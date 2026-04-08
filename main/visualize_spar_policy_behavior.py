"""
Visualize SPAR policy behavior by recording long gameplay videos.

Usage examples:
  python main/visualize_spar_policy_behavior.py \
      --model_source /path/to/model_or_folder \
      --episodes 50 \
      --output_video main/videos/spar_behavior.mp4

`--model_source` can be:
  1) a specific checkpoint file path, or
  2) a directory, in which case one checkpoint is chosen randomly.
"""

import argparse
import os
import random
import sys
from types import SimpleNamespace
from typing import List, Tuple


def _peek_torch_device_argv(argv: List[str]):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


_peeked_dev = _peek_torch_device_argv(sys.argv[1:])
if _peeked_dev is not None and str(_peeked_dev).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import av
import numpy as np
import torch as th
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import obs_as_tensor

from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from ippo import env_generator


def _candidate_model_files(folder: str) -> List[str]:
    exts = (".zip", ".task")
    candidates = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(exts):
            candidates.append(path)
    return sorted(candidates)


def _resolve_model_path(model_source: str, seed: int) -> str:
    if os.path.isfile(model_source):
        return model_source
    if not os.path.isdir(model_source):
        raise FileNotFoundError(f"model_source is neither file nor directory: {model_source}")
    candidates = _candidate_model_files(model_source)
    if not candidates:
        raise FileNotFoundError(f"No .zip or .task files found in directory: {model_source}")
    rng = random.Random(seed)
    return rng.choice(candidates)


def _extract_state_list_from_checkpoint(model_path: str, device: str) -> List[str]:
    data, _, _ = load_from_zip_file(model_path, device=device)
    state_list = data.get("state_list")
    if not isinstance(state_list, list) or len(state_list) == 0:
        raise ValueError(f"Checkpoint does not include a valid non-empty state_list: {model_path}")
    return list(state_list)


def _extract_step_outputs(step_output) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle both custom 2-player VecEnv signatures and Gym-style signatures.
    Returns: (obs, done)
    """
    if len(step_output) == 5:
        obs, _, _, done, _ = step_output
    else:
        obs, _, done, _ = step_output
    done = np.asarray(done).reshape(-1).astype(bool)
    return obs, done


def _first_frame(render_output):
    if isinstance(render_output, (list, tuple)):
        if len(render_output) == 0:
            raise ValueError("render() returned an empty list/tuple.")
        return np.asarray(render_output[0], dtype=np.uint8)
    return np.asarray(render_output, dtype=np.uint8)


def _write_frame(container, stream, frame_np: np.ndarray) -> None:
    frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_source",
        type=str,
        required=True,
        help="Path to a model file, or a directory to sample a random model from.",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=os.path.join("main", "videos", "spar_policy_behavior.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="How many episodes to record.")
    parser.add_argument("--max_steps_per_episode", type=int, default=5000, help="Safety cap per episode.")
    parser.add_argument(
        "--state",
        type=str,
        default="",
        help="Optional explicit state path. If omitted, all states from checkpoint state_list are used.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model/state selection.")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for model loading/inference.")

    # Env args forwarded into ippo.env_generator(...)
    parser.add_argument("--reset", choices=["round", "match", "game"], default="round")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--render", choices=["True", "False"], default="False")
    parser.add_argument("--enable_combo", choices=["True", "False"], default="True")
    parser.add_argument("--null_combo", choices=["True", "False"], default="False")
    parser.add_argument("--transform_action", choices=["True", "False"], default="False")

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.manual_seed(args.seed)

    model_path = _resolve_model_path(args.model_source, seed=args.seed)
    all_states = _extract_state_list_from_checkpoint(model_path, device=args.device)
    selected_states = [args.state] if args.state else list(all_states)

    game_args = SimpleNamespace(
        reset=args.reset,
        side=args.side,
        render=args.render == "True",
        enable_combo=args.enable_combo == "True",
        null_combo=args.null_combo == "True",
        transform_action=args.transform_action == "True",
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    env = env_generator(game_args, STATE=selected_states, n_envs=1)
    model = CleanDerivativeFreeSPAR.load(model_path, env=env, num_perturbed=1, device=args.device)

    print(f"Selected model: {model_path}")
    if args.state:
        print(f"Selected state override: {args.state}")
    else:
        print(f"Using all states from checkpoint state_list (count={len(selected_states)})")
    print(f"Recording {args.episodes} episodes to: {args.output_video}")

    obs = env.reset()
    first = _first_frame(env.render(mode="rgb_array"))
    h, w = int(first.shape[0]), int(first.shape[1])
    container = av.open(args.output_video, mode="w")
    stream = container.add_stream("mpeg4", rate=args.fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    episodes_done = 0
    episode_steps = 0
    _write_frame(container, stream, first)

    while episodes_done < args.episodes:
        with th.no_grad():
            ego_action, _, adv_action, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
        clipped_action = np.hstack([ego_action.cpu().numpy(), adv_action.cpu().numpy()])
        obs, done = _extract_step_outputs(env.step(clipped_action))
        frame = _first_frame(env.render(mode="rgb_array"))
        _write_frame(container, stream, frame)

        episode_steps += 1
        if np.any(done) or episode_steps >= args.max_steps_per_episode:
            episodes_done += 1
            episode_steps = 0
            print(f"Completed episode {episodes_done}/{args.episodes}")
            if episodes_done < args.episodes:
                obs = env.reset()
                frame = _first_frame(env.render(mode="rgb_array"))
                _write_frame(container, stream, frame)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
