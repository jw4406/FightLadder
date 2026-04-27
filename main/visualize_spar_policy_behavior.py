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
from common.justin.clean_derivative_free_spar_ippo import CleanDerivativeFreeSPARIPPO
from ippo import env_generator
from copy import deepcopy

from new_br_worker import (
    _infer_cds_architecture,
    _infer_league_matchup_states_from_dir,
    _load_league_checkpoint,
    _load_left_side_checkpoint,
    _load_right_side_checkpoints,
    load_league_model,
)


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


def _detect_model_type(model_path: str, device: str) -> str:
    """
    Identify the checkpoint's model family.

    Returns one of:
      - "spar"   : CDS SPAR (CleanDerivativeFreeSPAR), SB3 zip format.
      - "ippo"   : CDS IPPO (CleanDerivativeFreeSPARIPPO), SB3 zip format.
      - "league" : LeaguePPO, torch.save({cls_name, kwargs}) format.

    Detection order: try the SB3 zip loader first (covers both CDS arches);
    on failure, try the league loader and look for the league marker key.
    Raises ValueError if neither matches.
    """
    try:
        data, _, _ = load_from_zip_file(model_path, device=device)
    except Exception:
        data = None
    if data is not None:
        return _infer_cds_architecture(data, model_path)

    try:
        league_obj = _load_league_checkpoint(model_path, device=device)
    except Exception as exc:
        raise ValueError(
            f"Could not detect model type for {model_path!r}: not a SB3 zip "
            f"(CDS) and not a torch-saved league checkpoint ({exc})."
        )
    if isinstance(league_obj, dict) and "cls_name" in league_obj:
        return "league"
    raise ValueError(
        f"Could not detect model type for {model_path!r}: file loaded but "
        "lacks the league 'cls_name' marker."
    )


def _extract_cds_state_list(model_path: str, device: str) -> List[str]:
    """SB3-zip CDS checkpoints (SPAR/IPPO) carry their state_list in `data`."""
    data, _, _ = load_from_zip_file(model_path, device=device)
    state_list = data.get("state_list")
    if not isinstance(state_list, list) or len(state_list) == 0:
        raise ValueError(f"Checkpoint does not include a valid non-empty state_list: {model_path}")
    return list(state_list)


def _extract_league_state_list(model_path: str, device: str) -> List[str]:
    """
    League state_list source order:
      1) sibling MA*_right_*.pt files (the canonical roster — captures all
         matchups in the run, not just the loaded checkpoint's own).
      2) saved kwargs `state_name` from the loaded checkpoint (single
         matchup; useful when siblings have been moved/cleaned up).
    """
    try:
        return list(_infer_league_matchup_states_from_dir(model_path))
    except FileNotFoundError:
        pass
    saved = _load_league_checkpoint(model_path, device=device)
    state_name = saved.get("kwargs", {}).get("state_name")
    if not state_name:
        raise ValueError(
            f"League checkpoint {model_path!r} has no sibling MA*_right_*.pt "
            "files and no `state_name` in saved kwargs; cannot derive a "
            "matchup state for the env."
        )
    return [state_name]


def _extract_state_list_from_checkpoint(model_path: str, device: str, model_type: str) -> List[str]:
    if model_type == "league":
        return _extract_league_state_list(model_path, device=device)
    return _extract_cds_state_list(model_path, device=device)


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


def _filter_state_dict_keys(agent_dict: dict, prefix: str) -> dict:
    """
    Player.get_parameters() on a LeaguePPO returns a dict whose top-level
    keys are state-dict names like "policy" / "policy.optimizer" /
    "policy_other" / "policy_other.optimizer" (see LeaguePPO.state_dicts in
    common/algorithms.py). This helper keeps only the keys belonging to one
    side so we can recombine left+right from different MA snapshots.
    """
    return {k: v for k, v in agent_dict.items() if k == prefix or k.startswith(prefix + ".")}


def _load_per_matchup_right_policies(
    left_model,
    model_dir: str,
    league_matchup_states: List[str],
    device: str,
):
    """
    Build a list of standalone right-side policy modules, one per matchup
    state in *league_matchup_states* (same order). Each module is a clone of
    `left_model.policy_other` with the matchup's trained right-side weights
    loaded from sibling MA*_right_*.pt files in *model_dir*.

    Returns (policies, missing_keys) where:
      - policies[i] is the right policy for env-slot i (state league_matchup_states[i]).
      - missing_keys is the list of canonical matchup keys with no sibling
        MA right snapshot — caller can warn or fall back.

    The per-slot routing happens in _step_action: each env in the multi-env
    vec env is on a different matchup state, so we look up the right policy
    for that slot and forward only its observation through it.
    """
    from train_ma import _extract_chars_from_state_name, _sanitize_matchup_token

    right_checkpoints = _load_right_side_checkpoints(
        model_dir, league_matchup_states, device=device
    )

    policies = []
    missing = []
    template = left_model.policy_other
    for state_name in league_matchup_states:
        left_char, right_char = _extract_chars_from_state_name(state_name)
        canonical_key = (
            f"{_sanitize_matchup_token(left_char)}_vs_{_sanitize_matchup_token(right_char)}"
        )
        ckpt = right_checkpoints.get(canonical_key)
        if ckpt is None:
            # No sibling for this matchup — reuse template (will be the
            # wrong opponent but at least stays the right shape so the
            # visualizer keeps running). Caller decides whether to warn.
            missing.append(canonical_key)
            policies.append(template)
            continue

        right_weights = _filter_state_dict_keys(ckpt["agent_dict"], "policy_other")
        if not right_weights:
            # Schema drift: agent_dict didn't have "policy_other" keys.
            # Fall back to the "policy" keys (some checkpoints store the
            # trained right side under "policy" depending on save context).
            right_weights = _filter_state_dict_keys(ckpt["agent_dict"], "policy")
            # Translate "policy.*" -> "policy_other.*" so set_parameters
            # routes them into the right module on a fresh LeaguePPO.
            right_weights = {
                k.replace("policy", "policy_other", 1): v for k, v in right_weights.items()
            }

        policy_clone = deepcopy(template).to(device)
        policy_clone.set_training_mode(False)
        # Build a one-shot LeaguePPO-style state_dict ingestion: we already
        # filtered to the policy_other keys; load just into the cloned
        # module's state_dict.
        sd = right_weights.get("policy_other")
        if sd is None:
            # Some saves nest the actual nn.Module state under .pth-like
            # keys. Fall through to a best-effort: try the top-level dict.
            sd = right_weights
        try:
            policy_clone.load_state_dict(sd, strict=False)
        except Exception as exc:
            print(f"[viz/league] WARNING: failed to load right weights for "
                  f"{canonical_key}: {exc}; using template (stale).")
        policies.append(policy_clone)

    return policies, missing


def _apply_trained_left_to_model(model, left_agent_dict: dict, device: str) -> bool:
    """
    Overwrite *model.policy* (left side) with the trained left MA's weights
    from *left_agent_dict* (a Player.get_parameters() dict). Returns True
    on success, False if no usable weights were found.
    """
    left_weights = _filter_state_dict_keys(left_agent_dict, "policy")
    sd = left_weights.get("policy")
    if sd is None:
        return False
    try:
        model.policy.to(device).load_state_dict(sd, strict=False)
        model.policy.set_training_mode(False)
        return True
    except Exception as exc:
        print(f"[viz/league] WARNING: failed to load trained left weights: {exc}")
        return False


def _load_model(model_path: str, model_type: str, env, game_args, device: str):
    """
    Build a fully-initialized model of the requested type, with `model.env`
    pointed at *env* (the visualize-script's 2-player VecEnv) so the
    rendering/stepping loop can drive it directly.

    For CDS arches `.load(env=env, ...)` already attaches the env.

    For league we need extra care: a single saved Player .task only contains
    the trained side's weights well-populated; the other side's weights are
    stale (last sampled opponent). We:
      1) Build a base model via load_league_model (gets us LeaguePPO shape).
      2) Overwrite model.policy with the trained left MA from the sibling
         MA*_left_*.pt snapshot.
      3) Build a list of per-matchup right policy clones from sibling
         MA*_right_*.pt snapshots, attached as model._per_matchup_right_policies.
    The episode loop's _step_action then routes each env slot through its
    corresponding right policy.
    """
    if model_type == "spar":
        return CleanDerivativeFreeSPAR.load(model_path, env=env, num_perturbed=1, device=device)
    if model_type == "ippo":
        return CleanDerivativeFreeSPARIPPO.load(model_path, env=env, num_perturbed=1, device=device)
    if model_type == "league":
        league_states = list(getattr(env, "state_list", []))
        if not league_states:
            league_states = _extract_league_state_list(model_path, device=device)
        model = load_league_model(
            vars(game_args),
            model_path,
            league_matchup_states=league_states,
            n_envs=env.num_envs,
            device=device,
            use_wandb=False,
        )
        model.env = env

        # Overwrite the left side with the trained left MA from the sibling
        # snapshot. Without this, model.policy is whatever load_league_model
        # populated (only correct if the loaded .task happened to be the
        # left MA itself — and even then policy_other would still be stale).
        model_dir = os.path.dirname(os.path.abspath(model_path))
        left_ckpt = _load_left_side_checkpoint(model_dir, device=device)
        if left_ckpt is None:
            print(f"[viz/league] WARNING: no sibling MA*_left_*.pt found in "
                  f"{model_dir}; left side will use load_league_model's defaults.")
        else:
            ok = _apply_trained_left_to_model(model, left_ckpt["agent_dict"], device=device)
            if ok:
                print(f"[viz/league] Loaded trained left MA "
                      f"(step={left_ckpt['checkpoint_step']}) into model.policy.")

        # Per-matchup right policies, aligned with env state slot order.
        right_policies, missing = _load_per_matchup_right_policies(
            model, model_dir, league_states, device=device
        )
        model._per_matchup_right_policies = right_policies
        model._per_matchup_states = list(league_states)
        if missing:
            print(f"[viz/league] WARNING: no sibling right MA snapshot for "
                  f"matchup(s): {missing}; those env slots will reuse the "
                  "base model.policy_other (stale).")
        else:
            print(f"[viz/league] Loaded per-matchup right policies for "
                  f"{len(right_policies)} matchup(s).")
        return model
    raise ValueError(f"Unsupported model_type={model_type!r}")


def _step_action(model, obs, model_type: str, device) -> np.ndarray:
    """
    Compute the joint [ego/left, adv/right] action for one env step.

    CDS SPAR/IPPO expose a single multi-head policy that returns a 6-tuple
    `(ego_action, _, adv_action, _, _, _)` from one forward.

    For league: the trained left MA is shared across all env slots, but the
    trained right MA differs per matchup. _load_model attaches a list of
    per-matchup right policy clones in the same order as the env's state
    slots; we route each env's observation through its corresponding right
    policy. Left actions come from one batched forward of model.policy.
    """
    obs_t = obs_as_tensor(obs, device)
    with th.no_grad():
        if model_type in ("spar", "ippo"):
            ego_action, _, adv_action, _, _, _ = model.policy(obs_t)
        elif model_type == "league":
            # Left: one batched forward — same trained left MA for all envs.
            ego_action, _, _ = model.policy(obs_t)
            # Right: per-slot routing. obs_t is shape [num_envs, ...]; each
            # row goes through its matchup's policy. Sequential is fine
            # here because viz num_envs is small (one per matchup).
            per_matchup = getattr(model, "_per_matchup_right_policies", None)
            if per_matchup:
                n = int(obs_t.shape[0])
                right_chunks = []
                for i in range(n):
                    policy_i = per_matchup[i % len(per_matchup)]
                    a_i, _, _ = policy_i(obs_t[i:i + 1])
                    right_chunks.append(a_i)
                adv_action = th.cat(right_chunks, dim=0)
            else:
                # Fallback (shouldn't trigger when _load_model ran cleanly).
                adv_action, _, _ = model.policy_other(obs_t)
        else:
            raise ValueError(f"Unsupported model_type={model_type!r}")
    return np.hstack([ego_action.cpu().numpy(), adv_action.cpu().numpy()])


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
        default="",
        help="Output MP4 path. If omitted, defaults to "
             "main/videos/<policy_name>.mp4 where policy_name is the "
             "resolved checkpoint's basename without extension.",
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
    model_type = _detect_model_type(model_path, device=args.device)
    all_states = _extract_state_list_from_checkpoint(model_path, device=args.device, model_type=model_type)
    selected_states = [args.state] if args.state else list(all_states)

    # Default output path = main/videos/<policy_name>.mp4 where policy_name
    # is the resolved checkpoint's basename without extension. Explicit
    # --output_video always wins.
    if not args.output_video:
        policy_name = os.path.splitext(os.path.basename(model_path))[0]
        args.output_video = os.path.join("main", "videos", f"{policy_name}.mp4")

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
    # Stash the resolved state list on the env so _load_model can recover it
    # for league construction without re-reading the checkpoint.
    env.state_list = list(selected_states)
    model = _load_model(model_path, model_type, env=env, game_args=game_args, device=args.device)

    print(f"Selected model: {model_path} (type={model_type})")
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
        clipped_action = _step_action(model, obs, model_type, model.device)
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
