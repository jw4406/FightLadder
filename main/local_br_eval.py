import os
import sys


def _peek_torch_device_argv(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


_br_eval_dev = _peek_torch_device_argv(sys.argv[1:])
if _br_eval_dev is not None and str(_br_eval_dev).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import ast
import re

from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.algorithms import Exploiter
from ippo import env_generator
import json
import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
import argparse
from gymnasium.spaces import Box
from new_br_worker import _FixedMatchupPolicyAdapter, _dedupe_preserve_order


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_prot", type=str, required=True)
    parser.add_argument("--main_checkpoint_model_path", type=str, required=True)
    parser.add_argument("--done_model_checkpoint_path", type=str, required=True)
    parser.add_argument("--br_checkpoint_model_path", type=str, required=True)
    parser.add_argument("--full_state_list", type=str, required=True)
    parser.add_argument("--state_list", type=str, required=True)
    parser.add_argument("--dedicated_exploiter", type=str, required=True)
    parser.add_argument("--br_index", type=int, required=True)
    parser.add_argument("--game_args", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for loaded models (e.g. cpu, cuda:0)",
    )
    return parser


def _extract_step_outputs(step_output):
    """Handle both 2-player VecEnv step signatures and standard Gym-style signatures."""
    if len(step_output) == 5:
        obs, reward, _, done, info = step_output
    else:
        obs, reward, done, info = step_output
    reward = np.asarray(reward).reshape(-1)
    done = np.asarray(done).reshape(-1).astype(bool)
    return obs, reward, done, info


def _collect_episode_returns(model, target_episodes, action_fn):
    """Collect per-episode returns from vectorized envs that finish asynchronously."""
    obs = model.env.reset()
    n_envs = model.env.num_envs
    running_returns = np.zeros(n_envs, dtype=np.float32)
    finished_returns = []

    while len(finished_returns) < target_episodes:
        clipped_action = action_fn(obs)
        obs, reward, done, info = _extract_step_outputs(model.env.step(clipped_action))
        running_returns += reward

        done_indices = np.where(done)[0]
        for idx in done_indices:
            finished_returns.append(float(running_returns[idx]))
            running_returns[idx] = 0.0
            print(f"Episode {len(finished_returns)} completed", flush=True)
            if len(finished_returns) >= target_episodes:
                break

    return finished_returns


def _extract_left_right_names_from_state(state):
    if not state:
        return "unknown_left", "unknown_right"
    basename = os.path.basename(str(state))
    matchup_match = re.search(r"\.([A-Za-z]+)Vs([A-Za-z]+)\.2Player\.state$", basename)
    if matchup_match:
        return matchup_match.group(1), matchup_match.group(2)
    return "unknown_left", "unknown_right"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data_dict = json.loads(args.game_args)
    args.game_args = argparse.Namespace(**data_dict)
    args.state_list = ast.literal_eval(args.state_list)
    args.full_state_list = ast.literal_eval(args.full_state_list)
    args.dedicated_exploiter = args.dedicated_exploiter == "True"
    args.eval_prot = args.eval_prot == "True"

    main_checkpoint_model_path = args.main_checkpoint_model_path
    done_model_checkpoint_path = args.done_model_checkpoint_path
    br_model_path = args.br_checkpoint_model_path

    br_rewards_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "br_rewards")
    selfplay_rewards_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfplay_rewards")
    os.makedirs(br_rewards_folder, exist_ok=True)
    os.makedirs(selfplay_rewards_folder, exist_ok=True)

    env = env_generator(args.game_args, STATE=args.state_list)
    full_env = env_generator(args.game_args, STATE=args.full_state_list)
    # ENV_ID = args.env_id
    try:
        model = CleanDerivativeFreeSPAR.load(
            main_checkpoint_model_path, env=env, num_perturbed=1, device=args.device
        )
    except FileNotFoundError:
        model = CleanDerivativeFreeSPAR.load(
            done_model_checkpoint_path, env=env, num_perturbed=1, device=args.device
        )

# if args.eval_prot is True: # we're training an optimal adversary
#     dstb_action_space = Box(low=model.dstb_action_space.low, high=model.dstb_action_space.high, shape=model.dstb_action_space.shape)
#     env.action_space = dstb_action_space
# else:
#     assert args.eval_prot is False 
#     # we're training an optimal ego against the current adversary
#     ego_action_space = Box(low=model.action_space.low, high=model.action_space.high, shape=model.action_space.shape)
#     env.action_space = ego_action_space

# if args.exploiter_is_cds:
#     pass
# else:
#     env.action_space = model.dstb_action_space

    if not args.dedicated_exploiter:
        br_model = CleanDerivativeFreeSPAR.load(
            br_model_path, env=env, num_perturbed=1, device=args.device
        )
    else:
        # if args.eval_prot is True:  # we're training an optimal adversary
        #     dstb_action_space = Box(
        #         low=model.dstb_action_space.low,
        #         high=model.dstb_action_space.high,
        #         shape=model.dstb_action_space.shape,
        #     )
        #     env.action_space = dstb_action_space
        # else:
        #     assert args.eval_prot is False
        #     # we're training an optimal ego against the current adversary
        #     ego_action_space = Box(
        #         low=model.action_space.low,
        #         high=model.action_space.high,
        #         shape=model.action_space.shape,
        #     )
        #     env.action_space = ego_action_space
        # print("#$%*&^%$EVAL PROT: %s$%^&*", args.eval_prot)
        # print("$@#$%^&*()(*&^%$#@)%s$#%^&*()(*&^%$#@", args.exploiter_is_cds)
        br_model = Exploiter.load(br_model_path, env=env, n_envs=1, device=args.device)

    nr = 50
    exploiter_rewards, selfplay_rewards = [], []
    model_policy_for_eval = model.policy
    use_fixed_matchup_adapter = False
    if args.dedicated_exploiter:
        # Dedicated runs pass a repeated singleton state list; intercept policy
        # forward calls on the exploited CDS model through one fixed matchup.
        eval_unique_states = _dedupe_preserve_order(args.state_list)
        if len(eval_unique_states) == 1:
            dedicated_state = eval_unique_states[0]
            model_unique_states = _dedupe_preserve_order(getattr(model, "state_list", []))
            if dedicated_state in model_unique_states:
                fixed_matchup_idx = model_unique_states.index(dedicated_state)
                model_policy_for_eval = _FixedMatchupPolicyAdapter(
                    model.policy, fixed_matchup_idx=fixed_matchup_idx
                )
                use_fixed_matchup_adapter = True
                print(
                    "Configured fixed-matchup eval adapter: "
                    f"state={dedicated_state}, fixed_matchup_idx={fixed_matchup_idx}",
                    flush=True,
                )

    def exploiter_action_fn(obs):
        with th.no_grad():
            obs_model_tensor = obs_as_tensor(obs, model.device)
            if args.dedicated_exploiter and use_fixed_matchup_adapter:
                action, _ = model_policy_for_eval(
                    obs_model_tensor,
                    deterministic=False,
                    ego_forward=args.eval_prot,
                    adv_forward=not args.eval_prot,
                )
            else:
                if args.eval_prot:
                    action, _, _, _, _, _ = model.policy(obs_model_tensor)
                else:
                    _, _, action, _, _, _ = model.policy(obs_model_tensor)
            if args.dedicated_exploiter:
                action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
            else:
                if args.eval_prot:
                    _, _, action_br, _, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
                else:
                    action_br, _, _, _, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
        action = action.cpu().numpy()
        action_br = action_br.cpu().numpy()
        if args.eval_prot:
            return np.hstack([action, action_br])
        return np.hstack([action_br, action])
# for i in range(nr):
#     curr_reward = 0
#     obs = model.env.reset()
#     obs = np.expand_dims(obs, 0)
#     done = False
#     while not done:
#         with th.no_grad():
#             if args.exploiter_is_cds:
#                 ego_actions, ego_log_probs, action_br, adv_log_probs, values, q_values = br_model.policy(obs_as_tensor(obs, br_model.device), deterministic=False, ego_forward=True, adv_forward=True, zero_ego_action=False, zero_adv_action=True)
#             else:
#                 action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
#             action, _, adv_action, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
#         action = action.cpu().numpy()

#         clipped_action = np.hstack([action_br, adv_action])
#         obs, reward, done, info = model.env.step(clipped_action)
#         curr_reward += reward
#     exploiting_adv_rewards.append(curr_reward)
#     print(f"Episode {i+1} completed")

    def selfplay_action_fn(obs):
        with th.no_grad():
            action, _, adv_action, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
        action = action.cpu().numpy()
        adv_action = adv_action.cpu().numpy()
        return np.hstack([action, adv_action])

    exploiter_rewards = _collect_episode_returns(model, nr, exploiter_action_fn)
    model.env = full_env
    selfplay_rewards = _collect_episode_returns(model, nr, selfplay_action_fn)

    # TODO: write out to a file and then aggregate the results and plot
    # os.makedirs(rewards_folder, exist_ok=True)
    main_side = "left" if args.eval_prot else "right"
    exploiter_side = "left" if not args.eval_prot else "right"
    tested_states = _dedupe_preserve_order(args.state_list)
    tested_state = tested_states[0] if tested_states else ""
    left_name, right_name = _extract_left_right_names_from_state(tested_state)
    main_name = left_name if main_side == "left" else right_name
    exploiter_name = right_name if exploiter_side == "right" else left_name
    filename = (
        f"{model.num_timesteps}_main_{main_side}_{main_name}_"
        f"exploiter_{exploiter_side}_{exploiter_name}_.txt"
    )
    with open(os.path.join(br_rewards_folder, filename), "w") as f:
        f.write(str(np.mean(exploiter_rewards)))
    with open(os.path.join(selfplay_rewards_folder, filename), "w") as f:
        f.write(str(np.mean(selfplay_rewards)))

    eval_target = "ego" if args.eval_prot else "adv"
    tested_state_for_print = tested_state if len(tested_states) == 1 else tested_states
    run_summary = {
        "checkpoint_num_timesteps": int(model.num_timesteps),
        "br_index": args.br_index,
        "tested_state": tested_state_for_print,
        "left_name": left_name,
        "right_name": right_name,
        "output_filename": filename,
        "eval_target": eval_target,
        "dedicated_exploiter": args.dedicated_exploiter,
        "device": args.device,
        "main_checkpoint_model_path": args.main_checkpoint_model_path,
        "done_model_checkpoint_path": args.done_model_checkpoint_path,
        "br_checkpoint_model_path": args.br_checkpoint_model_path,
        "full_state_list": args.full_state_list,
        "state_list": args.state_list,
    }
    print(
        f"local br eval complete for checkpoint {model.num_timesteps} | "
        f"br_index={args.br_index} | state={tested_state_for_print} | eval_target={eval_target}",
        flush=True,
    )
    print("local br eval args:", flush=True)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)

    
if __name__ == "__main__":
    main()
    print("local br eval complete")