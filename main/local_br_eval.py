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

from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.algorithms import Exploiter
from ippo import env_generator
import json
import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
import argparse
from gymnasium.spaces import Box


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_prot", type=str, required=True)
    parser.add_argument("--main_checkpoint_model_path", type=str, required=True)
    parser.add_argument("--done_model_checkpoint_path", type=str, required=True)
    parser.add_argument("--br_checkpoint_model_path", type=str, required=True)
    parser.add_argument("--state_list", type=str, required=True)
    parser.add_argument("--exploiter_is_cds", type=str, required=True)
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data_dict = json.loads(args.game_args)
    args.game_args = argparse.Namespace(**data_dict)
    args.state_list = ast.literal_eval(args.state_list)
    args.exploiter_is_cds = args.exploiter_is_cds == "True"
    args.eval_prot = args.eval_prot == "True"

    main_checkpoint_model_path = args.main_checkpoint_model_path
    done_model_checkpoint_path = args.done_model_checkpoint_path
    br_model_path = args.br_checkpoint_model_path

    br_rewards_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "br_rewards")
    selfplay_rewards_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfplay_rewards")
    os.makedirs(br_rewards_folder, exist_ok=True)
    os.makedirs(selfplay_rewards_folder, exist_ok=True)

    env = env_generator(args.game_args, STATE=args.state_list)
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

    if args.exploiter_is_cds:
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

    def exploiter_action_fn(obs):
        with th.no_grad():
            action, _, _, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
            if args.exploiter_is_cds:
                left_br_action, _, right_br_action, _, _, _ = br_model.policy(
                    obs_as_tensor(obs, br_model.device),
                    deterministic=False,
                    ego_forward=True,
                    adv_forward=True,
                    zero_ego_action=False,
                    zero_adv_action=True,
                )
                action_br = right_br_action if args.eval_prot else left_br_action
            else:
                action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
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
    selfplay_rewards = _collect_episode_returns(model, nr, selfplay_action_fn)

    # TODO: write out to a file and then aggregate the results and plot
    # os.makedirs(rewards_folder, exist_ok=True)
    if args.eval_prot:
        with open(os.path.join(br_rewards_folder, "%s_br%d_adv.txt" % (str(model.num_timesteps), args.br_index)), "w") as f:
            f.write(str(np.mean(exploiter_rewards)))
    else:
        with open(os.path.join(br_rewards_folder, "%s_br%d_ego.txt" % (str(model.num_timesteps), args.br_index)), "w") as f:
            f.write(str(np.mean(exploiter_rewards)))
    with open(os.path.join(selfplay_rewards_folder, "%s_br%d.txt" % (str(model.num_timesteps), args.br_index)), "w") as f:
        f.write(str(np.mean(selfplay_rewards)))


if __name__ == "__main__":
    main()
