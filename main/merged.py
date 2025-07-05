import os
import av
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import copy
import retro
import multiprocessing
import traceback

# Your existing imports
from FightLadder.main.common.justin.Generalist_SPAR import Generalist_SPAR
from stable_baselines3.common.save_util import load_from_zip_file
from common.const import *
# Using SubprocVecEnv2P as requested, and DummyVecEnv2P for the simple worker envs
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P, DummyVecEnv2P
from common.algorithms import Exploiter
from common.retro_wrappers import SFWrapper, Monitor2P


# --- Schedule functions (unchanged) ---
def critic_decay_schedule(initial_value: float):
    def func(curr_step: int) -> float:
        return initial_value / curr_step

    return func


def actor_decay_schedule(initial_value: float):
    def func(curr_step: int) -> float:
        return initial_value / (curr_step ** (2 / 3))

    return func


# --- Environment creation function (unchanged) ---
def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=False,
             null_combo=False, transform_action=False, seed=0):
    def _init():
        players = 2
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE, players=players
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level,
                        state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo,
                        transform_action=transform_action)
        env = Monitor2P(env)
        env.seed(seed)
        return env

    return _init


# --- NEW: Worker function to play a single game in parallel ---
# --- CORRECTED: Worker function that creates only ONE environment ---
def play_one_episode(worker_args):
    """
    This function is executed by each parallel worker.
    It creates only one environment and cleverly manages its ownership
    to avoid resource conflicts.
    """
    (
        args_dict, sf_game, curr_state, episode_num,
        ego_model_path, exploiter_model_path, PLAYER, OPPONENT_LIST
    ) = worker_args

    # Recreate a lightweight args object
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)

    args = Args(**args_dict)

    # This single environment will be shared for setup
    vec_env = None
    try:
        # 1. CREATE ONLY ONE VecEnv for this worker.
        vec_env = DummyVecEnv2P([make_env(sf_game, state=curr_state, side=args.side, reset_type=args.reset,
                                          rendering=args.render, seed=episode_num)])
        raw_env = vec_env.envs[0]

        # 2. LOAD models from paths inside the worker.
        _, ego_params, _ = load_from_zip_file(ego_model_path)
        _, br_params, _ = load_from_zip_file(exploiter_model_path)

        # 3. INITIALIZE both models with the SAME environment.
        ego = Generalist_SPAR("AACCnnPolicy",
                vec_env,
                device="cuda",
                verbose=2,
                n_steps=1536,  # 1408,
                batch_size=768,  # 2816,  # 512,
                n_epochs=10,
                gamma=0.94,
                v_learning_rate=5e-3, c_learning_rate=1e-4,
                d_learning_rate=5e-4, v_learning_rate_decay=critic_decay_schedule(1e-3),
                c_learning_rate_decay=critic_decay_schedule(1e-4),
                d_learning_rate_decay=critic_decay_schedule(5e-4),
                clip_range=None,
                tensorboard_log=args.log_dir,
                seed=args.seed,
                ent_coef=0,
                dstb_ent_coef=0,
                I_AM_LEFT=True,
                I_AM_RIGHT=False,
                num_adversary=1,
                n_global_env=1,
                n_env_per_adv=1,
                opp_list=OPPONENT_LIST,
                player=PLAYER,
                use_mirror=False
            )
        exploiter = Exploiter('CnnPolicy', vec_env, device='cuda', exploited=ego)

        # 4. *** THE CRUCIAL FIX ***
        # Detach the environment from the exploiter. This prevents its __del__
        # method from closing our shared environment.
        exploiter.env = None

        # Now we can safely set the parameters
        ego.set_parameters({k: v for k, v in ego_params.items() if 'optimizer' not in k}, exact_match=False)
        exploiter.set_parameters(br_params, exact_match=False)

        # 5. PLAY one episode using the raw environment.
        obs = raw_env.reset()
        done = False
        while not done:
            (action, _), (action_other, _) = ego.predict(obs, env_index=0, deterministic=False)
            br_action, _ = exploiter.predict(obs, deterministic=False)
            action_other = br_action
            obs, reward, reward_other, done, info = raw_env.step(np.hstack([action, action_other]))
            #info = info[0]  # The info from the raw env

        # 6. Return the result
        return 1 if info['enemy_hp'] < info['agent_hp'] else 0

    except Exception as e:
        print(f"--- WORKER ERROR in episode {episode_num} ---")
        traceback.print_exc()
        return 0  # Count as a loss if an error occurs

    finally:
        # 7. ENSURE the single environment is always closed when the worker is done.
        if vec_env is not None:
            vec_env.close()


# --- NEW: Evaluation function that manages the parallel pool ---
def evaluate_sa_parallel(args, sf_game, curr_state, ego_model_path, exploiter_model_path, PLAYER, OPPONENT_LIST):
    """
    Manages a pool of workers to play all episodes in parallel.
    """
    print(f"  Starting parallel evaluation for {args.num_episodes} episodes...")

    args_dict = vars(args)
    tasks = [(
        args_dict, sf_game, curr_state, i,
        ego_model_path, exploiter_model_path, PLAYER, OPPONENT_LIST
    ) for i in range(args.num_episodes)]

    # You can tune the number of processes. It's often best to match your number of CPU cores.
    num_processes = min(args.num_episodes, 16)
    print(f"  Dispatching to {num_processes} worker processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(play_one_episode, tasks)

    win_count = sum(results)
    win_rate = win_count / args.num_episodes

    print(f"  Evaluation complete. Victories: {win_count}/{args.num_episodes}. Win Rate: {win_rate:.2%}")
    return win_rate


# --- Main loop that iterates SERIALLY over model pairs ---
def run_evaluation_loop(args, sf_game, PLAYER):
    """
    Main function to set up and run the evaluation.
    The outer loop over model files is serial.
    The inner loop over game episodes is parallel.
    """
    OPPONENT_LIST = ["Guile"]
    SIDE = "left"
    ego_folder = '/home/jw4406/codebase/FightLadder/main/trained_models/ego_models/'
    exploiter_folder = "/home/jw4406/codebase/FightLadder/main/trained_models/br_models/"

    for dir_path in [args.save_dir, args.log_dir, args.video_dir, args.finetune_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print("--- Discovering model pairs to evaluate... ---")
    nums = []
    ego_beginning, ego_ending = "ppo_Guile_", "_steps.task"
    for fname in os.listdir(ego_folder):
        if fname.startswith(ego_beginning) and fname.endswith(ego_ending):
            try:
                num_str = fname.strip(ego_beginning).strip(ego_ending)
                nums.append(int(num_str))
            except ValueError:
                continue
    nums.sort()
    print(f"Found {len(nums)} model pairs.")

    wrs = []
    for num in nums:
        print(f"\n--- Evaluating model pair number: {num} ---")
        ego_model_path = os.path.join(ego_folder, f"{ego_beginning}{num}{ego_ending}")
        br_beginning, br_ending = "br_to_ppo_Guile_", "_steps.task.zip_8200000_steps.zip"
        exploiter_model_path = os.path.join(exploiter_folder, f"{br_beginning}{num}{br_ending}")

        if not os.path.exists(ego_model_path) or not os.path.exists(exploiter_model_path):
            print(f"  ERROR: Could not find model files for pair {num}. Skipping.")
            continue

        eval_state = 'two_player/Guile_left/Champion.Level1.GuileVsGuile.2Player.state'

        # Call the new parallel evaluation function
        win_rate = evaluate_sa_parallel(args, sf_game, eval_state, ego_model_path, exploiter_model_path, PLAYER,
                                        OPPONENT_LIST)

        wrs.append(win_rate)
        with open(f"{args.finetune_dir}/{args.model_name_prefix}_{num}_results.txt", 'w') as f:
            f.write(str(win_rate))

    print("\n\n--- All model pairs evaluated. ---")
    print("Collected Win Rates:", wrs)


if __name__ == "__main__":
    # --- MUST set start method to 'spawn' for this to work reliably ---
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("--- Multiprocessing start method set to 'spawn' ---")
    except RuntimeError:
        print("--- Multiprocessing context already set ---")

    parser = argparse.ArgumentParser(description='Run parallel episode evaluation of Street Fighter models.')
    parser.add_argument("--player", type=str, required=True)
    parser.add_argument('--reset', choices=['round', 'match', 'game'], default='round')
    parser.add_argument('--save-dir', default="trained_models/default_save")
    parser.add_argument('--log-dir', default="logs")
    parser.add_argument('--model-name-prefix', default="ppo_default")
    parser.add_argument('--state', default='Champion.Level1.RyuVsGuile')
    parser.add_argument('--side', default='both', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true')  # Note: Render won't show in parallel workers
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--video-dir', default='videos/default_video')
    parser.add_argument('--finetune-dir', default='finetune')
    parser.add_argument('--enable-combo', action='store_true')
    parser.add_argument('--null-combo', action='store_true')
    parser.add_argument('--transform-action', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.model_name_prefix = f'ppo_{args.player}'
    args.video_dir = f'videos/spar_spar_{args.player}'

    print("--- Initializing Evaluation ---")
    print(f"Command Line Args: {args}")

    sf_game = 'StreetFighterIISpecialChampionEdition-Genesis'
    run_evaluation_loop(args, sf_game, args.player)