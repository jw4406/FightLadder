import os
import av
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import copy
import retro

# Your existing imports
from FightLadder.main.common.justin.Generalist_SPAR import Generalist_SPAR
from stable_baselines3.common.save_util import load_from_zip_file
from common.const import *
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P
from common.algorithms import Exploiter
from common.retro_wrappers import SFWrapper, Monitor2P


# --- Schedule functions (unchanged) ---
def critic_decay_schedule(initial_value: float):
    # ... (code is correct)
    def func(curr_step: int) -> float:
        return initial_value / curr_step

    return func


def actor_decay_schedule(initial_value: float):
    # ... (code is correct)
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


# --- OPTIMIZATION: Modified evaluate_sa to reuse a single environment ---
@torch.no_grad()
def evaluate_sa(eval_env, curr_state: str, args: argparse.Namespace, model, exploiter_model, env_index,
                greedy=0, record=True):
    """
    Runs evaluation episodes using a single, pre-created environment.
    This avoids the massive overhead of re-creating the emulator for each episode.
    """
    # --- OPTIMIZATION ---
    # The environment is now passed in, so we don't create it here.
    # We just need to set its statename for logging purposes if it changes.
    eval_env.statename = curr_state

    win_cnt = 0
    for j in range(1, args.num_episodes + 1):
        # This is now extremely fast compared to retro.make()
        obs = eval_env.reset()
        done = False

        if record:
            video_log = [Image.fromarray(eval_env.render(mode="rgb_array"))]

        while not done:
            # Your prediction and step logic remains the same
            if model.use_mirror:
                (action, _states), (_, _) = model.predict(obs, env_index, deterministic=False)
                exploit_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)
                action_other = exploit_action
            else:
                (action, _), (action_other, _) = model.predict(obs, env_index, deterministic=False)

            br_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)
            action_other = br_action

            obs, reward, reward_other, done, info = eval_env.step(np.hstack([action, action_other]))
            info = info[0]
            if record:
                video_log.append(Image.fromarray(eval_env.render(mode="rgb_array")))

            if done:
                # Video saving logic is unchanged
                if record:
                    try:
                        name = curr_state.split("/")[1]
                    except:
                        name = curr_state
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{args.video_dir}/{name}_episode_{j}.mp4", mode='w')
                    stream = container.add_stream('h264', rate=10)
                    stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
                    for img in video_log:
                        frame = av.VideoFrame.from_image(img)
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    container.mux(stream.encode(None))
                    container.close()

        if info['enemy_hp'] < info['agent_hp']:
            print(f"  Episode {j}: Victory!")
            win_cnt += 1

    win_rate = win_cnt / args.num_episodes
    print(f"  Winning rate over {args.num_episodes} episodes: {win_rate:.2%}")
    # We do NOT call env.close() here, as it's managed outside the loop.
    return win_rate


def run_serial_evaluation(args, sf_game, PLAYER):
    """
    Main function to set up and run the optimized serial evaluation.
    """
    # --- Configuration ---
    OPPONENT_LIST = ["Guile"]
    SIDE = "left"
    ego_folder = '/home/jw4406/codebase/FightLadder/main/trained_models/ego_models/'
    exploiter_folder = "/home/jw4406/codebase/FightLadder/main/trained_models/br_models/"

    # --- Create directories ---
    for dir_path in [args.save_dir, args.log_dir, args.video_dir, args.finetune_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # --- Discover model pairs ---
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
    nums = nums[-20:]
    print(f"Found {len(nums)} model pairs.")

    # --- OPTIMIZATION: Create models and envs ONCE before the loop ---
    print("--- Setting up reusable infrastructure (models and environments)... ---")

    # Even in serial, this follows the "create-use-destroy" pattern for safety
    temp_env = VecTransposeImage2P(SubprocVecEnv2P(
        [make_env(sf_game, state=args.state, side=args.side, reset_type=args.reset, rendering=args.render)]))
    eval_vec_env = VecTransposeImage2P(SubprocVecEnv2P(
        [make_env(sf_game, state=args.state, side=args.side, reset_type=args.reset, rendering=args.render)]))

    ego = Generalist_SPAR("AACCnnPolicy",
                          eval_vec_env,
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
    exploiter = Exploiter('CnnPolicy', temp_env, device='cuda', exploited=ego)  # Add other params
    temp_env.close()  # Close the temporary env

    evaluation_env = eval_vec_env  # This is the raw retro.Env for our loop

    # --- Main Evaluation Loop ---
    wrs = []
    for num in nums:
        print(f"\n--- Evaluating pair number: {num} ---")
        ego_model_path = os.path.join(ego_folder, f"{ego_beginning}{num}{ego_ending}")
        br_beginning, br_ending = "br_to_ppo_Guile_", "_steps.task.zip_8200000_steps.zip"
        exploiter_model_path = os.path.join(exploiter_folder, f"{br_beginning}{num}{br_ending}")

        print("  Loading parameters...")
        try:
            _, ego_params, _ = load_from_zip_file(ego_model_path)
            _, br_params, _ = load_from_zip_file(exploiter_model_path)
        except FileNotFoundError as e:
            print(f"  ERROR: Could not find model files for pair {num}. Skipping. Details: {e}")
            continue

        # --- OPTIMIZATION: Load weights into existing models instead of recreating them ---
        ego.set_parameters({k: v for k, v in ego_params.items() if 'optimizer' not in k}, exact_match=False)
        exploiter.set_parameters(br_params, exact_match=False)

        eval_state = 'two_player/Guile_left/Champion.Level1.GuileVsGuile.2Player.state'
        results = evaluate_sa(evaluation_env, eval_state, args, ego, exploiter, 0, record=True)

        wrs.append(results)
        with open("/home/jw4406/codebase/FightLadder/main/trained_models/_start_results.txt", 'a') as f:
            f.write('\n')
            f.write(str(results))

    print("\n--- Evaluation complete. Closing final environment. ---")
    eval_vec_env.close()



if __name__ == "__main__":
    # --- OPTIMIZATION: Argument parser is defined and called only once at startup ---
    parser = argparse.ArgumentParser(description='Run serial Street Fighter model evaluation.')
    parser.add_argument("--player", type=str, required=True)
    parser.add_argument('--reset', choices=['round', 'match', 'game'], default='round')
    parser.add_argument('--save-dir', default="trained_models/default_save")
    parser.add_argument('--log-dir', default="logs")
    parser.add_argument('--model-name-prefix', default="ppo_default")
    parser.add_argument('--state', default='Champion.Level1.RyuVsGuile')
    parser.add_argument('--side', default='both', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--video-dir', default='videos/default_video')
    parser.add_argument('--finetune-dir', default='finetune')
    parser.add_argument('--enable-combo', action='store_true')
    parser.add_argument('--null-combo', action='store_true')
    parser.add_argument('--transform-action', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Dynamically set some paths based on the player argument
    args.model_name_prefix = f'ppo_{args.player}'
    args.video_dir = f'videos/spar_spar_{args.player}'

    print("--- Initializing Serial Evaluation ---")
    print(f"Command Line Args: {args}")

    sf_game = 'StreetFighterIISpecialChampionEdition-Genesis'
    run_serial_evaluation(args, sf_game, args.player)