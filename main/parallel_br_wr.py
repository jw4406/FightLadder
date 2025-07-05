import os
import av
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import copy
import retro  # Import retro here

# Assuming these are available in your environment, otherwise they'll cause issues
from FightLadder.main.common.justin.Generalist_SPAR import Generalist_SPAR
from stable_baselines3.common.save_util import load_from_zip_file

from common.const import *
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P, DummyVecEnv2P
from common.game import get_next_level
from common.algorithms import IPPO, MAGICS_PPO, RARL_PPO, TSS_PPO, Specialized_Agent, Specialized_Agent_IPPO, eepy, \
    Exploiter
from stable_baselines3 import MAGICS_AL
from common.retro_wrappers import SFWrapper, Monitor2P

# --- Add multiprocessing import ---
import multiprocessing

# ----------------------------------

# --- Global variables (if needed, but try to pass as arguments) ---
# It's better to avoid global mutable state when possible in multiprocessing.
# If sf_game, PLAYER, REMOVAL are truly global and constant, they can remain.
# Otherwise, consider passing them as arguments to the worker function.
sf_game = 'StreetFighterIISpecialChampionEdition-Genesis'  # Example, replace with your actual sf_game
PLAYER = None  # Will be set by command line arg
REMOVAL = None  # Will be set within main if needed


# ------------------------------------------------------------------

def const_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return initial_value

    return func


def critic_decay_schedule(initial_value: float):
    def func(curr_step: int) -> float:
        return initial_value / curr_step

    return func


def actor_decay_schedule(initial_value: float):
    def func(curr_step: int) -> float:
        return initial_value / (curr_step ** (2 / 3))

    return func


def constructor(args, side, log_name=None, single_env=False):
    pass


def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True,
             null_combo=False, transform_action=False, seed=0):
    def _init():
        players = 2
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level,
                        state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo,
                        transform_action=transform_action)
        env = Monitor2P(env)
        env.seed(seed)
        return env

    return _init


@torch.no_grad()
def evaluate_sa(env, curr_state, args, model, exploiter_model, env_index, greedy=0, record=True):
    # This function remains largely the same, but it's crucial that
    # any objects passed into it (like models) are either picklable
    # or re-initialized within the worker process.
    # In this case, `model` and `exploiter_model` are passed, which are
    # likely complex objects. This is where pickling can be an issue.
    # A better approach for multiprocessing for this specific structure
    # is to pass the *paths* to the models, and load them inside this function.

    args.num_episodes = 50
    win_cnt = 0
    vic = np.zeros((50,))
    for j in range(1, args.num_episodes + 1):
        #env = make_env(sf_game, state=curr_state, side='both', reset_type=args.reset, rendering=args.render,
        #               enable_combo=args.enable_combo, null_combo=args.null_combo,
        #               transform_action=args.transform_action, seed=0)().env
        #env = model.env
        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:
            if model.use_mirror is True:
                (action, _states), (_, _) = model.predict(obs, env_index, deterministic=False)
                exploit_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)
                action_other = exploit_action
            else:
                if np.random.uniform() > greedy:
                    (action, _states), (action_other, _states_other) = model.predict(obs, env_index,
                                                                                     deterministic=False)
                else:
                    (action, _states), (action_other, _states_other) = model.predict(obs, env_index,
                                                                                     deterministic=False)
            br_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)

            action_other = br_action
            obs, reward, reward_other, done, info = env.step(np.hstack([action, action_other]))
            #info = info[0]
            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))

            if done:
                if record:
                    try:
                        name = curr_state.split("/")[1]
                    except:
                        name = curr_state
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{args.video_dir}/{name}_episode_{j}.mp4", mode='w')
                    stream = container.add_stream('h264', rate=10)
                    stream.width = width
                    stream.height = height
                    stream.pix_fmt = 'yuv420p'
                    for img in video_log:
                        frame = av.VideoFrame.from_image(img)
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    remain_packets = stream.encode(None)
                    container.mux(remain_packets)
                    container.close()

        if info['enemy_hp'] < info['agent_hp']:
            print("Victory!")
            win_cnt += 1

        env.close()

    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate

def check_env_alive(vec_env, step_name: str):
    """A helper function to check if the underlying retro emulator is alive."""
    try:
        raw_env = vec_env.envs[0]
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env
            if isinstance(raw_env, retro.RetroEnv):
                break

        if hasattr(raw_env, 'em') and raw_env.em is not None:
            print(f"--- [SUCCESS] ENV IS ALIVE after: '{step_name}' ---")
            return True
        else:
            print(f"!!! [FAILURE] ENV IS DEAD after: '{step_name}' !!!")
            return False
    except Exception as e:
        print(f"!!! [FAILURE] FAILED to check env status after '{step_name}': {e}")
        return False


#
# ---- THIS IS THE TEST FUNCTION ----
#
def run_minimal_test(args_tuple):
    # Unpack just what's needed for the environment
    ego_model_path, exploiter_model_path, current_num, player_arg, opponent_list_arg, side_arg, state_list_arg, all_args_dict = args_tuple

    class DummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args = DummyArgs(**all_args_dict)
    eval_state = state_list_arg[0]

    # Use the same make_env function you already have
    def make_env_fn():
        return [
            make_env(sf_game, state=eval_state, side=args.side, reset_type=args.reset, rendering=args.render,
                     enable_combo=args.enable_combo, null_combo=args.null_combo,
                     transform_action=args.transform_action, seed=0)
        ]

    try:
        print(f"\n--- MINIMAL TEST STARTING IN WORKER {os.getpid()} ---")

        # === PHASE 1: Create and destroy the first environment ===
        print("\n[PHASE 1] Creating temporary environment...")
        temp_env = VecTransposeImage2P(DummyVecEnv2P(make_env_fn()))
        if not check_env_alive(temp_env, "temp env creation"): return "FAILED"

        print("[PHASE 1] Closing temporary environment...")
        temp_env.close()
        print("[PHASE 1] Temporary environment closed.")

        # === PHASE 2: Create the second environment ===
        print("\n[PHASE 2] Creating final evaluation environment...")
        final_env = VecTransposeImage2P(DummyVecEnv2P(make_env_fn()))
        if not check_env_alive(final_env, "final env creation"): return "FAILED"

        # === PHASE 3: Attempt to use the second environment ===
        print("\n[PHASE 3] Attempting to reset the final environment...")
        final_env.reset()
        print("--- [SUCCESS] Final environment was reset successfully. ---")

        final_env.close()
        return "SUCCESS"

    except Exception as e:
        print(f"\n--- MINIMAL TEST FAILED IN WORKER {os.getpid()} ---")
        traceback.print_exc()
        return f"FAILED"
# --- Worker function for parallel execution ---
# --- Worker function for parallel execution ---
def run_evaluation_for_pair(args_tuple):
    # Unpack arguments
    # Now, the last element of args_tuple is the dictionary itself.
    ego_model_path, exploiter_model_path, current_num, player_arg, opponent_list_arg, side_arg, state_list_arg, all_args_dict = args_tuple

    # Recreate the args object from the dictionary
    class DummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    args = DummyArgs(**all_args_dict) # This is now correct!

    # Re-initialize these globals if they are used by make_env or other functions called here
    global PLAYER
    PLAYER = player_arg # Ensure PLAYER is set correctly in each process
    global REMOVAL
    # REMOVAL is probably set based on some condition, if it's constant for all, pass it.
    # If it's dynamic based on the loop, it needs to be part of args_tuple.

    # Redefine the env generators if they depend on PLAYER or other dynamic args
    def env_generator():
        each_env_count = 1
        env = []
        for i in range(len(state_list_arg)): # Use the passed state_list_arg
            for j in range(each_env_count):
                env.append(
                    make_env(sf_game, state=state_list_arg[i], side=args.side, reset_type=args.reset, rendering=args.render,
                             enable_combo=args.enable_combo, null_combo=args.null_combo,
                             transform_action=args.transform_action, seed=0))
        return VecTransposeImage2P(DummyVecEnv2P(env))

    eval_state = state_list_arg[0]
    def make_shared_env_fn():
        env_fn = [
            make_env(sf_game, state=eval_state, side=args.side, reset_type=args.reset, rendering=args.render,
                     enable_combo=args.enable_combo, null_combo=args.null_combo,
                     transform_action=args.transform_action, seed=0)
        ]
        return env_fn

    # 2. Create the single, shared VecEnv instance
    print(f"Worker {os.getpid()}: Creating shared environment...")
    #dummy = VecTransposeImage2P(DummyVecEnv2P(make_shared_env_fn()))
    class Dummy:
        pass
    # Load models within the worker process
    print(f"Worker {os.getpid()}: Loading models for num={current_num}")
    _, ego_params, _ = load_from_zip_file(ego_model_path)
    _, br_params, _ = load_from_zip_file(exploiter_model_path)

    #dummy.close()
    #del dummy
    shared_vec_env = VecTransposeImage2P(DummyVecEnv2P(make_shared_env_fn()))
    #env=env_generator()
    ego = Generalist_SPAR(
        "AACCnnPolicy",
        shared_vec_env,
        device="cuda", # Ensure CUDA is set up correctly for multiprocessing if using multiple GPUs, or use "cpu"
        verbose=2,
        n_steps=1536,
        batch_size=768,
        n_epochs=10,
        gamma=0.94,
        v_learning_rate=5e-3, c_learning_rate=1e-4,
        d_learning_rate=5e-4, v_learning_rate_decay=critic_decay_schedule(1e-3),
        c_learning_rate_decay=critic_decay_schedule(1e-4),
        d_learning_rate_decay=critic_decay_schedule(5e-4),
        clip_range=0.1, # Use the actual clip_range_schedule value or pass it
        tensorboard_log=args.log_dir,
        seed=args.seed,
        ent_coef=0,
        dstb_ent_coef=0,
        I_AM_LEFT=True,
        I_AM_RIGHT=False,
        num_adversary=1,
        n_global_env=args.num_env,
        n_env_per_adv=args.num_env // 1,
        opp_list=opponent_list_arg, # Use the passed opponent_list_arg
        player=player_arg, # Use the passed player_arg
        use_mirror=False
    )

    del ego_params['policy.ctrl_optimizer']
    del ego_params['policy.value_optimizer']
    del ego_params['policy.dstb_optimizer']
    ego.set_parameters(ego_params, exact_match=False, device=ego.device)
    exploiter = Exploiter(
        'CnnPolicy',
        shared_vec_env,
        device='cuda',  # Ensure CUDA is set up correctly for multiprocessing if using multiple GPUs, or use "cpu"
        exploited=Dummy(),  # Pass the actual ego model
        n_steps=1024,
        batch_size=512,
        n_epochs=1
    )
    exploiter.env = None
    exploiter.set_parameters(br_params, exact_match=False, device=exploiter.device)
    exploiter.exploited = ego
    eval_env = shared_vec_env.envs[0]
    # Perform the evaluation
    current_win_rates = []
    for j in range(len(state_list_arg)):
        results = evaluate_sa(eval_env, state_list_arg[j], args, ego, exploiter, j, record=True)
        current_win_rates.append(results)

    # Save results for this specific pair
    output_filename = f"{args.finetune_dir}/{args.model_name_prefix}_{current_num}_results.txt"
    with open(output_filename, 'w') as f:
        f.write(str(current_win_rates[0])) # Assuming only one result per pair evaluation

    print(f"Worker {os.getpid()}: Finished evaluation for num={current_num}. Win rate: {current_win_rates[0]}")
    return current_win_rates[0] # Return the win rate for this pair
def main(PLAYER_MAIN):  # Renamed to avoid confusion with global PLAYER
    global PLAYER
    PLAYER = PLAYER_MAIN  # Set the global PLAYER here

    global REMOVAL  # Not currently used for dynamic removal in this context, but kept for consistency
    use_mirror = False
    REMOVAL = None

    if use_mirror is True:
        OPPONENT_LIST = ["Sagat", "EHonda", "MBison"]
    else:
        OPPONENT_LIST = ["Guile"]
    SIDE = "left"
    player_folder_name = PLAYER + '_' + SIDE
    if REMOVAL is not None:
        OPPONENT_LIST.remove(REMOVAL)

    if use_mirror is True:
        STATE_prot_left = [
            "two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name, PLAYER, OPPONENT_LIST[i]) for i
            in range(len(OPPONENT_LIST))]

        opp_left_folder_name = []
        for i in range(len(OPPONENT_LIST)):
            opp_left_folder_name.append(OPPONENT_LIST[i] + "_" + SIDE)
        STATE_prot_right = [
            "two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (opp_left_folder_name[i], OPPONENT_LIST[i], PLAYER)
            for i in range(len(OPPONENT_LIST))]
        STATE = [val for pair in zip(STATE_prot_left, STATE_prot_right) for val in pair]
    else:
        STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name, PLAYER, OPPONENT_LIST[i])
                 for i in range(len(OPPONENT_LIST))]

    # Get command line arguments (important to do this ONCE in the main process)
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'],
                        help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument('--model-file', help='The model to continue to learn from')
    parser.add_argument('--save-dir', help='The directory to save the trained models',
                        default="trained_models/exploiting_%s_12_ippo_match" % PLAYER_MAIN)
    parser.add_argument('--log-dir', help='The directory to save logs', default="logs")
    parser.add_argument('--model-name-prefix', help='The prefix of the model names to save',
                        default="ppo_%s" % PLAYER_MAIN)
    parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile',
                        default=SF_DEFAULT_STATE)
    parser.add_argument('--side', help='The side for AI to control. By default both', default='both',
                        choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=64)
    parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(10e8))
    parser.add_argument('--video-dir', help='The path to save videos', default='videos/spar_spar_%s' % PLAYER_MAIN)
    parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    parser.add_argument('--init-level', type=int,
                        help='Initial level to load from. By default 0, starting from pretrain', default=0)
    parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain',
                        default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    parser.add_argument('--update-left', type=int, help='Update left policy', default=1)
    parser.add_argument('--update-right', type=int, help='Update right policy', default=1)
    parser.add_argument('--left-model-file', help='The left model to continue to learn from')
    parser.add_argument('--right-model-file', help='The right model to continue to learn from')
    parser.add_argument('--other-timescale', type=float, help='Other agent learning rate scale', default=1.0)
    parser.add_argument('--fsp', action='store_true', help='Fictitious self-play')
    parser.add_argument('--fsp-threshold', type=float, help='Fictitious self-play threshold', default=0.5)
    parser.add_argument('--async-update', action='store_true', help='Update left and right asynchronously')
    parser.add_argument("--player", type=str, required=True)
    args = parser.parse_args()

    print("command line args:" + str(args))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.finetune_dir, exist_ok=True)

    ego_folder = '/home/jw4406/codebase/FightLadder/main/trained_models/ego_models/'
    exploiter_folder = "/home/jw4406/codebase/FightLadder/main/trained_models/br_models/"

    nums = []
    ego_beginning = "ppo_Guile_"
    ego_ending = "_steps.task"
    br_beginning = "br_to_ppo_Guile_"
    br_ending = "_steps.task.zip_8200000_steps.zip"

    for fname in os.listdir(ego_folder):
        if fname.startswith(ego_beginning) and fname.endswith(ego_ending):
            try:
                num_str = fname.replace(ego_beginning, "").replace(ego_ending, "")
                nums.append(int(num_str))
            except ValueError:
                continue  # Skip files that don't match the expected naming convention

    nums.sort()

    # Prepare tasks for the pool
    tasks = []
    # The original loop iterates through `nums` and then within that loop,
    # sets `state_list` to a single fixed state for `evaluate_sa`.
    # This means each iteration `i` in `nums` corresponds to evaluating
    # a specific ego/exploiter pair against the *same* fixed state.
    fixed_eval_state_list = ['two_player/Guile_left/Champion.Level1.GuileVsGuile.2Player.state']

    # Convert args object to a dictionary for safe passing to child processes
    # Only include arguments that are needed by the worker function
    args_dict_for_worker = {
        'reset': args.reset, 'model_file': args.model_file, 'save_dir': args.save_dir,
        'log_dir': args.log_dir, 'model_name_prefix': args.model_name_prefix, 'state': args.state,
        'side': args.side, 'render': args.render, 'num_env': args.num_env,
        'num_episodes': args.num_episodes, 'num_epoch': args.num_epoch, 'total_steps': args.total_steps,
        'video_dir': args.video_dir, 'finetune_dir': args.finetune_dir, 'init_level': args.init_level,
        'resume_epoch': args.resume_epoch, 'enable_combo': args.enable_combo, 'null_combo': args.null_combo,
        'transform_action': args.transform_action, 'seed': args.seed, 'update_left': args.update_left,
        'update_right': args.update_right, 'left_model_file': args.left_model_file,
        'right_model_file': args.right_model_file,
        'other_timescale': args.other_timescale, 'fsp': args.fsp, 'fsp_threshold': args.fsp_threshold,
        'async_update': args.async_update, 'player': args.player
    }

    for i in range(len(nums)):
        current_num = nums[i]
        ego_model_path = ego_folder + ego_beginning + str(current_num) + ego_ending
        exploiter_model_path = exploiter_folder + br_beginning + str(current_num) + br_ending

        # We need to pass all information required by `run_evaluation_for_pair`
        # including the specific state_list for evaluation
        tasks.append((ego_model_path, exploiter_model_path, current_num,
                      PLAYER_MAIN, OPPONENT_LIST, SIDE, fixed_eval_state_list,
                      args_dict_for_worker))

    wrs = []
    # Use a multiprocessing Pool
    # You can set the number of processes, e.g., processes=multiprocessing.cpu_count()
    # or a specific number like processes=4
    num_processes = 8  # Or choose a specific number
    print(f"Starting parallel evaluation with {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # map applies the function to each item in the iterable in parallel
        # It returns results in the order of the input iterable
        all_results = pool.map(run_evaluation_for_pair, tasks)
        print("minimal test results", all_results)
        wrs.extend(all_results)  # Collect all win rates

    print("All parallel evaluations complete.")
    print("Collected Win Rates:", wrs)


if __name__ == "__main__":
    # Ensure this part is simple and only handles command line arguments
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("--- Multiprocessing start method set to 'spawn' ---")
    except RuntimeError:
        # This will happen if the context is already set and can't be changed.
        # It's usually fine, but setting it explicitly is best practice.
        print("--- Multiprocessing context already set ---")

    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=str, required=True)
    args = parser.parse_args()

    # Call main function with the player argument
    main(args.player)