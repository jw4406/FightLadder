# br_worker.py
import argparse
import os
from annotated_types import Ge
import time, av
import copy
from train_ma import constructor
import types
from common.league import PayoffManager, League, FSPLeague, PSROLeague, Learner
import random
import retro
from PIL import Image
from stable_baselines3.common.utils import obs_as_tensor
import wandb
import torch
import torch.multiprocessing as mp
from multiprocessing.managers import DictProxy
from common.justin.Generalist_SPAR import Generalist_SPAR, generalist_SPAR_predict
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.const import *
from common.retro_wrappers import SFWrapper, Monitor2P
from common.algorithms import Exploiter
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P
from stable_baselines3.common.callbacks import CheckpointCallback, ExploiterCheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file
from utils import agent_win, select_device
import subprocess
# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
TASK_DIR = os.path.join(current_dir, "trained_models/tasks")
#TASK_DIR = '/n/fs/magics/2415498/FightLadder/main/trained_models/tasks/'
PROCESSING_DIR = os.path.join(current_dir, "trained_models/tasks/processing")
DONE_DIR = os.path.join(current_dir, "trained_models/tasks/done")
ERROR_DIR = os.path.join(current_dir, "trained_models/tasks/error")
BR_MODEL_DIR = os.path.join(current_dir, "trained_models/tasks/br_models")
WR_STATS_DIR = os.path.join(current_dir, "trained_models/wr_stats")
MEAN_REW_STATS_DIR = os.path.join(current_dir, "trained_models/mean_rew_stats")
os.makedirs(BR_MODEL_DIR, exist_ok=True)
os.makedirs(TASK_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)
os.makedirs(DONE_DIR, exist_ok=True)
os.makedirs(WR_STATS_DIR, exist_ok=True)
os.makedirs(MEAN_REW_STATS_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

if not os.listdir(TASK_DIR):
    print("Warning: The TASK_DIR is empty. Please run ippo.py --player PLAYER to generate a task file.")

POLL_INTERVAL = 5  # Seconds to wait before checking for new tasks
BR_TRAINING_STEPS = 10000000
BR_TRAINING_STEPS = 100

PLAYER = "Guile"
OPPONENT_LIST = ["Guile"]
SIDE = "left"
player_folder_name = PLAYER + '_' + SIDE
video_dir = 'videos/single_1v2_%s' % PLAYER
STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name, PLAYER, OPPONENT_LIST[i]) for i in range(len(OPPONENT_LIST))]

# TODO: this is static right now. need to make this dynamic based on the state list from the task file.

def load_league_models(character_names: list, model_dir: str = None) -> dict:
    """
    Load model checkpoint files from trained_models/ma directory, separated by player type (LE, MA, ME)
    and side (left, right). Includes all LE, MA, and ME files regardless of whether they contain character names.
    
    Args:
        character_names: List of character names (e.g., ["ryu", "bison", "guile"]). 
                         Currently not used for filtering - all LE/MA/ME files are included.
        model_dir: Directory to search for model files. Defaults to trained_models/ma relative to current_dir.
    
    Returns:
        Nested dictionary with structure:
        {
            'LE': {'left': {...}, 'right': {...}},
            'MA': {'left': {...}, 'right': {...}},
            'ME': {'left': {...}, 'right': {...}},
            'payoff': <latest payoff file path or None>
        }
        Keys are filenames without .pt extension (trailing step removed, prefix/index normalized),
        values are full file paths.
    """
    if model_dir is None:
        model_dir = os.path.join(current_dir, "trained_models", "ma")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Separate files by player type and side; track latest payoff
    model_files = {
        'LE': {'left': {}, 'right': {}},  # League Exploiters
        'MA': {'left': {}, 'right': {}},  # Main Agents
        'ME': {'left': {}, 'right': {}},  # Main Exploiters
        'payoff': None,
    }
    latest_payoff_mtime = 0
    
    # Get all .pt files in the directory
    for filename in os.listdir(model_dir):
        if not filename.endswith('.pt'):
            continue
        
        # Track latest payoff file
        if filename.startswith('payoff_'):
            payoff_path = os.path.join(model_dir, filename)
            mtime = os.path.getmtime(payoff_path)
            if mtime > latest_payoff_mtime:
                latest_payoff_mtime = mtime
                model_files['payoff'] = payoff_path
            continue
        
        full_path = os.path.join(model_dir, filename)
        
        # Build key from filename:
        # - Strip '.pt'
        # - Remove trailing segment (typically a step index like '_0')
        # - Split the first token into prefix letters and index digits
        #   e.g. 'LE1_left_0' -> ['LE1', 'left', '0']
        #        'LE1' -> prefix='LE', idx='1' -> 'LE_1_left'
        base_name = filename[:-3]  # Remove '.pt'
        parts = base_name.split('_')
        if len(parts) < 2:
            # Fallback: just use base_name
            file_key = base_name
        else:
            # Drop the last segment (trailing numeric step)
            core_parts = parts[:-1]
            first = core_parts[0]
            prefix = ''.join(ch for ch in first if ch.isalpha())
            index = ''.join(ch for ch in first if ch.isdigit())
            if prefix and index:
                # e.g. 'MA0' -> prefix='MA', index='0' -> 'MA0'
                new_first = f"{prefix}{index}"
            else:
                new_first = first
            file_key = '_'.join([new_first] + core_parts[1:])
        
        # Determine side (left or right)
        if '_left_' in filename:
            side = 'left'
        elif '_right_' in filename:
            side = 'right'
        else:
            # Skip files that don't have a clear left/right designation
            continue
        
        # Categorize by prefix - include ALL LE, MA, ME files
        # (both with and without character names)
        if filename.startswith('LE'):
            model_files['LE'][side][file_key] = full_path
        elif filename.startswith('MA'):
            model_files['MA'][side][file_key] = full_path
        elif filename.startswith('ME'):
            model_files['ME'][side][file_key] = full_path
    
    return model_files, payoff_path

def gen_dummy_policy(exploiter_model: Exploiter) -> torch.nn.Module:
    """
    This function creates a dummy copy of exploiter_model.
    It is needed when we need to pass a copy of exploiter_model.policy (cannot use deepcopy).

    Args: TODO: Complete the docstring

    Returns:
        A copy of exploiter_model.policy.
    """

    # Step 1: Get the class of the policy
    policy_cls = type(exploiter_model.policy)
    
    # Step 2: Instantiate a new policy with the same architecture
    # (This is the tricky part: SB3 policies require obs_space, act_space, net args, etc.)
    # The easiest way is to pull those from the existing policy:
    new_policy = policy_cls(
                            observation_space=exploiter_model.policy.observation_space,
                            action_space=exploiter_model.policy.action_space,
                            lr_schedule=lambda _: 0.0,  # Doesn't matter for inference
                            )
    
    # Step 3: Load weights
    new_policy.load_state_dict(exploiter_model.policy.state_dict())

    # Step 4: Move to CPU
    return new_policy.cpu()
    

@torch.no_grad()
def save_episode_video(curr_state, video_log: list, name: str, worker_number: int = 0, episode_number: int = 0):
    try:
        name = curr_state.split("/")[1]
    except:
        name = curr_state
    height, width, layers = np.array(video_log[0]).shape
    container = av.open(f"{video_dir}/{name}_worker_num_{worker_number}_episode_{episode_number}.mp4", mode='w')
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

def evaluate_single_iter(curr_state: str, use_mirror: bool, model: torch.nn.Module,record: bool, env_index: int, exploiter_model: torch.nn.Module = None, greedy: int=0, eval_prot: bool = False, video_dir: str = None)-> bool:
    # flag checks

    # cases:
    # 1. use_mirror is True and eval_prot is True: play prot against prot, exploiter model is None
    # 2. use_mirror is True and eval_prot is False: play prot against adv, exploiter model is None
    # 3. exploiter_model is not None and eval_prot is True: play prot against exploiter, use_mirror is False
    # 4. exploiter_model is not None and eval_prot is False: play adv against exploiter, use_mirror is False
    # 5. use_mirror is False, exploiter model is None, eval_prot is False: play prot against adv (this is the default case)
    # 6. use_mirror is False, exploiter model is None, eval_prot is True: raise error (this is not possible)

    if use_mirror is True:
        if exploiter_model is not None:
            raise ValueError("Exploiter model should not be used when use_mirror is True -- we just play prot against adv")
    if use_mirror is False:
        if exploiter_model is None:
            raise ValueError("Exploiter model should not be None when use_mirror is False")
    if use_mirror is False and exploiter_model is None and eval_prot is False:
        print("playing default: prot against adv")
    if use_mirror is False and exploiter_model is None and eval_prot is True:
        raise ValueError("Eval prot should not be used when use_mirror is False and exploiter model is None")
    if record:
        if video_dir is None:
            raise ValueError("Video directory is not set but record is True")
    env = env_generator()
    obs = env.reset()
    done = False
    video_log = []
    left_rew = np.zeros(len(OPPONENT_LIST))
    while not np.any(done):
        obs_tensor = obs_as_tensor(obs, model.device)
        if exploiter_model is not None:
            (action, _states), (action_other, _states_other) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
            exploiter_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)
            if eval_prot is True:
                left_action = action
                right_action = action_other
            else:
                left_action = action_other
                right_action = action
        else:
            if use_mirror is True and eval_prot is True:
                (action, _states), (_, _) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
                (action_other, _states_other), (_, _) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
                left_action = action
                right_action = action_other
            elif use_mirror is True and eval_prot is False:
                (_, _), (action, _states_other) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
                (_, _), (action_other, _states) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
                left_action = action
                right_action = action_other
            elif use_mirror is False and eval_prot is False:
                (action, _states), (action_other, _states_other) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
                left_action = action
                right_action = action_other
            else:
                print("current flags: use_mirror={use_mirror}, eval_prot={eval_prot}, exploiter_model={exploiter_model}")
                raise ValueError("got impossible flag combination")
        left_action = left_action.cpu().numpy()
        right_action = right_action.cpu().numpy()
        obs, reward, reward_other, done, info = env.step(np.hstack([left_action, right_action]))
        left_rew += reward
        if record:
            video_log.append(Image.fromarray(env.render(mode="rgb_array")))
    
    return [agent_win(info[i]) for i in range(len(OPPONENT_LIST))], left_rew, video_log


def evaluate_sa_worker(curr_state: str, use_mirror: bool, model: torch.nn.Module, exploiter_model: torch.nn.Module, env_index: int, return_list: DictProxy, pid: int, episodes: int, greedy: int=0, record: bool=False, eval_prot: bool = False, video_dir: str = None):
    try:
        device = select_device()
        model.eval().to(device)
        if exploiter_model is not None:
            exploiter_model.eval().to(device)
        else:
            exploiter_model = None

        win_count = np.zeros(len(OPPONENT_LIST))
        rew_arr = []
        video_log = []
        for ep_num in range(episodes):
            exploiter_model = None if use_mirror is True else exploiter_model
            joined_win_rew = evaluate_single_iter(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record, eval_prot=eval_prot, video_dir=video_dir)
            win_count += joined_win_rew[0]
            rew_arr.append(joined_win_rew[1])
            video_log.append(joined_win_rew[2])
            # if type(model) is type(exploiter_model):
            #     joined_win_rew = evaluate_single_iter_exploiter(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record)
            #     win_count += joined_win_rew[0]
            #     rew_arr.append(joined_win_rew[1])
            #     #win_count += evaluate_single_iter_exploiter(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record)
            # elif use_mirror is True and eval_prot is True:
            #     joined_win_rew = evaluate_single_iter_prot_prot(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record)
            #     win_count += joined_win_rew[0]
            #     rew_arr.append(joined_win_rew[1])
            #     #win_count += evaluate_single_iter_prot_prot(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record)
            # else:
            #     #assert use_mirror is True
            #     joined_win_rew = evaluate_single_iter_prot_adv(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record, worker_number=pid, episode_number=ep_num)
            #     win_count += joined_win_rew[0]
            #     rew_arr.append(joined_win_rew[1])
            #     #win_count += evaluate_single_iter_prot_adv(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record)
        return_list[pid] = [win_count, rew_arr, video_log]
    except Exception as e:
        print(f"Worker {pid} failed with exception {e}")
        raise

def evaluate_sa_parallel(curr_state: str, model: Generalist_SPAR, exploiter_model: Exploiter, env_index: int, greedy: int=0, record: bool=False, num_episodes: int=50, num_workers: int=10, use_mirror: bool=False, eval_prot: bool = False, video_dir: str = None) -> float:
    #Set up multiprocessing
    if __name__=="__main__":
        mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    return_list = manager.dict()
    processes = []

    #Calculate episodes per worker
    episodes = num_episodes//num_workers
    if num_episodes % num_workers > 0:
        print(f"Warning: The total number of episodes ({num_episodes}) is not divisible by the number of workers ({num_workers}).")

    for pid in range(num_workers):
        #Sharing the models across processes is not safe - create copies
        #Create CPU versions of the models for deepcopy (safer than deepcopying GPU models)
        policy_copy = copy.deepcopy(model.policy).cpu()
        exploiter_policy_copy = gen_dummy_policy(exploiter_model)
        
        #The following line can be used for serial debugging
        #evaluate_sa_worker(curr_state, use_mirror, policy_copy, exploiter_policy_copy, env_index, return_list, pid, episodes, greedy, record, eval_prot, video_dir)
        p = mp.Process(
                target=evaluate_sa_worker,
                args=(curr_state, use_mirror, policy_copy, exploiter_policy_copy, env_index, return_list, pid, episodes, greedy, record, eval_prot, video_dir)
                )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    rew_list = []
    total_wins = sum(return_list.values()[i][0] for i in range(len(return_list)))
    video_logs = [return_list.values()[i][2] for i in range(len(return_list))]
    count = 0
    for i in range(len(video_logs)):
        for j in range(len(video_logs[i])):
            save_episode_video(curr_state, video_logs[i][j], PLAYER, worker_number=i, episode_number=j)
            count += 1
    rew_list.extend(return_list.values()[i][1] for i in range(len(return_list)))
    flat_rew_list = [item for sublist in rew_list for item in sublist]
    avg_rew = sum(flat_rew_list) / len(flat_rew_list)
    win_rate = total_wins / num_episodes
    print(f"Winning rate: {win_rate:.2f}")
    return win_rate, avg_rew    


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

def env_generator(STATE=None):
    # STATE
    each_env_count = 1
    env = []
    for i in range(len(STATE)):
        for j in range(each_env_count):
            env.append(
                make_env(sf_game, state=STATE[i], side='both', reset_type='round', rendering=False,
                         enable_combo=False, null_combo=False,
                         transform_action=False, seed=0))
    # env = [make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=0) for i in range(args.num_env)]
    # env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render,
    #         enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action,
    #         seed=0)
    return VecTransposeImage2P(SubprocVecEnv2P(env))

def exploiter_env_generator(STATE=None):
    # STATE
    each_env_count = 1
    env = []
    for i in range(len(STATE)):
        for j in range(each_env_count):
            env.append(
                make_env(sf_game, state=STATE[i], side='both', reset_type='round', rendering=False,
                         enable_combo=False, null_combo=False,
                         transform_action=False, seed=0))
    # env = [make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=0) for i in range(args.num_env)]
    # env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render,
    #         enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action,
    #         seed=0)
    return VecTransposeImage2P(SubprocVecEnv2P(env))
# --- Worker Logic ---
def instantiate_league_models(files, character_names: list):
    opponent_names = character_names
    right_models = {}
    args = types.SimpleNamespace()
    args.enable_combo = False
    args.null_combo = False
    args.transform_action = False
    args.seed = 0
    args.fsp_league = False
    args.psro_league = False
    args.num_env = 1
    args.side = "both"
    args.reset = "round"
    args.render = False
    args.log_dir = os.path.join(current_dir, "trained_models", "br_logs")
    args.save_dir = os.path.join(current_dir, "trained_models", "br_payoffs")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    for opponent in opponent_names:
        right_models[opponent] = constructor(args, "right", log_name=None, single_env=True, opponent=opponent)
    left_model = constructor(args, "left", log_name=None, single_env=True)
    initial_agents = {'left': left_model, 'right': right_models}

    with PayoffManager() as manager:
        shared_payoff = manager.Payoff(args.save_dir)
        if args.fsp_league:
            league = FSPLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        elif args.psro_league:
            league = PSROLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        else:
            league = League(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1, main_exploiters=1, league_exploiters=2)
        # Create a flat lookup dictionary from the nested structure
        # Maps player name (filename without .pt) to full file path
        model_files_flat = {}
        for player_type in ['LE', 'MA', 'ME']:
            for side in ['left', 'right']:
                model_files_flat.update(files[player_type][side])
        
        processes = []
        for idx in range(league.size()):
            player = league.get_player(idx)
            if player.name in model_files_flat:
                player.load(model_files_flat[player.name])

            else:
                print(f"Warning: No checkpoint file found for player {player.name}")
        
        print(f"Loaded models for players: {list(model_files_flat.keys())}\n")
        # Load latest payoff file if available
        payoff_file = files.get('payoff')
        if payoff_file:
            shared_payoff.load(payoff_file)
        else:
            print("Warning: No payoff file found to load.")
        [league.get_player(i).construct_agent() for i in range(league.size())]
        print("hello")
        return league

def load_spar_model(task_file_path: str) -> None:
    worker_id = os.getpid()
    print(f"WORKER [{worker_id}]: Processing task: {os.path.basename(task_file_path)}")

    #try:
    # The task file IS the model checkpoint file, just renamed.
    checkpoint_path = task_file_path

    # Extract timestep from the checkpoint filename for wandb logging
    try:
        basename = os.path.basename(checkpoint_path)
        # Assumes format like '..._12345_steps.task'
        timestep_str = basename.replace('.task', '').split('_')[-2]
        ego_timestep = int(timestep_str)
    except (IndexError, ValueError):
        print(f"WORKER [{worker_id}]: Could not parse timestep from filename: {basename}. BR win rate will not be logged against a specific step.")
        ego_timestep = None

    """ finetune_model = Generalist_SPAR(
        "AACCnnPolicy",
        env_generator(),
        device="cuda",
        verbose=2,
        n_steps=96,  # 1408,
        batch_size=192,  # 2816,  # 512,
        n_epochs=5,
        gamma=0.99,
        v_learning_rate=5e-5, c_learning_rate=1e-6,
        d_learning_rate=2e-6, v_learning_rate_decay=critic_decay_schedule(1e-3),
        c_learning_rate_decay=critic_decay_schedule(1e-4),
        d_learning_rate_decay=critic_decay_schedule(5e-4),
        clip_range=linear_schedule(0.075, 0.025),
        tensorboard_log='logs',
        seed=0,
        ent_coef=.01,
        dstb_ent_coef=.01,
        I_AM_LEFT=True,
        I_AM_RIGHT=False,
        num_adversary=1,
        n_global_env=4,
        n_env_per_adv=4,
        opp_list=[PLAYER],
        player=PLAYER,
        use_mirror=False
    ) """
    data, _, _ = load_from_zip_file(checkpoint_path)
    # get the saved matchups -- make this automated so that we dont hit stupid user errors
    #matchups = data['matchups']
    uniques = list(dict.fromkeys(data['state_list']).keys())
    STATE = uniques
    env = env_generator(STATE=STATE)
    env.num_envs = 1 # HACKY FOR NOW!
    try:
        ftm = CleanDerivativeFreeSPAR.load(path=checkpoint_path, env=env, num_perturbed=1)
        #if ftm.policy.num_env_per_adv is None:
        #    ftm.policy.num_env_per_adv = ftm.envs_per_matchup
    except Exception as e:
        data, params, pytorch_variables = load_from_zip_file(
            checkpoint_path)
        ftm = CleanDerivativeFreeSPAR(
            "AACCnnPolicy",
            env,
            device="cuda",
            verbose=2,
            n_steps=256,
            batch_size=512,
            n_epochs=1,
            state_list=STATE,
            envs_per_matchup=1,
            env_generator_func=env_generator,
            num_adversaries=1,
            n_env_per_adv=1,
            seed= 0,
            target_kl=0.025,
            use_mirror=False
        )
        ftm.set_parameters(params, exact_match=True, device=ftm.device)
    return ftm
def train_best_response(model_to_exploit, task_file_path: str, eval_prot: bool, use_mirror: bool, eval_only: bool, proj_name: str, analysis_upload_proj_name: str, is_spar: bool = False) -> None:
    """
    The core logic for a single best-response training run.

    Args:
        TODO: Complete this.
    """
    checkpoint_path = task_file_path
    ftm = model_to_exploit
    # --- This is where your specific BR logic goes ---
    # 1. Load the frozen opponent
    # fixed_opponent = PPO.load(checkpoint_path)
    wandb.init(project=proj_name,
                entity='jw4406',
                group="br_workers",
                config={"eval_rew": 0,
                        "exploiter_rew": 0,
                        "epochs": 0,
                        "br_wr": 0,
                        "main_training_epoch": 0,
                        })
    # 2. Create your environment, passing the frozen opponent to it
    #    so the BR agent can play against it.
    # env = YourStreetFighterEnv(opponent_policy=fixed_opponent)
    env = exploiter_env_generator(STATE=STATE)

    # 3. Create a new agent to be the best response
    br_agent = Exploiter('CnnPolicy', exploiter_env_generator(STATE=STATE), device='cuda', exploited=ftm, n_steps=1024, batch_size=512, n_epochs=10, exploiting='ego')
    br_agent.is_spar = is_spar # TODO: This is a stupid hack to get the BR agent to know if it is a SPAR model or not. Remove this once we have a better way to do this.
    # 4. Train the BR agent
    br_model_name = f"br_to_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.zip"
    exploiter_callback = ExploiterCheckpointCallback(save_freq=100000, save_path=BR_MODEL_DIR, name_prefix=br_model_name)
     
    if eval_only == False:
        print("eval_only was passed as False. Training the BR agent.")
        br_agent.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback)
        agg_file = os.path.join(current_dir, "aggregate_to_wandb.py")
        subprocess.Popen(["python", agg_file, "--read_from_proj_name", proj_name, "--upload_to_proj_name", analysis_upload_proj_name])
    # eval BR against ego right here! both models are already in namespace.

    """ wr, mean_rew = evaluate_sa_parallel(curr_state=STATE[0], model=ftm, exploiter_model=br_agent, env_index=0, record=True, use_mirror=use_mirror, eval_prot=eval_prot, video_dir=video_dir)
    # delete this line this is hacky. we should use command flags to determine what agents to video

    
    if ego_timestep is not None:
        wr_filename = os.path.join(WR_STATS_DIR, f"{ego_timestep}.txt")
        with open(wr_filename, 'w') as f:
            f.write(str(wr))
        
        mean_rew_filename = os.path.join(MEAN_REW_STATS_DIR, f"{ego_timestep}.txt")
        with open(mean_rew_filename, 'w') as f:
            f.write(str(mean_rew))
            
    #TODO: Remove the following line once debugging is done
    # wr = evaluate_sa(STATE[0], finetune_model, br_agent, 0, record=False) # do not change False to True
    if use_mirror is False:
        rew_arr = np.zeros(len(br_agent.ep_info_buffer))
        for i in range(len(rew_arr)):
            rew_arr[i] = br_agent.ep_info_buffer[i]['r']
        mean_rew = np.mean(rew_arr)
    if ego_timestep is not None:
        try:
            wandb.log({"br_win_rate_vs_%s" % (br_agent.exploiting): wr, "global_step": ego_timestep})
        except Exception as e:
            print(f"wandb.log() failed with error: {e}")
        
        if not use_mirror:
            wandb.log({"br_mean_reward_vs_%s" % (br_agent.exploiting): mean_rew, "global_step": ego_timestep})
    else:
        wandb.log({"br_win_rate_vs_ego": wr})

    # 5. Save the trained BR model
    # br_agent.save(os.path.join(BR_MODEL_DIR, br_model_name))
    # -------------------------------------------------

    print(f"WORKER [{worker_id}]: Successfully trained and saved {br_model_name}") """
    wandb.finish()

    # except Exception as e:
    #     print(f"WORKER [{worker_id}]: FAILED to process task {os.path.basename(task_file_path)}. Error: {e}")
    #     # Optionally, move the failed task to an "error" directory instead of "done"
    #     # to inspect it later.

if __name__ == "__main__":
    #Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_prot", action="store_true")
    parser.add_argument("--use_mirror", action="store_true")
    parser.add_argument("--eval_only", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--proj_name", type=str, required=True)
    parser.add_argument("--analysis_upload_proj_name", type=str, required=True)
    parser.add_argument("--is_league", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_br", choices=['True', 'False'], default='False', required=True)
    args = parser.parse_args()

    if args.eval_only == 'True':
        print("WARNING!")
        print("This is an EVAL ONLY run. No exploiter training will be performed.")
        print("WARNING!")
    args.eval_only = args.eval_only == 'True'
    wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    todo_dir = os.path.join(TASK_DIR, "todo")
    processing_dir = os.path.join(TASK_DIR, "processing")
    error_dir = os.path.join(TASK_DIR, "error")
    done_dir = os.path.join(TASK_DIR, "done")
    stop_file = os.path.join(TASK_DIR, "STOP")
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # if os.path.isfile(curr_dir + "/myfile.txt"):
    #     import json
    #     test = json.load(open(curr_dir + "/myfile.txt"))
    #     print(test)
    # else:
    #     print("myfile.txt does not exist")

    print(f"WORKER [{os.getpid()}]: Starting. Watching {todo_dir} for tasks.")
    if args.is_league == 'False' and args.load_br == 'False':
        while not os.path.exists(stop_file):
            tasks = [f for f in os.listdir(todo_dir) if f.endswith(".task")]

            if not tasks:
                time.sleep(POLL_INTERVAL)
                continue

            # Grab a random task to reduce the chance of multiple workers grabbing the same one
            task_filename = random.choice(tasks)
            todo_path = os.path.join(todo_dir, task_filename)
            processing_path = os.path.join(processing_dir, task_filename)
            error_path = os.path.join(error_dir, task_filename)

            try:
                # Atomically move the task file to claim it
                os.rename(todo_path, processing_path)

                # Now that we've claimed it, process it
                loaded_model = load_spar_model(processing_path)
                train_best_response(loaded_model,processing_path, eval_prot=args.eval_prot, use_mirror=args.use_mirror, eval_only=args.eval_only, proj_name=args.proj_name, is_spar=True, analysis_upload_proj_name=args.analysis_upload_proj_name)

                # Move it to 'done' when finished
                done_path = os.path.join(done_dir, task_filename)
                os.rename(processing_path, done_path)

            except FileNotFoundError:
                # Another worker grabbed this file first. No problem.
                continue
            except Exception as e:
                print(f"WORKER [{os.getpid()}]: A critical error occurred. Error: {e}")
                # Move the failed task back to todo or to an error folder
                try:
                    os.rename(processing_path, error_path)
                except:
                    pass
    elif args.is_league == 'True' and args.load_br == 'False':
        model_files, payoff_path = load_league_models(model_dir=args.model_dir, character_names=["ryu", "bison", "guile"])
        loaded_league = instantiate_league_models(model_files, character_names=["ryu", "bison", "guile"])
        main_agent_left = loaded_league.get_player(0).agent
        train_best_response(main_agent_left, payoff_path, eval_prot=args.eval_prot, use_mirror=args.use_mirror, eval_only=args.eval_only, proj_name=args.proj_name, is_spar=False, analysis_upload_proj_name=args.analysis_upload_proj_name)
    else:
        while not os.path.exists(stop_file):
            tasks = [f for f in os.listdir(BR_MODEL_DIR) if f.endswith(".task")]

            if not tasks:
                time.sleep(POLL_INTERVAL)
                continue
            ep_info_buffers = []
            rew_arr = np.zeros(len(tasks))
            # Grab a random task to reduce the chance of multiple workers grabbing the same one
            for i in range(len(tasks)):
                loaded_model = Exploiter.load(os.path.join(BR_MODEL_DIR, tasks[i]), env=exploiter_env_generator(STATE=STATE))
                ep_info_buffers.append(loaded_model.ep_info_buffer)
                for j in range(len(loaded_model.ep_info_buffer)):
                    rew_arr[i] = rew_arr[i] + loaded_model.ep_info_buffer[j]['r']
                rew_arr[i] = rew_arr[i] / len(loaded_model.ep_info_buffer)
            
            print("hello")
    print(f"WORKER [{os.getpid()}]: Stop file detected. Shutting down.")

