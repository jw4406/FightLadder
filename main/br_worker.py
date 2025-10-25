# br_worker.py
import argparse
import os
from annotated_types import Ge
import time, av
import copy
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
# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
TASK_DIR = os.path.join(current_dir, "trained_models/tasks")
PROCESSING_DIR = os.path.join(current_dir, "trained_models/tasks/processing")
DONE_DIR = os.path.join(current_dir, "trained_models/tasks/done")
ERROR_DIR = os.path.join(current_dir, "trained_models/tasks/error")
BR_MODEL_DIR = os.path.join(current_dir, "trained_models/br_models")
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
BR_TRAINING_STEPS = 100
BR_TRAINING_STEPS = 10000000

PLAYER = "Guile"
SIDE = "left"
player_folder_name = PLAYER + '_' + SIDE
video_dir = 'videos/spar_spar_%s' % PLAYER
STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name, PLAYER, PLAYER) for i in range(1)]

# TODO: this is static right now. need to make this dynamic based on the state list from the task file.

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
def evaluate_single_iter_prot_prot(curr_state: str, use_mirror: bool, model: torch.nn.Module, exploiter_model: torch.nn.Module, env_index: int, greedy: int=0, record: bool=False) -> bool:
        """
        This function evaluates a single episode and returns win or loss.
        
        Args: TODO: Complete the docstring

        Returns True if won and False otherwise.
        """
        
        env = env_generator()
        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]
        left_rew = 0
        while not done:
            obs_tensor = obs_as_tensor(obs, model.device)
            #obs_tensor = torch.unsqueeze(obs_tensor, 0)
            #TODO: This if is not very clean: can probably be replaced with a single call to predict.
            if use_mirror is True:
                (action, _states), (_, _) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
                (action_other, _), (_, _) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
                #exploit_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)
                #action_other = exploit_action
            else:
                (action, _states), (action_other, _states_other) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
            #br_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)

            #action_other = br_action
            obs, reward, reward_other, done, info = env.step(np.hstack([action, action_other]))
            left_rew += reward
            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))

            if done:
                if record:
                    try:
                        name = curr_state.split("/")[1]
                    except:
                        name = curr_state
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{video_dir}/{name}_episode_{j}.mp4", mode='w')
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

        env.close()
        return agent_win(info), left_rew
@torch.no_grad()
def evaluate_single_iter_prot_adv(curr_state: str, use_mirror: bool, model: torch.nn.Module, exploiter_model: torch.nn.Module, env_index: int, greedy: int=0, record: bool=False, worker_number: int=0, episode_number: int=0) -> bool:
        """
        This function evaluates a single episode and returns win or loss.
        
        Args: TODO: Complete the docstring

        Returns True if won and False otherwise.
        """
        
        env = make_env(sf_game, state=curr_state, side='both', reset_type='round', rendering=False,
                 enable_combo=False, null_combo=False,
                 transform_action=False, seed=0)().env
        env = env_generator()
        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]
        left_rew = 0
        while not np.any(done):
            obs_tensor = obs_as_tensor(obs, model.device)
            #obs_tensor = torch.unsqueeze(obs_tensor, 0)
            #TODO: This if is not very clean: can probably be replaced with a single call to predict.
            if use_mirror is True:
                (action, _states), (action_other, _) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
                #exploit_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)
                #action_other = exploit_action
            else:
                (action, _states), (action_other, _states_other) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs_tensor, env_index=env_index, deterministic=False)
            #br_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)
            action = action.cpu().numpy()
            action_other = action_other.cpu().numpy()
            #action_other = br_action
            obs, reward, reward_other, done, info = env.step(np.hstack([action, action_other]))
            left_rew += reward
            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))

            if np.any(done):
                if record:
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

        env.close()
        return [agent_win(info[0]), left_rew] # TODO: is a tuple 

@torch.no_grad()
def evaluate_single_iter_exploiter(curr_state: str, use_mirror: bool, model: torch.nn.Module, exploiter_model: torch.nn.Module, env_index: int, greedy: int=0, record: bool=False) -> bool:
        """
        This function evaluates a single episode and returns win or loss.
        
        Args: TODO: Complete the docstring

        Returns True if won and False otherwise.
        """
        
        env = make_env(sf_game, state=curr_state, side='both', reset_type='round', rendering=False,
                 enable_combo=False, null_combo=False,
                 transform_action=False, seed=0)().env
        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]
        left_rew = 0
        while not done:
            #TODO: This if is not very clean: can probably be replaced with a single call to predict.
            if use_mirror is True:
                (action, _states), (_, _) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs, env_index=env_index, deterministic=False)
                exploit_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)
                action_other = exploit_action
            else:
                (action, _states), (action_other, _states_other) = generalist_SPAR_predict(use_mirror=use_mirror, policy=model, obs=obs, env_index=env_index, deterministic=False)
            br_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)

            action_other = br_action
            obs, reward, reward_other, done, info = env.step(np.hstack([action, action_other]))
            left_rew += reward
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

        env.close()
        return agent_win(info), left_rew
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

def evaluate_single_iter(curr_state: str, use_mirror: bool, model: torch.nn.Module,record: bool, env_index: int, exploiter_model: torch.nn.Module = None, greedy: int=0, eval_prot: bool = False)-> bool:
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

    env = env_generator()
    obs = env.reset()
    done = False
    video_log = []
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
        
        obs, reward, reward_other, done, info = env.step(np.hstack([left_action, right_action]))
        left_rew += reward
        if record:
            video_log.append(Image.fromarray(env.render(mode="rgb_array")))
    return agent_win(info[0]), left_rew


def evaluate_sa_worker(curr_state: str, use_mirror: bool, model: torch.nn.Module, exploiter_model: torch.nn.Module, env_index: int, return_list: DictProxy, pid: int, episodes: int, greedy: int=0, record: bool=False, eval_prot: bool = False):
    try:
        device = select_device()
        model.eval().to(device)
        exploiter_model.eval().to(device)

        win_count = 0
        rew_arr = []
        for ep_num in range(episodes):
            joined_win_rew = evaluate_single_iter(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record, eval_prot=eval_prot)
            win_count += joined_win_rew[0]
            rew_arr.append(joined_win_rew[1])
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
        return_list[pid] = [win_count, rew_arr]
    except Exception as e:
        print(f"Worker {pid} failed with exception {e}")
        raise

def evaluate_sa_parallel(curr_state: str, model: Generalist_SPAR, exploiter_model: Exploiter, env_index: int, greedy: int=0, record: bool=False, num_episodes: int=50, num_workers: int=10, use_mirror: bool=False, eval_prot: bool = False) -> float:
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
        # evaluate_sa_worker(curr_state, model.use_mirror, policy_copy, exploiter_policy_copy, env_index, return_list, pid, episodes, greedy, record)
        p = mp.Process(
                target=evaluate_sa_worker,
                args=(curr_state, use_mirror, policy_copy, exploiter_policy_copy, env_index, return_list, pid, episodes, greedy, record, eval_prot)
                )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    rew_list = []
    total_wins = sum(return_list.values()[i][0] for i in range(len(return_list)))
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

def env_generator():
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

def exploiter_env_generator():
    # STATE
    each_env_count = 4
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
def train_best_response(task_file_path: str, eval_prot: bool, use_mirror: bool) -> None:
    """
    The core logic for a single best-response training run.

    Args:
        TODO: Complete this.
    """
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
    env = env_generator()
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
    use_mirror = ftm.use_mirror
    
    #OVERRIDEN HERE
    
    
    # Read the path of the frozen policy from the task file
    #data, params, pytorch_variables = load_from_zip_file(
    #    checkpoint_path)

    #del params['policy.ctrl_optimizer']
    #del params['policy.value_optimizer']
    #del params['policy.dstb_optimizer']
    # finetune_model.warmstarted_cont_MAGICS = True
    # finetune_model.warmstart_setup(finetune_model.lr_schedule)
    #finetune_model.set_parameters(params, exact_match=False, device=finetune_model.device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    # --- This is where your specific BR logic goes ---
    # 1. Load the frozen opponent
    # fixed_opponent = PPO.load(checkpoint_path)
    wandb.init(project="exploiter",
                entity='jw4406',
                group="br_workers",
                config={"eval_rew": 0,
                        "epochs": 0,
                        "br_wr": 0})
    # 2. Create your environment, passing the frozen opponent to it
    #    so the BR agent can play against it.
    # env = YourStreetFighterEnv(opponent_policy=fixed_opponent)
    env = exploiter_env_generator()

    # 3. Create a new agent to be the best response
    br_agent = Exploiter('CnnPolicy', exploiter_env_generator(), device='cuda', exploited=ftm, n_steps=1024, batch_size=512, n_epochs=10, exploiting='ego')

    # 4. Train the BR agent
    br_model_name = f"br_to_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.zip"
    exploiter_callback = ExploiterCheckpointCallback(save_freq=100, save_path=BR_MODEL_DIR, name_prefix=br_model_name)
    #br_agent.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback)

    # eval BR against ego right here! both models are already in namespace.

    wr, mean_rew = evaluate_sa_parallel(curr_state=STATE[0], model=ftm, exploiter_model=br_agent, env_index=0, record=True, use_mirror=use_mirror, eval_prot=eval_prot)
    
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

    print(f"WORKER [{worker_id}]: Successfully trained and saved {br_model_name}")
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
    args = parser.parse_args()


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
            train_best_response(processing_path, eval_prot=args.eval_prot, use_mirror=args.use_mirror)

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

    print(f"WORKER [{os.getpid()}]: Stop file detected. Shutting down.")
