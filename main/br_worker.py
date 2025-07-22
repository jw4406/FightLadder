# br_worker.py
import os
import time, av
import copy
import random
import retro
from PIL import Image
import wandb
import torch
import torch.multiprocessing as mp
from multiprocessing.managers import DictProxy
from common.justin.Generalist_SPAR import Generalist_SPAR, generalist_SPAR_predict
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
PROCESSING_DIR = os.path.join(current_dir, "trained_models/processing")
DONE_DIR = os.path.join(current_dir, "trained_models/done")
BR_MODEL_DIR = os.path.join(current_dir, "trained_models/br_models")
os.makedirs(BR_MODEL_DIR, exist_ok=True)
os.makedirs(TASK_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)
os.makedirs(DONE_DIR, exist_ok=True)

if not os.listdir(TASK_DIR):
    print("Warning: The TASK_DIR is empty. Please run ippo.py --player PLAYER to generate a task file.")

POLL_INTERVAL = 5  # Seconds to wait before checking for new tasks
BR_TRAINING_STEPS = 100
BR_TRAINING_STEPS = 10000000

PLAYER = "Guile"
SIDE = "left"
player_folder_name = PLAYER + '_' + SIDE

STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name, PLAYER, PLAYER)]

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
def evaluate_single_iter(curr_state: str, use_mirror: bool, model: torch.nn.Module, exploiter_model: torch.nn.Module, env_index: int, greedy: int=0, record: bool=False) -> bool:
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
        return agent_win(info)

def evaluate_sa_worker(curr_state: str, use_mirror: bool, model: torch.nn.Module, exploiter_model: torch.nn.Module, env_index: int, return_list: DictProxy, pid: int, episodes: int, greedy: int=0, record: bool=False):
    try:
        device = select_device()
        model.eval().to(device)
        exploiter_model.eval().to(device)

        win_count = 0
        for _ in range(episodes):
            win_count += evaluate_single_iter(curr_state=curr_state, use_mirror=use_mirror, model=model, exploiter_model=exploiter_model, env_index=env_index, greedy=greedy, record=record)
        return_list[pid] = win_count
    except Exception as e:
        print(f"Worker {pid} failed with exception {e}")
        raise

def evaluate_sa_parallel(curr_state: str, model: Generalist_SPAR, exploiter_model: Exploiter, env_index: int, greedy: int=0, record: bool=False, num_episodes: int=50, num_workers: int=12) -> float:
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
                args=(curr_state, model.use_mirror, policy_copy, exploiter_policy_copy, env_index, return_list, pid, episodes, greedy, record)
                )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_wins = sum(return_list.values())
    win_rate = total_wins / num_episodes
    print(f"Winning rate: {win_rate:.2f}")
    return win_rate    

#TODO: Once we are satisfied with evaluate_sa_parallel, we can remove this function (to avoid code bloating).
@torch.no_grad()
def evaluate_sa(curr_state: str, model: Generalist_SPAR, exploiter_model: Exploiter, env_index: int, greedy: int=0, record: bool=False):
    #assert isinstance(model, Specialized_Agent)
    # global STATE
    num_episodes = 50
    win_cnt = 0
    vic = np.zeros((50,))
    # env = []
    for j in range(1, num_episodes + 1):
        env = make_env(sf_game, state=curr_state, side='both', reset_type='round', rendering=False,
                 enable_combo=False, null_combo=False,
                 transform_action=False, seed=0)().env
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
            # vic[j-1] = 1
            win_cnt += 1

        env.close()

    win_rate = win_cnt / num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate

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
def train_best_response(task_file_path: str):
    """
    The core logic for a single best-response training run.
    """
    worker_id = os.getpid()
    print(f"WORKER [{worker_id}]: Processing task: {os.path.basename(task_file_path)}")

    try:
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

        finetune_model = Generalist_SPAR(
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
        )

        # Read the path of the frozen policy from the task file
        data, params, pytorch_variables = load_from_zip_file(
            checkpoint_path)

        del params['policy.ctrl_optimizer']
        del params['policy.value_optimizer']
        del params['policy.dstb_optimizer']
        # finetune_model.warmstarted_cont_MAGICS = True
        # finetune_model.warmstart_setup(finetune_model.lr_schedule)
        finetune_model.set_parameters(params, exact_match=False, device=finetune_model.device)
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
        br_agent = Exploiter('CnnPolicy', exploiter_env_generator(), device='cuda', exploited=finetune_model, n_steps=1024, batch_size=512, n_epochs=10, exploiting='ego')

        # 4. Train the BR agent
        br_model_name = f"br_to_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.zip"
        exploiter_callback = ExploiterCheckpointCallback(save_freq=100, save_path=BR_MODEL_DIR, name_prefix=br_model_name)
        br_agent.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback)

        # eval BR against ego right here! both models are already in namespace.

        wr = evaluate_sa_parallel(curr_state=STATE[0], model=finetune_model, exploiter_model=br_agent, env_index=0, record=False)
        #TODO: Remove the following line once debugging is done
        # wr = evaluate_sa(STATE[0], finetune_model, br_agent, 0, record=False) # do not change False to True
        rew_arr = np.zeros(len(br_agent.ep_info_buffer))
        for i in range(len(rew_arr)):
            rew_arr[i] = br_agent.ep_info_buffer[i]['r']
        mean_rew = np.mean(rew_arr)
        if ego_timestep is not None:
            wandb.log({"br_win_rate_vs_%s" % (br_agent.exploiting): wr, "global_step": ego_timestep})
            wandb.log({"br_mean_reward_vs_%s" % (br_agent.exploiting): mean_rew, "global_step": ego_timestep})
        else:
            wandb.log({"br_win_rate_vs_ego": wr})

        # 5. Save the trained BR model
        # br_agent.save(os.path.join(BR_MODEL_DIR, br_model_name))
        # -------------------------------------------------

        print(f"WORKER [{worker_id}]: Successfully trained and saved {br_model_name}")
        wandb.finish()

    except Exception as e:
        print(f"WORKER [{worker_id}]: FAILED to process task {os.path.basename(task_file_path)}. Error: {e}")
        # Optionally, move the failed task to an "error" directory instead of "done"
        # to inspect it later.

if __name__ == "__main__":
    wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    todo_dir = os.path.join(TASK_DIR, "todo")
    processing_dir = os.path.join(TASK_DIR, "processing")
    done_dir = os.path.join(TASK_DIR, "done")
    stop_file = os.path.join(TASK_DIR, "STOP")

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

        try:
            # Atomically move the task file to claim it
            os.rename(todo_path, processing_path)

            # Now that we've claimed it, process it
            train_best_response(processing_path)

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
                os.rename(processing_path, todo_path)
            except:
                pass

    print(f"WORKER [{os.getpid()}]: Stop file detected. Shutting down.")
