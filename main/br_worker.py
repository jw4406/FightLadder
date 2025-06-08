# br_worker.py
import os
import time
import random
import retro
import wandb

from common.justin.Generalist_SPAR import Generalist_SPAR
from common.const import *
from common.retro_wrappers import SFWrapper, Monitor2P
from FightLadder.main.common.algorithms import Exploiter
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P
from stable_baselines3.common.callbacks import CheckpointCallback, ExploiterCheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file
# --- Configuration ---
current_dir = os.getcwd()
TASK_DIR = current_dir + "/trained_models/tasks"
BR_MODEL_DIR = current_dir + "/trained_models/br_models"
POLL_INTERVAL = 5  # Seconds to wait before checking for new tasks
BR_TRAINING_STEPS = 50_000_000

PLAYER = "Guile"
SIDE = "left"
player_folder_name = PLAYER + '_' + SIDE

STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name, PLAYER, PLAYER)]

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
            task_file_path)

        del params['policy.ctrl_optimizer']
        del params['policy.value_optimizer']
        del params['policy.dstb_optimizer']
        # finetune_model.warmstarted_cont_MAGICS = True
        # finetune_model.warmstart_setup(finetune_model.lr_schedule)
        finetune_model.set_parameters(params, exact_match=False, device=finetune_model.device)
        if not os.path.exists(task_file_path):
            raise FileNotFoundError(f"Checkpoint file not found at {task_file_path}")

        # --- This is where your specific BR logic goes ---
        # 1. Load the frozen opponent
        # fixed_opponent = PPO.load(checkpoint_path)
        wandb.init(project="exploiter",
                   entity='jw4406',
                   config={"eval_rew": 0,
                           "epochs": 0})
        # 2. Create your environment, passing the frozen opponent to it
        #    so the BR agent can play against it.
        # env = YourStreetFighterEnv(opponent_policy=fixed_opponent)
        env = exploiter_env_generator()

        # 3. Create a new agent to be the best response
        br_agent = Exploiter('CnnPolicy', exploiter_env_generator(), device='cuda', exploited=finetune_model, n_steps=1024, batch_size=512, n_epochs=1)

        # 4. Train the BR agent
        br_model_name = f"br_to_{os.path.basename(task_file_path).replace('.zip', '')}.zip"
        exploiter_callback = ExploiterCheckpointCallback(save_freq=10000, save_path=BR_MODEL_DIR, name_prefix=br_model_name)
        br_agent.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback)

        # 5. Save the trained BR model
        br_agent.save(os.path.join(BR_MODEL_DIR, br_model_name))
        # -------------------------------------------------

        print(f"WORKER [{worker_id}]: Successfully trained and saved {br_model_name}")

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