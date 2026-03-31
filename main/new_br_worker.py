from common.algorithms import Exploiter
import argparse
import os
import time
import json
import random
import numpy as np
import wandb
import subprocess
import multiprocessing as mp
from stable_baselines3.common.preprocessing import is_image_space
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from stable_baselines3.common.save_util import load_from_zip_file
from ippo import env_generator
from stable_baselines3.common.callbacks import ExploiterCheckpointCallback
from gymnasium.spaces import Box
# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)
TASK_DIR = os.path.join(current_dir, "trained_models/tasks")
#TASK_DIR = '/n/fs/magics/2415498/FightLadder/main/trained_models/tasks/'
PROCESSING_DIR = os.path.join(current_dir, "main/trained_models/tasks/processing")
DONE_DIR = os.path.join(current_dir, "main/trained_models/tasks/done")
ERROR_DIR = os.path.join(current_dir, "main/trained_models/tasks/error")
BR_MODEL_DIR = os.path.join(current_dir, "main/trained_models/tasks/br_models")
WR_STATS_DIR = os.path.join(current_dir, "main/trained_models/wr_stats")
MEAN_REW_STATS_DIR = os.path.join(current_dir, "main/trained_models/mean_rew_stats")
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
BR_TRAINING_STEPS = 120000


def load_spar_model(game_args: dict, task_file_path: str, n_envs: int = 2) -> None:
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

    
    data, _, _ = load_from_zip_file(checkpoint_path)
    # get the saved matchups -- make this automated so that we dont hit stupid user errors
    #matchups = data['matchups']
    uniques = list(dict.fromkeys(data['state_list']).keys())
    # need to get the strengths as well
    STATE = uniques
    game_args = argparse.Namespace(**game_args)
    env = env_generator(game_args, STATE=STATE, n_envs=n_envs)
    #env.num_envs = 1 # HACKY FOR NOW!
    try:
        ftm = CleanDerivativeFreeSPAR.load(path=checkpoint_path, env=env, game_args=game_args, num_perturbed=1)
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


def train_best_response(
    game_args: dict,
    model_to_exploit,
    task_file_path: str,
    eval_prot: bool,
    use_mirror: bool,
    eval_only: bool,
    proj_name: str,
    analysis_upload_proj_name: str,
    n_envs: int,
    is_spar: bool = False,
    br_index: int = 0,
    from_scratch: bool = False,
    exploiter_save_freq: int = 100000,
) -> None:
    """
    The core logic for a single best-response training run.

    Args:
        TODO: Complete this.
    """
    checkpoint_path = task_file_path
    done_model_checkpoint_path = os.path.join(DONE_DIR, os.path.basename(checkpoint_path))
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
    if is_spar == True:
        game_args = argparse.Namespace(**game_args)
        env = env_generator(game_args, STATE=ftm.state_list, n_envs=n_envs)
        # if eval_prot is True: # we're training an optimal adversary
        #     dstb_action_space = Box(low=ftm.dstb_action_space.low, high=ftm.dstb_action_space.high, shape=ftm.dstb_action_space.shape)
        #     env.action_space = dstb_action_space
        # else:
        #     assert eval_prot is False 
        #     # we're training an optimal ego against the current adversary
        #     ego_action_space = Box(low=ftm.action_space.low, high=ftm.action_space.high, shape=ftm.action_space.shape)
        #     env.action_space = ego_action_space
    else:
        # NOT SURE WHAT TO DO HERE ABOUT LEAGUE MODELS
        env = env_generator(STATE=STATE)

    # 3. Create a new agent to be the best response
    br_agent = Exploiter('CnnPolicy' if is_image_space(env.observation_space) else 'MlpPolicy', env, device='cuda', exploited=ftm, n_steps=2048, batch_size=512, n_epochs=5, exploiting='ego' if eval_prot is True else 'adv')
    br_agent.is_spar = is_spar # TODO: This is a stupid hack to get the BR agent to know if it is a SPAR model or not. Remove this once we have a better way to do this.
    # 4. Train the BR agent
    br_model_name = f"br{br_index}_to_{os.path.splitext(os.path.basename(checkpoint_path))[0]}_exploiting_{'ego' if eval_prot is True else 'adv'}.zip"
    exploiter_callback = ExploiterCheckpointCallback(save_freq=exploiter_save_freq // n_envs, save_path=BR_MODEL_DIR, name_prefix=br_model_name)
     
    if eval_only == False:
        print("eval_only was passed as False. Training the BR agent.")
        if from_scratch == True:
            br_agent.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback)
        else:
            # if eval prot is True we are training an optimal adversary so we need to update the adversary
            # if eval prot is False we are training an optimal ego against the current adversary so we need to update the ego
            ftm.envs_per_matchup = len(env.envs)
            ftm.policy.num_env_per_adv = len(env.envs)
            ftm.policy.envs_per_matchup = len(env.envs)
            ftm.exploited = None
            ftm.training_br = True
            ftm.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback, update_ego=not eval_prot, update_adversary=eval_prot)
        #br_agent.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback)

        local_plot_and_eval_file = os.path.join(current_dir, "local_br_eval.py")
        
        br_interval_num = exploiter_callback.n_calls * env.num_envs // exploiter_callback.save_freq
        #br_model_path = os.path.join(BR_MODEL_DIR, f"{br_model_name}_{br_interval_num}000_steps.zip")
        br_model_path = exploiter_callback.model_path
        subprocess.Popen(["python", local_plot_and_eval_file, 
        "--eval_prot", str(eval_prot),
        "--main_checkpoint_model_path", checkpoint_path,
        "--done_model_checkpoint_path", done_model_checkpoint_path,
        "--br_checkpoint_model_path", br_model_path,
        "--state_list", str(ftm.state_list),
        "--exploiter_is_cds", str(not from_scratch),
        "--br_index", str(br_index),
        "--game_args", json.dumps(vars(game_args)),
        ])
        #agg_file = os.path.join(current_dir, "aggregate_to_wandb.py")
        #subprocess.Popen(["python", agg_file, "--read_from_proj_name", proj_name, "--upload_to_proj_name", analysis_upload_proj_name])


def run_br_for_task_in_subprocess(
    game_args: dict,
    task_file_path: str,
    eval_prot: bool,
    use_mirror: bool,
    eval_only: bool,
    proj_name: str,
    analysis_upload_proj_name: str,
    n_envs: int,
    is_spar: bool,
    br_index: int,
    from_scratch: bool = False,
    exploiter_save_freq: int = 100000,
) -> None:
    """
    Worker function for running a single BR training instance in a separate process.
    Each subprocess loads its own copy of the model to avoid pickling issues.
    """
    if is_spar:
        loaded_model = load_spar_model(game_args, task_file_path, n_envs=n_envs)# LEARNING RATE
        if exploiter_save_freq * n_envs * len(loaded_model.matchups) > BR_TRAINING_STEPS:
            print("-------------------------------------------\n\n ")
            print("ERROR!")
            print("ERROR! Exploiter save frequency is greater than BR training steps. This will result in no exploiter checkpoints being saved AND AN ERROR RIGHT BEFORE LOCAL BR EVAL")
            print("ERROR!")
            print("-------------------------------------------\n\n ")
            quit() # QUIT THE PROGRAM
        loaded_model.c_learning_rate = 1e-4
        loaded_model.d_learning_rate = 2e-4
        loaded_model.v_learning_rate = 5e-4
        loaded_model.policy.ctrl_optimizer.param_groups[0]['lr'] = 1e-4
        loaded_model.policy.dstb_optimizer.param_groups[0]['lr'] = 1e-4
        loaded_model.policy.value_optimizer.param_groups[0]['lr'] = 2e-4
        loaded_model.use_lr_annealing = False
    else:
        raise NotImplementedError("Non-SPAR multiprocessing BR training is not implemented.")

    train_best_response(
        game_args,
        loaded_model,
        task_file_path,
        eval_prot=eval_prot,
        use_mirror=use_mirror,
        eval_only=eval_only,
        proj_name=proj_name,
        analysis_upload_proj_name=analysis_upload_proj_name,
        n_envs=n_envs,
        is_spar=is_spar,
        br_index=br_index,
        from_scratch = from_scratch,
        exploiter_save_freq=exploiter_save_freq,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_prot", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--eval_adv", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--eval_only", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--proj_name", type=str, required=True)
    parser.add_argument("--analysis_upload_proj_name", type=str, required=True)
    parser.add_argument("--load_br", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--which_env", choices=['my_pendulum', 'my_walker2d_v4', 'my_mountain_car_continuous', 'my_half_cheetah', 'my_hopper', 'my_ant'], required=True)
    parser.add_argument("--is_league", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--use_mirror", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--task_dir", type=str, required=False)
    parser.add_argument("--num_brs", type=int, default=6, help="Number of independent BR agents to train per main checkpoint.")
    parser.add_argument("--n_envs", type=int, default=2, help="Number of environments to run in parallel.")
    parser.add_argument("--DEBUG", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--num_full_exploiters", type=int, default=4, help="Number of full exploiters to train.")
    parser.add_argument("--num_continue_exploiters", type=int, default=4, help="Number of continue exploiters to train.")
    parser.add_argument("--dedicated_exploiter", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--continue_exploiters", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--exploiter_save_freq", type=int, required=True, default=100000, help="Frequency of exploiter checkpoint saves.")

    parser.add_argument('--reset', choices=['round', 'match', 'game'],help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument("--side", type=str, help="Side", default="left", required=True, choices=["left", "right", "both"])

    parser.add_argument('--render', choices=['True', 'False'], help='Whether to render the game screen', default='False')
    parser.add_argument('--enable_combo', choices=['True', 'False'], help='Enable special move action space for environment', default='True')
    parser.add_argument('--null_combo', choices=['True', 'False'], help='Null action space for special move', default='False')
    parser.add_argument('--transform_action', choices=['True', 'False'], help='Transform action space to MultiDiscrete', default='False')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    # whether or not we want to do full br training or continuing the CDS as exploiter
    args = parser.parse_args()
    args.render = True if args.render == 'True' else False
    args.enable_combo = True if args.enable_combo == 'True' else False
    args.null_combo = True if args.null_combo == 'True' else False
    args.transform_action = True if args.transform_action == 'True' else False
    game_args = {
        "reset": args.reset,
        "side": args.side,
        "render": args.render,
        "enable_combo": args.enable_combo,
        "null_combo": args.null_combo,
        "transform_action": args.transform_action,
        "seed": args.seed,
    }
    args.DEBUG = args.DEBUG == 'True'
    args.dedicated_exploiter = args.dedicated_exploiter == 'True'
    args.continue_exploiters = args.continue_exploiters == 'True'


    args.eval_prot = args.eval_prot == 'True'
    args.eval_adv = args.eval_adv == 'True'
    if args.eval_only == 'True':
        print("WARNING!")
        print("This is an EVAL ONLY run. No exploiter training will be performed.")
        print("WARNING!")
    if not args.eval_prot or not args.eval_adv:
        print("WARNING!")
        training_type = "adversary" if args.eval_prot else "ego " # if eval_prot is True we are training an optimal adversary so we need to update the adversary
        print(f"WARNING! Only {training_type} training will be performed.")
    print("Dedicated exploiter: ", args.dedicated_exploiter)
    print("Number of full exploiters: ", args.num_full_exploiters)
    print("Continue exploiters: ", args.continue_exploiters)
    print("Number of continue exploiters: ", args.num_continue_exploiters)
    print("Number of environments: ", args.n_envs)
    if args.exploiter_save_freq * args.n_envs > BR_TRAINING_STEPS:
        print("WARNING! Exploiter save frequency is greater than BR training steps. This will result in no exploiter checkpoints being saved AND AN ERROR RIGHT BEFORE LOCAL BR EVAL")
    args.eval_only = args.eval_only == 'True'
    wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    todo_dir = os.path.join(TASK_DIR, "todo")

    if args.task_dir is not None and args.task_dir != "":
        print(f"WARNING: Using custom task directory: {args.task_dir}")
        todo_dir = os.path.join(args.task_dir, "todo")
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
                # if args.num_brs == 1:
                #     loaded_model = load_spar_model(processing_path)
                #     train_best_response(
                #         loaded_model,
                #         processing_path,
                #         eval_prot=args.eval_prot,
                #         use_mirror=args.use_mirror,
                #         eval_only=args.eval_only,
                #         proj_name=args.proj_name,
                #         analysis_upload_proj_name=args.analysis_upload_proj_name,
                #         is_spar=True,
                #         br_index=2,
                #         from_scratch=True,
                #     )
                # else:
                processes = []
                if args.eval_prot == True:
                    if args.continue_exploiters == True:
                        from_scratch = False
                        for br_idx in range(args.num_continue_exploiters):
                            target = run_br_for_task_in_subprocess
                            training_args = (
                                game_args,
                                processing_path,
                                args.eval_prot,
                                args.use_mirror,
                                args.eval_only,
                                args.proj_name,
                                args.analysis_upload_proj_name,
                                args.n_envs,
                                True,  # is_spar
                                br_idx,
                                from_scratch,
                                args.exploiter_save_freq,
                            )
                            if args.DEBUG:
                                print(f"DEBUG: Running BR {br_idx} for task {task_filename}")
                                target(*training_args)
                            else:
                                p = mp.Process(target=target, args=training_args)
                                p.start()
                                processes.append(p)
                    if args.dedicated_exploiter == True:
                        from_scratch = True
                        for br_idx in range(args.num_full_exploiters):
                            target = run_br_for_task_in_subprocess
                            training_args = (
                                game_args,
                                processing_path,
                                args.eval_prot,
                                args.use_mirror,
                                args.eval_only,
                                args.proj_name,
                                args.analysis_upload_proj_name,
                                args.n_envs,
                                True,  # is_spar
                                br_idx,
                                from_scratch,
                                args.exploiter_save_freq,
                            )
                            if args.DEBUG:
                                print(f"DEBUG: Running BR {br_idx} for task {task_filename}")
                                target(*training_args)
                            else:
                                p = mp.Process(target=target, args=training_args)
                                p.start()
                                processes.append(p)
                if args.eval_adv == True:
                    if args.continue_exploiters == True:   
                        from_scratch = False
                        for br_idx in range(args.num_continue_exploiters):
                            target = run_br_for_task_in_subprocess
                            training_args = (
                                game_args,
                                processing_path,
                                args.eval_adv,
                                args.use_mirror,
                                args.eval_only,
                                args.proj_name,
                                args.analysis_upload_proj_name,
                                args.n_envs,
                                True,  # is_spar
                                br_idx,
                                from_scratch,
                                args.exploiter_save_freq,
                            )
                            if args.DEBUG:
                                print(f"DEBUG: Running BR {br_idx} for task {task_filename}")
                                target(*training_args)
                            else:
                                p = mp.Process(target=target, args=training_args)
                                p.start()
                                processes.append(p)
                    if args.dedicated_exploiter == True:
                        from_scratch = True
                        for br_idx in range(args.num_full_exploiters):
                            target = run_br_for_task_in_subprocess
                            training_args = (
                                game_args,
                                processing_path,
                                args.eval_adv,
                                args.use_mirror,
                                args.eval_only,
                                args.proj_name,
                                args.analysis_upload_proj_name,
                                args.n_envs,
                                True,  # is_spar
                                br_idx,
                                from_scratch,
                                args.exploiter_save_freq,
                            )
                            if args.DEBUG:
                                print(f"DEBUG: Running BR {br_idx} for task {task_filename}")
                                target(*training_args)
                            else:
                                p = mp.Process(target=target, args=training_args)
                                p.start()
                                processes.append(p)
                    # Wait for all BR processes to finish before marking task as done
                if not args.DEBUG:
                    for p in processes:
                        p.join()

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
        model_files, payoff_path = load_league_models(model_dir=args.league_model_dir, character_names=["ryu", "bison", "guile"])
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
