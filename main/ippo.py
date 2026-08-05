import os
import av
import sys
import torch
import multiprocessing as mp
import argparse
from pprint import pformat
import time
import random
import numpy as np
from PIL import Image
import copy
from common.justin.bare_derivative_free_spar import BareDerivativeFreeSPAR
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.justin.clean_derivative_free_spar_ippo import CleanDerivativeFreeSPARIPPO
from utils import merge_models
import retro
from stable_baselines3.common.callbacks import CheckpointCallback, SACheckpointCallback, FileQueueTriggerCallback, CreateVideoCallback
from stable_baselines3.common.buffers import AdvRolloutBuffer
from stable_baselines3.common.utils import get_schedule_fn
from torch.backends.cudnn import deterministic
from common.const import *
import itertools
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P, reset_child_params
from common.game import get_next_level
from common.algorithms import IPPO, MAGICS_PPO, RARL_PPO, TSS_PPO, Specialized_Agent, Specialized_Agent_IPPO, eepy
from common.justin.spar import Single_SPAR
from common.justin.Generalist_SPAR import Generalist_SPAR
from common.justin.derivative_free_spar import Derivative_Free_SPAR
#from common.justin.derivative_free_spar_parallel import Derivative_Free_SPAR
from stable_baselines3 import MAGICS_AL
from common.retro_wrappers import SFWrapper, Monitor2P, InfoObsWrapper, EgoCentricImageWrapper
import wandb

PRETRAIN = True
FINETUNE = False
EVAL = False
SAVE_FREQ = 10000  # Save a checkpoint every 10,000 steps
#TOTAL_TIMESTEPS = 150_000_000


current_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(current_dir, "main_checkpoints")
# .task checkpoints are written here by FileQueueTriggerCallback -- NOT to
# --save_dir, which only governs .mp4 output. Because this is derived from the
# SCRIPT's path it also ignores cwd, so two concurrent runs of the same
# (arch, player, opponent) silently overwrite each other's checkpoints: they
# share model_name_prefix, hence identical filenames. That destroyed a baseline
# checkpoint during the timescale experiment. Overridable per-run via the
# environment; default is unchanged. It must be an env var rather than a CLI
# flag because this constant is evaluated at import, before argparse runs.
TASK_DIR = os.environ.get("FIGHTLADDER_TASK_DIR") or os.path.join(
    current_dir, "trained_models/tasks")
#print("#$%^&*()*&^%$#@$%^&*(&^%$#@"+ current_dir)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TASK_DIR, exist_ok=True)

MODEL_NAME = "streetfighter_v1"

if EVAL is False:
    assert PRETRAIN != FINETUNE
else:
    assert (PRETRAIN is False) and (FINETUNE is False)

# STATE = "Champion.RyuVsRyu.2Player.align"
global REMOVAL
#global STATE
#torch.backends.cudnn.enabled = False
'''
PLAYER = "Blanka" # "Blanka
global REMOVAL
REMOVAL = None
OPPONENT_LIST = ["Vega", "Balrog", "Guile", "EHonda", "Blanka", "Ryu", "Sagat", "MBison", "Dhalsim", "Zangief", "ChunLi", "Ken"]
SIDE = "left" # "right"
player_folder_name = PLAYER + '_' + SIDE
if REMOVAL is not None:
    OPPONENT_LIST.remove(REMOVAL)

global STATE
#files  = os.listdir
STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name, PLAYER, OPPONENT_LIST[i]) for i in range(len(OPPONENT_LIST))]
'''


# STATE = "two_player/Champion.Level1.RyuVsBlanka.2Player.state"
# STATE = ["two_player/Champion.Level1.RyuVsVega.2Player.state", "two_player/Champion.Level1.RyuVsBalrog.2Player.state", "two_player/Champion.Level1.RyuVsGuile.2Player.state", \
#    "two_player/Champion.Level1.RyuVsEHonda.2Player.state", "two_player/Champion.Level1.RyuVsBlanka.2Player.state", "two_player/Champion.Level1.RyuVsRyu.2Player.state", \
#         "two_player/Champion.Level1.RyuVsSagat.2Player.state", "two_player/Champion.Level1.RyuVsMBison.2Player.state", "two_player/Champion.Level1.RyuVsDhalsim.2Player.state", \
#         "two_player/Champion.Level1.RyuVsZangief.2Player.state", "two_player/Champion.Level1.RyuVsChunLi.2Player.state", "two_player/Champion.Level1.RyuVsKen.2Player.state"]
# state_list = STATE
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
             null_combo=False, transform_action=False, seed=0, obs_type='image', ego_is_left=True):
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
        #if obs_type == 'info':
        #    env = InfoObsWrapper(env, ego_is_left=ego_is_left)
        #elif not ego_is_left:
        #    env = EgoCentricImageWrapper(env)
        env = Monitor2P(env)
        env.seed(seed)
        return env

    return _init


@torch.no_grad()
def evaluate(args, model, greedy=0, record=True):
    global STATE
    win_cnt = 0
    # env = []
    for i in range(1, args.num_episodes + 1):
        env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render,
                       enable_combo=args.enable_combo, null_combo=args.null_combo,
                       transform_action=args.transform_action, seed=1)().env
        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:
            if np.random.uniform() > greedy:
                (action, _states), (action_other, _states_other) = model.predict(obs, deterministic=True)
            else:
                (action, _states), (action_other, _states_other) = model.predict(obs, deterministic=True)

            obs, reward, reward_other, done, info = env.step(np.hstack([action, action_other]))
            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))
            # print(info)
            # if done:
            #     video_log[-1].save(f"{args.video_dir}/episode_{i}.png")

            if done:
                if record:
                    try:
                        name = STATE.split("/")[1]
                    except:
                        name = STATE
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{args.video_dir}/{name}_episode_{i}.mp4", mode='w')
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

        # print("Total reward: {}\n".format(total_reward))
        # episode_reward_sum += total_reward

        env.close()

    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate


@torch.no_grad()
def evaluate_sa(curr_state, args, model, env_index, greedy=0, record=True):
    assert isinstance(model, Specialized_Agent)
    # global STATE
    win_cnt = 0
    # env = []
    for j in range(1, args.num_episodes + 1):
        env = make_env(sf_game, state=curr_state, side=args.side, reset_type=args.reset, rendering=args.render,
                       enable_combo=args.enable_combo, null_combo=args.null_combo,
                       transform_action=args.transform_action, seed=None)().env
        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:
            from stable_baselines3.common.save_util import load_from_zip_file
            if model.use_mirror is True:

                data, params, pytorch_variables = load_from_zip_file(

                    "/home/jw4406/codebase/FightLadder/main/trained_models/ippo_mirror_pre_%s/ppo_%s_27894000_steps.zip" % (

                        PLAYER, PLAYER))
                del params['policy.ctrl_optimizer']
                del params['policy.value_optimizer']
                del params['policy.dstb_optimizer']

                not_ego = model
                not_ego.set_parameters(params, exact_match=False, device=model.device)

                (action, _states), (_,_) = model.predict(obs, env_index, deterministic=False)
                (not_ego_action, _), (_,_) = not_ego.predict(obs, env_index, deterministic=False)

                action_other = not_ego_action

            else:

                if np.random.uniform() > greedy:
                    (action, _states), (action_other, _states_other) = model.predict(obs, env_index, deterministic=False)
                else:
                    (action, _states), (action_other, _states_other) = model.predict(obs, env_index, deterministic=False)

            obs, reward, reward_other, done, info = env.step(np.hstack([action, action_other]))
            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))
            # print(info)
            # if done:
            #     video_log[-1].save(f"{args.video_dir}/episode_{i}.png")

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

        # print("Total reward: {}\n".format(total_reward))
        # episode_reward_sum += total_reward

        env.close()

    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate


@torch.no_grad()
def evaluate_cross(args, model1, model2, greedy=0.5, record=True):
    win_cnt = 0

    for i in range(1, args.num_episodes + 1):
        env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render,
                       enable_combo=args.enable_combo, null_combo=args.null_combo,
                       transform_action=args.transform_action, seed=None)().env

        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:
            # if np.random.uniform() > greedy:
            (action, _states), (action_other, _states_other) = model1.predict(obs, deterministic=False)
            (_, _), (action_other, _states_other) = model2.predict(obs, deterministic=False)
            # else:
            #    (action, _states), (action_other, _states_other) = model1.predict(obs, deterministic=True)
            #    (_, _), (action_other, _states_other) = model2.predict(obs, deterministic=False)

            obs, reward, reward_other, done, info = env.step(np.hstack([action, action_other]))
            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))
            # print(info)
            # if done:
            #     video_log[-1].save(f"{args.video_dir}/episode_{i}.png")

            if done:
                if record:
                    try:
                        name = STATE.split("/")[1]
                    except:
                        name = STATE
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{args.video_dir}/{name}episode_{i}.mp4", mode='w')
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

        # print("Total reward: {}\n".format(total_reward))
        # episode_reward_sum += total_reward

        env.close()

    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate

def env_generator(args, max_envs: int = 0, i_start: int = 0, j_start: int = 0, STATE=None, n_envs: int = 1):
        #global STATE
        """
        TODO: Complete the docstring

        Args:
            max_envs (int):
                Maximum environments to generator. If 0, unbounded.
            i_start (int):
                Index of state to start at.
            j_start (int):
                index of env_count to start at.
        """
        def exceed_max_envs(env_count: int, max_envs: int) -> bool:
            """
            This is a helper function that returns True if max_envs is active (not 0) and count exceeds it.
            """
            if max_envs == 0:
                return False
            return env_count >= max_envs
        
        # STATE
        each_env_count = n_envs
        print("Generating %d envs per character matchup:" % each_env_count)
        env = []
        env_count = 0
        obs_type = getattr(args, 'obs_type', 'image')
        halfway = len(STATE) // 2
        use_mirror = getattr(args, 'use_mirror', False)
        for i in range(i_start, len(STATE)):
            if exceed_max_envs(env_count, max_envs):
                break
            ego_is_left = (i < halfway) if use_mirror else True
            env.append(
                make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render,
                            enable_combo=args.enable_combo, null_combo=args.null_combo,
                            transform_action=args.transform_action, seed=0, obs_type=obs_type,
                            ego_is_left=ego_is_left))
            env_count += 1
            if exceed_max_envs(env_count, max_envs):
                break
        # env = [make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=0) for i in range(args.num_env)]
        # env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render,
        #         enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action,
        #         seed=0)
        vec_env = SubprocVecEnv2P(env)
        if getattr(args, 'obs_type', 'image') == 'image':
            return VecTransposeImage2P(vec_env)
        return vec_env
def main(args):
    PLAYER = args.player
    # global REMOVAL
    # PLAYER = "Blanka"  # "Blanka

    global REMOVAL
    use_mirror = args.use_mirror
    
    REMOVAL = None
    if args.opponents:
        OPPONENT_LIST = args.opponents
    elif use_mirror is True:
        OPPONENT_LIST = ["Guile"]#, "Ryu", "Dhalsim", "Zangief", "ChunLi", "Guile", "Ken", "Balrog", "MBison"]
    else:
        #OPPONENT_LIST = ["Sagat", "EHonda", "MBison", "Blanka", "Ryu", "Dhalsim", "Zangief", "ChunLi", "Guile", "Ken", "Balrog", "MBison"]
        OPPONENT_LIST = ["Guile", "Sagat"]#,"ChunLi", "MBison", "Blanka", "Ryu", "Dhalsim", "Zangief", "Ken", "Balrog", "Vega", "EHonda"]
    player_short = ''.join(player[:2] for player in PLAYER)
    opponent_short = ''.join(opponent[:2] for opponent in OPPONENT_LIST)
    # Preserve each path's existing default when --gamma is unset.
    _gamma_spar = 0.99 if args.gamma is None else args.gamma
    _gae_lambda = 0.95 if args.gae_lambda is None else args.gae_lambda
    _gamma_ippo = 0.94 if args.gamma is None else args.gamma
    model_name_prefix = "%s_%s_%s" % (args.model_arch_type, player_short, opponent_short)
    # parser = argparse.ArgumentParser(description='Reset game stats')
    # parser.add_argument('--reset', choices=['round', 'match', 'game'],
    #                     help='Reset stats for a round, a match, or the whole game', default='round')
    # parser.add_argument('--model-file', help='The model to continue to learn from')
    # parser.add_argument('--save-dir', help='The directory to save the trained models',
    #                     default="trained_models/single_test_large_%s_%s" % (PLAYER, OPPONENT_LIST[0]))

    # #if use_mirror is True:
    # #    OPPONENT_LIST = ["Sagat", "EHonda"]# "MBison", "Blanka"]#, "Ryu", "Dhalsim", "Zangief", "ChunLi", "Guile", "Ken", "Balrog", "MBison"]
    # #else:
    #     #OPPONENT_LIST = ["Sagat", "EHonda", "MBison", "Blanka", "Ryu", "Dhalsim", "Zangief", "ChunLi", "Guile", "Ken", "Balrog", "MBison"]
    # #    OPPONENT_LIST = ["Guile", "EHonda", "Sagat","ChunLi"]#, "MBison", "Blanka", "Ryu", "Dhalsim", "Zangief", "Ken", "Balrog", "Vega"]
    # parser.add_argument('--log-dir', help='The directory to save logs', default="logs")
    # parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="ppo_%s" % '_'.join(PLAYER))
    # parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile',
    #                     default=SF_DEFAULT_STATE)
    # parser.add_argument('--side', help='The side for AI to control. By default both', default='both',
    #                     choices=['left', 'right', 'both'])
    # parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    # parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=64)
    # parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    # parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    # parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(10e8))
    # parser.add_argument('--video-dir', help='The path to save videos', default='videos/spar_spar_%s' % '_'.join(PLAYER))
    # parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    # parser.add_argument('--init-level', type=int,
    #                     help='Initial level to load from. By default 0, starting from pretrain', default=0)
    # parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain',
    #                     default=0)
    # parser.add_argument('--envs-per-matchup', type=int, help='How many environments to create per matchup', default=2)
    # parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    # parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    # parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    # parser.add_argument('--seed', type=int, help='Seed', default=0)
    # parser.add_argument('--update-left', type=int, help='Update left policy', default=1)
    # parser.add_argument('--update-right', type=int, help='Update right policy', default=1)
    # parser.add_argument('--left-model-file', help='The left model to continue to learn from')
    # parser.add_argument('--right-model-file', help='The right model to continue to learn from')
    # parser.add_argument('--other-timescale', type=float, help='Other agent learning rate scale', default=1.0)
    # parser.add_argument('--fsp', action='store_true', help='Fictitious self-play')
    # parser.add_argument('--fsp-threshold', type=float, help='Fictitious self-play threshold', default=0.5)
    # parser.add_argument('--async-update', action='store_true', help='Update left and right asynchronously')
    # parser.add_argument('--num_env_steps', type=int, help='Number of env steps to run', default=300)
    # #parser.add_argument("--player", type=str, required=True)
    # parser.add_argument("--player", type=str, nargs='+', required=True, help="One or more protagonist players.")
    # parser.add_argument("--num_env_to_load", type=int, required=False, help="Number of envs to load", default=1)
    # parser.add_argument("--env_batch_size", type=int, required=True, help="Environment back size", default=100)
    # parser.add_argument("--num_perturbs", type=int, help="Number of perturbed policies to be created.", default=32)
    # parser.add_argument("--c_lr", type=float, help="ego learning rate", default=1e-5)
    # parser.add_argument("--d_lr", type=float, help="adversary learning rate", default=2e-5)
    # parser.add_argument("--v_lr", type=float, help="value learning rate", default=1e-4)
    # parser.add_argument("--use_mirror", action='store_true', help='Use mirror')
    # parser.add_argument("--num_workers", type=int, help="Number of workers", default=5)
    # parser.add_argument("--load_path", type=str, help="Path to load the model from", default=None)
    # #parser.add_argument("--left_model_file", type=str, help="Path to load the left model from", default=None)
    # #parser.add_argument("--right_model_file", type=str, help="Path to load the right model from", default=None)
    # parser.add_argument("--training_style", type=str, required=True, help="Training style", default="L3", choices=["L3", "L2", "L1"])
    # parser.add_argument("--continue_training", help='Continue training', default=False)
    # parser.add_argument("--use_lr_annealing", choices=['True', 'False'], help='Use lr annealing', default=True)
    # parser.add_argument("--lr_anneal_coeff", type=float, help="Learning rate anneal coefficient", default=0.995)
    # parser.add_argument("--checkpoint_interval", type=int, help="Checkpoint interval", default=10000)
    # args = parser.parse_args()
    SIDE = "left"  # "right"
    player_folder_name = [PLAYER[i] + '_' + SIDE for i in range(len(PLAYER))]
    if REMOVAL is not None:
        OPPONENT_LIST.remove(REMOVAL)

    # files  = os.listdir

    if use_mirror is True:
        # Full STATE (with envs_per_matchup replication) is used by env_generator;
        # keep that semantics intact.
        STATE_prot_left = [
            "two_player/%s/Champion.Level1.%sVs%s.2Player.state"
            % (player_folder_name[i], PLAYER[i], OPPONENT_LIST[j])
            for i in range(len(PLAYER))
            for j in range(len(OPPONENT_LIST))
            for _ in range(args.envs_per_matchup)
        ]

        opp_left_folder_name = [opponent + "_" + SIDE for opponent in OPPONENT_LIST]
        STATE_prot_right = [
            "two_player/%s/Champion.Level1.%sVs%s.2Player.state"
            % (opp_left_folder_name[i], OPPONENT_LIST[i], PLAYER[j])
            for i in range(len(OPPONENT_LIST))
            for j in range(len(PLAYER))
            for _ in range(args.envs_per_matchup)
        ]

        STATE = STATE_prot_left + STATE_prot_right
    else:

        if args.ego_side == 'right':
            opp_left_folder_name = [opponent + "_" + SIDE for opponent in OPPONENT_LIST]
            STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (opp_left_folder_name[j], OPPONENT_LIST[j], PLAYER[i])
                     for i in range(len(PLAYER))
                     for j in range(len(OPPONENT_LIST))
                     for k in range(args.envs_per_matchup)]
        else:
            STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name[i], PLAYER[i], OPPONENT_LIST[j])
                     for i in range(len(PLAYER))
                     for j in range(len(OPPONENT_LIST))
                     for k in range(args.envs_per_matchup)]
    state_list = STATE

    # PLAYER = args.player

    #args = parser.parse_args()
    #print("command line args:" + str(args))
    num_steps = args.num_env_steps
    os.makedirs(args.save_dir, exist_ok=True)
    #os.makedirs(args.log_dir, exist_ok=True)
    #os.makedirs(args.video_dir, exist_ok=True)
    #os.makedirs(args.finetune_dir, exist_ok=True)
    #args.model_name_prefix = 
    # Set up the environment and model
    # def env_generator(max_envs: int = 0, i_start: int = 0, j_start: int = 0, STATE=None):
    #     #global STATE
    #     """
    #     TODO: Complete the docstring

    #     Args:
    #         max_envs (int):
    #             Maximum environments to generator. If 0, unbounded.
    #         i_start (int):
    #             Index of state to start at.
    #         j_start (int):
    #             index of env_count to start at.
    #     """
    #     def exceed_max_envs(env_count: int, max_envs: int) -> bool:
    #         """
    #         This is a helper function that returns True if max_envs is active (not 0) and count exceeds it.
    #         """
    #         if max_envs == 0:
    #             return False
    #         return env_count >= max_envs
        
    #     # STATE
    #     each_env_count = args.envs_per_matchup
    #     print("Generating %d envs per character matchup:" % each_env_count)
    #     env = []
    #     env_count = 0
    #     for i in range(i_start, len(STATE)):
    #         if exceed_max_envs(env_count, max_envs):
    #             break
    #         env.append(
    #             make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render,
    #                         enable_combo=args.enable_combo, null_combo=args.null_combo,
    #                         transform_action=args.transform_action, seed=0))
    #         env_count += 1
    #         if exceed_max_envs(env_count, max_envs):
    #             break
    #     # env = [make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=0) for i in range(args.num_env)]
    #     # env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render,
    #     #         enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action,
    #     #         seed=0)
    #     return VecTransposeImage2P(SubprocVecEnv2P(env))
        # return SubprocVecEnv2P(env)

    def many_char_env_generator():
        obs_type = getattr(args, 'obs_type', 'image')
        halfway = len(STATE) // 2
        env = [make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render,
                        enable_combo=args.enable_combo, null_combo=args.null_combo,
                        transform_action=args.transform_action, seed=0, obs_type=obs_type,
                        ego_is_left=(i < halfway) if use_mirror else True) for i in range(len(STATE))]
        vec_env = SubprocVecEnv2P(env)
        if obs_type == 'image':
            return VecTransposeImage2P(vec_env)
        return vec_env
        # return SubprocVecEnv2P(env)

    checkpoint_interval = args.checkpoint_interval # checkpoint_interval * num_envs = total_steps_per_checkpoint

    def finetune_model_generator(model_file=None, reinit_adversary=False, lr_schedule=linear_schedule(5.0e-5, 2.5e-6),
                                 other_lr_schedule=linear_schedule(5.0e-5, 2.5e-6),
                                 clip_range_schedule=linear_schedule(0.075, 0.025), STATE=None, model_arch_type=None):
        REMOVAL
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        finetune_env = env_generator(args, STATE=STATE, n_envs=args.envs_per_matchup)
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        # remove seeds

        
        _ippo_policy = "MlpPolicy" if getattr(args, 'obs_type', 'image') == 'info' else "CnnPolicy"
        finetune_model = IPPO(
            _ippo_policy,
            finetune_env,
            device="cpu",
            verbose=1,
            n_steps=192,
            batch_size=384,  # 512,
            n_epochs=5,
            gamma=_gamma_ippo,
            gae_lambda=_gae_lambda,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            #tensorboard_log=args.log_dir,
            seed=args.seed,
            update_left=True if args.ego_style == 'learning' else False,
            update_right=True if args.adv_style == 'learning' else False,
            other_learning_rate=0.0
        )

        # finetune_model = TSS_PPO(
        #     "AACCnnPolicy",
        #     finetune_env,
        #     device="cpu",
        #     verbose=2,
        #     n_steps=48,
        #     batch_size=24,  # 512,
        #     n_epochs=1,
        #     gamma=0.94,
        #     v_learning_rate=1e-3, c_learning_rate=1e-4,
        #     d_learning_rate=5e-4, v_learning_rate_decay=critic_decay_schedule(1e-3),
        #     c_learning_rate_decay=critic_decay_schedule(1e-4),
        #     d_learning_rate_decay=critic_decay_schedule(5e-4),
        #     clip_range=clip_range_schedule,
        #     tensorboard_log=args.log_dir,
        #     seed=args.seed,
        #     ent_coef=.01,
        #     dstb_ent_coef=.01,
        #     update_left=bool(args.update_left),
        #     update_right=bool(args.update_right),
        # )

        if REMOVAL is None:
            num_adversary = len(OPPONENT_LIST) * len(PLAYER)
            if use_mirror is True:
                num_adversary = num_adversary * 2
        else:
            if isinstance(REMOVAL, str):
                num_adversary = 11
            else:
                assert isinstance(REMOVAL, list)
                num_adversary = 12 - len(REMOVAL)

        # finetune_model = Derivative_Free_SPAR(
        #     "AACCnnPolicy",
        #     finetune_env,
        #     env_batch_size=args.env_batch_size,
        #     envs_per_matchup=args.envs_per_matchup,
        #     state_len=len(STATE),
        #     device="cuda",
        #     verbose=2,
        #     n_steps=num_steps,  # 1408,
        #     batch_size=int(num_steps * len(state_list)),  # 2816,  # 512,
        #     n_epochs=5,
        #     gamma=0.94,
        #     v_learning_rate=1e-3, c_learning_rate=5e-4,
        #     d_learning_rate=0.0, v_learning_rate_decay=critic_decay_schedule(1e-3),
        #     c_learning_rate_decay=critic_decay_schedule(1e-4),
        #     d_learning_rate_decay=critic_decay_schedule(5e-4),
        #     clip_range=clip_range_schedule,
        #     tensorboard_log=args.log_dir,
        #     seed=args.seed,
        #     ent_coef=.01,
        #     dstb_ent_coef=.01,
        #     I_AM_LEFT=True,
        #     I_AM_RIGHT=False,
        #     num_adversary=num_adversary,
        #     n_global_env=args.num_env,
        #     n_env_per_adv=args.num_env // num_adversary,
        #     opp_list=OPPONENT_LIST,
        #     player='_'.join(PLAYER),
        #     use_mirror=use_mirror,
        #     env_generator_func=env_generator,
        #     state_list=state_list,
        # )
        if model_arch_type == "spar":
            finetune_model = CleanDerivativeFreeSPAR(
                "AACCnnPolicy",
                finetune_env,
                device="cuda",
                gamma=_gamma_spar,
                gae_lambda=_gae_lambda,
                c_learning_rate=args.c_lr,
                d_learning_rate=args.d_lr,
                v_learning_rate=args.v_lr,
                verbose=2,
                n_steps=args.num_env_steps,
                batch_size=args.training_batch_size,
                n_epochs=4,
                state_list=state_list,
                envs_per_matchup=args.envs_per_matchup,
                env_generator_func=env_generator,
                num_adversaries=num_adversary,
                clip_range=clip_range_schedule,
                n_env_per_adv=args.num_env // num_adversary,
                seed= 0,
                target_kl=0.05,
                use_mirror=use_mirror,
                ego_side=args.ego_side,
                use_lr_annealing=args.use_lr_annealing,
                lr_anneal_coeff=args.lr_anneal_coeff,
                use_stagnation_early_stop=args.use_stagnation_early_stop,
                use_stagnation_velocity_signal=args.use_stagnation_velocity_signal,
                use_stagnation_entropy_signal=args.use_stagnation_entropy_signal,
                stagnation_patience=args.stagnation_patience,
                stagnation_tolerance=args.stagnation_tolerance,
                stagnation_rel_tolerance=args.stagnation_rel_tolerance,
                stagnation_ema_beta=args.stagnation_ema_beta,
                stagnation_eps=args.stagnation_eps,
                stagnation_eval_games=args.stagnation_eval_games,
                entropy_stagnation_weight=args.entropy_stagnation_weight,
                stagnation_lr_factor=args.stagnation_lr_factor,
                stagnation_lr_patience=args.stagnation_lr_patience,
                vtrace_enabled=True,
                vtrace_replay_capacity=args.vtrace_replay_capacity,
                vtrace_seq_len=args.vtrace_seq_len,
                vtrace_c_bar=args.vtrace_c_bar,
                vtrace_rho_bar=args.vtrace_rho_bar,
                blend_adversary_heads=(args.blend_adversary_heads == 'True'),
            )
        elif model_arch_type == "ippo":
            finetune_model = CleanDerivativeFreeSPARIPPO(
                "AACCnnPolicy",
                finetune_env,
                device="cuda",
                c_learning_rate=args.c_lr,
                d_learning_rate=args.c_lr,
                v_learning_rate=args.c_lr,
                verbose=2,
                n_steps=args.num_env_steps,
                batch_size=args.training_batch_size,
                n_epochs=6,
                state_list=state_list,
                envs_per_matchup=args.envs_per_matchup,
                env_generator_func=env_generator,
                num_adversaries=num_adversary,
                n_env_per_adv=args.num_env // num_adversary,
                seed= 0,
                target_kl=None,
                use_mirror=use_mirror,
                ego_side=args.ego_side,
                use_lr_annealing=args.use_lr_annealing,
                lr_anneal_coeff=args.lr_anneal_coeff,
                # Stagnation gates: CleanDerivativeFreeSPARIPPO inherits
                # __init__ from CleanDerivativeFreeSPAR whose defaults are
                # use_stagnation_velocity_signal=True and
                # use_stagnation_entropy_signal=True. Without forwarding
                # the args here, master_use_stag=False would not actually
                # silence the EMA/entropy logs in learn() for the ippo arch.
                use_stagnation_early_stop=args.use_stagnation_early_stop,
                use_stagnation_velocity_signal=args.use_stagnation_velocity_signal,
                use_stagnation_entropy_signal=args.use_stagnation_entropy_signal,
            )
        elif model_arch_type == '2timescale':
            finetune_model = CleanDerivativeFreeSPARIPPO(
                "AACCnnPolicy",
                finetune_env,
                device="cuda",
                c_learning_rate=args.c_lr,
                d_learning_rate=args.d_lr,
                v_learning_rate=args.d_lr, # this is adversary's v; ego is created separately
                verbose=2,
                n_steps=args.num_env_steps,
                batch_size=args.training_batch_size,
                n_epochs=6,
                state_list=state_list,
                envs_per_matchup=args.envs_per_matchup,
                env_generator_func=env_generator,
                num_adversaries=num_adversary,
                n_env_per_adv=args.num_env // num_adversary,
                seed= 0,
                target_kl=None,
                use_mirror=use_mirror,
                ego_side=args.ego_side,
                use_lr_annealing=args.use_lr_annealing,
                lr_anneal_coeff=args.lr_anneal_coeff,
                # See note above on the ippo branch — same inheritance
                # default issue applies to the 2timescale arch.
                use_stagnation_early_stop=args.use_stagnation_early_stop,
                use_stagnation_velocity_signal=args.use_stagnation_velocity_signal,
                use_stagnation_entropy_signal=args.use_stagnation_entropy_signal,
            )
        else:
            raise ValueError(f"Invalid model arch type: {model_arch_type}. Valid choices are \'spar\', \'ippo\', \'2timescale\'.")

        # if args.load_path and args.continue_training:
        #     from stable_baselines3.common.save_util import load_from_zip_file
        #     from utils import state2matchup
        #     try:
        #         data, params, pytorch_variables = load_from_zip_file(
        #             args.load_path)
        #         if 'state_list' not in data.keys():
        #             print("WARNING: state_list not found in load_path. Using default state list with states %s" % (state_list))
        #             #data['state_list'] = state_list
        #             finetune_model = CleanDerivativeFreeSPAR.load(path=args.load_path, env=finetune_env, num_perturbed=1, state_list=state_list)
        #             matchups = [state2matchup(state) for state in state_list]
        #             assert matchups == finetune_model.matchups
        #         else:
        #             #global STATE
        #             STATE = data['state_list']
        #             my_env_test = env_generator(STATE=STATE)
        #             finetune_model = CleanDerivativeFreeSPAR.load(path=args.load_path, env=my_env_test, num_perturbed=1)
        #             finetune_model.policy.ctrl_optimizer.defaults['lr'] = args.c_lr
        #             finetune_model.policy.dstb_optimizer.defaults['lr'] = args.d_lr
        #             finetune_model.policy.value_optimizer.defaults['lr'] = args.v_lr
        #             finetune_model.ctrl_scheduler.base_lrs = [args.c_lr]
        #             finetune_model.dstb_scheduler.base_lrs = [args.d_lr]
        #             finetune_model.value_scheduler.base_lrs = [args.v_lr]
        #     except Exception as e:
        #         data, params, pytorch_variables = load_from_zip_file(
        #             args.load_path)
        #         finetune_model.set_parameters(params, exact_match=True, device=finetune_model.device)
        #         finetune_model.__dict__.update(data)
            # else:
            #     # need to load two models here
            #     # because curriculum is training the models individually
            #     # load left and then right and then make the left model the correct model
            #     # and point at the right model's parameters
            #     left_model = CleanDerivativeFreeSPAR.load(path=args.left_model_file, env=finetune_env, num_perturbed=1)
            #     right_model = CleanDerivativeFreeSPAR.load(path=args.right_model_file, env=finetune_env, num_perturbed=1)
            #     finetune_model = merge_models(left_model, right_model)
            #     # TODO: Make the left model the correct model and point at the right model's parameters

        #TODO: This is commented out per Justin's comment - should be uncommented in the future.
        # finetune_model = Specialized_Agent_IPPO("IPPOAACCnnPolicy",
        #     finetune_env,
        #     env_batch_size=args.env_batch_size,
        #     envs_per_matchup=args.envs_per_matchup,
        #     state_len=len(STATE),
        #     device="cuda",
        #     verbose=2,
        #     n_steps=num_steps,  # 1408,
        #     batch_size=int(num_steps * len(state_list) / 10),  # 2816,  # 512,
        #     n_epochs=5,
        #     gamma=0.94,
        #     v_learning_rate=5e-3, c_learning_rate=1e-4,
        #     d_learning_rate=5e-4, v_learning_rate_decay=critic_decay_schedule(1e-3),
        #     c_learning_rate_decay=critic_decay_schedule(1e-4),
        #     d_learning_rate_decay=critic_decay_schedule(5e-4),
        #     clip_range=clip_range_schedule,
        #     tensorboard_log=args.log_dir,
        #     seed=args.seed,
        #     ent_coef=0.0,
        #     dstb_ent_coef=0.0,
        #     I_AM_LEFT=True,
        #     I_AM_RIGHfT=False,
        #     num_adversary=num_adversary,
        #     n_global_env=args.num_env,
        #     n_env_per_adv=args.num_env // num_adversary,
        #     opp_list=OPPONENT_LIST,
        #     player='_'.join(PLAYER),
        #     use_mirror=False,
        #     env_generator_func=env_generator,
        #     state_list=state_list,
        # )

        # if use_mirror is True:
        #     with open(current_dir + "miror_indicator.txt", "w") as f:
        #         f.write("1")
        # else:
        #     with open(current_dir + "miror_indicator.txt", "w") as f:
        #         f.write("0")
        #props = finetune_model.dump_properties()
        #with open(current_dir + '/myfile.txt', 'w') as f:
        #    print(props, file=f)


        ''' 
        finetune_model = eepy(
            "AACCnnPolicy",
            finetune_env,
            device="cuda",
            verbose=2,
            n_steps=96,
            batch_size=192,  # 512,
            n_epochs=20,
            gamma=_gamma_ippo,
            gae_lambda=_gae_lambda,
            v_learning_rate=7.5e-2, c_learning_rate=5e-3,
            d_learning_rate=2.5e-2, v_learning_rate_decay=critic_decay_schedule(1e-3),
            c_learning_rate_decay=critic_decay_schedule(1e-4),
            d_learning_rate_decay=critic_decay_schedule(5e-4),
            clip_range=clip_range_schedule,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            ent_coef=0,
            dstb_ent_coef=0,
            update_left=bool(args.update_left),
            update_right=bool(args.update_right),
            #warmstarted_cont_MAGICS=True
        )
        '''

        '''
        finetune_model = MAGICS_AL("MLPAACCNNPolicy", dstb_action_space=finetune_env.action_space, ent_coef='auto',
                      learning_starts=100000, env=finetune_env, verbose=2, v_learning_rate=5e-4, c_learning_rate=10e-4,
                      d_learning_rate=50e-4, v_learning_rate_decay=critic_decay_schedule(5e-4),
                      c_learning_rate_decay=critic_decay_schedule(10e-4),
                      d_learning_rate_decay=critic_decay_schedule(50e-4), 
                      policy_kwargs={'net_arch': dict(pi=[64,64,64], qf=[64,64,64])},
                      buffer_size=100000, batch_size=256, train_freq=32, gradient_steps=1000, gamma=0.9,
                      tau=0.01, use_sde=False, use_stackelberg=True, device='auto', diag=True, use_ef=False, zofo=False, seed=0)
        '''
        if model_file:
            print("load model from " + model_file)
            if model_file.endswith(".pt"):
                model_file = torch.load(model_file, map_location=torch.device('cpu'))["kwargs"]["agent_dict"]
            finetune_model.set_parameters(model_file)
            if reinit_adversary:
                # set_parameters() loads the WHOLE policy, so a checkpoint whose
                # adversary has collapsed (advH ~ 0.02) resumes collapsed. For
                # "can the adversary learn against a frozen strong ego?" that
                # confounds the answer: a negative result could mean the frozen
                # opponent didn't help, or just that a degenerate policy has too
                # little exploration left to recover. Re-init gives a fresh
                # adversary at maximum entropy (ln 22 = 3.09).
                pol = finetune_model.policy
                reset_child_params(pol.pi_dstb_features_extractor)
                reset_child_params(pol.dstb_action_net)
                # Rebuild dstb_optimizer so Adam moment estimates from the
                # collapsed policy don't carry over into the fresh one.
                lr_now = pol.dstb_optimizer.param_groups[0]['lr']
                pol.dstb_optimizer = pol.optimizer_class(
                    itertools.chain(pol.pi_dstb_features_extractor.parameters(),
                                    pol.dstb_action_net.parameters()),
                    lr_now, maximize=False)
                n = sum(p.numel() for p in pol.pi_dstb_features_extractor.parameters()) \
                  + sum(p.numel() for p in pol.dstb_action_net.parameters())
                print(f"[reinit_adversary] re-initialized {n:,} adversary params "
                      f"(pi_dstb_features_extractor + dstb_action_net) and rebuilt "
                      f"dstb_optimizer at lr={lr_now}; ego and critic kept from the "
                      f"checkpoint", flush=True)
        print("model generated")
        finetune_model.model_arch_type = model_arch_type
        # Ego value-head LR: only ippo/2timescale build a dedicated ego_value_optimizer.
        # Guarded (arch check + hasattr) so CDS/spar models -- which have no such
        # optimizer -- never touch it and never fail on a missing attribute.
        if model_arch_type in ("ippo", "2timescale") and hasattr(finetune_model.policy, "ego_value_optimizer"):
            for _pg in finetune_model.policy.ego_value_optimizer.param_groups:
                _pg["lr"] = args.ego_value_head_lr
            print(f"[ippo] ego_value_optimizer lr set to {args.ego_value_head_lr}", flush=True)
        return finetune_model

    #finetune_epoch_model_path = os.path.join(args.save_dir, args.model_name_prefix + f"_final_steps")
    lr_schedule = 1e-4  # if args.async_update else linear_schedule(2.5e-4, 2.5e-6)
    other_lr_schedule = 1e-4  # if args.async_update else linear_schedule(2.5e-4/args.other_timescale, 2.5e-6/args.other_timescale)
    clip_range_schedule = linear_schedule(0.15, 0.025)  # if args.async_update else linear_schedule(0.15, 0.025)
    if REMOVAL is None:
        temp_env = env_generator(args, STATE=STATE)
        args.num_env = temp_env.num_envs
        temp_env.close()
    else:

        if isinstance(REMOVAL, str):
            args.num_env = 11 * 4
        else:
            assert isinstance(REMOVAL, list)
            args.num_env = (12 - len(REMOVAL)) * 4
    model = finetune_model_generator(args.model_file, reinit_adversary=(args.reinit_adversary == 'True'), lr_schedule=lr_schedule, other_lr_schedule=other_lr_schedule,
                                     clip_range_schedule=clip_range_schedule, STATE=STATE, model_arch_type=args.model_arch_type)
    if REMOVAL is not None:
        model.REMOVAL = REMOVAL
    # if args.left_model_file and args.right_model_file:
    #     print("load model from " + args.left_model_file + " and " + args.right_model_file)
    #     model.set_parameters_2p(args.left_model_file, args.right_model_file)

    checkpoint_callback = SACheckpointCallback(save_freq=checkpoint_interval, save_path=args.save_dir,
                                               name_prefix=f"{model_name_prefix}") if hasattr(model,
                                                                                                   "num_adversaries") else CheckpointCallback(
        save_freq=checkpoint_interval, save_path=args.save_dir, name_prefix=f"{model_name_prefix}")

    file_queue_callback = FileQueueTriggerCallback(
        task_dir=TASK_DIR,
        use_mirror=args.use_mirror,
        num_workers=1,
        save_freq=checkpoint_interval,
        save_path=args.save_dir,
        name_prefix=f"{model_name_prefix}"
    )
    video_callback = CreateVideoCallback(save_path=args.save_dir, save_freq=checkpoint_interval)

    if (FINETUNE is True) or (EVAL is True):
        finetune_model = finetune_model_generator(args.model_file, reinit_adversary=(args.reinit_adversary == 'True'), lr_schedule=lr_schedule,
                                                  other_lr_schedule=other_lr_schedule,
                                                  clip_range_schedule=clip_range_schedule)

        # finetune_model.warmstart_setup(finetune_model.lr_schedule)
        # finetune_model = Specialized_Agent.load("/home/jw4406/codebase/FightLadder/main/trained_models/ma/ppo_ryu_4545792_steps.zip", env=env_generator())

        from stable_baselines3.common.save_util import load_from_zip_file
        # data, params, pytorch_variables = load_from_zip_file(
        #    "/home/jw4406/codebase/FightLadder/main/trained_models/ws3_8/ppo_ryu_1668096_steps.zip")
        # if FINETUNE is True:
        # finetune_model.warmstarted_cont_MAGICS = True
        # finetune_model.warmstart_setup(finetune_model.lr_schedule)
        data, params, pytorch_variables = load_from_zip_file(
            "/home/jw4406/codebase/FightLadder/main/trained_models/tasks/first_9mil/ppo_Guile_9880000_steps.task")
        '''
        data, params, pytorch_variables = load_from_zip_file(

            "/home/jw4406/codebase/FightLadder/main/trained_models/ppo_%s_8064000_steps.zip" % (

            PLAYER))
        '''

        #data, params, pytorch_variables = load_from_zip_file(

        #    "/home/jw4406/codebase/FightLadder/main/trained_models/sa_mirror_ft_2_174000cont_%s/ppo_%s_84000_steps.zip" % (

        #        PLAYER, PLAYER))

        # data, params, pytorch_variables = load_from_zip_file(
        #        "/home/jw4406/codebase/FightLadder/main/trained_models/guile_tss_test/ppo_%s_1728000_steps.zip" % (PLAYER))
        if EVAL is True or FINETUNE is True:
            del params['policy.ctrl_optimizer']
            del params['policy.value_optimizer']
            del params['policy.dstb_optimizer']
        finetune_model.set_parameters(params, exact_match=False, device=finetune_model.device)
        model = finetune_model

    if not EVAL:
        if args.async_update:
            model.async_learn(
                total_timesteps=args.total_steps,
                callback=[checkpoint_callback],
                fsp=args.fsp,
                fsp_threshold=args.fsp_threshold,
            )
        else:
            #if hasattr(model, "num_adversaries"):
            #    for i in range(model.num_adversaries):
            #        model.adversaries[i]._setup_learn(model.adversaries[i].num_timesteps)
            if args.use_wandb:
                wandb.init(project="dfs_simple_ego_only",
                           entity='jw4406',
                           config={"eval_rew": 0,
                                   "epochs": 0})
            #test = CleanDerivativeFreeSPAR.load("/home/jw4406/codebase/FightLadder/main/trained_models/tasks/todo/ppo_Guile_32000_steps.task")
            model.policy.to(model.device)

            if args.ego_style == 'learning':
                update_ego=True
                zero_ego_action=False
                random_ego_action=False
            elif args.ego_style == 'zero_action':
                update_ego=False
                zero_ego_action=True
                random_ego_action=False
            elif args.ego_style == 'random_action':
                update_ego=False
                zero_ego_action=False
                random_ego_action=True
            elif args.ego_style == 'frozen':
                # Play the ego policy as-is and never update it. Separates
                # "the opponent is MOVING" (non-stationarity) from "the opponent
                # is STRONG": zero_action already covers a frozen-weak opponent,
                # learning covers a moving one, this covers frozen-strong.
                #
                # Supply the weights with --model_file; without it the ego stays
                # at random init, which is a weaker opponent than zero_action.
                update_ego=False
                zero_ego_action=False
                random_ego_action=False
                if not args.model_file:
                    print("[WARN] --ego_style frozen without --model_file: the ego "
                          "is frozen at RANDOM INIT, not at a trained policy.",
                          flush=True)
                # train() already gates the ego update on update_ego (and
                # USE_PERTURBED is False), so ctrl_optimizer is never stepped.
                # Detaching as well is belt-and-braces against any future path
                # that steps it unconditionally.
                _ego_frozen = [model.policy.pi_ctrl_features_extractor,
                               model.policy.mlp_extractor.policy_net,
                               model.policy.action_net]
                _n_frozen = 0
                for _m in _ego_frozen:
                    for _p in _m.parameters():
                        _p.requires_grad_(False)
                        _n_frozen += _p.numel()
                print(f"[ego_style=frozen] froze {_n_frozen:,} ego actor params "
                      f"(pi_ctrl_features_extractor + mlp_extractor.policy_net + action_net); "
                      f"critic and adversary remain trainable", flush=True)
            else:
                raise ValueError(f"Invalid ego style: {args.ego_style}")
            if args.adv_style == 'learning':
                update_adversary=True
                zero_adv_action=False
                random_adv_action=False
            elif args.adv_style == 'zero_action':
                update_adversary=False
                zero_adv_action=True
                random_adv_action=False
            elif args.adv_style == 'random_action':
                update_adversary=False
                zero_adv_action=False
                random_adv_action=True
            else:
                raise ValueError(f"Invalid adv style: {args.adv_style}")
            model.debug_frame_dir = os.path.join(args.save_dir, "debug_frames")
            model.learn(
                total_timesteps=args.total_timesteps,
                num_perturbs = args.num_perturbs,
                callback=[checkpoint_callback, file_queue_callback, video_callback], update_ego=update_ego, update_adversary=update_adversary, run_ego_forward=True, run_adv_forward=True,
                zero_ego_action=zero_ego_action, zero_adv_action=zero_adv_action, random_ego_action=random_ego_action, random_adv_action=random_adv_action
            )
            #model.learn(total_timesteps=args.total_steps, callback=None)
        # for i in range(len(model.adversaries)):
        #     model.adversaries[i].save("enemy_policy_%d.pt" % i)
        # model.adversaries = []
        # model.save(finetune_epoch_model_path)
    else:
        state_list = ['two_player/EHonda_left/Champion.Level1.EHondaVsEHonda.2Player.state']
        for i in range(len(state_list)):
            # global STATE
            # STATE = state_list[i]
            results = evaluate_sa(state_list[i], args, finetune_model, i, record=True)
        print(results)
        with open(f"{args.finetune_dir}/{args.model_name_prefix}_start_results.txt", 'w') as f:
            f.write(str(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--player", type=str, required=True)
    parser.add_argument("--player", type=str, nargs='+', required=True, help="One or more protagonist players.")
    parser.add_argument("--opponents", type=str, nargs='+', required=False, default=None, help="One or more opponents.")
    parser.add_argument("--num_env_to_load", type=int, required=False, help="Number of envs to load", default=1)
    parser.add_argument("--env_batch_size", type=int, required=True, help="Environment back size", default=32)
    parser.add_argument("--num_perturbs", type=int, help="Number of perturbed policies to be created.", default=1)
    parser.add_argument("--num_env_steps", type=int, help="Number of env steps to run", default=192, required=True)
    # V-trace value-worker sequence length (T). Optional; only used by the spar
    # arch (the only one with vtrace_enabled=True). None => the class default
    # min(max(1, n_steps//4), 64). Lower T => cheaper B*(T+1) forwards => more
    # critic updates/sec (the far end of the trace washes out at c_bar=1 anyway).
    parser.add_argument("--vtrace_seq_len", type=int, default=None, required=False,
                        help="V-trace worker sequence length T (spar arch only). "
                             "None => class default min(n_steps//4, 64).")
    # The two V-trace truncation bars. They do DIFFERENT jobs (vtrace.py:56-57):
    #   rho_bar truncates the TD-error ratio and SETS THE FIXED POINT -- changing
    #           it changes which policy's value function you converge to.
    #   c_bar   truncates the trace ratio and SETS THE VARIANCE -- changing it
    #           changes how far credit propagates, NOT what you converge to.
    # So c_bar is the safe knob: raising it cannot bias the solution.
    #
    # Measured on the spar_Ry_Sa baseline: rho_sat_frac ~0.003 (rho_bar=5 barely
    # binds) but c_sat_frac ~0.47 -- roughly half the traces clip at c_bar=1, and
    # since the trace coefficient at lag k is prod(c_i) with every c_i <= 1, the
    # effective horizon is far shorter than seq_len=64 or gamma=0.99 imply. See
    # also the note at clean_derivative_free_spar.py:249.
    # Raising c_bar trades bias for variance; vtrace_ratio_max reached 119 as the
    # policies diverged, so sweep (1.0 -> 2.0 -> 5.0) rather than removing it.
    # gamma was never exposed. The three model constructors in this file
    # disagree: IPPO and CleanDerivativeFreeSPARIPPO pass gamma=0.94 explicitly,
    # while CleanDerivativeFreeSPAR passes nothing and silently inherits the
    # class default of 0.99 (clean_derivative_free_spar.py:117). So `spar` runs
    # have trained at a ~13 s horizon while the other two paths use ~2.2 s.
    #
    # Measured on arm A's own checkpoints (critic_ceiling D1, episode splits):
    # the return-prediction ceiling is 0.18-0.20 at gamma 0.75-0.9 versus 0.055
    # at 0.99 -- roughly 4x. NOTE that measures how PREDICTABLE the return is,
    # not how good a policy trained on it will be; a shorter horizon also makes
    # the agent myopic.
    #
    # Default None = "leave each path at whatever it uses today", so adding this
    # flag changes nothing until someone passes it.
    # GAE lambda. Never referenced in this file before now: all three model
    # constructors inherited the class default 0.95
    # (clean_derivative_free_spar.py:118).
    #
    # It matters when sweeping gamma, because the ADVANTAGE horizon is
    # ~1/(1-gamma*lambda) while the CRITIC target horizon is ~1/(1-gamma):
    #   gamma 0.99 lambda 0.95 -> critic 100 steps (13.3 s), GAE 16.8 (2.2 s)
    #   gamma 0.94 lambda 0.95 -> critic  16.7      (2.2 s), GAE  9.4 (1.2 s)
    #   gamma 0.94 lambda 1.00 -> critic  16.7      (2.2 s), GAE 16.7 (2.2 s)
    # So lambda=1.0 HOLDS THE ADVANTAGE HORIZON FIXED while shortening only the
    # critic's target -- which isolates the critic change from the policy one.
    # Cost: lambda=1.0 is pure Monte Carlo advantage, higher variance.
    # V-trace replay capacity. Was hardcoded 15000 at the SPAR construction.
    # Measured: ~88,500 updates against a 15,000-capacity buffer with mean_age
    # ~7,400, and an in-batch/held-out EV gap that reached 0.45 on the baseline.
    # That reuse is the mechanism behind the memorization signature.
    parser.add_argument("--vtrace_replay_capacity", type=int, default=15000,
                        required=False,
                        help="V-trace replay buffer capacity (spar only). "
                             "Default 15000 reproduces every run to date.")
    parser.add_argument("--gae_lambda", type=float, default=None, required=False,
                        help="GAE lambda. Unset => class default 0.95. Set to 1.0 "
                             "alongside --gamma 0.94 to hold the advantage horizon "
                             "at ~17 steps while the critic target drops 100 -> 17.")
    parser.add_argument("--gamma", type=float, default=None, required=False,
                        help="Discount factor. Unset => spar keeps 0.99, the "
                             "IPPO paths keep 0.94. Setting it overrides ALL of "
                             "them. Effective horizon ~1/(1-gamma) agent steps; "
                             "at ~0.13 s/step, 0.99 ~ 13 s and 0.94 ~ 2.2 s "
                             "against rounds of ~27 s.")
    parser.add_argument("--vtrace_c_bar", type=float, default=1.0, required=False,
                        help="V-trace trace-ratio truncation (variance / effective "
                             "credit horizon). Does NOT move the fixed point.")
    parser.add_argument("--vtrace_rho_bar", type=float, default=5.0, required=False,
                        help="V-trace TD-error-ratio truncation. SETS THE FIXED "
                             "POINT -- changing this changes what is learned.")
    # Blend the multi-head adversary update so the shared dstb_net trunk gets one
    # mean-over-heads step per batch instead of N sequential per-head steps
    # (removes the ordering bias where later matchups inherit earlier ones' trunk
    # drift). spar arch only; 'False' => unchanged sequential behavior.
    parser.add_argument("--blend_adversary_heads", choices=['True', 'False'], default='True', required=False,
                        help="Blend multi-head adversary trunk update (spar arch only). "
                             "'False' => unchanged sequential per-head update.")
    parser.add_argument("--ego_style", type=str, help="Ego style", default="learning", required=True, choices=["learning", "zero_action", "random_action", "frozen"])
    parser.add_argument("--adv_style", type=str, help="Adv style", default="learning", required=True, choices=["learning", "zero_action", "random_action"])
    parser.add_argument("--c_lr", type=float, help="ego learning rate", default=1e-4, required=True)
    parser.add_argument("--d_lr", type=float, help="adversary learning rate", default=7e-4, required=True)
    parser.add_argument("--v_lr", type=float, help="value learning rate", default=7e-4, required=True)
    # Ego value-head LR. Conditionally required (see validation after parse_args):
    # REQUIRED for --model_arch_type ippo/2timescale, optional (ignored) otherwise.
    parser.add_argument("--ego_value_head_lr", type=float, default=None, required=False,
                        help="Ego value-head learning rate. REQUIRED for --model_arch_type "
                             "ippo/2timescale; optional and ignored for spar/other.")
    parser.add_argument("--training_batch_size", type=int, help="Training batch size", default=256, required=True)
    parser.add_argument("--checkpoint_interval", type=int, help="Checkpoint interval", default=10000, required=True)
    parser.add_argument("--save_dir", type=str, help="Save directory", default="trained_models/ippo", required=True)
    #parser.add_argument("--log_dir", type=str, help="Log directory", default="logs/ippo", required=True)
    parser.add_argument("--envs_per_matchup", type=int, help="Number of envs per matchup", default=1, required=True)
    parser.add_argument("--use_lr_annealing", choices=['True', 'False'], help='Use lr annealing', default=True, required=True)
    parser.add_argument("--lr_anneal_coeff", type=float, help="Learning rate anneal coefficient", default=0.995, required=True)
    parser.add_argument('--reset', choices=['round', 'match', 'game'],help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument("--side", type=str, help="Side", default="left", required=True, choices=["left", "right", "both"])
    # Master gate for ALL stagnation behavior (EMA, entropy weighting,
    # plateau/slope checks, LR adjustment, early-stop, and the associated
    # logger/wandb records inside CleanDerivativeFreeSPAR.learn). When
    # 'False', the per-signal use_stagnation_* flags below are forced to
    # False after parsing so use_elo_tracker collapses to False and the
    # entire stagnation block in learn() is skipped.
    parser.add_argument("--master_use_stag", choices=['True', 'False'], help='Master switch for all stagnation logic and logging in main training', default='True', required=False)
    parser.add_argument("--use_stagnation_early_stop", choices=['True', 'False'], help='Use stagnation signal for early stopping', default=True, required=True)
    parser.add_argument("--use_stagnation_velocity_signal", choices=['True', 'False'], help='Use rating-movement velocity in stagnation metric', default=True, required=True)
    parser.add_argument("--use_stagnation_entropy_signal", choices=['True', 'False'], help='Use policy entropy in stagnation metric', default=True, required=True)
    parser.add_argument("--stagnation_patience", type=int, help="Stagnation patience checks", default=20, required=True)
    parser.add_argument("--stagnation_tolerance", type=float, help="Absolute stagnation tolerance", default=1e-4, required=True)
    parser.add_argument("--stagnation_rel_tolerance", type=float, help="Relative tolerance for dynamic stagnation threshold", default=0.05, required=True)
    parser.add_argument("--stagnation_ema_beta", type=float, help="EMA beta for dynamic stagnation threshold scaling", default=0.99, required=True)
    parser.add_argument("--stagnation_eps", type=float, help="Numerical epsilon floor for dynamic stagnation threshold", default=1e-8, required=True)
    parser.add_argument("--stagnation_eval_games", type=int, help="Games before each stagnation evaluation", default=0, required=True)
    parser.add_argument("--entropy_stagnation_weight", type=float, help="Weight for entropy contribution in stagnation metric", default=100.0, required=True)
    parser.add_argument("--stagnation_lr_factor", type=float, help="Target factor for stagnation-triggered LR reduction", default=0.5, required=True)
    parser.add_argument("--stagnation_lr_patience", type=int, help="Plateau checks before stagnation-triggered LR reduction", default=5, required=True)
    parser.add_argument('--render', choices=['True', 'False'], help='Whether to render the game screen', default='False')
    parser.add_argument('--enable_combo', choices=['True', 'False'], help='Enable special move action space for environment', default='True')
    parser.add_argument('--null_combo', choices=['True', 'False'], help='Null action space for special move', default='False')
    parser.add_argument('--transform_action', choices=['True', 'False'], help='Transform action space to MultiDiscrete', default='False')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    parser.add_argument('--model_file', type=str, help='Model file', default=None)
    parser.add_argument('--reinit_adversary', type=str, default='False',
                        help="After --model_file loads the whole policy, re-initialize the "
                             "adversary head (pi_dstb_features_extractor + dstb_action_net) "
                             "and rebuild dstb_optimizer. Use with --ego_style frozen to ask "
                             "'can a FRESH adversary learn against a strong stationary ego?' "
                             "without inheriting a collapsed adversary from the checkpoint.")
    parser.add_argument('--use_mirror', choices=['True', 'False'], help='Use mirror', required=True, default='False')
    parser.add_argument('--ego_side', type=str, choices=['left', 'right'], help='Which side the ego controls in cds-style training', required=True, default='left')
    parser.add_argument('--async_update', choices=['True', 'False'], help='Async update', required=True, default='False')
    parser.add_argument('--model_arch_type', type=str, help='Model architecture type', default="spar", required=True, choices=["spar", "ippo", "2timescale"])
    parser.add_argument('--total_timesteps', type=int, help='How many total steps to train', default=int(1e8))
    parser.add_argument('--use_wandb', choices=['True', 'False'], help='Enable Weights & Biases logging', default='False')
    parser.add_argument('--obs_type', type=str, help='Observation type: image or info', default='image', choices=['image', 'info'])
    #parser.add_argument('--num_workers', type=int, help='Number of workers', default=5)
    #parser.add_argument('--num_adversary', type=int, help='Number of adversaries', default=1)
    #parser.add_argument('--n_global_env', type=int, help='Number of global environments', default=1)
    args = parser.parse_args()
    # ego_value_head_lr is required for arch types that have a dedicated ego value
    # head (ippo/2timescale); optional otherwise.
    if args.model_arch_type in ("ippo", "2timescale") and args.ego_value_head_lr is None:
        parser.error(
            "--ego_value_head_lr is required when --model_arch_type is 'ippo' or '2timescale'"
        )
    args.use_wandb = args.use_wandb == 'True'
    if not args.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    else:
        wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    args.async_update = True if args.async_update == 'True' else False
    args.use_mirror = True if args.use_mirror == 'True' else False
    args.render = True if args.render == 'True' else False
    args.enable_combo = True if args.enable_combo == 'True' else False
    args.null_combo = True if args.null_combo == 'True' else False
    args.transform_action = True if args.transform_action == 'True' else False
    args.use_lr_annealing = True if args.use_lr_annealing == 'True' else False
    args.use_stagnation_early_stop = True if args.use_stagnation_early_stop == 'True' else False
    args.use_stagnation_velocity_signal = True if args.use_stagnation_velocity_signal == 'True' else False
    args.use_stagnation_entropy_signal = True if args.use_stagnation_entropy_signal == 'True' else False
    args.stagnation_eval_games = None if args.stagnation_eval_games <= 0 else args.stagnation_eval_games

    # Master switch override. Forcing all three use_stagnation_* flags to
    # False makes use_elo_tracker (in CleanDerivativeFreeSPAR.learn) evaluate
    # False, which short-circuits the entire stagnation block: no
    # tracker.check(), no logger.record / wandb.log of stagnation metrics,
    # no LR adjustment, no early-stop break.
    args.master_use_stag = True if args.master_use_stag == 'True' else False
    if not args.master_use_stag:
        print(
            "[master_use_stag=False] Disabling all main-training stagnation "
            "logic and logging: forcing use_stagnation_early_stop, "
            "use_stagnation_velocity_signal, use_stagnation_entropy_signal -> False."
        )
        args.use_stagnation_early_stop = False
        args.use_stagnation_velocity_signal = False
        args.use_stagnation_entropy_signal = False

    # Print all runtime CLI settings in a readable way for debugging/repro.
    def _print_args_human_readable(parsed_args):
        args_dict = vars(parsed_args)
        print("\n========== IPPO CLI Arguments ==========")
        max_key_len = max(len(key) for key in args_dict)
        for key in sorted(args_dict):
            value = args_dict[key]
            value_str = pformat(value, compact=True)
            print(f"{key:<{max_key_len}} : {value_str}")
        print("========================================\n")

    _print_args_human_readable(args)

    PLAYER = args.player
    mp.set_start_method("spawn", force=True) #A lot of stable_baseline3 objects don't support the default "fork".
    main(args)
