import os
import av
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import copy

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from FightLadder.main.common.justin.Generalist_SPAR import Generalist_SPAR
from stable_baselines3.common.save_util import load_from_zip_file

import wandb
import retro
from stable_baselines3.common.callbacks import CheckpointCallback, SACheckpointCallback, ExploiterCheckpointCallback, \
    Exploiter_EvalCallback, CallbackList
from stable_baselines3.common.buffers import AdvRolloutBuffer
from stable_baselines3.common.utils import get_schedule_fn
from torch.backends.cudnn import deterministic

from common.const import *
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P
from common.game import get_next_level
from common.algorithms import IPPO, MAGICS_PPO, RARL_PPO, TSS_PPO, Specialized_Agent, Specialized_Agent_IPPO, eepy, \
    Exploiter
from stable_baselines3 import MAGICS_AL
from common.retro_wrappers import SFWrapper, Monitor2P

PRETRAIN = False

FINETUNE = False
EVAL = True

if EVAL is False:
    assert PRETRAIN != FINETUNE
else:
    assert (PRETRAIN is False) and (FINETUNE is False)

# STATE = "Champion.RyuVsRyu.2Player.align"
global REMOVAL
global STATE


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
def evaluate_sa(curr_state, args, model, exploiter_model, env_index, greedy=0, record=True):
    #assert isinstance(model, Specialized_Agent)
    # global STATE
    args.num_episodes = 50
    win_cnt = 0
    vic = np.zeros((50,))
    # env = []
    for j in range(1, args.num_episodes + 1):
        env = make_env(sf_game, state=curr_state, side=args.side, reset_type=args.reset, rendering=args.render,
                       enable_combo=args.enable_combo, null_combo=args.null_combo,
                       transform_action=args.transform_action, seed=1)().env
        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:

            (action, _states), (_, _) = model.predict(obs, env_index, deterministic=False)
            (_, _), (adv_action, _) = exploiter_model.predict(obs, env_index, deterministic=False)

            #action_other = exploit_action


            obs, reward, reward_other, done, info = env.step(np.hstack([action, adv_action]))
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
            # vic[j-1] = 1
            win_cnt += 1

        # print("Total reward: {}\n".format(total_reward))
        # episode_reward_sum += total_reward

        env.close()

    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate


def evaluate_single(curr_state, args, model, env_index, greedy=0, record=True):
    #assert isinstance(model, TSS_PPO) or isinstance(model, IPPO)
    # global STATE
    args.num_episodes = 50
    win_cnt = 0
    vic = np.zeros((50,))
    # env = []
    for j in range(1, args.num_episodes + 1):
        env = make_env(sf_game, state=curr_state, side=args.side, reset_type=args.reset, rendering=args.render,
                       enable_combo=args.enable_combo, null_combo=args.null_combo,
                       transform_action=args.transform_action, seed=0)().env
        done = False

        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:

            (action, _states), (adv_action, _) = model.predict(obs, env_index, deterministic=False)
            #exploit_action, _ = exploiter_model.predict(obs, env_index, deterministic=False)

            #action_other = exploit_action


            obs, reward, reward_other, done, info = env.step(np.hstack([action, adv_action]))
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
            # vic[j-1] = 1
            win_cnt += 1

        # print("Total reward: {}\n".format(total_reward))
        # episode_reward_sum += total_reward

        env.close()

    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate

def main(PLAYER):
    # global REMOVAL
    # PLAYER = "Blanka"  # "Blanka
    global REMOVAL
    use_mirror = False
    REMOVAL = None

    if use_mirror is True:
        OPPONENT_LIST = ["Sagat", "EHonda", "MBison"]
    else:
        OPPONENT_LIST = ["Guile"]
    SIDE = "left"  # "right"
    player_folder_name = PLAYER + '_' + SIDE
    if REMOVAL is not None:
        OPPONENT_LIST.remove(REMOVAL)

    # files  = os.listdir

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
        # STATE = STATE_prot_left + STATE_prot_right

        # chunking requires same adversaries to be next to each other

        # interleave
        STATE = [val for pair in zip(STATE_prot_left, STATE_prot_right) for val in pair]


    else:

        STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name, PLAYER, OPPONENT_LIST[i])
                 for i
                 in range(len(OPPONENT_LIST))]
    state_list = STATE
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'],
                        help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument('--model-file', help='The model to continue to learn from')
    parser.add_argument('--save-dir', help='The directory to save the trained models',
                        default="trained_models/exploiting_%s_12_ippo_match" % PLAYER)

    parser.add_argument('--log-dir', help='The directory to save logs', default="logs")
    parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="ppo_%s" % PLAYER)
    parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile',
                        default=SF_DEFAULT_STATE)
    parser.add_argument('--side', help='The side for AI to control. By default both', default='both',
                        choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=64)
    parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(10e8))
    parser.add_argument('--video-dir', help='The path to save videos', default='videos/spar_spar_%s' % PLAYER)
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

    # PLAYER = args.player

    args = parser.parse_args()
    print("command line args:" + str(args))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.finetune_dir, exist_ok=True)

    # Set up the environment and model
    def env_generator():
        # STATE
        each_env_count = 4
        env = []
        for i in range(len(STATE)):
            for j in range(each_env_count):
                env.append(
                    make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render,
                             enable_combo=args.enable_combo, null_combo=args.null_combo,
                             transform_action=args.transform_action, seed=0))
        # env = [make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=0) for i in range(args.num_env)]
        # env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render,
        #         enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action,
        #         seed=0)
        return VecTransposeImage2P(SubprocVecEnv2P(env))
        # return SubprocVecEnv2P(env)

    def exploiter_env_generator():
        # STATE
        each_env_count = 4
        env = []
        for i in range(len(STATE)):
            for j in range(each_env_count):
                env.append(
                    make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render,
                             enable_combo=args.enable_combo, null_combo=args.null_combo,
                             transform_action=args.transform_action, seed=0))
        # env = [make_env(sf_game, state=STATE[i], side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=0) for i in range(args.num_env)]
        # env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render,
        #         enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action,
        #         seed=0)
        return VecTransposeImage2P(SubprocVecEnv2P(env))
        # return SubprocVecEnv2P(env)

    checkpoint_interval = 10000  # checkpoint_interval * num_envs = total_steps_per_checkpoint

    def finetune_model_generator(model_file=None, lr_schedule=linear_schedule(5.0e-5, 2.5e-6),
                                 other_lr_schedule=linear_schedule(5.0e-5, 2.5e-6),
                                 clip_range_schedule=linear_schedule(0.075, 0.025)):
        REMOVAL
        finetune_env = env_generator()
        finetune_model = IPPO(
            "CnnPolicy",
            finetune_env,
            device="cuda",
            verbose=1,
            n_steps=48,
            batch_size=24,  # 512,
            n_epochs=100,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            update_left=bool(args.update_left),
            update_right=bool(args.update_right),
            other_learning_rate=other_lr_schedule
        )

        finetune_model = TSS_PPO(
            "AACCnnPolicy",
            finetune_env,
            device="cuda",
            verbose=2,
            n_steps=48,
            batch_size=24,  # 512,
            n_epochs=1,
            gamma=0.94,
            v_learning_rate=1e-3, c_learning_rate=1e-4,
            d_learning_rate=5e-4, v_learning_rate_decay=critic_decay_schedule(1e-3),
            c_learning_rate_decay=critic_decay_schedule(1e-4),
            d_learning_rate_decay=critic_decay_schedule(5e-4),
            clip_range=clip_range_schedule,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            ent_coef=.01,
            dstb_ent_coef=.01,
            update_left=bool(args.update_left),
            update_right=bool(args.update_right),
        )

        if REMOVAL is None:
            num_adversary = len(OPPONENT_LIST)
        else:
            if isinstance(REMOVAL, str):
                num_adversary = 11
            else:
                assert isinstance(REMOVAL, list)
                num_adversary = 12 - len(REMOVAL)
        finetune_model = Specialized_Agent_IPPO(
            "AACCnnPolicy",
            finetune_env,
            device="cuda",
            verbose=2,
            n_steps=2048,  # 1408,
            batch_size=256,  # 2816,  # 512,
            n_epochs=1,
            gamma=0.94,
            v_learning_rate=5e-4, c_learning_rate=5e-4,
            d_learning_rate=5e-4, v_learning_rate_decay=critic_decay_schedule(1e-3),
            c_learning_rate_decay=critic_decay_schedule(1e-4),
            d_learning_rate_decay=critic_decay_schedule(5e-4),
            clip_range=clip_range_schedule,
            tensorboard_log=args.log_dir,
            seed=1,
            ent_coef=0,
            dstb_ent_coef=0,
            I_AM_LEFT=True,
            I_AM_RIGHT=False,
            num_adversary=num_adversary,
            n_global_env=args.num_env,
            n_env_per_adv=args.num_env // num_adversary,
            opp_list=OPPONENT_LIST,
            use_mirror=True
        )

        # finetune_model = Exploiter('CnnPolicy',finetune_env, device='cuda', exploited=)

        ''' 
        finetune_model = eepy(
            "AACCnnPolicy",
            finetune_env,
            device="cuda",
            verbose=2,
            n_steps=96,
            batch_size=192,  # 512,
            n_epochs=20,
            gamma=0.94,
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
        return finetune_model

    finetune_epoch_model_path = os.path.join(args.save_dir, args.model_name_prefix + f"_final_steps")
    lr_schedule = 1e-4  # if args.async_update else linear_schedule(2.5e-4, 2.5e-6)
    other_lr_schedule = 1e-4  # if args.async_update else linear_schedule(2.5e-4/args.other_timescale, 2.5e-6/args.other_timescale)
    clip_range_schedule = 0.1  # if args.async_update else linear_schedule(0.15, 0.025)
    if REMOVAL is None:
        args.num_env = env_generator().num_envs
    else:

        if isinstance(REMOVAL, str):
            args.num_env = 11 * 4
        else:
            assert isinstance(REMOVAL, list)
            args.num_env = (12 - len(REMOVAL)) * 4
    model = finetune_model_generator(args.model_file, lr_schedule=lr_schedule, other_lr_schedule=other_lr_schedule,
                                     clip_range_schedule=clip_range_schedule)
    if REMOVAL is not None:
        model.REMOVAL = REMOVAL
    if args.left_model_file and args.right_model_file:
        print("load model from " + args.left_model_file + " and " + args.right_model_file)
        model.set_parameters_2p(args.left_model_file, args.right_model_file)
    '''
    #model = IPPO.load("/home/jw4406/codebase/FightLadder/main/trained_models/single_ippo_Vega_Vega/ppo_Vega_18000000_steps.zip")
    model = TSS_PPO.load("/home/jw4406/codebase/FightLadder/main/trained_models/single_tss_Guile_Vega/ppo_Guile_45500000_steps.zip")
    model = TSS_PPO.load(
        "/home/jw4406/codebase/FightLadder/main/trained_models/single_tss_Guile_Guile/ppo_Guile_51800000_steps.zip")
    other_model = IPPO.load(
        "/home/jw4406/codebase/FightLadder/main/trained_models/single_ippo_Guile_Guile/ppo_Guile_17200000_steps.zip")
    #other_model = IPPO.lo
    #model = TSS_PPO.load("/home/jw4406/codebase/FightLadder/main/trained_models/single_tss_ft_magics_Guile_Guile/ppo_Guile_760000_steps.zip")
    data, params, pytorch_variables = load_from_zip_file("/home/jw4406/codebase/FightLadder/main/trained_models/single_tss_ft_magics_Guile_Guile/ppo_Guile_760000_steps.zip")
    model = TSS_PPO(
            "AACCnnPolicy",
            env_generator(),
            device="cuda",
            verbose=2,
            n_steps=48,
            batch_size=24,  # 512,
            n_epochs=1,
            gamma=0.94,
            v_learning_rate=1e-3, c_learning_rate=1e-4,
            d_learning_rate=5e-4, v_learning_rate_decay=critic_decay_schedule(1e-3),
            c_learning_rate_decay=critic_decay_schedule(1e-4),
            d_learning_rate_decay=critic_decay_schedule(5e-4),
            clip_range=clip_range_schedule,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            ent_coef=.01,
            dstb_ent_coef=.01,
            update_left=bool(args.update_left),
            update_right=bool(args.update_right),
        )
    '''
    data, params, pytorch_variables = load_from_zip_file(
        "/home/jw4406/codebase/FightLadder/main/trained_models/tasks/todo/ppo_Guile_78480000_steps.task")
    #model = Generalist_SPAR.load("/home/jw4406/codebase/FightLadder/main/trained_models/sa_cont_league_question_Guile/ppo_Guile_236000_steps.zip", env=env_generator())
    model = Generalist_SPAR("AACCnnPolicy",
            env_generator(),
            device="cuda",
            verbose=2,
            n_steps=768,  # 1408,
            batch_size=1536,  # 2816,  # 512,
            n_epochs=10,
            gamma=0.94,
            v_learning_rate=5e-3, c_learning_rate=1e-4,
            d_learning_rate=5e-4, v_learning_rate_decay=critic_decay_schedule(1e-3),
            c_learning_rate_decay=critic_decay_schedule(1e-4),
            d_learning_rate_decay=critic_decay_schedule(5e-4),
            clip_range=clip_range_schedule,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            ent_coef=0,
            dstb_ent_coef=0,
            I_AM_LEFT=True,
            I_AM_RIGHT=False,
            num_adversary=1,
            n_global_env=args.num_env,
            n_env_per_adv=args.num_env // 1,
            opp_list=OPPONENT_LIST,
            use_mirror=False
        )
    #model.warmstarted_cont_MAGICS=True
    #model.warmstart_setup(model.lr_schedule)
    del params['policy.ctrl_optimizer']
    del params['policy.value_optimizer']
    del params['policy.dstb_optimizer']
    model.set_parameters(params, exact_match=False, device=model.device)
    state_list = ['two_player/Guile_left/Champion.Level1.GuileVsGuile.2Player.state']
    if not EVAL:
        if args.async_update:
            model.async_learn(
                total_timesteps=args.total_steps,
                callback=[checkpoint_callback],
                fsp=args.fsp,
                fsp_threshold=args.fsp_threshold,
            )
        else:
            if hasattr(model, "num_adversaries"):
                for i in range(model.num_adversaries):
                    model.adversaries[i]._setup_learn(model.adversaries[i].num_timesteps)
            model.learn(
                total_timesteps=args.total_steps,
                callback=[checkpoint_callback])

        for i in range(len(model.adversaries)):
            model.adversaries[i].save("enemy_policy_%d.pt" % i)
        model.adversaries = []
        model.save(finetune_epoch_model_path)
    else:
        for i in range(len(state_list)):
            # global STATE
            # STATE = state_list[i]
            results = evaluate_single(state_list[i], args, model, i, record=True)
        print(results)
        with open(f"{args.finetune_dir}/{args.model_name_prefix}_start_results.txt", 'w') as f:
            f.write(str(results))


if __name__ == "__main__":
    # wandb.init(entity='jw4406')
    # wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=str, required=True)
    args = parser.parse_args()

    PLAYER = args.player
    main(PLAYER)
