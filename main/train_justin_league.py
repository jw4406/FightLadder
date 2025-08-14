import argparse
import multiprocessing as mp
import os
from functools import partial

import torch
#from common.utils.common_utils import sf_game, state_list
from common.league import (League, Learner, MainExploiter, MainPlayer,
                           PayoffManager)
from common.retro_wrappers import Monitor2P, SFWrapper
from common.utils import SubprocVecEnv2P, VecTransposeImage2P
from stable_baselines3.common.policies import \
    ActorActorCriticCnnGeneralistPolicy
from common.const import *
from common.justin.role_based_spar import RoleBasedSPAR

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

def unified_constructor(args, side, log_name, state_list):
    """
    A single factory function to create a RoleBasedSPAR agent.
    All players in the league will use this constructor.
    """
    env = VecTransposeImage2P(SubprocVecEnv2P([make_env(sf_game, state=state_list[0], side=side, rendering=args.render, reset_type=args.reset, seed=i) for i in range(args.num_env)]))
    
    policy_kwargs = {"num_adversaries": 1}

    return RoleBasedSPAR(
        policy=ActorActorCriticCnnGeneralistPolicy,
        env=env,
        device="cuda",
        verbose=1,
        n_steps=512,
        batch_size=1024,
        n_epochs=4,
        gamma=0.94,
        c_learning_rate=1e-4,
        d_learning_rate=1e-4,
        v_learning_rate=1e-4,
        clip_range=0.1,
        tensorboard_log=None if log_name is None else os.path.join(args.log_dir, log_name),
        num_adversary=len(state_list),
        n_global_env=len(state_list),
        n_env_per_adv=1,
        policy_kwargs=policy_kwargs,
        I_AM_LEFT=(side=='left'),
        I_AM_RIGHT=(side=='right'),
        envs_per_matchup=args.envs_per_matchup,
        state_len=len(state_list),
    )

def worker(idx, player, shared_payoff, total_steps, rollout_opponent_num):
    """
    Worker function that runs a learner for a given player, with role-based training.
    """
    print(f"Worker {player.name} start")
    with torch.cuda.device(idx % torch.cuda.device_count()):
        player._create_agent()
        player._payoff = shared_payoff

        if isinstance(player, MainPlayer):
            update_ego = True
            update_adversary = False
        elif isinstance(player, MainExploiter):
            update_ego = False
            update_adversary = True
        else: # Default for LeagueExploiter or other types
            update_ego = True
            update_adversary = True

        learner = Learner(player)
        learner.run(
            total_timesteps=total_steps,
            rollout_opponent_num=rollout_opponent_num,
            update_ego=update_ego,
            update_adversary=update_adversary
        )

    print(f"Worker {player.name} finished.")

def main():
    # --- Argument Parsing (from ippo.py) ---
    parser = argparse.ArgumentParser(description='Train a multi-character agent in a league.')
    parser.add_argument('--player', type=str, nargs='+', required=True, help="Protagonist player(s).")
    parser.add_argument('--opponent-list', type=str, nargs='+', default=["Guile", "EHonda"], help="List of opponent characters.")
    parser.add_argument('--side', help='The side for AI to control', default='left', choices=['left', 'right'])
    parser.add_argument('--num-env', type=int, help='Total number of environments to run in parallel', default=64)
    parser.add_argument('--envs-per-matchup', type=int, help='How many parallel environments for each matchup', default=1)
    parser.add_argument('--num-main-exploiters', type=int, help="Number of main exploiters in the league.", default=2)
    parser.add_argument('--num-league-exploiters', type=int, help="Number of league exploiters in the league.", default=4)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train per agent per generation', default=int(1e7))
    parser.add_argument('--rollout-opponent-num', type=int, help='For exploiters, numbers of opponents for each update', default=5)
    
    parser.add_argument('--save-dir', help='The directory to save the trained models', default="trained_models/justin_league")
    parser.add_argument('--log-dir', help='The directory to save logs', default="logs/justin_league")
    parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="mh_ppo")
    
    parser.add_argument('--reset', choices=['round', 'match', 'game'], default='round')
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    args = parser.parse_args()
    print("Command line args:" + str(args))
    os.makedirs(args.save_dir, exist_ok=True)
    # --- Environment State Generation (from ippo.py) ---
    player_folder_name = [p + '_' + args.side for p in args.player]
    args = parser.parse_args()

    mp.set_start_method("spawn")

    os.makedirs(args.log_dir, exist_ok=True)
    
    payoff_manager = PayoffManager()
    payoff_manager.start()
    shared_payoff = payoff_manager.Payoff()

    PLAYER = args.player
    OPPONENT_LIST = args.opponent_list
    constructor_partial = partial(unified_constructor, args=args)
    SIDE = "left"  # "right"
    player_folder_name = [PLAYER[i] + '_' + SIDE for i in range(len(PLAYER))]
    #if REMOVAL is not None:
    #    OPPONENT_LIST.remove(REMOVAL)

    # files  = os.listdir

    # if use_mirror is True:
    #     STATE_prot_left = [
    #         "two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name[i], PLAYER[i], OPPONENT_LIST[j]) for i in range(len(PLAYER)) for j in range(len(OPPONENT_LIST))]

    #     opp_left_folder_name = []
    #     for i in range(len(OPPONENT_LIST)):
    #         opp_left_folder_name.append(OPPONENT_LIST[i] + "_" + SIDE)
    #     STATE_prot_right = [
    #         "two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (opp_left_folder_name[i], OPPONENT_LIST[i], PLAYER[j])
    #         for i in range(len(OPPONENT_LIST)) for j in range(len(PLAYER))]
    #     # STATE = STATE_prot_left + STATE_prot_right

    #     # chunking requires same adversaries to be next to each other

    #     # interleave
    #     STATE = STATE_prot_left + STATE_prot_right
    #     #STATE = STATE_prot_right

    #else:

    STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name[i], PLAYER[i], OPPONENT_LIST[j])
            for i in range(len(PLAYER))
            for j in range(len(OPPONENT_LIST))]
    state_list = STATE
    initial_agents = {"main_0": constructor_partial(side='left', log_name="main_0", state_list=state_list)}

    league = League(
        args=args,
        initial_agents=initial_agents,
        constructor=constructor_partial,
        payoff=shared_payoff,
        main_agents=1,
        main_exploiters=args.num_main_exploiters,
        league_exploiters=args.num_league_exploiters
    )

    players = league.get_players()

    with mp.Pool(processes=args.num_workers) as pool:
        pool.starmap(worker, [(idx, player, shared_payoff, args.total_steps, args.rollout_opponent_num) for idx, player in enumerate(players)])

    payoff_manager.shutdown()

if __name__ == "__main__":
    main() 