import os
import torch
import argparse
import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)
from multiprocessing import Process
from typing import Dict, List

import retro

from common.const import *
#from common.utils import SubprocVecEnv2P, VecTransposeImage2P
from common.algorithms import LeaguePPO
from common.retro_wrappers import SFWrapper, Monitor2P
from common.league import PayoffManager, League, Learner, GeneralistMainPlayer, GeneralistLearner, MainExploiter, LeagueExploiter
from common.multi_head_ppo import PPO_From_SPAR
from stable_baselines3.common.policies import ActorActorCriticCnnGeneralistPolicy
from common.utils import SubprocVecEnv2P, VecTransposeImage2P

class GeneralistLeague(League):
    """
    A League that specifically creates a GeneralistMainPlayer for the main agent
    while using standard players for exploiters. This avoids modifying the base
    League class.
    """
    def __init__(self,
                 args,
                 generalist_constructor,
                 exploiter_constructor,
                 payoff=None,
                 main_agents=1,
                 main_exploiters=1,
                 league_exploiters=2):
        if payoff is None:
            self._payoff = Payoff()
        else:
            self._payoff = payoff
            
        self._learning_agents = []
        for side in ['left', 'right']: # Assuming agents on both sides
            # Create the generalist main agent
            for idx in range(main_agents):
                # The agent is initialized as None, it will be created later in the worker process
                main_agent = GeneralistMainPlayer(f"MA{idx}_{side}", side, generalist_constructor, args, None, self._payoff)
                self._learning_agents.append(main_agent)
                self.add_player(main_agent)
                self.add_player(main_agent.checkpoint())
            
            # Create Standard Main Exploiters
            for idx in range(main_exploiters):
                self._learning_agents.append(
                    MainExploiter(f"ME{idx}_{side}", side, exploiter_constructor, args, None, self._payoff))
            
            # Create Standard League Exploiters
            for idx in range(league_exploiters):
                self._learning_agents.append(
                    LeagueExploiter(f"LE{idx}_{side}", side, exploiter_constructor, args, None, self._payoff))
        
        for player in self._learning_agents:
            self.add_player(player)


def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, seed=0):
    def _init():
        players = 2
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action)
        env = Monitor2P(env)
        env.seed(seed)
        return env
    return _init

def generalist_constructor(args, side, log_name, state_list, n_env_per_adv, num_adversaries):
    """
    A factory function to create a PPO_From_SPAR agent.
    It selectively pulls the required arguments from the command-line args
    to prevent TypeErrors from unexpected keyword arguments.
    """
    env = VecTransposeImage2P(SubprocVecEnv2P([make_env(sf_game, state=s, side=side, rendering=args.render, reset_type=args.reset, seed=i) for i, s in enumerate(state_list) for _ in range(n_env_per_adv)]))
    
    policy_kwargs = {
        "num_adversaries": num_adversaries
    }

    return PPO_From_SPAR(
        policy="AACCnnPolicy",
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
        num_adversary=num_adversaries,
        n_global_env=len(state_list) * n_env_per_adv,
        n_env_per_adv=n_env_per_adv,
        policy_kwargs=policy_kwargs,
        I_AM_LEFT=(side=='left'),
        I_AM_RIGHT=(side=='right'),
    )

def exploiter_constructor(constructor_args, side, log_name, state_list):
    env = VecTransposeImage2P(SubprocVecEnv2P([make_env(sf_game, state=state_list[0], side=side, rendering=constructor_args.render, reset_type=constructor_args.reset, seed=i) for i in range(constructor_args.num_env)]))
    return LeaguePPO(
        side,
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=1024,
        n_epochs=4,
        gamma=0.94,
        learning_rate=1e-4,
        clip_range=0.1,
        tensorboard_log=None if log_name is None else os.path.join(constructor_args.log_dir, log_name),
    )

def worker(idx, player, shared_payoff, total_steps, rollout_opponent_num):
    """
    Generic worker function that initializes and runs a learner for a given player.
    """
    print(f"Worker {player.name} start")
    with torch.cuda.device(idx % torch.cuda.device_count()):
        # The agent must be created inside the worker process to avoid pickling issues.
        player._create_agent()

        # The player from the main process doesn't have a reference to the shared payoff
        # object, so we must set it here.
        player._payoff = shared_payoff

        # Choose the correct learner for the player type
        if isinstance(player, GeneralistMainPlayer):
            learner = GeneralistLearner(player)
            learner.run(total_timesteps=total_steps)
        else:
            # The 'rollout_opponent_num' is not used by the GeneralistLearner,
            # but the standard Learner expects it.
            learner = Learner(player)
            learner.run(total_timesteps=total_steps, rollout_opponent_num=rollout_opponent_num)

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
    os.makedirs(args.log_dir, exist_ok=True)
    
    # --- Environment State Generation (from ippo.py) ---
    player_folder_name = [p + '_' + args.side for p in args.player]
    STATE = [
        f"two_player/{p_folder}/Champion.Level1.{p_name}Vs{opp_name}.2Player.state"
        for p_folder, p_name in zip(player_folder_name, args.player)
        for opp_name in args.opponent_list
    ]
    
    num_adversaries = len(args.opponent_list)
    
    # Ensure total envs is divisible by number of adversaries for clean slicing
    if args.num_env % num_adversaries != 0:
        raise ValueError(f"Total number of environments ({args.num_env}) must be divisible by the number of adversaries ({num_adversaries}).")
    n_env_per_adv = args.num_env // num_adversaries
    
    # --- Agent Constructors ---
    # Moved to top level to be pickleable

    # --- League Setup and Execution (from train_ma.py) ---
    
    # Create lightweight "template" agents. The full agents will be constructed in subprocesses.
    # We need to bind the arguments to the constructor functions
    from functools import partial
    gen_constructor_partial = partial(generalist_constructor, state_list=STATE, n_env_per_adv=n_env_per_adv, num_adversaries=num_adversaries)
    exp_constructor_partial = partial(exploiter_constructor, state_list=STATE)

    with PayoffManager() as manager:
        shared_payoff = manager.Payoff(args.save_dir)
        
        league = GeneralistLeague(
            args=args, 
            generalist_constructor=gen_constructor_partial,
            exploiter_constructor=exp_constructor_partial,
            payoff=shared_payoff, 
            main_agents=1, 
            main_exploiters=args.num_main_exploiters, 
            league_exploiters=args.num_league_exploiters
        )
        
        processes = []
        for idx in range(league.size()):
            player = league.get_player(idx)
            
            # The player object is passed to the worker, which will then
            # create the agent and the appropriate learner. This avoids pickling the agent/env.
            process = Process(target=worker, args=(idx, player, shared_payoff, args.total_steps, args.rollout_opponent_num))
            processes.append(process)

        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    main() 