import sys
import retro
import numpy as np
import torch
from gym.spaces import MultiBinary
import multiprocessing as mp
sys.path.append(".")
#sys.path.append("..")
#sys.path.append("../..")
from main.common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P
from main.common.retro_wrappers import SFWrapper, Monitor2P
from main.common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import safe_mean
from main.utils import state2matchup
from main.common.const import *
import wandb
from main.common.justin.calc_F import _calculate_policy_loss, _get_buffers_and_keys
from main.common.justin.clean_derivative_free_spar import DummyCallback

PLAYER = ["Guile"]
OPPONENT_LIST = ["Blanka"]
SIDE = "left"
envs_per_matchup = 2
num_adversaries = len(OPPONENT_LIST)
n_env_per_adv = envs_per_matchup



player_folder_name = [PLAYER[i] + '_' + SIDE for i in range(len(PLAYER))]
STATE = ["two_player/%s/Champion.Level1.%sVs%s.2Player.state" % (player_folder_name[i], PLAYER[i], OPPONENT_LIST[j])
                 for i in range(len(PLAYER))
                 for j in range(len(OPPONENT_LIST))
                 for k in range(envs_per_matchup)]



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

def env_generator(max_envs: int = 0, i_start: int = 0, j_start: int = 0):
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
        each_env_count = envs_per_matchup
        print("Generating %d envs per character matchup:" % each_env_count)
        env = []
        env_count = 0
        for i in range(i_start, len(STATE)):
            if exceed_max_envs(env_count, max_envs):
                break
            env.append(
                make_env(sf_game, state=STATE[i], side='both', reset_type="round", rendering=False,
                            enable_combo=False, null_combo=False,
                            transform_action=False, seed=0))
            env_count += 1
            if exceed_max_envs(env_count, max_envs):
                break
        return VecTransposeImage2P(SubprocVecEnv2P(env))


def test_df_policy_loss_():
    env = env_generator()
    model = CleanDerivativeFreeSPAR(
        policy="AACCnnPolicy",
        env=env,
        n_steps=2048,
        batch_size=256,
        n_epochs=1,
        num_adversaries=len(OPPONENT_LIST),
        n_env_per_adv=envs_per_matchup,
        state_list=STATE,
        envs_per_matchup=envs_per_matchup,
        env_generator_func=env_generator,
        dstb_action_space=MultiBinary(15),
        verbose=1
    )
    cb = DummyCallback()
    model._setup_learn(100, cb, True)

    perturbed_agents = [model._create_perturbed_agent()[0] for _ in range(1)] #TODO: Parallelize this.
    perturbed_bufs, perturbed_adv_bufs = zip(*[perturbed_agent.env_perturb_params() for perturbed_agent in perturbed_agents])
    model.collect_rollouts(env, cb, model.rollout_buffer, model.adversary_buffers, model.n_steps)
    model.perturbed_agents = perturbed_agents
    model.perturbed_bufs = perturbed_bufs
    model.perturbed_adv_bufs = perturbed_adv_bufs
    model.perturbed_agents_policy = [perturbed_agent.policy for perturbed_agent in perturbed_agents]
    num_runs_count = 1 if True else model.num_adversaries
    clip_range = model.clip_range(model._current_progress_remaining)
    for j in range(model.n_epochs):
        for i in range(num_runs_count):
            for perturbed_buf, perturbed_policy in zip(model.perturbed_bufs, model.perturbed_agents_policy): 
                network_keys, curr_buf, curr_perturbed_buf = _get_buffers_and_keys(model.rollout_buffer, perturbed_buf, True, i, num_adversaries)
                for ori_rollout_data, perturbed_rollout_data in zip(curr_buf.get(model.batch_size), curr_perturbed_buf.get(model.batch_size)):                    
                    policy_loss, log_prob, entropy = _calculate_policy_loss(
                        ori_rollout_data, model.policy, True, clip_range, model.use_sde, model.device, model.batch_size, model.envs_per_matchup
                    )
    policy_loss, log_prob, entropy = _calculate_policy_loss(
        model.rollout_buffer, model.policy, True, model.clip_range, model.use_sde, model.device, model.batch_size, model.envs_per_matchup
    )
    print(policy_loss, log_prob, entropy)

if __name__ == "__main__":
    test_df_policy_loss_()