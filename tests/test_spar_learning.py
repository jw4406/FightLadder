import sys
import retro
import numpy as np
import torch
from gym.spaces import MultiBinary
import multiprocessing as mp
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from main.common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P
from main.common.retro_wrappers import SFWrapper, Monitor2P
from main.common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import safe_mean
from main.utils import state2matchup
from main.common.const import *
import wandb

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


def test_ego_forward():
    print("--- Running Test: Ego Forward ---")
    
    #def env_generator_func():
    #    return DummyVecEnv([lambda: make_env(side="left")])

    env = env_generator()
    
    #state_list = ["Champion.Level1.RyuVsRyu"]
    #num_adversaries = 1
    #n_env_per_adv = 1
    #envs_per_matchup = 1
    
    model = CleanDerivativeFreeSPAR(
        policy="AACCnnPolicy",
        env=env,
        n_steps=2048,
        batch_size=512,
        n_epochs=1,
        num_adversaries=num_adversaries,
        n_env_per_adv=n_env_per_adv,
        state_list=STATE,
        envs_per_matchup=envs_per_matchup,
        env_generator_func=env_generator,
        dstb_action_space=MultiBinary(15),
        verbose=1
    )

    model.learn(total_timesteps=model.n_steps * 15, update_ego=True, update_adversary=False)
    
    mean_reward = safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer])
    print(f"Ego forward final mean reward: {mean_reward}")
    
    assert mean_reward > 0.3, f"Mean reward {mean_reward} is not > 0.3"
    env.close()

def test_adversary_forward():
    print("--- Running Test: Adversary Forward ---")

    #def env_generator_func():
    #    return DummyVecEnv([lambda: make_env(side="right")])

    env = env_generator()
    

    model = CleanDerivativeFreeSPAR(
        policy="AACCnnPolicy",
        env=env,
        n_steps=2048,
        batch_size=512,
        n_epochs=1,
        num_adversaries=num_adversaries,
        n_env_per_adv=n_env_per_adv,
        state_list=STATE,
        envs_per_matchup=envs_per_matchup,
        env_generator_func=env_generator,
        dstb_action_space=MultiBinary(15),
        verbose=1
    )

    model.learn(total_timesteps=model.n_steps * 15, update_ego=False, update_adversary=True)

    mean_reward = safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer])
    print(f"Adversary forward final mean reward: {mean_reward}")
    
    assert mean_reward < -0.3, f"Mean reward {mean_reward} is not < -0.3"
    env.close()

def test_advantage_value_rewards_sign():

    env = env_generator()
    

    model = CleanDerivativeFreeSPAR(
        policy="AACCnnPolicy",
        env=env,
        n_steps=2048,
        batch_size=512,
        n_epochs=1,
        num_adversaries=num_adversaries,
        n_env_per_adv=n_env_per_adv,
        state_list=STATE,
        envs_per_matchup=envs_per_matchup,
        env_generator_func=env_generator,
        dstb_action_space=MultiBinary(15),
        verbose=1
    )
    model.learn(total_timesteps=model.n_steps, update_ego=False, update_adversary=True)
    assert torch.allclose(model.adversary_buffers[0].values + model.rollout_buffer.values, torch.zeros_like(model.adversary_buffers[0].values))
    assert np.allclose(model.adversary_buffers[0].rewards + model.rollout_buffer.rewards, np.zeros_like(model.adversary_buffers[0].rewards))
    summed_advantages = model.rollout_buffer.advantages + model.adversary_buffers[0].advantages
    assert torch.allclose(summed_advantages, torch.zeros_like(summed_advantages))
    print(f"Summed advantages: {summed_advantages.mean().item():.4f}") 

if __name__ == "__main__":
    wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    wandb.init(project="spar-learning", name="test_ego_forward",config={"eval_rew": 0,
                               "epochs": 0})
    mp.set_start_method("spawn", force=True) #A lot of stable_baseline3 objects don't support the default "fork".
    #test_ego_forward()
    #test_adversary_forward()
    test_advantage_value_rewards_sign()