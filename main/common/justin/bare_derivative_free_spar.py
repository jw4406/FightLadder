import torch, torch as th, torch.autograd as autograd
import sys, time, random
from venv import create
import wandb
import numpy as np
import torch.nn as nn
from anyio import value
from gym import spaces
import torch.nn.functional as F
from copy import deepcopy
from collections import deque
from functorch import vmap as eepy
from retro.examples.brute import rollout
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.policies import ActorActorCriticCnnGeneralistPolicy, bare_AACCNNGP
from typing import Union, Type, Optional, Dict, Any
from .spar import Single_SPAR
from .Doubly_TSS_SPAR import Doubly_TSS_SPAR
from typing import List
from typing import Callable
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn, \
    update_learning_rate, is_vectorized_observation, polyak_update
from utils import move_policy, select_device, get_n_workers, state2matchup, select_matchup_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback


class BareDerivativeFreeSPAR(Single_SPAR):
    policy_aliases = {
        "AACCnnPolicy": bare_AACCNNGP
    }
    def __init__(self,
        policy: Union[str, Type[ActorActorCriticCnnGeneralistPolicy]],
        env: Union[GymEnv, str],
        c_learning_rate: Union[float, Schedule] = 1e-4,
        d_learning_rate: Union[float, Schedule] = 7e-4,
        v_learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        batch_size: int = 64,
        n_epochs: int = 1,
        clip_range: float = 0.2,
        clip_range_vf: float = 0.2,
        normalize_advantage: bool = False,
        c_learning_rate_decay: float = 1e-4 * 7e-4,
        d_learning_rate_decay: float = 7e-4,
        v_learning_rate_decay: float = 7e-4,
        I_AM_LEFT: bool = True,
        I_AM_RIGHT: bool = False,
        num_adversary: int = 4,
        n_global_env: int = None,
        n_env_per_adv: int = 1,
        opp_list: List[str] = None,
        player: str = None,
        use_mirror: bool = False,
        env_generator_func: Callable = None,
        dstb_ent_coef: float = 0.0,
        dstb_action_space: spaces.Space = None,
        update_left: bool = True,
        update_right: bool = True   ,
        sde_sample_freq: int = -1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        env_batch_size: int = 32,
        envs_per_matchup: int = 1,
        state_len: int = 1,
        state_list: List[List[str]] = None,

    ):
        self.matchups = [state2matchup(state) for state in state_list] #This needs to happen before the super().__init__
        self.envs_per_matchup = envs_per_matchup
        self.num_adversaries = num_adversary
        self.state_len = state_len
        self.env_generator_func = env_generator_func
        self.n_global_env = n_global_env
        self.use_mirror = use_mirror
        super().__init__(
            policy=policy,
            env=env,
            c_learning_rate=c_learning_rate,
            d_learning_rate=d_learning_rate,
            v_learning_rate=v_learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.n_env_per_adv = n_env_per_adv
        adversary_buffers = []
        for i in range(num_adversary):
            overwrite = Doubly_TSS_SPAR("AACCnnPolicy",
                                       self.env,
                                       device=self.device,
                                       verbose=self.verbose,
                                       n_steps=self.n_steps,
                                       batch_size=self.batch_size // self.n_envs,  # 512,
                                       n_epochs=self.n_epochs,
                                       gamma=self.gamma,
                                       v_learning_rate=v_learning_rate, c_learning_rate=c_learning_rate,
                                       d_learning_rate=d_learning_rate, v_learning_rate_decay=v_learning_rate_decay,
                                       c_learning_rate_decay=c_learning_rate_decay,
                                       d_learning_rate_decay=d_learning_rate_decay,
                                       clip_range=self.clip_range,
                                       tensorboard_log=self.tensorboard_log,
                                       seed=self.seed,
                                       ent_coef=self.ent_coef,
                                       dstb_ent_coef=self.dstb_ent_coef,
                                       update_left=not self.update_left,
                                       update_right=not self.update_right,
                                       warmstarted_cont_MAGICS=False,
                                       matchups=self.matchups,
                                       envs_per_matchup=self.envs_per_matchup
                                       )
            overwrite.rollout_buffer.n_envs = self.n_env_per_adv
            adversary_buffers.append(overwrite.rollout_buffer)
        self.adversary_buffers = adversary_buffers
        self.env_batch_size = env_batch_size
        print("hello")
        if self.policy is not None: 
            self.policy.num_env_per_adv = self.envs_per_matchup
            self.policy.num_global_env = self.n_global_env

    def collect_rollouts(
                        self,
                        env: VecEnv,
                        callback: BaseCallback,
                        rollout_buffer: RolloutBuffer,
                        adversary_buffers,
                        n_rollout_steps: int,
                        ) -> bool:
        """Override to use batched environments for memory management"""
        def _calc_i_start_j_start(env_cnt: int, envs_per_matchup: int) -> tuple:
            """
            This helper functino calcaultes the start of i and j to be used in env_generator_func.

            Args:
                env_cnt (int):
                    How many environments were created.
                envs_per_matchup (int):
                    Environments to create per matchup.
            
            Returns:
                i_start (int):
                    i to start env_generator_func.
                j_start (int):
                    j to start env_generator_func.
            """
            #i_start = env_cnt // envs_per_matchup
            #j_start = env_cnt  % envs_per_matchup
            #i_start = 0
            j_start = 0
            i_start = env_cnt
            return i_start, j_start
        ct = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        for i in range(self.rollout_buffer.buffer_size):
            _, test, _ = self.policy.evaluate_actions(self.rollout_buffer.observations[i].to(self.device), self.rollout_buffer.actions[i].to(self.device))
            assert th.allclose(self.rollout_buffer.log_probs[i], test.cpu())
        #self.rollout_buffer.prepare_data_for_training()
        return ct
        # Create rollout environments in batches using the stored generator function
        total_envs_needed = self.state_len# * self.envs_per_matchp
        # we modified state_list to hold the master list of envs with all the diversity duplicates and everything. 
        rollout_buffer.reset()
        for buf in adversary_buffers:
            buf.reset()
        #total_batches = (total_envs_needed + self.env_batch_size - 1) // self.env_batch_size
        # i dont quite understand why this computation is correct
        total_batches = total_envs_needed // self.env_batch_size
        if total_envs_needed % self.env_batch_size != 0:
            raise ValueError("total_envs_needed must be divisible by env_batch_size")
        #if self.envs_per_matchup % self.env_batch_size != 0:
        #    raise ValueError("env_batch_size must be divisible by envs_per_matchup")
            # do not allow grabbing splits (i.e., A A B B A | A B B A B ) 
            # only allow (A A | B B ...) or (A A B B | A A B B ...)
        # need to do a rem check here

        env_cnt = 0 #how many environments were created
        #if total_batches != 1:
        flat_elements = []
        adv_flat_elements = list(adversary_buffers[0].obs_shape)
        adv_flat_elements.insert(0, adversary_buffers[0].buffer_size)
        adv_flat_elements.insert(1, self.envs_per_matchup)
        for item in (rollout_buffer.buffer_size,total_envs_needed, rollout_buffer.obs_shape):
            if isinstance(item, tuple):
                flat_elements.extend(item)  # Recursively flatten nested tuples
            else:
                flat_elements.append(item)  # Add integers directly
        #shape = np.flatten((rollout_buffer.buffer_size,total_envs_needed, rollout_buffer.obs_shape))
        ego_vertical_batch_obs = th.empty(flat_elements, pin_memory=True)
        adv_vertical_batch_obs = [th.empty(adv_flat_elements, pin_memory=True) for _ in range(self.num_adversaries)]
        ego_vertical_batch_rewards = th.empty(np.shape(rollout_buffer.rewards), pin_memory=True)
        adv_vertical_batch_rewards = [th.empty(np.shape(adversary_buffers[i].rewards), pin_memory=True) for i in range(self.num_adversaries)]
        #vertical_batch_rewards_other = np.empty(np.shape(rollout_buffer.rewards_other))
        ego_vertical_batch_dones = th.empty(np.shape(rollout_buffer.dones), pin_memory=True)
        adv_vertical_batch_dones = [th.empty(np.shape(adversary_buffers[i].dones), pin_memory=True) for i in range(self.num_adversaries)]
        #vertical_batch_infos = np.empty(np.shape(rollout_buffer.infos))
        ego_vertical_batch_log_probs = th.empty(np.shape(rollout_buffer.log_probs))
        adv_vertical_batch_log_probs = [th.empty(np.shape(adversary_buffers[i].log_probs)) for i in range(self.num_adversaries)]
        ego_vertical_batch_values = th.empty(np.shape(rollout_buffer.values)).to(self.device)
        adv_vertical_batch_values = [th.empty(np.shape(adversary_buffers[i].values), pin_memory=True) for i in range(self.num_adversaries)]
        ego_vertical_batch_dstb_log_probs = th.empty(np.shape(rollout_buffer.dstb_log_probs), pin_memory=True)
        adv_vertical_batch_dstb_log_probs = [th.empty(np.shape(adversary_buffers[i].dstb_log_probs), pin_memory=True) for i in range(self.num_adversaries)]
        ego_vertical_batch_last_ep_starts = th.empty(np.shape(rollout_buffer.episode_starts)).to(self.device)
        adv_vertical_batch_last_ep_starts = [th.empty(np.shape(adversary_buffers[i].episode_starts), pin_memory=True) for i in range(self.num_adversaries)]
        last_ep_starts = th.empty(np.shape(rollout_buffer.episode_starts), pin_memory=True)
        adv_last_ep_starts = [th.empty(np.shape(adversary_buffers[i].episode_starts), pin_memory=True) for i in range(self.num_adversaries)]
        ego_vertical_batch_actions = th.empty(np.shape(rollout_buffer.actions), pin_memory=True)
        adv_vertical_batch_actions = [th.empty(np.shape(adversary_buffers[i].actions), pin_memory=True) for i in range(self.num_adversaries)]
        ego_vertical_batch_adversary_actions = th.empty(np.shape(rollout_buffer.dstb_actions), pin_memory=True)
        adv_vertical_batch_adversary_actions = [th.empty(np.shape(adversary_buffers[i].dstb_actions), pin_memory=True) for i in range(self.num_adversaries)]

        final_obs_all_envs = th.empty(rollout_buffer.observations[0].shape, device=self.device)
        final_dones_all_envs = np.empty(rollout_buffer.dones[0].shape)
        for batch_idx in range(total_batches):
            # if total_batches != 1:
            #     flat_elements = []
            #     for item in (rollout_buffer.buffer_size,total_envs_needed, rollout_buffer.obs_shape):
            #         if isinstance(item, tuple):
            #             flat_elements.extend(item)  # Recursively flatten nested tuples
            #         else:
            #             flat_elements.append(item)  # Add integers directly
            #     #shape = np.flatten((rollout_buffer.buffer_size,total_envs_needed, rollout_buffer.obs_shape))
            #     vertical_batch_obs = np.empty(flat_elements)
            i_start, j_start = _calc_i_start_j_start(env_cnt, self.envs_per_matchup)
            rollout_env = self.env_generator_func(max_envs=self.env_batch_size, i_start=i_start, j_start=j_start)
            network_keys = [i // self.envs_per_matchup for i in range(i_start, i_start + self.env_batch_size)]
            # indexing state_list will use i_start : i_start + self.env_batch_size

            env_cnt += rollout_env.num_envs
            self._last_obs = rollout_env.reset()  # Set initial observations for this batch
            self._last_episode_starts = np.ones(rollout_env.num_envs)
            # Call the parent's collect_rollouts with our batched environment
            
            env = rollout_env
            assert self._last_obs is not None, "No previous observation was provided"
            # Switch to eval mode (this affects batch norm / dropout)
            self.policy.set_training_mode(True)

            n_steps = 0
            #rollout_buffer.reset()
            for i in range(self.num_adversaries):
                adversary_buffers[i].reset()
            # Sample new weights for the state dependent exploration
            if self.use_sde:
                self.policy.reset_noise(env.num_envs)

            # need to sample leader policy here
            #for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
            #    self.policy.ctrl_optimizer.param_groups[0]['params'][i] = torch.nn.init.uniform_(self.policy.ctrl_optimizer.param_groups[0]['params'][i], a=-1., b=1.)

            callback.on_rollout_start()
            count = 0
            while n_steps < n_rollout_steps:
                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.policy.reset_noise(env.num_envs)

                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict

                    # PROBLEM HERE:
                    # we need to only call the right heads here cause adversary list may not be the 
                    # full thing since we're chunking/cycling the envs!


                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    s_actions, s_log_probs, s_values, s_dstb_actions, s_dstb_log_probs = self.policy(obs_tensor, network_keys=network_keys)
                    all_adv_critic_values = torch.zeros((self.n_global_env, 1), device=self.device)
                actions = s_actions
                adversary_actions = s_dstb_actions
                log_probs = s_log_probs
                adversary_log_probs = s_dstb_log_probs
                actions = actions.cpu().numpy()
                adversary_actions = adversary_actions.cpu().numpy()
                all_adv_critic_values = s_values

                if self.use_mirror is True:
                    mirror_master_copy_actions = deepcopy(actions)
                    mirror_master_copy_adv_actions = deepcopy(adversary_actions)

                # upper half, lower half

                
                if self.use_mirror is True:
                    # print("SINGLE TRAIN EXTRACTOR MIRROR")

                    '''
                    assume wlog Ehonda is the prot.

                    action right now is:                  adv_action right now is:
                    EHonda left                                              Sagat    right
                    EHonda left                                              Sagat    right
                    EHonda left                                             MBison    right
                    EHonda left                                             MBison    right

                    EHonda v Sagat       0
                    Sagat v. EHonda      1
                    EHonda v. MBison     2
                    MBison v. EHonda     3

                    action[odds] needs to go to the other side because our design makes prot actions left

                    same with adversary[odds] -- adversary is on the right so adv[ods] is backwards

                    '''
                    halfway = actions.shape[0] // 2 #halfway split between upper & lower + left & right
                    
                    if DEBUG:
                        #test = np.zeros_like(actions)
                        #other_test = np.ones_like(actions)
                        #test_left = test[halfway:, :]
                        #test_right = other_test[:halfway, :]
                        #temp = np.zeros((self.num_adversaries, self.action_space.shape[0]))
                        #temp[:halfway, :] = test_left
                        #temp[halfway:, :] = test_right

                        test2 = np.zeros_like(actions)
                        count = 0
                        for i in range(test2.shape[0]):
                            for j in range(test2.shape[1]):
                                test2[i, j] = count
                                count += 1
                        other_test2 = np.zeros_like(actions)
                        count = other_test2.size - 1
                        for i in range(other_test2.shape[0]):
                            for j in range(other_test2.shape[1]):
                                other_test2[i, j] = count
                                count -= 1
                        prot_left = test2[:halfway, :]  # actions for the prot when he is on the left
                        prot_left_pre = test2[halfway:, :]  

                        adv_right = other_test2[:halfway, :]
                        adv_right_pre = other_test2[halfway:, :]

                        prot_actions = np.empty_like(actions)
                        prot_actions[:halfway, :] = prot_left
                        prot_actions[halfway:, :] = adv_right_pre

                        adv_actions = np.empty_like(actions)
                        adv_actions[:halfway, :] = adv_right
                        adv_actions[halfway:, :] = prot_left_pre

                        #print("temp2", temp2)
                        #print("other_test2", other_test2)
                        #print("test2_left", test2_left)
                        #print("test2_right", test2_right)
                        #print("actions", actions)
                        #print("temp", temp)
                        #print("other_test", other_test)
                        #print("test_left", test_left)
                        #print("test_right", test_right)
                        #print("actions", actions)

                    prot_left = actions[:halfway, :]  # actions for the prot when he is on the left
                    prot_left_pre = actions[halfway:, :]  

                    adv_right = adversary_actions[:halfway, :]
                    adv_right_pre = adversary_actions[halfway:, :]

                    prot_actions = np.empty_like(actions)
                    #temp = prot_right
                    prot_actions[:halfway, :] = prot_left
                    prot_actions[halfway:, :] = adv_right_pre

                    adv_actions = np.empty_like(actions)
                    adv_actions[:halfway, :] = adv_right
                    adv_actions[halfway:, :] = prot_left_pre

                    actions = prot_actions
                    adversary_actions = adv_actions

                # Rescale and perform action
                if self.update_left is True:
                    # MESSY
                    clipped_actions = np.hstack([actions, adversary_actions])
                else:
                    clipped_actions = np.hstack([adversary_actions, actions])
                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, spaces.Box):
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                new_obs, rewards, rew_other, dones, infos = env.step(clipped_actions)

                # if mirroring:
                # rew = (r,r,r, -r, -r, -r)^T 
                if self.use_mirror is True:
                    halfway = len(rewards) // 2
                    rewards[halfway:] = -rewards[halfway:]
                    # now rew = (r, r, r, r, r, r)^T
                    # this is the correct ego reward
                
                # if mirror is false
                # rew is already (r,r,r,r,r,r)^T and we dont need to do anything
                
                #if np.any(rewards != 0):
                #    print("Reward is not 0")
                ego_vertical_batch_obs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :, :, :] = th.unsqueeze(th.from_numpy(self._last_obs), 0)
                ego_vertical_batch_rewards[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(rewards), 0)
                #vertical_batch_rewards_other[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :] = th.unsqueeze(th.from_numpy(rew_other), 0)
                ego_vertical_batch_dones[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(dones), 0)
                #vertical_batch_infos[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :] = th.unsqueeze(th.from_numpy(infos), 0)
                ego_vertical_batch_log_probs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(log_probs, 0).cpu()
                ego_vertical_batch_values[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(s_values, 0)
                ego_vertical_batch_dstb_log_probs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(s_dstb_log_probs, 0).cpu()
                ego_vertical_batch_last_ep_starts[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(self._last_episode_starts), 0)
                ego_vertical_batch_actions[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(actions), 0)
                ego_vertical_batch_adversary_actions[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(adversary_actions), 0)
                
                # For each environment in the batch, assign its data to the correct adversary buffer and slot.
                for j in range(env.num_envs):
                    # Calculate the global index of the environment across all batches.
                    global_env_idx = i_start + j
                    #print(global_env_idx)
                    
                    # Determine the matchup this environment belongs to. This is the index for the adversary buffer.
                    matchup_idx = global_env_idx // self.envs_per_matchup
                    
                    # Determine the local index of the environment within its matchup group. This is the slot in the buffer.
                    local_env_idx = global_env_idx % self.envs_per_matchup

                    # Place the observation, reward, and done status into the correct buffer and slot.
                    # `count` is the current step in the rollout.
                    adv_vertical_batch_obs[matchup_idx][count, local_env_idx] = obs_as_tensor(new_obs[j], device='cpu')
                    # we need to flip adversary rewards because adversary always gets -r
                    # recall right now that rew = (r, r, r, r, r, r)^T in BOTH cases! (mirror or not)

                    # so we flip every element 
                    adv_vertical_batch_rewards[matchup_idx][count, local_env_idx] = -rewards[j]
                    #adv_vertical_batch_dones[matchup_idx][count, local_env_idx] = dones[j]
                    adv_vertical_batch_log_probs[matchup_idx][count, local_env_idx] = log_probs[j]
                    adv_vertical_batch_values[matchup_idx][count, local_env_idx] = -all_adv_critic_values[j]
                    adv_vertical_batch_dstb_log_probs[matchup_idx][count, local_env_idx] = s_dstb_log_probs[j]
                    adv_vertical_batch_last_ep_starts[matchup_idx][count, local_env_idx] = th.from_numpy(self._last_episode_starts)[j]
                    #last_ep_starts[global_env_idx] = th.from_numpy(np.round(dones[j]).astype(bool))

                    adv_vertical_batch_actions[matchup_idx][count, local_env_idx].copy_(th.from_numpy(actions[j]))
                    adv_vertical_batch_adversary_actions[matchup_idx][count, local_env_idx].copy_(th.from_numpy(adversary_actions[j]))


                self.num_timesteps += env.num_envs
                wandb.log({"epochs": self.num_timesteps})
                # Give access to local variables
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

                self._update_info_buffer(infos)
                n_steps += 1

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)
                count += 1

                self._last_obs = new_obs
                self._last_episode_starts = dones
            i_start = batch_idx * self.env_batch_size
            final_obs_all_envs[i_start : i_start + self.env_batch_size] = obs_as_tensor(new_obs, self.device)
            final_dones_all_envs[i_start : i_start + self.env_batch_size] = dones

            rollout_env.close()  # Clean up this batch
            result = True
            
            if not result:  # If parent method returned False, propagate it
                return False
        
        rollout_buffer.observations = ego_vertical_batch_obs
        rollout_buffer.rewards.copy_(ego_vertical_batch_rewards)
        #rollout_buffer.dones = ego_vertical_batch_dones
        rollout_buffer.log_probs = ego_vertical_batch_log_probs
        rollout_buffer.values = ego_vertical_batch_values
        rollout_buffer.dstb_log_probs = ego_vertical_batch_dstb_log_probs
        rollout_buffer.episode_starts = ego_vertical_batch_last_ep_starts
        rollout_buffer.actions = ego_vertical_batch_actions
        rollout_buffer.adversary_actions = ego_vertical_batch_adversary_actions
        

        for i in range(len(adversary_buffers)):
            adversary_buffers[i].observations = adv_vertical_batch_obs[i]
            adversary_buffers[i].rewards = adv_vertical_batch_rewards[i]
            #adversary_buffers[i].dones = adv_vertical_batch_dones[i]
            adversary_buffers[i].log_probs = adv_vertical_batch_log_probs[i]
            adversary_buffers[i].values = adv_vertical_batch_values[i]
            adversary_buffers[i].dstb_log_probs = adv_vertical_batch_dstb_log_probs[i]
            adversary_buffers[i].episode_starts = adv_vertical_batch_last_ep_starts[i]
            adversary_buffers[i].actions = adv_vertical_batch_actions[i]
            adversary_buffers[i].adversary_actions = adv_vertical_batch_adversary_actions[i]

        rollout_buffer.full = True
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].full = True

        
        with th.no_grad():
            # Compute value for the last time:w
            # step
            #values = torch.zeros((self.n_global_env,))
            values = self.policy.predict_values(final_obs_all_envs)

        rollout_buffer.values = rollout_buffer.values.to(self.device, non_blocking=True)
        rollout_buffer.rewards = rollout_buffer.rewards.to(self.device, non_blocking=True)
        rollout_buffer.advantages = rollout_buffer.advantages.to(self.device, non_blocking=True)
        rollout_buffer.episode_starts = rollout_buffer.episode_starts.to(self.device, non_blocking=True)
        #rollout_buffer.vectorized_compute_returns_and_advantages(last_values=values, dones=torch.Tensor(dones).to(self.device))
        rollout_buffer.vectorized_compute_returns_and_advantages(last_values=values, dones=final_dones_all_envs)
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].values = adversary_buffers[i].values.to(self.device, non_blocking=True)
            adversary_buffers[i].rewards = adversary_buffers[i].rewards.to(self.device, non_blocking=True)
            adversary_buffers[i].advantages = adversary_buffers[i].advantages.to(self.device, non_blocking=True)
            adversary_buffers[i].episode_starts = adversary_buffers[i].episode_starts.to(self.device, non_blocking=True)
            #adversary_buffers[i].vectorized_compute_returns_and_advantages(last_values=values, dones=final_dones_all_envs)
            

            start_idx = i * self.envs_per_matchup
            end_idx = (i + 1) * self.envs_per_matchup
            adv_last_values = values[start_idx:end_idx]
            adv_dones = final_dones_all_envs[start_idx:end_idx]
            adversary_buffers[i].vectorized_compute_returns_and_advantages(last_values=-adv_last_values, dones=adv_dones)
        
        callback.on_rollout_end()

        rollout_buffer.prepare_data_for_training()
        for buf in adversary_buffers:
            buf.prepare_data_for_training()
        
        return True  

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.train_standard(adversary_index=None)
        
    def train_standard(self, adversary_index: int = None) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [self.policy.ctrl_optimizer, self.policy.dstb_optimizer, self.policy.value_optimizer])
        clip_range = self.clip_range(self._current_progress_remaining)
        #clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses = []
        pg_losses, value_losses = [], []
        approx_kl_divs = []
        clip_fractions = []

        self.policy.set_training_mode(False)
        # with th.no_grad():
        #     _, recomputed_log_probs, _ = self.policy.evaluate_actions(self.rollout_buffer.observations, self.rollout_buffer.actions)

        # stored_log_probs = self.rollout_buffer.log_probs
        
        # if not th.allclose(recomputed_log_probs, stored_log_probs):
        #     print("!!! LOG PROBS DO NOT MATCH !!!")
        #     diff = th.abs(recomputed_log_probs - stored_log_probs)
        #     print(f"Max difference: {th.max(diff).item()}")
        #     print(f"Mean difference: {th.mean(diff).item()}")
        #     max_diff_idx = th.argmax(diff).item()
        #     print(f"Index of max difference: {max_diff_idx}")
        #     print(f"Recomputed log_prob at index {max_diff_idx}: {recomputed_log_probs[max_diff_idx].item()}")
        #     print(f"Stored log_prob at index {max_diff_idx}: {stored_log_probs[max_diff_idx].item()}")
        #     print(f"Observation at index {max_diff_idx}: {self.rollout_buffer.observations[max_diff_idx]}")
        #     print(f"Action at index {max_diff_idx}: {self.rollout_buffer.actions[max_diff_idx]}")

        # assert th.allclose(recomputed_log_probs, stored_log_probs), "Log probabilities do not match between collection and training."
        self.policy.set_training_mode(True)
        for i in range(self.rollout_buffer.buffer_size):    
            _, test, _ = self.policy.evaluate_actions(self.rollout_buffer.observations[i].to(self.device), self.rollout_buffer.actions[i].to(self.device))
            assert th.allclose(self.rollout_buffer.log_probs[i], test.cpu())
        test_obs = deepcopy(self.rollout_buffer.observations)
        test_actions = deepcopy(self.rollout_buffer.actions)
        test_old_log_prob = deepcopy(self.rollout_buffer.log_probs)
        #check oirder
        #test_actions = deepcopy(self.rollout_buffer.actions[0])
        #test_old_log_prob = deepcopy(self.rollout_buffer.log_probs[0].flatten())
        self.rollout_buffer.observations = self.rollout_buffer.observations.cpu().numpy()
        self.rollout_buffer.actions = self.rollout_buffer.actions.cpu().numpy()
        self.rollout_buffer.log_probs = self.rollout_buffer.log_probs.cpu().numpy()
        for epoch in range(self.n_epochs):

            for i in range(self.rollout_buffer.buffer_size):    
                _, pre_test, _ = self.policy.evaluate_actions(th.from_numpy(self.rollout_buffer.observations[i]).to(self.device), th.from_numpy(self.rollout_buffer.actions[i]).to(self.device))
                assert th.allclose(th.from_numpy(self.rollout_buffer.log_probs[i]), pre_test.cpu())
            for rollout_data in self.rollout_buffer.get(batch_size=1):

                for i in range(self.rollout_buffer.buffer_size):    
                    _, post_test, _ = self.policy.evaluate_actions(th.from_numpy(self.rollout_buffer.observations).to(self.device), th.from_numpy(self.rollout_buffer.actions).to(self.device))
                    assert th.allclose(th.from_numpy(self.rollout_buffer.log_probs), post_test.cpu())
                count = 0
                for j in range(8):
                    for i in range(300):
                        #assert False not in (rollout_data.observations[count] == test_obs[i, j])
                        #assert False not in (rollout_data.actions[count] == test_actions[i, j])
                        #assert False not in (rollout_data.old_log_prob[count] == test_old_log_prob[i, j])
                        count = count + 1
                #assert False not in (rollout_data.observations[0:8] == test_obs)
                #assert False not in (rollout_data.actions[0:8] == test_actions)
                #assert False not in (rollout_data.old_log_prob[0:8] == test_old_log_prob)
                _,rollout_data_test, _ = self.policy.evaluate_actions(th.from_numpy(rollout_data.observations[0:8]).to(self.device), th.from_numpy(rollout_data.actions[0:8]).to(self.device))
                _, buf_test, _ = self.policy.evaluate_actions(th.from_numpy(self.rollout_buffer.observations[0:8]).to(self.device), th.from_numpy(self.rollout_buffer.actions[0:8]).to(self.device))

                #assert th.allclose(self.rollout_buffer.log_probs[0:8], buf_test.cpu())
                #assert th.allclose(rollout_data.old_log_prob[0:8], rollout_data_test.cpu())
                actions = rollout_data.actions
                dstb_actions = rollout_data.dstb_actions
                
                # values, log_prob, entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                #     rollout_data.observations,
                #     actions,
                #     dstb_actions,
                #     shuffle_keys=rollout_data.env_indices,
                #     network_keys=[adversary_index] if adversary_index is not None else list(range(self.num_adversaries)),
                #     envs_per_matchup=self.envs_per_matchup
                # )

                values, log_prob, entropy = self.policy.evaluate_actions(
                    th.from_numpy(rollout_data.observations).to(self.device), th.from_numpy(actions).to(self.device))
                
                values = values.flatten()
                value_loss = F.mse_loss(rollout_data.returns.to(self.device), values)
                value_losses.append(value_loss.item())
                
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                ratio = torch.exp(log_prob - torch.Tensor(rollout_data.old_log_prob).to(self.device))
                # policy_loss_1 = advantages.to(self.device) * ratio
                # policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # pg_losses.append(policy_loss.item())
                # clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                # clip_fractions.append(clip_fraction)

                # if self.clip_range_vf is None:
                #     values_pred = values
                # else:
                #     values_pred = rollout_data.old_values + torch.clamp(
                #         values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                #     )
                # value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred)
                # value_losses.append(value_loss.item())

                # entropy_loss = -entropy.mean()
                # entropy_losses.append(entropy_loss.item())
                
                # #dstb_entropy_loss = -dstb_entropy.mean()
                # #dstb_entropy_losses.append(dstb_entropy_loss.item())
                
                # entropy_loss = -entropy.mean()
                # entropy_losses.append(entropy_loss.item())
                
                # #dstb_entropy_loss = -dstb_entropy.mean()
                # #dstb_entropy_losses.append(dstb_entropy_loss.item())
                
                # entropy_loss = -entropy.mean()
                # entropy_losses.append(entropy_loss.item())
                # with th.no_grad():
                #     log_ratio = log_prob.detach().cpu() - rollout_data.old_log_prob.cpu()
                #     approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                #     approx_kl_divs.append(approx_kl_div)
                
                # #dstb_entropy_loss = -dstb_entropy.mean()
                # #dstb_entropy_losses.append(dstb_entropy_loss.item())
                
                # loss = policy_loss + value_loss

                # self.policy.ctrl_optimizer.zero_grad()
                # #self.policy.dstb_optimizer.zero_grad()
                # self.policy.value_optimizer.zero_grad()
                
                # loss.backward()
                
                # self.policy.ctrl_optimizer.step()
                # #self.policy.dstb_optimizer.step()
                # self.policy.value_optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten().detach().cpu().numpy(), self.rollout_buffer.returns.flatten().detach().cpu().numpy())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        update_ego: bool = True,
        update_adversary: bool = True,
    ):
        #try:
        iteration = 0
        #from common.algorithms import Exploiter
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        self.callback = callback

        window = 250
        tolerance = .05 # movable
        rews = []

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            #perturbed_agent, other_ego, other_adv = self._create_perturbed_agent()
            print("perturbed agent created!", flush=True)
            #self._initialize_parallel_updater()      
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps) #TODO: This is sequential - remove when done.
            # perturbed_buf, perturbed_adv_buf = perturbed_agent.env_perturb_params() #TODO: This is a sequential original line, delete it when done.
            #continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps) #TODO: This is sequential - remove when done.

            # Run env_perturb_params and collect_rollouts in different threads (cannot be done in different processes because they contain unpickleable objects)
            # with ThreadPoolExecutor(max_workers=2) as executor:
            #     future_perturbed = executor.submit(perturbed_agent.env_perturb_params)
            #     future_collect = executor.submit(self.collect_rollouts, self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps)
                
            #     perturbed_buf, perturbed_adv_buf = future_perturbed.result()
            #     continue_training = future_collect.result()
            # self.perturbed_agent = perturbed_agent
            # self.perturbed_buf = perturbed_buf
            # self.perturbed_adv_buf = perturbed_adv_buf
            # self.perturbed_agent_policy = perturbed_agent.policy
            # print("main agent and perturbed agent rollout done!", flush=True)
            
            #if isinstance(self, Exploiter):
            #    if len(rews) > 2000:
            #        if (max(rews[-window:]) - min(rews[-window:])) <= tolerance * 2:
            #            continue_training = False
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    rews.append(safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    wandb.log({"eval_rew": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])})
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
        

            self.train()
            #self.perturbed_agent.env.close()
            #del self.perturbed_agent

        callback.on_training_end()
        
        # finally:
        #     #IMPORTANT! Persistent workers must be cleaned up.
        #     self.cleanup()
        #     torch.cuda.empty_cache()

        #except Exception as e:
        #    print(e)
        return self
    def dump_properties(self):
        
        PRIMITIVE_TYPES = (int, float, bool, str, type(None))
        
        primitive_attrs = {}

        for attr_name in dir(self):
            if attr_name.startswith('__'):
                continue
                
            try:
                attr_value = getattr(self, attr_name)
            except AttributeError:
                continue

            if callable(attr_value):
                continue

            if isinstance(attr_value, PRIMITIVE_TYPES):
                primitive_attrs[attr_name] = attr_value
                continue
            
            if isinstance(attr_value, (list, tuple)):
                if all(isinstance(item, PRIMITIVE_TYPES) for item in attr_value):
                    primitive_attrs[attr_name] = attr_value



        return primitive_attrs