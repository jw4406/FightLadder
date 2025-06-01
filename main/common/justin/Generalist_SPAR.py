import torch
import torch as th
import itertools
from torch import autograd
import sys
import time
import random
from venv import create
import wandb
import numpy as np
import torch.nn as nn
from anyio import value
from gym import spaces
from copy import deepcopy
from collections import deque
from functorch import vmap as eepy
from retro.examples.brute import rollout
from torch.nn import functional as F
from typing import Any, Dict, Mapping, Optional, Tuple, Union, Type, List, TypeVar

from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from stable_baselines3.common.policies import BasePolicy, ActorActorCriticCnnPolicy, ActorActorCriticCnnGeneralistPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn, \
    update_learning_rate, is_vectorized_observation
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, \
    save_to_zip_file
from stable_baselines3.common.vec_env import VecEnv
from .Doubly_TSS_SPAR import Doubly_TSS_SPAR

class Generalist_SPAR(Doubly_TSS_SPAR):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "AACCnnPolicy": ActorActorCriticCnnGeneralistPolicy
    }

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            c_learning_rate: Union[float, Schedule] = 1e-4,
            d_learning_rate: Union[float, Schedule] = 7e-4,
            v_learning_rate: Union[float, Schedule] = 7e-4,
            c_learning_rate_decay: Union[float, Schedule] = 1e-4,
            d_learning_rate_decay: Union[float, Schedule] = 7e-4,
            v_learning_rate_decay: Union[float, Schedule] = 7e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 1,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            dstb_ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            I_AM_LEFT=True,
            I_AM_RIGHT=False,
            dstb_action_space=None,
            num_adversary=4,
            n_global_env=None,
            n_env_per_adv=None,
            warmstarted_cont_MAGICS=False,
            opp_list=None,
            use_mirror=False
    ):
        assert I_AM_LEFT != I_AM_RIGHT
        super().__init__(
            policy,
            env,
            v_learning_rate=v_learning_rate,
            c_learning_rate=c_learning_rate,
            d_learning_rate=d_learning_rate,
            v_learning_rate_decay=v_learning_rate_decay,
            c_learning_rate_decay=c_learning_rate_decay,
            d_learning_rate_decay=d_learning_rate_decay,
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
            seed=seed,
            _init_setup_model=False,
            batch_size=batch_size,
            warmstarted_cont_MAGICS=warmstarted_cont_MAGICS
        )
        self.update_left = I_AM_LEFT
        self.dstb_ent_coef = dstb_ent_coef
        self.dstb_action_space = dstb_action_space
        self.update_right = I_AM_RIGHT
        self.n_epochs = n_epochs
        # self.learning_rate = [v_learning_rate, c_learning_rate, d_learning_rate]
        # self.learning_rate_decay_phase = [v_learning_rate_decay, c_learning_rate_decay, d_learning_rate_decay]
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        '''
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.smart = True
        self.adversarial = True
        '''
        self.n_global_env = n_global_env
        self.n_env_per_adv = n_env_per_adv
        self.learning_rate = [c_learning_rate, d_learning_rate, v_learning_rate]
        self.num_adversaries = num_adversary
        if _init_setup_model:
            self._setup_model()

        # at this point in the code, the Specialized_Agent's policy and value function are set up (we don't care about hte other one)
        # now we need to create the adversaries
        adversary_buffers = []
        self.env.num_envs = self.n_env_per_adv

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
                                       warmstarted_cont_MAGICS=self.warmstarted_cont_MAGICS
                                       )
            overwrite.rollout_buffer.n_envs = self.n_env_per_adv
            adversary_buffers.append(overwrite.rollout_buffer)
        self.adversary_buffers = adversary_buffers
        self.env.num_envs = self.n_envs
        print("created %d adversaries" % self.num_adversaries)
        #self.adversaries = adversaries
        # self._setup_learn(self._total_timesteps)
        self.vf_coef = 1
        self.use_mirror = use_mirror

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        # self._setup_learn()
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        for i in range(self.num_adversaries):
            self.adversary_buffers[i].reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                s_actions, s_log_probs, s_values, s_dstb_actions, s_dstb_log_probs = self.policy(obs_tensor, network_keys=[i for i in range(self.num_adversaries)])
                all_adv_left_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
                all_adv_right_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
                all_adv_critic_values = torch.zeros((self.n_global_env, 1), device=self.device)
                all_adv_log_probs = torch.zeros((self.n_global_env,), device=self.device)
                all_adv_dstb_log_probs = torch.zeros((self.n_global_env,), device=self.device)
                '''
                for i in range(self.num_adversaries):
                    actions, log_probs, values, dstb_actions, dstb_log_probs = self.adversaries[i].policy(obs_tensor)
                    # actions = actions.cpu()
                    # dstb_actions = dstb_actions.cpu()
                    chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
                    all_adv_left_actions[chunk] = actions[chunk]
                    all_adv_log_probs[chunk] = log_probs[chunk]
                    all_adv_critic_values[chunk] = values[chunk]
                    all_adv_right_actions[chunk] = dstb_actions[chunk]
                    all_adv_dstb_log_probs[chunk] = dstb_log_probs[chunk]
                

                if self.update_left is True:
                    # specialized agent is playing left
                    # all adversaries are playing right.
                    all_adv_left_actions = []
                    all_adv_log_probs = []
                    actions = s_actions
                    log_probs = s_log_probs
                    adversary_actions = all_adv_right_actions
                    adversary_log_probs = all_adv_dstb_log_probs
                else:
                    all_adv_right_actions = []
                    all_adv_dstb_log_probs = []
                    actions = s_dstb_actions
                    log_probs = s_dstb_log_probs
                    adversary_actions = all_adv_left_actions
                    adversary_log_probs = all_adv_log_probs
                '''
            actions = s_actions
            adversary_actions = s_dstb_actions
            log_probs = s_log_probs
            adversary_log_probs = s_dstb_log_probs
            actions = actions.cpu().numpy()
            adversary_actions = adversary_actions.cpu().numpy()

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

                prot_left = actions[0::2, :]  # actions for the prot when he is on the left
                prot_reversed = actions[1::2,
                                :]  # actions for prot when he is on the right... but its backwards right now!

                adv_right = adversary_actions[0::2, :]
                adv_reversed = adversary_actions[1::2, :]

                temp = np.zeros((self.num_adversaries, self.action_space.shape[0]))
                temp = prot_reversed

                actions[1::2, :] = adversary_actions[1::2, :]
                adversary_actions[1::2, :] = temp

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
            # assert np.allclose(rewards + rew_other, np.zeros(rewards.shape))
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

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            if self.use_mirror is True:
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    mirror_master_copy_actions,
                    mirror_master_copy_adv_actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    all_adv_critic_values.squeeze(),
                    log_probs,
                    adversary_log_probs
                )
            else:
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    adversary_actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    all_adv_critic_values.squeeze(),
                    log_probs,
                    adversary_log_probs
                )

            for i in range(self.num_adversaries):
                chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
                if self.use_mirror is True:
                    self.adversaries[i].rollout_buffer.add(
                        self._last_obs[chunk],
                        mirror_master_copy_actions[chunk],
                        mirror_master_copy_adv_actions[chunk],
                        rewards[chunk],
                        self._last_episode_starts[chunk],
                        all_adv_critic_values[chunk],
                        log_probs[chunk],
                        adversary_log_probs[chunk]  # not done
                    )
                else:
                    self.adversary_buffers[i].add(
                        self._last_obs[chunk],
                        actions[chunk],
                        adversary_actions[chunk],
                        rew_other[chunk],
                        self._last_episode_starts[chunk],
                        all_adv_critic_values[chunk],
                        log_probs[chunk],
                        adversary_log_probs[chunk]  # not done
                    )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            #values = torch.zeros((self.n_global_env,))
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            #for i in range(self.num_adversaries):
            #    chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
            #    values[chunk] = self.policy.predict_values(obs_as_tensor(new_obs, self.device))[
            #        chunk].to('cpu')
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        for i in range(self.num_adversaries):  # is this a bug?
            chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
            self.adversary_buffers[i].compute_returns_and_advantage(last_values=values[chunk],
                                                                             dones=dones[chunk])

        callback.on_rollout_end()

        return True

    def train(self):
        # train the special agent and the adversaries

        # main agent needs its own training routine.
        # adversaries can just call their own methods

        # main agent
        # need to query adversary critics

        assert self.update_left != self.update_right
        self.policy.num_adversaries = self.num_adversaries
        self.train_ma()
        #self.policy.num_adversaries = 1
        self.train_advs()
        #for i in range(self.num_adversaries):
        #    self.adversaries[i].train_one_adversary(self.policy, ma_left=self.update_left, ma_right=self.update_right),
        # adversaries
        # test(self)
        '''for i in range(self.num_adversaries):
            self.adversaries[i].train_one_adversary(self.policy, ma_left=self.update_left, ma_right=self.update_right)'''

        return

    def train_ma(self):
        # helper function

        """
        Update policy using the currently gathered rollout buffer.
        """

        '''
        if self.warmstarted_cont_MAGICS is True:
            if self.warmstarted_cont_MAGICS is True:
                print("this model is warmstarted! now running magics_ppo training", flush=True)
            return super().train()
        '''
        self.warmstarted_cont_MAGICS = False
        self._update_learning_rate(
            [self.policy.ctrl_optimizer, self.policy.dstb_optimizer, self.policy.value_optimizer])
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        if self.warmstarted_cont_MAGICS is True:
            buf = deepcopy(self.rollout_buffer)
            buf.values = torch.from_numpy(self.rollout_buffer.values).to(self.device)
            buf.rewards = torch.from_numpy(buf.rewards).to(self.device)
            buf.advantages = torch.from_numpy(buf.advantages).to(self.device)
            buf.episode_starts = torch.from_numpy(buf.episode_starts).to(self.device)
            for i in range(buf.buffer_size):
                # location = np.nonzero(rollout_data.env_indices == i)
                #adversary_id = buf.env_indices[i] // self.n_env_per_adv
                #for j in range(self.num_adversaries):
                #    _, _, values, _, _ = self.policy(
                #        torch.Tensor(buf.observations[i][adversary_id == j]).to(self.device))
                    # _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                #    buf.values[i][adversary_id == j] = values.squeeze()
                _, _, buf.values[i], _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device), network_keys=[i for i in range(self.num_adversaries)])
            # _, _, last_values, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
            buf.compute_returns_and_advantage_pt(buf.values[i], torch.Tensor(buf.dones[-1]).to(self.device))
            rollout_advantages_copy = deepcopy(self.rollout_buffer.advantages)
            # buf.compute_returns_and_advantage_pt_test(last_values, torch.Tensor(buf.dones[-1]).to(self.device))
            self.rollout_buffer.advantages = buf.advantages
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            count = 0
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                start = time.time()
                actions = torch.Tensor(rollout_data.actions).to(self.device)
                dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(self.device)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                if self.update_left is True:
                    # main player is the left player
                    # adversaries are "dstb" role
                    #values = torch.zeros((self.batch_size, 1), device=self.device)
                    #dstb_log_prob = torch.zeros((self.batch_size,), device=self.device)
                    #dstb_entropy = torch.zeros((self.batch_size,), device=self.device)

                    '''
                    for i in range(self.n_global_env):
                        location = np.nonzero(rollout_data.env_indices == i)
                        adversary_id = i // self.n_env_per_adv
                        temp_values, _, _, temp_dstb_log_prob, temp_dstb_entropy = self.adversaries[
                            adversary_id].policy.evaluate_actions(
                            torch.Tensor(rollout_data.observations[location]).to(self.device), actions[location],
                            dstb_actions[location])
                        values[location] = temp_values  #
                        dstb_log_prob[location] = temp_dstb_log_prob
                        dstb_entropy[location] = temp_dstb_entropy
                    _, ctrl_log_prob, ctrl_entropy, _, _ = self.policy.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                    '''
                    self.policy.num_global_env = self.n_global_env
                    self.policy.num_adv = self.num_adversaries
                    values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions, shuffle_keys=rollout_data.env_indices, network_keys=[i for i in range(self.num_adversaries)])

                    values = values.flatten()
                else:
                    assert self.update_right is True
                    # main player is the right player
                    # adversaries are the control role
                    values = torch.zeros((self.batch_size, 1), device=self.device)
                    ctrl_log_prob = torch.zeros((self.batch_size,), device=self.device)
                    ctrl_entropy = torch.zeros((self.batch_size,), device=self.device)
                    for i in range(self.n_global_env):
                        location = np.nonzero(rollout_data.env_indices == i)
                        adversary_id = i // self.n_env_per_adv
                        temp_values, temp_ctrl_log_prob, temp_ctrl_entropy, _, _ = self.adversaries[
                            adversary_id].policy.evaluate_actions(
                            torch.Tensor(rollout_data.observations[location]).to(self.device), actions[location],
                            dstb_actions[location])
                        values[location] = temp_values  #
                        ctrl_log_prob[location] = temp_ctrl_log_prob
                        ctrl_entropy[location] = temp_ctrl_entropy
                    _, _, _, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                    values = values.flatten()
                # Normalize advantage
                if type(rollout_data.advantages) is np.ndarray:
                    advantages = torch.from_numpy(rollout_data.advantages).to(self.device)
                else:
                    advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ctrl_ratio = th.exp(ctrl_log_prob - torch.Tensor(rollout_data.old_log_prob).to(self.device))
                dstb_ratio = th.exp(dstb_log_prob - torch.Tensor(rollout_data.old_dstb_log_prob).to(self.device))

                # clipped surrogate loss
                policy_loss_1 = advantages * ctrl_ratio
                policy_loss_2 = advantages * th.clamp(ctrl_ratio, 1 - clip_range, 1 + clip_range)
                #dstb_policy_loss_1 = advantages * dstb_ratio
                #dstb_policy_loss_2 = advantages * th.clamp(dstb_ratio, 1 - clip_range, 1 + clip_range)
                ctrl_policy_loss = th.min(policy_loss_1, policy_loss_2).mean()
                #dstb_policy_loss = th.min(dstb_policy_loss_1, dstb_policy_loss_2).mean()

                # Logging
                pg_losses.append(ctrl_policy_loss.item())
                clip_fraction = th.mean((th.abs(ctrl_ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred)
                value_losses.append(value_loss.item())
                all_adv_val_params = self.policy.value_optimizer.param_groups[0]['params'][-self.num_adversaries*2:]
                if self.warmstarted_cont_MAGICS is True:
                    L_ctrl_grad_batched = autograd.grad(value_loss, all_adv_val_params,
                                                        create_graph=True, retain_graph=True)
                    L_ctrl_grad = torch.cat([t.flatten() for t in L_ctrl_grad_batched], dim=0)
                    # L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])
                    full_hessian = True
                    n = sum(p.numel() for p in all_adv_val_params)
                    if full_hessian is False:

                        k = 50
                        # n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])

                        rademacher = torch.bernoulli(torch.from_numpy(np.ones((n, k)) * .5)).to(self.device)
                        rademacher[rademacher == 0] = -1
                        # grad_batched = autograd.grad(L_ctrl_grad, flat_params, rademacher,0,1, is_grads_batched=True)
                        grad_batched = autograd.grad(L_ctrl_grad, all_adv_val_params,
                                                     torch.transpose(rademacher.to(self.device), 0, 1),
                                                     is_grads_batched=True,
                                                     retain_graph=True, create_graph=True)

                    else:
                        grad_batched = autograd.grad(L_ctrl_grad, all_adv_val_params,
                                                     torch.eye(n).to(self.device),
                                                     is_grads_batched=True,
                                                     retain_graph=True, create_graph=True)
                    if full_hessian is False:
                        reshaped_grads = self.matrix_unbatch(grad_batched, k, size2=n).T
                        reshaped_grads = reshaped_grads * rademacher
                        L_ctrl_hessian = torch.mean(reshaped_grads, dim=1)
                        L_ctrl_hessian = L_ctrl_hessian + 10
                    else:
                        L_ctrl_hessian = self.matrix_unbatch(grad_batched, n)
                        L_ctrl_hessian.diag().add_(10)
                    d2f1_ctrl_batched = autograd.grad(ctrl_policy_loss, all_adv_val_params,
                                                      create_graph=True, retain_graph=True)
                    # d2f1_dstb_batched = autograd.grad(dstb_policy_loss,
                    #                                  self.policy.value_optimizer.param_groups[0]['params'],
                    #                                  create_graph=True, retain_graph=True)
                    d2f1_ctrl = torch.hstack([t.flatten() for t in d2f1_ctrl_batched])

                    # d2f1_dstb = torch.hstack([t.flatten() for t in d2f1_dstb_batched])
                    # d2f1_ctrl = torch.rand(d2f1_dstb.shape).to(self.device)

                    # diag, no other option
                    #iHvp_ctrl = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_ctrl)

                    iHvp_ctrl = torch.linalg.solve(L_ctrl_hessian, d2f1_ctrl)

                    # iHvp_dstb = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_dstb)
                    # assert not np.any(self.rollout_buffer.current_shot - self.rollout_buffer.indices[
                    #                                                     count * self.batch_size: count * self.batch_size + self.batch_size])
                    # assert self.rollout_buffer.current_shot == self.rollout_buffer.indices[count * self.batch_size: count * self.batch_size + self.batch_size]
                    traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[
                                                               count * self.batch_size: count * self.batch_size + self.batch_size]].squeeze()
                    assert (np.max(traj_ids - rollout_data.env_indices) == 0) and (np.min(traj_ids - rollout_data.env_indices) == 0)
                    x0_states = self.rollout_buffer.X0_VALUES_MASTER[traj_ids]
                    x0_returns = buf.X0_RETURNS_MASTER[traj_ids]
                    x0_values, _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device),
                                                                         torch.Tensor(actions).to(self.device),
                                                                         torch.Tensor(dstb_actions).to(self.device), shuffle_keys=traj_ids, network_keys=[i for i in range(self.num_adversaries)])

                    # clipped surrogate loss

                    '''policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()'''
                    # autograd.grad(L_ctrl_grad, self.policy.ctrl_optimizer.param_groups[0]['params'], iHvp_ctrl, is_grads_batched=False, create_graph=True, retain_graph=True)

                    # surr_L_ctrl = self.prep_grad_theta_L(advantages, ctrl_log_prob, x0_values.squeeze(), torch.tensor(x0_returns).to(self.device))
                    '''values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(self.device), torch.Tensor(actions).to(self.device),
                        torch.Tensor(dstb_actions).to(self.device))
                    x0_values, _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device),
                                                                         torch.Tensor(actions[0]).to(self.device),
                                                                         torch.Tensor(dstb_actions[0]).to(self.device))
                    '''
                    # surr_L_dstb = self.prep_grad_psi_L(advantages, dstb_log_prob, x0_values.squeeze(), torch.tensor(x0_returns).to(self.device))
                    # d1f2_ctrl_batched = autograd.grad(surr_L, self.policy.ctrl_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)
                    surr_L_ctrl, surr_L_dstb = self.estimators(advantages, ctrl_log_prob, dstb_log_prob,
                                                               x0_values.squeeze(),
                                                               torch.Tensor(x0_returns).to(self.device))
                    d1f2_ctrl_batched = autograd.grad(surr_L_ctrl, all_adv_val_params,
                                                      create_graph=True, retain_graph=True)
                    d1f2_ctrl = torch.hstack([t.flatten() for t in d1f2_ctrl_batched])
                    # d1f2_dstb_batched = autograd.grad(surr_L_dstb, self.policy.value_optimizer.param_groups[0]['params'],
                    #                                  create_graph=True, retain_graph=True)
                    # d1f2_dstb = torch.hstack([u.flatten() for u in d1f2_dstb_batched])
                    # d1f2_dstb = d1f2_dstb.dot(dstb_log_prob)
                    # ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.value_optimizer.param_groups[0]['params'], torch.eye(d1f2_ctrl.shape[0], device=self.device), is_grads_batched=True, create_graph=True, retain_graph=True)
                    ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.ctrl_optimizer.param_groups[0]['params'], iHvp_ctrl,
                                             is_grads_batched=False, create_graph=True, retain_graph=True)
                    # dstb_imp = autograd.grad(d1f2_dstb, self.policy.dstb_optimizer.param_groups[0]['params'], iHvp_dstb,
                    #                         is_grads_batched=False, create_graph=True, retain_graph=True)

                # Entropy loss favor exploration
                if (ctrl_entropy is None) or (dstb_entropy is None):
                    # Approximate entropy when no analytical form
                    ctrl_entropy_loss = -th.mean(-ctrl_log_prob)
                    dstb_entropy_loss = -th.mean(-dstb_log_prob)
                else:
                    ctrl_entropy_loss = -th.mean(ctrl_entropy)
                    dstb_entropy_loss = -th.mean(dstb_entropy)

                entropy_losses.append(ctrl_entropy_loss.item())

                loss = ctrl_policy_loss# + self.ent_coef * ctrl_entropy_loss + self.vf_coef * value_loss# + dstb_policy_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    ctrl_log_ratio = ctrl_log_prob - torch.from_numpy(rollout_data.old_log_prob).to(self.device)
                    ctrl_approx_kl_div = th.mean((th.exp(ctrl_log_ratio) - 1) - ctrl_log_ratio).cpu().numpy()
                    dstb_log_ratio = dstb_log_prob - torch.from_numpy(rollout_data.old_dstb_log_prob).to(self.device)
                    dstb_approx_kl_div = th.mean((th.exp(dstb_log_ratio) - 1) - dstb_log_ratio).cpu().numpy()
                    approx_kl_divs.append(ctrl_approx_kl_div)

                if self.target_kl is not None and torch.max(ctrl_approx_kl_div,
                                                            dstb_approx_kl_div) > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.ctrl_optimizer.zero_grad()
                self.policy.dstb_optimizer.zero_grad()
                self.policy.value_optimizer.zero_grad()
                loss.backward()
                if self.warmstarted_cont_MAGICS is True:
                    for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                        self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad = \
                            self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad - ctrl_imp[i]
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.ctrl_optimizer.step()
                #self.policy.dstb_optimizer.step()
                self.policy.value_optimizer.step()
                if self.warmstarted_cont_MAGICS is True:
                    advantage_test = []
                    #vf = torch.zeros_like(buf.values[-1])
                    adversary_id = buf.env_indices[-1] // self.n_env_per_adv
                    #for j in range(self.num_adversaries):
                    #    _, _, values, _, _ = self.policy(
                    #        torch.Tensor(buf.observations[-1][adversary_id == j]).to(self.device))
                    #    # _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                    #    vf[adversary_id == j] = values.squeeze()

                    _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device), network_keys=[i for i in range(self.num_adversaries)])
                    last_values = vf.flatten()
                    last_gae_lam = th.zeros_like(last_values)
                    dones = torch.Tensor(buf.dones[-1]).to(self.device)
                    for step in reversed(range(buf.buffer_size)):
                        next_values = torch.zeros_like(buf.values[-1])
                        # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                        if step == buf.buffer_size - 1:
                            next_non_terminal = 1.0 - dones.float()
                            next_values = last_values
                        else:
                            next_non_terminal = 1.0 - buf.episode_starts[step + 1].float()
                            #adversary_id = buf.env_indices[-1] // self.n_env_per_adv
                            #for j in range(self.num_adversaries):
                            #    _, _, temp_values, _, _ = self.policy(
                            #        torch.Tensor(buf.observations[step + 1][adversary_id == j]).to(self.device))
                                # _, _, temp_values, _, _ = self.policy(torch.Tensor(buf.observations[step + 1]).to(self.device))
                            #    next_values[adversary_id == j] = temp_values.flatten()
                            _, _, next_values, _, _ = self.policy(
                                torch.Tensor(buf.observations[step + 1]).to(self.device), network_keys=[i for i in range(self.num_adversaries)])
                        #value_query = torch.zeros_like(buf.values[-1])
                        # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                        adversary_id = buf.env_indices[step] // self.n_env_per_adv
                        #for j in range(self.num_adversaries):
                        #    _, _, temp_values, _, _ = self.policy(
                        #        torch.Tensor(buf.observations[step][adversary_id == j]).to(self.device))
                        #    value_query[adversary_id == j] = temp_values.squeeze()
                        _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device), network_keys=[i for i in range(self.num_adversaries)])

                        delta = buf.rewards[step] + buf.gamma * next_values * next_non_terminal - value_query.squeeze()
                        last_gae_lam = delta + buf.gamma * buf.gae_lambda * next_non_terminal * last_gae_lam
                        advantage_test.append(last_gae_lam)
                        # buf.advantages[step] = last_gae_lam
                    advantages = torch.stack(advantage_test, dim=0)
                    # buf.returns = buf.advantages + buf.values
                    end = time.time()
                    print("batch complete, elapsed = %f" % (start - end))
                    # TEST - DO NOT COMMIT

                    # buf.compute_returns_and_advantage_pt(values, torch.Tensor(buf.dones[-1]).to(self.device))
                    # self.rollout_buffer.advantages = torch.zeros_like(self.rollout_buffer.advantages)
                    # self.rollout_buffer.flat_advantages = buf.swap_and_flatten(buf.advantages)
                    self.rollout_buffer.advantages = self.rollout_buffer.swap_and_flatten_pt(advantages)
                    count = count + 1

                if not continue_training:
                    break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record(f"train/entropy_loss", np.mean(entropy_losses))
        self.logger.record(f"train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record(f"train/value_loss", np.mean(value_losses))
        self.logger.record(f"train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record(f"train/clip_fraction", np.mean(clip_fractions))
        self.logger.record(f"train/loss", loss.item())
        self.logger.record(f"train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(f"train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def train_advs(self):
        # helper function

        """
        Update policy using the currently gathered rollout buffer.
        """

        '''
        if self.warmstarted_cont_MAGICS is True:
            if self.warmstarted_cont_MAGICS is True:
                print("this model is warmstarted! now running magics_ppo training", flush=True)
            return super().train()
        '''
        ego_buffer = self.rollout_buffer
        for k in range(self.num_adversaries):
            self.rollout_buffer = self.adversary_buffers[k]
            self.warmstarted_cont_MAGICS = False
            self._update_learning_rate(
                [self.policy.ctrl_optimizer, self.policy.dstb_optimizer, self.policy.value_optimizer])
            # Compute current clip range
            clip_range = self.clip_range(self._current_progress_remaining)
            # Optional: clip range for the value function
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []

            continue_training = True
            if self.warmstarted_cont_MAGICS is True:
                buf = deepcopy(self.rollout_buffer)
                buf.values = torch.from_numpy(self.rollout_buffer.values).to(self.device)
                buf.rewards = torch.from_numpy(buf.rewards).to(self.device)
                buf.advantages = torch.from_numpy(buf.advantages).to(self.device)
                buf.episode_starts = torch.from_numpy(buf.episode_starts).to(self.device)
                for i in range(buf.buffer_size):
                    # location = np.nonzero(rollout_data.env_indices == i)
                    #adversary_id = buf.env_indices[i] // self.n_env_per_adv
                    #for j in range(self.num_adversaries):
                    #    _, _, values, _, _ = self.policy(
                    #        torch.Tensor(buf.observations[i][adversary_id == j]).to(self.device))
                        # _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                    #    buf.values[i][adversary_id == j] = values.squeeze()
                    _, _, buf.values[i], _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device), network_keys=[k])
                # _, _, last_values, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
                buf.compute_returns_and_advantage_pt(buf.values[i], torch.Tensor(buf.dones[-1]).to(self.device))
                rollout_advantages_copy = deepcopy(self.rollout_buffer.advantages)
                # buf.compute_returns_and_advantage_pt_test(last_values, torch.Tensor(buf.dones[-1]).to(self.device))
                self.rollout_buffer.advantages = buf.advantages
            # train for n_epochs epochs
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                count = 0
                # Do a complete pass on the rollout buffer
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    start = time.time()
                    actions = torch.Tensor(rollout_data.actions).to(self.device)
                    dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(self.device)
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    if self.update_left is True:
                        # main player is the left player
                        # adversaries are "dstb" role
                        #values = torch.zeros((self.batch_size, 1), device=self.device)
                        #dstb_log_prob = torch.zeros((self.batch_size,), device=self.device)
                        #dstb_entropy = torch.zeros((self.batch_size,), device=self.device)

                        '''
                        for i in range(self.n_global_env):
                            location = np.nonzero(rollout_data.env_indices == i)
                            adversary_id = i // self.n_env_per_adv
                            temp_values, _, _, temp_dstb_log_prob, temp_dstb_entropy = self.adversaries[
                                adversary_id].policy.evaluate_actions(
                                torch.Tensor(rollout_data.observations[location]).to(self.device), actions[location],
                                dstb_actions[location])
                            values[location] = temp_values  #
                            dstb_log_prob[location] = temp_dstb_log_prob
                            dstb_entropy[location] = temp_dstb_entropy
                        _, ctrl_log_prob, ctrl_entropy, _, _ = self.policy.evaluate_actions(
                            torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                        '''
                        self.policy.num_global_env = self.n_env_per_adv
                        self.policy.num_adv = 1
                        values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions, shuffle_keys=rollout_data.env_indices, network_keys=[k])

                        values = values.flatten()
                    else:
                        assert self.update_right is True
                        # main player is the right player
                        # adversaries are the control role
                        values = torch.zeros((self.batch_size, 1), device=self.device)
                        ctrl_log_prob = torch.zeros((self.batch_size,), device=self.device)
                        ctrl_entropy = torch.zeros((self.batch_size,), device=self.device)
                        for i in range(self.n_global_env):
                            location = np.nonzero(rollout_data.env_indices == i)
                            adversary_id = i // self.n_env_per_adv
                            temp_values, temp_ctrl_log_prob, temp_ctrl_entropy, _, _ = self.adversaries[
                                adversary_id].policy.evaluate_actions(
                                torch.Tensor(rollout_data.observations[location]).to(self.device), actions[location],
                                dstb_actions[location])
                            values[location] = temp_values  #
                            ctrl_log_prob[location] = temp_ctrl_log_prob
                            ctrl_entropy[location] = temp_ctrl_entropy
                        _, _, _, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                            torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                        values = values.flatten()
                    # Normalize advantage
                    if type(rollout_data.advantages) is np.ndarray:
                        advantages = torch.from_numpy(rollout_data.advantages).to(self.device)
                    else:
                        advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ctrl_ratio = th.exp(ctrl_log_prob - torch.Tensor(rollout_data.old_log_prob).to(self.device))
                    dstb_ratio = th.exp(dstb_log_prob - torch.Tensor(rollout_data.old_dstb_log_prob).to(self.device))

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ctrl_ratio
                    policy_loss_2 = advantages * th.clamp(ctrl_ratio, 1 - clip_range, 1 + clip_range)
                    dstb_policy_loss_1 = advantages * dstb_ratio
                    dstb_policy_loss_2 = advantages * th.clamp(dstb_ratio, 1 - clip_range, 1 + clip_range)
                    ctrl_policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    dstb_policy_loss = th.min(dstb_policy_loss_1, dstb_policy_loss_2).mean()

                    # Logging
                    pg_losses.append(ctrl_policy_loss.item())
                    clip_fraction = th.mean((th.abs(ctrl_ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred)
                    value_losses.append(value_loss.item())
                    all_adv_val_params = list(self.policy.value_net[k].parameters())
                    this_dstb_params = list(itertools.chain(list(self.policy.mlp_extractor.dstb_net.parameters()), self.policy.dstb_action_net[k].parameters()))
                    if self.warmstarted_cont_MAGICS is True:
                        L_dstb_grad_batched = autograd.grad(value_loss, all_adv_val_params,
                                                            create_graph=True, retain_graph=True)
                        L_dstb_grad = torch.cat([t.flatten() for t in L_dstb_grad_batched], dim=0)
                        # L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])
                        full_hessian = True
                        n = sum(p.numel() for p in all_adv_val_params)
                        if full_hessian is False:

                            k = 50
                            # n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])

                            rademacher = torch.bernoulli(torch.from_numpy(np.ones((n, k)) * .5)).to(self.device)
                            rademacher[rademacher == 0] = -1
                            # grad_batched = autograd.grad(L_ctrl_grad, flat_params, rademacher,0,1, is_grads_batched=True)
                            grad_batched = autograd.grad(L_dstb_grad, all_adv_val_params,
                                                         torch.transpose(rademacher.to(self.device), 0, 1),
                                                         is_grads_batched=True,
                                                         retain_graph=True, create_graph=True)

                        else:
                            grad_batched = autograd.grad(L_dstb_grad, all_adv_val_params,
                                                         torch.eye(n).to(self.device),
                                                         is_grads_batched=True,
                                                         retain_graph=True, create_graph=True)
                        if full_hessian is False:
                            reshaped_grads = self.matrix_unbatch(grad_batched, k, size2=n).T
                            reshaped_grads = reshaped_grads * rademacher
                            L_ctrl_hessian = torch.mean(reshaped_grads, dim=1)
                            L_ctrl_hessian = L_ctrl_hessian + 10
                        else:
                            L_dstb_hessian = self.matrix_unbatch(grad_batched, n)
                            #L_dstb_hessian.diag().add_(10)
                        d2f1_dstb_batched = autograd.grad(dstb_policy_loss, all_adv_val_params,
                                                          create_graph=True, retain_graph=True)
                        # d2f1_dstb_batched = autograd.grad(dstb_policy_loss,
                        #                                  self.policy.value_optimizer.param_groups[0]['params'],
                        #                                  create_graph=True, retain_graph=True)
                        d2f1_dstb = torch.hstack([t.flatten() for t in d2f1_dstb_batched])

                        # d2f1_dstb = torch.hstack([t.flatten() for t in d2f1_dstb_batched])
                        # d2f1_ctrl = torch.rand(d2f1_dstb.shape).to(self.device)

                        # diag, no other option
                        #iHvp_ctrl = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_ctrl)

                        iHvp_dstb = torch.linalg.solve(L_dstb_hessian, d2f1_dstb)

                        # iHvp_dstb = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_dstb)
                        # assert not np.any(self.rollout_buffer.current_shot - self.rollout_buffer.indices[
                        #                                                     count * self.batch_size: count * self.batch_size + self.batch_size])
                        # assert self.rollout_buffer.current_shot == self.rollout_buffer.indices[count * self.batch_size: count * self.batch_size + self.batch_size]
                        traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[
                                                                   count * self.batch_size: count * self.batch_size + self.batch_size]].squeeze()
                        assert (np.max(traj_ids - rollout_data.env_indices) == 0) and (np.min(traj_ids - rollout_data.env_indices) == 0)
                        x0_states = self.rollout_buffer.X0_VALUES_MASTER[traj_ids]
                        x0_returns = buf.X0_RETURNS_MASTER[traj_ids]
                        x0_values, _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device),
                                                                             torch.Tensor(actions).to(self.device),
                                                                             torch.Tensor(dstb_actions).to(self.device), shuffle_keys=traj_ids, network_keys=[k])

                        # clipped surrogate loss

                        '''policy_loss_1 = advantages * ratio
                        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()'''
                        # autograd.grad(L_ctrl_grad, self.policy.ctrl_optimizer.param_groups[0]['params'], iHvp_ctrl, is_grads_batched=False, create_graph=True, retain_graph=True)

                        # surr_L_ctrl = self.prep_grad_theta_L(advantages, ctrl_log_prob, x0_values.squeeze(), torch.tensor(x0_returns).to(self.device))
                        '''values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                            torch.Tensor(rollout_data.observations).to(self.device), torch.Tensor(actions).to(self.device),
                            torch.Tensor(dstb_actions).to(self.device))
                        x0_values, _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device),
                                                                             torch.Tensor(actions[0]).to(self.device),
                                                                             torch.Tensor(dstb_actions[0]).to(self.device))
                        '''
                        # surr_L_dstb = self.prep_grad_psi_L(advantages, dstb_log_prob, x0_values.squeeze(), torch.tensor(x0_returns).to(self.device))
                        # d1f2_ctrl_batched = autograd.grad(surr_L, self.policy.ctrl_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)
                        surr_L_ctrl, surr_L_dstb = self.estimators(advantages, ctrl_log_prob, dstb_log_prob,
                                                                   x0_values.squeeze(),
                                                                   torch.Tensor(x0_returns).to(self.device))
                        d1f2_dstb_batched = autograd.grad(surr_L_dstb, all_adv_val_params,
                                                          create_graph=True, retain_graph=True)
                        d1f2_dstb = torch.hstack([t.flatten() for t in d1f2_dstb_batched])
                        # d1f2_dstb_batched = autograd.grad(surr_L_dstb, self.policy.value_optimizer.param_groups[0]['params'],
                        #                                  create_graph=True, retain_graph=True)
                        # d1f2_dstb = torch.hstack([u.flatten() for u in d1f2_dstb_batched])
                        # d1f2_dstb = d1f2_dstb.dot(dstb_log_prob)
                        # ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.value_optimizer.param_groups[0]['params'], torch.eye(d1f2_ctrl.shape[0], device=self.device), is_grads_batched=True, create_graph=True, retain_graph=True)
                        dstb_imp = autograd.grad(d1f2_dstb, this_dstb_params, iHvp_dstb,
                                                 is_grads_batched=False, create_graph=True, retain_graph=True)
                        # dstb_imp = autograd.grad(d1f2_dstb, self.policy.dstb_optimizer.param_groups[0]['params'], iHvp_dstb,
                        #                         is_grads_batched=False, create_graph=True, retain_graph=True)

                    # Entropy loss favor exploration
                    if (ctrl_entropy is None) or (dstb_entropy is None):
                        # Approximate entropy when no analytical form
                        ctrl_entropy_loss = -th.mean(-ctrl_log_prob)
                        dstb_entropy_loss = -th.mean(-dstb_log_prob)
                    else:
                        ctrl_entropy_loss = -th.mean(ctrl_entropy)
                        dstb_entropy_loss = -th.mean(dstb_entropy)

                    entropy_losses.append(ctrl_entropy_loss.item())

                    loss = self.vf_coef * value_loss - dstb_policy_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        ctrl_log_ratio = ctrl_log_prob - torch.from_numpy(rollout_data.old_log_prob).to(self.device)
                        ctrl_approx_kl_div = th.mean((th.exp(ctrl_log_ratio) - 1) - ctrl_log_ratio).cpu().numpy()
                        dstb_log_ratio = dstb_log_prob - torch.from_numpy(rollout_data.old_dstb_log_prob).to(self.device)
                        dstb_approx_kl_div = th.mean((th.exp(dstb_log_ratio) - 1) - dstb_log_ratio).cpu().numpy()
                        approx_kl_divs.append(ctrl_approx_kl_div)

                    if self.target_kl is not None and torch.max(ctrl_approx_kl_div,
                                                                dstb_approx_kl_div) > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    self.policy.ctrl_optimizer.zero_grad()
                    self.policy.dstb_optimizer.zero_grad()
                    self.policy.value_optimizer.zero_grad()
                    loss.backward()
                    if self.warmstarted_cont_MAGICS is True:
                        for i in range(len(this_dstb_params)):
                            this_dstb_params[i].grad = this_dstb_params[i].grad + dstb_imp[i]
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    #self.policy.ctrl_optimizer.step()
                    self.policy.dstb_optimizer.step()
                    self.policy.value_optimizer.step()
                    if self.warmstarted_cont_MAGICS is True:
                        advantage_test = []
                        #vf = torch.zeros_like(buf.values[-1])
                        adversary_id = buf.env_indices[-1] // self.n_env_per_adv
                        #for j in range(self.num_adversaries):
                        #    _, _, values, _, _ = self.policy(
                        #        torch.Tensor(buf.observations[-1][adversary_id == j]).to(self.device))
                        #    # _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                        #    vf[adversary_id == j] = values.squeeze()

                        _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device), network_keys=[k])
                        last_values = vf.flatten()
                        last_gae_lam = th.zeros_like(last_values)
                        dones = torch.Tensor(buf.dones[-1]).to(self.device)
                        for step in reversed(range(buf.buffer_size)):
                            next_values = torch.zeros_like(buf.values[-1])
                            # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                            if step == buf.buffer_size - 1:
                                next_non_terminal = 1.0 - dones.float()
                                next_values = last_values
                            else:
                                next_non_terminal = 1.0 - buf.episode_starts[step + 1].float()
                                #adversary_id = buf.env_indices[-1] // self.n_env_per_adv
                                #for j in range(self.num_adversaries):
                                #    _, _, temp_values, _, _ = self.policy(
                                #        torch.Tensor(buf.observations[step + 1][adversary_id == j]).to(self.device))
                                    # _, _, temp_values, _, _ = self.policy(torch.Tensor(buf.observations[step + 1]).to(self.device))
                                #    next_values[adversary_id == j] = temp_values.flatten()
                                _, _, next_values, _, _ = self.policy(
                                    torch.Tensor(buf.observations[step + 1]).to(self.device), network_keys=[k])
                            #value_query = torch.zeros_like(buf.values[-1])
                            # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                            adversary_id = buf.env_indices[step] // self.n_env_per_adv
                            #for j in range(self.num_adversaries):
                            #    _, _, temp_values, _, _ = self.policy(
                            #        torch.Tensor(buf.observations[step][adversary_id == j]).to(self.device))
                            #    value_query[adversary_id == j] = temp_values.squeeze()
                            _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device), network_keys=[k])

                            delta = buf.rewards[step] + buf.gamma * next_values * next_non_terminal - value_query.squeeze()
                            last_gae_lam = delta + buf.gamma * buf.gae_lambda * next_non_terminal * last_gae_lam
                            advantage_test.append(last_gae_lam)
                            # buf.advantages[step] = last_gae_lam
                        advantages = torch.stack(advantage_test, dim=0)
                        # buf.returns = buf.advantages + buf.values
                        end = time.time()
                        print("batch complete, elapsed = %f" % (start - end))
                        # TEST - DO NOT COMMIT

                        # buf.compute_returns_and_advantage_pt(values, torch.Tensor(buf.dones[-1]).to(self.device))
                        # self.rollout_buffer.advantages = torch.zeros_like(self.rollout_buffer.advantages)
                        # self.rollout_buffer.flat_advantages = buf.swap_and_flatten(buf.advantages)
                        self.rollout_buffer.advantages = self.rollout_buffer.swap_and_flatten_pt(advantages)
                        count = count + 1

                    if not continue_training:
                        break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.rollout_buffer = ego_buffer

        # Logs
        self.logger.record(f"train/entropy_loss", np.mean(entropy_losses))
        self.logger.record(f"train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record(f"train/value_loss", np.mean(value_losses))
        self.logger.record(f"train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record(f"train/clip_fraction", np.mean(clip_fractions))
        self.logger.record(f"train/loss", loss.item())
        self.logger.record(f"train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(f"train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def predict(self, obs, env_index, deterministic=False):
        if self.use_mirror is True:
            # when mirror is true, ego is fighting ego
            # we need to query the policy twice

            (ego_action, state), (right_action, _) = self.policy.predict(obs, deterministic=deterministic)
            left_action = ego_action
        else:
            (left_action, state), (right_action, _) = self.policy.predict(obs, deterministic=deterministic, network_keys=[env_index])
            #(_, _), (right_action, _) = self.adversaries[env_index].predict(obs, deterministic=deterministic)
        return (left_action, state), (right_action, state)