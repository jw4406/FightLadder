import sys
import time
import random
from venv import create

import torch
import torch as th
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
from anyio import value
from gym import spaces
from copy import deepcopy
from collections import deque

from retro.examples.brute import rollout
from torch.nn import functional as F
from typing import Any, Dict, Mapping, Optional, Tuple, Union, Type, List, TypeVar

from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from stable_baselines3.common.policies import BasePolicy, ActorActorCriticCnnPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer
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
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn, update_learning_rate, is_vectorized_observation
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.vec_env import VecEnv

from .const import *
from .nash import compute_nash


SelfIPPO = TypeVar("SelfIPPO", bound="IPPO")
SelfLeaguePPO = TypeVar("SelfLeaguePPO", bound="LeaguePPO")
MAGICS_PPO = TypeVar("MAGICS_PPO", bound="MAGICS_PPO")

class IPPO(PPO):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
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
        update_left = True,
        update_right = True,
        other_learning_rate = None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.update_left = update_left
        self.update_right = update_right
        self.other_learning_rate = other_learning_rate

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer_other = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.other_lr_schedule = self.lr_schedule if self.other_learning_rate is None else get_schedule_fn(self.other_learning_rate)
        self.policy_other = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.other_lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy_other = self.policy_other.to(self.device)
    
    def _update_other_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        self.logger.record("train/other_learning_rate", self.other_lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.other_lr_schedule(self._current_progress_remaining))
    
    def _excluded_save_params(self) -> List[str]:
        return [
            "policy",
            "policy_other",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "rollout_buffer_other",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer", "policy_other", "policy_other.optimizer"]

        return state_dicts, []

    def set_parameters_2p(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        load_path_or_dict_other: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)
        params_other = None
        if isinstance(load_path_or_dict_other, dict):
            params_other = load_path_or_dict_other
        else:
            _, params_other, _ = load_from_zip_file(load_path_or_dict_other, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)
        
        for name in params_other:
            attr = None
            name_other = name.replace("policy", "policy_other")
            try:
                attr = recursive_getattr(self, name_other)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name_other} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params_other[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params_other[name], strict=exact_match)
            updated_objects.add(name_other)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.policy.predict(observation, state, episode_start, deterministic), self.policy_other.predict(observation, state, episode_start, deterministic)
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        rollout_buffer_other: RolloutBuffer,
        n_rollout_steps: int,
        policy = None,
        policy_other = None,
        coordinate_fn = None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        rollout_policy = self.policy if policy is None else policy
        rollout_policy_other = self.policy_other if policy_other is None else policy_other
        rollout_policy.set_training_mode(False)
        rollout_policy_other.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        rollout_buffer_other.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            rollout_policy.reset_noise(env.num_envs)
            rollout_policy_other.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                rollout_policy.reset_noise(env.num_envs)
                rollout_policy_other.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = rollout_policy(obs_tensor)
                actions_other, values_other, log_probs_other = rollout_policy_other(obs_tensor)
            actions = actions.cpu().numpy()
            actions_other = actions_other.cpu().numpy()

            # Rescale and perform action
            clipped_actions = np.hstack([actions, actions_other])
            print(clipped_actions, flush=True)
            print(np.shape(clipped_actions),flush=True)
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low, self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
                actions_other = actions_other.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and coordinate_fn is not None
                ):
                    coordinate_fn(infos[idx]["outcome"])
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    # print(f"[PPO] idx: {idx}, done: {done}, outcome: {infos[idx]['outcome']}", flush=True)
                    terminal_obs = rollout_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    terminal_obs_other = rollout_policy_other.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = rollout_policy.predict_values(terminal_obs)[0]
                        terminal_value_other = rollout_policy_other.predict_values(terminal_obs_other)[0]
                    rewards[idx] += self.gamma * terminal_value
                    rewards_other[idx] += self.gamma * terminal_value_other                        

            # from IPython import embed; embed()
            if self.update_left:
                rollout_buffer.add(self._last_obs.copy(), actions, rewards, self._last_episode_starts, values, log_probs)
            if self.update_right:
                rollout_buffer_other.add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts, values_other, log_probs_other)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = rollout_policy.predict_values(obs_as_tensor(new_obs, self.device))
            values_other = rollout_policy_other.predict_values(obs_as_tensor(new_obs, self.device))

        if self.update_left:
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if self.update_right:
            rollout_buffer_other.compute_returns_and_advantage(last_values=values_other, dones=dones)

        callback.on_rollout_end()

        return True
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.policy_other.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        self._update_other_learning_rate(self.policy_other.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        policies = [self.policy, self.policy_other]
        rollout_buffers = [self.rollout_buffer, self.rollout_buffer_other]
        suffixes = ["", "_other"]
        update_flags = [self.update_left, self.update_right]
        # policies = [self.policy_other, self.policy]
        # rollout_buffers = [self.rollout_buffer_other, self.rollout_buffer]
        # suffixes = ["_other", ""]
        # update_flags = [self.update_right, self.update_left]

        for policy, rollout_buffer, suffix, update_flag in zip(policies, rollout_buffers, suffixes, update_flags):
            if not update_flag:
                continue
            
            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []

            continue_training = True

            # train for n_epochs epochs
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                for rollout_data in rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        policy.reset_noise(self.batch_size)

                    values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
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
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                    policy.optimizer.step()

                if not continue_training:
                    break

            self._n_updates += self.n_epochs
            explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

            # Logs
            self.logger.record(f"train/entropy_loss{suffix}", np.mean(entropy_losses))
            self.logger.record(f"train/policy_gradient_loss{suffix}", np.mean(pg_losses))
            self.logger.record(f"train/value_loss{suffix}", np.mean(value_losses))
            self.logger.record(f"train/approx_kl{suffix}", np.mean(approx_kl_divs))
            self.logger.record(f"train/clip_fraction{suffix}", np.mean(clip_fractions))
            self.logger.record(f"train/loss{suffix}", loss.item())
            self.logger.record(f"train/explained_variance{suffix}", explained_var)
            if hasattr(policy, "log_std"):
                self.logger.record(f"train/std{suffix}", th.exp(policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfIPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "IPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfIPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps)

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
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_other_mean", safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self
    
    def async_learn(
        self: SelfIPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "IPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        fsp: bool = False, # NOTE: this method implements an approximate version of FSP, the full version is implemented in league.py
        max_fsp_num: int = 50,
        fsp_threshold: float = 0.3,
    ) -> SelfIPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps * 10, # Async learning is much slower
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        if fsp:
            left_state_dicts = [deepcopy(self.policy.state_dict())]
            right_state_dicts = [deepcopy(self.policy_other.state_dict())]
            tmp_left_policy = deepcopy(self.policy)
            tmp_right_policy = deepcopy(self.policy_other)

        while self.num_timesteps < total_timesteps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            
            self.update_left = True
            self.update_right = False
            rew_diff = 0
            while (rew_diff < fsp_threshold) and (self.num_timesteps < total_timesteps):
                rew_diff = 0
                for _ in range(10):
                    if fsp:
                        tmp_right_policy.load_state_dict(random.choice(right_state_dicts))
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps, policy_other=tmp_right_policy)
                    else:
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps)
                    if continue_training is False:
                        break
                    iteration += 1
                    # Display training infos
                    if log_interval is not None and iteration % log_interval == 0:
                        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                        self.logger.record("time/iterations", iteration, exclude="tensorboard")
                        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_rew_other_mean", safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("time/fps", fps)
                        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                        self.logger.dump(step=self.num_timesteps)
                    rew_diff = rew_diff + safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) - safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer])
                    self.train()
                rew_diff = rew_diff / 10
                if continue_training is False:
                    break
            print("[Left] rew_diff: ", rew_diff, flush=True)
            if continue_training is False:
                break
            if fsp:
                left_state_dicts.append(deepcopy(self.policy.state_dict()))
                if len(left_state_dicts) > max_fsp_num:
                    left_state_dicts.pop(random.randrange(len(left_state_dicts)))

            self.update_left = False
            self.update_right = True
            rew_diff = 0
            while (rew_diff < fsp_threshold) and (self.num_timesteps < total_timesteps):
                rew_diff = 0
                for _ in range(10):
                    if fsp:
                        tmp_left_policy.load_state_dict(random.choice(left_state_dicts))
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps, policy=tmp_left_policy)
                    else:
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps)
                    if continue_training is False:
                        break
                    iteration += 1
                    # Display training infos
                    if log_interval is not None and iteration % log_interval == 0:
                        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                        self.logger.record("time/iterations", iteration, exclude="tensorboard")
                        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_rew_other_mean", safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("time/fps", fps)
                        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                        self.logger.dump(step=self.num_timesteps)
                    rew_diff = rew_diff + safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]) - safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    self.train()
                rew_diff = rew_diff / 10
                if continue_training is False:
                    break
            print("[Right] rew_diff: ", rew_diff, flush=True)
            if continue_training is False:
                break
            if fsp:
                right_state_dicts.append(deepcopy(self.policy_other.state_dict()))
                if len(right_state_dicts) > max_fsp_num:
                    right_state_dicts.pop(random.randrange(len(right_state_dicts)))

        callback.on_training_end()

        return self


class BRIPPO(IPPO):

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        rollout_buffer_other: RolloutBuffer,
        n_rollout_steps: int,
        policy = None,
        policy_other = None,
        # coordinate_fn = None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        rollout_policy = self.policy if policy is None else policy
        rollout_policy_other = self.policy_other if policy_other is None else policy_other
        rollout_policy.set_training_mode(False)
        rollout_policy_other.set_training_mode(False)

        round_results = {'win': 0, 'lose': 0, 'draw': 0}
        round_start_steps = self.num_timesteps

        n_steps = 0
        rollout_buffer.reset()
        rollout_buffer_other.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            rollout_policy.reset_noise(env.num_envs)
            rollout_policy_other.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                rollout_policy.reset_noise(env.num_envs)
                rollout_policy_other.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = rollout_policy(obs_tensor)
                actions_other, values_other, log_probs_other = rollout_policy_other(obs_tensor)
            actions = actions.cpu().numpy()
            actions_other = actions_other.cpu().numpy()

            # Rescale and perform action
            clipped_actions = np.hstack([actions, actions_other])
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low, self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
                actions_other = actions_other.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    # and coordinate_fn is not None
                ):
                    round_results[infos[idx]["outcome"]] += 1
                    # coordinate_fn(infos[idx]["outcome"])
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    # print(f"[PPO] idx: {idx}, done: {done}, outcome: {infos[idx]['outcome']}", flush=True)
                    terminal_obs = rollout_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    terminal_obs_other = rollout_policy_other.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = rollout_policy.predict_values(terminal_obs)[0]
                        terminal_value_other = rollout_policy_other.predict_values(terminal_obs_other)[0]
                    rewards[idx] += self.gamma * terminal_value
                    rewards_other[idx] += self.gamma * terminal_value_other                        

            # from IPython import embed; embed()
            if self.update_left:
                rollout_buffer.add(self._last_obs.copy(), actions, rewards, self._last_episode_starts, values, log_probs)
            if self.update_right:
                rollout_buffer_other.add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts, values_other, log_probs_other)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = rollout_policy.predict_values(obs_as_tensor(new_obs, self.device))
            values_other = rollout_policy_other.predict_values(obs_as_tensor(new_obs, self.device))

        if self.update_left:
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if self.update_right:
            rollout_buffer_other.compute_returns_and_advantage(last_values=values_other, dones=dones)

        callback.on_rollout_end()

        round_end_steps = self.num_timesteps
        round_results['start_steps'] = round_start_steps
        round_results['end_steps'] = round_end_steps
        with open(os.path.join(self.tensorboard_log, "round_results.txt"), "a") as f:
            f.write(str(round_results) + "\n")

        return True


class LeaguePPO(IPPO):

    def __init__(
        self,
        side,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
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
        other_learning_rate = None,
    ):
        if side == "left":
            update_left = True
            update_right = False
        elif side == "right":
            update_left = False
            update_right = True
        else:
            raise ValueError("side should be 'left' or 'right'")
        self.side = side

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            update_left=update_left,
            update_right=update_right,
            other_learning_rate=other_learning_rate,
        )
    
    def train(self, rollout_buffer: RolloutBuffer) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        policy = self.policy if self.side == "left" else self.policy_other
        suffix = "" if self.side == "left" else "_other"
        # Switch to train mode (this affects batch norm / dropout)
        policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    policy.reset_noise(self.batch_size)

                values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
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
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

        # Logs
        self.logger.record(f"train/entropy_loss{suffix}", np.mean(entropy_losses))
        self.logger.record(f"train/policy_gradient_loss{suffix}", np.mean(pg_losses))
        self.logger.record(f"train/value_loss{suffix}", np.mean(value_losses))
        self.logger.record(f"train/approx_kl{suffix}", np.mean(approx_kl_divs))
        self.logger.record(f"train/clip_fraction{suffix}", np.mean(clip_fractions))
        self.logger.record(f"train/loss{suffix}", loss.item())
        self.logger.record(f"train/explained_variance{suffix}", explained_var)
        if hasattr(policy, "log_std"):
            self.logger.record(f"train/std{suffix}", th.exp(policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
    
    def learn(
        self: SelfLeaguePPO,
        total_timesteps: int,
        rollout_opponent_num: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "IPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        get_kwargs_fn = None,
    ) -> SelfLeaguePPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer
        all_rollouts = buffer_cls(
            self.n_steps * rollout_opponent_num,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        while self.num_timesteps < total_timesteps:
            
            all_rollouts.reset()

            for i in range(rollout_opponent_num):
                kwargs = get_kwargs_fn()

                # NOTE: reset env before each rollout to avoid cross-episodic interference among different opponents
                self._last_obs = self.env.reset()
                self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
                if self._vec_normalize_env is not None:
                    self._last_original_obs = self._vec_normalize_env.get_original_obs()

                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps, policy=kwargs.get("policy"), policy_other=kwargs.get("policy_other"), coordinate_fn=kwargs.get("coordinate_fn"))
                if continue_training is False:
                    break

                collected_rollouts = self.rollout_buffer if self.side == "left" else self.rollout_buffer_other
                assert collected_rollouts.full, "rollout buffer should be full"
                curr_pos = all_rollouts.pos
                next_pos = all_rollouts.pos + collected_rollouts.size()
                all_rollouts.observations[curr_pos:next_pos] = collected_rollouts.observations[:]
                all_rollouts.actions[curr_pos:next_pos] = collected_rollouts.actions[:]
                all_rollouts.rewards[curr_pos:next_pos] = collected_rollouts.rewards[:]
                all_rollouts.returns[curr_pos:next_pos] = collected_rollouts.returns[:]
                all_rollouts.episode_starts[curr_pos:next_pos] = collected_rollouts.episode_starts[:]
                all_rollouts.values[curr_pos:next_pos] = collected_rollouts.values[:]
                all_rollouts.log_probs[curr_pos:next_pos] = collected_rollouts.log_probs[:]
                all_rollouts.advantages[curr_pos:next_pos] = collected_rollouts.advantages[:]
                all_rollouts.pos = next_pos
                if all_rollouts.pos == all_rollouts.buffer_size:
                    all_rollouts.full = True

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
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_other_mean", safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train(all_rollouts)
            kwargs["sync_fn"]()

        callback.on_training_end()

        return self

    def get_steps(self) -> int:
        return self.num_timesteps
    
    def set_steps(self, steps: int) -> None:
        self.num_timesteps = steps
    
    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        self.policy.to("cpu")
        self.policy_other.to("cpu")
        params = super().get_parameters()
        self.policy.to(self.device)
        self.policy_other.to(self.device)
        return params


class MAGICS_PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "AACCnnPolicy": ActorActorCriticCnnPolicy
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
        update_left = True,
        update_right = True,
        dstb_action_space =None
    ):

        super().__init__(
            policy,
            env,
            learning_rate=v_learning_rate,
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
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.update_left = update_left
        self.dstb_ent_coef = dstb_ent_coef
        self.dstb_action_space = dstb_action_space
        self.update_right = update_right
        self.learning_rate = [v_learning_rate, c_learning_rate, d_learning_rate]
        self.learning_rate_decay_phase = [v_learning_rate_decay, c_learning_rate_decay, d_learning_rate_decay]
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
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.smart = True
        self.adversarial = True
        if _init_setup_model:
            self._setup_model()

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
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
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
                actions, log_probs, values, dstb_actions, dstb_log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            dstb_actions = dstb_actions.cpu().numpy()
            # Rescale and perform action
            clipped_actions = np.hstack([actions, dstb_actions])
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, _, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

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

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                dstb_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                dstb_log_probs
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        def hook_fn(grad):
            raise RuntimeError("GRADIENT MODIFIED")

        '''for p in self.policy.value_optimizer.param_groups[0]['params']:
            p.register_hook(hook_fn)'''
        continue_training = True

        # train for n_epochs epochs

        #self.rollout_buffer.advantages = buf.advantages
        buf = deepcopy(self.rollout_buffer)
        buf.values = torch.from_numpy(self.rollout_buffer.values).to(self.device)
        buf.rewards = torch.from_numpy(buf.rewards).to(self.device)
        buf.advantages = torch.from_numpy(buf.advantages).to(self.device)
        buf.episode_starts = torch.from_numpy(buf.episode_starts).to(self.device)
        for i in range(buf.buffer_size):
            _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
            buf.values[i] = values.squeeze()
        _, _, last_values, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
        buf.compute_returns_and_advantage_pt(last_values, torch.Tensor(buf.dones[-1]).to(self.device))
        rollout_advantages_copy = deepcopy(self.rollout_buffer.advantages)
        #buf.compute_returns_and_advantage_pt_test(last_values, torch.Tensor(buf.dones[-1]).to(self.device))
        self.rollout_buffer.advantages = buf.advantages
        #self.rollout_buffer.flat_advantages =
        env_indices = np.random.permutation(self.rollout_buffer.buffer_size * self.n_envs)
        #buffer = self.rollout_buffer.flatten()
        for epoch in range(self.n_epochs):
            #if epoch == 0:
            #    self.rollout_buffer.advantages = buf.advantages
            #else:
            #    self.rollout_buffer.advantages = buf.swap_and_flatten_pt(buf.advantages)
            #self.rollout_buffer.advantages = buf.swap_and_flatten_pt(buf.advantages)
            #torch.autograd.set_detect_anomaly(True)
            approx_kl_divs = []
            #self.rollout_buffer.advantages = buf.advantages
            # Do a complete pass on the rollout buffer
            count = 0
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                #start_idx = epoch * self.batch_size
                #selection = env_indices[start_idx: start_idx + self.batch_size]
                #rollout_data = buffer.sample(selection)
                #torch.autograd.set_detect_anomaly(True)
                #self.train_loop(rollout_data, clip_range, pg_losses, clip_fractions, None,value_losses,buf, entropy_losses,approx_kl_divs)

                #torch.autograd.set_detect_anomaly(True)
                self.normalize_advantage = False
                #torch.autograd.set_detect_anomaly(True)
                actions = torch.from_numpy(rollout_data.actions).to(self.device)
                dstb_actions = torch.from_numpy(rollout_data.dstb_actions).to(self.device)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                #traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[0:self.batch_size]].squeeze()
                #x0_states = self.rollout_buffer.X0_VALUES_MASTER[traj_ids]
                #x0_returns = buf.X0_RETURNS_MASTER[traj_ids]
                #x0_values, _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device),
                #                                                     torch.Tensor(actions[0]).to(self.device),
                #                                                     torch.Tensor(dstb_actions[0]).to(self.device))
                values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(torch.from_numpy(rollout_data.observations).to(self.device), actions, dstb_actions)
                #_,test = self.estimators(rollout_data.advantages, ctrl_log_prob, dstb_log_prob, x0_values.squeeze(), torch.Tensor(x0_returns).to(self.device))
                #d1f2_dstb_batched = autograd.grad(test, self.policy.value_optimizer.param_groups[0]['params'],
                #                                  create_graph=True, retain_graph=True)
                #d1f2_dstb = torch.hstack([u.flatten() for u in d1f2_dstb_batched])
                #autograd.grad(d1f2_dstb[0], self.policy.dstb_optimizer.param_groups[0]['params'])
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(ctrl_log_prob - torch.Tensor(rollout_data.old_log_prob).to(self.device))
                dstb_ratio = th.exp(dstb_log_prob - torch.Tensor(rollout_data.old_dstb_log_prob).to(self.device))

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                dstb_policy_loss_1 = advantages * dstb_ratio
                dstb_policy_loss_2 = advantages * th.clamp(dstb_ratio, 1 - clip_range, 1 + clip_range)
                dstb_policy_loss = th.min(dstb_policy_loss_1, dstb_policy_loss_2).mean()
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
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
                #_, _, values_pred, _, _ = self.policy(torch.Tensor(rollout_data.observations).to(self.device))
                value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values)
                value_losses.append(value_loss.item())
                L_ctrl_grad_batched = autograd.grad(value_loss, self.policy.value_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)
                L_ctrl_grad = torch.cat([t.flatten() for t in L_ctrl_grad_batched], dim=0)
                #L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])
                k = 5
                n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])

                rademacher = torch.bernoulli(torch.from_numpy(np.ones((n, k)) * .5)).to(self.device)
                rademacher[rademacher == 0] = -1
                # grad_batched = autograd.grad(L_ctrl_grad, flat_params, rademacher,0,1, is_grads_batched=True)
                grad_batched = autograd.grad(L_ctrl_grad, self.policy.value_optimizer.param_groups[0]['params'],
                                             torch.transpose(rademacher.to(self.device), 0, 1),
                                             is_grads_batched=True,
                                             retain_graph=True, create_graph=True)

                reshaped_grads = self.matrix_unbatch(grad_batched, k, size2=n).T
                reshaped_grads = reshaped_grads * rademacher
                L_ctrl_hessian = torch.mean(reshaped_grads, dim=1)
                L_ctrl_hessian = L_ctrl_hessian + 5

                d2f1_ctrl_batched = autograd.grad(policy_loss, self.policy.value_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)
                d2f1_dstb_batched = autograd.grad(dstb_policy_loss, self.policy.value_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)
                d2f1_ctrl = torch.hstack([t.flatten() for t in d2f1_ctrl_batched])

                d2f1_dstb = torch.hstack([t.flatten() for t in d2f1_dstb_batched])
                #d2f1_ctrl = torch.rand(d2f1_dstb.shape).to(self.device)



                # diag, no other option
                iHvp_ctrl = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_ctrl)
                iHvp_dstb = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_dstb)
                traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[count * self.batch_size: count * self.batch_size + self.batch_size]].squeeze()
                x0_states = self.rollout_buffer.X0_VALUES_MASTER[traj_ids]
                x0_returns = buf.X0_RETURNS_MASTER[traj_ids]
                x0_values,  _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device), torch.Tensor(actions[0]).to(self.device), torch.Tensor(dstb_actions[0]).to(self.device))




                # clipped surrogate loss

                '''policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()'''
                #autograd.grad(L_ctrl_grad, self.policy.ctrl_optimizer.param_groups[0]['params'], iHvp_ctrl, is_grads_batched=False, create_graph=True, retain_graph=True)

                #surr_L_ctrl = self.prep_grad_theta_L(advantages, ctrl_log_prob, x0_values.squeeze(), torch.tensor(x0_returns).to(self.device))
                '''values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                    torch.Tensor(rollout_data.observations).to(self.device), torch.Tensor(actions).to(self.device),
                    torch.Tensor(dstb_actions).to(self.device))
                x0_values, _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device),
                                                                     torch.Tensor(actions[0]).to(self.device),
                                                                     torch.Tensor(dstb_actions[0]).to(self.device))
                '''
                #surr_L_dstb = self.prep_grad_psi_L(advantages, dstb_log_prob, x0_values.squeeze(), torch.tensor(x0_returns).to(self.device))
                #d1f2_ctrl_batched = autograd.grad(surr_L, self.policy.ctrl_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)
                surr_L_ctrl, surr_L_dstb = self.estimators(advantages, ctrl_log_prob, dstb_log_prob, x0_values.squeeze(), torch.Tensor(x0_returns).to(self.device))
                d1f2_ctrl_batched = autograd.grad(surr_L_ctrl, self.policy.value_optimizer.param_groups[0]['params'],
                                                  create_graph=True, retain_graph=True)
                d1f2_ctrl = torch.hstack([t.flatten() for t in d1f2_ctrl_batched])
                d1f2_dstb_batched = autograd.grad(surr_L_dstb, self.policy.value_optimizer.param_groups[0]['params'],
                                                  create_graph=True, retain_graph=True)
                d1f2_dstb = torch.hstack([u.flatten() for u in d1f2_dstb_batched])
                #d1f2_dstb = d1f2_dstb.dot(dstb_log_prob)
                #ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.value_optimizer.param_groups[0]['params'], torch.eye(d1f2_ctrl.shape[0], device=self.device), is_grads_batched=True, create_graph=True, retain_graph=True)
                ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.ctrl_optimizer.param_groups[0]['params'], iHvp_ctrl, is_grads_batched=False, create_graph=True, retain_graph=True)
                dstb_imp = autograd.grad(d1f2_dstb, self.policy.dstb_optimizer.param_groups[0]['params'], iHvp_dstb, is_grads_batched=False, create_graph=True, retain_graph=True)

                # Entropy loss favor exploration
                if ctrl_entropy is None:
                    # Approximate entropy when no analytical form
                    ctrl_entropy_loss = -th.mean(-ctrl_log_prob)
                    dstb_entropy_loss = -th.mean(-dstb_log_prob)
                else:
                    ctrl_entropy_loss = -th.mean(ctrl_entropy)
                    dstb_entropy_loss = -th.mean(dstb_entropy)

                entropy_losses.append(ctrl_entropy_loss.item())
                #policy_loss_1 = advantages * ratio
                #policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                #policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                ctrl_loss = policy_loss + self.ent_coef * ctrl_entropy_loss
                dstb_loss = dstb_policy_loss + self.dstb_ent_coef * dstb_entropy_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = ctrl_log_prob.detach().cpu() - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                critic_loss = self.vf_coef * value_loss
                #big_loss = ctrl_loss + dstb_loss + critic_loss
                self.policy.ctrl_optimizer.zero_grad()
                self.policy.dstb_optimizer.zero_grad()
                self.policy.value_optimizer.zero_grad()
                ctrl_tensors = autograd.grad(ctrl_loss, self.policy.ctrl_optimizer.param_groups[0]['params'])
                dstb_tensors = autograd.grad(dstb_loss, self.policy.dstb_optimizer.param_groups[0]['params'])
                value_tensors = autograd.grad(critic_loss, self.policy.value_optimizer.param_groups[0]['params'])#, create_graph=True, retain_graph=True)

                for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                    self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad = ctrl_tensors[i]
                    self.policy.dstb_optimizer.param_groups[0]['params'][i].grad = dstb_tensors[i]
                th.nn.utils.clip_grad_norm_(self.policy.ctrl_optimizer.param_groups[0]['params'], self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(self.policy.dstb_optimizer.param_groups[0]['params'], self.max_grad_norm)
                for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                    self.policy.value_optimizer.param_groups[0]['params'][i].grad = value_tensors[i]
                th.nn.utils.clip_grad_norm_(self.policy.value_optimizer.param_groups[0]['params'], self.max_grad_norm)
                #big_loss.backward()
                print("e done")
                #ctrl_loss.backward(retain_graph=True)

                with (torch.no_grad()):
                    #ctrl_partials = autograd.grad(ctrl_loss, self.policy.ctrl_optimizer.param_groups[0]['params'])
                    for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                        self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad = \
                        self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad - ctrl_imp[i]
                    th.nn.utils.clip_grad_norm_(self.policy.ctrl_optimizer.param_groups[0]['params'], self.max_grad_norm)

                    for i in range(len(self.policy.dstb_optimizer.param_groups[0]['params'])):
                        self.policy.dstb_optimizer.param_groups[0]['params'][i].grad = \
                        self.policy.dstb_optimizer.param_groups[0]['params'][i].grad - dstb_imp[i]
                    th.nn.utils.clip_grad_norm_(self.policy.dstb_optimizer.param_groups[0]['params'], self.max_grad_norm)
                    th.nn.utils.clip_grad_norm_(self.policy.value_optimizer.param_groups[0]['params'], self.max_grad_norm)

                '''
                for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                    self.policy.ctrl_optimizer.param_groups[0]['params'][i] = self.policy.ctrl_optimizer.param_groups[0]['params'][i] - \
                                              self.policy.ctrl_optimizer.param_groups[0]['lr'] * self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad

                for i in range(len(self.policy.dstb_optimizer.param_groups[0]['params'])):
                    self.policy.dstb_optimizer.param_groups[0]['params'][i] = self.policy.dstb_optimizer.param_groups[0]['params'][i] - \
                                              self.policy.dstb_optimizer.param_groups[0]['lr'] * self.policy.dstb_optimizer.param_groups[0]['params'][i].grad

                for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                    self.policy.value_optimizer.param_groups[0]['params'][i] = self.policy.value_optimizer.param_groups[0]['params'][i] - \
                                              self.policy.value_optimizer.param_groups[0]['lr'] * self.policy.value_optimizer.param_groups[0]['params'][i].grad
                '''
                self.policy.ctrl_optimizer.step()
                self.policy.dstb_optimizer.step()
                self.policy.value_optimizer.step()
                '''
                ctrl_list, dstb_list, value_list = [], [], []
                with torch.no_grad():
                    count = 0
                    for param in self.policy.value_optimizer.param_groups[0]['params']:
                        param.copy_(self.policy.value_optimizer.param_groups[0]['params'][count])
                        count = count + 1  # Assign new random values
                        value_list.append(param)

                    # Reassign parameters to the optimizer (clear old state)
                    self.policy.value_optimizer.param_groups[0]['params'] = value_list
                    count = 0
                    for param in self.policy.ctrl_optimizer.param_groups[0]['params']:
                        param.copy_(self.policy.ctrl_optimizer.param_groups[0]['params'][count])
                        count = count + 1  # Assign new random values
                        ctrl_list.append(param)
                    # Reassign parameters to the optimizer (clear old state)
                    self.policy.ctrl_optimizer.param_groups[0]['params'] = ctrl_list
                    count = 0
                    for param in self.policy.dstb_optimizer.param_groups[0]['params']:
                        param.copy_(self.policy.dstb_optimizer.param_groups[0]['params'][count])
                        count = count + 1  # Assign new random values
                        dstb_list.append(param)
                    # Reassign parameters to the optimizer (clear old state)
                    self.policy.dstb_optimizer.param_groups[0]['params'] = dstb_list
                '''
                #self.policy.dstb_optimizer.zero_grad()
                #dstb_partials = autograd.grad(dstb_loss, self.policy.dstb_optimizer.param_groups[0]['params'])
                #dstb_loss.backward(retain_graph=True)
                #self.policy.dstb_optimizer.step()
                #values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                #    torch.from_numpy(rollout_data.observations).to(self.device), actions, dstb_actions)
                #values_pred = values.flatten()
                #value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred)
                #critic_loss = self.vf_coef * value_loss
                #self.policy.value_optimizer.zero_grad()
                '''
                critic_partials = autograd.grad(critic_loss, self.policy.value_optimizer.param_groups[0]['params'])
                for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                    self.policy.value_optimizer.param_groups[0]['params'][i].grad = critic_partials[i]'''
                #critic_loss.backward(retain_graph=True)

                #loss.backward()
                # Clip grad norm
                #self.policy.value_optimizer.step()
                '''
                with torch.no_grad():
                    for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                        self.policy.value_optimizer.param_groups[0]['params'][i] = torch.tensor(self.policy.value_optimizer.param_groups[0]['params'][i].data, requires_grad=True)
                    for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                        self.policy.ctrl_optimizer.param_groups[0]['params'][i] = torch.tensor(self.policy.ctrl_optimizer.param_groups[0]['params'][i].data, requires_grad=True)
                    for i in range(len(self.policy.dstb_optimizer.param_groups[0]['params'])):
                        self.policy.dstb_optimizer.param_groups[0]['params'][i] = torch.tensor(self.policy.dstb_optimizer.param_groups[0]['params'][i].data, requires_grad=True)
    
                    """self.ctrl_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.policy_net.parameters(), self.action_net.parameters()), joint_schedule[1](1),maximize=False)
                    self.dstb_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.dstb_net.parameters(), self.dstb_action_net.parameters()), joint_schedule[2](1), maximize=False)
                    self.value_optimizer = self.optimizer_class(
                        itertools.chain(self.mlp_extractor.value_net.parameters(), self.value_net.parameters()),
                        joint_schedule[0](1), **self.optimizer_kwargs)
                        """
                    evens = 0
                    for i in range(len(self.policy.mlp_extractor.value_net)):
                        if i % 2 == 1:
                            evens = evens + 2
                            continue
                        self.policy.mlp_extractor.value_net[evens].weight.data = self.policy.value_optimizer.param_groups[0]['params'][evens].data
                        self.policy.mlp_extractor.value_net[evens].bias.data = self.policy.value_optimizer.param_groups[0]['params'][evens + 1].data
                    #evens = 0
                    self.policy.value_net.bias.data = self.policy.value_optimizer.param_groups[0]['params'][-1].data
                    self.policy.value_net.weight.data = self.policy.value_optimizer.param_groups[0]['params'][-2].data
                '''


                '''
                del policy_loss
                del dstb_policy_loss
                del d2f1_ctrl_batched
                del d2f1_dstb_batched
                del policy_loss_1
                del policy_loss_2
                del dstb_policy_loss_1
                del dstb_policy_loss_2
                del L_ctrl_grad_batched
                del L_ctrl_grad
                del d2f1_ctrl
                del d2f1_dstb
                del d1f2_ctrl_batched
                del d1f2_dstb_batched
                del d1f2_ctrl
                del advantages
                del values, values_pred
                '''

                #buf = deepcopy(self.rollout_buffer)
                #buf.values = torch.from_numpy(self.rollout_buffer.values).to(self.device)
                #buf.rewards = torch.from_numpy(buf.rewards).to(self.device)
                #buf.advantages = torch.from_numpy(buf.advantages).to(self.device)
                #buf.episode_starts = torch.from_numpy(buf.episode_starts).to(self.device)

                #for i in range(buf.buffer_size):
                #    _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                #    buf.values[i] = values.squeeze()
                #_, _, last_values, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
                
                #TEST - DO NOT COMMIT
                advantage_test = []
                _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
                last_values = vf.flatten()
                last_gae_lam = th.zeros_like(last_values)
                dones = torch.Tensor(buf.dones[-1]).to(self.device)
                for step in reversed(range(buf.buffer_size)):
                    #_, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                    if step == buf.buffer_size - 1:
                        next_non_terminal = 1.0 - dones.float()
                        next_values = last_values
                    else:
                        next_non_terminal = 1.0 - buf.episode_starts[step + 1].float()
                        _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[step+1]).to(self.device))
                        next_values = vf.flatten()
                    _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))

                    delta = buf.rewards[step] + buf.gamma * next_values * next_non_terminal - value_query.squeeze()
                    last_gae_lam = delta + buf.gamma * buf.gae_lambda * next_non_terminal * last_gae_lam
                    advantage_test.append(last_gae_lam)
                    #buf.advantages[step] = last_gae_lam
                advantages = torch.stack(advantage_test, dim=0)
                #buf.returns = buf.advantages + buf.values
                print("")
                #TEST - DO NOT COMMIT
                
                #buf.compute_returns_and_advantage_pt(values, torch.Tensor(buf.dones[-1]).to(self.device))
                #self.rollout_buffer.advantages = torch.zeros_like(self.rollout_buffer.advantages)
                #self.rollout_buffer.flat_advantages = buf.swap_and_flatten(buf.advantages)
                self.rollout_buffer.advantages = self.rollout_buffer.swap_and_flatten_pt(advantages)
                #elf.rollout_buffer.flat_advantages = self.rollout_buffer.swap_and_flatten_pt(self.rollout_buffer.advantages)
                #self.rollout_buffer.flat_advantages = buf.swap_and_flatten_pt(buf.advantages)

                if not continue_training:
                    break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", ctrl_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: MAGICS_PPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> MAGICS_PPO:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def prep_grad_theta_L(self, advantages, ctrl_logp, x0_values, x0_returns):
        #TODO: self.rollout_buffer.X0_VALUES_MASTER + env_indices
        grad_estimator = 2 * (advantages * ctrl_logp).mean() * (x0_returns - x0_values).mean()
        return grad_estimator
    def prep_grad_psi_L(self, advantages, dstb_logp, x0_values, x0_returns):
        grad_estimator = 2 * (advantages * dstb_logp).mean() * (x0_returns - x0_values).mean()
        return grad_estimator
    def estimators(self, advantages, ctrl_logp, dstb_logp, x0_values, x0_returns):
        return 2 * (advantages * ctrl_logp).mean() * (x0_returns - x0_values).mean(), 2 * (advantages * dstb_logp).mean() * (x0_returns - x0_values).mean()
    def matrix_unbatch(self, to_be_unbatched, size1, size2=None):
        if size2 is None:
            size2 = size1
        unbatched = torch.zeros((size1, size2), device=self.device)
        for jac_row_count in range(size1):
            curr = 0
            for count in range(len(to_be_unbatched)):
                unbatched[jac_row_count,
                curr:curr + len(
                    torch.flatten(to_be_unbatched[count][jac_row_count, :]))] = torch.flatten(
                    to_be_unbatched[count][jac_row_count, :])
                curr = curr + len(torch.flatten(to_be_unbatched[count][jac_row_count, :]))
        return unbatched


    def train_loop(self, rollout_data, clip_range, pg_losses, clip_fractions, clip_range_vf,value_losses,buf, entropy_losses,approx_kl_divs):
        # torch.autograd.set_detect_anomaly(True)
        self.normalize_advantage = False
        # torch.autograd.set_detect_anomaly(True)
        actions = torch.from_numpy(rollout_data.actions).to(self.device)
        dstb_actions = torch.from_numpy(rollout_data.dstb_actions).to(self.device)
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)
        # traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[0:self.batch_size]].squeeze()
        # x0_states = self.rollout_buffer.X0_VALUES_MASTER[traj_ids]
        # x0_returns = buf.X0_RETURNS_MASTER[traj_ids]
        # x0_values, _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device),
        #                                                     torch.Tensor(actions[0]).to(self.device),
        #                                                     torch.Tensor(dstb_actions[0]).to(self.device))
        values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
            torch.from_numpy(rollout_data.observations).to(self.device), actions, dstb_actions)
        # _,test = self.estimators(rollout_data.advantages, ctrl_log_prob, dstb_log_prob, x0_values.squeeze(), torch.Tensor(x0_returns).to(self.device))
        # d1f2_dstb_batched = autograd.grad(test, self.policy.value_optimizer.param_groups[0]['params'],
        #                                  create_graph=True, retain_graph=True)
        # d1f2_dstb = torch.hstack([u.flatten() for u in d1f2_dstb_batched])
        # autograd.grad(d1f2_dstb[0], self.policy.dstb_optimizer.param_groups[0]['params'])
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(ctrl_log_prob - torch.Tensor(rollout_data.old_log_prob).to(self.device))
        dstb_ratio = th.exp(dstb_log_prob - torch.Tensor(rollout_data.old_dstb_log_prob).to(self.device))

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        dstb_policy_loss_1 = advantages * dstb_ratio
        dstb_policy_loss_2 = advantages * th.clamp(dstb_ratio, 1 - clip_range, 1 + clip_range)
        dstb_policy_loss = th.min(dstb_policy_loss_1, dstb_policy_loss_2).mean()
        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
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
        value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred.flatten())
        value_losses.append(value_loss.item())
        L_ctrl_grad_batched = autograd.grad(value_loss, self.policy.value_optimizer.param_groups[0]['params'],
                                            create_graph=True, retain_graph=True)
        L_ctrl_grad = torch.cat([t.flatten() for t in L_ctrl_grad_batched], dim=0)
        # L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])
        k = 30
        n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])

        rademacher = torch.bernoulli(torch.from_numpy(np.ones((n, k)) * .5)).to(self.device)
        rademacher[rademacher == 0] = -1
        # grad_batched = autograd.grad(L_ctrl_grad, flat_params, rademacher,0,1, is_grads_batched=True)
        grad_batched = autograd.grad(L_ctrl_grad, self.policy.value_optimizer.param_groups[0]['params'],
                                     torch.transpose(rademacher.to(self.device), 0, 1),
                                     is_grads_batched=True,
                                     retain_graph=False, create_graph=False)

        reshaped_grads = self.matrix_unbatch(grad_batched, k, size2=n).T
        reshaped_grads = reshaped_grads * rademacher
        L_ctrl_hessian = torch.mean(reshaped_grads, dim=1)
        L_ctrl_hessian = L_ctrl_hessian + 5

        d2f1_ctrl_batched = autograd.grad(policy_loss, self.policy.value_optimizer.param_groups[0]['params'],
                                          create_graph=True, retain_graph=True)
        d2f1_dstb_batched = autograd.grad(dstb_policy_loss, self.policy.value_optimizer.param_groups[0]['params'],
                                          create_graph=True, retain_graph=True)
        d2f1_ctrl = torch.hstack([t.flatten() for t in d2f1_ctrl_batched])

        d2f1_dstb = torch.hstack([t.flatten() for t in d2f1_dstb_batched])
        # d2f1_ctrl = torch.rand(d2f1_dstb.shape).to(self.device)

        # diag, no other option
        iHvp_ctrl = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_ctrl)
        iHvp_dstb = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_dstb)
        traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[0:self.batch_size]].squeeze()
        x0_states = self.rollout_buffer.X0_VALUES_MASTER[traj_ids]
        x0_returns = buf.X0_RETURNS_MASTER[traj_ids]
        x0_values, _, _, _, _ = self.policy.evaluate_actions(torch.Tensor(x0_states).to(self.device),
                                                             torch.Tensor(actions[0]).to(self.device),
                                                             torch.Tensor(dstb_actions[0]).to(self.device))

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
        surr_L_ctrl, surr_L_dstb = self.estimators(advantages, ctrl_log_prob, dstb_log_prob, x0_values.squeeze(),
                                                   torch.Tensor(x0_returns).to(self.device))
        d1f2_ctrl_batched = autograd.grad(surr_L_ctrl, self.policy.value_optimizer.param_groups[0]['params'],
                                          create_graph=True, retain_graph=True)
        d1f2_ctrl = torch.hstack([t.flatten() for t in d1f2_ctrl_batched])
        d1f2_dstb_batched = autograd.grad(surr_L_dstb, self.policy.value_optimizer.param_groups[0]['params'],
                                          create_graph=True, retain_graph=True)
        d1f2_dstb = torch.hstack([u.flatten() for u in d1f2_dstb_batched])
        # d1f2_dstb = d1f2_dstb.dot(dstb_log_prob)
        # ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.value_optimizer.param_groups[0]['params'], torch.eye(d1f2_ctrl.shape[0], device=self.device), is_grads_batched=True, create_graph=True, retain_graph=True)
        ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.ctrl_optimizer.param_groups[0]['params'], iHvp_ctrl,
                                 is_grads_batched=False, create_graph=True, retain_graph=True)
        dstb_imp = autograd.grad(d1f2_dstb, self.policy.dstb_optimizer.param_groups[0]['params'], iHvp_dstb,
                                 is_grads_batched=False, create_graph=True, retain_graph=True)

        # Entropy loss favor exploration
        if ctrl_entropy is None:
            # Approximate entropy when no analytical form
            ctrl_entropy_loss = -th.mean(-ctrl_log_prob)
            dstb_entropy_loss = -th.mean(-dstb_log_prob)
        else:
            ctrl_entropy_loss = -th.mean(ctrl_entropy)
            dstb_entropy_loss = -th.mean(dstb_entropy)

        entropy_losses.append(ctrl_entropy_loss.item())
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        ctrl_loss = policy_loss + self.ent_coef * ctrl_entropy_loss
        dstb_loss = dstb_policy_loss + self.dstb_ent_coef * dstb_entropy_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
            log_ratio = ctrl_log_prob.detach().cpu() - rollout_data.old_log_prob
            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
            continue_training = False
            if self.verbose >= 1:
                print("")
                #print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
            return

        # Optimization step
        self.policy.ctrl_optimizer.zero_grad()
        ctrl_loss.backward(retain_graph=True)
        # ctrl_partials = autograd.grad(ctrl_loss, self.policy.ctrl_optimizer.param_groups[0]['params'])
        for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
            self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad = \
            self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad - ctrl_imp[i]
        th.nn.utils.clip_grad_norm_(self.policy.ctrl_optimizer.param_groups[0]['params'], self.max_grad_norm)
        self.policy.ctrl_optimizer.step()

        self.policy.dstb_optimizer.zero_grad()
        # dstb_partials = autograd.grad(dstb_loss, self.policy.dstb_optimizer.param_groups[0]['params'])
        dstb_loss.backward(retain_graph=True)
        for i in range(len(self.policy.dstb_optimizer.param_groups[0]['params'])):
            self.policy.dstb_optimizer.param_groups[0]['params'][i].grad = \
            self.policy.dstb_optimizer.param_groups[0]['params'][i].grad - dstb_imp[i]
        th.nn.utils.clip_grad_norm_(self.policy.dstb_optimizer.param_groups[0]['params'], self.max_grad_norm)
        self.policy.dstb_optimizer.step()
        values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
            torch.from_numpy(rollout_data.observations).to(self.device), actions, dstb_actions)
        values_pred = values.flatten()
        value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred)
        critic_loss = self.vf_coef * value_loss
        self.policy.value_optimizer.zero_grad()
        '''
        critic_partials = autograd.grad(critic_loss, self.policy.value_optimizer.param_groups[0]['params'])
        for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
            self.policy.value_optimizer.param_groups[0]['params'][i].grad = critic_partials[i]'''
        critic_loss.backward(retain_graph=True)

        # loss.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.value_optimizer.param_groups[0]['params'], self.max_grad_norm)
        self.policy.value_optimizer.step()
        '''
        with torch.no_grad():
            for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                self.policy.value_optimizer.param_groups[0]['params'][i] = torch.tensor(self.policy.value_optimizer.param_groups[0]['params'][i].data, requires_grad=True)
            for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                self.policy.ctrl_optimizer.param_groups[0]['params'][i] = torch.tensor(self.policy.ctrl_optimizer.param_groups[0]['params'][i].data, requires_grad=True)
            for i in range(len(self.policy.dstb_optimizer.param_groups[0]['params'])):
                self.policy.dstb_optimizer.param_groups[0]['params'][i] = torch.tensor(self.policy.dstb_optimizer.param_groups[0]['params'][i].data, requires_grad=True)

            """self.ctrl_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.policy_net.parameters(), self.action_net.parameters()), joint_schedule[1](1),maximize=False)
self.dstb_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.dstb_net.parameters(), self.dstb_action_net.parameters()), joint_schedule[2](1), maximize=False)
self.value_optimizer = self.optimizer_class(
    itertools.chain(self.mlp_extractor.value_net.parameters(), self.value_net.parameters()),
    joint_schedule[0](1), **self.optimizer_kwargs)"""
            evens = 0
            for i in range(len(self.policy.mlp_extractor.value_net)):
                if i % 2 == 1:
                    evens = evens + 2
                    continue
                self.policy.mlp_extractor.value_net[evens].weight.data = self.policy.value_optimizer.param_groups[0]['params'][evens].data
                self.policy.mlp_extractor.value_net[evens].bias.data = self.policy.value_optimizer.param_groups[0]['params'][evens + 1].data
            #evens = 0
            self.policy.value_net.bias.data = self.policy.value_optimizer.param_groups[0]['params'][-1].data
            self.policy.value_net.weight.data = self.policy.value_optimizer.param_groups[0]['params'][-2].data
        '''

        del policy_loss
        del dstb_policy_loss
        del d2f1_ctrl_batched
        del d2f1_dstb_batched
        del policy_loss_1
        del policy_loss_2
        del dstb_policy_loss_1
        del dstb_policy_loss_2
        del L_ctrl_grad_batched
        del L_ctrl_grad
        del d2f1_ctrl
        del d2f1_dstb
        del d1f2_ctrl_batched
        del d1f2_dstb_batched
        del d1f2_ctrl
        del advantages
        del values


class RARL_PPO(MAGICS_PPO):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "AACCnnPolicy": ActorActorCriticCnnPolicy
    }

    def __init__(self,
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
        update_left = True,
        update_right = True,
        dstb_action_space =None
    ):

        super().__init__(
            policy,
            env,
            c_learning_rate= c_learning_rate,
            d_learning_rate= d_learning_rate,
            v_learning_rate= v_learning_rate,
            c_learning_rate_decay= c_learning_rate_decay,
            d_learning_rate_decay= d_learning_rate_decay,
            v_learning_rate_decay= v_learning_rate_decay,
            n_steps=n_steps,
            batch_size = batch_size,
            n_epochs = n_epochs,
            gamma = gamma,
            gae_lambda = gae_lambda,
            clip_range = clip_range,
            clip_range_vf= clip_range_vf,
            normalize_advantage= normalize_advantage,
            ent_coef= ent_coef,
            dstb_ent_coef = dstb_ent_coef,
            vf_coef = vf_coef,
            max_grad_norm = max_grad_norm,
            use_sde = use_sde,
            sde_sample_freq = sde_sample_freq,
            target_kl = target_kl,
            tensorboard_log = tensorboard_log,
            policy_kwargs = policy_kwargs,
            verbose= verbose,
            seed = seed,
            device = device,
            _init_setup_model = _init_setup_model,
            update_left = update_left,
            update_right = update_right,
            dstb_action_space = dstb_action_space,
        )
        self.update_ctrl = True
        self.update_dstb = False

        print("")

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """

        self._update_learning_rate(self.policy.ctrl_optimizer) if self.update_ctrl is True else self._update_learning_rate(self.policy.dstb_optimizer)
        self._update_learning_rate(self.policy.value_optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = torch.Tensor(rollout_data.actions).to(self.device)
                dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(self.device)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                if self.update_ctrl is True:
                    ratio = th.exp(ctrl_log_prob - torch.Tensor(rollout_data.old_log_prob).to(self.device))
                else:
                    ratio = th.exp(dstb_log_prob - torch.Tensor(rollout_data.old_dstb_log_prob).to(self.device))

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                if self.update_ctrl is True:
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                else:
                    policy_loss = th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
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

                # Entropy loss favor exploration
                if (ctrl_entropy is None) or (dstb_entropy is None):
                    # Approximate entropy when no analytical form
                    if self.update_ctrl is True:
                        entropy_loss = -th.mean(-ctrl_log_prob)
                    else:
                        entropy_loss = -th.mean(-dstb_log_prob)
                else:
                    if self.update_ctrl is True:
                        entropy_loss = -th.mean(ctrl_entropy)
                    else:
                        entropy_loss = -th.mean(dstb_entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    if self.update_ctrl is True:
                        log_ratio = ctrl_log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    else:
                        log_ratio = dstb_log_prob - rollout_data.old_dstb_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                if self.update_ctrl is True:
                    self.policy.ctrl_optimizer.zero_grad()
                else:
                    self.policy.dstb_optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if self.update_ctrl is True:
                    self.policy.ctrl_optimizer.step()
                else:
                    self.policy.dstb_optimizer.step()
                self.policy.value_optimizer.step()

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
        self.update_ctrl = not self.update_ctrl

class TSS_PPO(MAGICS_PPO):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "AACCnnPolicy": ActorActorCriticCnnPolicy
    }

    def __init__(self,
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
                 update_left=True,
                 update_right=True,
                 dstb_action_space=None
                 ):
        super().__init__(
            policy,
            env,
            c_learning_rate=c_learning_rate,
            d_learning_rate=d_learning_rate,
            v_learning_rate=v_learning_rate,
            c_learning_rate_decay=c_learning_rate_decay,
            d_learning_rate_decay=d_learning_rate_decay,
            v_learning_rate_decay=v_learning_rate_decay,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            dstb_ent_coef=dstb_ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            update_left=update_left,
            update_right=update_right,
            dstb_action_space=dstb_action_space,
        )

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """

        self._update_learning_rate([self.policy.ctrl_optimizer, self.policy.dstb_optimizer,self.policy.value_optimizer])
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = torch.Tensor(rollout_data.actions).to(self.device)
                dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(self.device)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                    torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                values = values.flatten()
                # Normalize advantage
                advantages = torch.from_numpy(rollout_data.advantages).to(self.device)
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

                # Entropy loss favor exploration
                if (ctrl_entropy is None) or (dstb_entropy is None):
                    # Approximate entropy when no analytical form
                    ctrl_entropy_loss = -th.mean(-ctrl_log_prob)
                    dstb_entropy_loss = -th.mean(-dstb_log_prob)
                else:
                    ctrl_entropy_loss = -th.mean(ctrl_entropy)
                    dstb_entropy_loss = -th.mean(dstb_entropy)

                entropy_losses.append(ctrl_entropy_loss.item())

                loss = ctrl_policy_loss + self.ent_coef * ctrl_entropy_loss + self.dstb_ent_coef * dstb_entropy_loss + self.vf_coef * value_loss + dstb_policy_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    ctrl_log_ratio = ctrl_log_prob - rollout_data.old_log_prob
                    ctrl_approx_kl_div = th.mean((th.exp(ctrl_log_ratio) - 1) - ctrl_log_ratio).cpu().numpy()
                    dstb_log_ratio = dstb_log_prob - rollout_data.old_dstb_log_prob
                    dstb_approx_kl_div = th.mean((th.exp(dstb_log_ratio) - 1) - dstb_log_ratio).cpu().numpy()
                    approx_kl_divs.append(ctrl_approx_kl_div)

                if self.target_kl is not None and torch.max(ctrl_approx_kl_div, dstb_approx_kl_div) > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.ctrl_optimizer.zero_grad()
                self.policy.dstb_optimizer.zero_grad()
                self.policy.value_optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.ctrl_optimizer.step()
                self.policy.dstb_optimizer.step()
                self.policy.value_optimizer.step()

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
        #self.update_ctrl = not self.update_ctrl
