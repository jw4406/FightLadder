import sys
import time
import random
from venv import create
import wandb
import torch
import torch as th
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
from anyio import value
from gym import spaces
from copy import deepcopy
from collections import deque
from torch import vmap as eepy
from retro.examples.brute import rollout
from torch.nn import functional as F
from typing import Any, Dict, Mapping, Optional, Tuple, Union, Type, List, TypeVar

from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from stable_baselines3.common.policies import BasePolicy, ActorActorCriticCnnPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy, ActorActorCriticCnnGeneralistPolicy, IPPOActorCriticCnnGeneralistPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from common.justin.Generalist_SPAR import Generalist_SPAR
from common.justin.derivative_free_spar import Derivative_Free_SPAR
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn, \
    update_learning_rate, is_vectorized_observation
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, \
    save_to_zip_file
from stable_baselines3.common.vec_env import VecEnv

from .const import *
from .nash import compute_nash

import itertools

SelfIPPO = TypeVar("SelfIPPO", bound="IPPO")
SelfLeaguePPO = TypeVar("SelfLeaguePPO", bound="LeaguePPO")
MAGICS_PPO = TypeVar("MAGICS_PPO", bound="MAGICS_PPO")
SelfMultiHeadLeaguePPO = TypeVar("SelfMultiHeadLeaguePPO", bound="MultiHeadLeaguePPO")


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
            update_left=True,
            update_right=True,
            other_learning_rate=None,
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
        self.adversarial = False
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
        self.other_lr_schedule = self.lr_schedule if self.other_learning_rate is None else get_schedule_fn(
            self.other_learning_rate)
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
        return self.policy.predict(observation, state, episode_start, deterministic), self.policy_other.predict(
            observation, state, episode_start, deterministic)

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            rollout_buffer_other: RolloutBuffer,
            n_rollout_steps: int,
            policy=None,
            policy_other=None,
            coordinate_fn=None,
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
            # print(clipped_actions, flush=True)
            # print(np.shape(clipped_actions),flush=True)
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low,
                                          self.action_space.high)

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
                rollout_buffer.add(self._last_obs.copy(), actions, rewards, self._last_episode_starts, values,
                                   log_probs)
            if self.update_right:
                rollout_buffer_other.add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts,
                                         values_other, log_probs_other)
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

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      self.rollout_buffer_other, n_rollout_steps=self.n_steps)

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
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_other_mean",
                                       safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
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
            fsp: bool = False,
            # NOTE: this method implements an approximate version of FSP, the full version is implemented in league.py
            max_fsp_num: int = 50,
            fsp_threshold: float = 0.3,
    ) -> SelfIPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps * 10,  # Async learning is much slower
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
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                                  self.rollout_buffer_other,
                                                                  n_rollout_steps=self.n_steps,
                                                                  policy_other=tmp_right_policy)
                    else:
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                                  self.rollout_buffer_other,
                                                                  n_rollout_steps=self.n_steps)
                    if continue_training is False:
                        break
                    iteration += 1
                    # Display training infos
                    if log_interval is not None and iteration % log_interval == 0:
                        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                        self.logger.record("time/iterations", iteration, exclude="tensorboard")
                        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                            self.logger.record("rollout/ep_rew_mean",
                                               safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_rew_other_mean",
                                               safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_len_mean",
                                               safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("time/fps", fps)
                        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                        self.logger.dump(step=self.num_timesteps)
                    rew_diff = rew_diff + safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) - safe_mean(
                        [ep_info["ro"] for ep_info in self.ep_info_buffer])
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
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                                  self.rollout_buffer_other,
                                                                  n_rollout_steps=self.n_steps, policy=tmp_left_policy)
                    else:
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                                  self.rollout_buffer_other,
                                                                  n_rollout_steps=self.n_steps)
                    if continue_training is False:
                        break
                    iteration += 1
                    # Display training infos
                    if log_interval is not None and iteration % log_interval == 0:
                        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                        self.logger.record("time/iterations", iteration, exclude="tensorboard")
                        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                            self.logger.record("rollout/ep_rew_mean",
                                               safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_rew_other_mean",
                                               safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_len_mean",
                                               safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("time/fps", fps)
                        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                        self.logger.dump(step=self.num_timesteps)
                    rew_diff = rew_diff + safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]) - safe_mean(
                        [ep_info["r"] for ep_info in self.ep_info_buffer])
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
            policy=None,
            policy_other=None,
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
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low,
                                          self.action_space.high)

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
                rollout_buffer.add(self._last_obs.copy(), actions, rewards, self._last_episode_starts, values,
                                   log_probs)
            if self.update_right:
                rollout_buffer_other.add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts,
                                         values_other, log_probs_other)
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
            other_learning_rate=None,
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
        self.current_opponent = None
        self.constructor_fn = None
        self.constructor_args = None        
        self.current_opponent = None
        self.constructor_fn = None
        self.constructor_args = None        

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
            get_kwargs_fn=None,
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

                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                          self.rollout_buffer_other, n_rollout_steps=self.n_steps,
                                                          policy=kwargs.get("policy"),
                                                          policy_other=kwargs.get("policy_other"),
                                                          coordinate_fn=kwargs.get("coordinate_fn"))
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
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_other_mean",
                                       safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
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

    def set_opponent_character(self, opponent_character: str) -> None:
        """Recreate environment with new opponent character if needed."""
        if hasattr(self, 'current_opponent') and self.current_opponent == opponent_character:
            return
        
        self.current_opponent = opponent_character
        # Close current environment
        self.env.close()
        # Recreate with new opponent
        new_agent = self.constructor_fn(self.constructor_args, self.side, opponent=opponent_character, single_env=False)
        self.env = new_agent.env
        # Reset episode tracking
        self._last_obs = None
        self._last_episode_starts = None

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
            update_left=True,
            update_right=True,
            dstb_action_space=None
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
        self.learning_rate = [c_learning_rate, d_learning_rate, v_learning_rate]
        self.learning_rate_decay_phase = [c_learning_rate_decay, d_learning_rate_decay, v_learning_rate_decay]
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

            new_obs, rewards, rew_other, dones, infos = env.step(clipped_actions)
            # assert np.allclose(rewards + rew_other, np.zeros(rewards.shape))
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

        def hook_fn(grad):
            raise RuntimeError("GRADIENT MODIFIED")

        '''for p in self.policy.value_optimizer.param_groups[0]['params']:
            p.register_hook(hook_fn)'''
        continue_training = True

        # train for n_epochs epochs

        # self.rollout_buffer.advantages = buf.advantages

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
        # buf.compute_returns_and_advantage_pt_test(last_values, torch.Tensor(buf.dones[-1]).to(self.device))
        self.rollout_buffer.advantages = buf.advantages
        '''
        buf = deepcopy(self.rollout_buffer)
        buf.rewards = torch.from_numpy(buf.rewards).to(self.device)
        buf.episode_starts = torch.from_numpy(buf.episode_starts).to(self.device)
        advantage_test = []
        _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
        last_values = vf.flatten()
        last_gae_lam = th.zeros_like(last_values)
        dones = torch.Tensor(buf.dones[-1]).to(self.device)
        for step in reversed(range(buf.buffer_size)):
            # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
            if step == buf.buffer_size - 1:
                next_non_terminal = 1.0 - dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - buf.episode_starts[step + 1].float()
                _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[step + 1]).to(self.device))
                next_values = vf.flatten()
            _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))

            delta = buf.rewards[step] + buf.gamma * next_values * next_non_terminal - value_query.squeeze()
            last_gae_lam = delta + buf.gamma * buf.gae_lambda * next_non_terminal * last_gae_lam
            advantage_test.append(last_gae_lam)
            # buf.advantages[step] = last_gae_lam
        advantages = torch.stack(advantage_test, dim=0)
        # buf.returns = buf.advantages + buf.values
        print("")
        self.rollout_buffer.advantages = advantages
        '''
        # TEST - DO NOT COMMIT

        # buf.compute_returns_and_advantage_pt(values, torch.Tensor(buf.dones[-1]).to(self.device))
        # self.rollout_buffer.advantages = torch.zeros_like(self.rollout_buffer.advantages)
        # self.rollout_buffer.flat_advantages = buf.swap_and_flatten(buf.advantages)
        env_indices = np.random.permutation(self.rollout_buffer.buffer_size * self.n_envs)
        # buffer = self.rollout_buffer.flatten()
        for epoch in range(self.n_epochs):
            # if epoch == 0:
            #    self.rollout_buffer.advantages = buf.advantages
            # else:
            #    self.rollout_buffer.advantages = buf.swap_and_flatten_pt(buf.advantages)
            # self.rollout_buffer.advantages = buf.swap_and_flatten_pt(buf.advantages)
            # torch.autograd.set_detect_anomaly(True)
            approx_kl_divs = []
            # self.rollout_buffer.advantages = buf.advantages
            # Do a complete pass on the rollout buffer
            count = 0
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # start_idx = epoch * self.batch_size
                # selection = env_indices[start_idx: start_idx + self.batch_size]
                # rollout_data = buffer.sample(selection)
                # torch.autograd.set_detect_anomaly(True)
                # self.train_loop(rollout_data, clip_range, pg_losses, clip_fractions, None,value_losses,buf, entropy_losses,approx_kl_divs)

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
                # _, _, values_pred, _, _ = self.policy(torch.Tensor(rollout_data.observations).to(self.device))
                value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values)
                value_losses.append(value_loss.item())
                L_ctrl_grad_batched = autograd.grad(value_loss, self.policy.value_optimizer.param_groups[0]['params'],
                                                    create_graph=True, retain_graph=True)
                L_ctrl_grad = torch.cat([t.flatten() for t in L_ctrl_grad_batched], dim=0)
                # L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])
                full_hessian = False
                n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])
                if full_hessian is False:

                    k = 50
                    # n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])

                    rademacher = torch.bernoulli(torch.from_numpy(np.ones((n, k)) * .5)).bfloat16().to(self.device)
                    rademacher[rademacher == 0] = -1
                    # grad_batched = autograd.grad(L_ctrl_grad, flat_params, rademacher,0,1, is_grads_batched=True)
                    grad_batched = autograd.grad(L_ctrl_grad, self.policy.value_optimizer.param_groups[0]['params'],
                                                 torch.transpose(rademacher.to(self.device), 0, 1),
                                                 is_grads_batched=True,
                                                 retain_graph=True, create_graph=True)

                else:
                    # import torch

                    def compute_jacobian_batched(x, params, batch_size=128):
                        N = x.shape[0]
                        param_vec_len = sum(p.numel() for p in params)

                        jacobian = torch.zeros(N, param_vec_len, device=x.device, dtype=x.dtype)

                        for start in range(0, N, batch_size):
                            end = min(start + batch_size, N)
                            batch_indices = torch.arange(start, end, device=x.device)

                            batch_x = x[batch_indices]  # shape (batch_size,)

                            grads = []
                            for i in range(batch_x.shape[0]):
                                # Compute grad of x[i] w.r.t params
                                grad_outputs = torch.zeros_like(batch_x)
                                grad_outputs[i] = 1.0  # select only the i-th output
                                g = torch.autograd.grad(
                                    batch_x, params, grad_outputs=grad_outputs,
                                    retain_graph=True, create_graph=False
                                )
                                grads.append(torch.cat([gg.flatten() for gg in g]))

                            grads = torch.stack(grads, dim=0)  # shape (batch_size, param_vec_len)
                            jacobian[start:end] = grads

                        return jacobian

                    # Now use vmap over all elements of x
                    jacobian = compute_jacobian_batched(L_ctrl_grad,
                                                        self.policy.value_optimizer.param_groups[0]['params'])
                    print("eee")
                    # grad_batched = autograd.grad(L_ctrl_grad, self.policy.value_optimizer.param_groups[0]['params'],
                    #                             torch.eye(n).to(self.device),
                    #                             is_grads_batched=True,
                    #                             retain_graph=True, create_graph=True)
                if full_hessian is False:
                    reshaped_grads = self.matrix_unbatch(grad_batched, k, size2=n).T
                    reshaped_grads = reshaped_grads * rademacher
                    L_ctrl_hessian = torch.mean(reshaped_grads, dim=1)
                    L_ctrl_hessian = L_ctrl_hessian + 10
                else:
                    L_ctrl_hessian = self.matrix_unbatch(grad_batched, n)
                    L_ctrl_hessian.diag().add_(5)
                d2f1_ctrl_batched = autograd.grad(-policy_loss, self.policy.value_optimizer.param_groups[0]['params'],
                                                  create_graph=True, retain_graph=True)
                d2f1_dstb_batched = autograd.grad(dstb_policy_loss,
                                                  self.policy.value_optimizer.param_groups[0]['params'],
                                                  create_graph=True, retain_graph=True)
                d2f1_ctrl = torch.hstack([t.flatten() for t in d2f1_ctrl_batched])

                d2f1_dstb = torch.hstack([t.flatten() for t in d2f1_dstb_batched])
                # d2f1_ctrl = torch.rand(d2f1_dstb.shape).to(self.device)

                # diag, no other option
                iHvp_ctrl = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_ctrl)
                iHvp_dstb = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_dstb)
                #assert not np.any(self.rollout_buffer.current_shot - self.rollout_buffer.indices[
                #                                                     count * self.batch_size: count * self.batch_size + self.batch_size])
                # assert self.rollout_buffer.current_shot == self.rollout_buffer.indices[count * self.batch_size: count * self.batch_size + self.batch_size]
                traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[
                                                           count * self.batch_size: count * self.batch_size + self.batch_size]].squeeze()
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
                surr_L_ctrl, surr_L_dstb = self.estimators(advantages, ctrl_log_prob, dstb_log_prob,
                                                           x0_values.squeeze(),
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
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
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
                # big_loss = ctrl_loss + dstb_loss + critic_loss
                self.policy.ctrl_optimizer.zero_grad()
                self.policy.dstb_optimizer.zero_grad()
                self.policy.value_optimizer.zero_grad()
                ctrl_tensors = autograd.grad(ctrl_loss, self.policy.ctrl_optimizer.param_groups[0]['params'])
                dstb_tensors = autograd.grad(dstb_loss, self.policy.dstb_optimizer.param_groups[0]['params'])
                value_tensors = autograd.grad(critic_loss, self.policy.value_optimizer.param_groups[0][
                    'params'])  # , create_graph=True, retain_graph=True)

                for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                    self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad = ctrl_tensors[i]
                    self.policy.dstb_optimizer.param_groups[0]['params'][i].grad = dstb_tensors[i]
                th.nn.utils.clip_grad_norm_(self.policy.ctrl_optimizer.param_groups[0]['params'], self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(self.policy.dstb_optimizer.param_groups[0]['params'], self.max_grad_norm)
                for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                    self.policy.value_optimizer.param_groups[0]['params'][i].grad = value_tensors[i]
                th.nn.utils.clip_grad_norm_(self.policy.value_optimizer.param_groups[0]['params'], self.max_grad_norm)
                # big_loss.backward()
                print("e done")
                # ctrl_loss.backward(retain_graph=True)

                with (torch.no_grad()):
                    # ctrl_partials = autograd.grad(ctrl_loss, self.policy.ctrl_optimizer.param_groups[0]['params'])
                    for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                        self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad = \
                            self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad + ctrl_imp[i]
                    th.nn.utils.clip_grad_norm_(self.policy.ctrl_optimizer.param_groups[0]['params'],
                                                self.max_grad_norm)

                    for i in range(len(self.policy.dstb_optimizer.param_groups[0]['params'])):
                        self.policy.dstb_optimizer.param_groups[0]['params'][i].grad = \
                            self.policy.dstb_optimizer.param_groups[0]['params'][i].grad - dstb_imp[i]
                    th.nn.utils.clip_grad_norm_(self.policy.dstb_optimizer.param_groups[0]['params'],
                                                self.max_grad_norm)
                    th.nn.utils.clip_grad_norm_(self.policy.value_optimizer.param_groups[0]['params'],
                                                self.max_grad_norm)

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
                # self.policy.dstb_optimizer.zero_grad()
                # dstb_partials = autograd.grad(dstb_loss, self.policy.dstb_optimizer.param_groups[0]['params'])
                # dstb_loss.backward(retain_graph=True)
                # self.policy.dstb_optimizer.step()
                # values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                #    torch.from_numpy(rollout_data.observations).to(self.device), actions, dstb_actions)
                # values_pred = values.flatten()
                # value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred)
                # critic_loss = self.vf_coef * value_loss
                # self.policy.value_optimizer.zero_grad()
                '''
                critic_partials = autograd.grad(critic_loss, self.policy.value_optimizer.param_groups[0]['params'])
                for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                    self.policy.value_optimizer.param_groups[0]['params'][i].grad = critic_partials[i]'''
                # critic_loss.backward(retain_graph=True)

                # loss.backward()
                # Clip grad norm
                # self.policy.value_optimizer.step()
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

                # buf = deepcopy(self.rollout_buffer)
                # buf.values = torch.from_numpy(self.rollout_buffer.values).to(self.device)
                # buf.rewards = torch.from_numpy(buf.rewards).to(self.device)
                # buf.advantages = torch.from_numpy(buf.advantages).to(self.device)
                # buf.episode_starts = torch.from_numpy(buf.episode_starts).to(self.device)

                # for i in range(buf.buffer_size):
                #    _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                #    buf.values[i] = values.squeeze()
                # _, _, last_values, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))

                # TEST - DO NOT COMMIT

                advantage_test = []
                _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
                last_values = vf.flatten()
                last_gae_lam = th.zeros_like(last_values)
                dones = torch.Tensor(buf.dones[-1]).to(self.device)
                for step in reversed(range(buf.buffer_size)):
                    # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                    if step == buf.buffer_size - 1:
                        next_non_terminal = 1.0 - dones.float()
                        next_values = last_values
                    else:
                        next_non_terminal = 1.0 - buf.episode_starts[step + 1].float()
                        _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[step + 1]).to(self.device))
                        next_values = vf.flatten()
                    _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))

                    delta = buf.rewards[step] + buf.gamma * next_values * next_non_terminal - value_query.squeeze()
                    last_gae_lam = delta + buf.gamma * buf.gae_lambda * next_non_terminal * last_gae_lam
                    advantage_test.append(last_gae_lam)
                    # buf.advantages[step] = last_gae_lam
                advantages = torch.stack(advantage_test, dim=0)
                # buf.returns = buf.advantages + buf.values
                print("")
                # TEST - DO NOT COMMIT

                # buf.compute_returns_and_advantage_pt(values, torch.Tensor(buf.dones[-1]).to(self.device))
                # self.rollout_buffer.advantages = torch.zeros_like(self.rollout_buffer.advantages)
                # self.rollout_buffer.flat_advantages = buf.swap_and_flatten(buf.advantages)
                self.rollout_buffer.advantages = self.rollout_buffer.swap_and_flatten_pt(advantages)
                count = count + 1
                # elf.rollout_buffer.flat_advantages = self.rollout_buffer.swap_and_flatten_pt(self.rollout_buffer.advantages)
                # self.rollout_buffer.flat_advantages = buf.swap_and_flatten_pt(buf.advantages)

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

    def train_one_adversary(self, main_agent, ma_left=False, ma_right=False):
        # helper function

        """
        Update policy using the currently gathered rollout buffer.
        """
        assert ma_left != ma_right

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
        # buf.compute_returns_and_advantage_pt_test(last_values, torch.Tensor(buf.dones[-1]).to(self.device))
        self.rollout_buffer.advantages = buf.advantages
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            count = 0
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

                if ma_left is True:
                    # main player is the left player
                    # adversaries are "dstb" role
                    '''
                    values = torch.zeros((self.batch_size, 1), device=self.device)
                    dstb_log_prob = torch.zeros((self.batch_size,), device=self.device)
                    dstb_entropy = torch.zeros((self.batch_size,), device=self.device)
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
                        '''
                    values, _, _, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                    _, ctrl_log_prob, ctrl_entropy, _, _ = main_agent.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                    values = values.flatten()

                else:
                    # assert self.update_right is True
                    # main player is the right player
                    # adversaries are the control role
                    '''
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
                    '''
                    _, _, _, dstb_log_prob, dstb_entropy = main_agent.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                    values, ctrl_log_prob, ctrl_entropy, _, _ = self.policy.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions)
                    values = values.flatten()
                # Normalize advantage
                if type(rollout_data.advantages) is np.ndarray:
                    advantages = torch.from_numpy(rollout_data.advantages).to(self.device)
                else:
                    advantages = rollout_data.advantages
                # advantages = torch.from_numpy(rollout_data.advantages).to(self.device)
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

                L_ctrl_grad_batched = autograd.grad(value_loss, self.policy.value_optimizer.param_groups[0]['params'],
                                                    create_graph=True, retain_graph=True)
                L_ctrl_grad = torch.cat([t.flatten() for t in L_ctrl_grad_batched], dim=0)
                # L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])
                full_hessian = False
                n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])
                if full_hessian is False:

                    k = 50
                    # n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])

                    rademacher = torch.bernoulli(torch.from_numpy(np.ones((n, k)) * .5)).to(self.device)
                    rademacher[rademacher == 0] = -1
                    # grad_batched = autograd.grad(L_ctrl_grad, flat_params, rademacher,0,1, is_grads_batched=True)
                    grad_batched = autograd.grad(L_ctrl_grad, self.policy.value_optimizer.param_groups[0]['params'],
                                                 torch.transpose(rademacher.to(self.device), 0, 1),
                                                 is_grads_batched=True,
                                                 retain_graph=True, create_graph=True)

                else:
                    grad_batched = autograd.grad(L_ctrl_grad, self.policy.value_optimizer.param_groups[0]['params'],
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
                # d2f1_ctrl_batched = autograd.grad(policy_loss, self.policy.value_optimizer.param_groups[0]['params'],
                #                                  create_graph=True, retain_graph=True)
                d2f1_dstb_batched = autograd.grad(dstb_policy_loss,
                                                  self.policy.value_optimizer.param_groups[0]['params'],
                                                  create_graph=True, retain_graph=True)
                # d2f1_ctrl = torch.hstack([t.flatten() for t in d2f1_ctrl_batched])

                d2f1_dstb = torch.hstack([t.flatten() for t in d2f1_dstb_batched])
                # d2f1_ctrl = torch.rand(d2f1_dstb.shape).to(self.device)

                # diag, no other option
                # iHvp_ctrl = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_ctrl)
                if not full_hessian:
                    iHvp_dstb = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_dstb)
                else:
                    iHvp_dstb = torch.linalg.solve(L_ctrl_hessian, d2f1_dstb)
                # assert not np.any(self.rollout_buffer.current_shot - self.rollout_buffer.indices[
                #                                                     count * self.batch_size: count * self.batch_size + self.batch_size])
                # assert self.rollout_buffer.current_shot == self.rollout_buffer.indices[count * self.batch_size: count * self.batch_size + self.batch_size]
                traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[
                                                           count * self.batch_size: count * self.batch_size + self.batch_size]].squeeze()
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
                surr_L_ctrl, surr_L_dstb = self.estimators(advantages, ctrl_log_prob, dstb_log_prob,
                                                           x0_values.squeeze(),
                                                           torch.Tensor(x0_returns).to(self.device))
                # d1f2_ctrl_batched = autograd.grad(surr_L_ctrl, self.policy.value_optimizer.param_groups[0]['params'],
                #                                  create_graph=True, retain_graph=True)
                # d1f2_ctrl = torch.hstack([t.flatten() for t in d1f2_ctrl_batched])
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

                for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                    self.policy.dstb_optimizer.param_groups[0]['params'][i].grad = \
                        self.policy.dstb_optimizer.param_groups[0]['params'][i].grad - dstb_imp[i]

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.ctrl_optimizer.step()
                self.policy.dstb_optimizer.step()
                self.policy.value_optimizer.step()

                advantage_test = []
                _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
                last_values = vf.flatten()
                last_gae_lam = th.zeros_like(last_values)
                dones = torch.Tensor(buf.dones[-1]).to(self.device)
                for step in reversed(range(buf.buffer_size)):
                    # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                    if step == buf.buffer_size - 1:
                        next_non_terminal = 1.0 - dones.float()
                        next_values = last_values
                    else:
                        next_non_terminal = 1.0 - buf.episode_starts[step + 1].float()
                        _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[step + 1]).to(self.device))
                        next_values = vf.flatten()
                    _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))

                    delta = buf.rewards[step] + buf.gamma * next_values * next_non_terminal - value_query.squeeze()
                    last_gae_lam = delta + buf.gamma * buf.gae_lambda * next_non_terminal * last_gae_lam
                    advantage_test.append(last_gae_lam)
                    # buf.advantages[step] = last_gae_lam
                advantages = torch.stack(advantage_test, dim=0)
                # buf.returns = buf.advantages + buf.values
                # print("")
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
        # TODO: self.rollout_buffer.X0_VALUES_MASTER + env_indices
        grad_estimator = 2 * (advantages * ctrl_logp).mean() * (x0_returns - x0_values).mean()
        return grad_estimator

    def prep_grad_psi_L(self, advantages, dstb_logp, x0_values, x0_returns):
        grad_estimator = 2 * (advantages * dstb_logp).mean() * (x0_returns - x0_values).mean()
        return grad_estimator

    def estimators(self, advantages, ctrl_logp, dstb_logp, x0_values, x0_returns):
        return 2 * (advantages * ctrl_logp).mean() * (x0_returns - x0_values).mean(), 2 * (
                    advantages * dstb_logp).mean() * (x0_returns - x0_values).mean()

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
        self.update_ctrl = True
        self.update_dstb = False

        # print("")

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # set flags once and for all
        self.update_lr_ctrl = self.update_ctrl
        self.update_lr_critic = False
        self.update_lr_dstb = self.update_dstb
        # modify flags after lr update is done
        self._update_learning_rate(
            self.policy.ctrl_optimizer) if self.update_ctrl is True else self._update_learning_rate(
            self.policy.dstb_optimizer)
        self.update_lr_ctrl = False
        self.update_lr_dstb = False
        self.update_lr_critic = True
        self._update_learning_rate(self.policy.value_optimizer)
        self.update_lr_critic = False
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
                        log_ratio = ctrl_log_prob - torch.from_numpy(rollout_data.old_log_prob).to(self.device)
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    else:
                        log_ratio = dstb_log_prob - torch.from_numpy(rollout_data.old_dstb_log_prob).to(self.device)
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
        self.update_dstb = not self.update_ctrl


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
                 dstb_action_space=None,
                 warmstarted_cont_MAGICS=False
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
        self.warmstarted_cont_MAGICS = warmstarted_cont_MAGICS
        if self.warmstarted_cont_MAGICS is True:
            print("this model is warmstarted! now running magics_ppo training", flush=True)
        self.learning_rate = [c_learning_rate, d_learning_rate, v_learning_rate]
        # self.learning_rate_decay_phase =

    def warmstart_setup(self, joint_schedule, use_policy_extractor=True):

        assert self.warmstarted_cont_MAGICS == True
        # can't call this method if not warmstarting

        # use only value cnn extractor

        for param in self.policy.vf_features_extractor.parameters():
            param.requires_grad = False
        for param in self.policy.pi_ctrl_features_extractor.parameters():
            param.requires_grad = False
        for param in self.policy.pi_dstb_features_extractor.parameters():
            param.requires_grad = False

        self.policy.pi_ctrl_features_extractor = self.policy.vf_features_extractor
        self.policy.pi_dstb_features_extractor = self.policy.vf_features_extractor

        self.policy.ctrl_optimizer = torch.optim.AdamW(
            itertools.chain(self.policy.mlp_extractor.policy_net.parameters(),
                            self.policy.action_net.parameters()), joint_schedule[0](1), maximize=False)
        self.policy.dstb_optimizer = torch.optim.AdamW(
            itertools.chain(self.policy.mlp_extractor.dstb_net.parameters(),
                            self.policy.dstb_action_net.parameters()), joint_schedule[1](1), maximize=False)
        self.policy.value_optimizer = torch.optim.AdamW(
            itertools.chain(self.policy.mlp_extractor.value_net.parameters(),
                            self.policy.value_net.parameters()),
            joint_schedule[2](1), **self.policy.optimizer_kwargs)

    def warmstart_buffer_setup(self, n_steps, n_envs, batch_size):
        buffer = AdvRolloutBuffer(n_steps, self.observation_space, self.action_space, device=self.device,
                                  gamma=self.gamma,
                                  gae_lambda=self.gae_lambda,
                                  n_envs=n_envs,
                                  **self.rollout_buffer_kwargs
                                  )

        self.batch_size = batch_size
        self.n_envs = n_envs
        self.n_steps = n_steps

        return buffer

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        if self.warmstarted_cont_MAGICS is True:
            if self.warmstarted_cont_MAGICS is True:
                print("this model is warmstarted! now running magics_ppo training", flush=True)
            return super().train()

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


class Specialized_Agent(TSS_PPO):
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
        if _init_setup_model:
            self._setup_model()

        # at this point in the code, the Specialized_Agent's policy and value function are set up (we don't care about hte other one)
        # now we need to create the adversaries
        self.num_adversaries = num_adversary
        adversaries = []
        self.env.num_envs = self.n_env_per_adv
        for i in range(num_adversary):
            adversaries.append(TSS_PPO("AACCnnPolicy",
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
                                       ))
            adversaries[i].this_adv = opp_list[i]
            adversaries[i].rollout_buffer.n_envs = self.n_env_per_adv

        self.env.num_envs = self.n_envs
        print("created %d adversaries" % self.num_adversaries)
        self.adversaries = adversaries
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
            self.adversaries[i].rollout_buffer.reset()
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
                s_actions, s_log_probs, s_values, s_dstb_actions, s_dstb_log_probs = self.policy(obs_tensor)
                all_adv_left_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
                all_adv_right_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
                all_adv_critic_values = torch.zeros((self.n_global_env, 1), device=self.device)
                all_adv_log_probs = torch.zeros((self.n_global_env,), device=self.device)
                all_adv_dstb_log_probs = torch.zeros((self.n_global_env,), device=self.device)
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
            actions = actions.cpu().numpy()
            adversary_actions = adversary_actions.cpu().numpy()

            if self.use_mirror is True:
                mirror_master_copy_actions = deepcopy(actions)
                mirror_master_copy_adv_actions = deepcopy(adversary_actions)

            # upper half, lower half

            if self.use_mirror is True:
                #print("SINGLE TRAIN EXTRACTOR MIRROR")

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
                prot_reversed = actions[1::2, :] # actions for prot when he is on the right... but its backwards right now!

                adv_right = adversary_actions[0::2, :]
                adv_reversed = adversary_actions[1::2, :]

                temp = np.zeros((self.num_adversaries, self.action_space.shape[0]))
                temp = prot_reversed

                actions[1::2, :] = adversary_actions[1::2, :]
                adversary_actions[1::2, :] = temp

            # Rescale and perform action
            if self.update_left is True:
                #MESSY
                clipped_actions = np.hstack([actions, adversary_actions])
            else:
                clipped_actions = np.hstack([adversary_actions, actions])
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, rew_other, dones, infos = env.step(clipped_actions)
            # assert np.allclose(rewards + rew_other, np.zeros(rewards.shape))
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
                    self.adversaries[i].rollout_buffer.add(
                        self._last_obs[chunk],
                        actions[chunk],
                        adversary_actions[chunk],
                        rewards[chunk],
                        self._last_episode_starts[chunk],
                        all_adv_critic_values[chunk],
                        log_probs[chunk],
                        adversary_log_probs[chunk]  # not done
                    )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = torch.zeros((self.n_global_env, 1))
            for i in range(self.num_adversaries):
                chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
                values[chunk] = self.adversaries[i].policy.predict_values(obs_as_tensor(new_obs, self.device))[
                    chunk].to('cpu')
            # not bootstrapped correctly
            # use adversary critics not ma critic
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        for i in range(self.num_adversaries):  # is this a bug?
            chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
            self.adversaries[i].rollout_buffer.compute_returns_and_advantage(last_values=values[chunk],
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

        self.train_ma()
        for i in range(self.num_adversaries):
            self.adversaries[i].train_one_adversary(self.policy, ma_left=self.update_left, ma_right=self.update_right),
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
                adversary_id = buf.env_indices[i] // self.n_env_per_adv
                for j in range(self.num_adversaries):
                    _, _, values, _, _ = self.adversaries[j].policy(
                        torch.Tensor(buf.observations[i][adversary_id == j]).to(self.device))
                    # _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                    buf.values[i][adversary_id == j] = values.squeeze()
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
                    values = torch.zeros((self.batch_size, 1), device=self.device)
                    dstb_log_prob = torch.zeros((self.batch_size,), device=self.device)
                    dstb_entropy = torch.zeros((self.batch_size,), device=self.device)
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
                all_adv_val_params = list(itertools.chain.from_iterable(
                    [self.adversaries[i].policy.value_optimizer.param_groups[0]['params'] for i in
                     range(self.num_adversaries)]))
                if self.warmstarted_cont_MAGICS is True:
                    L_ctrl_grad_batched = autograd.grad(value_loss, all_adv_val_params,
                                                        create_graph=True, retain_graph=True)
                    L_ctrl_grad = torch.cat([t.flatten() for t in L_ctrl_grad_batched], dim=0)
                    # L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])
                    full_hessian = False
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
                        grad_batched = autograd.grad(L_ctrl_grad, self.policy.value_optimizer.param_groups[0]['params'],
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
                    iHvp_ctrl = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_ctrl)
                    # iHvp_dstb = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_dstb)
                    # assert not np.any(self.rollout_buffer.current_shot - self.rollout_buffer.indices[
                    #                                                     count * self.batch_size: count * self.batch_size + self.batch_size])
                    # assert self.rollout_buffer.current_shot == self.rollout_buffer.indices[count * self.batch_size: count * self.batch_size + self.batch_size]
                    traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[
                                                               count * self.batch_size: count * self.batch_size + self.batch_size]].squeeze()
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

                loss = ctrl_policy_loss + self.ent_coef * ctrl_entropy_loss + self.dstb_ent_coef * dstb_entropy_loss + self.vf_coef * value_loss + dstb_policy_loss

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
                self.policy.dstb_optimizer.step()
                self.policy.value_optimizer.step()
                if self.warmstarted_cont_MAGICS is True:
                    advantage_test = []
                    vf = torch.zeros_like(buf.values[-1])
                    adversary_id = buf.env_indices[-1] // self.n_env_per_adv
                    for j in range(self.num_adversaries):
                        _, _, values, _, _ = self.adversaries[j].policy(
                            torch.Tensor(buf.observations[-1][adversary_id == j]).to(self.device))
                        # _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                        vf[adversary_id == j] = values.squeeze()
                    # _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
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
                            adversary_id = buf.env_indices[-1] // self.n_env_per_adv
                            for j in range(self.num_adversaries):
                                _, _, temp_values, _, _ = self.adversaries[j].policy(
                                    torch.Tensor(buf.observations[step + 1][adversary_id == j]).to(self.device))
                                # _, _, temp_values, _, _ = self.policy(torch.Tensor(buf.observations[step + 1]).to(self.device))
                                next_values[adversary_id == j] = temp_values.flatten()
                        value_query = torch.zeros_like(buf.values[-1])
                        # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                        adversary_id = buf.env_indices[step] // self.n_env_per_adv
                        for j in range(self.num_adversaries):
                            _, _, temp_values, _, _ = self.adversaries[j].policy(
                                torch.Tensor(buf.observations[step][adversary_id == j]).to(self.device))
                            value_query[adversary_id == j] = temp_values.squeeze()

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

    def predict(self, obs, env_index, deterministic=False):
        if self.use_mirror is True:
            # when mirror is true, ego is fighting ego
            # we need to query the policy twice

            (ego_action, state), (right_action, _) = self.policy.predict(obs, deterministic=deterministic)
            left_action = ego_action
        else:
            (left_action, state), (_, _) = self.policy.predict(obs, deterministic=deterministic)
            (_, _), (right_action, _) = self.adversaries[env_index].predict(obs, deterministic=deterministic)
        return (left_action, state), (right_action, state)


class Specialized_Agent_IPPO(Derivative_Free_SPAR):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "AACCnnPolicy": ActorActorCriticCnnPolicy,
        "IPPOAACCnnPolicy": IPPOActorCriticCnnGeneralistPolicy
    }

    def __init__(self,
                 policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 env_batch_size: int = 32,
                 envs_per_matchup: int = 1,
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
                 use_mirror=False,
                 state_len=None,
                 player=None,
                 env_generator_func=None,
                 state_list=None
                 ):

        if warmstarted_cont_MAGICS is True:
            print(
                "warmstarted_cont_MAGICS is True but this is IPPO-specialized. MAGICS training not supported. Overriding to False.")
            warmstarted_cont_MAGICS = False

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
            I_AM_LEFT=I_AM_LEFT,
            I_AM_RIGHT=I_AM_RIGHT,
            dstb_action_space=dstb_action_space,
            num_adversary=num_adversary,
            n_global_env=n_global_env,
            n_env_per_adv=n_env_per_adv,
            warmstarted_cont_MAGICS=warmstarted_cont_MAGICS,
            opp_list=opp_list,
            use_mirror=use_mirror
        )
        self.vf_coef = .5
        self.state_len = state_len
        self.env_batch_size = env_batch_size
        self.envs_per_matchup = envs_per_matchup
        self.state_list = state_list
        self.env_generator_func = env_generator_func
        if self.policy is not None: 
            self.policy.num_env_per_adv = self.envs_per_matchup

    # def collect_rollouts(
    #         self,
    #         env: VecEnv,
    #         callback: BaseCallback,
    #         rollout_buffer: RolloutBuffer,
    #         n_rollout_steps: int,
    # ) -> bool:
    #     # self._setup_learn()
    #     assert self._last_obs is not None, "No previous observation was provided"
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)

    #     n_steps = 0
    #     rollout_buffer.reset()
    #     for i in range(self.num_adversaries):
    #         self.adversaries[i].rollout_buffer.reset()
    #     # Sample new weights for the state dependent exploration
    #     if self.use_sde:
    #         self.policy.reset_noise(env.num_envs)

    #     callback.on_rollout_start()

    #     while n_steps < n_rollout_steps:
    #         if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.policy.reset_noise(env.num_envs)

    #         with th.no_grad():
    #             # Convert to pytorch tensor or to TensorDict
    #             obs_tensor = obs_as_tensor(self._last_obs, self.device)
    #             s_actions, s_log_probs, s_values_ego, s_values_adv, s_dstb_actions, s_dstb_log_probs = self.policy(obs_tensor)
    #             all_adv_left_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
    #             all_adv_right_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
    #             all_adv_critic_values_ego = torch.zeros((self.n_global_env, 1), device=self.device)
    #             all_adv_critic_values_adv = torch.zeros((self.n_global_env, 1), device=self.device)
    #             all_adv_log_probs = torch.zeros((self.n_global_env,), device=self.device)
    #             all_adv_dstb_log_probs = torch.zeros((self.n_global_env,), device=self.device)
    #         actions = s_actions
    #         adversary_actions = s_dstb_actions
    #         log_probs = s_log_probs
    #         adversary_log_probs = s_dstb_log_probs
    #         actions = actions.cpu().numpy()
    #         adversary_actions = adversary_actions.cpu().numpy()
    #         all_adv_critic_values_ego = s_values_ego
    #         all_adv_critic_values_adv = s_values_adv
    #             # for i in range(self.num_adversaries):
    #             #     actions, log_probs, values, dstb_actions, dstb_log_probs = self.adversaries[i].policy(obs_tensor)
    #             #     # actions = actions.cpu()
    #             #     # dstb_actions = dstb_actions.cpu()
    #             #     chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
    #             #     all_adv_left_actions[chunk] = actions[chunk]
    #             #     all_adv_log_probs[chunk] = log_probs[chunk]
    #             #     all_adv_critic_values[chunk] = values[chunk]
    #             #     all_adv_right_actions[chunk] = dstb_actions[chunk]
    #             #     all_adv_dstb_log_probs[chunk] = dstb_log_probs[chunk]

    #         # if self.update_left is True:
    #         #     # specialized agent is playing left
    #         #     # all adversaries are playing right.
    #         #     all_adv_left_actions = []
    #         #     all_adv_log_probs = []
    #         #     actions = s_actions
    #         #     log_probs = s_log_probs
    #         #     adversary_actions = all_adv_right_actions
    #         #     adversary_log_probs = all_adv_dstb_log_probs
    #         # else:
    #         #     all_adv_right_actions = []
    #         #     all_adv_dstb_log_probs = []
    #         #     actions = s_dstb_actions
    #         #     log_probs = s_dstb_log_probs
    #         #     adversary_actions = all_adv_left_actions
    #         #     adversary_log_probs = all_adv_log_probs
    #         actions = actions.cpu().numpy()
    #         adversary_actions = adversary_actions.cpu().numpy()
    #         if self.use_mirror is True:
    #             mirror_master_copy_actions = deepcopy(actions)
    #             mirror_master_copy_adv_actions = deepcopy(adversary_actions)

    #         # upper half, lower half

    #         if self.use_mirror is True:
    #             # print("SINGLE TRAIN EXTRACTOR MIRROR")

    #             '''
    #             assume wlog Ehonda is the prot.

    #             action right now is:                  adv_action right now is:
    #             EHonda left                                              Sagat    right
    #             EHonda left                                              Sagat    right
    #             EHonda left                                             MBison    right
    #             EHonda left                                             MBison    right

    #             EHonda v Sagat       0
    #             Sagat v. EHonda      1
    #             EHonda v. MBison     2
    #             MBison v. EHonda     3

    #             action[odds] needs to go to the other side because our design makes prot actions left

    #             same with adversary[ods] -- adversary is on the right so adv[ods] is backwards

    #             '''

    #             prot_left = actions[0::2, :]  # actions for the prot when he is on the left
    #             prot_reversed = actions[1::2,
    #                             :]  # actions for prot when he is on the right... but its backwards right now!

    #             adv_right = adversary_actions[0::2, :]
    #             adv_reversed = adversary_actions[1::2, :]

    #             temp = np.zeros((self.num_adversaries, self.action_space.shape[0]))
    #             temp = prot_reversed

    #             actions[1::2, :] = adversary_actions[1::2, :]
    #             adversary_actions[1::2, :] = temp
    #         # Rescale and perform action
    #         if self.update_left is True:
    #             clipped_actions = np.hstack([actions, adversary_actions])
    #         else:
    #             clipped_actions = np.hstack([adversary_actions, actions])
    #         # Clip the actions to avoid out of bound error
    #         if isinstance(self.action_space, spaces.Box):
    #             clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

    #         new_obs, rewards, rew_other, dones, infos = env.step(clipped_actions)
    #         # assert np.allclose(rewards + rew_other, np.zeros(rewards.shape))
    #         self.num_timesteps += env.num_envs
    #         wandb.log({"epochs": self.num_timesteps})            # Give access to local variables
    #         callback.update_locals(locals())
    #         if callback.on_step() is False:
    #             return False

    #         self._update_info_buffer(infos)
    #         n_steps += 1

    #         if isinstance(self.action_space, spaces.Discrete):
    #             # Reshape in case of discrete action
    #             actions = actions.reshape(-1, 1)

    #         # Handle timeout by bootstraping with value function
    #         # see GitHub issue #633
    #         for idx, done in enumerate(dones):
    #             if (
    #                     done
    #                     and infos[idx].get("terminal_observation") is not None
    #                     and infos[idx].get("TimeLimit.truncated", False)
    #             ):
    #                 terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
    #                 with th.no_grad():
    #                     terminal_value = self.policy.predict_values(terminal_obs)[0]
    #                 rewards[idx] += self.gamma * terminal_value
    #         if self.use_mirror is True:
    #             rollout_buffer.add(
    #                 self._last_obs,  # type: ignore[arg-type]
    #                 mirror_master_copy_actions,
    #                 mirror_master_copy_adv_actions,
    #                 rewards,
    #                 self._last_episode_starts,  # type: ignore[arg-type]
    #                 all_adv_critic_values.squeeze(),
    #                 log_probs,
    #                 adversary_log_probs
    #             )
    #         else:
    #             rollout_buffer.add(
    #             self._last_obs,  # type: ignore[arg-type]
    #             actions,
    #             adversary_actions,
    #             rewards,
    #             self._last_episode_starts,  # type: ignore[arg-type]
    #             all_adv_critic_values_ego.squeeze(),
    #             log_probs,
    #             adversary_log_probs
    #             )

    #         for i in range(self.num_adversaries):
    #             chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
    #             if self.use_mirror is True:
    #                 self.adversaries[i].rollout_buffer.add(
    #                     self._last_obs[chunk],
    #                     mirror_master_copy_actions[chunk],
    #                     mirror_master_copy_adv_actions[chunk],
    #                     rewards[chunk],
    #                     self._last_episode_starts[chunk],
    #                     all_adv_critic_values_adv[chunk],
    #                     log_probs[chunk],
    #                     adversary_log_probs[chunk]  # not done
    #                 )
    #             else:
    #                 self.adversaries[i].rollout_buffer.add(
    #                     self._last_obs[chunk],
    #                     actions[chunk],
    #                     adversary_actions[chunk],
    #                     -rewards[chunk],
    #                     self._last_episode_starts[chunk],
    #                     -all_adv_critic_values_adv[chunk],
    #                     log_probs[chunk],
    #                     adversary_log_probs[chunk]  # not done
    #                 )

    #         self._last_obs = new_obs
    #         self._last_episode_starts = dones

    #     with th.no_grad():
    #         # Compute value for the last timestep
    #         # values = torch.zeros((self.n_global_env,1))
    #         # for i in range(self.num_adversaries):
    #         #    chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
    #         #    values[chunk] = self.adversaries[i].policy.predict_values(obs_as_tensor(new_obs, self.device))[chunk].to('cpu')
    #         # not bootstrapped correctly
    #         # use adversary critics not ma critic
    #         values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
    #     rollout_buffer.compute_returns_and_advantage(last_values=values[0], dones=dones)

    #     for i in range(self.num_adversaries):  # is this a bug?
    #         chunk = range(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
    #         self.adversaries[i].rollout_buffer.compute_returns_and_advantage(last_values=-values[1][chunk],
    #                                                                          dones=dones[chunk])

    #     callback.on_rollout_end()

    #     return True

    # def train(self):
    #     TSS_PPO.train(self)
    #     for i in range(self.num_adversaries):
    #         self.adversaries[i].train_one_adversary(self.policy, ma_left=self.update_left, ma_right=self.update_right)

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
            # if total_batches == 1:
            #     result = super().collect_rollouts(
            #         rollout_env,
            #         callback,
            #         rollout_buffer,
            #         adversary_buffers,
            #         n_rollout_steps
            #     )
            #     if not result:  # If parent method returned False, propagate it
            #         return False
        
            #     return True
            # else:
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
                    s_actions, s_log_probs, s_values_ego, s_values_adv,s_dstb_actions, s_dstb_log_probs = self.policy(obs_tensor, network_keys=network_keys)
                    all_adv_left_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
                    all_adv_right_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
                    all_adv_critic_values = torch.zeros((self.n_global_env, 1), device=self.device)
                    all_adv_log_probs = torch.zeros((self.n_global_env,), device=self.device)
                    all_adv_dstb_log_probs = torch.zeros((self.n_global_env,), device=self.device)
                actions = s_actions
                adversary_actions = s_dstb_actions
                log_probs = s_log_probs
                adversary_log_probs = s_dstb_log_probs
                actions = actions.cpu().numpy()
                adversary_actions = adversary_actions.cpu().numpy()
                all_adv_critic_values_ego = s_values_ego
                all_adv_critic_values_adv = s_values_adv

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
                
                if np.any(rewards != 0):
                    print("Reward is not 0")
                ego_vertical_batch_obs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :, :, :] = th.unsqueeze(th.from_numpy(new_obs), 0)
                ego_vertical_batch_rewards[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(rewards), 0)
                #vertical_batch_rewards_other[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :] = th.unsqueeze(th.from_numpy(rew_other), 0)
                ego_vertical_batch_dones[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(dones), 0)
                #vertical_batch_infos[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :] = th.unsqueeze(th.from_numpy(infos), 0)
                ego_vertical_batch_log_probs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(log_probs, 0).cpu()
                ego_vertical_batch_values[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(s_values_ego, 0)
                ego_vertical_batch_dstb_log_probs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(s_dstb_log_probs, 0).cpu()
                ego_vertical_batch_last_ep_starts[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(self._last_episode_starts), 0)
                ego_vertical_batch_actions[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(actions), 0)
                ego_vertical_batch_adversary_actions[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(adversary_actions), 0)
                
                # For each environment in the batch, assign its data to the correct adversary buffer and slot.
                for j in range(env.num_envs):
                    # Calculate the global index of the environment across all batches.
                    global_env_idx = i_start + j
                    print(global_env_idx)
                    
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
                    adv_vertical_batch_values[matchup_idx][count, local_env_idx] = -all_adv_critic_values_adv[j]
                    adv_vertical_batch_dstb_log_probs[matchup_idx][count, local_env_idx] = s_dstb_log_probs[j]
                    adv_vertical_batch_last_ep_starts[matchup_idx][count, local_env_idx] = th.from_numpy(self._last_episode_starts)[j]
                    #last_ep_starts[global_env_idx] = th.from_numpy(np.round(dones[j]).astype(bool))

                    adv_vertical_batch_actions[matchup_idx][count, local_env_idx].copy_(th.from_numpy(actions[j]))
                    adv_vertical_batch_adversary_actions[matchup_idx][count, local_env_idx].copy_(th.from_numpy(adversary_actions[j]))


                self.num_timesteps += env.num_envs
                #wandb.log({"epochs": self.num_timesteps})
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
            final_ego_values, final_adv_values = self.policy.predict_values(final_obs_all_envs)

        rollout_buffer.values = rollout_buffer.values.to(self.device, non_blocking=True)
        rollout_buffer.rewards = rollout_buffer.rewards.to(self.device, non_blocking=True)
        rollout_buffer.advantages = rollout_buffer.advantages.to(self.device, non_blocking=True)
        rollout_buffer.episode_starts = rollout_buffer.episode_starts.to(self.device, non_blocking=True)
        #rollout_buffer.vectorized_compute_returns_and_advantages(last_values=values, dones=torch.Tensor(dones).to(self.device))
        rollout_buffer.vectorized_compute_returns_and_advantages(last_values=final_ego_values, dones=final_dones_all_envs)
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].values = adversary_buffers[i].values.to(self.device, non_blocking=True)
            adversary_buffers[i].rewards = adversary_buffers[i].rewards.to(self.device, non_blocking=True)
            adversary_buffers[i].advantages = adversary_buffers[i].advantages.to(self.device, non_blocking=True)
            adversary_buffers[i].episode_starts = adversary_buffers[i].episode_starts.to(self.device, non_blocking=True)
            #adversary_buffers[i].vectorized_compute_returns_and_advantages(last_values=values, dones=final_dones_all_envs)
            

            start_idx = i * self.envs_per_matchup
            end_idx = (i + 1) * self.envs_per_matchup
            adv_last_values = final_adv_values[start_idx:end_idx]
            adv_dones = final_dones_all_envs[start_idx:end_idx]
            adversary_buffers[i].vectorized_compute_returns_and_advantages(last_values=adv_last_values, dones=adv_dones)
        
        callback.on_rollout_end()

        rollout_buffer.prepare_data_for_training()
        for buf in adversary_buffers:
            buf.prepare_data_for_training()
        
        return True
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
        try:
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
                #print("perturbed agent created!", flush=True)
                #self._initialize_parallel_updater()                
                # perturbed_buf, perturbed_adv_buf = perturbed_agent.env_perturb_params() #TODO: This is a sequential original line, delete it when done.
                #continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps) #TODO: This is sequential - remove when done.

                # Run env_perturb_params and collect_rollouts in different threads (cannot be done in different processes because they contain unpickleable objects)
                #with ThreadPoolExecutor(max_workers=2) as executor:
                #    future_perturbed = executor.submit(perturbed_agent.env_perturb_params)
                #self.collect_rollouts, self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps)
                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps) #TODO: This is sequential - remove when done.    
                    #perturbed_buf, perturbed_adv_buf = future_perturbed.result()
                    #continue_training = future_collect.result()
                #self.perturbed_agent = perturbed_agent
                #self.perturbed_buf = perturbed_buf
                #self.perturbed_adv_buf = perturbed_adv_buf
                #self.perturbed_agent_policy = perturbed_agent.policy
                #print("main agent and perturbed agent rollout done!", flush=True)
                
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
                        #wandb.log({"eval_rew": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])})
                        self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("time/fps", fps)
                    self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                    self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                    self.logger.dump(step=self.num_timesteps)
            

                self.train()
                #self.perturbed_agent.env.close()
                #del self.perturbed_agent

            callback.on_training_end()
        except Exception as e:
            print(e)
            print(self.dump_properties())
            raise e
        #finally:
        #    #IMPORTANT! Persistent workers must be cleaned up.
        #    self.cleanup()
        #    torch.cuda.empty_cache()

        return self
    
    def train(self):

        self.policy.num_global_env = self.n_global_env
        self.update(self.rollout_buffer, self.policy, True, self.clip_range)
        self.update(self.adversary_buffers, self.policy, False, self.clip_range)
    

    def update(self, rollout_buffer, ori_policy, ego, clip_range):
        #network_keys, curr_buf = self._get_buffers_and_keys(rollout_buffer, ego)
        for epoch in range(self.n_epochs):

            num_runs_count = 1 if ego else self.num_adversaries
            for i in range(num_runs_count):
                network_keys, curr_buf = self._get_buffers_and_keys(rollout_buffer, ego, i)

                for ori_rollout_data in curr_buf.get(self.batch_size):
                    policy_loss, log_prob, entropy = self._calculate_policy_loss(
                            ori_rollout_data, ori_policy, ego, network_keys, clip_range
                        )
                    optimizer = self.policy.ctrl_optimizer if ego else self.policy.dstb_optimizer
                    optimizer.zero_grad()
                    policy_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    optimizer.step()
                    value_loss = self._update_value_functions(ori_rollout_data, ego, network_keys, clip_range)
                    # need to update value functions here 
                #self.policy.value_optimizer.step()
        
        return
    
    def _get_buffers_and_keys(self, buf, ego, index):
        if ego:
            network_keys = [k for k in range(self.num_adversaries)]
            curr_buf = buf
        else:
            network_keys = [index]
            curr_buf = buf[index]
        return network_keys, curr_buf
    
    def _update_value_functions(self, batch, ego, network_keys, clip_range):
        if ego:
            values, _, _, _, _, _ = self.policy.evaluate_actions(batch.observations, batch.actions, batch.dstb_actions, network_keys=network_keys, shuffle_keys=batch.env_indices)
        else:
            _, values, _, _, _, _ = self.policy.evaluate_actions(batch.observations, batch.actions, batch.dstb_actions, network_keys=network_keys, shuffle_keys=batch.env_indices)
        loss = F.mse_loss(values, batch.returns)
        optimizer = self.policy.ego_value_optimizer if ego else self.policy.adv_value_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def _calculate_policy_loss(self, rollout_data, policy, ego, network_keys, clip_range):
        clip_range = self.clip_range(self._current_progress_remaining)
        actions = torch.Tensor(rollout_data.actions).to(self.device)
        dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(self.device)

        if self.use_sde:
            policy.reset_noise(self.batch_size)

        #with torch.no_grad():
        if ego:
            old_log_prob = rollout_data.old_log_prob
            _, _, log_prob, entropy, _, _ = policy.evaluate_actions(
                torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions,
                shuffle_keys=rollout_data.env_indices, network_keys=network_keys
            )
        else:
            old_log_prob = rollout_data.old_dstb_log_prob
            _, _, _, _, log_prob, entropy = policy.evaluate_actions(
                torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions,
                shuffle_keys=rollout_data.env_indices, network_keys=network_keys
            )
        
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - torch.Tensor(old_log_prob).to(self.device))
        
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = torch.min(policy_loss_1, policy_loss_2).mean()
        
        return policy_loss, log_prob, entropy

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.ego_value_optimizer", "policy.adv_value_optimizer", "policy.ctrl_optimizer", "policy.dstb_optimizer"]
        return state_dicts, []


class eepy(MAGICS_PPO):
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
            update_left=True,
            update_right=True,
            dstb_action_space=None,
    ):
        super().__init__(
            policy=policy,
            env=env,
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
            dstb_action_space=dstb_action_space
        )

        return

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
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

        def hook_fn(grad):
            raise RuntimeError("GRADIENT MODIFIED")

        '''for p in self.policy.value_optimizer.param_groups[0]['params']:
            p.register_hook(hook_fn)'''
        continue_training = True

        # train for n_epochs epochs

        # self.rollout_buffer.advantages = buf.advantages

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
        # buf.compute_returns_and_advantage_pt_test(last_values, torch.Tensor(buf.dones[-1]).to(self.device))
        self.rollout_buffer.advantages = buf.advantages
        '''
        buf = deepcopy(self.rollout_buffer)
        buf.rewards = torch.from_numpy(buf.rewards).to(self.device)
        buf.episode_starts = torch.from_numpy(buf.episode_starts).to(self.device)
        advantage_test = []
        _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
        last_values = vf.flatten()
        last_gae_lam = th.zeros_like(last_values)
        dones = torch.Tensor(buf.dones[-1]).to(self.device)
        for step in reversed(range(buf.buffer_size)):
            # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
            if step == buf.buffer_size - 1:
                next_non_terminal = 1.0 - dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - buf.episode_starts[step + 1].float()
                _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[step + 1]).to(self.device))
                next_values = vf.flatten()
            _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))

            delta = buf.rewards[step] + buf.gamma * next_values * next_non_terminal - value_query.squeeze()
            last_gae_lam = delta + buf.gamma * buf.gae_lambda * next_non_terminal * last_gae_lam
            advantage_test.append(last_gae_lam)
            # buf.advantages[step] = last_gae_lam
        advantages = torch.stack(advantage_test, dim=0)
        # buf.returns = buf.advantages + buf.values
        print("")
        self.rollout_buffer.advantages = advantages
        '''
        # TEST - DO NOT COMMIT

        # buf.compute_returns_and_advantage_pt(values, torch.Tensor(buf.dones[-1]).to(self.device))
        # self.rollout_buffer.advantages = torch.zeros_like(self.rollout_buffer.advantages)
        # self.rollout_buffer.flat_advantages = buf.swap_and_flatten(buf.advantages)
        # self.rollout_buffer.advantages = self.rollout_buffer.swap_and_flatten_pt(advantages)
        # self.rollout_buffer.flat_advantages =
        env_indices = np.random.permutation(self.rollout_buffer.buffer_size * self.n_envs)
        # buffer = self.rollout_buffer.flatten()
        for epoch in range(self.n_epochs):
            # if epoch == 0:
            #    self.rollout_buffer.advantages = buf.advantages
            # else:
            #    self.rollout_buffer.advantages = buf.swap_and_flatten_pt(buf.advantages)
            # self.rollout_buffer.advantages = buf.swap_and_flatten_pt(buf.advantages)
            # torch.autograd.set_detect_anomaly(True)
            approx_kl_divs = []
            # self.rollout_buffer.advantages = buf.advantages
            # Do a complete pass on the rollout buffer
            count = 0
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # start_idx = epoch * self.batch_size
                # selection = env_indices[start_idx: start_idx + self.batch_size]
                # rollout_data = buffer.sample(selection)
                # torch.autograd.set_detect_anomaly(True)
                # self.train_loop(rollout_data, clip_range, pg_losses, clip_fractions, None,value_losses,buf, entropy_losses,approx_kl_divs)

                # torch.autograd.set_detect_anomaly(True)
                # self.normalize_advantage = False
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
                # _, _, values_pred, _, _ = self.policy(torch.Tensor(rollout_data.observations).to(self.device))
                value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values)
                value_losses.append(value_loss.item())

                head_indices = (0, 1, 2, 3, 12, 13)
                extractor_indices = np.setdiff1d(range(len(self.policy.value_optimizer.param_groups[0]['params'])),
                                                 head_indices)

                value_head_params = [self.policy.value_optimizer.param_groups[0]['params'][idx] for idx in head_indices]
                value_cnn_extractor_params = [self.policy.value_optimizer.param_groups[0]['params'][idx] for idx in
                                              extractor_indices]

                ctrl_head_params = [self.policy.ctrl_optimizer.param_groups[0]['params'][idx] for idx in head_indices]
                ctrl_cnn_extractor_params = [self.policy.ctrl_optimizer.param_groups[0]['params'][idx] for idx in
                                             extractor_indices]

                dstb_head_params = [self.policy.dstb_optimizer.param_groups[0]['params'][idx] for idx in head_indices]
                dstb_cnn_extractor_params = [self.policy.dstb_optimizer.param_groups[0]['params'][idx] for idx in
                                             extractor_indices]

                L_ctrl_grad_batched = autograd.grad(value_loss, value_head_params,
                                                    create_graph=True, retain_graph=True)
                L_ctrl_grad = torch.cat([t.flatten() for t in L_ctrl_grad_batched], dim=0)
                # L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])

                # BLENDING TIME

                full_hessian = False
                # ee = [x for xs in value_head_params for x in xs]
                n = sum(p.numel() for p in value_head_params)
                if full_hessian is False:

                    k = 50
                    # n = sum(p.numel() for p in self.policy.value_optimizer.param_groups[0]['params'])

                    rademacher = torch.bernoulli(torch.from_numpy(np.ones((n, k)) * .5)).to(self.device)
                    rademacher[rademacher == 0] = -1
                    # grad_batched = autograd.grad(L_ctrl_grad, flat_params, rademacher,0,1, is_grads_batched=True)
                    grad_batched = autograd.grad(L_ctrl_grad, value_head_params,
                                                 torch.transpose(rademacher.to(self.device), 0, 1),
                                                 is_grads_batched=True,
                                                 retain_graph=True, create_graph=True)

                else:
                    # import torch
                    '''
                    def compute_jacobian_batched(x, params, batch_size=128):
                        N = x.shape[0]
                        param_vec_len = sum(p.numel() for p in params)

                        jacobian = torch.zeros(N, param_vec_len, device=x.device, dtype=x.dtype)

                        for start in range(0, N, batch_size):
                            end = min(start + batch_size, N)
                            batch_indices = torch.arange(start, end, device=x.device)

                            batch_x = x[batch_indices]  # shape (batch_size,)

                            grads = []
                            for i in range(batch_x.shape[0]):
                                # Compute grad of x[i] w.r.t params
                                grad_outputs = torch.zeros_like(batch_x)
                                grad_outputs[i] = 1.0  # select only the i-th output
                                g = torch.autograd.grad(
                                    batch_x, params, grad_outputs=grad_outputs,
                                    retain_graph=True, create_graph=False
                                )
                                grads.append(torch.cat([gg.flatten() for gg in g]))

                            grads = torch.stack(grads, dim=0)  # shape (batch_size, param_vec_len)
                            jacobian[start:end] = grads

                        return jacobian

                    # Now use vmap over all elements of x
                    jacobian = compute_jacobian_batched(L_ctrl_grad, self.policy.value_optimizer.param_groups[0]['params'])
                    print("eee")
                    '''
                    grad_batched = autograd.grad(L_ctrl_grad, value_head_params,
                                                 torch.eye(n).to(self.device),
                                                 is_grads_batched=True,
                                                 retain_graph=True, create_graph=True)
                if full_hessian is False:
                    reshaped_grads = self.matrix_unbatch(grad_batched, k, size2=n).T
                    reshaped_grads = reshaped_grads * rademacher
                    L_ctrl_hessian = torch.mean(reshaped_grads, dim=1)
                    L_ctrl_hessian = L_ctrl_hessian + 1
                else:
                    L_ctrl_hessian = self.matrix_unbatch(grad_batched, n)
                    L_ctrl_hessian.diag().add_(1)
                d2f1_ctrl_batched = autograd.grad(policy_loss, value_head_params, create_graph=True, retain_graph=True)
                d2f1_dstb_batched = autograd.grad(dstb_policy_loss, value_head_params, create_graph=True,
                                                  retain_graph=True)
                d2f1_ctrl = torch.hstack([t.flatten() for t in d2f1_ctrl_batched])

                d2f1_dstb = torch.hstack([t.flatten() for t in d2f1_dstb_batched])
                # d2f1_ctrl = torch.rand(d2f1_dstb.shape).to(self.device)

                # diag, no other option
                iHvp_ctrl = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_ctrl)
                iHvp_dstb = torch.mul(torch.pow(L_ctrl_hessian, -1), d2f1_dstb)
                # assert not np.any(self.rollout_buffer.current_shot - self.rollout_buffer.indices[
                #                                                     count * self.batch_size: count * self.batch_size + self.batch_size])
                # assert self.rollout_buffer.current_shot == self.rollout_buffer.indices[count * self.batch_size: count * self.batch_size + self.batch_size]
                traj_ids = self.rollout_buffer.env_indices[self.rollout_buffer.indices[
                                                           count * self.batch_size: count * self.batch_size + self.batch_size]].squeeze()
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
                surr_L_ctrl, surr_L_dstb = self.estimators(advantages, ctrl_log_prob, dstb_log_prob,
                                                           x0_values.squeeze(),
                                                           torch.Tensor(x0_returns).to(self.device))
                d1f2_ctrl_batched = autograd.grad(surr_L_ctrl, value_head_params,
                                                  create_graph=True, retain_graph=True)
                d1f2_ctrl = torch.hstack([t.flatten() for t in d1f2_ctrl_batched])
                d1f2_dstb_batched = autograd.grad(surr_L_dstb, value_head_params,
                                                  create_graph=True, retain_graph=True)
                d1f2_dstb = torch.hstack([u.flatten() for u in d1f2_dstb_batched])
                # d1f2_dstb = d1f2_dstb.dot(dstb_log_prob)
                # ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.value_optimizer.param_groups[0]['params'], torch.eye(d1f2_ctrl.shape[0], device=self.device), is_grads_batched=True, create_graph=True, retain_graph=True)
                ctrl_imp = autograd.grad(d1f2_ctrl, ctrl_head_params, iHvp_ctrl, is_grads_batched=False,
                                         create_graph=True, retain_graph=True)
                dstb_imp = autograd.grad(d1f2_dstb, dstb_head_params, iHvp_dstb, is_grads_batched=False,
                                         create_graph=True, retain_graph=True)

                # Entropy loss favor exploration
                if ctrl_entropy is None:
                    # Approximate entropy when no analytical form
                    ctrl_entropy_loss = -th.mean(-ctrl_log_prob)
                    dstb_entropy_loss = -th.mean(-dstb_log_prob)
                else:
                    ctrl_entropy_loss = -th.mean(ctrl_entropy)
                    dstb_entropy_loss = -th.mean(dstb_entropy)

                entropy_losses.append(ctrl_entropy_loss.item())
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
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
                # big_loss = ctrl_loss + dstb_loss + critic_loss
                self.policy.ctrl_optimizer.zero_grad()
                self.policy.dstb_optimizer.zero_grad()
                self.policy.value_optimizer.zero_grad()
                ctrl_tensors = autograd.grad(ctrl_loss, self.policy.ctrl_optimizer.param_groups[0]['params'])

                # ctrl_head_grads = autograd.grad(ctrl_loss, ctrl_head_params)
                # ctrl_cnn_extractor_grads = autograd.grad(ctrl_loss, ctrl_cnn_extractor_params)

                dstb_tensors = autograd.grad(dstb_loss, self.policy.dstb_optimizer.param_groups[0]['params'])

                # dstb_head_grads = autograd.grad(dstb_loss, dstb_head_params)
                # dstb_cnn_extractor_grads = autograd.grad(dstb_loss, dstb_cnn_extractor_params)

                value_tensors = autograd.grad(critic_loss, self.policy.value_optimizer.param_groups[0][
                    'params'])  # , create_graph=True, retain_graph=True)

                # value_head_grads = autograd.grad(value_loss, value_head_params)
                # value_cnn_extractor_grads = autograd.grad(value_loss, value_cnn_extractor_params)

                for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                    self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad = ctrl_tensors[i]
                    self.policy.dstb_optimizer.param_groups[0]['params'][i].grad = dstb_tensors[i]
                th.nn.utils.clip_grad_norm_(self.policy.ctrl_optimizer.param_groups[0]['params'], self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(self.policy.dstb_optimizer.param_groups[0]['params'], self.max_grad_norm)
                for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                    self.policy.value_optimizer.param_groups[0]['params'][i].grad = value_tensors[i]
                th.nn.utils.clip_grad_norm_(self.policy.value_optimizer.param_groups[0]['params'], self.max_grad_norm)
                # big_loss.backward()
                print("e done")
                # ctrl_loss.backward(retain_graph=True)

                with (torch.no_grad()):
                    # ctrl_partials = autograd.grad(ctrl_loss, self.policy.ctrl_optimizer.param_groups[0]['params'])
                    for i in range(len(ctrl_head_params)):
                        self.policy.ctrl_optimizer.param_groups[0]['params'][head_indices[i]].grad = \
                            self.policy.ctrl_optimizer.param_groups[0]['params'][head_indices[i]].grad - ctrl_imp[i]
                    # th.nn.utils.clip_grad_norm_(self.policy.ctrl_optimizer.param_groups[0]['params'],
                    #                            self.max_grad_norm)

                    for i in range(len(dstb_head_params)):
                        self.policy.dstb_optimizer.param_groups[0]['params'][head_indices[i]].grad = \
                            self.policy.dstb_optimizer.param_groups[0]['params'][head_indices[i]].grad - dstb_imp[i]
                    # th.nn.utils.clip_grad_norm_(self.policy.dstb_optimizer.param_groups[0]['params'],
                    #                            self.max_grad_norm)
                    # th.nn.utils.clip_grad_norm_(self.policy.value_optimizer.param_groups[0]['params'],
                    #                            self.max_grad_norm)

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
                # self.policy.dstb_optimizer.zero_grad()
                # dstb_partials = autograd.grad(dstb_loss, self.policy.dstb_optimizer.param_groups[0]['params'])
                # dstb_loss.backward(retain_graph=True)
                # self.policy.dstb_optimizer.step()
                # values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                #    torch.from_numpy(rollout_data.observations).to(self.device), actions, dstb_actions)
                # values_pred = values.flatten()
                # value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred)
                # critic_loss = self.vf_coef * value_loss
                # self.policy.value_optimizer.zero_grad()
                '''
                critic_partials = autograd.grad(critic_loss, self.policy.value_optimizer.param_groups[0]['params'])
                for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                    self.policy.value_optimizer.param_groups[0]['params'][i].grad = critic_partials[i]'''
                # critic_loss.backward(retain_graph=True)

                # loss.backward()
                # Clip grad norm
                # self.policy.value_optimizer.step()
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

                # buf = deepcopy(self.rollout_buffer)
                # buf.values = torch.from_numpy(self.rollout_buffer.values).to(self.device)
                # buf.rewards = torch.from_numpy(buf.rewards).to(self.device)
                # buf.advantages = torch.from_numpy(buf.advantages).to(self.device)
                # buf.episode_starts = torch.from_numpy(buf.episode_starts).to(self.device)

                # for i in range(buf.buffer_size):
                #    _, _, values, _, _ = self.policy(torch.Tensor(buf.observations[i]).to(self.device))
                #    buf.values[i] = values.squeeze()
                # _, _, last_values, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))

                # TEST - DO NOT COMMIT

                advantage_test = []
                _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[-1]).to(self.device))
                last_values = vf.flatten()
                last_gae_lam = th.zeros_like(last_values)
                dones = torch.Tensor(buf.dones[-1]).to(self.device)
                for step in reversed(range(buf.buffer_size)):
                    # _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))
                    if step == buf.buffer_size - 1:
                        next_non_terminal = 1.0 - dones.float()
                        next_values = last_values
                    else:
                        next_non_terminal = 1.0 - buf.episode_starts[step + 1].float()
                        _, _, vf, _, _ = self.policy(torch.Tensor(buf.observations[step + 1]).to(self.device))
                        next_values = vf.flatten()
                    _, _, value_query, _, _ = self.policy(torch.Tensor(buf.observations[step]).to(self.device))

                    delta = buf.rewards[step] + buf.gamma * next_values * next_non_terminal - value_query.squeeze()
                    last_gae_lam = delta + buf.gamma * buf.gae_lambda * next_non_terminal * last_gae_lam
                    advantage_test.append(last_gae_lam)
                    # buf.advantages[step] = last_gae_lam
                advantages = torch.stack(advantage_test, dim=0)
                # buf.returns = buf.advantages + buf.values
                print("")
                # TEST - DO NOT COMMIT

                # buf.compute_returns_and_advantage_pt(values, torch.Tensor(buf.dones[-1]).to(self.device))
                # self.rollout_buffer.advantages = torch.zeros_like(self.rollout_buffer.advantages)
                # self.rollout_buffer.flat_advantages = buf.swap_and_flatten(buf.advantages)
                self.rollout_buffer.advantages = self.rollout_buffer.swap_and_flatten_pt(advantages)
                count = count + 1
                # elf.rollout_buffer.flat_advantages = self.rollout_buffer.swap_and_flatten_pt(self.rollout_buffer.advantages)
                # self.rollout_buffer.flat_advantages = buf.swap_and_flatten_pt(buf.advantages)

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

class Exploiter(PPO):
    def __init__(self,
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
        exploited: Union[Specialized_Agent, Specialized_Agent_IPPO]=None,
        exploiting = None
    ):

        super().__init__(policy=policy,
        env=env,
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
        _init_setup_model=_init_setup_model)

        #assert exploited is not None
        assert exploiting == "ego" or exploiting == "adv"
        self.exploiting = exploiting
        self.exploited = exploited

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
                actions, values, log_probs, = self.policy(obs_tensor)
                ego_id = self.exploited.opp_list.index(self.exploited.player)
                if self.exploiting == "ego":
                    EXPLOITED_ACTIONS, _, _, _, _ = self.exploited.policy(obs_tensor, network_keys=[ego_id])
                else:
                    _, _, _, EXPLOITED_ACTIONS, _ = self.exploited.policy(obs_tensor, network_keys=[ego_id])

            actions = actions.cpu().numpy()
            EXPLOITED_ACTIONS = EXPLOITED_ACTIONS.cpu().numpy()
            #dstb_actions = dstb_actions.cpu().numpy()
            # Rescale and perform action
            if self.exploiting == "ego":
                clipped_actions = np.hstack([EXPLOITED_ACTIONS, actions])
            else:
                clipped_actions = np.hstack([actions, EXPLOITED_ACTIONS])
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, rew_other, dones, infos = env.step(clipped_actions)
            #print(rewards)
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
            if self.exploiting == "ego":
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rew_other,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs
                    )
            else:
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs
                    )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True