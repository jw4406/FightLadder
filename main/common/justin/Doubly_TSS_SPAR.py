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
from gym import spaces
from copy import deepcopy
from collections import deque
from functorch import vmap as eepy
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

from .spar import Single_SPAR


class Doubly_TSS_SPAR(Single_SPAR):
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
                 warmstarted_cont_MAGICS=False,
                 matchups=None,
                 envs_per_matchup=None,
                 ):
        self.matchups=matchups
        self.envs_per_matchup=envs_per_matchup
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
        # self.update_ctrl = not self.update_ctrl

    def train_one_adversary(self, main_agent, ma_left=False, ma_right=False):
        # helper function

        """
        Update policy using the currently gathered rollout buffer.
        """
        assert ma_left != ma_right
        if self.warmstarted_cont_MAGICS is True:
            if self.warmstarted_cont_MAGICS is True:
                print("this model is warmstarted! now running magics_ppo training", flush=True)
            return super().train_one_adversary(main_agent, ma_left=ma_left, ma_right=ma_right)

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

                if ma_left is True:
                    # main player is the left player
                    # adversaries are "dstb" role
                    # right now we need to update adversaries
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