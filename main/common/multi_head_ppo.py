import sys
import time
import numpy as np
import torch as th
from gym import spaces
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.buffers import AdvRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance
from torch.nn import functional as F
from stable_baselines3.common.vec_env import VecEnv
# It's better to have this import specific to where it's needed.
# Assuming this custom policy is in the same directory or accessible via python path.
from stable_baselines3.common.policies import ActorActorCriticCnnGeneralistPolicy
from .justin.Generalist_SPAR import Generalist_SPAR

SelfPPO_From_SPAR = TypeVar("SelfPPO_From_SPAR", bound="PPO_From_SPAR")

class PPO_From_SPAR(Generalist_SPAR):
    """
    This class provides a standard PPO training algorithm while inheriting the
    multi-headed, multi-buffer agent structure from Generalist_SPAR.
    It is designed for one-vs-many training in a league setting.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom attribute to track timesteps
        self._total_timesteps = 0

    def get_steps(self) -> int:
        """
        Returns the current number of timesteps.
        """
        return self.num_timesteps

    def set_steps(self, num_timesteps: int) -> None:
        """
        Sets the current number of timesteps.
        """
        self.num_timesteps = num_timesteps
    
    def set_active_opponents(self, opponent_policies: List[BasePolicy]):
        """
        Sets the opponent policies for the upcoming training generation.
        This is a critical part of the one-vs-many training setup.
        """
        # The Generalist_SPAR architecture uses the policy network itself 
        # to manage different opponents via network_keys, so we don't need
        # to store the policies here. This method is for API consistency.
        pass

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ):
        self.start_time = time.time()
        if self.ep_info_buffer is None or reset_num_timesteps:
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._n_updates = 0
        else:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling learn() multiple times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        callback = self._init_callback(callback, progress_bar)
        callback.on_training_start(locals(), globals())
        return total_timesteps, callback

    def learn(
        self: SelfPPO_From_SPAR,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO_From_SPAR",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO_From_SPAR:
        
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.adversary_buffers[0], self.adversary_buffers, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            self.train()

        callback.on_training_end()

        return self

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffers with a standard
        PPO loss function.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.policy.ctrl_optimizer, self.policy.dstb_optimizer, self.policy.value_optimizer])

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        for i in range(self.num_adversaries):
            pg_losses, value_losses, entropy_losses = [], [], []
            approx_kl_divs, clip_fractions = [], []

            for epoch in range(self.n_epochs):
                for rollout_data in self.adversary_buffers[i].get(self.batch_size):
                    actions = th.as_tensor(rollout_data.actions, device=self.device)
                    dstb_actions = th.as_tensor(rollout_data.dstb_actions, device=self.device)

                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, 
                        actions,
                        dstb_actions=dstb_actions,
                        network_keys=[i]
                    )

                    values = values.flatten()
                    advantages = th.as_tensor(rollout_data.advantages, device=self.device)
                    if self.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    old_log_prob = th.as_tensor(rollout_data.old_log_prob, device=self.device)
                    ratio = th.exp(log_prob - old_log_prob)

                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        old_values = th.as_tensor(rollout_data.old_values, device=self.device)
                        values_pred = old_values + th.clamp(
                            values - old_values, -clip_range_vf, clip_range_vf
                        )
                    
                    returns = th.as_tensor(rollout_data.returns, device=self.device)
                    value_loss = F.mse_loss(returns, values_pred)
                    value_losses.append(value_loss.item())

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    
                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Since Generalist_SPAR has separate optimizers, we calculate gradients for each part
                    # Note: This assumes the generalist policy computes gradients for all parts.
                    # This might need adjustment based on the policy's `evaluate_actions` implementation.
                    
                    # For a simple PPO, we'd have one optimizer.
                    # Here we mimic that by applying the loss to all relevant components.
                    # This is a simplification; a true PPO would likely have a single optimizer over all params.
                    self.policy.ctrl_optimizer.zero_grad()
                    self.policy.dstb_optimizer.zero_grad()
                    self.policy.value_optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.ctrl_optimizer.step()
                    self.policy.dstb_optimizer.step()
                    self.policy.value_optimizer.step()

                    with th.no_grad():
                        log_ratio = log_prob - old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
            
            # Logging for each adversary head
            self.logger.record(f"train_adv_{i}/entropy_loss", np.mean(entropy_losses))
            self.logger.record(f"train_adv_{i}/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record(f"train_adv_{i}/value_loss", np.mean(value_losses))
            self.logger.record(f"train_adv_{i}/approx_kl", np.mean(approx_kl_divs))
            self.logger.record(f"train_adv_{i}/clip_fraction", np.mean(clip_fractions))
        
        self._n_updates += self.n_epochs
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard") 