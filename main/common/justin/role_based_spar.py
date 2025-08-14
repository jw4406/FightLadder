import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Type, Union

import torch
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean, explained_variance

from .derivative_free_spar import Derivative_Free_SPAR


class RoleBasedSPAR(Derivative_Free_SPAR):
    """
    An extension of Derivative_Free_SPAR that uses a standard PPO training update
    instead of the derivative-free method. It supports role-based training, allowing
    for selective training of the ego (actor) or adversary (critic/disturber)
    components based on flags.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        envs_per_matchup: int,
        state_len: int,
        env_batch_size: int = 32,
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
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        I_AM_LEFT=True,
        I_AM_RIGHT=False,
        dstb_action_space=None,
        num_adversary=4,
        n_global_env=None,
        n_env_per_adv=1,
        warmstarted_cont_MAGICS=False,
        opp_list=None,
        player=None,
        use_mirror=False,
        env_generator_func=None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            envs_per_matchup=envs_per_matchup,
            state_len=state_len,
            env_batch_size=env_batch_size,
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
            player=player,
            use_mirror=use_mirror,
            env_generator_func=env_generator_func,
        )

    def train_agent(self, rollout_buffer, part_to_train: str, adversary_index: int = None):
        """
        Performs a PPO update on a specific part of the policy.

        :param rollout_buffer: The buffer containing the rollouts.
        :param part_to_train: Which part to train: "ego" or "adversary".
        :param adversary_index: The index of the adversary if training the adversary part.
        """
        self.policy.set_training_mode(True)
        
        # Update optimizer learning rates
        if part_to_train == "ego":
            self._update_learning_rate(self.policy.ctrl_optimizer, self.c_lr_schedule)
        else:
            self._update_learning_rate(self.policy.dstb_optimizer, self.d_lr_schedule)
        self._update_learning_rate(self.policy.value_optimizer, self.v_lr_schedule)

        # Compute schedule for clipping
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses = []
        pg_losses = []
        value_losses = []

        for epoch in range(self.n_epochs):
            for rollout_data in rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                dstb_actions = rollout_data.dstb_actions

                # Forward pass through policy
                values, log_prob, entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    dstb_actions,
                    shuffle_keys=rollout_data.env_indices,
                    network_keys=[adversary_index] if adversary_index is not None else list(range(self.num_adversaries))
                )

                # Value loss
                values = values.flatten()
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())

                # Policy loss
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                if part_to_train == "ego":
                    old_log_prob = rollout_data.old_log_prob
                    current_log_prob = log_prob
                    current_entropy = entropy
                    ent_coef = self.ent_coef
                    policy_optimizer = self.policy.ctrl_optimizer
                else:  # "adversary"
                    old_log_prob = rollout_data.old_dstb_log_prob
                    current_log_prob = dstb_log_prob
                    current_entropy = dstb_entropy
                    ent_coef = self.dstb_ent_coef
                    policy_optimizer = self.policy.dstb_optimizer
                
                ratio = torch.exp(current_log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                # Entropy loss
                entropy_loss = -torch.mean(current_entropy)
                entropy_losses.append(entropy_loss.item())

                # Total loss
                loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                policy_optimizer.zero_grad()
                self.policy.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                policy_optimizer.step()
                self.policy.value_optimizer.step()

        self._n_updates += self.n_epochs
        explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

        # Logs
        self.logger.record(f"train/{part_to_train}_policy_gradient_loss", safe_mean(pg_losses))
        self.logger.record(f"train/{part_to_train}_value_loss", safe_mean(value_losses))
        self.logger.record(f"train/{part_to_train}_entropy_loss", safe_mean(entropy_losses))
        self.logger.record(f"train/{part_to_train}_explained_variance", explained_var)

    def train(self, update_ego: bool = True, update_adversary: bool = True):
        """
        Dispatches training to the appropriate agent components.
        """
        if update_ego:
            self.train_agent(self.rollout_buffer, "ego")

        if update_adversary:
            for i, adv_buffer in enumerate(self.adversary_buffers):
                self.train_agent(adv_buffer, "adversary", adversary_index=i)

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
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = super(Derivative_Free_SPAR, self).collect_rollouts(
                self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps
            )

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.dump(step=self.num_timesteps)

            self.train(update_ego=update_ego, update_adversary=update_adversary)

        callback.on_training_end()

        return self 
    
    def set_steps(self, steps: int) -> None:
        self.num_timesteps = steps
    
    def get_steps(self) -> int:
        return self.num_timesteps