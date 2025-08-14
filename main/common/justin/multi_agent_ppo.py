import torch
import torch as th
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, safe_mean
from torch.nn import functional as F
from gym import spaces

from .Generalist_SPAR import Generalist_SPAR

class TwoSidedMultiAgentPPO(Generalist_SPAR):
    """
    A multi-agent PPO that trains a single agent with two distinct policies
    for playing on the 'left' and 'right' sides of the game.
    
    It uses the multi-opponent data collection from Generalist_SPAR and a 
    standard PPO update rule.
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        side: str, # 'left' or 'right'
        c_learning_rate: Union[float, Schedule] = 1e-4,
        d_learning_rate: Union[float, Schedule] = 7e-4,
        v_learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
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
        **kwargs,
    ):
        # We pass a placeholder learning_rate; the real ones are defined in the policy's optimizer
        super(Generalist_SPAR, self).__init__(
            policy=policy,
            env=env,
            learning_rate=c_learning_rate,
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
            _init_setup_model=False, # We will setup model manually
        )
        
        self.side = side
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = get_schedule_fn(clip_range)
        self.clip_range_vf = get_schedule_fn(clip_range_vf) if clip_range_vf is not None else None
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.dstb_ent_coef = dstb_ent_coef

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """
        Create the policy networks and optimizers.
        This agent has two separate policies, one for each side.
        """
        # Create 'left' policy
        self.policy_left = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy_left.to(self.device)

        # Create 'right' policy
        self.policy_other = self.policy_class( # Naming to match IPPO
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy_other.to(self.device)

        # The active policy is determined by the side this agent is playing on
        if self.side == 'left':
            self.policy = self.policy_left
        elif self.side == 'right':
            self.policy = self.policy_other
        else:
            raise ValueError(f"Invalid side '{self.side}'. Must be 'left' or 'right'.")

    def train(self) -> None:
        """
        Update the active policy using the currently gathered rollout buffer.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        entropy_losses, pg_losses, value_losses = [], [], []
        clip_fractions = []
        
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                # Use the network_keys from the buffer to route to the correct head
                network_keys = rollout_data.network_keys

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, 
                    actions,
                    network_keys=network_keys
                )
                values = values.flatten()
                
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(self, **kwargs):
        # Use the learn method from the grandparent class to avoid SPAR logic
        return super(Generalist_SPAR, self).learn(**kwargs) 