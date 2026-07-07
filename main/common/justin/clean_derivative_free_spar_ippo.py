import time
from typing import Dict, Type

import numpy as np
import torch as th
from PIL import Image
from stable_baselines3.common.buffers import Q_RolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.clean_new_policies_ippo import CleanIPPOActorActorCriticPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from common.justin.clean_derivative_free_spar import (
    _BoxTypes,
    _DiscreteTypes,
    CleanDerivativeFreeSPAR,
)


class CleanDerivativeFreeSPARIPPO(CleanDerivativeFreeSPAR):
    """
    Minimal IPPO variant:
    - ego gets a dedicated value head
    - ego rollout buffer stores ego values
    - ego training evaluates ego-specific values
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "AACCnnPolicy": CleanIPPOActorActorCriticPolicy
    }

    def _setup_model(self) -> None:
        super()._setup_model()
        if isinstance(self.observation_space, spaces.Dict):
            return

    def collect_rollouts_standard(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        adversary_buffers,
        n_rollout_steps: int,
        run_ego_forward: bool = True,
        run_adv_forward: bool = True,
        zero_ego_action: bool = False,
        zero_adv_action: bool = False,
        random_ego_action: bool = False,
        random_adv_action: bool = False,
    ) -> bool:
        _ = time.time()
        _ = [Image.fromarray(env.render(mode="rgb_array"))]
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        ego_entropy_sum = 0.0
        adv_entropy_sum = 0.0
        entropy_count = 0
        for i in range(self.num_adversaries):
            adversary_buffers[i].reset()
        rollout_terminal_stats = [
            {"wins": 0, "losses": 0, "draws": 0, "games": 0}
            for _ in range(self.num_adversaries)
        ]
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                ego_actions, ego_log_probs, adv_actions, adv_log_probs, values, q_values = self.policy(
                    obs_tensor,
                    deterministic=False,
                    ego_forward=run_ego_forward,
                    adv_forward=run_adv_forward,
                    zero_ego_action=zero_ego_action,
                    zero_adv_action=zero_adv_action,
                )
                ego_entropy_sum += float((-ego_log_probs.detach()).mean().item())
                adv_entropy_sum += float((-adv_log_probs.detach()).mean().item())
                entropy_count += 1
                ego_values = self.policy.ego_value_forward(obs_tensor)
                other_values = -values

            actions = ego_actions.cpu().numpy()
            actions_other = adv_actions.cpu().numpy()
            left_actions = actions if self.ego_side == "left" else actions_other
            right_actions = actions_other if self.ego_side == "left" else actions

            clipped_actions = np.hstack([left_actions, right_actions])
            if isinstance(self.action_space, _BoxTypes):
                clipped_actions = np.clip(
                    np.hstack([left_actions, right_actions]),
                    self.action_space.low,
                    self.action_space.high,
                )

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)

            if self.ego_side == 'right':
                rewards = rewards_other
                rewards_other = -rewards

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            if self.ego_side == 'right':
                for idx in range(len(infos)):
                    if "episode" in infos[idx]:
                        infos[idx]["episode"]["r"], infos[idx]["episode"]["ro"] = infos[idx]["episode"]["ro"], infos[idx]["episode"]["r"]
                    if "outcome" in infos[idx]:
                        o = infos[idx]["outcome"]
                        infos[idx]["outcome"] = "lose" if o == "win" else ("win" if o == "lose" else o)

            self._update_info_buffer(infos)
            n_steps += 1

            for idx, done in enumerate(dones):
                if not done:
                    continue
                adv_idx = idx // self.n_env_per_adv
                if adv_idx < 0 or adv_idx >= self.num_adversaries:
                    continue
                ego_score = self._ego_score_from_terminal(
                    infos[idx],
                    float(rewards[idx]),
                    float(rewards_other[idx]),
                )
                if ego_score >= 1.0:
                    rollout_terminal_stats[adv_idx]["wins"] += 1
                elif ego_score <= 0.0:
                    rollout_terminal_stats[adv_idx]["losses"] += 1
                else:
                    rollout_terminal_stats[adv_idx]["draws"] += 1
                rollout_terminal_stats[adv_idx]["games"] += 1

            if isinstance(self.action_space, _DiscreteTypes):
                actions = actions.reshape(-1, 1)
                actions_other = actions_other.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs.copy(),
                actions,
                actions_other,
                rewards,
                new_obs,
                dones,
                self._last_episode_starts,
                ego_values,
                ego_log_probs,
                q_values,
            )
            for i in range(self.num_adversaries):
                indices = slice(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
                adversary_buffers[i].add(
                    self._last_obs[indices].copy(),
                    actions[indices],
                    actions_other[indices],
                    rewards_other[indices],
                    new_obs[indices],
                    dones[indices],
                    self._last_episode_starts[indices],
                    other_values[indices],
                    adv_log_probs[indices],
                    -q_values[indices],
                )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            last_adv_values = self.policy.value_forward(obs_as_tensor(new_obs, self.device))
            last_ego_values = self.policy.ego_value_forward(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=last_ego_values, dones=dones)
        for i in range(self.num_adversaries):
            adversary_buffers[i].compute_returns_and_advantage(
                last_values=-last_adv_values[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv],
                dones=dones[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv],
            )

        callback.on_rollout_end()
        rollout_buffer.prepare_data_for_training()
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].prepare_data_for_training()
        if entropy_count > 0:
            self._last_rollout_entropy_ego = ego_entropy_sum / entropy_count
            self._last_rollout_entropy_adv = adv_entropy_sum / entropy_count
            self._last_rollout_policy_entropy = (
                self._last_rollout_entropy_ego + self._last_rollout_entropy_adv
            ) / 2.0
        else:
            self._last_rollout_policy_entropy = None
        n_games = self._update_elo_from_rollout_stats(rollout_terminal_stats)
        if getattr(self, "stagnation_tracker", None) is not None:
            self.stagnation_tracker.register_games(n_games)
        return True

    def train_standard(self, update_ego: bool = True, update_adversary: bool = True) -> None:
        if update_ego and not update_adversary:
            original_evaluate_states = self.policy.evaluate_states

            def _evaluate_ego_values(obs, buf_num=None, env_indices=None, side_flag=None):
                return self.policy.evaluate_ego_values(obs)

            self.policy.evaluate_states = _evaluate_ego_values
            try:
                super().train_standard(update_ego=update_ego, update_adversary=update_adversary)
            finally:
                self.policy.evaluate_states = original_evaluate_states
            return
        super().train_standard(update_ego=update_ego, update_adversary=update_adversary)
