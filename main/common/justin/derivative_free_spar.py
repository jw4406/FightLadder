import torch
import torch as th, time, sys
import numpy as np
from gym import spaces
from copy import deepcopy
from stable_baselines3.common.callbacks import ConvertCallback
from torch.nn import functional as F
from .Generalist_SPAR import Generalist_SPAR
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from typing import Any, Dict, Mapping, Optional, Tuple, Union, Type, List, TypeVar
from stable_baselines3.common.policies import BasePolicy, ActorActorCriticCnnPolicy, ActorActorCriticCnnGeneralistPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
class Derivative_Free_SPAR(Generalist_SPAR):
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
            I_AM_LEFT=True,
            I_AM_RIGHT=False,
            dstb_action_space=None,
            num_adversary=4,
            n_global_env=None,
            n_env_per_adv=None,
            warmstarted_cont_MAGICS=False,
            opp_list=None,
            player=None,
            use_mirror=False
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
            I_AM_LEFT=I_AM_LEFT,
            I_AM_RIGHT=I_AM_RIGHT,
            dstb_action_space=dstb_action_space,
            num_adversary=num_adversary,
            n_global_env=n_global_env,
            n_env_per_adv=n_env_per_adv,
            warmstarted_cont_MAGICS=warmstarted_cont_MAGICS,
            opp_list=opp_list,
            player=player,
            use_mirror=use_mirror
        )

    def copy_constructor(self, retain_callback=False):

        import copy
        from copy import deepcopy

        test = copy.copy(self)
        test.policy = self.policy_class(self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs)
        test.policy.load_state_dict(self.policy.state_dict())
        if hasattr(self, "num_adversaries"):
            for i in range(test.num_adversaries):
                test.policy.value_net[i] = test.policy.value_net[i].to(test.device)
                test.policy.dstb_action_net[i] = test.policy.dstb_action_net[i].to(test.device)
        test.policy.ctrl_optimizer = self.policy.optimizer_class(test.policy.ctrl_optimizer.param_groups[0]['params'], maximize=True)
        test.policy.dstb_optimizer = self.policy.optimizer_class(test.policy.dstb_optimizer.param_groups[0]['params'], maximize=True)
        test.policy.value_optimizer = self.policy.optimizer_class(test.policy.value_optimizer.param_groups[0]['params'])
        test.adversary_buffers = deepcopy(self.adversary_buffers)
        test.rollout_buffer = deepcopy(self.rollout_buffer)
        if retain_callback is True:
            pass
        else:
            test.callback = ConvertCallback(None)
            test.callback.init_callback(test)
        test.policy = test.policy.to(self.device)
        return test

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.inner_loop()
        self.leader_grads(self.rollout_buffer, self.perturbed_buf, self.policy, self.perturbed_agent.policy, ego=True)
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
                ctrl_policy_loss = th.min(policy_loss_1, policy_loss_2).mean()
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

                loss = ctrl_policy_loss - self.ent_coef * ctrl_entropy_loss - self.dstb_ent_coef * dstb_entropy_loss + self.vf_coef * value_loss + dstb_policy_loss

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

    def perturb_params(self, param_list):
        count = 0
        for i in range(len(param_list)):
            count = count + torch.numel(param_list[i])
        delta = 999999
        select = torch.from_numpy(np.random.uniform(low=-1, high=1, size=count)).to(self.device)
        v = delta * select / torch.linalg.norm(select)
        self.delta = delta
        self.v = v
        self.d = count
        count = 0
        with torch.no_grad():
            for p in param_list:
                p.copy_(p + torch.reshape(v[count:count + torch.numel(p)], p.shape).to(self.device))
                count = count + torch.numel(p)
        return
    def env_perturb_params(self):
        buf = deepcopy(self.rollout_buffer)
        buf.reset()
        adv_buf = deepcopy(self.adversary_buffers)
        [adv_buf[i].reset() for i in range(len(adv_buf))]
        self.collect_rollouts(self.env, self.callback, buf, adv_buf, n_rollout_steps=self.n_steps)
        return buf, adv_buf

    def inner_loop(self):
        other_ego = deepcopy(self.policy.ctrl_optimizer.param_groups[0]['params'])
        other_adv = deepcopy(self.policy.dstb_optimizer.param_groups[0]['params'])
        self.perturb_params(other_ego)
        self.perturb_params(other_adv)
        perturbed_agent = self.copy_constructor()
        #other_adv_optimizer = self.policy.optimizer_class(other_adv, maximize=True)
        with torch.no_grad():
            for i in range(len(perturbed_agent.policy.dstb_optimizer.param_groups[0]['params'])):
                perturbed_agent.policy.ctrl_optimizer.param_groups[0]['params'][i].copy_(other_ego[i])
                perturbed_agent.policy.dstb_optimizer.param_groups[0]['params'][i].copy_(other_adv[i])
        perturbed_buf, perturbed_adv_buf = perturbed_agent.env_perturb_params()
        self.perturbed_buf = perturbed_buf
        self.perturbed_adv_buf = perturbed_adv_buf
        for i in range(len(self.adversary_buffers)):
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                for rollout_data in self.adversary_buffers[i].get(self.batch_size):
                    actions = torch.Tensor(rollout_data.actions).to(self.device)
                    dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(self.device)

                    self.policy.num_global_env = self.n_env_per_adv
                    self.policy.num_adv = 1
                    values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions,
                        shuffle_keys=rollout_data.env_indices, network_keys=[i])

                    values = values.flatten()

                    if type(rollout_data.advantages) is np.ndarray:
                        advantages = torch.from_numpy(rollout_data.advantages).to(self.device)
                    else:
                        advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    value_loss = F.mse_loss(torch.Tensor(-rollout_data.returns).to(self.device), values)
                    self.policy.value_optimizer.zero_grad()
                    value_loss.backward()

                    #loss.backward()
                    self.policy.ctrl_optimizer.zero_grad()
                    self.policy.dstb_optimizer.zero_grad()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    # self.policy.ctrl_optimizer.step()
                    # self.policy.dstb_optimizer.step()
                    self.policy.value_optimizer.step()


                for rollout_data in perturbed_adv_buf[i].get(self.batch_size):
                    actions = torch.Tensor(rollout_data.actions).to(self.device)
                    dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(self.device)

                    perturbed_agent.policy.num_global_env = perturbed_agent.n_env_per_adv
                    perturbed_agent.policy.num_adv = 1
                    values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = perturbed_agent.policy.evaluate_actions(
                        torch.Tensor(rollout_data.observations).to(perturbed_agent.device), actions, dstb_actions,
                        shuffle_keys=rollout_data.env_indices, network_keys=[i])

                    values = values.flatten()

                    if type(rollout_data.advantages) is np.ndarray:
                        advantages = torch.from_numpy(rollout_data.advantages).to(perturbed_agent.device)
                    else:
                        advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if perturbed_agent.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    value_loss = F.mse_loss(torch.Tensor(-rollout_data.returns).to(perturbed_agent.device), values)
                    perturbed_agent.policy.value_optimizer.zero_grad()
                    value_loss.backward()

                    #loss.backward()
                    perturbed_agent.policy.ctrl_optimizer.zero_grad()
                    perturbed_agent.policy.dstb_optimizer.zero_grad()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(perturbed_agent.policy.parameters(), perturbed_agent.max_grad_norm)
                    # self.policy.ctrl_optimizer.step()
                    # self.policy.dstb_optimizer.step()
                    perturbed_agent.policy.value_optimizer.step()
        self.perturbed_agent = perturbed_agent

    def leader_grads(self, ori_buf, perturbed_buf, ori_policy, perturbed_policy, ego=True):
        clip_range = self.clip_range(self._current_progress_remaining)
        # F = d/delta * (f(x-hat, g(x-hat, y-hat)) - f(x, g(x,y))) * v
        # need ot run forward pass twice (one for normal network and one for perturbed network)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer

            # ego
            for (ori_rollout_data, perturbed_rollout_data) in zip(ori_buf.get(self.batch_size), perturbed_buf.get(self.batch_size)):
                if ego is False:
                    ori_rollout_data.old_log_prob = ori_rollout_data.old_dstb_log_prob
                ori_actions = torch.Tensor(ori_rollout_data.actions).to(self.device)
                ori_dstb_actions = torch.Tensor(ori_rollout_data.dstb_actions).to(self.device)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, ctrl_entropy, _, _ = ori_policy.evaluate_actions(
                    torch.Tensor(ori_rollout_data.observations).to(self.device), ori_actions, ori_dstb_actions, shuffle_keys=ori_rollout_data.env_indices, network_keys=[i for i in range(self.num_adversaries)])
                values = values.flatten()
                # Normalize advantage
                advantages = torch.from_numpy(ori_rollout_data.advantages).to(self.device)
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - torch.Tensor(ori_rollout_data.old_log_prob).to(self.device))
                #dstb_ratio = th.exp(dstb_log_prob - torch.Tensor(rollout_data.old_dstb_log_prob).to(self.device))

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                #dstb_policy_loss_1 = advantages * dstb_ratio
                #dstb_policy_loss_2 = advantages * th.clamp(dstb_ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = th.min(policy_loss_1, policy_loss_2).mean()
                #dstb_policy_loss = th.min(dstb_policy_loss_1, dstb_policy_loss_2).mean()

                perturbed_actions = torch.Tensor(perturbed_rollout_data.actions).to(self.device)
                perturbed_dstb_actions = torch.Tensor(perturbed_rollout_data.dstb_actions).to(self.device)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, perturbed_log_prob, perturbed_entropy, _, _ = perturbed_policy.evaluate_actions(
                    torch.Tensor(perturbed_rollout_data.observations).to(self.device), perturbed_actions, perturbed_dstb_actions, shuffle_keys=perturbed_rollout_data.env_indices, network_keys=[i for i in range(self.num_adversaries)])
                values = values.flatten()
                # Normalize advantage
                perturbed_advantages = torch.from_numpy(perturbed_rollout_data.advantages).to(self.device)
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(perturbed_advantages) > 1:
                    perturbed_advantages = (perturbed_advantages - perturbed_advantages.mean()) / (perturbed_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                perturbed_ratio = th.exp(perturbed_log_prob - torch.Tensor(perturbed_rollout_data.old_log_prob).to(self.device))
                # dstb_ratio = th.exp(dstb_log_prob - torch.Tensor(rollout_data.old_dstb_log_prob).to(self.device))

                # clipped surrogate loss
                perturbed_policy_loss_1 = perturbed_advantages * ratio
                perturbed_policy_loss_2 = perturbed_advantages * th.clamp(perturbed_ratio, 1 - clip_range, 1 + clip_range)
                # dstb_policy_loss_1 = advantages * dstb_ratio
                # dstb_policy_loss_2 = advantages * th.clamp(dstb_ratio, 1 - clip_range, 1 + clip_range)
                perturbed_policy_loss = th.min(perturbed_policy_loss_1, perturbed_policy_loss_2).mean()
                # dstb_policy_loss = th.min(dstb_policy_loss_1, dstb_policy_loss_2).mean()

                F = self.d / self.delta * (perturbed_policy_loss - policy_loss) * self.v


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
                #value_loss = F.mse_loss(torch.Tensor(rollout_data.returns).to(self.device), values_pred)
                #value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if (ctrl_entropy is None) or (dstb_entropy is None):
                    # Approximate entropy when no analytical form
                    ctrl_entropy_loss = -th.mean(-ctrl_log_prob)
                    #dstb_entropy_loss = -th.mean(-dstb_log_prob)
                else:
                    ctrl_entropy_loss = -th.mean(ctrl_entropy)
                    #dstb_entropy_loss = -th.mean(dstb_entropy)

                entropy_losses.append(ctrl_entropy_loss.item())

                loss = ctrl_policy_loss - self.ent_coef * ctrl_entropy_loss# - self.dstb_ent_coef * dstb_entropy_loss + self.vf_coef * value_loss + dstb_policy_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    ctrl_log_ratio = ctrl_log_prob - torch.from_numpy(rollout_data.old_log_prob).to(self.device)
                    ctrl_approx_kl_div = th.mean((th.exp(ctrl_log_ratio) - 1) - ctrl_log_ratio).cpu().numpy()
                    #dstb_log_ratio = dstb_log_prob - torch.from_numpy(rollout_data.old_dstb_log_prob).to(self.device)
                    #dstb_approx_kl_div = th.mean((th.exp(dstb_log_ratio) - 1) - dstb_log_ratio).cpu().numpy()
                    #approx_kl_divs.append(ctrl_approx_kl_div)

                if self.target_kl is not None and torch.max(ctrl_approx_kl_div,
                                                            dstb_approx_kl_div) > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.ctrl_optimizer.zero_grad()
                #self.policy.dstb_optimizer.zero_grad()
                #self.policy.value_optimizer.zero_grad()
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

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
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

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, n_rollout_steps=self.n_steps)
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

        callback.on_training_end()

        return self