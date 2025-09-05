import torch as th
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
import warnings
#from ../.const import *
#from ../.nash import compute_nash

import itertools

SelfIPPO = TypeVar("SelfIPPO", bound="IPPO")
SelfLeaguePPO = TypeVar("SelfLeaguePPO", bound="LeaguePPO")
MAGICS_PPO = TypeVar("MAGICS_PPO", bound="MAGICS_PPO")

class Single_SPAR(OnPolicyAlgorithm):
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
            normalize_advantage: bool = False,
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
        #super()._setup_model()
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else AdvRolloutBuffer
        self.rollout_buffer_class = buffer_cls
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            dstb_action_space=self.dstb_action_space
        )

        if hasattr(self, "num_adversaries"):
            self.policy_kwargs['num_adversaries'] = self.num_adversaries
            #self.policy_kwargs['num_env_per_adv'] = self.num_env_per_adv

        self.policy_kwargs['matchups'] = self.matchups
        self.policy_kwargs['envs_per_matchup'] = self.envs_per_matchup

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )

        self.policy = self.policy.to(self.device)
        # try:
        #     self.policy.value_net
        #     if hasattr(self, "num_adversaries"):
        #         assert isinstance(self.policy.value_net, nn.ModuleList)
        #     else:
        #         pass
        #     #assert isinstance(self.policy.value_net, nn.ModuleList)
        # except:
        #     assert isinstance(self.policy.ego_value_net, nn.ModuleList)
        # if isinstance(self.policy.value_net, nn.ModuleList):
        #     for i in range(len(self.policy.value_net)):
        #         self.policy.value_net[i] = self.policy.value_net[i].to(self.device)
        #         self.policy.dstb_action_net[i] = self.policy.dstb_action_net[i].to(self.device)
        #for i in range(len(self.policy.value_targ) - 1):
        #    self.policy.value_targ[i] = self.policy.value_targ[i].to(self.device)
        #for i in range(len(self.policy.value_targ[-1])):
        #    self.policy.value_targ[-1][i] = self.policy.value_targ[-1][i].to(self.device)
        # if hasattr(self, "num_adversaries"):
        #     for i in range(self.num_adversaries):
        #         self.policy.value_net[i] = self.policy.value_net[i].to(self.device)
        #         self.policy.dstb_action_net[i] = self.policy.dstb_action_net[i].to(self.device)



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
                # ctrl_imp = autograd.grad(d1f2_ctrl, self.policy.ctrl_optimizer.param_groups[0]['params'], iHvp_ctrl,
                #                         is_grads_batched=False, create_graph=True, retain_graph=True)
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
        unbatched = th.zeros((size1, size2), device=self.device)
        for jac_row_count in range(size1):
            curr = 0
            for count in range(len(to_be_unbatched)):
                unbatched[jac_row_count,
                curr:curr + len(
                    th.flatten(to_be_unbatched[count][jac_row_count, :]))] = th.flatten(
                    to_be_unbatched[count][jac_row_count, :])
                curr = curr + len(th.flatten(to_be_unbatched[count][jac_row_count, :]))
        return unbatched