import torch
import torch as th
import torch.autograd as autograd
import sys
import time
import random
from venv import create
import wandb
import warnings
from typing import Union, Type, Optional, Dict, Any
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.clean_new_policies import CleanActorActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer
from utils import state2matchup
from common.justin.Doubly_TSS_SPAR import Doubly_TSS_SPAR as dtss

import numpy as np
import torch.nn as nn
from anyio import value
from gym import spaces
from stable_baselines3 import PPO

class CleanDerivativeFreeSPAR(PPO):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "AACCnnPolicy": CleanActorActorCriticPolicy
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
            dstb_action_space=None,
            matchups=None,
            envs_per_matchup=None,
            state_list=None,
            env_generator_func=None,
            num_adversaries=None,
            n_env_per_adv=None,
    ):

        self.matchups = [state2matchup(state) for state in state_list] #This needs to happen before the super().__init__
        self.envs_per_matchup = envs_per_matchup
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
        self.num_adversaries = num_adversaries
        self.n_env_per_adv = n_env_per_adv
        if _init_setup_model:
            self._setup_model()
        
        adversary_buffers = []
        for i in range(self.num_adversaries):
            overwrite = dtss("AACCnnPolicy",
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
                                       warmstarted_cont_MAGICS=False,
                                       matchups=matchups,
                                       envs_per_matchup=self.envs_per_matchup
                                       )
            overwrite.rollout_buffer.n_envs = self.n_env_per_adv
            adversary_buffers.append(overwrite.rollout_buffer)
        self.adversary_buffers = adversary_buffers
        self.env.num_envs = self.n_envs

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