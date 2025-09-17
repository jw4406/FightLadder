import torch
import torch as th
import torch.autograd as autograd
import sys
import time
import random
from functools import partial
from venv import create
import wandb
import itertools
import numpy as np
import torch.nn as nn
from anyio import value
from gym import spaces
import warnings
from stable_baselines3.common.preprocessing import preprocess_obs
from .policies import BasePolicy, SelectLastLSTMOutput
from .distributions import BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution, MultiCategoricalDistribution, StateDependentNoiseDistribution, make_proba_distribution
from .preprocessing import maybe_transpose
from .type_aliases import GymEnv, MaybeCallback, Schedule
from .utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn, \
    update_learning_rate, is_vectorized_observation
from .save_util import load_from_zip_file, recursive_getattr, recursive_setattr, \
    save_to_zip_file
from .vec_env import VecEnv
from .distributions import Distribution

from .buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer
from .callbacks import BaseCallback
from .noise import ActionNoise
from .policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from typing import Union, Type, Optional, Dict, Any, List, Tuple
#from stable_baselines3.common.clean_new_policies import CleanActorActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractorAdv, NatureCNN
from utils import select_matchup_env

class CleanActorActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        # TODO(antonin): update type annotation when we remove shared network support
        net_arch: Union[List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        matchups=None,
        envs_per_matchup=None,
        num_adversaries=None,
        dstb_action_space=None,
    ):
        # self.matchups = matchups
        # self.envs_per_matchup = envs_per_matchup
        # self.num_adversaries = num_adversaries
        # self.dstb_action_space = dstb_action_space
        # self.use_sde = use_sde
        # self.dist_kwargs = None
        # self.f
        # if dstb_action_space is None:
        #     self.dstb_action_space = action_space
        #     dstb_action_space = action_space
        # self.dstb_action_dist = [make_proba_distribution(self.dstb_action_space, use_sde=self.use_sde, dist_kwargs=None) for i in range(self.num_adversaries)]
        # self.pi_dstb_features_extractor = self.make_features_extractor()
        super().__init__(observation_space = observation_space,
        action_space = action_space,
        lr_schedule = lr_schedule[0],
        # TODO(antonin): update type annotation when we remove shared network support
        net_arch = net_arch,
        activation_fn = activation_fn,
        ortho_init = ortho_init,
        use_sde = use_sde,
        log_std_init = log_std_init,
        full_std = full_std,
        use_expln = use_expln,
        squash_output = squash_output,
        features_extractor_class = features_extractor_class,
        features_extractor_kwargs = features_extractor_kwargs,
        share_features_extractor = share_features_extractor,
        normalize_images = normalize_images,
        optimizer_class = optimizer_class,
        optimizer_kwargs = optimizer_kwargs,
    )
        self.dstb_action_space = dstb_action_space
        if dstb_action_space is None:
            self.dstb_action_space = action_space
            dstb_action_space = action_space
        # self.use_sde = use_sde
        # self.dist_kwargs = None
        self.num_adversaries = num_adversaries
        self.matchups = matchups
        self.envs_per_matchup = envs_per_matchup
        self.dstb_action_dist = [make_proba_distribution(self.dstb_action_space, use_sde=self.use_sde, dist_kwargs=None) for i in range(self.num_adversaries)]
        self.pi_dstb_features_extractor = self.make_features_extractor()
        self.pi_ctrl_features_extractor = self.features_extractor
        #self.vf_features
        self._build_network(lr_schedule)

        print("hello")
    def _build_mlp_extractor(self, extra=False) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractorAdv(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device='auto',
            adversarial=True,
            context_dim=0
        ) 
    def _build_network(self, joint_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        lstm_hidden_size = 256
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
            if self.adversarial: # we are doing adversarial!
                self.dstb_action_net, self.dstb_log_std = self.dstb_action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            #self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

            self.action_net = nn.Sequential(
                nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                SelectLastLSTMOutput(),
                nn.Linear(lstm_hidden_size, lstm_hidden_size),
                self.activation_fn(),
                nn.Linear(lstm_hidden_size, latent_dim_pi),
                self.activation_fn(),
                self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
            )

            self.dstb_action_net = nn.ModuleDict()
            self.head_length = 10 # lstm = 4, 2 linear layers = 2 + 2, proba_dist is also a linear, so 2, total 10
            for i in range(self.num_adversaries):
                matchup_key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
                self.dstb_action_net[matchup_key] = nn.Sequential(
                    nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                    SelectLastLSTMOutput(),
                    nn.Linear(lstm_hidden_size, lstm_hidden_size),
                    self.activation_fn(),
                    nn.Linear(lstm_hidden_size, latent_dim_pi),
                    self.activation_fn(),
                    self.dstb_action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi))
                    
                if i == 0:
                    assert len(next(iter(self.dstb_action_net.values()))) == 7 and self.head_length == 10

                #self.dstb_action_net.append(self.dstb_action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi))
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        
        self.value_net = nn.ModuleDict()
        for i in range(self.num_adversaries):
            matchup_key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
            self.value_net[matchup_key] = nn.Sequential(
                nn.LSTM(input_size=self.mlp_extractor.latent_dim_vf, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                SelectLastLSTMOutput(),
                nn.Linear(lstm_hidden_size, lstm_hidden_size),
                self.activation_fn(),
                nn.Linear(lstm_hidden_size, lstm_hidden_size),
                self.activation_fn(),
                nn.Linear(lstm_hidden_size, 1))
            #self.value_net.append(nn.Linear(self.mlp_extractor.latent_dim_vf, 1))
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.dstb_action_net: 0.01,
                self.value_net: 1,
            }
            
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        
        if len(self.mlp_extractor.policy_net) == 0: # we're using naturecnn for images
            self.ctrl_optimizer = self.optimizer_class(
                itertools.chain(self.pi_ctrl_features_extractor.parameters(), self.action_net.parameters()),
                joint_schedule[1](1), maximize=False)
            self.dstb_optimizer = self.optimizer_class(
                itertools.chain(self.pi_dstb_features_extractor.parameters(), self.dstb_action_net.parameters()),
                joint_schedule[2](1), maximize=False)
            self.value_optimizer = self.optimizer_class(
                itertools.chain(self.vf_features_extractor.parameters(), self.value_net.parameters()),
                joint_schedule[0](1), **self.optimizer_kwargs)
        else:
            self.ctrl_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.policy_net.parameters(), self.pi_ctrl_features_extractor.parameters(),self.action_net.parameters()), joint_schedule[0](1),maximize=True)
            self.dstb_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.dstb_net.parameters(), self.pi_dstb_features_extractor.parameters(), self.dstb_action_net.parameters()), joint_schedule[1](1), maximize=False)
            self.extractor_and_trunk_length = 12
            assert self.extractor_and_trunk_length == 12 and len(self.mlp_extractor.dstb_net) + len(self.pi_dstb_features_extractor.cnn) + len(self.pi_dstb_features_extractor.linear) == 13
            #self.value_optimizer = self.optimizer_class(
            #    itertools.chain(self.mlp_extractor.value_net.parameters(), self.vf_features_extractor.parameters(), itertools.chain.from_iterable([self.value_net[i].parameters() for i in range(self.num_adversaries)])),
            #    joint_schedule[2](1), **self.optimizer_kwargs)
            self.value_optimizer = self.optimizer_class(
                itertools.chain(self.mlp_extractor.value_net.parameters(), self.vf_features_extractor.parameters(), self.value_net.parameters()),
                joint_schedule[2](1), **self.optimizer_kwargs)
            #self.value_targ = [copy.deepcopy(self.vf_features_extractor).requires_grad_(False).to('cuda'),
            #                   copy.deepcopy(self.mlp_extractor.value_net).requires_grad_(False).to('cuda'),
            #                   [copy.deepcopy(self.value_net)[i].requires_grad_(False).to('cuda') for i in range(len(self.value_net))]]

    def _get_ego_action_dist_from_latent(self, latent_pi) -> Tuple[Distribution, Distribution]:
        mean_actions = self.action_net(latent_pi)
        
        if isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        raise ValueError("Invalid action distribution")

    def _get_adv_action_dist_from_latent(self, latent_pi_dstb) -> Tuple[Distribution, Distribution]:
        dstb_actions = th.zeros_like(latent_pi_dstb)
        num_env_per_adv = self.num_env_per_adv
        for i in range(self.num_adversaries):
            key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
            dstb_actions[i * num_env_per_adv : (i+1) * num_env_per_adv, :] = self.dstb_action_net[key](latent_pi_dstb[i * num_env_per_adv : (i+1) * num_env_per_adv, :])
        return dstb_actions

    def ego_forward(self, obs, deterministic=False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        new_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        pi_ctrl_features = self.pi_ctrl_features_extractor(new_obs)
        latent_pi = self.mlp_extractor.ego_forward(pi_ctrl_features)
        ctrl_distribution = self._get_ego_action_dist_from_latent(latent_pi)
        ctrl_actions = ctrl_distribution.get_actions(deterministic=deterministic)
        ctrl_log_prob = ctrl_distribution.log_prob(ctrl_actions)
        return ctrl_actions, ctrl_log_prob

    def adv_forward(self, obs, deterministic=False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        new_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        pi_dstb_features = self.pi_dstb_features_extractor(new_obs)
        latent_pi_dstb = self.mlp_extractor.adv_forward(pi_dstb_features)
        dstb_distribution = self._get_adv_action_dist_from_latent(latent_pi_dstb)
        dstb_actions = dstb_distribution.get_actions(deterministic=deterministic)
        dstb_log_prob = dstb_distribution.log_prob(dstb_actions)
        return dstb_actions, dstb_log_prob

    def value_forward(self, obs) -> Tuple[th.Tensor, th.Tensor]:
        new_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        vf_features = self.vf_features_extractor(new_obs)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        for i in range(self.num_adversaries):
            # NOT DONE YET! NEED TO CHUNK
            key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
            values = self.value_net[key](latent_vf)
        return values

    def forward(self, obs, deterministic=False, ego_forward=False, adv_forward=False, network_keys=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        if ego_forward:
            ego_actions, ego_log_prob = self.ego_forward(obs, deterministic)
        else:
            ego_actions = th.zeros()
            ego_log_prob = th.zeros()
            #ego_entropy = th.zeros()
        if adv_forward:
            adv_actions, adv_log_prob = self.adv_forward(obs, deterministic, network_keys=network_keys, envs_per_matchup=envs_per_matchup, shuffle_keys=shuffle_keys)
        else:
            adv_actions = th.zeros_like(ego_actions)
            adv_log_prob = th.zeros_like(ego_log_prob)
            #adv_entropy = th.zeros()
        

        values = self.value_forward(obs)
        return ego_actions, ego_log_prob, adv_actions, adv_log_prob, values