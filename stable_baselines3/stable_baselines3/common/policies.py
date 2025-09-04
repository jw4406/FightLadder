"""Policies: abstract base class and concrete implementations."""

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from pyexpat import features
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import itertools

import numpy as np
import torch as th
from gym import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
    MlpExtractorAdv,
    IPPOMlpExtractorAdv
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")


class BaseModel(nn.Module):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[nn.Module] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[th.optim.Optimizer]

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        # Automatically deactivate dtype and bounds checks
        if normalize_images is False and issubclass(features_extractor_class, (NatureCNN, CombinedExtractor)):
            self.features_extractor_kwargs.update(dict(normalized_image=True))

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor()
        net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
        return net_kwargs

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def extract_features(self, obs: th.Tensor, features_extractor: Optional[BaseFeaturesExtractor] = None) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

         :param obs: The observation
         :param features_extractor: The features extractor to use. If it is set to None,
            the features extractor of the policy is used.
         :return: The features
        """
        if features_extractor is None:
            warnings.warn(
                (
                    "When calling extract_features(), you should explicitely pass a features_extractor as parameter. "
                    "This will be mandatory in Stable-Baselines v1.8.0"
                ),
                DeprecationWarning,
            )

        features_extractor = features_extractor or self.features_extractor
        assert features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls: Type[SelfBaseModel], path: str, device: Union[th.device, str] = "auto") -> SelfBaseModel:
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        """
        th.nn.utils.vector_to_parameters(th.as_tensor(vector, dtype=th.float, device=self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env


class BasePolicy(BaseModel, ABC):
    """The base policy object.

    Parameters are mostly the same as `BaseModel`; additions are documented below.

    :param args: positional arguments passed through to `BaseModel`.
    :param kwargs: keyword arguments passed through to `BaseModel`.
    :param squash_output: For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function.
    """

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        if hasattr(self, "smart") and self.smart is True:
            with th.no_grad():
                actions, dstb_actions = self._predict(obs_tensor, deterministic=deterministic)
            # Convert to numpy, and reshape to the original action shape
            actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]
            dstb_actions = dstb_actions.cpu().numpy().reshape((-1, *self.dstb_action_space.shape))
            if isinstance(self.action_space, spaces.Box):
                if self.squash_output:
                    # Rescale to proper domain when using squashing
                    actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
                    dstb_actions = self.unscale_dstb_action(dstb_actions)
                else:
                    # Actions could be on arbitrary scale, so clip the actions to avoid
                    # out of bound error (e.g. if sampling from a Gaussian distribution)
                    actions = np.clip(actions, self.action_space.low,
                                      self.action_space.high)  # type: ignore[assignment, arg-type]
                    dstb_actions = np.clip(dstb_actions, self.dstb_action_space.low, self.dstb_action_space.high)

            # Remove batch dimension if needed
            if not vectorized_env:
                assert isinstance(actions, np.ndarray)
                assert isinstance(dstb_actions, np.ndarray)
                actions = actions.squeeze(axis=0)
                dstb_actions = dstb_actions.squeeze(axis=0)

            return (actions, state), (dstb_actions, state)  # type: ignore[return-value]
        else:
            with th.no_grad():
                actions = self._predict(obs_tensor, deterministic=deterministic)
            # Convert to numpy, and reshape to the original action shape
            actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

            if isinstance(self.action_space, spaces.Box):
                if self.squash_output:
                    # Rescale to proper domain when using squashing
                    actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
                else:
                    # Actions could be on arbitrary scale, so clip the actions to avoid
                    # out of bound error (e.g. if sampling from a Gaussian distribution)
                    actions = np.clip(actions, self.action_space.low,
                                      self.action_space.high)  # type: ignore[assignment, arg-type]

            # Remove batch dimension if needed
            if not vectorized_env:
                assert isinstance(actions, np.ndarray)
                actions = actions.squeeze(axis=0)

            return actions, state  # type: ignore[return-value]

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
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
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        # Convert [dict()] to dict() as shared network are deprecated
        if isinstance(net_arch, list) and len(net_arch) > 0:
            if isinstance(net_arch[0], dict):
                warnings.warn(
                    (
                        "As shared layers in the mlp_extractor are deprecated and will be removed in SB3 v1.8.0, "
                        "you should now pass directly a dictionary and not a list "
                        "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                    ),
                )
                net_arch = net_arch[0]
            else:
                # Note: deprecation warning will be emitted
                # by the MlpExtractor constructor
                pass

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                #net_arch = dict(pi=[16,16], vf=[64,64])
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()
            # if the features extractor is not shared, there cannot be shared layers in the mlp_extractor
            # TODO(antonin): update the check once we change net_arch behavior
            if isinstance(net_arch, list) and len(net_arch) > 0:
                raise ValueError(
                    "Error: if the features extractor is not shared, there cannot be shared layers in the mlp_extractor"
                )

        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
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

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward p ass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the features extractor(s)
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor)
        else:
            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

class ActorActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        ortho_init: bool = False,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        adversarial=True,
        dstb_action_space: spaces.Space = None,
        policy_memory_size: Optional[int] = None,
        num_adversaries=None,
        device='auto'
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam or optimizer_class == th.optim.AdamW:
                optimizer_kwargs["eps"] = 1e-5
                optimizer_kwargs["weight_decay"] = 5e-3

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = dict(pi=[256,256], vf=[256,256])
            else:

                net_arch = dict(pi=[16,16], vf=[64,64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.adversarial = adversarial
        self.num_adversaries = num_adversaries
        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        #self.features_extractor_class=
        #self.context_extractor = self.m
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_ctrl_features_extractor = self.features_extractor
            self.pi_dstb_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_ctrl_features_extractor = self.features_extractor
            self.pi_dstb_features_extractor = self.make_features_extractor()
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None
        self.policy_memory_size = policy_memory_size
        if self.policy_memory_size is not None:
            self.policy_memory = np.zeros((self.policy_memory_size,), dtype=object)
        assert not (squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        self.dstb_action_space = dstb_action_space
        if dstb_action_space is None:
            self.dstb_action_space = action_space
            dstb_action_space = action_space
        if hasattr(self, "num_adversaries") and self.num_adversaries is not None:
            self.dstb_action_dist = []
            for i in range(self.num_adversaries):
                self.dstb_action_dist.append(make_proba_distribution(dstb_action_space, use_sde=use_sde, dist_kwargs=dist_kwargs))
        else:

            self.dstb_action_dist = make_proba_distribution(dstb_action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        #self.device = device
        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)
        self.dstb_action_dist.sample_weights(self.dstb_log_std, batch_size=n_envs)
    def _build_mlp_extractor(self) -> None:
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

    def _build(self, joint_schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

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
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi).to('cuda')
            if self.adversarial: # we are doing adversarial!

                self.dstb_action_net= self.dstb_action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi).to('cuda')
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1).to('cuda')

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            if self.adversarial:
                module_gains = {
                    self.features_extractor: np.sqrt(2),
                    self.mlp_extractor: np.sqrt(2),
                    self.action_net: 0.01,
                    self.dstb_action_net: 0.01,
                    self.value_net: 1,
                }
            else:
                module_gains = {
                    self.features_extractor: np.sqrt(2),
                    self.mlp_extractor: np.sqrt(2),
                    self.action_net: 0.01,
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

        # Setup optimizer with initial learning rate
        #self.ctrl_optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
        #itertools.chain([self.log_std], self.mlp_extractor.policy_net.parameters(),
        #                self.mlp_extractor.value_net.parameters(), self.action_net.parameters(),
        #                self.value_net.parameters())
        #itertools.chain([self.dstb_log_std], self.mlp_extractor.dstb_net.parameters(), self.dstb_action_net.parameters())
        #itertools.chain([self.log_std], self.mlp_extractor.policy_net.parameters(), self.action_net.parameters())
        #itertools.chain(self.mlp_extractor.value_net.parameters(), self.value_net.parameters())
        #self.optimizer = self.optimizer_class(self.parameters(), joint_schedule[0](1), **self.optimizer_kwargs)
        #if self.policy_memory_size is not None:
        #    for i in range(self.policy_memory_size):
        #        torch.save(self, "weights.pt")
        #        self.policy_memory[i] = torch.load("weights.pt")
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
            self.ctrl_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.policy_net.parameters(), self.pi_ctrl_features_extractor.parameters(),self.action_net.parameters()), joint_schedule[1](1),maximize=False)
            self.dstb_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.dstb_net.parameters(), self.pi_dstb_features_extractor.parameters(), self.dstb_action_net.parameters()), joint_schedule[2](1), maximize=False)
            self.value_optimizer = self.optimizer_class(
                itertools.chain(self.mlp_extractor.value_net.parameters(), self.vf_features_extractor.parameters(), self.value_net.parameters()),
                joint_schedule[0](1), **self.optimizer_kwargs)
        '''
        if self.policy_memory_size is not None:
            for i in range(self.policy_memory_size):
                torch.save(self, "weights.pt")
                self.policy_memory[i] = torch.load("weights.pt")
        '''
        #self.policy_memory_optimizers = np.zeros((self.policy_memory_size,), dtype=object)
        #for i in range(self.policy_memory_size):
        #    self.policy_memory_optimizers[i] = self.optimizer_class(itertools.chain([self.policy_memory[i].dstb_log_std], self.policy_memory[i].mlp_extractor.dstb_net.parameters(), self.policy_memory[i].dstb_action_net.parameters()), joint_schedule[2](1), maximize=False)
        #TODO: ReduceLROnPlateau
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_both, latent_vf = self.mlp_extractor(features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
        else:
            pi_features, vf_features = features
            latent_both = self.mlp_extractor.forward_actor(pi_features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        ctrl_distribution, dstb_distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb)
        ctrl_actions = ctrl_distribution.get_actions(deterministic=deterministic)
        ctrl_log_prob = ctrl_distribution.log_prob(ctrl_actions)
        dstb_actions = dstb_distribution.get_actions(deterministic=deterministic)
        dstb_log_prob = dstb_distribution.log_prob(dstb_actions)
        ctrl_actions = ctrl_actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        dstb_actions = dstb_actions.reshape((-1, *self.dstb_action_space.shape))
        return ctrl_actions, ctrl_log_prob, values, dstb_actions, dstb_log_prob

    def extract_features(  # type: ignore[override]
        self, obs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_pi_dstb: th.Tensor) -> [Distribution, Distribution]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        dstb_mean_actions = self.dstb_action_net(latent_pi_dstb)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions), self.dstb_action_dist.proba_distribution(action_logits=dstb_mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi), self.dstb_action_dist.proba_distribution(dstb_mean_actions, self.dstb_log_std, latent_pi_dstb)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        ctrl_dstro, dstb_dstro = self.get_distribution(observation)
        return ctrl_dstro.get_actions(deterministic=deterministic), dstb_dstro.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions: th.Tensor, dstb_actions: th. Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :param dstb_actions: dstb Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_both, latent_vf = self.mlp_extractor(features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
        else:
            pi_features, vf_features = features
            latent_both = self.mlp_extractor.forward_actor(pi_features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        ctrl_distribution, dstb_distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb)
        ctrl_log_prob = ctrl_distribution.log_prob(actions)
        dstb_log_prob = dstb_distribution.log_prob(dstb_actions)
        values = self.value_net(latent_vf)
        ctrl_entropy = ctrl_distribution.entropy()
        dstb_entropy = dstb_distribution.entropy()
        return values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy

    def get_distribution(self, obs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        ctrl_features = super().extract_features(obs, self.pi_ctrl_features_extractor)

        latent_pi, latent_pi_dstb = self.mlp_extractor.forward_actor(features)
        ctrl_dstro, dstb_distro = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb)
        return ctrl_dstro, dstb_distro

    def predict_values(self, obs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, dstb_actions = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        dstb_actions = dstb_actions.cpu().numpy().reshape((-1, *self.dstb_action_space.shape))
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
                dstb_actions = self.unscale_action(dstb_actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]
                dstb_actions = np.clip(dstb_actions, self.dstb_action_space.low, self.dstb_action_space.high)
        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)
            dstb_actions = dstb_actions.squeeze(axis=0)

        return (actions, state), (dstb_actions, state)  # type: ignore[return-value]
class SelectLastLSTMOutput(nn.Module):
    """
    A helper module to select the last output from an LSTM layer.
    LSTMs return (output, (hidden, cell)), and this module extracts
    the `output` tensor and returns the last time step's output,
    which has a shape of (batch_size, hidden_size).
    """

    def forward(self, x: Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]) -> th.Tensor:
        output, _ = x
        return output

class ActorActorCriticGeneralistPolicy(ActorActorCriticPolicy):
    def __init__(self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        ortho_init: bool = False,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        adversarial=True,
        dstb_action_space: spaces.Space = None,
        policy_memory_size: Optional[int] = None,
        num_adversaries=None
    ):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         lr_schedule=lr_schedule,
                         net_arch=net_arch,
                         activation_fn=activation_fn,
                         ortho_init=ortho_init,
                         use_sde=use_sde,
                         log_std_init=log_std_init,
                         full_std=full_std,
                         use_expln=use_expln,
                         squash_output=squash_output,
                         features_extractor_class=features_extractor_class,
                         features_extractor_kwargs=features_extractor_kwargs,
                         share_features_extractor=share_features_extractor,
                         normalize_images=normalize_images,
                         optimizer_class=optimizer_class,
                         optimizer_kwargs=optimizer_kwargs,
                         adversarial=True,
                         dstb_action_space=dstb_action_space,
                         policy_memory_size=policy_memory_size,
                         num_adversaries=num_adversaries)

    def _build(self, joint_schedule) -> None:
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

            if self.adversarial: # we are doing adversarial!
                self.dstb_action_net = nn.ModuleList()
                self.head_length = 10 # lstm = 4, 2 linear layers = 2 + 2, proba_dist is also a linear, so 2, total 10
                for i in range(self.num_adversaries):
                    self.dstb_action_net.append(nn.Sequential(
                        nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                        SelectLastLSTMOutput(),
                        nn.Linear(lstm_hidden_size, lstm_hidden_size),
                        self.activation_fn(),
                        nn.Linear(lstm_hidden_size, latent_dim_pi),
                        self.activation_fn(),
                        self.dstb_action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi))
                    ) 
                    if i == 0:
                        assert len(self.dstb_action_net[0]) == 7 and self.head_length == 10

                    #self.dstb_action_net.append(self.dstb_action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi))
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        self.value_net = nn.ModuleList()
        for i in range(self.num_adversaries):
            self.value_net.append(nn.Sequential(
                nn.LSTM(input_size=self.mlp_extractor.latent_dim_vf, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                SelectLastLSTMOutput(),
                nn.Linear(lstm_hidden_size, lstm_hidden_size),
                self.activation_fn(),
                nn.Linear(lstm_hidden_size, lstm_hidden_size),
                self.activation_fn(),
                nn.Linear(lstm_hidden_size, 1))
            )
            #self.value_net.append(nn.Linear(self.mlp_extractor.latent_dim_vf, 1))
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            if self.adversarial:
                module_gains = {
                    self.features_extractor: np.sqrt(2),
                    self.mlp_extractor: np.sqrt(2),
                    self.action_net: 0.01,
                    self.dstb_action_net: 0.01,
                    self.value_net: 1,
                }
            else:
                module_gains = {
                    self.features_extractor: np.sqrt(2),
                    self.mlp_extractor: np.sqrt(2),
                    self.action_net: 0.01,
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

        # Setup optimizer with initial learning rate
        #self.ctrl_optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
        #itertools.chain([self.log_std], self.mlp_extractor.policy_net.parameters(),
        #                self.mlp_extractor.value_net.parameters(), self.action_net.parameters(),
        #                self.value_net.parameters())
        #itertools.chain([self.dstb_log_std], self.mlp_extractor.dstb_net.parameters(), self.dstb_action_net.parameters())
        #itertools.chain([self.log_std], self.mlp_extractor.policy_net.parameters(), self.action_net.parameters())
        #itertools.chain(self.mlp_extractor.value_net.parameters(), self.value_net.parameters())
        #self.optimizer = self.optimizer_class(self.parameters(), joint_schedule[0](1), **self.optimizer_kwargs)
        #if self.policy_memory_size is not None:
        #    for i in range(self.policy_memory_size):
        #        torch.save(self, "weights.pt")
        #        self.policy_memory[i] = torch.load("weights.pt")
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
            self.dstb_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.dstb_net.parameters(), self.pi_dstb_features_extractor.parameters(), self.dstb_action_net.parameters()), joint_schedule[1](1), maximize=True)
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
        '''
        if self.policy_memory_size is not None:
            for i in range(self.policy_memory_size):
                torch.save(self, "weights.pt")
                self.policy_memory[i] = torch.load("weights.pt")
        '''
        #self.policy_memory_optimizers = np.zeros((self.policy_memory_size,), dtype=object)
        #for i in range(self.policy_memory_size):
        #    self.policy_memory_optimizers[i] = self.optimizer_class(itertools.chain([self.policy_memory[i].dstb_log_std], self.policy_memory[i].mlp_extractor.dstb_net.parameters(), self.policy_memory[i].dstb_action_net.parameters()), joint_schedule[2](1), maximize=False)
        #TODO: ReduceLROnPlateau
        self.train()
class ActorCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
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
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

class ActorActorCriticCnnGeneralistPolicy(ActorActorCriticGeneralistPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.LeakyReLU,
            ortho_init: bool = False,
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
            adversarial=True,
            dstb_action_space: spaces.Space = None,
            policy_memory_size: Optional[int] = None,
            num_adversaries=None,
            num_env_per_adv=None
    ):
        #self.device
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            dstb_action_space=dstb_action_space,
            num_adversaries=num_adversaries
        )
        self.num_env_per_adv = num_env_per_adv

    def forward(self, obs: th.Tensor, deterministic: bool = False, network_keys=None) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        if network_keys is None:
            network_keys = self.network_keys
        num_adversaries = len(network_keys)
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_both, latent_vf = self.mlp_extractor(features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
        else:
            pi_ctrl_features, pi_dstb_features, vf_features = features
            latent_both = self.mlp_extractor.forward_actor(pi_ctrl_features, pi_dstb_features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        num_env_per_adv = latent_vf.shape[0] // num_adversaries
        values = th.zeros((latent_vf.shape[0],), device=self.device)
        for i in range(num_adversaries):
            values[i * num_env_per_adv : (i+1)*num_env_per_adv] = self.value_net[network_keys[i]](latent_vf[i * num_env_per_adv: (i + 1) * num_env_per_adv, :])[:, 0]
            #values[i * num_env_per_adv : (i+1)*num_env_per_adv] = non_chunked[i * num_env_per_adv : (i+1)*num_env_per_adv, 0]
        #values = self.value_net(latent_vf)
        ctrl_distribution, dstb_distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb, network_keys=network_keys)
        ctrl_actions = ctrl_distribution.get_actions(deterministic=deterministic)
        ctrl_log_prob = ctrl_distribution.log_prob(ctrl_actions)
        dstb_actions = [dstb_distribution[i].get_actions(deterministic=deterministic) for i in range(num_adversaries)]
        dstb_log_prob = [dstb_distribution[i].log_prob(dstb_actions[i]) for i in range(num_adversaries)]
        ctrl_actions = ctrl_actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        dstb_actions = [dstb_actions[i].reshape((-1, *self.dstb_action_space.shape)) for i in range(num_adversaries)]
        dstb_actions = th.vstack(dstb_actions)
        test = th.zeros((ctrl_actions.shape[0],))
        for i in range(num_adversaries):
            test[i * (num_env_per_adv): (i + 1) * num_env_per_adv] = dstb_log_prob[i][:]
        dstb_log_prob = test
        return ctrl_actions, ctrl_log_prob, values, dstb_actions, dstb_log_prob

    def extract_features(  # type: ignore[override]
            self, obs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(obs,
                                            self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )
            preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
            pi_ctrl_features = self.pi_ctrl_features_extractor(preprocessed_obs)
            pi_dstb_features = self.pi_dstb_features_extractor(preprocessed_obs)
            vf_features = self.vf_features_extractor(preprocessed_obs)
            return pi_ctrl_features, pi_dstb_features, vf_features

    def extract_context_features(self, context):
        #preprocessed_obs = preprocess_obs(context, self.observation_space, normalize_images=self.normalize_images)
        pi_ctrl_features = self.pi_ctrl_features_extractor(context)


        return context_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_pi_dstb: th.Tensor, network_keys=None) -> [Distribution,
                                                                                                Distribution]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if network_keys is None:
            network_keys = self.network_keys
        num_adversaries = len(network_keys)
        mean_actions = self.action_net(latent_pi)
        dstb_mean_actions = th.zeros_like(mean_actions)
        num_env_per_adv = mean_actions.shape[0] // num_adversaries
        for i in range(num_adversaries):
            dstb_mean_actions[i * num_env_per_adv : (i+1) * num_env_per_adv, :] = self.dstb_action_net[network_keys[i]](latent_pi_dstb[i * num_env_per_adv : (i+1) * num_env_per_adv, :])
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(
                action_logits=mean_actions), [self.dstb_action_dist[network_keys[i]].proba_distribution(action_logits=dstb_mean_actions[i * num_env_per_adv : (i + 1) * num_env_per_adv, :]) for i in range(num_adversaries)]
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std,
                                                       latent_pi), self.dstb_action_dist.proba_distribution(
                dstb_mean_actions, self.dstb_log_std, latent_pi_dstb)
        else:
            raise ValueError("Invalid action distribution")

    def _get_action_dist_from_latent_nonuniform(self, latent_pi: th.Tensor, latent_pi_dstb: th.Tensor,
                                                shuffle_keys, network_keys=None) -> [Distribution,
                                                                  Distribution]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if network_keys is None:
            network_keys = self.network_keys
        num_adversaries = len(network_keys)
        mean_actions = self.action_net(latent_pi)#[:, 0, :] # remove middle dim from lstm
        #num_env_per_adv = self.num_global_env // num_adversaries # WHY IS SELF.NUM_GLOBAL_ENV 1?
        num_env_per_adv = self.num_env_per_adv
        dstb_mean_actions = th.zeros_like(mean_actions)
        num_per = mean_actions.shape[0] // num_adversaries
        full = [i for i in range(self.num_global_env)]
        adv_distros = []
        for i in range(num_adversaries):
            which_envs = range(i * num_env_per_adv, (i + 1) * num_env_per_adv)
            indices = np.isin(shuffle_keys, which_envs)
            this_env_latent_pi_dstb = latent_pi_dstb[indices]
            self.dstb_action_net[network_keys[i]] = self.dstb_action_net[network_keys[i]].to(self.device)
            # chunk = full[i * num_per : (i+1) * num_per]
            this_adv_dstb_mean_actions = self.dstb_action_net[network_keys[i]](this_env_latent_pi_dstb)

            adv_distros.append(self.dstb_action_dist[network_keys[i]].proba_distribution(action_logits=this_adv_dstb_mean_actions))

        ctrl_distro = self.action_dist.proba_distribution(
            action_logits=mean_actions)
        return ctrl_distro, adv_distros

    def _predict(self, observation, deterministic: bool = False, network_keys=None) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        if network_keys is None:
            network_keys = self.network_keys
        ctrl_dstro, dstb_dstro = self.get_distribution(observation, network_keys=network_keys)
        return ctrl_dstro.get_actions(deterministic=deterministic), [dstb_dstro[i].get_actions(deterministic=deterministic) for i in range(len(network_keys))]

    def evaluate_actions(self, obs, actions: th.Tensor, dstb_actions: th.Tensor, shuffle_keys=None, network_keys=None) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :param dstb_actions: dstb Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        if network_keys is None:
            network_keys = self.network_keys
        # Preprocess the observation if needed
        with th.no_grad():
            num_adversaries = len(network_keys)
            #features = self.extract_features(obs)
            preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
            features = (self.pi_ctrl_features_extractor(preprocessed_obs), self.pi_dstb_features_extractor(preprocessed_obs))
            if self.share_features_extractor:
                latent_both, latent_vf = self.mlp_extractor(features)
                latent_pi = latent_both[0]
                latent_pi_dstb = latent_both[1]
            else:
                pi_ctrl_features, pi_dstb_features = features
                latent_both = self.mlp_extractor.forward_actor(pi_ctrl_features, pi_dstb_features)
                latent_pi = latent_both[0]
                latent_pi_dstb = latent_both[1]
                #latent_vf = self.mlp_extractor.forward_critic(vf_features)
            #latent_pi = latent_pi.unsqueeze(1)
            #latent_pi_dstb = latent_pi_dstb.unsqueeze(1)
            #latent_vf = latent_vf.unsqueeze(1)
            ctrl_distribution, dstb_distribution = self._get_action_dist_from_latent_nonuniform(latent_pi, latent_pi_dstb, shuffle_keys=shuffle_keys, network_keys=network_keys)
            ctrl_log_prob = ctrl_distribution.log_prob(actions)
            #dstb_log_prob = th.zeros_like(ctrl_log_prob)
            #num_global_env = np.max(shuffle_keys) + 1 #TODO HACKY
            num_global_env = self.num_global_env
            num_env_per_adv = num_global_env // num_adversaries
            full = [i for i in range(num_global_env)]
            concated_adv_log_probs = th.empty_like(ctrl_log_prob,dtype=th.float16) #empty_like is a bit faster than zeros_like
            for i in range(num_adversaries):
                chunk = full[i * num_env_per_adv : (i+1)*num_env_per_adv]
                #Try not to move stuff to CPU if can be avoided
                chunk_tensor = th.tensor(chunk, device=shuffle_keys.device if th.is_tensor(shuffle_keys) else 'cpu')
                shuffle_keys = shuffle_keys if th.is_tensor(shuffle_keys) else th.tensor(shuffle_keys, device=chunk_tensor.device)
                indices = th.isin(shuffle_keys, chunk_tensor) # is a bool array can be directly be used as keys

                concated_adv_log_probs[indices] = dstb_distribution[i].log_prob(dstb_actions[indices])

            dstb_log_prob = concated_adv_log_probs

        #dstb_log_prob = [dstb_distribution[i].log_prob(dstb_actions[i]) for i in range(self.num_adversaries)]
        #test = th.zeros((actions.shape[0],))
        # chunk_size = actions.shape[0] // num_adversaries
        #for i in range(self.num_adversaries):
        #    test[i * (chunk_size): (i + 1) * chunk_size] = dstb_log_prob[i][:]
        #dstb_log_prob = test
        vf_features = self.vf_features_extractor(preprocessed_obs)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = th.empty((latent_vf.shape[0],), device=self.device) #empty is a bit faster than zeros
        num_global_env = th.max(shuffle_keys) + 1 #latent_vf.shape[0] // self.num_adversaries
        num_env_per_adv = num_global_env // num_adversaries
        for i in range(num_adversaries):
            chunk = full[i * num_env_per_adv: (i + 1) * num_env_per_adv]
            indices = np.isin(shuffle_keys, chunk)
            values[indices] = self.value_net[network_keys[i]](latent_vf[indices]).flatten()
            #values[i * chunk_size: (i + 1) * chunk_size] = non_chunked[
            #                                                         i * chunk_size: (i + 1) * chunk_size, 0]
        ctrl_entropy = ctrl_distribution.entropy()
        stacked_dstb_entropy = [dstb_distribution[i].entropy() for i in range(num_adversaries)]
        #test = th.zeros((actions.shape[0],))
        #chunk_size = actions.shape[0] // self.num_adversaries
        #for i in range(self.num_adversaries):
        #    test[i * (chunk_size): (i + 1) * chunk_size] = stacked_dstb_entropy[i][:]
        #dstb_entropy = test
        dstb_entropy = th.hstack(stacked_dstb_entropy)
        return values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy

    def get_distribution(self, obs, network_keys=None) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        if network_keys is None:
            network_keys = self.network_keys
        ctrl_features, dstb_features, _ = self.extract_features(obs, self.pi_ctrl_features_extractor)
        #dstb_features = self.extract_features(obs, self.pi_dstb_features_extractor)
        latent_pi, latent_pi_dstb = self.mlp_extractor.forward_actor(ctrl_features, dstb_features)
        ctrl_dstro, dstb_distro = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb, network_keys=network_keys)
        return ctrl_dstro, dstb_distro

    def predict_values(self, obs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, _, vf_features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        num_env_per_adv = latent_vf.shape[0] // self.num_adversaries
        num_env_per_adv = latent_vf.shape[0] // self.num_adversaries
        values = th.zeros((latent_vf.shape[0],), device=self.device)
        for i in range(self.num_adversaries):
            values[i * num_env_per_adv: (i + 1) * num_env_per_adv] = self.value_net[i](latent_vf[i * num_env_per_adv: (i + 1) * num_env_per_adv, :])[:, 0]
            #values[i * num_env_per_adv: (i + 1) * num_env_per_adv] = non_chunked[
            #                                                         i * num_env_per_adv: (i + 1) * num_env_per_adv, 0]

        return values

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            network_keys=None
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if network_keys is None:
            network_keys = self.network_keys
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, dstb_actions = self._predict(obs_tensor, deterministic=deterministic, network_keys=network_keys)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        dstb_actions = [dstb_actions[i].cpu().numpy().reshape((-1, *self.dstb_action_space.shape)) for i in range(len(network_keys))]
        dstb_actions = np.vstack(dstb_actions)
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
                dstb_actions = self.unscale_action(dstb_actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low,
                                  self.action_space.high)  # type: ignore[assignment, arg-type]
                dstb_actions = np.clip(dstb_actions, self.dstb_action_space.low, self.dstb_action_space.high)
        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)
            dstb_actions = dstb_actions.squeeze(axis=0)

        return (actions, state), (dstb_actions, state)  # type: ignore[return-value]

class ActorActorCriticCnnPolicy(ActorActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.LeakyReLU,
            ortho_init: bool = False,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = False,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            adversarial=True,
            dstb_action_space: spaces.Space = None,
            policy_memory_size: Optional[int] = None,
            device='auto'
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            dstb_action_space=dstb_action_space,
            device=device

        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_both, latent_vf = self.mlp_extractor(features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
        else:
            pi_ctrl_features, pi_dstb_features, vf_features = features
            latent_both = self.mlp_extractor.forward_actor(pi_ctrl_features, pi_dstb_features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        ctrl_distribution, dstb_distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb)
        ctrl_actions = ctrl_distribution.get_actions(deterministic=deterministic)
        ctrl_log_prob = ctrl_distribution.log_prob(ctrl_actions)
        dstb_actions = dstb_distribution.get_actions(deterministic=deterministic)
        dstb_log_prob = dstb_distribution.log_prob(dstb_actions)
        ctrl_actions = ctrl_actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        dstb_actions = dstb_actions.reshape((-1, *self.dstb_action_space.shape))
        return ctrl_actions, ctrl_log_prob, values, dstb_actions, dstb_log_prob

    def extract_features(  # type: ignore[override]
            self, obs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(obs,
                                            self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )
            preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
            pi_ctrl_features = self.pi_ctrl_features_extractor(preprocessed_obs)
            pi_dstb_features = self.pi_dstb_features_extractor(preprocessed_obs)
            vf_features = self.vf_features_extractor(preprocessed_obs)
            return pi_ctrl_features, pi_dstb_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_pi_dstb: th.Tensor) -> [Distribution,
                                                                                                Distribution]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        dstb_mean_actions = self.dstb_action_net(latent_pi_dstb)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(
                action_logits=mean_actions), self.dstb_action_dist.proba_distribution(action_logits=dstb_mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std,
                                                       latent_pi), self.dstb_action_dist.proba_distribution(
                dstb_mean_actions, self.dstb_log_std, latent_pi_dstb)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        ctrl_dstro, dstb_dstro = self.get_distribution(observation)
        return ctrl_dstro.get_actions(deterministic=deterministic), dstb_dstro.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions: th.Tensor, dstb_actions: th.Tensor) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :param dstb_actions: dstb Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_both, latent_vf = self.mlp_extractor(features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
        else:
            pi_ctrl_features, pi_dstb_features, vf_features = features
            latent_both = self.mlp_extractor.forward_actor(pi_ctrl_features, pi_dstb_features)
            latent_pi = latent_both[0]
            latent_pi_dstb = latent_both[1]
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        ctrl_distribution, dstb_distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb)
        ctrl_log_prob = ctrl_distribution.log_prob(actions)
        dstb_log_prob = dstb_distribution.log_prob(dstb_actions)
        values = self.value_net(latent_vf)
        ctrl_entropy = ctrl_distribution.entropy()
        dstb_entropy = dstb_distribution.entropy()
        return values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy

    def get_distribution(self, obs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        ctrl_features, dstb_features, _ = self.extract_features(obs, self.pi_ctrl_features_extractor)
        #dstb_features = self.extract_features(obs, self.pi_dstb_features_extractor)
        latent_pi, latent_pi_dstb = self.mlp_extractor.forward_actor(ctrl_features, dstb_features)
        ctrl_dstro, dstb_distro = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb)
        return ctrl_dstro, dstb_distro

    def predict_values(self, obs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, _, vf_features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return self.value_net(latent_vf)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, dstb_actions = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        dstb_actions = dstb_actions.cpu().numpy().reshape((-1, *self.dstb_action_space.shape))
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
                dstb_actions = self.unscale_action(dstb_actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low,
                                  self.action_space.high)  # type: ignore[assignment, arg-type]
                dstb_actions = np.clip(dstb_actions, self.dstb_action_space.low, self.dstb_action_space.high)
        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)
            dstb_actions = dstb_actions.squeeze(axis=0)

        return (actions, state), (dstb_actions, state)  # type: ignore[return-value]

class MultiInputActorCriticPolicy(ActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space (Tuple)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Uses the CombinedExtractor
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Union[List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))

class ContinuousCriticAdv(BaseModel):
    """
    Critic network(s) for SMART-ISAAC.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        dstb_action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.dstb_action_space = dstb_action_space
        action_dim = get_action_dim(self.action_space)
        dstb_action_dim = get_action_dim(self.dstb_action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim + action_dim + dstb_action_dim, 1, net_arch, activation_fn, smart_bias_off=True)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor, dstb_actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions, dstb_actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor, dstb_actions) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions, dstb_actions], dim=1))

    def gpt_forward(self, obs: th.Tensor, actions: th.Tensor, dstb_actions: th.Tensor) -> th.Tensor:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions, dstb_actions], dim=1)
        q_values = [q_net(qvalue_input) for q_net in self.q_networks]
        # Aggregate Q-values in a way that involves both networks
        combined_q_value = sum(q_values) / len(q_values)
        return combined_q_value.mean()

# You can add this class at the end of stable_baselines3/stable_baselines3/common/policies.py

class IPPOActorCriticCnnGeneralistPolicy(ActorActorCriticCnnGeneralistPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.LeakyReLU,
            ortho_init: bool = False,
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
            adversarial=True,
            dstb_action_space: spaces.Space = None,
            policy_memory_size: Optional[int] = None,
            num_adversaries=None,
            num_env_per_adv=None
    ):
        # We need to replicate the __init__ logic to properly initialize the new feature extractors
        # before _build() is called.

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            dstb_action_space=dstb_action_space,
            num_adversaries=num_adversaries,
        )

        # super().__init__(
        #     observation_space,
        #     action_space,
        #     lr_schedule,
        #     net_arch,
        #     activation_fn,
        #     ortho_init,
        #     use_sde,
        #     log_std_init,
        #     full_std,
        #     use_expln,
        #     squash_output,
        #     features_extractor_class,
        #     features_extractor_kwargs,
        #     # Pass False to prevent parent feature extractor creation
        #     share_features_extractor=False,
        #     _init_setup_model=False,
        #     normalize_images=normalize_images,
        #     optimizer_class=optimizer_class,
        #     optimizer_kwargs=optimizer_kwargs,
        # )

        # self.adversarial = adversarial
        # self.dstb_action_space = dstb_action_space
        # self.num_adversaries = num_adversaries
        # self.num_env_per_adv = num_env_per_adv

        # # ---- MODIFICATION: Independent Feature Extractors ----
        # self.pi_ctrl_features_extractor = self.make_features_extractor()
        # self.pi_dstb_features_extractor = self.make_features_extractor()
        # self.ego_vf_features_extractor = self.make_features_extractor()
        # self.adv_vf_features_extractor = self.make_features_extractor()
        # # A bit of a hack: features_dim is needed by the MLP extractor, and we assume all extractors have the same output dim
        # self.features_dim = self.pi_ctrl_features_extractor.features_dim
        # # ---- END MODIFICATION ----

        # if self.adversarial:
        #     self.dstb_action_dist = []
        #     for i in range(self.num_adversaries):
        #         self.dstb_action_dist.append(make_proba_distribution(self.dstb_action_space, use_sde=use_sde, dist_kwargs=self.dist_kwargs))

        # self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = IPPOMlpExtractorAdv(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device='auto',
            adversarial=True,
            context_dim=0
        )

    def _build(self, joint_schedule) -> None:
        self.pi_ctrl_features_extractor = self.make_features_extractor()
        self.pi_dstb_features_extractor = self.make_features_extractor()
        self.ego_vf_features_extractor = self.make_features_extractor()
        self.adv_vf_features_extractor = self.make_features_extractor()
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        lstm_hidden_size = 256
        
        # Actor network setup is the same
        if isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = nn.Sequential(
                nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                SelectLastLSTMOutput(),
                nn.Linear(lstm_hidden_size, lstm_hidden_size),
                self.activation_fn(),
                nn.Linear(lstm_hidden_size, latent_dim_pi),
                self.activation_fn(),
                self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
            )

            if self.adversarial:
                self.dstb_action_net = nn.ModuleList()
                for i in range(self.num_adversaries):
                    self.dstb_action_net.append(nn.Sequential(
                        nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                        SelectLastLSTMOutput(),
                        nn.Linear(lstm_hidden_size, lstm_hidden_size),
                        self.activation_fn(),
                        nn.Linear(lstm_hidden_size, latent_dim_pi),
                        self.activation_fn(),
                        self.dstb_action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi))
                    )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # ---- MODIFICATION: Independent Value Networks (Trunk + Head) ----
        self.adv_value_net = nn.ModuleList()
        for i in range(self.num_adversaries):
            self.adv_value_net.append(nn.Sequential(
                nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                SelectLastLSTMOutput(),
                nn.Linear(lstm_hidden_size, lstm_hidden_size),
                self.activation_fn(),
                nn.Linear(lstm_hidden_size, 1))
            )

        self.ego_value_net = nn.Sequential(
            nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
            SelectLastLSTMOutput(),
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            self.activation_fn(),
            nn.Linear(lstm_hidden_size, 1)
        )
        # ---- END MODIFICATION ----
        
        self.ctrl_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.policy_net.parameters(), self.pi_ctrl_features_extractor.parameters(),self.action_net.parameters()), joint_schedule[0](1),maximize=True)
        self.dstb_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.dstb_net.parameters(), self.pi_dstb_features_extractor.parameters(), self.dstb_action_net.parameters()), joint_schedule[1](1), maximize=True)
        
        # ---- MODIFICATION: Independent Value Optimizers ----
        self.ego_value_optimizer = self.optimizer_class(
            itertools.chain(self.ego_vf_features_extractor.parameters(), self.mlp_extractor.ego_value_net.parameters(), self.ego_value_net.parameters()),
            joint_schedule[2](1), **self.optimizer_kwargs
        )
        self.adv_value_optimizer = self.optimizer_class(
            itertools.chain(self.adv_vf_features_extractor.parameters(), self.mlp_extractor.adv_value_net.parameters(), self.adv_value_net.parameters()),
            joint_schedule[2](1), **self.optimizer_kwargs
        )
        # ---- END MODIFICATION ----
        self.train()

    def extract_features(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        # ---- MODIFICATION: Extract all 4 features ----
        pi_ctrl_features = self.pi_ctrl_features_extractor(preprocessed_obs)
        pi_dstb_features = self.pi_dstb_features_extractor(preprocessed_obs)
        ego_vf_features = self.ego_vf_features_extractor(preprocessed_obs)
        adv_vf_features = self.adv_vf_features_extractor(preprocessed_obs)
        return pi_ctrl_features, pi_dstb_features, ego_vf_features, adv_vf_features
        # ---- END MODIFICATION ----

    def forward(self, obs: th.Tensor, deterministic: bool = False, network_keys=None) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        
        if network_keys is None:
            network_keys = self.network_keys
        num_adversaries = len(network_keys)
        
        pi_ctrl_features, pi_dstb_features, ego_vf_features, adv_vf_features = self.extract_features(obs)
        latent_pi, latent_pi_dstb = self.mlp_extractor.forward_actor(pi_ctrl_features, pi_dstb_features)
        latent_ego_vf, latent_adv_vf = self.mlp_extractor.forward_critic(ego_vf_features, adv_vf_features)
        
        ctrl_distribution, dstb_distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_dstb, network_keys=network_keys)
        ctrl_actions = ctrl_distribution.get_actions(deterministic=deterministic)
        ctrl_log_prob = ctrl_distribution.log_prob(ctrl_actions)
        
        dstb_actions = [dist.get_actions(deterministic=deterministic) for dist in dstb_distribution]
        dstb_log_prob = [dist.log_prob(act) for dist, act in zip(dstb_distribution, dstb_actions)]

        ctrl_actions = ctrl_actions.reshape((-1, *self.action_space.shape))
        dstb_actions = th.cat([act.reshape((-1, *self.dstb_action_space.shape)) for act in dstb_actions])
        
        # ---- MODIFICATION: Independent Value Forward Pass ----
        num_env_per_adv = ego_vf_features.shape[0] // num_adversaries
        adv_values = th.zeros((ego_vf_features.shape[0],), device=self.device)
        for i in range(num_adversaries):
            adv_values[i*num_env_per_adv:(i+1)*num_env_per_adv] = self.adv_value_net[network_keys[i]](latent_adv_vf[i*num_env_per_adv:(i+1)*num_env_per_adv, :])[:, 0]
        ego_values = self.ego_value_net(latent_ego_vf)[:, 0]
        # ---- END MODIFICATION ----
        
        dstb_log_prob_tensor = th.zeros((ctrl_actions.shape[0],), device=self.device)
        for i in range(num_adversaries):
            dstb_log_prob_tensor[i*num_env_per_adv:(i+1)*num_env_per_adv] = dstb_log_prob[i]
        
        return ctrl_actions, ctrl_log_prob, ego_values, adv_values, dstb_actions, dstb_log_prob_tensor

    def evaluate_actions(self, obs, actions: th.Tensor, dstb_actions: th.Tensor, shuffle_keys=None, network_keys=None) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        if network_keys is None:
            network_keys = self.network_keys
        
        num_adversaries = len(network_keys)
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        pi_ctrl_features = self.pi_ctrl_features_extractor(preprocessed_obs)
        pi_dstb_features = self.pi_dstb_features_extractor(preprocessed_obs)
        latent_pi, latent_pi_dstb = self.mlp_extractor.forward_actor(pi_ctrl_features, pi_dstb_features)

        ctrl_distribution, dstb_distribution = self._get_action_dist_from_latent_nonuniform(latent_pi, latent_pi_dstb, shuffle_keys=shuffle_keys, network_keys=network_keys)
        ctrl_log_prob = ctrl_distribution.log_prob(actions)
        ctrl_entropy = ctrl_distribution.entropy()

        num_global_env = self.num_global_env
        num_env_per_adv = num_global_env // num_adversaries
        
        concated_adv_log_probs = th.empty_like(ctrl_log_prob, dtype=th.float16)
        for i in range(num_adversaries):
            mask = (shuffle_keys >= i * num_env_per_adv) & (shuffle_keys < (i + 1) * num_env_per_adv)
            concated_adv_log_probs[mask] = dstb_distribution[i].log_prob(dstb_actions[mask])
        dstb_log_prob = concated_adv_log_probs
        dstb_entropy = th.cat([dist.entropy() for dist in dstb_distribution])

        # ---- MODIFICATION: Independent Value Evaluation ----
        ego_vf_features = self.ego_vf_features_extractor(preprocessed_obs)
        adv_vf_features = self.adv_vf_features_extractor(preprocessed_obs)
        latent_ego_vf, latent_adv_vf = self.mlp_extractor.forward_critic(ego_vf_features, adv_vf_features)
        
        adv_values = th.empty((ego_vf_features.shape[0],), device=self.device)
        for i in range(num_adversaries):
            mask = (shuffle_keys >= i*num_env_per_adv) & (shuffle_keys < (i+1)*num_env_per_adv)
            adv_values[mask] = self.adv_value_net[network_keys[i]](latent_adv_vf[mask]).flatten()
        
        ego_values = self.ego_value_net(latent_ego_vf).flatten()
        # ---- END MODIFICATION ----
        
        return ego_values, adv_values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy

    def predict_values(self, obs, network_keys=None) -> Tuple[th.Tensor, th.Tensor]:
        network_keys = list(range(self.num_adversaries))
        # if network_keys is None:
        #     network_keys = self.network_keys
        
        # ---- MODIFICATION: Independent Value Prediction ----
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        ego_vf_features = self.ego_vf_features_extractor(preprocessed_obs)
        adv_vf_features = self.adv_vf_features_extractor(preprocessed_obs)
        latent_ego_vf, latent_adv_vf = self.mlp_extractor.forward_critic(ego_vf_features, adv_vf_features)
        num_env_per_adv = ego_vf_features.shape[0] // len(network_keys)
        adv_values = th.zeros((ego_vf_features.shape[0],), device=self.device)
        for i, key in enumerate(network_keys):
            adv_values[i*num_env_per_adv:(i+1)*num_env_per_adv] = self.adv_value_net[key](latent_adv_vf[i*num_env_per_adv:(i+1)*num_env_per_adv, :])[:, 0]

        ego_values = self.ego_value_net(latent_ego_vf)[:, 0]
        # ---- END MODIFICATION ----
        return ego_values, adv_values