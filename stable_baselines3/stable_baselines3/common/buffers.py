import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Union
import time
import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
    AdvRolloutBufferSamples,
    Q_RolloutBufferSamples
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

class AdvReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    dstb_actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        dstb_action_space=None
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        if dstb_action_space is not None:
            self.dstb_action_dim = dstb_action_space.shape[0]
        else:
            self.dstb_action_dim = self.action_dim
        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.dstb_actions = np.zeros((self.buffer_size, self.n_envs, self.dstb_action_dim), dtype=self._maybe_cast_dtype(action_space.dtype))

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        dstb_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        dstb_action = dstb_action.reshape((self.n_envs, self.dstb_action_dim))
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.dstb_actions[self.pos] = np.array(dstb_action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self.dstb_actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return AdvReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype):
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.X0_VALUES_MASTER = np.zeros((self.n_envs, *self.obs_shape))
        self.env_indices = np.tile(np.arange(n_envs), (buffer_size, 1))  # Shape: (buffer_size, n_envs)
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.uint8)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # Same reshape, for actions
        action = action.reshape((self.n_envs, self.action_dim))
        new_episode_indices = np.where(episode_start)[0]
        if len(new_episode_indices) > 0:
            self.X0_VALUES_MASTER[new_episode_indices] = obs[new_episode_indices]
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "env_indices"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.env_indices[batch_inds].flatten()
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
    def prepare_data_for_training(self) -> None:
        """
        Prepares the buffer for training by swapping and flattening the data.
        This is a one-time operation that should be called after collecting rollouts.
        """
        #print("--- AdvRolloutBuffer PREPARED ---")
        if not self.generator_ready:
            _torch_tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            for tensor_name in _torch_tensor_names:
                tensor = self.__dict__[tensor_name]
                #if self.pin_memory:
                    # Data is already a tensor, just move to the correct device
                #    th_tensor = tensor.to(self.device, non_blocking=True)
                #else:
                    # th.as_tensor avoids a copy if the numpy array is on the CPU and writable
                    # which should be the case here
                th_tensor = th.as_tensor(tensor, device=self.device)
                
                shape = th_tensor.shape
                # .contiguous().view() is more efficient than reshape for non-contiguous tensors
                # We clone here to sever any computational graph links among buffer tensors
                self.__dict__[tensor_name] = th_tensor.transpose(0, 1).contiguous().view(shape[0] * shape[1], *shape[2:]).clone()

            # Handle env_indices separately as it remains a numpy array
            self.env_indices = self.swap_and_flatten(self.env_indices)
            self.generator_ready = True


def _numpy_cpu(x: Union[np.ndarray, th.Tensor]) -> np.ndarray:
    """Coerce rollout inputs to CPU numpy for buffer storage (avoids CUDA arrays in the buffer)."""
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Q_RolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space, device: Union[th.device, str] = "auto", gae_lambda: float = 1, gamma: float = 0.99, n_envs: int = 1):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
    def reset(self) -> None:
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.uint8)
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.uint8)
        self.ego_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.adv_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.q_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()
    def add(self, obs: np.ndarray, action: np.ndarray, adv_action: np.ndarray, reward: np.ndarray, next_obs: np.ndarray, dones, episode_start: np.ndarray, value: th.Tensor, log_prob: th.Tensor, q_value: th.Tensor) -> None:
        self.observations[self.pos] = _numpy_cpu(obs).copy()
        self.ego_actions[self.pos] = _numpy_cpu(action).copy()
        self.adv_actions[self.pos] = _numpy_cpu(adv_action).copy()
        self.rewards[self.pos] = _numpy_cpu(reward).copy()
        self.next_observations[self.pos] = _numpy_cpu(next_obs).copy()
        self.dones[self.pos] = _numpy_cpu(dones).copy()
        self.episode_starts[self.pos] = _numpy_cpu(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.q_values[self.pos] = q_value.clone().cpu().numpy().flatten()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, "Buffer is not full"
        np.random.seed(67)
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        #indices = [12, 14, 28, 38, 91, 92, 93, 94, 95, 96, 97, 98, 99, 17]
        #indices = list(range(self.buffer_size * self.n_envs))
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "ego_actions",
                "adv_actions",
                "next_observations",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "q_values",
                "env_indices",
                "rewards",
                "dones"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.ego_actions[batch_inds],
            self.adv_actions[batch_inds],
            self.next_observations[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.q_values[batch_inds].flatten(),
            self.env_indices[batch_inds].flatten(),
            self.rewards[batch_inds].flatten(),
            self.dones[batch_inds].flatten()
        )
        return Q_RolloutBufferSamples(*tuple(map(self.to_torch, data))) 
    def prepare_data_for_training(self) -> None:
        """
        Prepares the buffer for training by swapping and flattening the data.
        This is a one-time operation that should be called after collecting rollouts.
        """
        #print("--- AdvRolloutBuffer PREPARED ---")
        self.actions = self.ego_actions
        if not self.generator_ready:
            _torch_tensor_names = [
                "observations",
                "actions",
                "ego_actions",
                "adv_actions",
                "next_observations",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "q_values",
                "rewards",
                "dones"
            ]
            for tensor_name in _torch_tensor_names:
                tensor = self.__dict__[tensor_name]
                #if self.pin_memory:
                    # Data is already a tensor, just move to the correct device
                #    th_tensor = tensor.to(self.device, non_blocking=True)
                #else:
                    # th.as_tensor avoids a copy if the numpy array is on the CPU and writable
                    # which should be the case here
                th_tensor = th.as_tensor(tensor, device=self.device)
                
                shape = th_tensor.shape
                # .contiguous().view() is more efficient than reshape for non-contiguous tensors
                # We clone here to sever any computational graph links among buffer tensors
                self.__dict__[tensor_name] = th_tensor.transpose(0, 1).contiguous().view(shape[0] * shape[1], *shape[2:]).clone()

            # Handle env_indices separately as it remains a numpy array
            self.env_indices = self.swap_and_flatten(self.env_indices)
            self.generator_ready = True
class AdvRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    dstb_actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            dstb_action_space=None,
            pin_memory: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.dstb_action_space = dstb_action_space
        self.pin_memory = pin_memory
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.dstb_actions, self.dstb_log_probs = None, None
        self.generator_ready = False

        if dstb_action_space is not None:
            self.dstb_action_dim = dstb_action_space.shape[0]
        else:
            self.dstb_action_dim = self.action_dim
        self.X0_VALUES_MASTER = np.zeros((self.n_envs, *self.obs_shape))
        self.reset()
        self.X0_RETURNS_MASTER = np.zeros(self.returns[0, :].shape)
        print("")

    def reset(self) -> None:
        #print("--- AdvRolloutBuffer RESET ---")
        self.pos = 0
        self.full = False
        obs_shape = (self.buffer_size, self.n_envs) + self.obs_shape

        if self.pin_memory:
            # Create tensors directly on pinned CPU memory
            self.observations = th.zeros(obs_shape, dtype=th.uint8).pin_memory()
            self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32).pin_memory()
            self.dstb_actions = th.zeros((self.buffer_size, self.n_envs, self.dstb_action_dim), dtype=th.float32).pin_memory()
            self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
            self.returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
            self.episode_starts = th.zeros((self.buffer_size, self.n_envs), dtype=th.bool).pin_memory()
            self.values = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
            self.log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
            self.dstb_log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
            self.advantages = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
            self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        else:
            self.observations = np.zeros(obs_shape, dtype=np.uint8)
            self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
            self.dstb_actions = np.zeros((self.buffer_size, self.n_envs, self.dstb_action_dim), dtype=np.float32)
            self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
            self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.dstb_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.generator_ready = False
        self.env_indices = np.tile(np.arange(self.n_envs), (self.buffer_size, 1))  # Shape: (buffer_size, n_envs)
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        new_episode_indices = np.where(self.episode_starts)
        if len(new_episode_indices) > 0:
            self.X0_RETURNS_MASTER[new_episode_indices[1]] = self.returns[new_episode_indices[0], new_episode_indices[1]]
        self.returns = self.returns.clone()

    def compute_returns_and_advantage_pt(self, last_values: th.Tensor, dones: th.Tensor) -> None:
        """
        Compute lambda-return (TD(lambda) estimate) and GAE(lambda) advantage
        with full PyTorch gradient tracking.
        :param last_values: Tensor of state value estimations for the last step (one per env)
        :param dones: Tensor indicating if the last step was terminal (one bool per env)
        """
        last_values = last_values.flatten()
        last_gae_lam = th.zeros_like(last_values)

        # Convert numpy arrays to tensors
        rewards_pt = self.to_torch(self.rewards, copy=False)
        values_pt = self.to_torch(self.values, copy=False)
        episode_starts_pt = self.to_torch(self.episode_starts, copy=False)
        advantages_pt = th.zeros_like(rewards_pt)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - episode_starts_pt[step + 1].float()
                next_values = values_pt[step + 1]

            delta = rewards_pt[step] + self.gamma * next_values * next_non_terminal - values_pt[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages_pt[step] = last_gae_lam

        self.advantages = advantages_pt.clone().cpu().numpy()
        self.returns = self.advantages + self.values.clone().cpu().numpy()

    def compute_returns_and_advantage_pt_test(self, last_values: th.Tensor, dones: th.Tensor) -> None:
        """
        Compute lambda-return (TD(lambda) estimate) and GAE(lambda) advantage
        with full PyTorch gradient tracking.

        :param last_values: Tensor of state value estimations for the last step (one per env)
        :param dones: Tensor indicating if the last step was terminal (one bool per env)
        """
        advantages = th.zeros(self.buffer_size, self.n_envs, device=self.device)
        last_gae_lam = 0  # Initialize last advantage to 0

        # Compute delta (TD residual)
        deltas = self.returns + self.gamma * th.cat() - self.values

        # Compute advantage backwards
        for step in reversed(range(self.buffer_size)):
            next_not_done = 1.0 - dones[step]  # 1 if not done, 0 if done
            advantages[step] = deltas[step] + self.gamma * self.gae_lambda * next_not_done * last_gae_lam
            last_gae_lam = advantages[step]  # Update for next iteration

        print("GAE computed")

    def vectorized_compute_returns_and_advantages(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage using a vectorized PyTorch implementation.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert final dones to a tensor
        dones = self.to_torch(dones)
        last_values = last_values.flatten() # Shape: (n_envs,)

        # --- Vectorized Delta Calculation ---
        # The first step is to calculate the TD-residuals (deltas) for all steps at once.
        # To do this, we need the value of the next state (v(s_t+1)) for each state.

        # self.values is shape (buffer_size, n_envs)
        # We append last_values to create a tensor of V(s_t+1) for each s_t.
        next_values = th.cat(
            [self.to_torch(self.values)[1:], last_values.unsqueeze(0)], dim=0
        )

        # We also need to know if the next state is a terminal state.
        # self.episode_starts is shape (buffer_size, n_envs)
        next_non_terminal = 1.0 - th.cat(
            [self.to_torch(self.episode_starts)[1:], dones.unsqueeze(0).float()], dim=0
        )

        # Now calculate deltas for all steps in a single vectorized operation.
        deltas = self.to_torch(self.rewards) + self.gamma * next_values * next_non_terminal - self.to_torch(self.values)

        # --- Advantage Calculation ---
        # The GAE calculation has a recursive dependency: A_t depends on A_{t+1}.
        # A_t = delta_t + (gamma * gae_lambda * next_non_terminal_t) * A_{t+1}
        # While this part is difficult to fully vectorize without complex operations,
        # running a small Python loop over pre-computed tensors on a GPU is still very fast.

        advantages_pt = th.zeros_like(deltas, device=self.values.device)
        last_gae_lam = th.zeros_like(last_values) # Shape: (n_envs,)
        discount_factor = self.gamma * self.gae_lambda

        for t in reversed(range(self.buffer_size)):
            # Advantage at the next step is `last_gae_lam`.
            # The non-terminal condition is for the transition from t to t+1.
            last_gae_lam = deltas[t] + discount_factor * next_non_terminal[t] * last_gae_lam
            advantages_pt[t] = last_gae_lam

        # --- Finalization ---
        # Store the results back as numpy arrays in the buffer.
        self.advantages = advantages_pt#.clone().cpu().numpy()
        self.returns = self.advantages + self.values # self.values is already a numpy array

    def swap_and_flatten_pt(self, arr: th.Tensor) -> th.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintains the order)

        :param arr: A PyTorch tensor
        :return: A reshaped PyTorch tensor
        """
        if arr.dim() < 2:
            raise ValueError("Input tensor must have at least 2 dimensions")
        return arr.transpose(0, 1).reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            dstb_action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            dstb_log_prob: th.Tensor,
            dones = None
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param dstb_action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
            dstb_log_prob = dstb_log_prob.reshape(-1, 1)
        test = np.round(episode_start)
        #if any(episode_start!=0):
        #    print("oo")
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        dstb_action = dstb_action.reshape((self.n_envs, self.dstb_action_dim))

        new_episode_indices = np.where(np.round(episode_start))[0]
        if len(new_episode_indices) > 0:
            self.X0_VALUES_MASTER[new_episode_indices] = obs[new_episode_indices]

        if self.pin_memory:
            # Copy new data to the pinned memory tensors
            self.observations[self.pos].copy_(th.from_numpy(obs))
            self.actions[self.pos].copy_(th.from_numpy(action))
            self.dstb_actions[self.pos].copy_(th.from_numpy(dstb_action))
            self.rewards[self.pos].copy_(th.from_numpy(reward))
            self.episode_starts[self.pos].copy_(th.from_numpy(np.round(episode_start).astype(bool)))
            self.values[self.pos].copy_(value.cpu().flatten())
            self.log_probs[self.pos].copy_(log_prob.cpu())
            self.dstb_log_probs[self.pos].copy_(dstb_log_prob.cpu())
            if dones is not None:
                self.dones[self.pos].copy_(dones.cpu())
        else:
            self.observations[self.pos] = np.array(obs)
            self.actions[self.pos] = np.array(action)
            self.dstb_actions[self.pos] = np.array(dstb_action)
            self.rewards[self.pos] = np.array(reward)
            self.episode_starts[self.pos] = np.round(episode_start)
            self.values[self.pos] = value.clone().cpu().numpy().flatten()
            self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
            self.dstb_log_probs[self.pos] = dstb_log_prob.clone().cpu().numpy()
            if dones is not None:
                self.dones[self.pos] = dones.clone().cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def flatten(self):
        #buf_copy = deepcopy(self)
        len_count = self.swap_and_flatten(self.returns)
        self.flat_obs = th.zeros(len(len_count.squeeze())).to(self.device)
        self.flat_advantages = th.zeros_like(self.flat_obs)
        self.flat_values = th.zeros_like(self.flat_obs)
        _tensor_names = [
            "observations",
            "actions",
            "dstb_actions",
            "values",
            "log_probs",
            "dstb_log_probs",
            "flat_advantages",
            "returns",
            "env_indices"
        ]
        for tensor in _tensor_names:
            if tensor == 'flat_advantages':
                self.__dict__['flat_advantages'] = self.swap_and_flatten_pt(self.__dict__['advantages'])
                continue
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
        return self
    def sample(self, batch_inds):
        return self._get_samples(batch_inds)

    def prepare_data_for_training(self) -> None:
        """
        Prepares the buffer for training by swapping and flattening the data.
        This is a one-time operation that should be called after collecting rollouts.
        """
        #print("--- AdvRolloutBuffer PREPARED ---")
        if not self.generator_ready:
            _torch_tensor_names = [
                "observations",
                "actions",
                "dstb_actions",
                "values",
                "log_probs",
                "dstb_log_probs",
                "advantages",
                "returns",
            ]
            for tensor_name in _torch_tensor_names:
                tensor = self.__dict__[tensor_name]
                if self.pin_memory:
                    # Data is already a tensor, just move to the correct device
                    th_tensor = tensor.to(self.device, non_blocking=True)
                else:
                    # th.as_tensor avoids a copy if the numpy array is on the CPU and writable
                    # which should be the case here
                    th_tensor = th.as_tensor(tensor, device=self.device)
                
                shape = th_tensor.shape
                # .contiguous().view() is more efficient than reshape for non-contiguous tensors
                # We clone here to sever any computational graph links among buffer tensors
                self.__dict__[tensor_name] = th_tensor.transpose(0, 1).contiguous().view(shape[0] * shape[1], *shape[2:]).clone()

            # Handle env_indices separately as it remains a numpy array
            self.env_indices = self.swap_and_flatten(self.env_indices)
            self.generator_ready = True


    def old_get(self, batch_size: Optional[int] = None) -> Generator[AdvRolloutBufferSamples, None, None]:
        assert self.full, ""
        assert self.generator_ready, "You must call .prepare_data_for_training() before getting data from the buffer."
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        self.indices = indices
        # Prepare the data - MOVED TO prepare_data_for_training()
        
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            self.current_shot = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            #yield self._get_samples(list(range(start_idx, start_idx + batch_size)))
            start_idx += batch_size
    def get(self, batch_size: Optional[int] = None) -> Generator[AdvRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        #indices = list(range(self.buffer_size * self.n_envs))
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "dstb_actions",
                "values",
                "log_probs",
                "dstb_log_probs",
                "advantages",
                "returns",
                "env_indices"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None,
    ) -> AdvRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.dstb_actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.dstb_log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.env_indices[batch_inds].flatten()
        )
        return AdvRolloutBufferSamples(*data)

class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:  # pytype: disable=signature-mismatch
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        # Same reshape, for actions
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:  # pytype: disable=signature-mismatch
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictRolloutBufferSamples:

        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )
