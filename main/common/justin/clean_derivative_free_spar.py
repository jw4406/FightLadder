import torch
import torch as th
import torch.autograd as autograd
import sys
import time
import random
import math
import av
from PIL import Image
import pickle
from venv import create
import wandb
from copy import deepcopy
import gc
from queue import Empty
import warnings
from typing import Union, Type, Optional, Dict, Any, List, Tuple
from stable_baselines3.common.callbacks import ConvertCallback
from torch.multiprocessing import Process, Queue
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.clean_new_policies import CleanActorActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor, NatureCNN
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer, Q_RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance
from common.justin.Doubly_TSS_SPAR import Doubly_TSS_SPAR as dtss
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from common.justin.derivative_free_spar import ParallelUpdater
from .calc_F import _get_buffers_and_keys, _calculate_policy_loss, _compute_grads, calc_F_grad_single, _calculate_q_policy_loss
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau  #TODO: This can be changed to another scheduler.
from torch.optim.lr_scheduler import ExponentialLR
from anyio import value
from gymnasium import spaces
# Also import gym.spaces for backwards compatibility with FightLadder environments
try:
    from gym import spaces as gym_spaces
    # Create tuples for isinstance checks that work with both
    _BoxTypes = (spaces.Box, gym_spaces.Box)
    _DiscreteTypes = (spaces.Discrete, gym_spaces.Discrete)
except ImportError:
    _BoxTypes = spaces.Box
    _DiscreteTypes = spaces.Discrete
from stable_baselines3 import PPO
from utils import select_matchup_env, select_device, get_n_workers, move_policy, unpickle_policy, state2matchup, mirror_flip_attributes
from concurrent.futures import ThreadPoolExecutor
from .parallel_updater import ParallelUpdater
from br_tracker import BRConvergenceTracker

TIMING = False
DEBUG = False
PARALLEL_CALC_F = False
SAVE_TEST = True
USE_PERTURBED = False
class DummyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True

def _print_gpu(tag=""):
    if DEBUG:
        print(f"[{tag}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB | Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

def shard_indices(n_items: int, n_gpus: int) -> List[List[int]]:
    """
    Splits a range of indices [0, n_items) into n_gpus nearly equal-sized chunks.

    This is needed to distribute adversary buffer updates across multiple GPUs.

    Args:
        n_items (int):
            Total number of items to divide (e.g., adversary indices).
        n_gpus (int):
            Number of available GPUs to divide the work among.

    Returns:
        List[List[int]]: A list of `n_gpus` sublists, each containing integer indices.

    Raises:
        ValueError: If n_items < 0 or n_gpus <= 0.
    """
    if n_items < 0 or n_gpus <= 0:
        raise ValueError("n_items must be >= 0 and n_gpus must be > 0.")
    size = math.ceil(n_items / n_gpus)
    return [list(range(i * size, min((i + 1) * size, n_items))) for i in range(n_gpus)]

class CleanDerivativeFreeSPAR(PPO):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "AACCnnPolicy": CleanActorActorCriticPolicy
    }
    def __init__(self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            c_learning_rate: Union[float, Schedule] = 1e-4,
            d_learning_rate: Union[float, Schedule] = 2e-4,
            v_learning_rate: Union[float, Schedule] = 5e-4,
            c_learning_rate_decay: Union[float, Schedule] = 1e-4,
            d_learning_rate_decay: Union[float, Schedule] = 2e-4,
            v_learning_rate_decay: Union[float, Schedule] = 7e-4,
            use_lr_annealing: bool = False,
            lr_anneal_coeff: float = 0.995,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 1,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.1,
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
            seed: Optional[int] = 0,
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
            use_mirror=False,
            num_workers=None,
            scheduler_step_size: int=10, #TODO: 10 was chosen arbitrarily - should be changed.
    ):

        self.matchups = [state2matchup(state) for state in state_list] if state_list is not None else None #This needs to happen before the super().__init__
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
        self.use_lr_annealing = use_lr_annealing
        self.lr_anneal_coeff = lr_anneal_coeff
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
        self.state_list = state_list if state_list is not None else None
        if _init_setup_model:
            self.env.num_envs = self.n_envs
            self._setup_model()
        self.env_generator_func = env_generator_func
        self.parallel_updater = None
        self.n_global_env = self.n_envs
        adversary_buffers = []
        if self.num_adversaries is not None:
            for i in range(self.num_adversaries):
                # overwrite = dtss("AACCnnPolicy",
                #                            self.env,
                #                            device=self.device,
                #                            verbose=self.verbose,
                #                            n_steps=self.n_steps,
                #                            batch_size=self.batch_size // self.n_envs,  # 512,
                #                            n_epochs=self.n_epochs,
                #                            gamma=self.gamma,
                #                            v_learning_rate=v_learning_rate, c_learning_rate=c_learning_rate,
                #                            d_learning_rate=d_learning_rate, v_learning_rate_decay=v_learning_rate_decay,
                #                            c_learning_rate_decay=c_learning_rate_decay,
                #                            d_learning_rate_decay=d_learning_rate_decay,
                #                            clip_range=self.clip_range,
                #                            tensorboard_log=self.tensorboard_log,
                #                            seed=self.seed,
                #                            ent_coef=self.ent_coef,
                #                            dstb_ent_coef=self.dstb_ent_coef,
                #                            update_left=not self.update_left,
                #                            update_right=not self.update_right,
                #                            warmstarted_cont_MAGICS=False,
                #                            matchups=matchups,
                #                            envs_per_matchup=self.envs_per_matchup
                #                            )
                # overwrite.rollout_buffer.n_envs = self.n_env_per_adv
                # adversary_buffers.append(overwrite.rollout_buffer)
                adversary_buffers.append(self.rollout_buffer_class(self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.envs_per_matchup,
                #dstb_action_space=self.dstb_action_space
            ))
            self.adversary_buffers = adversary_buffers
        self.env.num_envs = self.n_envs
        self.use_mirror = use_mirror
        self.num_workers = num_workers

        #Create learning rate schedulers

    def _setup_model(self) -> None:
        assert self.state_list is not None
        assert self.num_adversaries is not None
        assert self.envs_per_matchup is not None
        #super()._setup_model()
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else Q_RolloutBuffer
        self.rollout_buffer_class = buffer_cls
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            #dstb_action_space=self.dstb_action_space
        )

        if hasattr(self, "num_adversaries"):
            self.policy_kwargs['num_adversaries'] = self.num_adversaries
            #self.policy_kwargs['num_env_per_adv'] = self.num_env_per_adv

        self.policy_kwargs['matchups'] = self.matchups
        self.policy_kwargs['envs_per_matchup'] = self.envs_per_matchup
        
        # Set features_extractor_class based on whether observation space is an image
        if is_image_space(self.observation_space):
            self.policy_kwargs['features_extractor_class'] = NatureCNN
        else:
            self.policy_kwargs['features_extractor_class'] = FlattenExtractor

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy.gamma = self.gamma

        self.policy = self.policy.to(self.device)
        if hasattr(self.policy, 'dstb_log_std'):
            self.policy.dstb_log_std = {key: self.policy.dstb_log_std[key].to(self.device) for key in self.policy.dstb_log_std}
        #self.ctrl_scheduler = ReduceLROnPlateau(self.policy.ctrl_optimizer, factor=0.5, patience=10)
        #self.dstb_scheduler = ReduceLROnPlateau(self.policy.dstb_optimizer, factor=0.5, patience=10)
        #self.value_scheduler = ReduceLROnPlateau(self.policy.value_optimizer, factor=0.5, patience=10)
        self.ctrl_scheduler = ExponentialLR(self.policy.ctrl_optimizer, gamma=self.lr_anneal_coeff)
        self.dstb_scheduler = ExponentialLR(self.policy.dstb_optimizer, gamma=self.lr_anneal_coeff)
        self.value_scheduler = ExponentialLR(self.policy.value_optimizer, gamma=self.lr_anneal_coeff)
    
    def _update_schedulers(self , step_ego, step_adv, step_val, skip=False):
        if skip:
            return
        """This functinon updates all schedulers and makes sure that ego_lr <= adv_lr <= value_lr is satisfied."""
        rew_std = np.std([ep_info["r"] for ep_info in self.ep_info_buffer])
        if step_ego:
            self.ctrl_scheduler.step()
        if step_adv:
            self.dstb_scheduler.step()
        if step_val:
            self.value_scheduler.step()
            

        # do we need to multiply by 3 here cause train standard is called twice and
        # train derivative free is called once?

        ego_lr = self.policy.ctrl_optimizer.param_groups[0]['lr']
        adv_lr = self.policy.dstb_optimizer.param_groups[0]['lr']
        value_lr = self.policy.value_optimizer.param_groups[0]['lr']
        
        #TODO: Justin - I don't know if this is the rule you want - please go over it and change it if necessary.
        # Clamp to maintain ordering
        ego_lr = min(ego_lr, adv_lr, value_lr)
        adv_lr = min(max(ego_lr, adv_lr), value_lr)
        value_lr = max(ego_lr, adv_lr, value_lr)
        
        self.policy.ctrl_optimizer.param_groups[0]['lr'] = ego_lr
        self.policy.dstb_optimizer.param_groups[0]['lr'] = adv_lr
        self.policy.value_optimizer.param_groups[0]['lr'] = value_lr

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, adversary_buffers, n_rollout_steps: int, run_ego_forward:bool, run_adv_forward:bool, use_mirror: bool, zero_ego_action, zero_adv_action) -> bool:
        if self.use_mirror:
            return self.collect_rollouts_mirror(env, callback, rollout_buffer, adversary_buffers, n_rollout_steps, run_ego_forward, run_adv_forward, zero_ego_action, zero_adv_action)
        else:
            return self.collect_rollouts_standard(env, callback, rollout_buffer, adversary_buffers, n_rollout_steps, run_ego_forward, run_adv_forward, zero_ego_action, zero_adv_action)
    
    def collect_rollouts_standard(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, adversary_buffers, n_rollout_steps: int, run_ego_forward: bool = True, run_adv_forward: bool = True, zero_ego_action=False, zero_adv_action=False) -> bool:
       
        video_log = [Image.fromarray(env.render(mode="rgb_array"))]
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        for i in range(self.num_adversaries):
            adversary_buffers[i].reset()
        #rollout_buffer_other.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        #np.random.seed(0)
        #random.seed(0)
        #torch.manual_seed(0)
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                ego_actions, ego_log_probs, adv_actions, adv_log_probs, values, q_values = self.policy(obs_tensor, deterministic=False, ego_forward=run_ego_forward, adv_forward=run_adv_forward, zero_ego_action=zero_ego_action, zero_adv_action=zero_adv_action)
                if th.any(adv_actions):
                    #print("Adv actions are not all zeros")
                    pass
                other_values = -values

            actions = ego_actions.cpu().numpy()
            actions_other = adv_actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = np.hstack([actions, actions_other])
            # print(clipped_actions, flush=True)
            # print(np.shape(clipped_actions),flush=True)
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, _BoxTypes):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low,
                                          self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)
            video_log.append(Image.fromarray(env.render(mode="rgb_array")))
            #env.render(mode="rgb_array")
            #np.random.seed(0)
            #random.seed(0)
            #torch.manual_seed(0)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, _DiscreteTypes):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
                actions_other = actions_other.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # for idx, done in enumerate(dones):
            #     if (
            #             done
            #             and coordinate_fn is not None
            #     ):
            #         coordinate_fn(infos[idx]["outcome"])
            #     if (
            #             done
            #             and infos[idx].get("terminal_observation") is not None
            #             and infos[idx].get("TimeLimit.truncated", False)
            #     ):
            #         # print(f"[PPO] idx: {idx}, done: {done}, outcome: {infos[idx]['outcome']}", flush=True)
            #         terminal_obs = rollout_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            #         terminal_obs_other = rollout_policy_other.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            #         with th.no_grad():
            #             terminal_value = rollout_policy.predict_values(terminal_obs)[0]
            #             terminal_value_other = rollout_policy_other.predict_values(terminal_obs_other)[0]
            #         rewards[idx] += self.gamma * terminal_value
            #         rewards_other[idx] += self.gamma * terminal_value_other

                    # from IPython import embed; embed()
            #rollout_buffer.add(self._last_obs.copy(), actions, rewards, self._last_episode_starts, values,
            #                       ego_log_probs)
            rollout_buffer.add(self._last_obs.copy(), actions, actions_other, rewards, new_obs, dones, self._last_episode_starts, values,
                                    ego_log_probs, q_values)
            for i in range(self.num_adversaries):
                indices = slice(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
                #adversary_buffers[i].add(self._last_obs[indices].copy(), actions_other[indices], rewards_other[indices], self._last_episode_starts[indices], other_values[indices],
                #                         adv_log_probs[indices])
                adversary_buffers[i].add(self._last_obs[indices].copy(), actions[indices], actions_other[indices], rewards_other[indices], new_obs[indices], dones[indices], self._last_episode_starts[indices], other_values[indices],
                                         adv_log_probs[indices], -q_values[indices])
            #for i in range(self.num_adversaries):
            #    adversary_buffers[i].add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts, values_other,
            #                             adv_log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.value_forward(obs_as_tensor(new_obs, self.device))
            #values_other = rollout_policy_other.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        for i in range(self.num_adversaries):
            adversary_buffers[i].compute_returns_and_advantage(last_values=-values[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv], dones=dones[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv])
        #if self.update_right:
        #    rollout_buffer_other.compute_returns_and_advantage(last_values=values_other, dones=dones)

        callback.on_rollout_end()

        rollout_buffer.prepare_data_for_training()
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].prepare_data_for_training()


        height, width, layers = np.array(video_log[0]).shape
        container = av.open(f"/home/jw4406/codebase/FightLadder/main/trained_models/tasks/todo/videos" + "_episode_{i}.mp4", mode='w')
        stream = container.add_stream('h264', rate=10)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        for img in video_log:
            frame = av.VideoFrame.from_image(img)
            for packet in stream.encode(frame):
                container.mux(packet)
        remain_packets = stream.encode(None)
        container.mux(remain_packets)
        container.close()
        return True
    
    def collect_rollouts_mirror(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, adversary_buffers, n_rollout_steps: int, update_ego: bool = True, update_adversary: bool = True) -> bool:
        assert self.use_mirror is True, "Use mirror is not True"
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        for i in range(self.num_adversaries):
            adversary_buffers[i].reset()

        callback.on_rollout_start()
        #np.random.seed(0)
        #random.seed(0)
        #torch.manual_seed(0)
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                ego_actions, ego_log_probs, adv_actions, adv_log_probs, values = self.policy(obs_tensor, deterministic=False, ego_forward=update_ego, adv_forward=update_adversary)
                
                # The value network predicts value from the left player's perspective.
                # In mirrored rollouts, the ego is on the right for the second half of envs.
                # Thus, `values` contains ego values for the top half and adversary values for the bottom half.
                # `other_values` is computed to hold adversary values for the top and ego values for the bottom.
                # We first negate, then use `mirror_flip_attributes` to swap the bottom halves,
                # consolidating all ego values in `values` and all adversary values in `other_values`.
                other_values = -values

            actions = ego_actions.cpu().numpy()
            actions_other = adv_actions.cpu().numpy()

            # Rescale and perform action

            mirror_master_copy_actions = deepcopy(actions)
            mirror_master_copy_adv_actions = deepcopy(adv_actions)

            # upper half, lower half

            
            # print("SINGLE TRAIN EXTRACTOR MIRROR")

            '''
            assume wlog Ehonda is the prot.

            action right now is:                  adv_action right now is:
            EHonda left                                              Sagat    right
            EHonda left                                              Sagat    right
            EHonda left                                             MBison    right
            EHonda left                                             MBison    right

            EHonda v Sagat       0
            Sagat v. EHonda      1
            EHonda v. MBison     2
            MBison v. EHonda     3

            action[odds] needs to go to the other side because our design makes prot actions left

            same with adversary[odds] -- adversary is on the right so adv[ods] is backwards

            '''
            halfway = actions.shape[0] // 2 #halfway split between upper & lower + left & right
            actions, actions_other = mirror_flip_attributes(actions, actions_other)
            ego_log_probs, adv_log_probs = mirror_flip_attributes(ego_log_probs, adv_log_probs)
            values, other_values = mirror_flip_attributes(values, other_values)
                
                
                

            clipped_actions = np.hstack([actions, actions_other])
            # print(clipped_actions, flush=True)
            # print(np.shape(clipped_actions),flush=True)
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, _BoxTypes):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low,
                                          self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)
            #rewards[halfway:] = -rewards[halfway:]
            rewards, rewards_other = mirror_flip_attributes(rewards, rewards_other)
            #np.random.seed(0)
            #random.seed(0)
            #torch.manual_seed(0)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, _DiscreteTypes):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
                actions_other = actions_other.reshape(-1, 1)
            
            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # for idx, done in enumerate(dones):
            #     if (
            #             done
            #             and coordinate_fn is not None
            #     ):
            #         coordinate_fn(infos[idx]["outcome"])
            #     if (
            #             done
            #             and infos[idx].get("terminal_observation") is not None
            #             and infos[idx].get("TimeLimit.truncated", False)
            #     ):
            #         # print(f"[PPO] idx: {idx}, done: {done}, outcome: {infos[idx]['outcome']}", flush=True)
            #         terminal_obs = rollout_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            #         terminal_obs_other = rollout_policy_other.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            #         with th.no_grad():
            #             terminal_value = rollout_policy.predict_values(terminal_obs)[0]
            #             terminal_value_other = rollout_policy_other.predict_values(terminal_obs_other)[0]
            #         rewards[idx] += self.gamma * terminal_value
            #         rewards_other[idx] += self.gamma * terminal_value_other

                    # from IPython import embed; embed()
            rollout_buffer.add(self._last_obs.copy(), actions, rewards, self._last_episode_starts, values,
                                   ego_log_probs)
            for i in range(self.num_adversaries):
                adversary_buffers[i].add(self._last_obs[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv].copy(), actions_other[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv], rewards_other[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv], self._last_episode_starts[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv], other_values[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv],
                                         adv_log_probs[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv])
            #for i in range(self.num_adversaries):
            #    adversary_buffers[i].add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts, values_other,
            #                             adv_log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.value_forward(obs_as_tensor(new_obs, self.device))
            values[halfway:] = -values[halfway:]
            #values_other = rollout_policy_other.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        for i in range(self.num_adversaries):
            adversary_buffers[i].compute_returns_and_advantage(last_values=-values[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv], dones=dones[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv])
        #if self.update_right:
        #    rollout_buffer_other.compute_returns_and_advantage(last_values=values_other, dones=dones)

        callback.on_rollout_end()

        rollout_buffer.prepare_data_for_training()
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].prepare_data_for_training()

        return True

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        run_ego_forward = True,
        run_adv_forward = True,
        update_ego: bool = True,
        update_adversary: bool = False,
        zero_ego_action: bool = False,
        zero_adv_action: bool = False,
        num_perturbs: int = 1,
    ):
        try:
            iteration = 0
            convergence_tracker = None
            if getattr(self, "training_br", False):
                convergence_tracker = BRConvergenceTracker(
                    patience=getattr(self, "br_tracker_patience", 10),
                    tolerance=getattr(self, "br_tracker_tolerance", 1e-4),
                    window_size=getattr(self, "br_tracker_window_size", 50),
                )
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
                if USE_PERTURBED:
                    self._create_all_perturbed_agents(num_perturbs)
                    self._initialize_parallel_updater()                

                if USE_PERTURBED:
                    with ThreadPoolExecutor(max_workers=num_perturbs + 1) as executor:
                        futures = [executor.submit(perturbed_agent.env_perturb_params, run_ego_forward, run_adv_forward, zero_ego_action, zero_adv_action) for perturbed_agent in self.perturbed_agents]
                        perturbed_bufs, perturbed_adv_bufs = zip(*[future.result() for future in futures])
                        future_standard = executor.submit(self.collect_rollouts, self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps, run_ego_forward, run_adv_forward,self.use_mirror, zero_ego_action, zero_adv_action)
                        continue_training = future_standard.result()
                    self.perturbed_bufs = list(perturbed_bufs)
                    self.perturbed_adv_bufs = list(perturbed_adv_bufs)

                    self.perturbed_agents_policy = [perturbed_agent.policy for perturbed_agent in self.perturbed_agents]
                else:
                    continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps, run_ego_forward, run_adv_forward, self.use_mirror, zero_ego_action, zero_adv_action)


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
                        wandb.log({"eval_rew": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])})
                        self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("time/fps", fps)
                    self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                    self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                    self.logger.dump(step=self.num_timesteps)
                if (
                    convergence_tracker is not None
                    and len(self.ep_info_buffer) > 0
                    and len(self.ep_info_buffer[0]) > 0
                ):
                    # Use negative mean reward as an exploitability proxy (lower is better).
                    br_metric = -safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    if convergence_tracker.check(br_metric):
                        print("BR convergence tracker triggered early stopping.")
                        break
            

                self.train(update_ego=update_ego, update_adversary=update_adversary)
                # uncomment perturbed agents
                if USE_PERTURBED:
                    [perturbed_agent.env.close() for perturbed_agent in self.perturbed_agents]
                    self.perturbed_agents.clear()
                    self.perturbed_bufs.clear()
                    self.perturbed_adv_bufs.clear()
                    self.perturbed_agents_policy.clear()


                gc.collect()
                torch.cuda.empty_cache()
                #self.perturbed_agent.env.close()
                #del self.perturbed_agent

            callback.on_training_end()
        
        except Exception as e:
            print(e)
        
        finally:
            #IMPORTANT! Persistent workers must be cleaned up.
            self.cleanup()
            torch.cuda.empty_cache()

        return self
    
    def train(self, update_ego: bool = True, update_adversary: bool = True) -> None:
        if update_ego:
            self.train_standard(update_ego=True, update_adversary=False)
            pass
        if update_adversary:
            self.train_standard(update_ego=False, update_adversary=True)
        #self.train_standard(update_ego, update_adversary)
        if USE_PERTURBED:
            self.train_derivative_free(update_ego, update_adversary)
    
    def train_derivative_free(self, update_ego: bool = True, update_adversary: bool = True) -> None:
        self.policy.set_training_mode(True)
        [self.perturbed_agents[i].policy.set_training_mode(True) for i in range(len(self.perturbed_agents))]

        #self.dummy_policy_update(update_ego, update_adversary)
        
        [self._update_value_functions(perturbed_agent, perturbed_adv_buf) for perturbed_agent, perturbed_adv_buf in zip(self.perturbed_agents, self.perturbed_adv_bufs)]
        futures = []
        # with ThreadPoolExecutor(max_workers=len(self.perturbed_bufs)) as executor:
        #     for policy, perturbed_buf, perturbed_buf_adv in zip(self.perturbed_agents_policy, self.perturbed_bufs, self.perturbed_adv_bufs):
        #         futures.append(executor.submit(self._update_advantages, policy, perturbed_buf, perturbed_buf_adv))
        #     for future in futures:
        #         future.result()
        self.perturbed_agents_policy = [perturbed_agent.policy for perturbed_agent in self.perturbed_agents]
        if update_ego:
            self.leader_grads(self.rollout_buffer, self.perturbed_bufs, self.policy, self.perturbed_agents_policy, ego=True)
        if update_adversary:
            self.leader_grads(self.adversary_buffers, self.perturbed_adv_bufs, self.policy, self.perturbed_agents_policy, ego=False)

    # we need to rewrite leader grads and update_advantages

    def leader_grads(self,
                     ori_buf: AdvRolloutBuffer,
                     perturbed_bufs: Tuple[AdvRolloutBuffer],
                     ori_policy: CleanActorActorCriticPolicy,
                     perturbed_policies: List[CleanActorActorCriticPolicy],
                     ego: bool=True) -> None:
        """TODO: Complete the docstring."""
        ori_policy = unpickle_policy(ori_policy)
        F_grad_temp = []
        if ego is True:
            print("Ego is true", flush=True)
        else:
            print("Ego is false", flush=True)
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_losses, pg_losses, approx_kl_divs_all = [], [], []

        num_runs_count = 1 if ego else self.num_adversaries
        for j in range(self.n_epochs):
            for i in range(num_runs_count):
                # i bug
                F_grad = 0
                futures = []
                num_bufs = self.n_envs * self.n_steps // self.batch_size if ego else self.n_env_per_adv * self.n_steps // self.batch_size
                for perturbed_buf_num in range(num_bufs):
                    F_grad_curr, pg_losses_curr, entropy_losses_curr, approx_kl_divs_curr, break_signal = calc_F_grad_single(ori_policy=ori_policy,
                                        perturbed_policies=perturbed_policies,
                                        ori_buf=ori_buf,
                                        perturbed_bufs=perturbed_bufs,
                                        ego=ego,
                                        i=i,
                                        perturbed_buf_num=perturbed_buf_num,
                                        num_adversaries=self.num_adversaries,
                                        batch_size=self.batch_size,
                                        clip_range=clip_range,
                                        use_sde=self.use_sde,
                                        device=self.device,
                                        envs_per_matchup=self.envs_per_matchup,
                                        d=self.ego_d if ego else self.adv_d,
                                        delta=self.delta,
                                        ego_v=self.ego_v,
                                        adv_v=self.adv_v,
                                        target_kl=self.target_kl,
                                        first_epoch=(j == 0),
                                        )

                        # Collect results
                    
                    if not DEBUG:
                        num_actual_bufs = len(self.perturbed_agents)
                        F_grad = F_grad_curr
                    else:
                        F_grad += F_grad_curr
                    pg_losses.extend(pg_losses_curr)
                    entropy_losses.extend(entropy_losses_curr)
                    approx_kl_divs_all.extend(approx_kl_divs_curr)
                    if break_signal:
                        print("Early stopping due to KL divergence", flush=True)
                        break 
                    


                    if DEBUG:
                        F_grad = F_grad_curr
                        for i in range(len(F_grad)):
                            assert th.max(th.abs(self.ego_grads_autograd_order[i] - F_grad[i])) < 1e-6, "Gradient mismatch"
                    else:
                        #F_grad = [F_grad[0][i]/(num_actual_bufs) for i in range(len(F_grad[0]))]#num_actual_bufs counts how many buffers participated to take the correct average, in case of early stopping.
                        #for i in range(len(F_grad)):
                        #    assert th.max(th.abs(self.ego_grads_autograd_order[perturbed_buf_num][i] - F_grad[i])) < 1e-6, "Gradient mismatch"
                        #F_grad = F_grad_curr
                        pass
                    param_list = self.policy.ctrl_optimizer.param_groups[0]['params'] if ego else self.policy.dstb_optimizer.param_groups[0]['params']
                    size_lists = [list(x.shape) for x in param_list]
                    
                    # reshaped_grad = []
                    # count = 0
                    # for k in range(len(size_lists)):
                    #     numel = np.prod(size_lists[k])
                    #     reshaped_grad.append(torch.reshape(F_grad[count: count + numel], size_lists[k]))
                    #     count += numel
                    if ego is False:
                        # heads_start_index = self.policy.extractor_and_trunk_length
                        # trunk_extractor_indices = [i for i in range(heads_start_index)]
                        # this_adv_indices = [i for i in range(heads_start_index + self.policy.head_length * adv_num , heads_start_index + self.policy.head_length * (adv_num + 1))]
                        # all_indices = trunk_extractor_indices + this_adv_indices
                        self.policy.dstb_optimizer.zero_grad()

                        for k in range(len(F_grad)):
                            self.policy.dstb_optimizer.param_groups[0]['params'][k].grad = F_grad[k].float().detach()
                    else:
                        self.policy.ctrl_optimizer.zero_grad()
                        for k in range(len(F_grad)):
                            self.policy.ctrl_optimizer.param_groups[0]['params'][k].grad = F_grad[k].float().detach()
                    
                    optimizer = self.policy.ctrl_optimizer if ego else self.policy.dstb_optimizer
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        log_prob, _ = self.policy.evaluate_ego_actions(ori_buf.observations, ori_buf.actions) if ego else self.policy.evaluate_adv_actions(ori_buf[i].observations, ori_buf[i].adv_actions, buf_num=[i])
                        log_ratio = log_prob - ori_buf.log_probs if ego else log_prob - ori_buf[i].log_probs
                        # 0 bug
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs_all.append(approx_kl_div)

        self._n_updates += self.n_epochs
        self._update_schedulers(step_ego=ego, step_adv=(not ego), step_val=True, skip=not self.use_lr_annealing)
        if hasattr(self.rollout_buffer, 'values') and self.rollout_buffer.values is not None and self.rollout_buffer.returns is not None:
             explained_var = explained_variance(self.rollout_buffer.values.flatten().detach().cpu().numpy(), self.rollout_buffer.returns.flatten().detach().cpu().numpy())
        else:
            explained_var = np.nan
        if ego is True:
            print("logging ego metrics", flush=True)
        self._log_leader_metrics(ego, entropy_losses, pg_losses, approx_kl_divs_all, explained_var, clip_range)

    def _update_advantages(self, policy, buf, adversary_buffers):
        updated_values = policy.evaluate_states(buf.observations, env_indices=buf.env_indices, buf_num=[i for i in range(self.num_adversaries)])
        buf.values = updated_values.reshape(buf.buffer_size, self.num_adversaries * self.envs_per_matchup).detach().cpu().numpy()
        buf.episode_starts = buf.episode_starts.reshape(buf.buffer_size, self.num_adversaries * self.envs_per_matchup)
        buf.advantages = buf.advantages.reshape(buf.buffer_size, self.num_adversaries * self.envs_per_matchup).detach().cpu().numpy()
        buf.compute_returns_and_advantage(th.from_numpy(buf.values[-1, :]).to(self.device), self._last_episode_starts)
        buf.advantages = buf.swap_and_flatten(buf.advantages)
        buf.values = buf.swap_and_flatten(buf.values)
        buf.returns = buf.swap_and_flatten(buf.returns)


        for i in range(len(adversary_buffers)):
            updated_values = policy.evaluate_states(adversary_buffers[i].observations, env_indices=adversary_buffers[i].env_indices, buf_num=[i])
            updated_values = -updated_values
            adversary_buffers[i].values = updated_values.reshape(adversary_buffers[i].buffer_size, self.envs_per_matchup).detach().cpu().numpy()
            adversary_buffers[i].episode_starts = adversary_buffers[i].episode_starts.reshape(adversary_buffers[i].buffer_size, self.envs_per_matchup)
            adversary_buffers[i].advantages = adversary_buffers[i].advantages.reshape(adversary_buffers[i].buffer_size, self.envs_per_matchup).detach().cpu().numpy()
            adversary_buffers[i].compute_returns_and_advantage(th.from_numpy(adversary_buffers[i].values[-1, :]).to(self.device), self._last_episode_starts[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv])
            adversary_buffers[i].advantages = adversary_buffers[i].swap_and_flatten(adversary_buffers[i].advantages)
            adversary_buffers[i].values = adversary_buffers[i].swap_and_flatten(adversary_buffers[i].values)
            adversary_buffers[i].returns = adversary_buffers[i].swap_and_flatten(adversary_buffers[i].returns)
        pass

    def train_standard(self, update_ego: bool = True, update_adversary: bool = True) -> None:
        self.ego_grads_autograd_order = []
        self.adv_grads_autograd_order = []
        self.policy.adv_grads_autograd_order = []
        self.policy.value_grads_autograd_order = []
        self.policy.value_loss = []
        first = True

        # afk test!
        assert update_ego != update_adversary

        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
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
        num_runs_count = 1 if update_ego else self.num_adversaries
        approx_kl_divs = []
        for i in range(num_runs_count):
            first = True
            if i == 1:
                pass
            if update_adversary:
                buf = self.adversary_buffers[i]
            else:
                buf = self.rollout_buffer
            for epoch in range(self.n_epochs):
                #approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                for rollout_data in buf.get(self.batch_size):
                    if update_ego and not update_adversary:
                        actions = rollout_data.actions
                    else:
                        actions = rollout_data.adv_actions
                    if isinstance(self.action_space, _DiscreteTypes):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)
                    
                    if update_ego:
                        log_prob, entropy = self.policy.evaluate_ego_actions(rollout_data.observations, actions)
                        #from stable_baselines3.common.save_util import load_from_zip_file
                        #data, params, pytorch_variables = load_from_zip_file("test_ego_save.pth", device=self.device)
                        #entropy = ego_entropy
                    if update_adversary:
                        log_prob, entropy = self.policy.evaluate_adv_actions(rollout_data.observations, actions, buf_num=[i])
                        #entropy = adv_entropy
                    if update_ego:
                        values = self.policy.evaluate_states(rollout_data.observations, env_indices=rollout_data.env_indices, buf_num=[i for i in range(self.num_adversaries)])
                    else:
                        values = self.policy.evaluate_states(rollout_data.observations, env_indices=rollout_data.env_indices, buf_num=[i])
                    if update_adversary:
                        values = -values
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    self.normalize_advantage = True
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    #if update_ego:  
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    if first:
                        print(f"[DEBUG @ train]: ratio: {ratio.mean().item():.4f}")
                        #assert th.allclose(log_prob, rollout_data.old_log_prob), "Log probabilities do not match between collection and training."
                        first = False
                    #if update_adversary:
                    #    ratio_adv = th.exp(adv_log_prob - rollout_data.old_dstb_log_prob)

                    # clipped surrogate loss
                    #if update_ego:
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    #if update_adversary:
                    #    policy_loss_adv_1 = advantages * ratio_adv
                    #    policy_loss_adv_2 = advantages * th.clamp(ratio_adv, 1 - clip_range, 1 + clip_range)
                    #    policy_loss_adv = th.min(policy_loss_adv_1, policy_loss_adv_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())# if update_ego else policy_loss_adv.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()# if update_ego else th.mean((th.abs(ratio_adv - 1) > clip_range).float()).item()
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
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())
                    pl = policy_loss#_ego if update_ego else policy_loss_adv
                    self.ego_params = self.policy.ctrl_optimizer.param_groups[0]['params']
                    loss = pl + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        #if update_ego:
                        #    log_ratio = ego_log_prob - rollout_data.old_log_prob
                        #else:
                        #    log_ratio = adv_log_prob - rollout_data.old_dstb_log_prob
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    self.policy.ctrl_optimizer.zero_grad()
                    self.policy.dstb_optimizer.zero_grad()
                    self.policy.value_optimizer.zero_grad()
                    loss.backward()

                    self.ego_grads_autograd_order.append([self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params']))])
                    self.policy.adv_grads_autograd_order.append([self.policy.dstb_optimizer.param_groups[0]['params'][i].grad for i in range(len(self.policy.dstb_optimizer.param_groups[0]['params']))])
                    self.policy.value_grads_autograd_order.append([self.policy.value_optimizer.param_groups[0]['params'][i].grad for i in range(len(self.policy.value_optimizer.param_groups[0]['params']))])
                    self.policy.value_loss.append(value_loss)
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if update_ego:
                        self.policy.ctrl_optimizer.step()
                    else:
                        self.policy.dstb_optimizer.step()
                    self.policy.value_optimizer.step()

                if not continue_training:
                    break
        self._update_schedulers(step_ego=update_ego, step_adv=(not update_ego), step_val=True, skip=not self.use_lr_annealing)
        # check location in train derivative free
        self._n_updates += self.n_epochs
        if th.is_tensor(buf.values):
            explained_var = explained_variance(buf.values.flatten().cpu().numpy(), buf.returns.flatten().cpu().numpy())
        else:
            explained_var = explained_variance(buf.values, buf.returns)
        self.policy.num_adversaries = self.num_adversaries

        # Logs
        #self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        #self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        #self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        #self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        #self.logger.record("train/loss", loss.item())
        #self.logger.record("train/explained_variance", explained_var)
        #if hasattr(self.policy, "log_std"):
        #    self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        #self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        #self.logger.record("train/clip_range", clip_range)
        #if self.clip_range_vf is not None:
        #    self.logger.record("train/clip_range_vf", clip_range_vf)
        
        self._log_leader_metrics(update_ego, entropy_losses, pg_losses, approx_kl_divs, explained_var, clip_range)
    def perturb_params(self, param_list, ego=True):
        count = 0
        for i in range(len(param_list)):
            count = count + torch.numel(param_list[i])
        delta = .5
        select = torch.from_numpy(np.random.uniform(low=-1, high=1, size=count)).to(self.device)
        v = delta * select / torch.linalg.norm(select)
        self.delta = delta
        if ego:
            self.ego_v = v
        else:
            self.adv_v = v
        # this works because we call leader_grads TWICE, once for ego and once for adv, so 
        # each time, we use a diff v and update each param list, so no need to double d here.
        if ego:
            self.ego_d = count
        else:
            self.adv_d = count
        count = 0
        with torch.no_grad():
            for p in param_list:
                p.copy_(p + torch.reshape(v[count:count + torch.numel(p)], p.shape).to(self.device))
                count = count + torch.numel(p)
        return
    
    def env_perturb_params(self, update_ego=True, update_adversary=True, zero_ego_action=False, zero_adv_action=False):
        buf = self.rollout_buffer_class(self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,)
        #buf = deepcopy(self.rollout_buffer)
        #buf.reset()
        #adv_buf = deepcopy(self.adversary_buffers)
        adv_buf = [self.rollout_buffer_class(self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs= self.n_env_per_adv) for i in range(self.num_adversaries)]
        #[adv_buf[i].reset() for i in range(len(adv_buf))]
        self.collect_rollouts(self.env, self.callback, buf, adv_buf, n_rollout_steps=self.n_steps, run_ego_forward=update_ego, run_adv_forward=update_adversary, use_mirror=self.use_mirror, zero_ego_action=zero_ego_action, zero_adv_action=zero_adv_action)
        
        #buf.prepare_data_for_training()
        #for i in range(len(adv_buf)):
        #    adv_buf[i].prepare_data_for_training()
            
        return buf, adv_buf

    def _update_value_functions(self, perturbed_agent, perturbed_adv_buf) -> None:
        """
        Updates value functions either serially (CPU or 1 GPU) or in parallel across multiple GPUs.

        Args:
            perturbed_agent:
                The agent with perturbed policy and its own buffer (`perturbed_adv_buf`).
            perturbed_adv_buf:
                Perturbed adversarial buffer.

        Returns:
            None
        """
        total_start_time = time.time()
        # Create updaters
        init_start_time = time.time()
        self._initialize_parallel_updater()
        init_end_time = time.time()
        if TIMING:
            print(f"    [Timing] _initialize_parallel_updater: {init_end_time - init_start_time:.4f}s")

        #The policies will be deeopcopied and so they won't have num_global_env, so these values need to be populated here
        self.policy.num_global_env = self.n_global_env
        perturbed_agent.policy.num_global_env = perturbed_agent.n_global_env
        
        update_start_time = time.time()
        results = self.parallel_updater.update_value_functions(
            self.policy, perturbed_agent, perturbed_adv_buf,
            self.adversary_buffers, self.batch_size, self.max_grad_norm,
            self.n_epochs, self.n_env_per_adv, self.first_run, self.envs_per_matchup
        )

        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            warnings.warn("No results from value function update workers.")
            return
        
        if len(valid_results) > 1:
            unperturbed_param_averages = {}
            perturbed_param_averages = {}
            for key in valid_results[0][0].keys():
                
                unperturbed_param_averages[key] = sum(result[0][key].to('cpu') for result in valid_results) / len(valid_results)
                perturbed_param_averages[key] = sum(result[1][key].to('cpu') for result in valid_results) / len(valid_results)
            valid_results = [(unperturbed_param_averages, perturbed_param_averages)]

        assert len(valid_results) == 1, f"Expected 1 result, got {len(valid_results)}"
        # Load the state dicts from the last valid result
        last_spar_state_dict, last_perturbed_state_dict = valid_results[-1]
        self.policy.load_state_dict(last_spar_state_dict)
        perturbed_agent.policy.load_state_dict(last_perturbed_state_dict)

        update_end_time = time.time()
        if TIMING:
            print(f"    [Timing] parallel_updater.update_value_functions: {update_end_time - update_start_time:.4f}s")

        self.policy.num_global_env = self.n_global_env
        perturbed_agent.policy.num_global_env = perturbed_agent.n_global_env
        self.first_run = False
        
        total_end_time = time.time()
        if TIMING:
            print(f"  [Timing] Total _update_value_functions: {total_end_time - total_start_time:.4f}s")

    def inner_loop(self):
        # 1. Create and configure the perturbed agent
        start_time = time.time()
        perturbed_agent, other_ego, other_adv = self._create_perturbed_agent()
        end_time = time.time()
        if TIMING:
            print(f"Time for _create_perturbed_agent: {end_time - start_time:.4f}s")
        
        # 2. Collect rollouts using the perturbed agent
        start_time = time.time()
        perturbed_buf, perturbed_adv_buf = perturbed_agent.env_perturb_params()
        end_time = time.time()
        if TIMING:
            print(f"Time for env_perturb_params: {end_time - start_time:.4f}s")
        self.perturbed_buf = perturbed_buf
        self.perturbed_adv_buf = perturbed_adv_buf

        # 3. Update value functions for both original and perturbed agents
        start_time = time.time()
        self._update_value_functions(perturbed_agent, perturbed_adv_buf)
        end_time = time.time()
        if TIMING:
            print(f"Time for _update_value_functions: {end_time - start_time:.4f}s")

        self.perturbed_agent_policy = perturbed_agent.policy

    def _create_perturbed_agent(self):
        # Deepcopy and perturb parameters for both ego and adversary policies
        other_ego = deepcopy(self.policy.ctrl_optimizer.param_groups[0]['params'])
        other_adv = deepcopy(self.policy.dstb_optimizer.param_groups[0]['params'])
        self.perturb_params(other_ego, ego=True)
        self.perturb_params(other_adv, ego=False)
        ego_norm = torch.linalg.norm(self.ego_v)
        adv_norm = torch.linalg.norm(self.adv_v)
        self.ego_v = self.ego_v / (ego_norm + adv_norm)
        self.adv_v = self.adv_v / (ego_norm + adv_norm)
        
        # Create a new agent instance with the perturbed parameters
        perturbed_agent = self.copy_constructor()
        perturbed_agent.policy.gamma = self.gamma
        with torch.no_grad():
            for i in range(len(perturbed_agent.policy.dstb_optimizer.param_groups[0]['params'])):
                #perturbed_agent.policy.ctrl_optimizer.param_groups[0]['params'][i].copy_(other_ego[i])
                perturbed_agent.policy.dstb_optimizer.param_groups[0]['params'][i].copy_(other_adv[i])
            for i in range(len(perturbed_agent.policy.ctrl_optimizer.param_groups[0]['params'])):
                perturbed_agent.policy.ctrl_optimizer.param_groups[0]['params'][i].copy_(other_ego[i])
        perturbed_agent.env = self._create_separate_env()
        # Since we have a new environment, we need new initial observations
        perturbed_agent._last_obs = perturbed_agent.env.reset()
        perturbed_agent._last_episode_starts = np.ones((perturbed_agent.env.num_envs,), dtype=bool)        
        return perturbed_agent, other_ego, other_adv

    def _create_all_perturbed_agents(self, num_perturbs: int) -> None:
        """This function creates perturbed agents and stores them in self."""
        #Don't create the perturbed agents if they already exist.
        if getattr(self, "perturbed_agents", None):
            return

        with ThreadPoolExecutor(max_workers=num_perturbs) as executor:
            futures = [executor.submit(self._create_perturbed_agent) for _ in range(num_perturbs)]
            perturbed_agents = [future.result()[0] for future in futures]
        self.perturbed_agents = perturbed_agents

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
                matchup_key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
                test.policy.value_net[matchup_key] = test.policy.value_net[matchup_key].to(test.device)
                test.policy.dstb_action_net[matchup_key] = test.policy.dstb_action_net[matchup_key].to(test.device)
        test.policy.ctrl_optimizer = self.policy.optimizer_class(test.policy.ctrl_optimizer.param_groups[0]['params'], maximize=True)
        test.policy.dstb_optimizer = self.policy.optimizer_class(test.policy.dstb_optimizer.param_groups[0]['params'], maximize=False)
        test.policy.value_optimizer = self.policy.optimizer_class(test.policy.value_optimizer.param_groups[0]['params'])
        for i in range(len(self.adversary_buffers)):
            self.adversary_buffers[i].reset()
        test.adversary_buffers = deepcopy(self.adversary_buffers)
        test.rollout_buffer = deepcopy(self.rollout_buffer.reset())
        if retain_callback is True:
            pass
        else:
            test.callback = ConvertCallback(None)
            test.callback.init_callback(test)
        test.policy = test.policy.to(self.device)
        # Copy observation states
        test._last_obs = self._last_obs.copy() if self._last_obs is not None else None
        test._last_episode_starts = self._last_episode_starts.copy() if self._last_episode_starts is not None else None
        test.policy.num_env_per_adv = self.envs_per_matchup
        return test
    
    def _create_separate_env(self):
        """Create a new environment instance using the stored generator function"""
        if self.env_generator_func is None:
            raise ValueError("No environment generator function provided")
        new_env = self.env_generator_func(args=self.game_args, STATE=self.state_list)
        new_env.reset()
        return new_env

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from save.
        """
        excluded = super()._excluded_save_params()
        excluded.extend(["parallel_updater", "callback", "perturbed_agents"])
        return excluded
    
    def cleanup(self):
        """
        Manually shutdown parallel workers when done.
        NOTE: This CANNOT be done in a destroctur, as the object my be killed earlier.
        """
        if hasattr(self, 'parallel_updater') and self.parallel_updater is not None:
            self.parallel_updater.shutdown()
            self.parallel_updater = None

    def _initialize_parallel_updater(self) -> None:
        """This function initializes the ParallelUpdater"""
        if self.parallel_updater is None:
            _, n_workers = get_n_workers()
            self.parallel_updater = ParallelUpdater(n_workers)
            self.first_run = True 

    def _log_leader_metrics(self, ego, entropy_losses, pg_losses, approx_kl_divs, explained_var, clip_range):
        prefix = "ego" if ego else "adv"

        self.logger.record(f"train/{prefix}_entropy_loss", np.mean(entropy_losses))
        self.logger.record(f"train/{prefix}_policy_gradient_loss", np.mean(pg_losses))
        self.logger.record(f"train/{prefix}_approx_kl", np.mean(approx_kl_divs))
        self.logger.record(f"train/{prefix}_explained_variance", explained_var)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            clip_range_vf_val = self.clip_range_vf(self._current_progress_remaining)
            self.logger.record("train/clip_range_vf", clip_range_vf_val)

    def dummy_policy_update(self, update_ego=True, update_adversary=True):
        first = True

        # afk test!
        assert update_ego != update_adversary

        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        if update_ego:
            buf = self.rollout_buffer
        else:
            self.policy.num_adversaries = 1
            buf = self.adversary_buffers[0]


        # train for n_epochs epochs
        num_runs_count = 1 if update_ego else self.num_adversaries
        for i in range(num_runs_count):
            if update_adversary:
                buf = self.adversary_buffers[i]
            else:
                buf = self.rollout_buffer
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                for rollout_data in buf.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, _DiscreteTypes):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    if update_ego:
                        log_prob, entropy = self.policy.evaluate_ego_actions(rollout_data.observations, actions)
                        #entropy = ego_entropy
                    if update_adversary:
                        log_prob, entropy = self.policy.evaluate_adv_actions(rollout_data.observations, actions, buf_num=[i])
                        #entropy = adv_entropy
                    if update_ego:
                        values = self.policy.evaluate_states(rollout_data.observations, env_indices=rollout_data.env_indices, buf_num=[i for i in range(self.num_adversaries)])
                    else:
                        values = self.policy.evaluate_states(rollout_data.observations, env_indices=rollout_data.env_indices, buf_num=[i])
                    if update_adversary:
                        values = -values
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    self.normalize_advantage = True
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    #if update_ego:  
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    if first:
                        #print(f"[DEBUG @ train]: ratio: {ratio.mean().item():.4f}")
                        assert th.allclose(log_prob, rollout_data.old_log_prob)
                        first = False
                    #if update_adversary:
                    #    ratio_adv = th.exp(adv_log_prob - rollout_data.old_dstb_log_prob)

                    # clipped surrogate loss
                    #if update_ego:
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    #if update_adversary:
                    #    policy_loss_adv_1 = advantages * ratio_adv
                    #    policy_loss_adv_2 = advantages * th.clamp(ratio_adv, 1 - clip_range, 1 + clip_range)
                    #    policy_loss_adv = th.min(policy_loss_adv_1, policy_loss_adv_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())# if update_ego else policy_loss_adv.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()# if update_ego else th.mean((th.abs(ratio_adv - 1) > clip_range).float()).item()
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
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())
                    pl = policy_loss#_ego if update_ego else policy_loss_adv
                    loss = pl
                    loss.backward()
                    self.policy.ctrl_optimizer.zero_grad()
                    self.policy.ctrl_optimizer.step()

    @classmethod
    def load(cls, path: str, num_perturbed: int, **kwargs):
        model = super().load(path, **kwargs)
        #model._create_all_perturbed_agents(num_perturbed)
        #TODO: Add a function that creates a callback and assigns it to self. Something like model._create_callback or passed in as an argument.
        return model
