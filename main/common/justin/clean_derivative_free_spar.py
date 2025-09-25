import torch
import torch as th
import torch.autograd as autograd
import sys
import time
import random
import math
import pickle
from venv import create
import wandb
from copy import deepcopy
import gc
from queue import Empty
import warnings
from typing import Union, Type, Optional, Dict, Any, List
from stable_baselines3.common.callbacks import ConvertCallback
from torch.multiprocessing import Process, Queue
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.clean_new_policies import CleanActorActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer
from utils import state2matchup
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance
from common.justin.Doubly_TSS_SPAR import Doubly_TSS_SPAR as dtss
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from anyio import value
from gym import spaces
from stable_baselines3 import PPO
from utils import select_matchup_env, select_device, get_n_workers, move_policy

TIMING = False

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

def _update_single_value_function(batch_size: int, max_grad_norm: float, policy, buffer, adversary_index: int, num_envs: int, device: torch.device, tag: str="", envs_per_matchup: int=None):
    """
    This function has to be placed outside of the object to enable parallel calls.
    TODO: Complete the docstring.
    TODO: Complete static types
    """
    def _prep_rollout_data_actions(batch_size: int, buffer) -> tuple:
        """
        This is a helper function that gets all the rollout data and actions once instead of batch by batch.
        """
        all_rollout_data = list(buffer.get(batch_size))
        all_actions = []
        all_dstb_actions = []
        all_observations = []
        all_returns = []
        all_env_indices = []

        for rollout_data in all_rollout_data:
            all_actions.append(torch.Tensor(rollout_data.actions))
            all_dstb_actions.append(torch.Tensor(rollout_data.dstb_actions))
            all_observations.append(rollout_data.observations)
            all_returns.append(torch.Tensor(rollout_data.returns))
            all_env_indices.extend(rollout_data.env_indices)
        
        actions_batch = torch.cat(all_actions).to(device)
        dstb_actions_batch = torch.cat(all_dstb_actions).to(device)
        observations_batch = torch.cat(all_observations).to(device)
        returns_batch = torch.cat(all_returns).to(device)

        return actions_batch, dstb_actions_batch, observations_batch, returns_batch, all_env_indices
    
    total_start_time = time.time()


    #Process all rollout data and actions at once instead of batch by batch.
    actions_batch, dstb_actions_batch, observations_batch, returns_batch, all_env_indices = _prep_rollout_data_actions(batch_size, buffer)
    policy.num_global_env = num_envs
    policy.num_adv = 1
    for i in range(len(returns_batch) // batch_size):
        values, _, _, _, _ = policy.evaluate_actions(
        observations_batch[i * batch_size:(i + 1) * batch_size],
        actions_batch[i * batch_size:(i + 1) * batch_size],
        dstb_actions_batch[i * batch_size:(i + 1) * batch_size],
        shuffle_keys=all_env_indices[i * batch_size:(i + 1) * batch_size],
        network_keys=[adversary_index], envs_per_matchup=envs_per_matchup
        )
        #policy.train(True)
        #torch.backends.cudnn.enabled = False
        values = values.flatten()
        offset = 12 # vf extractor and shared trunk are 12
        num_per_head = 10 # lstm = 6, 2 linear layers = 2 + 2, total 10
        value_loss = F.mse_loss(values, returns_batch[i * batch_size:(i + 1) * batch_size])
        indices = list(range(0, offset)) + list(range(offset + adversary_index * num_per_head, offset + (adversary_index + 1) * num_per_head))
        value_grads = th.autograd.grad(value_loss, [policy.value_optimizer.param_groups[0]['params'][j] for j in indices])
        #value_grads = th.cat([grad.view(-1) for grad in value_grads])
        policy.value_optimizer.zero_grad()
        for i in range(len(value_grads)):
            policy.value_optimizer.param_groups[0]['params'][indices[i]].grad = value_grads[i]
        #policy.value_optimizer.zero_grad()
        #for i in range(len(policy.value_optimizer.param_groups[0]['params'])):
        #    policy.value_optimizer.param_groups[0]['params'][i].grad = value_grads[i]
        #value_loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy.value_optimizer.step()

    total_end_time = time.time()
    if TIMING:
        print(f"      [Timing] Total _update_single_value_function ({tag}): {total_end_time - total_start_time:.4f}s")

class ParallelUpdater:
    """
    Manages persistent worker processes for parallel value function updates on multiple GPUs.
    
    Creates worker processes once and reuses them for subsequent calls, avoiding the overhead
    of process creation. Uses proper synchronization to wait for job completion.


    To add a new job type:
    1. Add job handler static method: _handle_your_job_type(job, device_id, done_queue, ...) -> Any
    2. Add elif case in _generic_worker_function: elif job_type == "YOUR_JOB_TYPE": ...
    3. Add necessary persistent state variables to persistent_state in _generic_worker_function (if needed).
    4. Add job creation method: _create_your_job_type_job(...) -> tuple (NOTE: This might not be necessary for every job)
    5. Add public method: your_job_type(self, active_jobs, ...) -> None (uses _submit_job and _wait_for_jobs). This method should use the _parallel_job_executor decorator (@_parallel_job_executor) and have jobs submitted to active_jobs. Use self.`_submit_job` with active_jobs.
    """
    
    def __init__(self, n_workers: int) -> None:
        """
        Initialize the parallel updater with persistent worker processes.
        
        Args:
            n_workers: Number of GPUs/workers to create
        """
        self.n_workers: int = n_workers
        self.processes: List[Process] = []
        self.input_queues: List[Queue] = []
        self.done_queue: Queue = Queue()  # Shared queue for completion signals
        self._initialize_processes()

    def _initialize_processes(self) -> None:
        """Initialize persistent worker processes once."""
        for device_id in range(self.n_workers):
            input_queue = Queue()
            # Create a custom worker that uses our generic function
            worker = Process(target=ParallelUpdater._generic_worker_function, 
                            args=(input_queue, self.done_queue, device_id))
            worker.daemon = False
            worker.start()
            self.processes.append(worker)
            self.input_queues.append(input_queue)

    @staticmethod
    def _generic_worker_function(input_queue: Queue, done_queue: Queue, device_id: int) -> None:
        """Generic worker that can handle different job types."""

        persistent_state = {}  # Worker-local persistent state - 

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.set_device(device_id)

        while True:
            try:
                job = input_queue.get(timeout=1)
            except Empty:
                continue
            if job == "STOP":
                print(f"Worker {device_id}: Received STOP signal, exiting")
                break

            if not isinstance(job, tuple) or len(job) < 2:
                done_queue.put(f"ERROR_INVALID_JOB_FORMAT")
                continue
                
            job_type = job[0]
            
            #NOTE: Add if statements here to be able to handle new jobs.
            if job_type == "UPDATE_VALUE_FUNCTIONS":
                ParallelUpdater._handle_update_value_functions(job, device_id, done_queue, persistent_state)

            else:
                print(f"Worker {device_id}: Unknown job type: {job_type}")
                done_queue.put(f"ERROR_UNKNOWN_JOB_TYPE_{job_type}")
    
    @staticmethod
    def _handle_update_value_functions(job: tuple, device_id: int, done_queue: Queue, persistent_state: dict):
        """
        Handle UPDATE_VALUE_FUNCTIONS job type.

        Processes value function updates for both main and perturbed policies on a specific device.
        Manages model loading/updating, moves models to appropriate device, and executes training loops.

        Args:
            job: (tuple) 
                A tuple containing (job_type, job_data) where job_data contains all training parameters
            device_id (int):
                GPU device ID for this worker
            done_queue: (Queue)
                A queue for signaling job completion or errors
            persistent_state: (dict)
                A dictionary storing worker-local persistent state (models, device, etc.)

        Returns:
            None: Updates persistent_state in-place and signals completion via done_queue
        """
        derivative_free_SPAR_policy = persistent_state.get('derivative_free_SPAR_policy')
        perturbed_agent_policy = persistent_state.get('perturbed_agent_policy')
        device = select_device(device_id)
        # Unpack the original job format for value function updates
        (
            derivative_free_SPAR_policy_data,
            perturbed_agent_policy_data,
            batch_size,
            max_grad_norm,
            i_list,
            adversary_buffers,
            perturbed_adv_buf,
            n_epochs,
            n_env_per_adv,
            n_env_per_pert,
            envs_per_matchup,
            job_id,
        ) = job[1]

        try:
            # Initialize models once (first run)
            if derivative_free_SPAR_policy is None:
                if isinstance(derivative_free_SPAR_policy_data, bytes):
                    derivative_free_SPAR_policy = pickle.loads(derivative_free_SPAR_policy_data)
                    perturbed_agent_policy = pickle.loads(perturbed_agent_policy_data)
                    persistent_state['derivative_free_SPAR_policy'] = derivative_free_SPAR_policy
                    persistent_state['perturbed_agent_policy'] = perturbed_agent_policy
                else:
                    # Handle case where first run sends state dict instead of pickled model
                    raise RuntimeError("First run should send pickled models, not state dicts")
                move_policy(derivative_free_SPAR_policy, device)
                move_policy(perturbed_agent_policy, device)
            else:
                # Update weights (subsequent runs)
                if isinstance(derivative_free_SPAR_policy_data, bytes):
                    derivative_free_SPAR_policy = pickle.loads(derivative_free_SPAR_policy_data)
                    perturbed_agent_policy = pickle.loads(perturbed_agent_policy_data)
                    move_policy(derivative_free_SPAR_policy, device)
                    move_policy(perturbed_agent_policy, device)
                else:
                    derivative_free_SPAR_policy.load_state_dict(derivative_free_SPAR_policy_data)
                    perturbed_agent_policy.load_state_dict(perturbed_agent_policy_data)
            persistent_state['derivative_free_SPAR_policy'] = derivative_free_SPAR_policy
            persistent_state['perturbed_agent_policy'] = perturbed_agent_policy

            # Do the actual work
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for i in i_list:
                for epoch in range(n_epochs):
                    _update_single_value_function(
                        batch_size, max_grad_norm, derivative_free_SPAR_policy, 
                        adversary_buffers[i], i, n_env_per_adv, device, envs_per_matchup=envs_per_matchup
                    )
                    _update_single_value_function(
                        batch_size, max_grad_norm, perturbed_agent_policy, 
                        perturbed_adv_buf[i], i, n_env_per_pert, device, envs_per_matchup=envs_per_matchup
                    )
            
            # Signal completion
            done_queue.put(job_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Worker {device_id} error: {e}")
            done_queue.put(f"ERROR_{job_id}")

    def _submit_job(self, job: str, device_id: int, active_jobs: List[int], *args) -> None:
        """
        This function is used to submit job to the ParallelUpdater.

        Args:
            job (str):
                The job to submit.
            device_id (int):
                Device ID to submit the job to.
            active_jobs (list):
                A list of all the active jobs to wait for. The submitted job will be appended to this list.
        """
        def _gen_job_id(device_id: int=0) -> str:
            """
            This helper function generates a random job ID.
            """
            return f"job_{device_id}_{int(time.time() * 1000000 + random.randint(0, 10000))}"
        
        job_id = _gen_job_id(device_id=device_id)
        self.input_queues[device_id].put((job, args + (job_id,)))
        active_jobs.append(job_id)
    
    def _create_update_values_function_job(self, policy: Any, perturbed_policy: Any, i_list: List[int], 
                   adversary_buffers: List[Any], perturbed_adv_buf: List[Any], 
                   batch_size: int, max_grad_norm: float, n_epochs: int, 
                   n_env_per_adv: int, n_env_per_pert: int, first_run: bool, envs_per_matchup: int) -> tuple:
        """
        Create job data for worker process - update_values_function.
        
        Args:
            policy: Main policy model
            perturbed_policy: Perturbed policy model
            i_list: List of adversary indices to process
            adversary_buffers: List of adversary buffers
            perturbed_adv_buf: List of perturbed adversary buffers
            batch_size: Batch size for training
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of training epochs
            n_env_per_adv: Number of environments per adversary
            n_env_per_pert: Number of environments per perturbed agent
            first_run: Whether this is the first run (send full models vs state dicts)
            envs_per_matchup: Number of environments per matchup
            
        Returns:
            Tuple containing all job data for the worker
        """
        if first_run:
            # First run: send full pickled models
            policy_data = pickle.dumps(policy)
            perturbed_policy_data = pickle.dumps(perturbed_policy)
        else:
            # Subsequent runs: send only state dicts
            policy_data = policy.state_dict()
            perturbed_policy_data = perturbed_policy.state_dict()

        return (
            policy_data,
            perturbed_policy_data,
            batch_size,
            max_grad_norm,
            i_list,
            adversary_buffers,
            perturbed_adv_buf,
            n_epochs,
            n_env_per_adv,
            n_env_per_pert,
            envs_per_matchup,
        )
    
    def _wait_for_jobs(self, active_jobs: List[str]) -> None:
        """
        This function waits for all the jobs submitted to active_jobs.

        Args:
            active_jobs (list[str]):
                List of active jobs.
        """
        completed_jobs = 0
        while completed_jobs < len(active_jobs):
            try:
                result = self.done_queue.get(timeout=60)  # 60 second timeout
                if isinstance(result, str) and result.startswith("ERROR_"):
                    print(f"Job failed: {result}")
                completed_jobs += 1
            except Empty:
                print("Warning: Timeout waiting for job completion")
                break
    
    def _parallel_job_executor(func):
        """
        This is a decorator to use 
        """
        def wrapper(self, *args, **kwargs):
            active_jobs = []
            func(self, active_jobs, *args, **kwargs)
            self._wait_for_jobs(active_jobs)
        return wrapper

    @_parallel_job_executor
    def update_value_functions(self, active_jobs: list, policy: Any, perturbed_agent: Any, perturbed_adv_buf: List[Any], 
                             adversary_buffers: List[Any], batch_size: int, max_grad_norm: float, 
                             n_epochs: int, n_env_per_adv: int, first_run: bool = False, envs_per_matchup: int = None) -> None:
        """
        Submit work to persistent processes and wait for completion.
        
        Args:
            policy: Main policy model
            perturbed_agent: Agent containing perturbed policy
            perturbed_adv_buf: List of perturbed adversary buffers
            adversary_buffers: List of main adversary buffers
            batch_size: Batch size for training
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of training epochs
            n_env_per_adv: Number of environments per adversary
            first_run: Whether this is the first run (affects data serialization)
        """
        # Set required attributes
        policy.num_global_env = n_env_per_adv
        perturbed_agent.policy.num_global_env = perturbed_agent.n_env_per_adv

        # Shard work across GPUs
        all_indices = list(range(len(adversary_buffers)))
        shards = shard_indices(len(all_indices), self.n_workers)

        # Submit jobs to worker processes
        for device_id, i_list in enumerate(shards):
            if not i_list:
                continue
            
            job = self._create_update_values_function_job(
                policy, perturbed_agent.policy, i_list, adversary_buffers,
                perturbed_adv_buf, batch_size, max_grad_norm, n_epochs,
                n_env_per_adv, perturbed_agent.n_env_per_adv, first_run, envs_per_matchup
            )
            self._submit_job("UPDATE_VALUE_FUNCTIONS", device_id, active_jobs, *job)

    def shutdown(self) -> None:
        """Clean shutdown of worker processes."""
        for queue in self.input_queues:
            queue.put("STOP")
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

class CleanDerivativeFreeSPAR(PPO):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "AACCnnPolicy": CleanActorActorCriticPolicy
    }
    def __init__(self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            c_learning_rate: Union[float, Schedule] = 1e-4,
            d_learning_rate: Union[float, Schedule] = 2e-4,
            v_learning_rate: Union[float, Schedule] = 7e-4,
            c_learning_rate_decay: Union[float, Schedule] = 1e-4,
            d_learning_rate_decay: Union[float, Schedule] = 2e-4,
            v_learning_rate_decay: Union[float, Schedule] = 7e-4,
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
            self.env.num_envs = self.n_envs
            self._setup_model()
        self.env_generator_func = env_generator_func
        self.parallel_updater = None
        self.n_global_env = self.n_envs
        adversary_buffers = []
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
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer
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

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )

        self.policy = self.policy.to(self.device)
    
    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, adversary_buffers, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        #rollout_policy = self.policy if policy is None else policy
        #rollout_policy_other = self.policy_other if policy_other is None else policy_other
        #rollout_policy.set_training_mode(False)
        #rollout_policy_other.set_training_mode(False)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        for i in range(self.num_adversaries):
            adversary_buffers[i].reset()
        #rollout_buffer_other.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            rollout_policy.reset_noise(env.num_envs)
            rollout_policy_other.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                rollout_policy.reset_noise(env.num_envs)
                rollout_policy_other.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                ego_actions, ego_log_probs, adv_actions, adv_log_probs, values = self.policy(obs_tensor, deterministic=False, ego_forward=False, adv_forward=True)
                other_values = -values
                #actions_other, values_other, log_probs_other = rollout_policy_other(obs_tensor)
            actions = ego_actions.cpu().numpy()
            actions_other = adv_actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = np.hstack([actions, actions_other])
            # print(clipped_actions, flush=True)
            # print(np.shape(clipped_actions),flush=True)
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low,
                                          self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)

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
        update_ego: bool = True,
        update_adversary: bool = True,
    ):
        #try:
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
            #perturbed_agent, other_ego, other_adv = self._create_perturbed_agent()
            print("perturbed agent created!", flush=True)
            #self._initialize_parallel_updater() 
                 
            self.inner_loop()
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps) #TODO: This is sequential - remove when done.
            # perturbed_buf, perturbed_adv_buf = perturbed_agent.env_perturb_params() #TODO: This is a sequential original line, delete it when done.
            #continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps) #TODO: This is sequential - remove when done.

            # Run env_perturb_params and collect_rollouts in different threads (cannot be done in different processes because they contain unpickleable objects)
            # with ThreadPoolExecutor(max_workers=2) as executor:
            #     future_perturbed = executor.submit(perturbed_agent.env_perturb_params)
            #     future_collect = executor.submit(self.collect_rollouts, self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps)
                
            #     perturbed_buf, perturbed_adv_buf = future_perturbed.result()
            #     continue_training = future_collect.result()
            # self.perturbed_agent = perturbed_agent
            # self.perturbed_buf = perturbed_buf
            # self.perturbed_adv_buf = perturbed_adv_buf
            # self.perturbed_agent_policy = perturbed_agent.policy
            # print("main agent and perturbed agent rollout done!", flush=True)
            
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
                    wandb.log({"eval_rew": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])})
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
        

            self.train(update_ego=False, update_adversary=True)
            #self.perturbed_agent.env.close()
            #del self.perturbed_agent

        callback.on_training_end()
        
        # finally:
        #     #IMPORTANT! Persistent workers must be cleaned up.
        #     self.cleanup()
        #     torch.cuda.empty_cache()

        #except Exception as e:
        #    print(e)
        return self
    
    def train(self, update_ego: bool = True, update_adversary: bool = True) -> None:
        #self.train_standard(update_ego, update_adversary)
        self.train_derivative_free(update_ego, update_adversary)
    
    def train_derivative_free(self, update_ego: bool = True, update_adversary: bool = True) -> None:
        #self._update_value_functions(self.policy, self.rollout_buffer, self.adversary_buffers)
        self._update_advantages(self.policy, self.rollout_buffer, self.adversary_buffers)
        self.leader_grads(self.rollout_buffer, self.adversary_buffers, self.policy, self.policy, ego=True)
        self.leader_grads(self.adversary_buffers, self.rollout_buffer, self.policy, self.policy, ego=False)
        #self.update_advantages(self.policy, self.rollout_buffer, self.adversary_buffers)
        #self.update_advantages(self.policy, self.rollout_buffer, self.adversary_buffers)
        self.perturbed_agent_policy = self.perturbed_agent.policy

    # we need to rewrite leader grads and update_advantages
    def leader_grads(self, ori_buf, perturbed_buf, ori_policy, perturbed_policy, ego=True):
        pass
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
            adversary_buffers[i].values = updated_values.reshape(adversary_buffers[i].buffer_size, self.envs_per_matchup).detach().cpu().numpy()
            adversary_buffers[i].episode_starts = adversary_buffers[i].episode_starts.reshape(adversary_buffers[i].buffer_size, self.envs_per_matchup)
            adversary_buffers[i].advantages = adversary_buffers[i].advantages.reshape(adversary_buffers[i].buffer_size, self.envs_per_matchup).detach().cpu().numpy()
            adversary_buffers[i].compute_returns_and_advantage(th.from_numpy(adversary_buffers[i].values[-1, :]).to(self.device), self._last_episode_starts[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv])
            adversary_buffers[i].advantages = adversary_buffers[i].swap_and_flatten(adversary_buffers[i].advantages)
            adversary_buffers[i].values = adversary_buffers[i].swap_and_flatten(adversary_buffers[i].values)
            adversary_buffers[i].returns = adversary_buffers[i].swap_and_flatten(adversary_buffers[i].returns)
        pass

    def train_standard(self, update_ego: bool = True, update_adversary: bool = True) -> None:
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
            buf = self.adversary_buffers[1]


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
                    if isinstance(self.action_space, spaces.Discrete):
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
                        print(f"[DEBUG @ train]: ratio: {ratio.mean().item():.4f}")
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
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if update_ego:
                        self.policy.ctrl_optimizer.step()
                    else:
                        self.policy.dstb_optimizer.step()
                    self.policy.value_optimizer.step()

                if not continue_training:
                    break

        self._n_updates += self.n_epochs
        if th.is_tensor(buf.values):
            explained_var = explained_variance(buf.values.flatten().cpu().numpy(), buf.returns.flatten().cpu().numpy())
        else:
            explained_var = explained_variance(buf.values, buf.returns)
        self.policy.num_adversaries = self.num_adversaries

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
    
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
        self.d = count
        count = 0
        with torch.no_grad():
            for p in param_list:
                p.copy_(p + torch.reshape(v[count:count + torch.numel(p)], p.shape).to(self.device))
                count = count + torch.numel(p)
        return
    
    def env_perturb_params(self):
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
        self.collect_rollouts(self.env, self.callback, buf, adv_buf, n_rollout_steps=self.n_steps)
        
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
        for i in range(self.n_epochs):
            self.parallel_updater.update_value_functions(
                                                        self.policy, perturbed_agent, perturbed_adv_buf, 
                                                        self.adversary_buffers, self.batch_size, self.max_grad_norm,
                                                        self.n_epochs, self.n_env_per_adv, self.first_run, self.envs_per_matchup
                                                        )
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
        new_env = self.env_generator_func()
        new_env.reset()
        return new_env

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from save.
        """
        excluded = super()._excluded_save_params()
        excluded.extend(["parallel_updater"])
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