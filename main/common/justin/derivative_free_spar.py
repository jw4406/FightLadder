import torch
from torch.multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
import gc
import torch as th, time, sys
import numpy as np
import pickle
from gym import spaces
import math
import random
from typing import List
from copy import deepcopy
from stable_baselines3.common.callbacks import ConvertCallback
from torch.nn import functional as F
from .Generalist_SPAR import Generalist_SPAR
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn, \
    update_learning_rate, is_vectorized_observation, polyak_update
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from typing import Any, Dict, Mapping, Optional, Tuple, Union, Type, List, TypeVar
from stable_baselines3.common.policies import BasePolicy, ActorActorCriticCnnPolicy, ActorActorCriticCnnGeneralistPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv
from utils import move_policy, select_device, get_n_workers, state2matchup, select_matchup_env

DEBUG = False
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
            all_returns.append(torch.Tensor(-rollout_data.returns))
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

class Derivative_Free_SPAR(Generalist_SPAR):
    def __init__(self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            envs_per_matchup: int=1,
            state_len: int=1,
            env_batch_size: int = 32,
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
            n_env_per_adv=1,
            warmstarted_cont_MAGICS=False,
            opp_list=None,
            player=None,
            use_mirror=False,
            env_generator_func=None, #The function is used to create a copy of the environment
            state_list=None,
    ):
        self.matchups = [state2matchup(state) for state in state_list] #This needs to happen before the super().__init__
        self.envs_per_matchup = envs_per_matchup
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
            use_mirror=use_mirror,
            matchups=self.matchups,
            envs_per_matchup=envs_per_matchup
        )
        self.parallel_updater = None
        self.first_run = False
        self.env_generator_func = env_generator_func
        self.state_len = state_len
        self.env_batch_size = env_batch_size
        if self.policy is not None: 
            self.policy.num_env_per_adv = self.envs_per_matchup

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
    
    def collect_rollouts(
                        self,
                        env: VecEnv,
                        callback: BaseCallback,
                        rollout_buffer: RolloutBuffer,
                        adversary_buffers,
                        n_rollout_steps: int,
                        ) -> bool:
        """Override to use batched environments for memory management"""
        def _calc_i_start_j_start(env_cnt: int, envs_per_matchup: int) -> tuple:
            """
            This helper functino calcaultes the start of i and j to be used in env_generator_func.

            Args:
                env_cnt (int):
                    How many environments were created.
                envs_per_matchup (int):
                    Environments to create per matchup.
            
            Returns:
                i_start (int):
                    i to start env_generator_func.
                j_start (int):
                    j to start env_generator_func.
            """
            #i_start = env_cnt // envs_per_matchup
            #j_start = env_cnt  % envs_per_matchup
            #i_start = 0
            j_start = 0
            i_start = env_cnt
            return i_start, j_start

        # Create rollout environments in batches using the stored generator function
        total_envs_needed = self.state_len# * self.envs_per_matchp
        # we modified state_list to hold the master list of envs with all the diversity duplicates and everything. 
        rollout_buffer.reset()
        for buf in adversary_buffers:
            buf.reset()
        #total_batches = (total_envs_needed + self.env_batch_size - 1) // self.env_batch_size
        # i dont quite understand why this computation is correct
        total_batches = total_envs_needed // self.env_batch_size
        if total_envs_needed % self.env_batch_size != 0:
            raise ValueError("total_envs_needed must be divisible by env_batch_size")
        #if self.envs_per_matchup % self.env_batch_size != 0:
        #    raise ValueError("env_batch_size must be divisible by envs_per_matchup")
            # do not allow grabbing splits (i.e., A A B B A | A B B A B ) 
            # only allow (A A | B B ...) or (A A B B | A A B B ...)
        # need to do a rem check here

        env_cnt = 0 #how many environments were created
        if total_batches != 1:
            flat_elements = []
            adv_flat_elements = list(adversary_buffers[0].obs_shape)
            adv_flat_elements.insert(0, adversary_buffers[0].buffer_size)
            adv_flat_elements.insert(1, self.envs_per_matchup)
            for item in (rollout_buffer.buffer_size,total_envs_needed, rollout_buffer.obs_shape):
                if isinstance(item, tuple):
                    flat_elements.extend(item)  # Recursively flatten nested tuples
                else:
                    flat_elements.append(item)  # Add integers directly
            #shape = np.flatten((rollout_buffer.buffer_size,total_envs_needed, rollout_buffer.obs_shape))
            ego_vertical_batch_obs = th.empty(flat_elements, pin_memory=True)
            adv_vertical_batch_obs = [th.empty(adv_flat_elements, pin_memory=True) for _ in range(self.num_adversaries)]
            ego_vertical_batch_rewards = th.empty(np.shape(rollout_buffer.rewards), pin_memory=True)
            adv_vertical_batch_rewards = [th.empty(np.shape(adversary_buffers[i].rewards), pin_memory=True) for i in range(self.num_adversaries)]
            #vertical_batch_rewards_other = np.empty(np.shape(rollout_buffer.rewards_other))
            ego_vertical_batch_dones = th.empty(np.shape(rollout_buffer.dones), pin_memory=True)
            adv_vertical_batch_dones = [th.empty(np.shape(adversary_buffers[i].dones), pin_memory=True) for i in range(self.num_adversaries)]
            #vertical_batch_infos = np.empty(np.shape(rollout_buffer.infos))
            ego_vertical_batch_log_probs = th.empty(np.shape(rollout_buffer.log_probs))
            adv_vertical_batch_log_probs = [th.empty(np.shape(adversary_buffers[i].log_probs)) for i in range(self.num_adversaries)]
            ego_vertical_batch_values = th.empty(np.shape(rollout_buffer.values)).to(self.device)
            adv_vertical_batch_values = [th.empty(np.shape(adversary_buffers[i].values), pin_memory=True) for i in range(self.num_adversaries)]
            ego_vertical_batch_dstb_log_probs = th.empty(np.shape(rollout_buffer.dstb_log_probs), pin_memory=True)
            adv_vertical_batch_dstb_log_probs = [th.empty(np.shape(adversary_buffers[i].dstb_log_probs), pin_memory=True) for i in range(self.num_adversaries)]
            ego_vertical_batch_last_ep_starts = th.empty(np.shape(rollout_buffer.episode_starts)).to(self.device)
            adv_vertical_batch_last_ep_starts = [th.empty(np.shape(adversary_buffers[i].episode_starts), pin_memory=True) for i in range(self.num_adversaries)]
            last_ep_starts = th.empty(np.shape(rollout_buffer.episode_starts), pin_memory=True)
            adv_last_ep_starts = [th.empty(np.shape(adversary_buffers[i].episode_starts), pin_memory=True) for i in range(self.num_adversaries)]
            ego_vertical_batch_actions = th.empty(np.shape(rollout_buffer.actions), pin_memory=True)
            adv_vertical_batch_actions = [th.empty(np.shape(adversary_buffers[i].actions), pin_memory=True) for i in range(self.num_adversaries)]
            ego_vertical_batch_adversary_actions = th.empty(np.shape(rollout_buffer.dstb_actions), pin_memory=True)
            adv_vertical_batch_adversary_actions = [th.empty(np.shape(adversary_buffers[i].dstb_actions), pin_memory=True) for i in range(self.num_adversaries)]

            final_obs_all_envs = th.empty(rollout_buffer.observations[0].shape, device=self.device)
            final_dones_all_envs = np.empty(rollout_buffer.dones[0].shape)
        for batch_idx in range(total_batches):
            # if total_batches != 1:
            #     flat_elements = []
            #     for item in (rollout_buffer.buffer_size,total_envs_needed, rollout_buffer.obs_shape):
            #         if isinstance(item, tuple):
            #             flat_elements.extend(item)  # Recursively flatten nested tuples
            #         else:
            #             flat_elements.append(item)  # Add integers directly
            #     #shape = np.flatten((rollout_buffer.buffer_size,total_envs_needed, rollout_buffer.obs_shape))
            #     vertical_batch_obs = np.empty(flat_elements)
            i_start, j_start = _calc_i_start_j_start(env_cnt, self.envs_per_matchup)
            rollout_env = self.env_generator_func(max_envs=self.env_batch_size, i_start=i_start, j_start=j_start)
            network_keys = [i // self.envs_per_matchup for i in range(i_start, i_start + self.env_batch_size)]
            # indexing state_list will use i_start : i_start + self.env_batch_size

            env_cnt += rollout_env.num_envs
            self._last_obs = rollout_env.reset()  # Set initial observations for this batch
            self._last_episode_starts = np.ones(rollout_env.num_envs)
            # Call the parent's collect_rollouts with our batched environment
            if total_batches == 1:
                result = super().collect_rollouts(
                    rollout_env,
                    callback,
                    rollout_buffer,
                    adversary_buffers,
                    n_rollout_steps
                )
                if not result:  # If parent method returned False, propagate it
                    return False
        
                return True
            else:
                env = rollout_env
                assert self._last_obs is not None, "No previous observation was provided"
                # Switch to eval mode (this affects batch norm / dropout)
                self.policy.set_training_mode(True)

                n_steps = 0
                #rollout_buffer.reset()
                for i in range(self.num_adversaries):
                    adversary_buffers[i].reset()
                # Sample new weights for the state dependent exploration
                if self.use_sde:
                    self.policy.reset_noise(env.num_envs)

                # need to sample leader policy here
                #for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params'])):
                #    self.policy.ctrl_optimizer.param_groups[0]['params'][i] = torch.nn.init.uniform_(self.policy.ctrl_optimizer.param_groups[0]['params'][i], a=-1., b=1.)

                callback.on_rollout_start()
                count = 0
                while n_steps < n_rollout_steps:
                    if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                        # Sample a new noise matrix
                        self.policy.reset_noise(env.num_envs)

                    with th.no_grad():
                        # Convert to pytorch tensor or to TensorDict

                        # PROBLEM HERE:
                        # we need to only call the right heads here cause adversary list may not be the 
                        # full thing since we're chunking/cycling the envs!


                        obs_tensor = obs_as_tensor(self._last_obs, self.device)
                        s_actions, s_log_probs, s_values, s_dstb_actions, s_dstb_log_probs = self.policy(obs_tensor, network_keys=network_keys)
                        all_adv_left_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
                        all_adv_right_actions = torch.zeros((self.n_global_env, self.action_space.n), device=self.device)
                        all_adv_critic_values = torch.zeros((self.n_global_env, 1), device=self.device)
                        all_adv_log_probs = torch.zeros((self.n_global_env,), device=self.device)
                        all_adv_dstb_log_probs = torch.zeros((self.n_global_env,), device=self.device)
                    actions = s_actions
                    adversary_actions = s_dstb_actions
                    log_probs = s_log_probs
                    adversary_log_probs = s_dstb_log_probs
                    actions = actions.cpu().numpy()
                    adversary_actions = adversary_actions.cpu().numpy()
                    all_adv_critic_values = s_values

                    if self.use_mirror is True:
                        mirror_master_copy_actions = deepcopy(actions)
                        mirror_master_copy_adv_actions = deepcopy(adversary_actions)

                    # upper half, lower half

                    
                    if self.use_mirror is True:
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
                        
                        if DEBUG:
                            #test = np.zeros_like(actions)
                            #other_test = np.ones_like(actions)
                            #test_left = test[halfway:, :]
                            #test_right = other_test[:halfway, :]
                            #temp = np.zeros((self.num_adversaries, self.action_space.shape[0]))
                            #temp[:halfway, :] = test_left
                            #temp[halfway:, :] = test_right

                            test2 = np.zeros_like(actions)
                            count = 0
                            for i in range(test2.shape[0]):
                                for j in range(test2.shape[1]):
                                    test2[i, j] = count
                                    count += 1
                            other_test2 = np.zeros_like(actions)
                            count = other_test2.size - 1
                            for i in range(other_test2.shape[0]):
                                for j in range(other_test2.shape[1]):
                                    other_test2[i, j] = count
                                    count -= 1
                            prot_left = test2[:halfway, :]  # actions for the prot when he is on the left
                            prot_left_pre = test2[halfway:, :]  

                            adv_right = other_test2[:halfway, :]
                            adv_right_pre = other_test2[halfway:, :]

                            prot_actions = np.empty_like(actions)
                            prot_actions[:halfway, :] = prot_left
                            prot_actions[halfway:, :] = adv_right_pre

                            adv_actions = np.empty_like(actions)
                            adv_actions[:halfway, :] = adv_right
                            adv_actions[halfway:, :] = prot_left_pre

                            #print("temp2", temp2)
                            #print("other_test2", other_test2)
                            #print("test2_left", test2_left)
                            #print("test2_right", test2_right)
                            #print("actions", actions)
                            #print("temp", temp)
                            #print("other_test", other_test)
                            #print("test_left", test_left)
                            #print("test_right", test_right)
                            #print("actions", actions)

                        prot_left = actions[:halfway, :]  # actions for the prot when he is on the left
                        prot_left_pre = actions[halfway:, :]  

                        adv_right = adversary_actions[:halfway, :]
                        adv_right_pre = adversary_actions[halfway:, :]

                        prot_actions = np.empty_like(actions)
                        #temp = prot_right
                        prot_actions[:halfway, :] = prot_left
                        prot_actions[halfway:, :] = adv_right_pre

                        adv_actions = np.empty_like(actions)
                        adv_actions[:halfway, :] = adv_right
                        adv_actions[halfway:, :] = prot_left_pre

                        actions = prot_actions
                        adversary_actions = adv_actions

                    # Rescale and perform action
                    if self.update_left is True:
                        # MESSY
                        clipped_actions = np.hstack([actions, adversary_actions])
                    else:
                        clipped_actions = np.hstack([adversary_actions, actions])
                    # Clip the actions to avoid out of bound error
                    if isinstance(self.action_space, spaces.Box):
                        clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                    new_obs, rewards, rew_other, dones, infos = env.step(clipped_actions)

                    # if mirroring:
                    # rew = (r,r,r, -r, -r, -r)^T 
                    if self.use_mirror is True:
                        halfway = len(rewards) // 2
                        rewards[halfway:] = -rewards[halfway:]
                        # now rew = (r, r, r, r, r, r)^T
                        # this is the correct ego reward
                    
                    # if mirror is false
                    # rew is already (r,r,r,r,r,r)^T and we dont need to do anything
                    
                    if np.any(rewards != 0):
                        print("Reward is not 0")
                    ego_vertical_batch_obs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :, :, :] = th.unsqueeze(th.from_numpy(new_obs), 0)
                    ego_vertical_batch_rewards[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(rewards), 0)
                    #vertical_batch_rewards_other[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :] = th.unsqueeze(th.from_numpy(rew_other), 0)
                    ego_vertical_batch_dones[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(dones), 0)
                    #vertical_batch_infos[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches)), :] = th.unsqueeze(th.from_numpy(infos), 0)
                    ego_vertical_batch_log_probs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(log_probs, 0).cpu()
                    ego_vertical_batch_values[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(s_values, 0)
                    ego_vertical_batch_dstb_log_probs[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(s_dstb_log_probs, 0).cpu()
                    ego_vertical_batch_last_ep_starts[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(self._last_episode_starts), 0)
                    ego_vertical_batch_actions[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(actions), 0)
                    ego_vertical_batch_adversary_actions[count, int(batch_idx * (total_envs_needed / total_batches)) : int((batch_idx + 1) * (total_envs_needed / total_batches))] = th.unsqueeze(th.from_numpy(adversary_actions), 0)
                    
                    # For each environment in the batch, assign its data to the correct adversary buffer and slot.
                    for j in range(env.num_envs):
                        # Calculate the global index of the environment across all batches.
                        global_env_idx = i_start + j
                        print(global_env_idx)
                        
                        # Determine the matchup this environment belongs to. This is the index for the adversary buffer.
                        matchup_idx = global_env_idx // self.envs_per_matchup
                        
                        # Determine the local index of the environment within its matchup group. This is the slot in the buffer.
                        local_env_idx = global_env_idx % self.envs_per_matchup

                        # Place the observation, reward, and done status into the correct buffer and slot.
                        # `count` is the current step in the rollout.
                        adv_vertical_batch_obs[matchup_idx][count, local_env_idx] = obs_as_tensor(new_obs[j], device='cpu')
                        # we need to flip adversary rewards because adversary always gets -r
                        # recall right now that rew = (r, r, r, r, r, r)^T in BOTH cases! (mirror or not)

                        # so we flip every element 
                        adv_vertical_batch_rewards[matchup_idx][count, local_env_idx] = -rewards[j]
                        #adv_vertical_batch_dones[matchup_idx][count, local_env_idx] = dones[j]
                        adv_vertical_batch_log_probs[matchup_idx][count, local_env_idx] = log_probs[j]
                        adv_vertical_batch_values[matchup_idx][count, local_env_idx] = -all_adv_critic_values[j]
                        adv_vertical_batch_dstb_log_probs[matchup_idx][count, local_env_idx] = s_dstb_log_probs[j]
                        adv_vertical_batch_last_ep_starts[matchup_idx][count, local_env_idx] = th.from_numpy(self._last_episode_starts)[j]
                        #last_ep_starts[global_env_idx] = th.from_numpy(np.round(dones[j]).astype(bool))

                        adv_vertical_batch_actions[matchup_idx][count, local_env_idx].copy_(th.from_numpy(actions[j]))
                        adv_vertical_batch_adversary_actions[matchup_idx][count, local_env_idx].copy_(th.from_numpy(adversary_actions[j]))


                    self.num_timesteps += env.num_envs
                    #wandb.log({"epochs": self.num_timesteps})
                    # Give access to local variables
                    callback.update_locals(locals())
                    if callback.on_step() is False:
                        return False

                    self._update_info_buffer(infos)
                    n_steps += 1

                    if isinstance(self.action_space, spaces.Discrete):
                        # Reshape in case of discrete action
                        actions = actions.reshape(-1, 1)
                    count += 1

                    self._last_obs = new_obs
                    self._last_episode_starts = dones
                i_start = batch_idx * self.env_batch_size
                final_obs_all_envs[i_start : i_start + self.env_batch_size] = obs_as_tensor(new_obs, self.device)
                final_dones_all_envs[i_start : i_start + self.env_batch_size] = dones

            rollout_env.close()  # Clean up this batch
            result = True
            
            if not result:  # If parent method returned False, propagate it
                return False
        
        rollout_buffer.observations = ego_vertical_batch_obs
        rollout_buffer.rewards.copy_(ego_vertical_batch_rewards)
        #rollout_buffer.dones = ego_vertical_batch_dones
        rollout_buffer.log_probs = ego_vertical_batch_log_probs
        rollout_buffer.values = ego_vertical_batch_values
        rollout_buffer.dstb_log_probs = ego_vertical_batch_dstb_log_probs
        rollout_buffer.episode_starts = ego_vertical_batch_last_ep_starts
        rollout_buffer.actions = ego_vertical_batch_actions
        rollout_buffer.adversary_actions = ego_vertical_batch_adversary_actions
        

        for i in range(len(adversary_buffers)):
            adversary_buffers[i].observations = adv_vertical_batch_obs[i]
            adversary_buffers[i].rewards = adv_vertical_batch_rewards[i]
            #adversary_buffers[i].dones = adv_vertical_batch_dones[i]
            adversary_buffers[i].log_probs = adv_vertical_batch_log_probs[i]
            adversary_buffers[i].values = adv_vertical_batch_values[i]
            adversary_buffers[i].dstb_log_probs = adv_vertical_batch_dstb_log_probs[i]
            adversary_buffers[i].episode_starts = adv_vertical_batch_last_ep_starts[i]
            adversary_buffers[i].actions = adv_vertical_batch_actions[i]
            adversary_buffers[i].adversary_actions = adv_vertical_batch_adversary_actions[i]

        rollout_buffer.full = True
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].full = True

        
        with th.no_grad():
            # Compute value for the last time:w
            # step
            #values = torch.zeros((self.n_global_env,))
            values = self.policy.predict_values(final_obs_all_envs)

        rollout_buffer.values = rollout_buffer.values.to(self.device, non_blocking=True)
        rollout_buffer.rewards = rollout_buffer.rewards.to(self.device, non_blocking=True)
        rollout_buffer.advantages = rollout_buffer.advantages.to(self.device, non_blocking=True)
        rollout_buffer.episode_starts = rollout_buffer.episode_starts.to(self.device, non_blocking=True)
        #rollout_buffer.vectorized_compute_returns_and_advantages(last_values=values, dones=torch.Tensor(dones).to(self.device))
        rollout_buffer.vectorized_compute_returns_and_advantages(last_values=values, dones=final_dones_all_envs)
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].values = adversary_buffers[i].values.to(self.device, non_blocking=True)
            adversary_buffers[i].rewards = adversary_buffers[i].rewards.to(self.device, non_blocking=True)
            adversary_buffers[i].advantages = adversary_buffers[i].advantages.to(self.device, non_blocking=True)
            adversary_buffers[i].episode_starts = adversary_buffers[i].episode_starts.to(self.device, non_blocking=True)
            #adversary_buffers[i].vectorized_compute_returns_and_advantages(last_values=values, dones=final_dones_all_envs)
            

            start_idx = i * self.envs_per_matchup
            end_idx = (i + 1) * self.envs_per_matchup
            adv_last_values = values[start_idx:end_idx]
            adv_dones = final_dones_all_envs[start_idx:end_idx]
            adversary_buffers[i].vectorized_compute_returns_and_advantages(last_values=-adv_last_values, dones=adv_dones)
        
        callback.on_rollout_end()

        rollout_buffer.prepare_data_for_training()
        for buf in adversary_buffers:
            buf.prepare_data_for_training()
        
        return True    

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
        test.policy.dstb_optimizer = self.policy.optimizer_class(test.policy.dstb_optimizer.param_groups[0]['params'], maximize=True)
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

    def train(self, update_ego: bool = True, update_adversary: bool = True):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # 3. Update value functions for both original and perturbed agents
        # This is called for all agents as the values are needed for advantage calculation.
        start_time = time.time()
        self._update_value_functions(self.perturbed_agent, self.perturbed_adv_buf)
        end_time = time.time()
        if TIMING:
            print(f"Time for _update_value_functions: {end_time - start_time:.4f}s")

        #need to swap and flatten here 
        self.update_advantages(self.policy, self.rollout_buffer, self.adversary_buffers)
        self.update_advantages(self.perturbed_agent.policy, self.perturbed_buf, self.perturbed_adv_buf)
        self.perturbed_agent_policy = self.perturbed_agent.policy
        #TODO: Serial mode - debug only
        self.leader_grads(self.rollout_buffer, self.perturbed_buf, self.policy, self.perturbed_agent_policy, ego=True)
        self.leader_grads(self.adversary_buffers, self.perturbed_adv_buf, self.policy, self.perturbed_agent_policy, ego=False)
        
        # Try to execute in parallel (on the same device)
        if update_ego:
            ego_policy_bytes: bytes = pickle.dumps(self.policy)
            ego_perturbed_agent_policy: bytes = pickle.dumps(self.perturbed_agent_policy)
        if update_adversary:
            adv_policy_bytes: bytes = pickle.dumps(self.policy)
            adv_perturbed_agent_policy: bytes = pickle.dumps(self.perturbed_agent_policy)
        futures = [] #A temporary list to store dummy results.
        #self.leader_grads(self.rollout_buffer, self.perturbed_buf, self.policy, self.perturbed_agent_policy, ego=True)
        #self.leader_grads(self.adversary_buffers, self.perturbed_adv_buf, self.policy, self.perturbed_agent_policy, ego=False)
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Selectively update the ego (actor) policy
            if update_ego:
                futures.append(executor.submit(self.leader_grads, self.rollout_buffer, self.perturbed_buf, ego_policy_bytes, ego_perturbed_agent_policy, ego=True))
            
            # Selectively update the adversary (disturber) policy
            if update_adversary:
                futures.append(executor.submit(self.leader_grads, self.adversary_buffers, self.perturbed_adv_buf, adv_policy_bytes, adv_perturbed_agent_policy, ego=False))
            
            #Wait for both jobs to finish
            for future in futures:
                future.result()

            # Copy optimizer states back to main policy
            if update_ego:
                self.policy.ctrl_optimizer.load_state_dict(pickle.loads(ego_policy_bytes).ctrl_optimizer.state_dict())
            if update_adversary:
                self.policy.dstb_optimizer.load_state_dict(pickle.loads(adv_policy_bytes).dstb_optimizer.state_dict())
        
        del self.perturbed_agent_policy
        del self.perturbed_buf
        del self.perturbed_adv_buf
        gc.collect()
        torch.cuda.empty_cache()
    
    def perturb_params(self, param_list, ego=True):
        count = 0
        for i in range(len(param_list)):
            count = count + torch.numel(param_list[i])
        delta = .7
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
            n_envs=self.n_envs,
            dstb_action_space=self.dstb_action_space)
        #buf = deepcopy(self.rollout_buffer)
        #buf.reset()
        #adv_buf = deepcopy(self.adversary_buffers)
        adv_buf = [self.rollout_buffer_class(self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs= self.n_env_per_adv,
            dstb_action_space=self.dstb_action_space) for i in range(self.num_adversaries)]
        #[adv_buf[i].reset() for i in range(len(adv_buf))]
        self.collect_rollouts(self.env, self.callback, buf, adv_buf, n_rollout_steps=self.n_steps)
        
        buf.prepare_data_for_training()
        for i in range(len(adv_buf)):
            adv_buf[i].prepare_data_for_training()
            
        return buf, adv_buf

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

    def leader_grads(self, ori_buf, perturbed_buf, ori_policy, perturbed_policy, ego=True):
        def _unpickle_policy(policy: Any) -> torch.nn.Module:
            """This is a helper function that unpickles a policy."""
            if isinstance(policy, bytes):
                policy = pickle.loads(policy)
            return policy
        #perturbed_policy might be pickled - if it is, unpickle it.
        ori_policy = _unpickle_policy(ori_policy)
        perturbed_policy = _unpickle_policy(perturbed_policy)
        
        total_start_time = time.time()
        if ego is True:
            print("Ego is true", flush=True)
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_losses, pg_losses, approx_kl_divs_all = [], [], []

        num_runs_count = 1 if ego else self.num_adversaries
        for j in range(self.n_epochs):
            epoch_start_time = time.time()
            for i in range(num_runs_count):
                run_start_time = time.time()
                network_keys, curr_buf, curr_perturbed_buf = self._get_buffers_and_keys(ori_buf, perturbed_buf, ego, i)
                
                approx_kl_divs_epoch = []
                
                batch_loop_start_time = time.time()
                for ori_rollout_data, perturbed_rollout_data in zip(curr_buf.get(self.batch_size), curr_perturbed_buf.get(self.batch_size)):
                    
                    calc_loss_start_time = time.time()
                    policy_loss, log_prob, entropy = self._calculate_policy_loss(
                        ori_rollout_data, ori_policy, ego, network_keys, clip_range
                    )
                    pg_losses.append(policy_loss.item())
                    entropy_losses.append(entropy.mean().item())

                    perturbed_policy_loss, _, _ = self._calculate_policy_loss(
                        perturbed_rollout_data, perturbed_policy, ego, network_keys, clip_range
                    )
                    calc_loss_end_time = time.time()
                    if TIMING:
                        print(f"            [Timing] _calculate_policy_loss (ori+pert): {calc_loss_end_time - calc_loss_start_time:.4f}s")
                    
                    compute_grads_start_time = time.time()
                    self._compute_and_apply_grads(policy_loss, perturbed_policy_loss, ego, i if ego is False else None)
                    compute_grads_end_time = time.time()
                    if TIMING:
                        print(f"            [Timing] _compute_and_apply_grads: {compute_grads_end_time - compute_grads_start_time:.4f}s")

                    kl_div_start_time = time.time()
                    with th.no_grad():
                        old_log_prob_tensor = ori_rollout_data.old_log_prob if ego else ori_rollout_data.old_dstb_log_prob
                        #run forward pass to get the log_prob
                        if ego:
                            _, log_prob, entropy, _, _ = ori_policy.evaluate_actions(
                            torch.Tensor(ori_rollout_data.observations).to(self.device), torch.Tensor(ori_rollout_data.actions).to(self.device), torch.Tensor(ori_rollout_data.dstb_actions).to(self.device),
                            shuffle_keys=ori_rollout_data.env_indices, network_keys=network_keys, envs_per_matchup=self.envs_per_matchup
                        )
                        else:
                           _, _, _, log_prob, entropy = ori_policy.evaluate_actions(
                            torch.Tensor(ori_rollout_data.observations).to(self.device), torch.Tensor(ori_rollout_data.actions).to(self.device), torch.Tensor(ori_rollout_data.dstb_actions).to(self.device),
                            shuffle_keys=ori_rollout_data.env_indices, network_keys=network_keys, envs_per_matchup=self.envs_per_matchup
                        ) 
                        #run forward pass to get the log_prob
                        #_, log_prob, entropy, _, _ = perturbed_policy.evaluate_actions(
                        log_ratio = log_prob - old_log_prob_tensor
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs_epoch.append(approx_kl_div)
                        #gc.collect()
                        #torch.cuda.empty_cache()
                    kl_div_end_time = time.time()
                    if TIMING:
                        print(f"            [Timing] KL-div calculation: {kl_div_end_time - kl_div_start_time:.4f}s")
                
                batch_loop_end_time = time.time()
                if TIMING:
                    print(f"          [Timing] Batch processing loop ({'ego' if ego else 'adv'} run {i}): {batch_loop_end_time - batch_loop_start_time:.4f}s")
                approx_kl_divs_all.extend(approx_kl_divs_epoch)
                run_end_time = time.time()
                if TIMING:
                    print(f"        [Timing] Adversary/Ego run {i}: {run_end_time - run_start_time:.4f}s")

            epoch_end_time = time.time()
            if TIMING:
                print(f"      [Timing] Epoch {j}: {epoch_end_time - epoch_start_time:.4f}s")


        self._n_updates += self.n_epochs
        if hasattr(self.rollout_buffer, 'values') and self.rollout_buffer.values is not None and self.rollout_buffer.returns is not None:
             explained_var = explained_variance(self.rollout_buffer.values.flatten().detach().cpu().numpy(), self.rollout_buffer.returns.flatten().detach().cpu().numpy())
        else:
            explained_var = np.nan
        if ego is True:
            print("logging ego metrics", flush=True)
        self._log_leader_metrics(ego, entropy_losses, pg_losses, approx_kl_divs_all, explained_var, clip_range)

        total_end_time = time.time()
        if TIMING:
            print(f"    [Timing] Total leader_grads ({'ego' if ego else 'adv'}): {total_end_time - total_start_time:.4f}s")

    def _get_buffers_and_keys(self, ori_buf, perturbed_buf, ego, index):
        if ego:
            network_keys = [k for k in range(self.num_adversaries)]
            curr_buf = ori_buf
            curr_perturbed_buf = perturbed_buf
        else:
            network_keys = [index]
            curr_buf = ori_buf[index]
            curr_perturbed_buf = perturbed_buf[index]
        return network_keys, curr_buf, curr_perturbed_buf

    def _calculate_policy_loss(self, rollout_data, policy, ego, network_keys, clip_range):
        actions = torch.Tensor(rollout_data.actions).to(self.device)
        dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(self.device)

        if self.use_sde:
            policy.reset_noise(self.batch_size)

        with torch.no_grad():
            if ego:
                old_log_prob = rollout_data.old_log_prob
                _, log_prob, entropy, _, _ = policy.evaluate_actions(
                    torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions,
                    shuffle_keys=rollout_data.env_indices, network_keys=network_keys, envs_per_matchup=self.envs_per_matchup
                )
            else:
                old_log_prob = rollout_data.old_dstb_log_prob
                _, _, _, log_prob, entropy = policy.evaluate_actions(
                    torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions,
                    shuffle_keys=rollout_data.env_indices, network_keys=network_keys, envs_per_matchup=self.envs_per_matchup
                )
        
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - torch.Tensor(old_log_prob).to(self.device))
        
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = torch.min(policy_loss_1, policy_loss_2).mean()
        
        return policy_loss, log_prob, entropy

    def _compute_and_apply_grads(self, policy_loss, perturbed_policy_loss, ego, adv_num=None):
        if ego is False:
            assert adv_num is not None
        if ego:
            F = self.d / self.delta * (perturbed_policy_loss - policy_loss) * self.ego_v
        else:
            F = self.d / self.delta * (perturbed_policy_loss - policy_loss) * self.adv_v
        
        param_list = self.policy.ctrl_optimizer.param_groups[0]['params'] if ego else self.policy.dstb_optimizer.param_groups[0]['params']
        size_lists = [list(x.shape) for x in param_list]
        
        reshaped_grad = []
        count = 0
        for i in range(len(size_lists)):
            numel = np.prod(size_lists[i])
            reshaped_grad.append(torch.reshape(F[count: count + numel], size_lists[i]))
            count += numel
        if ego is False:
            #all_heads_length = self.num_adversaries * self.policy.head_length
            heads_start_index = self.policy.extractor_and_trunk_length
            trunk_extractor_indices = [i for i in range(heads_start_index)]
            this_adv_indices = [i for i in range(heads_start_index + self.policy.head_length * adv_num , heads_start_index + self.policy.head_length * (adv_num + 1))]
            all_indices = trunk_extractor_indices + this_adv_indices
            self.policy.dstb_optimizer.zero_grad()

            for i in all_indices:
                self.policy.dstb_optimizer.param_groups[0]['params'][i].grad = reshaped_grad[i].float().detach()
        else:
            self.policy.ctrl_optimizer.zero_grad()
            for i in range(len(size_lists)):
                param_list[i].grad = reshaped_grad[i].float().detach()

        optimizer = self.policy.ctrl_optimizer if ego else self.policy.dstb_optimizer
        optimizer.step()

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
        try:
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
                perturbed_agent, other_ego, other_adv = self._create_perturbed_agent()
                print("perturbed agent created!", flush=True)
                self._initialize_parallel_updater()                
                # perturbed_buf, perturbed_adv_buf = perturbed_agent.env_perturb_params() #TODO: This is a sequential original line, delete it when done.
                #continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps) #TODO: This is sequential - remove when done.

                # Run env_perturb_params and collect_rollouts in different threads (cannot be done in different processes because they contain unpickleable objects)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_perturbed = executor.submit(perturbed_agent.env_perturb_params)
                    future_collect = executor.submit(self.collect_rollouts, self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps)
                    
                    perturbed_buf, perturbed_adv_buf = future_perturbed.result()
                    continue_training = future_collect.result()
                self.perturbed_agent = perturbed_agent
                self.perturbed_buf = perturbed_buf
                self.perturbed_adv_buf = perturbed_adv_buf
                self.perturbed_agent_policy = perturbed_agent.policy
                print("main agent and perturbed agent rollout done!", flush=True)
                
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
            

                self.train(update_ego=update_ego, update_adversary=update_adversary)
                self.perturbed_agent.env.close()
                del self.perturbed_agent

            callback.on_training_end()
        
        finally:
            #IMPORTANT! Persistent workers must be cleaned up.
            self.cleanup()
            torch.cuda.empty_cache()

        return self
    
    def dump_properties(self):
        
        PRIMITIVE_TYPES = (int, float, bool, str, type(None))
        
        primitive_attrs = {}

        for attr_name in dir(self):
            if attr_name.startswith('__'):
                continue
                
            try:
                attr_value = getattr(self, attr_name)
            except AttributeError:
                continue

            if callable(attr_value):
                continue

            if isinstance(attr_value, PRIMITIVE_TYPES):
                primitive_attrs[attr_name] = attr_value
                continue
            
            if isinstance(attr_value, (list, tuple)):
                if all(isinstance(item, PRIMITIVE_TYPES) for item in attr_value):
                    primitive_attrs[attr_name] = attr_value



        return primitive_attrs

    def update_advantages(self, policy, ego_buf, adv_bufs):
        # always pass ego and adv, do not pass regular and perturbed. 
        shuffle_keys = np.tile([i for i in range(self.num_adversaries * self.envs_per_matchup)], ego_buf.buffer_size) # becuase these track raw envs
        network_keys = [i for i in range(self.num_adversaries)]
        values, _, _, _, _ = policy.evaluate_actions(torch.Tensor(ego_buf.observations).to(self.device), torch.Tensor(ego_buf.actions).to(self.device), torch.Tensor(ego_buf.dstb_actions).to(self.device),
                                     shuffle_keys=shuffle_keys, network_keys=network_keys, envs_per_matchup=self.envs_per_matchup)
        #self.rollout_buffer.values = values.reshape(self.rollout_buffer.buffer_size, self.num_adversaries)
        #grabs_per_rep = len(shuffle_keys) // len(network_keys)
        grabs_per_rep = self.envs_per_matchup
        #skip = len(shuffle_keys) - grabs_per_rep
        for i in range(len(adv_bufs)):
            pointer = i * grabs_per_rep
            for j in range(ego_buf.buffer_size):
                len_chunk_to_scan = len(values) // self.rollout_buffer.buffer_size
                #self.adversary_buffers[i].values[] = values[j * pointer : j * (pointer + grabs_per_rep)]
                curr_chunk = values[len_chunk_to_scan * j : len_chunk_to_scan * (j + 1)]
                adv_bufs[i].values[j * grabs_per_rep : (j+1) * grabs_per_rep] = -curr_chunk[pointer : pointer + grabs_per_rep]
            
            adv_bufs[i].values =adv_bufs[i].values.reshape(self.adversary_buffers[i].buffer_size, grabs_per_rep)
            adv_bufs[i].dones = adv_bufs[i].dones.reshape(self.adversary_buffers[i].buffer_size, grabs_per_rep)
            adv_bufs[i].vectorized_compute_returns_and_advantages(adv_bufs[i].values[-1, :], adv_bufs[i].dones[-1, :])
            adv_bufs[i].advantages = adv_bufs[i].swap_and_flatten_pt(adv_bufs[i].advantages)
            adv_bufs[i].values = adv_bufs[i].swap_and_flatten_pt(adv_bufs[i].values)
            adv_bufs[i].returns = adv_bufs[i].swap_and_flatten_pt(adv_bufs[i].returns)
        
        ego_buf.values = values.reshape(self.rollout_buffer.buffer_size, self.num_adversaries * self.envs_per_matchup)
        ego_buf.dones = ego_buf.dones.reshape(self.rollout_buffer.buffer_size, self.num_adversaries * self.envs_per_matchup)

        ego_buf.vectorized_compute_returns_and_advantages(ego_buf.values[-1, :], ego_buf.dones[-1, :])
        ego_buf.advantages = ego_buf.swap_and_flatten_pt(ego_buf.advantages)
        ego_buf.values = ego_buf.swap_and_flatten_pt(ego_buf.values)
        ego_buf.returns = ego_buf.swap_and_flatten_pt(ego_buf.returns)

    def set_steps(self, steps: int) -> None:
        self.num_timesteps = steps
