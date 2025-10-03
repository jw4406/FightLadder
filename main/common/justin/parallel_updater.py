from typing import List, Any
import pickle
from queue import Empty
import time
import random
import gc

import numpy as np
import torch
from torch.multiprocessing import Process, Queue
import torch.nn.functional as F

from main.utils import move_policy, select_device, get_n_workers, state2matchup, select_matchup_env, unpickle_policy
from .update_value_functions import _update_single_value_function, shard_indices
from .calc_F import _get_buffers_and_keys, _calculate_policy_loss, _compute_grads, calc_F_grad_single

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
        #all_dstb_actions = []
        all_observations = []
        all_returns = []
        all_env_indices = []

        for rollout_data in all_rollout_data:
            all_actions.append(torch.Tensor(rollout_data.actions))
            #all_dstb_actions.append(torch.Tensor(rollout_data.dstb_actions))
            all_observations.append(rollout_data.observations)
            all_returns.append(torch.Tensor(rollout_data.returns))
            all_env_indices.extend(rollout_data.env_indices)
        
        actions_batch = torch.cat(all_actions).to(device)
        #dstb_actions_batch = torch.cat(all_dstb_actions).to(device)
        observations_batch = torch.cat(all_observations).to(device)
        returns_batch = torch.cat(all_returns).to(device)

        return actions_batch, observations_batch, returns_batch, np.array([env_ind.cpu() for env_ind in all_env_indices])
    
    #Process all rollout data and actions at once instead of batch by batch.
    actions_batch, observations_batch, returns_batch, all_env_indices = _prep_rollout_data_actions(batch_size, buffer)
    policy.num_global_env = num_envs
    policy.num_adv = 1
    for i in range(len(returns_batch) // batch_size):
        values = policy.evaluate_states(
        observations_batch[i * batch_size:(i + 1) * batch_size],
        buf_num=[adversary_index],
        env_indices=all_env_indices[i * batch_size:(i + 1) * batch_size]
        )
        values = values.flatten()
        # offset = 12 # vf extractor and shared trunk are 12
        # num_per_head = 10 # lstm = 6, 2 linear layers = 2 + 2, total 10
        value_loss = F.mse_loss(values, returns_batch[i * batch_size:(i + 1) * batch_size])
        # indices = list(range(0, offset)) + list(range(offset + adversary_index * num_per_head, offset + (adversary_index + 1) * num_per_head))
        # value_grads = th.autograd.grad(value_loss, [policy.value_optimizer.param_groups[0]['params'][j] for j in indices])
        #value_grads = th.cat([grad.view(-1) for grad in value_grads])
        policy.value_optimizer.zero_grad()
        # for i in range(len(value_grads)):
        #     policy.value_optimizer.param_groups[0]['params'][indices[i]].grad = value_grads[i]
        value_loss.backward()
        #policy.value_optimizer.zero_grad()
        #for i in range(len(policy.value_optimizer.param_groups[0]['params'])):
        #    policy.value_optimizer.param_groups[0]['params'][i].grad = value_grads[i]
        #value_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy.value_optimizer.step()


class ParallelUpdater:
    """
    Manages persistent worker processes for parallel value function updates on multiple GPUs.
    
    Creates worker processes once and reuses them for subsequent calls, avoiding the overhead
    of process creation. Uses proper synchronization to wait for job completion.


    To add a new job type:
    1. Add job handler static method: _handle_your_job_type(job, device_id, done_queue, persistent_state) -> Any
    2. Add handle functions to the HANDLE_FN dictionary in _generic_worker_function.
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

    def _select_device_id(self, ind: int) -> int:
        """This function transforms any index to device_id by performing ind%self.num_workers."""
        return ind%self.n_workers

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
    def _load_policy_from_persistent(persistent_state: dict, policy_data: Any, key: str, device: torch.device) -> torch.nn.Module:
        """
        This helper function loads a model from the persistent_state and moves it to the device.

        Args:
            persistent_state (dict):
                Persistent state dictionary.

            policay_data: (Any):
                Pickled model (bytes or weights).

            key (str):
                The key of the model in persistent_state.
            
            device (torch.device):
                The torch device to move to.
        
        Returns:
            The loaded policy.

        Raises:
            RuntimeError if the first run sends a state dict instead of a pickled model.
        """

        policy = persistent_state.get(key)

        # Initialize models once (first run)
        if policy is None:
            if isinstance(policy_data, bytes):
                policy = pickle.loads(policy_data)
                persistent_state[key] = policy
            else:
                # Handle case where first run sends state dict instead of pickled model
                raise RuntimeError("First run should send pickled models, not state dicts")
            move_policy(policy, device)
        else:
            # Update weights (subsequent runs)
            if isinstance(policy_data, bytes):
                policy = pickle.loads(policy_data)
                move_policy(policy, device)
            else:
                policy.load_state_dict(policy_data)
        persistent_state[key] = policy
        return policy

    @staticmethod
    def _generic_worker_function(input_queue: Queue, done_queue: Queue, device_id: int) -> None:
        """Generic worker that can handle different job types."""

        #NOTE: Update this dictionary to handle new jobs.
        HANDLE_FN = {
                    "UPDATE_VALUE_FUNCTIONS": ParallelUpdater._handle_update_value_functions,
                    }        
        
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
            
            job_handle_fn = HANDLE_FN.get(job_type)
            if job_handle_fn:
                job_handle_fn(job, device_id, done_queue, persistent_state)
            else:
                print(f"Worker {device_id}: Unknown job type: {job_type}")
                done_queue.put(f"ERROR_UNKNOWN_JOB_TYPE_{job_type}")
    
    @staticmethod
    def _handle_update_value_functions(job: tuple, device_id: int, done_queue: Queue, persistent_state: dict) -> None:
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

        device = select_device(device_id)
        # device = 'cpu' #I think this was introduced in debugging session, should probably be removed
        try:
            derivative_free_SPAR_policy = ParallelUpdater._load_policy_from_persistent(persistent_state=persistent_state, policy_data=derivative_free_SPAR_policy_data, key="derivative_free_SPAR_policy", device=device)
            perturbed_agent_policy = ParallelUpdater._load_policy_from_persistent(persistent_state=persistent_state, policy_data=perturbed_agent_policy_data, key="perturbed_agent_policy", device=device)

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
        job_results = {} #job_id -> result
        while completed_jobs < len(active_jobs):
            try:
                result = self.done_queue.get(timeout=60)  # 60 second timeout
                if isinstance(result, str):
                    if result.startswith("ERROR_"):
                        raise ValueError(f"Job failed: {result}")
                    elif result.startswith("job_"):
                        job_id = result
                    else:
                        raise ValueError(f"Unkonwn job result: {result}.")
                else:
                    job_id = result[0]
                    job_data = result[1:] if len(result)>1 else None
                    job_results[job_id] = job_data
                completed_jobs += 1
            except Empty:
                print("Warning: Timeout waiting for job completion")
                break
        return [job_results.get(job_id) for job_id in active_jobs]
    
    def _parallel_job_executor(func):
        """
        This is a decorator to use 
        """
        def wrapper(self, *args, **kwargs):
            active_jobs = []
            func(self, *args, active_jobs=active_jobs, **kwargs)
            self._wait_for_jobs(active_jobs)
        return wrapper

    @_parallel_job_executor
    def update_value_functions(self, policy: Any, perturbed_agent: Any, perturbed_adv_buf: List[Any], 
                            adversary_buffers: List[Any], batch_size: int, max_grad_norm: float, 
                            n_epochs: int, n_env_per_adv: int, first_run: bool = False, 
                            envs_per_matchup: int = None, *, active_jobs: list) -> None:
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
