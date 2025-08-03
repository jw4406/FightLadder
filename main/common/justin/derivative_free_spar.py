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
from utils import move_policy, select_device, get_n_workers

DEBUG = True
TIMING = True

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

def _update_single_value_function(batch_size: int, max_grad_norm: float, policy, buffer, adversary_index: int, num_envs: int, device: torch.device, tag: str=""):
    """
    This function has to be placed outside of the object to enable parallel calls.
    TODO: Complete the docstring.
    TODO: Complete static types
    """
    total_start_time = time.time()

    #get_start_time = time.time()
    rollout_data_list = list(buffer.get(batch_size))
    #get_end_time = time.time()
    #if TIMING:
    #    print(f"        [Timing] ({tag}) buffer.get(): {get_end_time - get_start_time:.4f}s")

    for rollout_data in buffer.get(batch_size):
        loop_start_time = time.time()

        prep_start_time = time.time()
        actions = torch.Tensor(rollout_data.actions).to(device)
        dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(device)
        observations_tensor = rollout_data.observations.to(device) 
        returns_tensor = torch.Tensor(-rollout_data.returns).to(device)
        prep_end_time = time.time()
        if TIMING:
            print(f"          [Timing] ({tag}) Data prep & to(device): {prep_end_time - prep_start_time:.4f}s")

        policy.num_global_env = num_envs
        policy.num_adv = 1
        
        eval_start_time = time.time()
        values, _, _, _, _ = policy.evaluate_actions(
            observations_tensor,
            actions,
            dstb_actions,
            shuffle_keys=rollout_data.env_indices,
            network_keys=[adversary_index]
        )
        values = values.flatten()
        eval_end_time = time.time()
        if TIMING:
            print(f"          [Timing] ({tag}) evaluate_actions: {eval_end_time - eval_start_time:.4f}s")

        loss_start_time = time.time()
        value_loss = F.mse_loss(returns_tensor, values)
        loss_end_time = time.time()
        if TIMING:
            print(f"          [Timing] ({tag}) loss_calculation: {loss_end_time - loss_start_time:.4f}s")

        zero_grad_start_time = time.time()
        policy.value_optimizer.zero_grad()
        if hasattr(policy, 'ctrl_optimizer') and policy.ctrl_optimizer:
            policy.ctrl_optimizer.zero_grad()
        if hasattr(policy, 'dstb_optimizer') and policy.dstb_optimizer:
            policy.dstb_optimizer.zero_grad()
        zero_grad_end_time = time.time()
        if TIMING:
            print(f"          [Timing] ({tag}) zero_grad: {zero_grad_end_time - zero_grad_start_time:.4f}s")

        backward_start_time = time.time()
        value_loss.backward()
        backward_end_time = time.time()
        if TIMING:
            print(f"          [Timing] ({tag}) backward: {backward_end_time - backward_start_time:.4f}s")

        clip_grad_start_time = time.time()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        clip_grad_end_time = time.time()
        if TIMING:
            print(f"          [Timing] ({tag}) clip_grad_norm: {clip_grad_end_time - clip_grad_start_time:.4f}s")

        step_start_time = time.time()
        policy.value_optimizer.step()
        step_end_time = time.time()
        if TIMING:
            print(f"          [Timing] ({tag}) optimizer.step: {step_end_time - step_start_time:.4f}s")
        
        loop_end_time = time.time()
        if TIMING:
            print(f"        [Timing] ({tag}) Total loop iteration: {loop_end_time - loop_start_time:.4f}s")

    total_end_time = time.time()
    if TIMING:
        print(f"      [Timing] Total _update_single_value_function ({tag}): {total_end_time - total_start_time:.4f}s")


class ParallelUpdater:
    """
    Manages persistent worker processes for parallel value function updates on multiple GPUs.
    
    Creates worker processes once and reuses them for subsequent calls, avoiding the overhead
    of process creation. Uses proper synchronization to wait for job completion.
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
        device = select_device(device_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.set_device(device_id)

        derivative_free_SPAR_policy = None
        perturbed_agent_policy = None

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
            
            if job_type == "UPDATE_VALUE_FUNCTIONS":
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
                    job_id,
                ) = job[1]

                try:
                    # Initialize models once (first run)
                    if derivative_free_SPAR_policy is None:
                        if isinstance(derivative_free_SPAR_policy_data, bytes):
                            derivative_free_SPAR_policy = pickle.loads(derivative_free_SPAR_policy_data)
                            perturbed_agent_policy = pickle.loads(perturbed_agent_policy_data)
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

                    # Do the actual work
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for i in i_list:
                        for epoch in range(n_epochs):
                            _update_single_value_function(
                                batch_size, max_grad_norm, derivative_free_SPAR_policy, 
                                adversary_buffers[i], i, n_env_per_adv, device
                            )
                            _update_single_value_function(
                                batch_size, max_grad_norm, perturbed_agent_policy, 
                                perturbed_adv_buf[i], i, n_env_per_pert, device
                            )
                    
                    # Signal completion
                    done_queue.put(job_id)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Worker {device_id} error: {e}")
                    done_queue.put(f"ERROR_{job_id}")

            else:
                print(f"Worker {device_id}: Unknown job type: {job_type}")
                done_queue.put(f"ERROR_UNKNOWN_JOB_TYPE_{job_type}")

    def _create_update_values_function_job(self, policy: Any, perturbed_policy: Any, i_list: List[int], 
                   adversary_buffers: List[Any], perturbed_adv_buf: List[Any], 
                   batch_size: int, max_grad_norm: float, n_epochs: int, 
                   n_env_per_adv: int, n_env_per_pert: int, first_run: bool, job_id: int) -> tuple:
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
            job_id: Unique identifier for this job
            
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
            job_id,
        )
    
    def update_value_functions(self, policy: Any, perturbed_agent: Any, perturbed_adv_buf: List[Any], 
                             adversary_buffers: List[Any], batch_size: int, max_grad_norm: float, 
                             n_epochs: int, n_env_per_adv: int, first_run: bool = False) -> None:
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
        total_start_time = time.time()

        # Set required attributes
        policy.num_global_env = n_env_per_adv
        perturbed_agent.policy.num_global_env = perturbed_agent.n_env_per_adv

        # Shard work across GPUs
        shard_start_time = time.time()
        all_indices = list(range(len(adversary_buffers)))
        shards = shard_indices(len(all_indices), self.n_workers)
        shard_end_time = time.time()
        if TIMING:
            print(f"      [Timing] Sharding work: {shard_end_time - shard_start_time:.4f}s")

        # Submit jobs to worker processes
        submit_start_time = time.time()
        active_jobs = []
        for device_id, i_list in enumerate(shards):
            if not i_list:
                continue
            
            job_id = f"job_{device_id}_{int(time.time() * 1000000)}"  # Unique job ID
            job = self._create_update_values_function_job(
                policy, perturbed_agent.policy, i_list, adversary_buffers,
                perturbed_adv_buf, batch_size, max_grad_norm, n_epochs,
                n_env_per_adv, perturbed_agent.n_env_per_adv, first_run, job_id
            )
            # self.input_queues[device_id].put(job)
            self.input_queues[device_id].put(("UPDATE_VALUE_FUNCTIONS", job))
            active_jobs.append(job_id)
        submit_end_time = time.time()
        if TIMING:
            print(f"      [Timing] Submitting jobs: {submit_end_time - submit_start_time:.4f}s")

        # Wait for all jobs to complete
        wait_start_time = time.time()
        completed_jobs = 0
        while completed_jobs < len(active_jobs):
            try:
                result = self.done_queue.get(timeout=60)  # 30 second timeout
                if isinstance(result, str) and result.startswith("ERROR_"):
                    print(f"Job failed: {result}")
                completed_jobs += 1
            except Empty:
                print("Warning: Timeout waiting for job completion")
                break
        wait_end_time = time.time()
        if TIMING:
            print(f"      [Timing] Waiting for jobs to complete: {wait_end_time - wait_start_time:.4f}s")
            
        total_end_time = time.time()
        if TIMING:
            print(f"    [Timing] Total ParallelUpdater.update_value_functions: {total_end_time - total_start_time:.4f}s")
            
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
            use_mirror=False,
            env_generator_func=None, #The function is used to create a copy of the environment
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
        self.parallel_updater = None
        self.first_run = False
        self.env_generator_func = env_generator_func

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
        # Copy observation states
        test._last_obs = self._last_obs.copy() if self._last_obs is not None else None
        test._last_episode_starts = self._last_episode_starts.copy() if self._last_episode_starts is not None else None
        return test

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        #self.inner_loop()
        #self.perturbed_buf = perturbed_buf
        #self.perturbed_adv_buf = perturbed_adv_buf

        # 3. Update value functions for both original and perturbed agents
        start_time = time.time()
        self._update_value_functions(self.perturbed_agent, self.perturbed_adv_buf)
        end_time = time.time()
        if TIMING:
            print(f"Time for _update_value_functions: {end_time - start_time:.4f}s")

        #self.perturbed_agent_policy = perturbed_agent.policy
        self.leader_grads(self.rollout_buffer, self.perturbed_buf, self.policy, self.perturbed_agent_policy, ego=True)
        self.leader_grads(self.adversary_buffers, self.perturbed_adv_buf, self.policy, self.perturbed_agent_policy, ego=False)
        del self.perturbed_agent_policy
        del self.perturbed_buf
        del self.perturbed_adv_buf
        gc.collect()
        torch.cuda.empty_cache()
    
    def perturb_params(self, param_list):
        count = 0
        for i in range(len(param_list)):
            count = count + torch.numel(param_list[i])
        delta = .1
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
        self.perturb_params(other_ego)
        self.perturb_params(other_adv)
        
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
                                                    self.n_epochs, self.n_env_per_adv, self.first_run
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
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_losses, pg_losses, approx_kl_divs_all = [], [], []

        num_runs_count = 1 if ego else self.num_adversaries

        for i in range(num_runs_count):
            network_keys, curr_buf, curr_perturbed_buf = self._get_buffers_and_keys(ori_buf, perturbed_buf, ego, i)
            
            approx_kl_divs_epoch = []
            
            for ori_rollout_data, perturbed_rollout_data in zip(curr_buf.get(self.batch_size), curr_perturbed_buf.get(self.batch_size)):
                
                policy_loss, log_prob, entropy = self._calculate_policy_loss(
                    ori_rollout_data, ori_policy, ego, network_keys, clip_range
                )
                pg_losses.append(policy_loss.item())
                entropy_losses.append(entropy.mean().item())

                perturbed_policy_loss, _, _ = self._calculate_policy_loss(
                    perturbed_rollout_data, perturbed_policy, ego, network_keys, clip_range
                )
                
                self._compute_and_apply_grads(policy_loss, perturbed_policy_loss, ego)
                
                with th.no_grad():
                    old_log_prob_tensor = ori_rollout_data.old_log_prob if ego else ori_rollout_data.old_dstb_log_prob
                    #run forward pass to get the log_prob
                    _, log_prob, entropy, _, _ = ori_policy.evaluate_actions(
                        torch.Tensor(ori_rollout_data.observations).to(self.device), torch.Tensor(ori_rollout_data.actions).to(self.device), torch.Tensor(ori_rollout_data.dstb_actions).to(self.device),
                        shuffle_keys=ori_rollout_data.env_indices, network_keys=network_keys
                    )
                    #run forward pass to get the log_prob
                    #_, log_prob, entropy, _, _ = perturbed_policy.evaluate_actions(
                    log_ratio = log_prob - old_log_prob_tensor
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs_epoch.append(approx_kl_div)
           
            approx_kl_divs_all.extend(approx_kl_divs_epoch)

        self._n_updates += self.n_epochs
        if hasattr(self.rollout_buffer, 'values') and self.rollout_buffer.values is not None and self.rollout_buffer.returns is not None:
             explained_var = explained_variance(self.rollout_buffer.values.flatten().cpu().numpy(), self.rollout_buffer.returns.flatten().cpu().numpy())
        else:
            explained_var = np.nan
        self._log_leader_metrics(ego, entropy_losses, pg_losses, approx_kl_divs_all, explained_var, clip_range)

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
                    shuffle_keys=rollout_data.env_indices, network_keys=network_keys
                )
            else:
                old_log_prob = rollout_data.old_dstb_log_prob
                _, _, _, log_prob, entropy = policy.evaluate_actions(
                    torch.Tensor(rollout_data.observations).to(self.device), actions, dstb_actions,
                    shuffle_keys=rollout_data.env_indices, network_keys=network_keys
                )
        
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - torch.Tensor(old_log_prob).to(self.device))
        
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = torch.min(policy_loss_1, policy_loss_2).mean()
        
        return policy_loss, log_prob, entropy

    def _compute_and_apply_grads(self, policy_loss, perturbed_policy_loss, ego):
        F = self.d / self.delta * (perturbed_policy_loss - policy_loss) * self.v
        
        param_list = self.policy.ctrl_optimizer.param_groups[0]['params'] if ego else self.policy.dstb_optimizer.param_groups[0]['params']
        size_lists = [list(x.shape) for x in param_list]
        
        reshaped_grad = []
        count = 0
        for i in range(len(size_lists)):
            numel = np.prod(size_lists[i])
            reshaped_grad.append(torch.reshape(F[count: count + numel], size_lists[i]))
            count += numel

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
                # continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, n_rollout_steps=self.n_steps) #TODO: This is sequential - remove when done.

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

                self.train()

            callback.on_training_end()
        
        finally:
            #IMPORTANT! Persistent workers must be cleaned up.
            self.cleanup()

        return self
