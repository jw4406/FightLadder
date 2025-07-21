import torch
import gc
import torch as th, time, sys
import numpy as np
from gym import spaces
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

DEBUG = True
TIMING = False

class DummyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True

def _print_gpu(tag=""):
    if DEBUG:
        print(f"[{tag}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB | Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

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
            use_mirror=False
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
        return test

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        if DEBUG:
            print("="*20)
        _print_gpu("Before inner_loop")
        self.inner_loop()
        _print_gpu("Before leader_grads #1")
        self.leader_grads(self.rollout_buffer, self.perturbed_buf, self.policy, self.perturbed_agent_policy, ego=True)
        _print_gpu("Before leader_grads #2")
        self.leader_grads(self.adversary_buffers, self.perturbed_adv_buf, self.policy, self.perturbed_agent_policy, ego=False)
        _print_gpu("Before del self.perturbed_agent")
        del self.perturbed_agent_policy
        _print_gpu("Before del self.perturbed_buf")
        del self.perturbed_buf
        _print_gpu("Before del self.perturbed_adv_buf")
        del self.perturbed_adv_buf
        _print_gpu("Before gc.collect()")
        gc.collect()
        _print_gpu("Before torch.cuda.empty_cache()")
        torch.cuda.empty_cache()
        _print_gpu("At the end of train")
    
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
        return perturbed_agent, other_ego, other_adv

    def _update_value_functions(self, perturbed_agent, perturbed_adv_buf):
        for i in range(len(self.adversary_buffers)):
            for epoch in range(self.n_epochs):
                # Update value function for the original agent
                self._update_single_value_function(self.policy, self.adversary_buffers[i], i, self.n_env_per_adv, self.device, "original")
                
                # Update value function for the perturbed agent
                self._update_single_value_function(perturbed_agent.policy, perturbed_adv_buf[i], i, perturbed_agent.n_env_per_adv, perturbed_agent.device, "perturbed")

    def _update_single_value_function(self, policy, buffer, adversary_index, num_envs, device, tag=""):
        total_start_time = time.time()

        get_start_time = time.time()
        rollout_data_list = list(buffer.get(self.batch_size))
        get_end_time = time.time()
        if TIMING:
            print(f"  Time for buffer.get() ({tag}): {get_end_time - get_start_time:.4f}s")

        for rollout_data in rollout_data_list:
            loop_start_time = time.time()

            start_time = time.time()
            actions = torch.Tensor(rollout_data.actions).to(device)
            dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(device)
            end_time = time.time()
            if TIMING:
                print(f"    Time for tensor conversion ({tag}): {end_time - start_time:.4f}s")

            policy.num_global_env = num_envs
            policy.num_adv = 1
            
            start_time = time.time()
            values, _, _, _, _ = policy.evaluate_actions(
                rollout_data.observations, #Changed to torch.from_numpy, a bit safer. #Big memory spike here
                actions,
                dstb_actions,
                shuffle_keys=rollout_data.env_indices,
                network_keys=[adversary_index]
            )
            values = values.flatten()
            end_time = time.time()
            if TIMING:
                print(f"    Time for policy.evaluate_actions ({tag}): {end_time - start_time:.4f}s")

            start_time = time.time()
            value_loss = F.mse_loss(torch.Tensor(-rollout_data.returns).to(device), values)
            end_time = time.time()
            if TIMING:
                print(f"    Time for F.mse_loss ({tag}): {end_time - start_time:.4f}s")

            start_time = time.time()
            policy.value_optimizer.zero_grad()
            if hasattr(policy, 'ctrl_optimizer') and policy.ctrl_optimizer:
                policy.ctrl_optimizer.zero_grad()
            if hasattr(policy, 'dstb_optimizer') and policy.dstb_optimizer:
                policy.dstb_optimizer.zero_grad()
            end_time = time.time()
            if TIMING:
                print(f"    Time for optimizers.zero_grad ({tag}): {end_time - start_time:.4f}s")
            
            start_time = time.time()
            value_loss.backward()
            end_time = time.time()
            if TIMING:
                print(f"    Time for value_loss.backward ({tag}): {end_time - start_time:.4f}s")
            
            start_time = time.time()
            th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            end_time = time.time()
            if TIMING:
                print(f"    Time for clip_grad_norm_ ({tag}): {end_time - start_time:.4f}s")

            start_time = time.time()
            policy.value_optimizer.step()
            end_time = time.time()
            if TIMING:
                print(f"    Time for value_optimizer.step ({tag}): {end_time - start_time:.4f}s")

            loop_end_time = time.time()
            if TIMING:
                print(f"  Time for one loop iteration ({tag}): {loop_end_time - loop_start_time:.4f}s")
        total_end_time = time.time()
        if TIMING:
            print(f"Time for _update_single_value_function ({tag}): {total_end_time - total_start_time:.4f}s")

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

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, n_rollout_steps=self.n_steps)
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

        return self
