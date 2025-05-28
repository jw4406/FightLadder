import os
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np, copy, scipy
import torch
import torch as th
from gym.spaces import MultiDiscrete, MultiBinary
from gymnasium import spaces
from torch.nn import functional as F
import torch.autograd as autograd
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy, MlPAACPolicy, MLPAACCNNPolicy
from functorch import make_functional_with_buffers, make_functional, vmap, grad, jacrev, hessian
# import multiprocess
from scipy.sparse.linalg import cg, bicgstab, LinearOperator
# multiprocess.set_start_method('spawn', force=True)
import time
#from multiprocess.pool import ThreadPool as bitx
#import multiprocess
#from multiprocess import shared_memory

torch.set_float32_matmul_precision('high')
SelfSAC = TypeVar("SelfSAC", bound="SAC")


# global pool
# pool = bitx(processes=os.cpu_count())

class MAGICS_AL(OffPolicyAlgorithm):
    """
    Stackelberg Minimax ARl Training
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "MlPAACPolicy": MlPAACPolicy,
        "MLPAACCNNPolicy": MLPAACCNNPolicy
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
            self,
            policy: Union[str, Type[SACPolicy]],
            env: Union[GymEnv, str],
            c_learning_rate: Union[float, Schedule] = 1e-4,
            d_learning_rate: Union[float, Schedule] = 7e-4,
            v_learning_rate: Union[float, Schedule] = 7e-4,
            c_learning_rate_decay: Union[float, Schedule] = 1e-4,
            d_learning_rate_decay: Union[float, Schedule] = 7e-4,
            v_learning_rate_decay: Union[float, Schedule] = 7e-4,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            use_stackelberg: bool = True,
            dstb_action_space: spaces.Space = None,
            linear_phase: bool = True,
            use_ef: bool = False,
            zofo: bool = False,
            diag: bool = True,
            d_see_u: bool = False
    ):
        super().__init__(
            policy,
            env,
            v_learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box, MultiDiscrete, MultiBinary),
            support_multi_env=True,
        )
        self.use_stackelberg = use_stackelberg
        self.linear_phase = linear_phase
        self.use_ef = use_ef
        self.zofo = zofo
        self.diag = diag
        self.d_see_u = d_see_u
        print("using e-fim: %r" % self.use_ef, flush=True)
        print("using zofo: %r" % self.zofo, flush=True)
        print("using diag: %r" % self.diag, flush=True)
        print("dstb sees ctrl input %r" % self.d_see_u, flush=True)
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        self.dstb_log_ent_coef = None
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.dstb_ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.c_learning_rate = c_learning_rate
        self.v_learning_rate = v_learning_rate
        self.d_learning_rate = d_learning_rate
        self.learning_rate = [v_learning_rate, c_learning_rate, d_learning_rate]
        self.learning_rate_decay_phase = [v_learning_rate_decay, c_learning_rate_decay, d_learning_rate_decay]
        self.smart = True
        self.use_leaderboard = False
        # TODO: leaderboard
        self.max_q_grad_norm = 1
        self.max_u_grad_norm = 1
        self.max_d_grad_norm = 1
        self.q_norm = 0
        self.d_norm = 0
        self.u_norm = 0
        self.policy_kwargs['dstb_action_space'] = dstb_action_space
        self.policy_kwargs['d_see_u'] = self.d_see_u
        #if self.d_see_u is True:
        #    self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.observation_space.shape[0] + self.action_space.shape[0],))
        if dstb_action_space is None:
            self.dstb_action_space = env.action_space
        else:
            self.dstb_action_space = dstb_action_space

        if _init_setup_model:
            self._setup_model()

        # self.cent_diff_compiled = torch.compile(self.cent_diff, mode='max-autotune', fullgraph=True)
        # self.forward_diff_compiled = torch.compile(self.forward_diff, mode='max-autotune')

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule[0](1))
            self.dstb_log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.dstb_ent_coef_optimizer = th.optim.Adam([self.dstb_log_ent_coef], lr=self.lr_schedule[0](1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
            self.dstb_ent_coef_tensor = th.tensor(float(self.dstb_ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.dstb_actor = self.policy.dstb_actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.critic.optimizer, self.actor.optimizer, self.dstb_actor.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
            optimizers += [self.dstb_ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs, dstb_ent_coefs = [], [], []
        actor_losses, critic_losses, dstb_actor_losses = [], [], []
        batch_start = time.time()
        for gradient_step in range(gradient_steps):
            start = time.time()
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()
                self.dstb_actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            if self.d_see_u is True:
                dstb_actions_pi, dstb_log_prob = self.dstb_actor.action_log_prob(th.cat((replay_data.observations, actions_pi),dim=1))
            else:
                dstb_actions_pi, dstb_log_prob = self.dstb_actor.action_log_prob(replay_data.observations)
            # dstb_actions_pi = th.zeros(dstb_actions_pi.shape, device=self.device)
            log_prob = log_prob.reshape(-1, 1)
            dstb_log_prob = dstb_log_prob.reshape(-1, 1)

            ent_coef_loss, dstb_ent_coef_loss = None, None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                dstb_ent_coef = th.exp(self.dstb_log_ent_coef.detach())
                dstb_ent_coef_loss = -(self.dstb_log_ent_coef * (dstb_log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
                dstb_ent_coef = self.dstb_ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            dstb_ent_coefs.append(dstb_ent_coef.item())
            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.dstb_ent_coef_optimizer.zero_grad()
                dstb_ent_coef_loss.backward()
                self.dstb_ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                if self.d_see_u is True:
                    next_dstb_actions, next_dstb_log_prob = self.dstb_actor.action_log_prob(th.cat((replay_data.next_observations, next_actions),dim=1))
                else:
                    next_dstb_actions, next_dstb_log_prob = self.dstb_actor.action_log_prob(replay_data.next_observations)
                # next_dstb_actions = th.zeros(next_dstb_actions.shape, device=self.device)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions, next_dstb_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # next_q_values = next_q_values[:, 0, None]
                # add entropy term
                # next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)# + dstb_ent_coef * next_dstb_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions, replay_data.dstb_actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            if self.use_stackelberg is True:
                num_ctrl_params = 0
                for ele in self.policy.actor.optimizer.param_groups[0]['params']:
                    num_ctrl_params = num_ctrl_params + torch.numel(ele)
                num_dstb_params = 0
                for ele in self.policy.dstb_actor.optimizer.param_groups[0]['params']:
                    num_dstb_params = num_dstb_params + torch.numel(ele)
                num_critic_params = 0
                for ele in self.policy.critic.optimizer.param_groups[0]['params']:
                    num_critic_params = num_critic_params + torch.numel(ele)


                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                if self.d_see_u is True:
                    dstb_actions_pi, dstb_log_prob = self.dstb_actor.action_log_prob(th.cat((replay_data.observations, actions_pi),dim=1))
                else:
                    dstb_actions_pi, dstb_log_prob = self.dstb_actor.action_log_prob(replay_data.observations)
                critic_pred = self.critic(replay_data.observations, actions_pi, dstb_actions_pi)
                # tmp1 = autograd.grad(critic_pred[0][0], self.actor.optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)
                # tmp2 = autograd.grad(tmp1[0][0][0], self.critic.parameters()[:6], create_graph=True, retain_graph=True)

                # critic_pred_sum = torch.add(critic_pred[0], critic_pred[1])
                # surr_q_value_pre_mean = torch.div(critic_pred_sum, 2)

                surr_q_values = torch.mean(torch.sum(torch.hstack((critic_pred[0], critic_pred[1])), dim=1))

                time_start = time.time()
                #ctrl_partial_batched = autograd.grad(-surr_q_values,
                #                                      self.policy.actor.optimizer.param_groups[0]['params'],
                #                                      create_graph=True, retain_graph=True)
                #ctrl_partial = torch.hstack([t.flatten() for t in ctrl_partial_batched])

                #dstb_partial_batched = autograd.grad(surr_q_values,
                #                                     self.policy.dstb_actor.optimizer.param_groups[0]['params'],
                 #                                     create_graph=True, retain_graph=True)
                #dstb_partial = torch.hstack([t.flatten() for t in dstb_partial_batched])
                #h1_pre_omega = torch.hstack((h1_upper, h1_lower))


                # Diagonal terms (Hessians) first
                if self.use_ef is False and self.zofo is False:  # compute true hessians OR diagonal only
                    #time_start = time.time()

                    L_ctrl_grad_batched = autograd.grad(critic_loss, self.critic.optimizer.param_groups[0]['params'],
                                                create_graph=True,retain_graph=True)


                    L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])

                    if self.diag is True:
                        k = 30
                        n = sum(p.numel() for p in self.critic.parameters())

                        rademacher = torch.bernoulli(torch.from_numpy(np.ones((n, k)) * .5)).to(self.device)
                        rademacher[rademacher == 0] = -1
                        # grad_batched = autograd.grad(L_ctrl_grad, flat_params, rademacher,0,1, is_grads_batched=True)
                        grad_batched = autograd.grad(L_ctrl_grad, self.critic.optimizer.param_groups[0]['params'],
                                                     torch.transpose(rademacher.to(self.device), 0, 1),
                                                     is_grads_batched=True,
                                                     retain_graph=True, create_graph=True)

                        reshaped_grads = self.matrix_unbatch(grad_batched, k, size2=n).T
                        reshaped_grads *= rademacher
                        L_ctrl_hessian = torch.mean(reshaped_grads, dim=1)
                        L_ctrl_hessian += 5
                    else:
                        L_ctrl_hessian_batched = autograd.grad(L_ctrl_grad, self.critic.optimizer.param_groups[0]['params'],
                                                       torch.eye(L_ctrl_grad.shape[0], device=self.device),
                                                             is_grads_batched=True)
                        L_ctrl_hessian = self.matrix_unbatch(L_ctrl_hessian_batched, L_ctrl_grad.shape[0]).detach()
                        del L_ctrl_hessian_batched
                        reg_param = 5
                        L_ctrl_hessian.diagonal().add_(reg_param)

                    '''
                    DO NOT USE!
                    DO NOT USE!
                    DO NOT USE!
                    def L_hessian_matvec(vec):
                        """
                        input:  numpy array
                        output: numpy array
                        """
                        vec = torch.from_numpy(vec).to(self.device)
                        _Avec = autograd.grad(L_ctrl_grad, self.critic.optimizer.param_groups[0]['params'], vec, retain_graph=True)
                        Avec = torch.cat([g.contiguous().view(-1) for g in _Avec])
                        reg_param = 5
                        Avec += reg_param * vec
                        return np.array(Avec.detach().to('cpu'))

                    #Dvvfv_lo = LinearOperator(shape=(num_critic_params, num_critic_params), matvec=L_hessian_matvec)
                    DO NOT USE!
                    DO NOT USE!
                    DO NOT USE!
                    '''
                    J_ctrl_critic_grad_batched = autograd.grad(-surr_q_values, self.critic.optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)

                    J_ctrl_critic_grad = torch.hstack([t.flatten() for t in J_ctrl_critic_grad_batched])

                    #J_dstb_critic_grad = -J_ctrl_critic_grad

                    # Select action according to policy
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    if self.d_see_u is True:
                        next_dstb_actions, next_dstb_log_prob = self.dstb_actor.action_log_prob(
                            th.cat((replay_data.next_observations, next_actions),dim=1))
                    else:
                        next_dstb_actions, next_dstb_log_prob = self.dstb_actor.action_log_prob(replay_data.next_observations)
                    # next_dstb_actions = th.zeros(next_dstb_actions.shape, device=self.device)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(
                        self.critic_target(replay_data.next_observations, next_actions, next_dstb_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                    # Get current Q-values estimates for each critic network
                    # using action from the replay buffer
                    current_q_values = self.critic(replay_data.observations, replay_data.actions,
                                                   replay_data.dstb_actions)

                    # Compute critic loss
                    critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

                    ctrl_stage_1_mixed_batched = autograd.grad(critic_loss, self.critic.optimizer.param_groups[0]['params'],
                                                       create_graph=True, retain_graph=True)

                    ctrl_stage_1_mixed = torch.hstack([t.flatten() for t in ctrl_stage_1_mixed_batched])

                    #iHvp_ctrl = self.kaczmarz(L_ctrl_grad, J_ctrl_critic_grad)
                    if len(L_ctrl_hessian.shape) == 1 and self.diag is True:
                        iHvp_ctrl = torch.mul(L_ctrl_hessian.pow_(-1), J_ctrl_critic_grad)
                    else:
                        iHvp_ctrl = torch.linalg.solve(L_ctrl_hessian, J_ctrl_critic_grad)
                    #iHvp_ctrl = torch.from_numpy(bicgstab(Dvvfv_lo, J_ctrl_critic_grad.detach().to('cpu').numpy())[0]).to(self.device)
                    #iHvp_ctrl = torch.from_numpy(
                    #    scipy.sparse.linalg.spsolve(Dvvfv_lo, J_ctrl_critic_grad.detach().to('cpu').numpy())).to(self.device)
                    iHvp_dstb = -iHvp_ctrl

                    ctrl_imp_batched = autograd.grad(ctrl_stage_1_mixed, self.policy.actor.optimizer.param_groups[0]['params'],
                                             iHvp_ctrl, is_grads_batched=False, create_graph=True, retain_graph=True)

                    dstb_imp_batched = autograd.grad(ctrl_stage_1_mixed, self.policy.dstb_actor.optimizer.param_groups[0]['params'],
                                                     iHvp_dstb, is_grads_batched=False, create_graph=True, retain_graph=True)

                    #ctrl_imp = torch.hstack([t.flatten() for t in ctrl_imp_batched])
                    #dstb_imp = torch.hstack([t.flatten() for t in dstb_imp_batched])
                    #ctrl_stackelberg_der = ctrl_partial - ctrl_imp
                    #dstb_stackelberg_der = dstb_partial - dstb_imp

                    min_qf_pi, _ = th.min(th.cat(current_q_values, dim=1), dim=1, keepdim=True)
                    # min_qf_pi = min_qf_pi.detach()
                    actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    for i in range(len(self.actor.optimizer.param_groups[0]['params'])):
                        self.actor.optimizer.param_groups[0]['params'][i].grad = \
                            self.actor.optimizer.param_groups[0]['params'][i].grad - ctrl_imp_batched[i]
                    self.actor.optimizer.step()
                    actor_losses.append(actor_loss.detach().cpu().numpy())
                    q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi, dstb_actions_pi), dim=1)
                    min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                    dstb_actor_loss = (dstb_ent_coef * dstb_log_prob + min_qf_pi).mean()
                    self.dstb_actor.optimizer.zero_grad()
                    grad = autograd.grad(dstb_actor_loss, self.dstb_actor.parameters())
                    self.dstb_actor.optimizer.zero_grad()
                    for i in range(len(grad)):
                        self.dstb_actor.optimizer.param_groups[0]['params'][i].grad = grad[i]
                    for i in range(len(self.dstb_actor.optimizer.param_groups[0]['params'])):
                        self.dstb_actor.optimizer.param_groups[0]['params'][i].grad = \
                            self.dstb_actor.optimizer.param_groups[0]['params'][i].grad - dstb_imp_batched[i]
                    self.dstb_actor.optimizer.step()
                    dstb_actor_losses.append(dstb_actor_loss)
                    end = time.time() - time_start
                    #print("elapsed: %.2f" % end)
                elif self.use_ef is True and self.zofo is False:
                    time_start = time.time()

                    # Step 1: Compute batched gradients using autograd with is_grads_batched=True
                    def compute_batched_grads(arb_log_probs, params):
                        batch_size = arb_log_probs.shape[0]
                        identity = torch.eye(batch_size, device=log_prob.device)  # Identity matrix for batching
                        batched_grads = torch.autograd.grad(
                            arb_log_probs,
                            params,
                            identity,  # Batching through the identity matrix
                            is_grads_batched=True,  # Enable batched gradient computation
                            create_graph=True,
                            retain_graph=True
                        )
                        # return torch.hstack([t.flatten() for t in batched_grads])
                        flattened_grads = [g.view(batch_size, -1) for g in batched_grads]
                        return torch.cat(flattened_grads, dim=1)

                    # Step 2: Calculate the gradients for the entire batch
                    actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                    dstb_actions_pi, dstb_log_prob = self.dstb_actor.action_log_prob(replay_data.observations)
                    critic_pred = self.critic(replay_data.observations, actions_pi, dstb_actions_pi)
                    grad_theta_flattened_batch = compute_batched_grads(log_prob,
                                                                       self.policy.actor.optimizer.param_groups[0][
                                                                           'params'])
                    grad_psi_flattened_batch = compute_batched_grads(dstb_log_prob,
                                                                     self.policy.dstb_actor.optimizer.param_groups[0][
                                                                         'params'])
                    # grad_theta_flattened_batch = torch.cat([g.view(log_prob.shape[0], -1) for g in grad_theta_batch], dim=1)
                    # grad_psi_flattened_batch = torch.cat([g.view(dstb_log_prob.shape[0], -1) for g in grad_psi_batch],
                    #                                     dim=1)

                    # Step 3: Compute e-FIM using outer products
                    fim_theta = torch.einsum('bi,bj->ij', grad_theta_flattened_batch,
                                             grad_theta_flattened_batch) / batch_size
                    fim_psi = torch.einsum('bi,bj->ij', grad_psi_flattened_batch, grad_psi_flattened_batch) / batch_size
                    fim_theta_psi = torch.einsum('bi,bj->ij', grad_theta_flattened_batch,
                                                 grad_psi_flattened_batch) / batch_size

                    # Step 4: Assemble the big H matrix
                    grad_theta_psi_J_t = torch.transpose(fim_theta_psi, 0, 1)  # this is the 2,1 position

                    upper_rows = torch.cat((fim_theta, fim_theta_psi), dim=1)
                    lower_rows = torch.cat((grad_theta_psi_J_t, fim_psi), dim=1)

                    H = torch.cat((upper_rows, lower_rows), dim=0)
                else:
                    ctrl_estimator, dstb_estimator = self.derivative_free(replay_data, num_ctrl_params, num_dstb_params)
                    self.actor.optimizer.zero_grad()
                    for i in range(len(ctrl_estimator)):
                        self.actor.optimizer.param_groups[0]['params'][i].grad = ctrl_estimator[i].float()
                    self.actor.optimizer.step()
                    for i in range(len(dstb_estimator)):
                        self.dstb_actor.optimizer.param_groups[0]['params'][i].grad = dstb_estimator[i].float()
                    self.dstb_actor.optimizer.step()

                    L_ctrl_grad_batched = autograd.grad(critic_loss, self.critic.optimizer.param_groups[0]['params'],
                                                        create_graph=True, retain_graph=True)

                    L_ctrl_grad = torch.hstack([t.flatten() for t in L_ctrl_grad_batched])


            else:
                current_q_values = self.critic(replay_data.observations, replay_data.actions, replay_data.dstb_actions)
                min_qf_pi, _ = th.min(th.cat(current_q_values, dim=1), dim=1, keepdim=True)
                # min_qf_pi = min_qf_pi.detach()
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self.use_stackelberg is True:
                    for i in range(len(self.actor.optimizer.param_groups[0]['params'])):
                        self.actor.optimizer.param_groups[0]['params'][i].grad = \
                            self.actor.optimizer.param_groups[0]['params'][i].grad - ctrl_imp_batched[i]
                self.actor.optimizer.step()
                actor_losses.append(actor_loss.detach().cpu().numpy())
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi, dstb_actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                dstb_actor_loss = (dstb_ent_coef * dstb_log_prob + min_qf_pi).mean()
                self.dstb_actor.optimizer.zero_grad()
                grad = autograd.grad(dstb_actor_loss, self.dstb_actor.parameters())
                self.dstb_actor.optimizer.zero_grad()
                for i in range(len(grad)):
                    self.dstb_actor.optimizer.param_groups[0]['params'][i].grad = grad[i]
                if self.use_stackelberg is True:

                    for i in range(len(self.dstb_actor.optimizer.param_groups[0]['params'])):
                        self.dstb_actor.optimizer.param_groups[0]['params'][i].grad = \
                            self.dstb_actor.optimizer.param_groups[0]['params'][i].grad - dstb_imp_batched[i]
                self.dstb_actor.optimizer.step()
                dstb_actor_losses.append(dstb_actor_loss)

            # Optimize the critic
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                if self.d_see_u is True:
                    next_dstb_actions, next_dstb_log_prob = self.dstb_actor.action_log_prob(th.cat((replay_data.next_observations, next_actions),dim=1))
                else:
                    next_dstb_actions, next_dstb_log_prob = self.dstb_actor.action_log_prob(replay_data.next_observations)
                # next_dstb_actions = th.zeros(next_dstb_actions.shape, device=self.device)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions, next_dstb_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # next_q_values = next_q_values[:, 0, None]
                # add entropy term
                # next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)# + dstb_ent_coef * next_dstb_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions, replay_data.dstb_actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
            '''
            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi, dstb_actions_pi), dim=1)
            if self.use_ef is True:
                _, log_prob = self.actor.action_log_prob(replay_data.observations)
                _, dstb_log_prob = self.dstb_actor.action_log_prob(replay_data.observations)

            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # min_qf_pi = min_qf_pi.detach()
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            # torch.autograd.set_detect_anomaly(True)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi, dstb_actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            dstb_actor_loss = (dstb_ent_coef * dstb_log_prob + min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            # dstb_actor_losses.append(dstb_actor_loss.item())

            # Optimize the actor

            # self.dstb_actor.optimizer.zero_grad()
            # dstb_actor_loss.backward()

            # FOR SOME REASON CALLING backward BREAKS THE CODE
            # cue the "fine, i'll do it myself" memes
            grad = autograd.grad(dstb_actor_loss, self.dstb_actor.parameters())
            self.dstb_actor.optimizer.zero_grad()
            for i in range(len(grad)):
                self.dstb_actor.optimizer.param_groups[0]['params'][i].grad = grad[i]

            # self.actor.optimizer.step()
            self.dstb_actor.optimizer.step()
            # self.dstb_actor.optimizer.zero_grad()
            # dstb_actor_loss.backward()
            '''
            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            q_norm = 0
            u_norm = 0
            d_norm = 0
            for i in range(len(self.critic.optimizer.param_groups[0]['params'])):
                q_norm = q_norm + torch.linalg.norm(self.critic.optimizer.param_groups[0]['params'][i].grad)
            for i in range(len(self.actor.optimizer.param_groups[0]['params'])):
                u_norm = u_norm + torch.linalg.norm(self.actor.optimizer.param_groups[0]['params'][i].grad)
            if self.use_leaderboard is True:
                for i in range(len(
                        self.policy.policy_memory[self.dstb_model_choice].dstb_optimizer.param_groups[0]['params'])):
                    d_norm = d_norm + torch.linalg.norm(
                        self.policy.policy_memory[self.dstb_model_choice].dstb_optimizer.param_groups[0]['params'][
                            i].grad)
            else:
                for i in range(len(self.dstb_actor.optimizer.param_groups[0]['params'])):
                    d_norm = d_norm + torch.linalg.norm(self.dstb_actor.optimizer.param_groups[0]['params'][i].grad)
                pass
            self.q_norm = q_norm
            self.d_norm = d_norm
            self.u_norm = u_norm
            if self.q_norm > self.max_q_grad_norm:
                self.max_q_grad_norm = self.q_norm
            if self.u_norm > self.max_u_grad_norm:
                self.max_u_grad_norm = self.u_norm
            if self.d_norm > self.max_d_grad_norm:
                self.max_d_grad_norm = self.d_norm
        elapsed = time.time() - batch_start
        print("batch elapsed: %.2f" % elapsed)
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/dstb_ent_coef", np.mean(dstb_ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
            self: SelfSAC,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "SAC",
            reset_num_timesteps: bool = False,
            progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def prep_grad_theta_J(self, curr_q_values, ctrl_log_prob, dstb_log_prob):
        return (- curr_q_values + self.ent_coef_tensor * ctrl_log_prob).mean()

    def prep_grad_psi_J(self, curr_q_values, ctrl_log_prob, dstb_log_prob):
        return (curr_q_values + self.ent_coef_tensor * dstb_log_prob).mean()

    def matrix_unbatch(self, to_be_unbatched, size1, size2=None):
        if size2 is None:
            size2 = size1
        unbatched = torch.zeros((size1, size2), device=self.device)
        for jac_row_count in range(size1):
            curr = 0
            for count in range(len(to_be_unbatched)):
                unbatched[jac_row_count,
                curr:curr + len(
                    torch.flatten(to_be_unbatched[count][jac_row_count, :]))] = torch.flatten(
                    to_be_unbatched[count][jac_row_count, :])
                curr = curr + len(torch.flatten(to_be_unbatched[count][jac_row_count, :]))
        return unbatched

    def compute_stateless_q_surr(self, ctrl_model_mu, ctrl_model_log_std, ctrl_model_latent_pi, dstb_model_mu,
                                 dstb_model_log_std,
                                 dstb_model_latent_pi, critic_model, ctrl_model_mu_params, ctrl_model_log_std_params,
                                 ctrl_model_latent_pi_params,
                                 dstb_model_mu_params, dstb_model_log_std_params, dstb_model_latent_pi_params,
                                 critic_params, critic_buffers, obs):
        # actions_pi, log_prob = self.actor.action_log_prob(obs)
        # dstb_actions_pi, dstb_log_prob = self.dstb_actor.action_log_prob(obs)

        ctrl_latent_pi = ctrl_model_latent_pi(ctrl_model_latent_pi_params, obs)
        ctrl_mu = ctrl_model_mu(ctrl_model_mu_params, ctrl_latent_pi)
        ctrl_log_std = ctrl_model_log_std(ctrl_model_log_std_params, ctrl_latent_pi)
        ctrl_std = ctrl_log_std.exp()
        ctrl_action = ctrl_mu + ctrl_std
        dstb_latent_pi = dstb_model_latent_pi(dstb_model_latent_pi_params, obs)
        dstb_mu = dstb_model_mu(dstb_model_mu_params, dstb_latent_pi)
        dstb_log_std = dstb_model_log_std(dstb_model_log_std_params, dstb_latent_pi)
        dstb_std = dstb_log_std.exp()
        dstb_action = dstb_mu + dstb_std

        q_values = critic_model(critic_params, critic_buffers, obs, ctrl_action, dstb_action)
        return q_values

    def cent_diff(self, critic_model, critic_params, critic_buffers, obs, u, d, ctrl_size, dstb_size,
                  use_parallel=True):
        pool = bitx(processes=os.cpu_count())
        delta = torch.rand(1, device=self.device)

        J = torch.zeros((ctrl_size + dstb_size, len(critic_params)), device=self.device)
        # In = I(n)
        # for j in range():
        #    J[:, j] = (f(x0 + delta * In[:, j]) - y0) / delta
        Ij = torch.eye(len(critic_params), device=self.device)
        global parallel_helper

        def parallel_helper(k):
            flat_critic_params_pos = critic_params + delta * Ij[:, k]
            flat_critic_params_neg = critic_params - delta * Ij[:, k]

            reshaped_critic_params_pos = self.critic_param_reshape(flat_critic_params_pos)
            forward_double_q_pred_pos = critic_model(reshaped_critic_params_pos, critic_buffers, obs, u, d)

            surr_q_pos = torch.mean(
                torch.sum(torch.hstack((forward_double_q_pred_pos[0], forward_double_q_pred_pos[1])), dim=1))
            h1_pos = self.compute_stage_1_grad(surr_q_pos)

            reshaped_critic_params_neg = self.critic_param_reshape(flat_critic_params_neg)
            forward_double_q_pred_neg = critic_model(reshaped_critic_params_neg, critic_buffers, obs, u, d)

            surr_q_neg = torch.mean(
                torch.sum(torch.hstack((forward_double_q_pred_neg[0], forward_double_q_pred_neg[1])), dim=1))
            h1_neg = self.compute_stage_1_grad(surr_q_neg)

            return (h1_pos - h1_neg) / (2 * delta)

        if use_parallel is True:
            for idx, val in enumerate(range(len(critic_params))):
                result = pool.map(parallel_helper, [val], chunksize=len(critic_params) // os.cpu_count())
                J[:, idx] = result[0]
            pool.close()
        else:
            for i in range(len(critic_params)):
                J[:, i] = parallel_helper(i)
        '''
        for k in range(len(critic_params)):
            flat_critic_params_pos = critic_params + delta * Ij[:, k]
            flat_critic_params_neg = critic_params - delta * Ij[:, k]

            reshaped_critic_params_pos = self.param_reshape(flat_critic_params_pos)
            forward_double_q_pred_pos = critic_model(reshaped_critic_params_pos, critic_buffers, obs, u, d)

            surr_q_pos = torch.mean(torch.sum(torch.hstack((forward_double_q_pred_pos[0], forward_double_q_pred_pos[1])), dim=1))
            h1_pos = self.compute_stage_1_grad(surr_q_pos)

            reshaped_critic_params_neg = self.param_reshape(flat_critic_params_neg)
            forward_double_q_pred_neg = critic_model(reshaped_critic_params_neg, critic_buffers, obs, u, d)

            surr_q_neg = torch.mean(torch.sum(torch.hstack((forward_double_q_pred_neg[0], forward_double_q_pred_neg[1])), dim=1))
            h1_neg = self.compute_stage_1_grad(surr_q_neg)

            J[:, k] = (h1_pos - h1_neg) / (2*delta)
            '''
        return J

    def critic_param_reshape(self, flat_critic_params):
        reshaped_critic_params = np.zeros(len(self.critic.qf0) + len(self.critic.qf1) + 2, dtype=object)
        params_pointer = 0

        i = 0
        for j in range(0, len(self.critic.qf0) + len(self.critic.qf1) + 2, 2):
            # if i == 0:
            #    reshaped_ctrl_params[i] = ctrl_params[0]
            #    params_pointer = 1
            #    i = i + 1

            if j <= len(self.critic.qf0):
                end = len(self.critic.qf0[j].weight.flatten())
                reshaped_critic_params[i] = torch.reshape(flat_critic_params[params_pointer:params_pointer + end],
                                                          self.critic.qf0[j].weight.shape)
                params_pointer = params_pointer + end
                i = i + 1
                # bias
                end = len(self.critic.qf0[j].bias.flatten())
                reshaped_critic_params[i] = torch.reshape(flat_critic_params[params_pointer:params_pointer + end],
                                                          self.critic.qf0[j].bias.shape)
                params_pointer = params_pointer + end
                i = i + 1
            else:
                m = j - len(self.critic.qf0) - 1
                end = len(self.critic.qf1[m].weight.flatten())
                reshaped_critic_params[i] = torch.reshape(flat_critic_params[params_pointer:params_pointer + end],
                                                          self.critic.qf1[m].weight.shape)
                params_pointer = params_pointer + end
                i = i + 1
                # bias
                end = len(self.critic.qf1[m].bias.flatten())
                reshaped_critic_params[i] = torch.reshape(flat_critic_params[params_pointer:params_pointer + end],
                                                          self.critic.qf1[m].bias.shape)
                params_pointer = params_pointer + end
                i = i + 1
        return reshaped_critic_params

    def compute_stage_1_grad(self, surr_q_values):
        h1_upper_grad_batched = autograd.grad(surr_q_values, list(self.actor.parameters()),
                                              create_graph=True, retain_graph=True)
        h1_upper = torch.hstack([t.flatten() for t in h1_upper_grad_batched])
        # autograd.grad(h1_upper, self.critic.parameters(), torch.eye(4545), is_grads_batched=True, create_graph=True, retain_graph=True)
        h1_lower_grad_batched = autograd.grad(surr_q_values, self.policy.dstb_actor.optimizer.param_groups[0]['params'],
                                              create_graph=True, retain_graph=True)
        h1_lower = torch.hstack([t.flatten() for t in h1_lower_grad_batched])

        h1_pre_omega = torch.hstack((h1_upper, h1_lower))

        return h1_pre_omega

    def forward_diff(self, f_x_0, critic_model, critic_params, critic_buffers, obs, u, d, ctrl_size, dstb_size,
                     use_parallel=False):
        global pool
        pool = bitx(processes=os.cpu_count())
        delta = torch.rand(1, device=self.device)

        J = torch.zeros((ctrl_size + dstb_size, len(critic_params)), device=self.device)
        global result
        result = Result()
        # In = I(n)
        # for j in range():
        #    J[:, j] = (f(x0 + delta * In[:, j]) - y0) / delta
        Ij = torch.eye(len(critic_params), device=self.device)
        global parallel_helper

        def parallel_helper(k):
            ej = torch.zeros((len(critic_params)), device=self.device)
            ej[k] = 1
            flat_critic_params_pos = critic_params + delta * ej
            # flat_critic_params_neg = critic_params - delta * Ij[:, k]

            reshaped_critic_params_pos = self.critic_param_reshape(flat_critic_params_pos)
            forward_double_q_pred_pos = critic_model(reshaped_critic_params_pos, critic_buffers, obs, u, d)

            surr_q_pos = torch.mean(
                torch.sum(torch.hstack((forward_double_q_pred_pos[0], forward_double_q_pred_pos[1])), dim=1))
            h1_pos = self.compute_stage_1_grad(surr_q_pos)

            # reshaped_critic_params_neg = self.param_reshape(flat_critic_params_neg)
            # forward_double_q_pred_neg = critic_model(reshaped_critic_params_neg, critic_buffers, obs, u, d)

            # surr_q_neg = torch.mean(
            #    torch.sum(torch.hstack((forward_double_q_pred_neg[0], forward_double_q_pred_neg[1])), dim=1))
            # h1_neg = self.compute_stage_1_grad(surr_q_neg)

            return k, (h1_pos - f_x_0) / delta

        if use_parallel is True:
            '''
            for idx, val in enumerate(range(len(critic_params))):
                result = bitx.map(parallel_helper, [val], chunksize=len(critic_params)//os.cpu_count())
                J[:, idx] = result[0]
            '''

            for idx, val in enumerate(range(len(critic_params))):
                # result.stash_col(idx)
                pool.apply_async(parallel_helper, (val,), callback=result.update_result)
            # pool.close()

        else:
            for i in range(len(critic_params)):
                J[:, i] = parallel_helper(i)

        return result

    def test_function(self, critic_model, critic_params, critic_buffers, obs, u, d, ctrl_size, dstb_size):
        # delta = torch.rand(1, device=self.device)
        # reshaped_critic_params_pos = self.param_reshape(critic_params)
        # J = torch.zeros((ctrl_size + dstb_size, len(critic_params)), device=self.device)
        # In = I(n)
        # for j in range():
        #    J[:, j] = (f(x0 + delta * In[:, j]) - y0) / delta
        Ij = torch.eye(len(critic_params), device=self.device)

        forward_double_q_pred_pos = critic_model(critic_params, critic_buffers, obs, u, d)

        surr_q_pos = torch.mean(
            torch.sum(torch.hstack((forward_double_q_pred_pos[0], forward_double_q_pred_pos[1])), dim=1))
        h1_pos = self.compute_stage_1_grad(surr_q_pos)
        return h1_pos

    def do_gradients_reversed(self, replay_data, num_critic_params, num_ctrl_params, num_dstb_params):
        delta = torch.rand(1, device=self.device) * .00001
        # player_params = torch.cat((flat_ctrl_params, flat_dstb_params), dim=0)
        J = torch.zeros(num_ctrl_params + num_dstb_params, num_critic_params, device=self.device)

        weights_path = 'weights_temp_large.pt'
        torch.save(self.policy, weights_path)
        hacky_trick = torch.load(weights_path)

        count = 0
        for i in range(2):
            if count <= num_ctrl_params:
                model = hacky_trick.actor
            else:
                model = hacky_trick.dstb_actor
            for p in model.parameters():
                for j in range(torch.numel(p)):
                    indices = np.unravel_index(j, p.shape)
                    with torch.no_grad():
                        old_val = p[indices]
                        new_val = p[indices] + delta
                        p[indices] = new_val[0]

                        if i == 0:
                            actions_pi, log_prob = hacky_trick.actor.action_log_prob(replay_data.observations)
                        if (i == 0 and count == 0) or (i == 1):
                            dstb_actions_pi, dstb_log_prob = hacky_trick.dstb_actor.action_log_prob(
                                replay_data.observations)
                        else:
                            pass
                        # log_prob = log_prob.reshape(-1, 1)
                        # dstb_log_prob = dstb_log_prob.reshape(-1, 1)
                    critic_pred = hacky_trick.critic(replay_data.next_observations, actions_pi, dstb_actions_pi)
                    surr_q_values = torch.mean(torch.sum(torch.hstack((critic_pred[0], critic_pred[1])), dim=1))
                    omega_grads_batched = autograd.grad(surr_q_values, hacky_trick.critic.parameters())
                    omega_grads_pos = torch.hstack([t.flatten() for t in omega_grads_batched])
                    with torch.no_grad():
                        p[indices] = old_val

                    with torch.no_grad():
                        old_val = p[indices]
                        new_val = p[indices] - delta
                        p[indices] = new_val[0]

                        if i == 0:
                            actions_pi, log_prob = hacky_trick.actor.action_log_prob(replay_data.observations)
                        if (i == 0 and count == 0) or (i == 1):
                            dstb_actions_pi, dstb_log_prob = hacky_trick.dstb_actor.action_log_prob(
                                replay_data.observations)
                        else:
                            pass
                        # log_prob = log_prob.reshape(-1, 1)
                        # dstb_log_prob = dstb_log_prob.reshape(-1, 1)
                    critic_pred = hacky_trick.critic(replay_data.next_observations, actions_pi, dstb_actions_pi)
                    surr_q_values = torch.mean(torch.sum(torch.hstack((critic_pred[0], critic_pred[1])), dim=1))
                    omega_grads_batched = autograd.grad(surr_q_values, hacky_trick.critic.parameters())
                    omega_grads_neg = torch.hstack([t.flatten() for t in omega_grads_batched])
                    with torch.no_grad():
                        p[indices] = old_val
                    J[count, :] = (omega_grads_pos - omega_grads_neg) / (2 * delta)

                    count = count + 1

        return J

    def jacobian_vector_product(self, f_x_0, v, epsilon=1e-5):
        # Get all parameters as a single vector
        params = torch.cat([p.view(-1) for p in self.critic.parameters()])
        params_size = params.numel()

        # Convert vector v to numpy array
        v = v.detach().numpy()

        # Perturb the parameters in the direction of v
        original_params = params.clone().detach().numpy()
        perturbed_params = original_params + epsilon * v

        # Update the network parameters with perturbed values
        start_idx = 0
        for param in self.critic.qf0.parameters():
            end_idx = start_idx + param.numel()
            param.data = torch.tensor(perturbed_params[start_idx:end_idx]).view(param.size())
            start_idx = end_idx
        start_idx = 0
        assert params_size // 2 == 0
        offset = params_size / 2

        for param in self.critic.qf1.parameters():
            end_idx = start_idx + param.numel()
            param.data = torch.tensor(original_params[offset + start_idx:offset + end_idx]).view(param.size())
            start_idx = end_idx

        # Compute the loss at perturbed parameters
        forward_double_q_pred_pos = self.critic(obs, u, d)

        surr_q_pos = torch.mean(
            torch.sum(torch.hstack((forward_double_q_pred_pos[0], forward_double_q_pred_pos[1])), dim=1))
        h1_perturbed = self.compute_stage_1_grad(surr_q_pos)
        # output = net(x)
        # perturbed_loss = vector_loss_fn(output, target)

        # Restore the original parameters
        start_idx = 0
        for param in self.critic.qf0.parameters():
            end_idx = start_idx + param.numel()
            param.data = torch.tensor(original_params[start_idx:end_idx]).view(param.size())
            start_idx = end_idx
        start_idx = 0
        for param in net.parameters():
            end_idx = start_idx + param.numel()
            param.data = torch.tensor(original_params[offset + start_idx:offset + end_idx]).view(param.size())
            start_idx = end_idx

        # Compute the original loss

        # get from pass in?

        original_loss = f_x_0

        # Compute the Jacobian-vector product
        jvp = (perturbed_loss - original_loss) / epsilon

        return jvp

    def jtv(self, f_x_0, critic_model, critic_params, critic_buffers, obs, u, d, v, epsilon=1e-5):
        J_T_v = torch.zeros(len(critic_params), dtype=torch.float32)
        for i in range(len(critic_params)):
            ej = torch.zeros((len(critic_params)), device=self.device)
            ej[i] = 1
            flat_critic_params_pos = critic_params + epsilon * ej
            # flat_critic_params_neg = critic_params - delta * Ij[:, k]

            reshaped_critic_params_pos = self.critic_param_reshape(flat_critic_params_pos)
            forward_double_q_pred_pos = critic_model(reshaped_critic_params_pos, critic_buffers, obs, u, d)

            surr_q_pos = torch.mean(
                torch.sum(torch.hstack((forward_double_q_pred_pos[0], forward_double_q_pred_pos[1])), dim=1))
            h1_pos = self.compute_stage_1_grad(surr_q_pos)

            # reshaped_critic_params_neg = self.param_reshape(flat_critic_params_neg)
            # forward_double_q_pred_neg = critic_model(reshaped_critic_params_neg, critic_buffers, obs, u, d)

            # surr_q_neg = torch.mean(
            #    torch.sum(torch.hstack((forward_double_q_pred_neg[0], forward_double_q_pred_neg[1])), dim=1))
            # h1_neg = self.compute_stage_1_grad(surr_q_neg)

            test = (h1_pos - f_x_0) / epsilon

            # Compute the dot product with v
            J_T_v[i] = torch.dot(test, v)

        return J_T_v

    def do_gradients_reversed_singleshot(self, surr_q_pre, replay_data, num_critic_params, num_ctrl_params,
                                         num_dstb_params):
        delta = 1e-5
        f_x_0_batched = autograd.grad(surr_q_pre, self.critic.parameters(), create_graph=True, retain_graph=True)
        f_x_0 = torch.hstack([p.flatten() for p in f_x_0_batched])
        # player_params = torch.cat((flat_ctrl_params, flat_dstb_params), dim=0)
        J = torch.zeros(num_ctrl_params + num_dstb_params, num_critic_params, device=self.device)

        weights_path = 'weights_temp_large.pt'
        torch.save(self.policy, weights_path)
        hacky_trick = torch.load(weights_path)
        count = 0
        for i in range(2):
            if count <= num_ctrl_params:
                model = hacky_trick.actor
            else:
                model = hacky_trick.dstb_actor
            for p in model.parameters():
                for j in range(torch.numel(p)):
                    indices = np.unravel_index(j, p.shape)
                    with torch.no_grad():
                        old_val = p[indices]
                        new_val = p[indices] + delta
                        p[indices] = new_val

                        if i == 0:
                            actions_pi, log_prob = hacky_trick.actor.action_log_prob(replay_data.observations)
                        if (i == 0 and count == 0) or (i == 1):
                            dstb_actions_pi, dstb_log_prob = hacky_trick.dstb_actor.action_log_prob(
                                replay_data.observations)
                        else:
                            pass
                        # log_prob = log_prob.reshape(-1, 1)
                        # dstb_log_prob = dstb_log_prob.reshape(-1, 1)
                    critic_pred = hacky_trick.critic(replay_data.next_observations, actions_pi, dstb_actions_pi)
                    surr_q_values = torch.mean(torch.sum(torch.hstack((critic_pred[0], critic_pred[1])), dim=1))
                    omega_grads_batched = autograd.grad(surr_q_values, hacky_trick.critic.parameters())
                    omega_grads_pos = torch.hstack([t.flatten() for t in omega_grads_batched])
                    with torch.no_grad():
                        p[indices] = old_val

                    J[count, :] = (omega_grads_pos - f_x_0) / delta

                    count = count + 1

        return J

    def derivative_free(self, replay_data, ctrl_d, dstb_d):
        delta = .1
        tol = .1
        K = 1000
        ctrl_select = torch.from_numpy(np.random.uniform(low=-1,high=1,size=ctrl_d)).to(self.device)
        dstb_select = torch.from_numpy(np.random.uniform(low=-1,high=1,size=dstb_d)).to(self.device)

        weights_path = 'descend_%d' % self.seed
        torch.save(self.policy, weights_path)
        v_ctrl = delta * ctrl_select / torch.linalg.norm(ctrl_select)
        v_dstb = delta * dstb_select / torch.linalg.norm(dstb_select)
        descend_copy_model = torch.load(weights_path)
        descend_copy_model.critic.optimizer.param_groups[0]['lr'] *= 10
        print("hello")
        norm = 1
        count = 0
        while norm > tol:
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = descend_copy_model.actor.action_log_prob(replay_data.next_observations)
                next_dstb_actions, next_dstb_log_prob = descend_copy_model.dstb_actor.action_log_prob(replay_data.next_observations)
                # next_dstb_actions = th.zeros(next_dstb_actions.shape, device=self.device)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    descend_copy_model.critic_target(replay_data.next_observations, next_actions, next_dstb_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # next_q_values = next_q_values[:, 0, None]
                # add entropy term
                # next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)# + dstb_ent_coef * next_dstb_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
            current_q_values = descend_copy_model.critic(replay_data.observations, replay_data.actions, replay_data.dstb_actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            #grads_batched = autograd.grad(critic_loss, descend_copy_model.critic.parameters())
            #grads = torch.hstack([t.flatten() for t in grads_batched])
            descend_copy_model.critic.optimizer.zero_grad()
            critic_loss.backward()
            descend_copy_model.critic.optimizer.step()
            norm = 0
            for j in range(len(descend_copy_model.critic.optimizer.param_groups[0]['params'])):
                norm = norm + torch.linalg.norm(descend_copy_model.critic.optimizer.param_groups[0]['params'][j].grad)
            count = count + 1
        x_q_vals = descend_copy_model.critic(replay_data.observations, replay_data.actions, replay_data.dstb_actions)
        f_x = torch.mean(torch.sum(torch.hstack((x_q_vals[0], x_q_vals[1])), dim=1))
        # tmp1 = autograd.grad(critic_pred[0][0], self.actor.optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True)
        # tmp2 = autograd.grad(tmp1[0][0][0], self.critic.parameters()[:6], create_graph=True, retain_graph=True)

        # critic_pred_sum = torch.add(critic_pred[0], critic_pred[1])
        # surr_q_value_pre_mean = torch.div(critic_pred_sum, 2)

        #surr_q_values = torch.mean(torch.sum(torch.hstack((critic_pred[0], critic_pred[1])), dim=1))
        descend_copy_model = torch.load(weights_path)
        descend_copy_model.critic.optimizer.param_groups[0]['lr'] *= 10
        count = 0
        with torch.no_grad():
            for p in descend_copy_model.actor.parameters():
                p.copy_(p + torch.reshape(v_ctrl[count:count + torch.numel(p)], p.shape).to(self.device))
        norm = 1
        while norm > tol:
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = descend_copy_model.actor.action_log_prob(
                    replay_data.next_observations)
                next_dstb_actions, next_dstb_log_prob = descend_copy_model.dstb_actor.action_log_prob(
                    replay_data.next_observations)
                # next_dstb_actions = th.zeros(next_dstb_actions.shape, device=self.device)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    descend_copy_model.critic_target(replay_data.next_observations, next_actions,
                                                     next_dstb_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # next_q_values = next_q_values[:, 0, None]
                # add entropy term
                # next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)# + dstb_ent_coef * next_dstb_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
            current_q_values = descend_copy_model.critic(replay_data.observations, replay_data.actions,
                                                         replay_data.dstb_actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            # grads_batched = autograd.grad(critic_loss, descend_copy_model.critic.parameters())
            # grads = torch.hstack([t.flatten() for t in grads_batched])
            descend_copy_model.critic.optimizer.zero_grad()
            critic_loss.backward()
            descend_copy_model.critic.optimizer.step()
            norm = 0
            for j in range(len(descend_copy_model.critic.optimizer.param_groups[0]['params'])):
                norm = norm + torch.linalg.norm(descend_copy_model.critic.optimizer.param_groups[0]['params'][j].grad)
            count = count + 1

        xhat_q_vals = descend_copy_model.critic(replay_data.observations, replay_data.actions,
                                             replay_data.dstb_actions)
        f_xhat = torch.mean(torch.sum(torch.hstack((xhat_q_vals[0], xhat_q_vals[1])), dim=1))

        ctrl_estimator = -ctrl_d / delta * (f_xhat - f_x) * v_ctrl
        count = 0
        descend_copy_model = torch.load(weights_path)
        descend_copy_model.critic.optimizer.param_groups[0]['lr'] *= 10
        with torch.no_grad():
            for p in descend_copy_model.dstb_actor.parameters():
                p.copy_(p + torch.reshape(v_dstb[count:count + torch.numel(p)], p.shape).to(self.device))
        norm = 1
        while norm > tol:
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = descend_copy_model.actor.action_log_prob(
                    replay_data.next_observations)
                next_dstb_actions, next_dstb_log_prob = descend_copy_model.dstb_actor.action_log_prob(
                    replay_data.next_observations)
                # next_dstb_actions = th.zeros(next_dstb_actions.shape, device=self.device)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    descend_copy_model.critic_target(replay_data.next_observations, next_actions,
                                                     next_dstb_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # next_q_values = next_q_values[:, 0, None]
                # add entropy term
                # next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)# + dstb_ent_coef * next_dstb_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
            current_q_values = descend_copy_model.critic(replay_data.observations, replay_data.actions,
                                                         replay_data.dstb_actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            # grads_batched = autograd.grad(critic_loss, descend_copy_model.critic.parameters())
            # grads = torch.hstack([t.flatten() for t in grads_batched])
            descend_copy_model.critic.optimizer.zero_grad()
            critic_loss.backward()
            descend_copy_model.critic.optimizer.step()
            norm = 0
            for j in range(len(descend_copy_model.critic.optimizer.param_groups[0]['params'])):
                norm = norm + torch.linalg.norm(descend_copy_model.critic.optimizer.param_groups[0]['params'][j].grad)
            count = count + 1

        xhat_q_vals = descend_copy_model.critic(replay_data.observations, replay_data.actions,
                                             replay_data.dstb_actions)
        f_xhat = torch.mean(torch.sum(torch.hstack((xhat_q_vals[0], xhat_q_vals[1])), dim=1))

        dstb_estimator = ctrl_d / delta * (f_xhat - f_x) * v_ctrl

        ctrl_size_lists = [list(x.shape) for x in self.actor.optimizer.param_groups[0]['params']]
        dstb_size_lists = [list(x.shape) for x in self.dstb_actor.optimizer.param_groups[0]['params']]
        reshaped_ctrl, reshaped_dstb = [], []
        count = 0
        for i in range(len(ctrl_size_lists)):
            numel = np.prod(ctrl_size_lists[i])
            reshaped_ctrl.append(torch.reshape(ctrl_estimator[count : count + numel], ctrl_size_lists[i]))
            count += numel
        count = 0
        for i in range(len(dstb_size_lists)):
            numel = np.prod(dstb_size_lists[i])
            reshaped_dstb.append(torch.reshape(dstb_estimator[count: count + numel], dstb_size_lists[i]))
            count += numel
        return reshaped_ctrl, reshaped_dstb
    def kaczmarz(self, grad_vec, b):
        tol = .00001
        x = torch.zeros_like(b)
        #P = torch.zeros(torch.numel(b), torch.numel(b), device=self.device)
        eye = torch.eye(torch.numel(b), device=self.device)
        old = 0
        for k in range(torch.numel(b)):
            '''
            a = autograd.grad(grad_vec, self.critic.optimizer.param_groups[0]['params'], eye[k, :], retain_graph=True)
            a = torch.hstack([t.flatten() for t in a])
            d = P.transpose(0,1) @ a
            c1 = torch.linalg.norm(a)
            c2 = torch.linalg.norm(d)
            c3 = (b[k] - torch.dot(x,a)) / ((c1 - c2)*(c1 + c2))
            p = c3*(a-P@(P.transpose(0,1)@a))
            P = torch.cat((P, torch.reshape(p/torch.linalg.norm(p), (torch.numel(b),1))),dim=1)
            x = x + p
            '''
            a = autograd.grad(grad_vec, self.critic.optimizer.param_groups[0]['params'], eye[k, :], retain_graph=True)
            a = torch.hstack([t.flatten() for t in a])
            r_k = b[k] - torch.dot(a,x)

            # Update the solution vector
            row_norm_sq = torch.norm(a)**2
            #old = x.detach().clone()
            x += (r_k / row_norm_sq) * a



        return x

