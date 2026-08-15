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
import json
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR  #TODO: This can be changed to another scheduler.
from torch.optim.lr_scheduler import ExponentialLR
DEBUG_VIDEO = False
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
from br_tracker import RatingStagnationTracker
from common.minimax import solve_matrix_game
from common.justin.vtrace import VTraceReplayBuffer, VTraceValueTrainer
import threading

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
            entropy_collapse_abort: bool = True,
            entropy_collapse_tol: float = 1e-6,
            entropy_collapse_patience: int = 20,
            enum_every: int = 0,
            enum_k: int = 484,
            enum_buffer: int = 8,
            enum_loss_coef: float = 1.0,
            enum_contact_only: bool = False,
            enum_probe: int = 0,
            enum_walk: int = 40,
            enum_probe_frac: float = 1.0,
            vf_coef: float = 1.0,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = 0.1,
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
            use_wandb: bool = False,
            use_stagnation_early_stop: bool = False,
            use_stagnation_velocity_signal: bool = True,
            use_stagnation_entropy_signal: bool = True,
            stagnation_patience: int = 200,
            stagnation_tolerance: float = 1e-4,
            stagnation_rel_tolerance: float = 0.05,
            stagnation_ema_beta: float = 0.99,
            stagnation_eps: float = 1e-8,
            stagnation_eval_games: Optional[int] = None,
            entropy_stagnation_weight: float = 100.0,
            stagnation_lr_factor: float = 0.5,
            stagnation_lr_patience: int = 5,
            stagnation_use_slope_early_stop: bool = False,
            stagnation_slope_window: int = 20,
            stagnation_slope_tolerance: float = 5e-3,
            stagnation_min_slope_checks: int = 10,
            entropy_ratio_only: bool = False,
            ego_side: str = "left",
            vtrace_enabled: bool = False,
            vtrace_replay_capacity: int = 200_000,
            vtrace_seq_len: Optional[int] = None,
            vtrace_batch_size: int = 256,
            vtrace_rho_bar: float = 5.0,
            vtrace_c_bar: float = 1.0,
            vtrace_keep_onpolicy_value: bool = True,
            blend_adversary_heads: bool = True,
            popart: bool = False,
            popart_beta: float = 3e-4,
            minimax_q: bool = False,
            minimax_head: str = "matrix",
            minimax_rank: int = 4,
            minimax_w_init: float = 0.01,
            minimax_embed: str = "",
            minimax_freeze_embed: bool = True,
            minimax_target: str = "returns",
            minimax_bootstrap_kappa: float = 0.0,
            minimax_bootstrap_warmup: int = 0,
            minimax_iters: int = 1024,
            minimax_eta: float = 0.5,
            minimax_stat_every: int = 10,
            minimax_stop_grad: bool = True,
    ):

        self.matchups = [state2matchup(state) for state in state_list] if state_list is not None else None #This needs to happen before the super().__init__
        self.envs_per_matchup = envs_per_matchup
        self.use_wandb = use_wandb
        super().__init__(
            policy,
            env,
            learning_rate=v_learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
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
        # Entropy saturation is an ABSORBING state: a policy at exactly zero
        # entropy has no probability mass to move, so its gradient is zero and it
        # can never recover. p1_clr1e5_winit hit it on the ADVERSARY at 3.77M and
        # spent the next 34M steps as single-agent RL against a frozen bot, with
        # a healthy-looking score curve throughout. Detect and stop, rather than
        # discover it days later. ent_coef/dstb_ent_coef are 0.0 by default here,
        # so nothing else prevents this.
        # COUNTERFACTUAL ACCESS. One transition trains exactly ONE of the 484
        # cells, and that is measurably fatal to the interaction term: fitting
        # the full Q from one cell per state recovers 0.95% of the true
        # interaction subspace -- BELOW the 3.40% a random subspace scores --
        # because a free per-state mean absorbs the lone observation and leaves
        # zero residual for gamma. Enumerating the matrix instead takes the real
        # head from ~5% to ~58%.
        #
        # This is PRIVILEGED information: it needs em.set_state(), which no
        # model-free agent has. It is training-time only (the deployed policy
        # queries nothing), but it changes the method's class, so every branch
        # step is charged and logged as train/enum_env_steps and comparisons
        # must be budget-matched.
        #
        # enum_every=0 disables it before any of the machinery is touched, so
        # the default is bitwise identical to not having it.
        self.enum_every = int(enum_every)
        self.enum_k = int(enum_k)
        self.enum_buffer = int(enum_buffer)
        self.enum_loss_coef = float(enum_loss_coef)
        # Keep only states where the joint action ACTUALLY affects reward. At
        # 6-12% contact in healthy self-play, ~93% of enumerated states have
        # gamma identically zero -- 484 copies of the same number -- and the
        # aux loss averages over them, so most of its gradient votes W=0. That
        # is the measured gating problem, and dropping those states removes it
        # without collecting anything extra.
        self.enum_contact_only = bool(enum_contact_only)
        self.enum_probe = int(enum_probe)
        self.enum_walk = int(enum_walk)
        # What FRACTION of envs to park on contact states. 1.0 = every state is
        # contact, which measured corrW(R) +0.028 -- WORSE than the natural 6.5%
        # (+0.050), with pred_std 1.8x tgt_std: a head trained only where
        # interaction exists learns to see it everywhere, then is scored on a
        # visitation that is ~93% interaction-free. Below 1.0 the unlocked envs
        # stay on ordinary on-policy states, so the buffer holds both.
        self.enum_probe_frac = float(enum_probe_frac)
        self._enum_store = []          # list of (obs np.ndarray, M np.ndarray)
        self._enum_env_steps = 0
        self._enum_next_at = 0
        self._enum_skipped = 0
        self._enum_walked = 0
        self.entropy_collapse_abort = bool(entropy_collapse_abort)
        self.entropy_collapse_tol = float(entropy_collapse_tol)
        self.entropy_collapse_patience = int(entropy_collapse_patience)
        self._entropy_zero_streak = {"ego": 0, "adv": 0}
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
        self.use_mirror = use_mirror
        self.state_list = state_list if state_list is not None else None
        self.vtrace_enabled = bool(vtrace_enabled)
        self.vtrace_replay_capacity = int(vtrace_replay_capacity)
        # Default T = n_steps // 4 but capped: long unrolls make each worker update
        # heavy (B*(T+1) CNN passes) and, with c_bar=1, the trace washes out the far end.
        self.vtrace_seq_len = int(vtrace_seq_len) if vtrace_seq_len is not None else min(max(1, n_steps // 4), 64)
        self.vtrace_batch_size = int(vtrace_batch_size)
        self.vtrace_rho_bar = float(vtrace_rho_bar)
        self.vtrace_c_bar = float(vtrace_c_bar)
        # Hybrid: also run the usual on-policy value update in train_standard on the
        # freshly collected rollout, anchoring V_theta to current data and correcting
        # the drift accumulated from stale replay. Safe because the async worker is
        # parked for the whole train() window (it is the sole value writer otherwise).
        self.vtrace_keep_onpolicy_value = bool(vtrace_keep_onpolicy_value)
        # Opt-in: blend the multi-head adversary update so the shared dstb_net
        # trunk gets one mean-over-heads step per batch instead of N sequential
        # per-head steps (removes the ordering bias where later heads inherit
        # earlier heads' trunk drift). Default False -> unchanged sequential path.
        self.blend_adversary_heads = bool(blend_adversary_heads)
        # Forwarded into policy_kwargs in _setup_model so the policy builds
        # PopArtHead-wrapped value heads. OFF by default -- turning it on changes
        # value_net state_dict keys and no existing .task can be loaded into it.
        self.popart = bool(popart)
        self.popart_beta = float(popart_beta)
        # Minimax-Q joint-action critic. PHASE 0 defaults: the head trains but
        # feeds NOTHING (stop_grad keeps its gradients off the shared vf trunk,
        # so the existing V head and the advantages are bit-identical to a run
        # with the flag off). Only flip stop_grad once the gate has been passed:
        # Q must beat SHUFFLED Q at branch selection, where V does not.
        self.minimax_q = bool(minimax_q)
        # 'matrix' (default) keeps the 484-cell free head. 'factored' selects the
        # ANOVA decomposition; see FactoredMinimaxHead.
        self.minimax_head = str(minimax_head)
        self.minimax_rank = int(minimax_rank)
        self.minimax_w_init = float(minimax_w_init)
        self.minimax_embed = str(minimax_embed)
        self.minimax_freeze_embed = bool(minimax_freeze_embed)
        # 'returns' (default) = option A, regress Q onto the lambda-returns:
        # DATA, never references Q, cannot diverge. 'minimax' = option B,
        # Littman's operator, target = r + gamma*V_mm(s'). Self-referential.
        # See _minimax_q_update.
        self.minimax_target = str(minimax_target)
        # PHASE 1. 0.0 = the head feeds NOTHING (diagnostic, the default and the
        # only mode measured so far). >0 blends V_minimax into the GAE bootstrap
        # and the head starts moving the policy. See _minimax_bootstrap.
        self.minimax_bootstrap_kappa = float(minimax_bootstrap_kappa)
        self.minimax_bootstrap_warmup = int(minimax_bootstrap_warmup)
        self.minimax_iters = int(minimax_iters)
        self.minimax_eta = float(minimax_eta)
        self.minimax_stat_every = int(minimax_stat_every)
        self.minimax_stop_grad = bool(minimax_stop_grad)
        self.vtrace_ego_replay = None
        self.vtrace_adv_replays = None
        self.vtrace_trainer = None
        self.vtrace_policy_lock = None  # serializes rollout vs worker policy forwards
        if _init_setup_model:
            self.env.num_envs = self.n_envs
            self._setup_model()
        self.env_generator_func = env_generator_func
        self.parallel_updater = None
        self.n_global_env = self.n_envs
        self.env.num_envs = self.n_envs
        self.num_workers = num_workers
        self.use_stagnation_early_stop = use_stagnation_early_stop
        self.use_stagnation_velocity_signal = bool(use_stagnation_velocity_signal)
        self.use_stagnation_entropy_signal = bool(use_stagnation_entropy_signal)
        self.stagnation_patience_cfg = int(stagnation_patience)
        self.stagnation_tolerance_cfg = float(stagnation_tolerance)
        self.stagnation_rel_tolerance_cfg = float(stagnation_rel_tolerance)
        self.stagnation_ema_beta_cfg = float(stagnation_ema_beta)
        self.stagnation_eps_cfg = float(stagnation_eps)
        self.stagnation_eval_games_cfg = stagnation_eval_games
        self.entropy_stagnation_weight_cfg = float(entropy_stagnation_weight)
        self.stagnation_lr_factor_cfg = float(stagnation_lr_factor)
        self.stagnation_lr_patience_cfg = int(stagnation_lr_patience)
        self.stagnation_use_slope_early_stop_cfg = bool(stagnation_use_slope_early_stop)
        self.stagnation_slope_window_cfg = int(stagnation_slope_window)
        self.stagnation_slope_tolerance_cfg = float(stagnation_slope_tolerance)
        self.stagnation_min_slope_checks_cfg = int(stagnation_min_slope_checks)
        self.entropy_ratio_only_cfg = bool(entropy_ratio_only)
        self.elo_initial_rating = 1000.0
        self.elo_k_factor = 24.0
        self._init_elo_trackers()
        default_eval_games = max(
            1,
            (int(self.num_adversaries) if self.num_adversaries is not None else 1) * 2,
        )
        stagnation_eval_games = int(
            default_eval_games if self.stagnation_eval_games_cfg is None else self.stagnation_eval_games_cfg
        )
        self.stagnation_tracker = RatingStagnationTracker(
            patience=self.stagnation_patience_cfg,
            tolerance=self.stagnation_tolerance_cfg,
            rel_tolerance=self.stagnation_rel_tolerance_cfg,
            ema_beta=self.stagnation_ema_beta_cfg,
            eps=self.stagnation_eps_cfg,
            eval_games=stagnation_eval_games,
            entropy_weight=self.entropy_stagnation_weight_cfg,
            lr_patience=self.stagnation_lr_patience_cfg,
            use_velocity_signal=self.use_stagnation_velocity_signal,
            use_entropy_signal=self.use_stagnation_entropy_signal,
            use_slope_early_stop=self.stagnation_use_slope_early_stop_cfg,
            slope_window=self.stagnation_slope_window_cfg,
            slope_tolerance=self.stagnation_slope_tolerance_cfg,
            min_slope_checks=self.stagnation_min_slope_checks_cfg,
            entropy_ratio_only=self.entropy_ratio_only_cfg,
            enable_local_entropy_plot=True,
            enable_local_reward_plot=True,
            enable_local_kl_plot=True,
            local_plot_prefix="continue_exploiter",
            local_plot_every_checks=1,
        )
        self.stagnation_tracker.reset(self.elo_adversary_ratings)
        self.ego_side = ego_side
        #Create learning rate schedulers

    def _init_elo_trackers(self) -> None:
        self.elo_ego_rating = float(self.elo_initial_rating)
        n_adv = int(self.num_adversaries) if self.num_adversaries is not None else 0
        self.elo_adversary_ratings = np.full(n_adv, self.elo_initial_rating, dtype=np.float64)
        self.elo_games_played = np.zeros(n_adv, dtype=np.int64)
        self._elo_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "elo_data",
        )
        os.makedirs(self._elo_data_dir, exist_ok=True)

    def _matchup_name_for_adversary(self, adversary_idx: int) -> str:
        if self.matchups is None or self.envs_per_matchup is None:
            return f"adversary_{adversary_idx}"
        state_idx = adversary_idx * self.envs_per_matchup
        if state_idx < 0 or state_idx >= len(self.matchups):
            return f"adversary_{adversary_idx}"
        return str(self.matchups[state_idx])

    def _ego_score_from_terminal(self, info: dict, reward: float, reward_other: float) -> float:
        if info is not None:
            outcome = info.get("outcome")
            if isinstance(outcome, str):
                normalized = outcome.lower()
                if any(k in normalized for k in ["agent", "ego", "left", "player1", "win"]):
                    if "lose" not in normalized and "loss" not in normalized:
                        return 1.0
                if any(k in normalized for k in ["enemy", "adv", "right", "player2", "lose", "loss"]):
                    return 0.0
                if any(k in normalized for k in ["draw", "tie"]):
                    return 0.5
            elif isinstance(outcome, (int, float)):
                if outcome > 0:
                    return 1.0
                if outcome < 0:
                    return 0.0
                return 0.5

            if "agent_hp" in info and "enemy_hp" in info:
                agent_hp = info["agent_hp"]
                enemy_hp = info["enemy_hp"]
                if agent_hp > enemy_hp:
                    return 1.0
                if agent_hp < enemy_hp:
                    return 0.0
                return 0.5

        if reward > reward_other:
            return 1.0
        if reward < reward_other:
            return 0.0
        return 0.5

    def _update_elo_from_rollout_stats(self, rollout_stats: List[dict]) -> int:
        if not rollout_stats or len(self.elo_adversary_ratings) == 0:
            return 0

        total_games = 0

        for adv_idx, stat in enumerate(rollout_stats):
            n_games = int(stat["games"])
            if n_games <= 0:
                continue
            total_games += n_games

            observed_score = (stat["wins"] + 0.5 * stat["draws"]) / n_games
            expected_score = 1.0 / (
                1.0 + 10.0 ** ((self.elo_adversary_ratings[adv_idx] - self.elo_ego_rating) / 400.0)
            )
            delta = self.elo_k_factor * (observed_score - expected_score)
            self.elo_ego_rating += delta
            self.elo_adversary_ratings[adv_idx] -= delta
            self.elo_games_played[adv_idx] += n_games

            matchup_name = self._matchup_name_for_adversary(adv_idx)
            rating_adv_display = int(round(float(self.elo_adversary_ratings[adv_idx])))
            rating_gap_display = float(self.elo_ego_rating - self.elo_adversary_ratings[adv_idx])
            self.logger.record(f"elo/adv/{matchup_name}/score_rollout", float(observed_score))
            self.logger.record(f"elo/adv/{matchup_name}/games_rollout", n_games)
            self.logger.record(f"elo/adv/{matchup_name}/rating_adv", rating_adv_display)
            self.logger.record(
                f"elo/adv/{matchup_name}/rating_gap",
                rating_gap_display,
            )
            if self.use_wandb:
                wandb.log(
                    {
                        f"elo/adv/{matchup_name}/score_rollout": float(observed_score),
                        f"elo/adv/{matchup_name}/games_rollout": n_games,
                        f"elo/adv/{matchup_name}/rating_adv": float(self.elo_adversary_ratings[adv_idx]),
                        f"elo/adv/{matchup_name}/rating_gap": rating_gap_display,
                    }
                )

        self.logger.record("elo/ego/global/rating_ego", int(round(float(self.elo_ego_rating))))
        if self.use_wandb:
            wandb.log({"elo/ego/global/rating_ego": float(self.elo_ego_rating)})

        self._save_elo_snapshot(rollout_stats)

        return total_games

    def _save_elo_snapshot(self, rollout_stats: List[dict]) -> None:
        try:
            timestep = int(getattr(self, "num_timesteps", 0))
            per_matchup = {}
            for adv_idx in range(len(self.elo_adversary_ratings)):
                matchup_name = self._matchup_name_for_adversary(adv_idx)
                stat = rollout_stats[adv_idx] if adv_idx < len(rollout_stats) else {}
                n_games = int(stat.get("games", 0))
                wins = int(stat.get("wins", 0))
                draws = int(stat.get("draws", 0))
                losses = n_games - wins - draws
                per_matchup[matchup_name] = {
                    "rating_adv": float(self.elo_adversary_ratings[adv_idx]),
                    "games_total": int(self.elo_games_played[adv_idx]),
                    "games_rollout": n_games,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                }
            record = {
                "timestep": timestep,
                "rating_ego": float(self.elo_ego_rating),
                "matchups": per_matchup,
                "wall_time": time.time(),
            }
            elo_file = os.path.join(self._elo_data_dir, "elo_history.jsonl")
            with open(elo_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _get_current_rollout_entropy(self, update_ego: bool, update_adversary: bool) -> Optional[float]:
        if update_ego and not update_adversary:
            return getattr(self, "_last_rollout_entropy_ego", None)
        if update_adversary and not update_ego:
            return getattr(self, "_last_rollout_entropy_adv", None)
        return getattr(self, "_last_rollout_policy_entropy", None)

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

        # Rebuild adversary buffers so device matches the model (pickled buffers from old saves
        # may still reference cuda after loading with device=cpu).
        adversary_buffers = []
        if self.num_adversaries is not None:
            for _ in range(self.num_adversaries):
                adversary_buffers.append(
                    buffer_cls(
                        self.n_steps,
                        self.observation_space,
                        self.action_space,
                        device=self.device,
                        gamma=self.gamma,
                        gae_lambda=self.gae_lambda,
                        n_envs=self.envs_per_matchup,
                    )
                )
            self.adversary_buffers = adversary_buffers

        if self.vtrace_enabled:
            from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
            _obs_shape = get_obs_shape(self.observation_space)
            _obs_dtype = self.observation_space.dtype
            _action_dim = get_action_dim(self.action_space)
            _action_dtype = np.float32
            self.vtrace_ego_replay = VTraceReplayBuffer(
                capacity=self.vtrace_replay_capacity,
                n_envs=self.n_envs,
                obs_shape=_obs_shape,
                obs_dtype=_obs_dtype,
                action_dim=_action_dim,
                action_dtype=_action_dtype,
                device=self.device,
            )
            self.vtrace_adv_replays = [
                VTraceReplayBuffer(
                    capacity=self.vtrace_replay_capacity,
                    n_envs=self.envs_per_matchup,
                    obs_shape=_obs_shape,
                    obs_dtype=_obs_dtype,
                    action_dim=_action_dim,
                    action_dtype=_action_dtype,
                    device=self.device,
                    env_index_offset=i * self.envs_per_matchup,
                )
                for i in range(self.num_adversaries)
            ]
            self.vtrace_policy_lock = threading.Lock()

        if hasattr(self, "num_adversaries"):
            self.policy_kwargs['num_adversaries'] = self.num_adversaries
            #self.policy_kwargs['num_env_per_adv'] = self.num_env_per_adv

        self.policy_kwargs['matchups'] = self.matchups
        self.policy_kwargs['envs_per_matchup'] = self.envs_per_matchup
        self.policy_kwargs['popart'] = getattr(self, "popart", False)
        self.policy_kwargs['popart_beta'] = getattr(self, "popart_beta", 3e-4)
        self.policy_kwargs['minimax_q'] = getattr(self, "minimax_q", False)
        self.policy_kwargs['minimax_head'] = getattr(self, "minimax_head", "matrix")
        self.policy_kwargs['minimax_rank'] = getattr(self, "minimax_rank", 4)
        self.policy_kwargs['minimax_w_init'] = getattr(self, "minimax_w_init", 0.01)
        self.policy_kwargs['minimax_embed'] = getattr(self, "minimax_embed", "")
        self.policy_kwargs['minimax_freeze_embed'] = getattr(self, "minimax_freeze_embed", True)
        # minimax_target is a TRAINER-side choice (it selects the loss target),
        # not a policy one, so it deliberately does NOT go into policy_kwargs --
        # adding it there would change the policy signature and break checkpoint
        # loading for no benefit.
        
        # Set features_extractor_class based on whether observation space is an image
        if is_image_space(self.observation_space):
            self.policy_kwargs['features_extractor_class'] = NatureCNN
        else:
            self.policy_kwargs['features_extractor_class'] = FlattenExtractor

        if self.use_mirror:
            self.policy_kwargs['side_dim'] = 1
            self.policy_kwargs['use_mirror'] = True
            print("Mirror mode training enabled: side_dim = %d" % self.policy_kwargs['side_dim'])

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

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, adversary_buffers, n_rollout_steps: int, run_ego_forward:bool, run_adv_forward:bool, use_mirror: bool, zero_ego_action, zero_adv_action, random_ego_action=False,random_adv_action=False) -> bool:
        if self.use_mirror:
            return self.collect_rollouts_mirror(env, callback, rollout_buffer, adversary_buffers, n_rollout_steps, run_ego_forward, run_adv_forward, zero_ego_action, zero_adv_action, random_ego_action, random_adv_action)
        else:
            return self.collect_rollouts_standard(env, callback, rollout_buffer, adversary_buffers, n_rollout_steps, run_ego_forward, run_adv_forward, zero_ego_action, zero_adv_action, random_ego_action, random_adv_action)
    
    def collect_rollouts_standard(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, adversary_buffers, n_rollout_steps: int, run_ego_forward: bool = True, run_adv_forward: bool = True, zero_ego_action=False, zero_adv_action=False, random_ego_action=False, random_adv_action=False) -> bool:
        if not hasattr(self, 'ego_side'):
            raise ValueError("Ego side not set. Please set ego_side when initializing the agent.")
        timenow = time.time()
        video_log = [Image.fromarray(env.render(mode="rgb_array"))]
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        ego_entropy_sum = 0.0
        adv_entropy_sum = 0.0
        entropy_count = 0
        for i in range(self.num_adversaries):
            adversary_buffers[i].reset()
        rollout_terminal_stats = [
            {"wins": 0, "losses": 0, "draws": 0, "games": 0}
            for _ in range(self.num_adversaries)
        ]
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
                # Serialize against the async V-trace worker's policy forward: SB3's
                # shared distribution objects are mutated by proba_distribution(), so a
                # concurrent worker forward would corrupt this sampling (mismatched
                # action counts -> hstack failure downstream). No-op when vtrace is off.
                _fwd_lock = self.vtrace_policy_lock
                if _fwd_lock is not None:
                    _fwd_lock.acquire()
                try:
                    ego_actions, ego_log_probs, adv_actions, adv_log_probs, values, q_values = self.policy(obs_tensor, deterministic=False, ego_forward=run_ego_forward, adv_forward=run_adv_forward, zero_ego_action=zero_ego_action, zero_adv_action=zero_adv_action, random_adv_action=random_adv_action)
                finally:
                    if _fwd_lock is not None:
                        _fwd_lock.release()
                ego_entropy_sum += float((-ego_log_probs.detach()).mean().item())
                adv_entropy_sum += float((-adv_log_probs.detach()).mean().item())
                entropy_count += 1
                # SPAR_DEBUG_LP=1 -- is the adversary's stored log-prob already
                # inconsistent AT WRITE TIME?
                #
                # The adversary's first-minibatch KL (before ANY gradient step,
                # where it is mathematically required to be 0) drifts 0.00000 ->
                # 0.137 and then aborts every update. This recomputes the same
                # quantity the trainer will recompute -- evaluate_adv_actions on
                # the SAME obs and the SAME actions -- microseconds after the
                # collection forward, with nothing modified in between.
                #   discrepancy HERE  -> policy() and evaluate_adv_actions
                #                        disagree; a forward-path bug
                #   discrepancy LATER -> something mutates between collection and
                #                        train(); the distribution objects are the
                #                        suspect (they are mutated in place by
                #                        proba_distribution(), which is why the
                #                        training path guards them with a lock)
                if os.environ.get("SPAR_DEBUG_LP") and self.num_timesteps % 12288 < 512:
                    with th.no_grad():
                        _re_lp, _ = self.policy.evaluate_adv_actions(
                            obs_tensor, adv_actions, buf_num=[0], side_flag=None)
                        _d = (_re_lp - adv_log_probs).abs()
                        _re_e, _ = self.policy.evaluate_ego_actions(
                            obs_tensor, ego_actions, side_flag=None)
                        _de = (_re_e - ego_log_probs).abs()
                        print(f"[LPDBG] t={self.num_timesteps} "
                              f"adv |re-lp - stored| mean={_d.mean().item():.6f} "
                              f"max={_d.max().item():.6f} "
                              f"| ego mean={_de.mean().item():.6f} "
                              f"max={_de.max().item():.6f}", flush=True)
                if th.any(adv_actions):
                    #print("Adv actions are not all zeros")
                    pass
                other_values = -values

            actions = ego_actions.cpu().numpy()
            actions_other = adv_actions.cpu().numpy()
            left_actions = actions if self.ego_side == "left" else actions_other
            right_actions = actions_other if self.ego_side == "left" else actions
            # Rescale and perform action
            clipped_actions = np.hstack([left_actions, right_actions])
            # print(clipped_actions, flush=True)
            # print(np.shape(clipped_actions),flush=True)
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, _BoxTypes):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low,
                                          self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)
            video_log.append(Image.fromarray(env.render(mode="rgb_array")))

            if self.ego_side == 'right':
                rewards = rewards_other
                rewards_other = -rewards
            
            #env.render(mode="rgb_array")
            #np.random.seed(0)
            #random.seed(0)
            #torch.manual_seed(0)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            if self.ego_side == 'right':
                for idx in range(len(infos)):
                    if "episode" in infos[idx]:
                        infos[idx]["episode"]["r"], infos[idx]["episode"]["ro"] = infos[idx]["episode"]["ro"], infos[idx]["episode"]["r"]
                    if "outcome" in infos[idx]:
                        o = infos[idx]["outcome"]
                        infos[idx]["outcome"] = "lose" if o == "win" else ("win" if o == "lose" else o)

            self._update_info_buffer(infos)
            n_steps += 1

            for idx, done in enumerate(dones):
                if not done:
                    continue
                adv_idx = idx // self.n_env_per_adv
                if adv_idx < 0 or adv_idx >= self.num_adversaries:
                    continue
                ego_score = self._ego_score_from_terminal(
                    infos[idx],
                    float(rewards[idx]),
                    float(rewards_other[idx]),
                )
                if ego_score >= 1.0:
                    rollout_terminal_stats[adv_idx]["wins"] += 1
                elif ego_score <= 0.0:
                    rollout_terminal_stats[adv_idx]["losses"] += 1
                else:
                    rollout_terminal_stats[adv_idx]["draws"] += 1
                rollout_terminal_stats[adv_idx]["games"] += 1

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
            if self.vtrace_enabled and self.vtrace_ego_replay is not None:
                _ego_log_probs_np = ego_log_probs.detach().cpu().numpy()
                _adv_log_probs_np = adv_log_probs.detach().cpu().numpy()
                self.vtrace_ego_replay.add(
                    obs=self._last_obs,
                    action=actions,
                    reward=rewards,
                    done=dones,
                    mu_log_prob=_ego_log_probs_np,
                )
                for i in range(self.num_adversaries):
                    indices = slice(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
                    self.vtrace_adv_replays[i].add(
                        obs=self._last_obs[indices],
                        action=actions_other[indices],
                        # Adversary-perspective reward (= -ego reward), matching the
                        # on-policy adversary_buffers.add() convention. The worker
                        # negates the value head to pair with this.
                        reward=rewards_other[indices],
                        done=dones[indices],
                        mu_log_prob=_adv_log_probs_np[indices],
                    )
            #for i in range(self.num_adversaries):
            #    adversary_buffers[i].add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts, values_other,
            #                             adv_log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.value_forward(obs_as_tensor(new_obs, self.device))
            #values_other = rollout_policy_other.predict_values(obs_as_tensor(new_obs, self.device))

        # PHASE 1: swap the bootstrap to V_minimax. No-op (and no solve) at
        # kappa 0, which is the default -- see _minimax_bootstrap.
        values = self._minimax_bootstrap(rollout_buffer, adversary_buffers,
                                         new_obs, values)
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        for i in range(self.num_adversaries):
            adversary_buffers[i].compute_returns_and_advantage(last_values=-values[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv], dones=dones[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv])
        if entropy_count > 0:
            self._last_rollout_entropy_ego = ego_entropy_sum / entropy_count
            self._last_rollout_entropy_adv = adv_entropy_sum / entropy_count
            if run_ego_forward and not run_adv_forward:
                self._last_rollout_policy_entropy = self._last_rollout_entropy_ego
            elif run_adv_forward and not run_ego_forward:
                self._last_rollout_policy_entropy = self._last_rollout_entropy_adv
            else:
                self._last_rollout_policy_entropy = (
                    self._last_rollout_entropy_ego + self._last_rollout_entropy_adv
                ) / 2.0
        else:
            self._last_rollout_policy_entropy = None
        #if self.update_right:
        #    rollout_buffer_other.compute_returns_and_advantage(last_values=values_other, dones=dones)

        callback.on_rollout_end()

        # Fingerprint the adversary policy at the END of collection, every
        # rollout. Sampling it inside the LPDBG window instead made
        # 'moved' report ordinary cross-iteration drift rather than the
        # collection-to-training gap that is actually in question.
        if os.environ.get("SPAR_DEBUG_LP"):
            self._dbg_adv_sig = float(sum(
                pp.detach().double().sum().item()
                for pp in list(self.policy.mlp_extractor.dstb_net.parameters())
                + list(self.policy.dstb_action_net.parameters())))
        rollout_buffer.prepare_data_for_training()
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].prepare_data_for_training()
        n_games = self._update_elo_from_rollout_stats(rollout_terminal_stats)
        if getattr(self, "stagnation_tracker", None) is not None:
            self.stagnation_tracker.register_games(n_games)

        return True
    
    def collect_rollouts_mirror(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, adversary_buffers, n_rollout_steps: int, run_ego_forward: bool = True, run_adv_forward: bool = True, zero_ego_action=False, zero_adv_action=False, random_ego_action=False,random_adv_action=False) -> bool:
        assert self.use_mirror is True, "Use mirror is not True"
        if self.policy.use_mirror is False:
            self.policy.use_mirror = True
            print("Mirror mode training enabled: policy use_mirror set to %d" % self.policy.use_mirror)
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        ego_entropy_sum = 0.0
        adv_entropy_sum = 0.0
        entropy_count = 0
        for i in range(self.num_adversaries):
            adversary_buffers[i].reset()
        rollout_terminal_stats = [
            {"wins": 0, "losses": 0, "draws": 0, "games": 0}
            for _ in range(self.num_adversaries)
        ]

        video_log = [Image.fromarray(env.render(mode="rgb_array"))]
        callback.on_rollout_start()

        halfway = env.num_envs // 2
        if not hasattr(self, '_ego_side_flags') or self._ego_side_flags.shape[0] != env.num_envs:
            self._ego_side_flags = np.zeros((env.num_envs, 1), dtype=np.float32)
            for idx in range(halfway, env.num_envs):
                self._ego_side_flags[idx, 0] = 1.0
        ego_side_flags = self._ego_side_flags

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                ego_side_tensor = th.tensor(ego_side_flags, device=self.device)
                adv_side_tensor = 1.0 - ego_side_tensor
                ego_actions, ego_log_probs, adv_actions, adv_log_probs, _, _ = self.policy(obs_tensor, deterministic=False, ego_forward=run_ego_forward, adv_forward=run_adv_forward, zero_ego_action=zero_ego_action, zero_adv_action=zero_adv_action, random_ego_action=random_ego_action, random_adv_action=random_adv_action, value_forward=False, ego_side_flag=ego_side_tensor, adv_side_flag=adv_side_tensor)
                values = self.policy.value_forward(obs_tensor, side_flag=ego_side_tensor)
                other_values = -values

            # ego_actions/adv_actions are N each (per-env conditioned)
            # Top half: ego=P1(left), adv=P2(right). Bottom half: adv=P1(left), ego=P2(right).
            actions = ego_actions.cpu().numpy()
            other_actions = adv_actions.cpu().numpy()
            halfway = actions.shape[0] // 2
            left_actions = np.concatenate([actions[:halfway], other_actions[halfway:]])
            right_actions = np.concatenate([other_actions[:halfway], actions[halfway:]])
            clipped_actions = np.hstack([left_actions, right_actions])
            if isinstance(self.action_space, _BoxTypes):
                clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)
            #rewards = np.ones_like(rewards) * 9
            #rewards_other = np.ones_like(rewards_other) * -9
            video_log.append(Image.fromarray(env.render(mode="rgb_array")))
            # Bottom half: ego is on right, adv is on left. Swap so rewards=ego, rewards_other=adv.
            temp_rew = rewards[halfway:].copy()
            rewards[halfway:] = rewards_other[halfway:]
            rewards_other[halfway:] = temp_rew

            corrected_ego_log_probs = ego_log_probs
            corrected_adv_log_probs = adv_log_probs
            corrected_ego_actions = actions
            corrected_adv_actions = other_actions
            #rewards, rewards_other = mirror_flip_attributes(rewards, rewards_other)
            #np.random.seed(0)
            #random.seed(0)
            #torch.manual_seed(0)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            # Swap episode reward and outcome for bottom-half envs so logger/ELO reports ego perspective
            for idx in range(halfway, len(infos)):
                if "episode" in infos[idx]:
                    infos[idx]["episode"]["r"], infos[idx]["episode"]["ro"] = infos[idx]["episode"]["ro"], infos[idx]["episode"]["r"]
                if "outcome" in infos[idx]:
                    o = infos[idx]["outcome"]
                    infos[idx]["outcome"] = "lose" if o == "win" else ("win" if o == "lose" else o)

            self._update_info_buffer(infos)
            n_steps += 1

            for idx, done in enumerate(dones):
                if not done:
                    continue
                adv_idx = idx // self.n_env_per_adv
                if adv_idx < 0 or adv_idx >= self.num_adversaries:
                    continue
                ego_score = self._ego_score_from_terminal(
                    infos[idx],
                    float(rewards[idx]),
                    float(rewards_other[idx]),
                )
                if ego_score >= 1.0:
                    rollout_terminal_stats[adv_idx]["wins"] += 1
                elif ego_score <= 0.0:
                    rollout_terminal_stats[adv_idx]["losses"] += 1
                else:
                    rollout_terminal_stats[adv_idx]["draws"] += 1
                rollout_terminal_stats[adv_idx]["games"] += 1

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
            rollout_buffer.add(self._last_obs.copy(), corrected_ego_actions, corrected_adv_actions, rewards, new_obs, dones, self._last_episode_starts, values,
                                   corrected_ego_log_probs, values, side_flags=ego_side_flags)
            adv_side_flags = 1.0 - ego_side_flags
            for i in range(self.num_adversaries):
                indices = slice(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
                adversary_buffers[i].add(self._last_obs[indices].copy(), corrected_ego_actions[indices], corrected_adv_actions[indices], -rewards[indices], new_obs[indices], dones[indices], self._last_episode_starts[indices], other_values[indices],
                                         corrected_adv_log_probs[indices], other_values[indices], side_flags=adv_side_flags[indices])

            for idx in range(env.num_envs):
                agent_x = infos[idx].get('agent_x', 0)
                enemy_x = infos[idx].get('enemy_x', 0)
                if idx < halfway:
                    ego_side_flags[idx, 0] = 0.0 if agent_x <= enemy_x else 1.0
                else:
                    ego_side_flags[idx, 0] = 0.0 if enemy_x <= agent_x else 1.0

            debug_dir = getattr(self, 'debug_frame_dir', None)
            if DEBUG_VIDEO and debug_dir is not None:
                import os
                os.makedirs(debug_dir, exist_ok=True)
                flags_str = ''.join([str(int(ego_side_flags[i, 0])) for i in range(env.num_envs)])
                positions = []
                for idx in range(env.num_envs):
                    ax = infos[idx].get('agent_x', 0)
                    ex = infos[idx].get('enemy_x', 0)
                    positions.append(f"e{idx}_ax{ax}_ex{ex}")
                pos_str = '_'.join(positions)
                frame = Image.fromarray(env.render(mode="rgb_array"))
                frame.save(os.path.join(debug_dir, f"step{n_steps:04d}_flags{flags_str}_{pos_str}.png"))

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute ego values for the last timestep, negate for adversary (zero-sum)
            last_obs_tensor = obs_as_tensor(new_obs, self.device)
            last_ego_side_tensor = th.tensor(ego_side_flags, device=self.device)
            last_adv_side_tensor = 1 - last_ego_side_tensor
            last_ego_values = self.policy.value_forward(last_obs_tensor, side_flag=last_ego_side_tensor)
            last_adv_values = -last_ego_values

        # PHASE 1, mirror path. side_flag is passed here because the mirror
        # encoder needs it; omitting it would silently evaluate the wrong seat.
        last_ego_values = self._minimax_bootstrap(
            rollout_buffer, adversary_buffers, new_obs, last_ego_values,
            side_flag=last_ego_side_tensor)
        last_adv_values = -last_ego_values
        rollout_buffer.compute_returns_and_advantage(last_values=last_ego_values, dones=dones)
        for i in range(self.num_adversaries):
            adversary_buffers[i].compute_returns_and_advantage(last_values=last_adv_values[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv], dones=dones[i * self.n_env_per_adv : (i + 1) * self.n_env_per_adv])
        if entropy_count > 0:
            self._last_rollout_entropy_ego = ego_entropy_sum / entropy_count
            self._last_rollout_entropy_adv = adv_entropy_sum / entropy_count
            if update_ego and not update_adversary:
                self._last_rollout_policy_entropy = self._last_rollout_entropy_ego
            elif update_adversary and not update_ego:
                self._last_rollout_policy_entropy = self._last_rollout_entropy_adv
            else:
                self._last_rollout_policy_entropy = (
                    self._last_rollout_entropy_ego + self._last_rollout_entropy_adv
                ) / 2.0
        else:
            self._last_rollout_policy_entropy = None
        #if self.update_right:
        #    rollout_buffer_other.compute_returns_and_advantage(last_values=values_other, dones=dones)

        callback.on_rollout_end()

        rollout_buffer.prepare_data_for_training()
        for i in range(len(adversary_buffers)):
            adversary_buffers[i].prepare_data_for_training()
        n_games = self._update_elo_from_rollout_stats(rollout_terminal_stats)
        if getattr(self, "stagnation_tracker", None) is not None:
            self.stagnation_tracker.register_games(n_games)

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
        random_ego_action: bool = False,
        random_adv_action: bool = False,
        num_perturbs: int = 1,
    ):
        try:
            iteration = 0
            use_elo_tracker = (
                bool(getattr(self, "use_stagnation_velocity_signal", True))
                or bool(getattr(self, "use_stagnation_entropy_signal", True))
            )
            if use_elo_tracker and getattr(self, "stagnation_tracker", None) is not None:
                self.stagnation_tracker.use_velocity_signal = bool(self.use_stagnation_velocity_signal)
                self.stagnation_tracker.use_entropy_signal = bool(self.use_stagnation_entropy_signal)
                self.stagnation_tracker.enable_local_entropy_plot = bool(getattr(self, "training_br", False))
                self.stagnation_tracker.enable_local_reward_plot = bool(getattr(self, "training_br", False))
                base_plot_prefix = (
                    "continue_exploiter_ego"
                    if update_ego and not update_adversary
                    else "continue_exploiter_adv"
                    if update_adversary and not update_ego
                    else "continue_exploiter_joint"
                )
                stop_key = getattr(self, "br_manual_stop_key", None)
                ckpt_tag = getattr(self, "_checkpoint_basename", None)
                if stop_key is not None and str(stop_key) != "":
                    prefix = f"{base_plot_prefix}_{str(stop_key)}"
                else:
                    prefix = base_plot_prefix
                if ckpt_tag is not None:
                    prefix = f"{prefix}_{ckpt_tag}"
                self.stagnation_tracker.local_plot_prefix = prefix
                self.stagnation_tracker.reset(self.elo_adversary_ratings)
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

            if self.vtrace_enabled and self.vtrace_ego_replay is not None and self.vtrace_trainer is None:
                self.vtrace_trainer = VTraceValueTrainer(
                    policy=self.policy,
                    value_optimizer=self.policy.value_optimizer,
                    ego_replay=self.vtrace_ego_replay,
                    adv_replays=self.vtrace_adv_replays,
                    num_adversaries=int(self.num_adversaries),
                    is_discrete_action=isinstance(self.action_space, _DiscreteTypes),
                    gamma=float(self.gamma),
                    rho_bar=self.vtrace_rho_bar,
                    c_bar=self.vtrace_c_bar,
                    seq_len=self.vtrace_seq_len,
                    batch_size=self.vtrace_batch_size,
                    max_grad_norm=float(self.max_grad_norm),
                    device=self.device,
                    warmup_transitions=self.vtrace_seq_len + 1,
                    policy_lock=self.vtrace_policy_lock,
                )
                self.vtrace_trainer.start()
                if self.verbose >= 1:
                    print(
                        f"[V-trace] worker started (T={self.vtrace_seq_len}, "
                        f"B={self.vtrace_batch_size}, capacity={self.vtrace_replay_capacity}, "
                        f"rho_bar={self.vtrace_rho_bar}, c_bar={self.vtrace_c_bar})",
                        flush=True,
                    )

            callback.on_training_start(locals(), globals())

            while self.num_timesteps < total_timesteps:
                if USE_PERTURBED:
                    self._create_all_perturbed_agents(num_perturbs)
                    self._initialize_parallel_updater()                

                if USE_PERTURBED:
                    with ThreadPoolExecutor(max_workers=num_perturbs + 1) as executor:
                        futures = [executor.submit(perturbed_agent.env_perturb_params, run_ego_forward, run_adv_forward, zero_ego_action, zero_adv_action, random_adv_action) for perturbed_agent in self.perturbed_agents]
                        perturbed_bufs, perturbed_adv_bufs = zip(*[future.result() for future in futures])
                        future_standard = executor.submit(self.collect_rollouts, self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps, run_ego_forward, run_adv_forward, self.use_mirror, zero_ego_action, zero_adv_action, random_adv_action)
                        continue_training = future_standard.result()
                    self.perturbed_bufs = list(perturbed_bufs)
                    self.perturbed_adv_bufs = list(perturbed_adv_bufs)

                    self.perturbed_agents_policy = [perturbed_agent.policy for perturbed_agent in self.perturbed_agents]
                else:
                    continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.adversary_buffers, self.n_steps, run_ego_forward, run_adv_forward, self.use_mirror, zero_ego_action, zero_adv_action, random_ego_action, random_adv_action)


                if continue_training is False:
                    break

                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

                # Display training infos
                if log_interval is not None and iteration % log_interval == 0:
                    time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                    fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                    self.logger.record("train/time/iterations", iteration, exclude="tensorboard")
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        rews.append(safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("train/rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                        if self.use_wandb:
                            wandb.log({"train/eval_rew": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])})
                        self.logger.record("train/rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("train/time/fps", fps)
                    self.logger.record("train/time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                    self.logger.record("train/time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                    self.logger.record("train/lr/ego", float(self.policy.ctrl_optimizer.param_groups[0]["lr"]))
                    self.logger.record("train/lr/adversary", float(self.policy.dstb_optimizer.param_groups[0]["lr"]))
                    self.logger.record("train/lr/critic", float(self.policy.value_optimizer.param_groups[0]["lr"]))
                    if hasattr(self.policy, "ego_value_optimizer"):
                        self.logger.record(
                            "train/lr/ego_critic",
                            float(self.policy.ego_value_optimizer.param_groups[0]["lr"]),
                        )
                    if self.vtrace_trainer is not None:
                        _vt_metrics = self.vtrace_trainer.drain_metrics()
                        if _vt_metrics:
                            _agg: Dict[str, List[float]] = {}
                            for _m in _vt_metrics:
                                for _k, _v in _m.items():
                                    _agg.setdefault(_k, []).append(_v)
                            for _k, _vs in _agg.items():
                                self.logger.record(f"train/{_k}", float(np.mean(_vs)))
                            self.logger.record("train/vtrace_updates_in_window", len(_vt_metrics))
                        self.logger.record("train/vtrace_updates_total", int(self.vtrace_trainer.updates_count))
                        self.logger.record("train/vtrace_ego_replay_size", int(self.vtrace_ego_replay.num_steps()))
                        if self.vtrace_adv_replays:
                            self.logger.record(
                                "train/vtrace_adv_replay_size_mean",
                                float(np.mean([b.num_steps() for b in self.vtrace_adv_replays])),
                            )
                    self.logger.dump(step=self.num_timesteps)
                if use_elo_tracker:
                    current_reward = None
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        current_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    current_entropy = self._get_current_rollout_entropy(update_ego, update_adversary)
                    def _lr_adjustment_callback() -> None:
                        if not self.use_lr_annealing:
                            return
                        lr_factor = float(self.stagnation_lr_factor_cfg)
                        if lr_factor <= 0.0 or lr_factor >= 1.0:
                            return
                        gamma = float(self.lr_anneal_coeff)
                        if gamma <= 0.0 or gamma >= 1.0:
                            extra_steps = 1
                        else:
                            extra_steps = int(np.ceil(np.log(lr_factor) / np.log(gamma)))
                            extra_steps = max(1, extra_steps)
                        for _ in range(extra_steps):
                            self._update_schedulers(
                                step_ego=update_ego,
                                step_adv=update_adversary,
                                step_val=True,
                                skip=not self.use_lr_annealing,
                            )

                    # train/{ego,adv}_approx_kl is snapshotted into
                    # self._last_train_approx_kl inside CDS.train() BEFORE the
                    # outer learn() calls logger.dump() (which would clear
                    # name_to_value). On the very first check (iter 1, no train
                    # has run yet) the attr is unset -> None -> tracker skips
                    # KL recording. From iter 2 onward each check gets the prev
                    # train's value.
                    _approx_kl = getattr(self, "_last_train_approx_kl", None)
                    stagnation_triggered, stagnation_logs = self.stagnation_tracker.check(
                        ratings=self.elo_adversary_ratings,
                        current_entropy=current_entropy,
                        lr_adjustment_callback=_lr_adjustment_callback,
                        timestep=float(self.num_timesteps),
                        current_reward=float(current_reward) if current_reward is not None else None,
                        current_kl=float(_approx_kl) if _approx_kl is not None else None,
                    )
                    if stagnation_logs is not None:
                        for key, value in stagnation_logs.items():
                            self.logger.record(key, value)
                        if self.use_wandb:
                            wandb.log(stagnation_logs)
                    if stagnation_triggered and self.use_stagnation_early_stop:
                        print("Elo stagnation tracker triggered early stopping.")
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                #self.perturbed_agent.env.close()
                #del self.perturbed_agent

            callback.on_training_end()
        
        except Exception as e:
            print(e)
            import traceback as _tb
            _tb.print_exc()

        finally:
            if self.vtrace_trainer is not None:
                try:
                    self.vtrace_trainer.stop()
                except Exception as _exc:
                    print(f"[V-trace] worker stop error: {_exc}", flush=True)
                self.vtrace_trainer = None
            #IMPORTANT! Persistent workers must be cleaned up.
            self.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self
    
    def train(self, update_ego: bool = True, update_adversary: bool = True) -> None:
        # Park the async V-trace value worker for the duration of the on-policy
        # ego/adv update so it does not write the value head while this thread
        # reads params / steps optimizers. Resumed in finally.
        _vt = getattr(self, "vtrace_trainer", None)
        if _vt is not None:
            _vt.pause()
        # BEFORE the updates and AFTER the rollout: the branch steps are rewound,
        # so nothing the policy update reads is contaminated, and the freshly
        # enumerated matrices are available to every minibatch this iteration.
        # Also inside the vtrace pause, so the value worker cannot write the
        # value head while enumeration reads V(s') for the targets.
        self._maybe_enumerate()
        try:
            if update_ego:
                self.train_standard(update_ego=True, update_adversary=False)
                pass
            if update_adversary:
                self.train_standard(update_ego=False, update_adversary=True)
            #self.train_standard(update_ego, update_adversary)
            if USE_PERTURBED:
                self.train_derivative_free(update_ego, update_adversary)
        finally:
            if _vt is not None:
                _vt.resume()
        # Snapshot approx_kl BEFORE the outer learn() loop calls Logger.dump(),
        # which would clear name_to_value before the next collect_rollouts ->
        # stagnation_tracker.check() can read it. CDS uses prefixed keys
        # (train/ego_approx_kl, train/adv_approx_kl) -- see _log_leader_metrics
        # at ~line 1730. Pick the side that was actually updated this step;
        # if both updated, take the mean.
        ego_kl = self.logger.name_to_value.get("train/ego_approx_kl") if update_ego else None
        adv_kl = self.logger.name_to_value.get("train/adv_approx_kl") if update_adversary else None
        vals = [v for v in (ego_kl, adv_kl) if v is not None]
        if vals:
            try:
                self._last_train_approx_kl = float(sum(float(v) for v in vals) / len(vals))
            except (TypeError, ValueError):
                self._last_train_approx_kl = None
    
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
                        if self.use_mirror:
                            df_side_flag = ori_buf.side_flags if ego else ori_buf[i].side_flags
                        else:
                            df_side_flag = None
                        log_prob, _ = self.policy.evaluate_ego_actions(ori_buf.observations, ori_buf.actions, side_flag=df_side_flag) if ego else self.policy.evaluate_adv_actions(ori_buf[i].observations, ori_buf[i].adv_actions, buf_num=[i], side_flag=df_side_flag)
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

    def _train_adversary_blended(self, clip_range) -> None:
        """Blended multi-head adversary update (opt-in via blend_adversary_heads).

        The sequential path trains heads one-by-one, and each head's step moves
        the SHARED dstb_net trunk, so later heads inherit earlier heads' drift.
        Here we interleave all heads into every optimizer step: forward each head
        on aligned minibatches, accumulate their gradients, then average ONLY the
        shared trunk gradient over the N heads before a single dstb_optimizer
        step. Per-head action nets keep full-rate grads (a head's forward touches
        only its own action net -> no cross-head conflict). The KL early-stop uses
        the mean approx-KL across heads, still measured vs the rollout baseline so
        the trust region is preserved.

        Loss math mirrors the sequential update_adversary path exactly; only the
        step structure differs.

        NOTE: the adversary's OWN feature extractor (pi_dstb_features_extractor)
        lives in dstb_optimizer alongside dstb_net, so trunk_params already
        includes it -- the /N averages the adversary's full shared stack (CNN +
        MLP trunk), not just the MLP. The value function's separate CNN
        (vf_features_extractor) is not an adversary-shared layer and is not
        blended here. When do_onpolicy_value is True the value optimizer is
        stepped in this same update (its grads summed over heads, not /N); those
        params are clipped via the do_onpolicy_value branch below, matching the
        sequential path (the async V-trace worker is parked during train(), so
        this thread owns the value grads and may clip them).
        """
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        n_adv = int(self.num_adversaries)
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        approx_kl_divs = []
        do_onpolicy_value = (not self.vtrace_enabled) or self.vtrace_keep_onpolicy_value

        # Shared adversary trunk = dstb_optimizer params minus the per-head
        # action nets (dstb_action_net). Those trunk params get the mean grad
        # over heads; the per-head action nets keep full-rate grads.
        try:
            _action_ids = {id(p) for p in self.policy.dstb_action_net.parameters()}
        except AttributeError:
            _action_ids = set()
        trunk_params = [
            p
            for group in self.policy.dstb_optimizer.param_groups
            for p in group["params"]
            if id(p) not in _action_ids
        ]

        stop = False
        for epoch in range(self.n_epochs):
            # One minibatch generator per head; zip -> an aligned tuple per step.
            gens = [self.adversary_buffers[i].get(self.batch_size) for i in range(n_adv)]
            for batch_tuple in zip(*gens):
                self.policy.ctrl_optimizer.zero_grad()
                self.policy.dstb_optimizer.zero_grad()
                if do_onpolicy_value:
                    self.policy.value_optimizer.zero_grad()
                    if hasattr(self.policy, "ego_value_optimizer"):
                        self.policy.ego_value_optimizer.zero_grad()

                head_kls = []
                for i, rollout_data in enumerate(batch_tuple):
                    actions = rollout_data.adv_actions
                    if isinstance(self.action_space, _DiscreteTypes):
                        actions = rollout_data.actions.long().flatten()
                    side_flag = rollout_data.side_flags if self.use_mirror else None

                    log_prob, entropy = self.policy.evaluate_adv_actions(
                        rollout_data.observations, actions, buf_num=[i], side_flag=side_flag)
                    # PopArt stats MUST be updated BEFORE this forward. update_stats
                    # mutates sigma and the final Linear's weights IN PLACE, and both
                    # are saved by autograd for the backward of `values` -- doing it
                    # after the forward raises "variable needed for gradient
                    # computation has been modified by an inplace operation".
                    # This block is the ADVERSARY frame (values negated just below),
                    # while the head's mu lives in the EGO frame, hence -returns.
                    _pa = self.policy.popart_for([i])
                    if _pa is not None:
                        _pa.update_stats(-rollout_data.returns)
                    values = self.policy.evaluate_states(
                        rollout_data.observations, env_indices=rollout_data.env_indices,
                        buf_num=[i], side_flag=side_flag)
                    values = (-values).flatten()

                    advantages = rollout_data.advantages
                    if len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    pg_losses.append(policy_loss.item())
                    clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                    if clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                    # PopArt: `values` was negated above, so this block is in the
                    # ADVERSARY frame; the head's mu lives in the EGO frame, hence
                    # the -returns when updating statistics. Only sigma enters the
                    # loss -- mu cancels in a difference of normalized quantities:
                    #   normalize(a) - normalize(b) == (a - b) / sigma
                    if _pa is not None:
                        _s = _pa.sigma.detach()
                        value_loss = F.mse_loss(rollout_data.returns / _s, values_pred / _s)
                    else:
                        value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    entropy_losses.append(entropy_loss.item())

                    if do_onpolicy_value:
                        loss = policy_loss + self.dstb_ent_coef * entropy_loss + self.vf_coef * value_loss
                    else:
                        loss = policy_loss + self.dstb_ent_coef * entropy_loss

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                    head_kls.append(approx_kl_div)

                    loss.backward()  # accumulate grads across heads

                # Average ONLY the shared trunk gradient over the heads.
                for p in trunk_params:
                    if p.grad is not None:
                        p.grad /= n_adv

                # Clip parity with the sequential path. When the on-policy value
                # update runs here the async V-trace worker is parked, so this
                # thread owns the value grads -> clip ALL params (incl value_net +
                # its CNN), which are stepped below. In pure-offload
                # (do_onpolicy_value False) the worker writes value grads on
                # another thread, so clip only ctrl+dstb to avoid racing them.
                if do_onpolicy_value:
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                else:
                    _clip_params = [
                        p
                        for opt in (self.policy.ctrl_optimizer, self.policy.dstb_optimizer)
                        for group in opt.param_groups
                        for p in group["params"]
                    ]
                    th.nn.utils.clip_grad_norm_(_clip_params, self.max_grad_norm)

                mean_kl = float(np.mean(head_kls)) if head_kls else 0.0
                if self.target_kl is not None and mean_kl > 1.5 * self.target_kl:
                    if self.verbose >= 1:
                        print(f"blended adversary training stopped (mean kl {mean_kl:.3f})")
                    stop = True
                    break

                self.policy.dstb_optimizer.step()
                if do_onpolicy_value:
                    self.policy.value_optimizer.step()
                # Blended path is adversary-only by construction, so the buffer
                # is always an adversary buffer and the returns are ADV frame.
                self._minimax_q_update(rollout_data, [i], adv_frame=True)
            if stop:
                break

        self._n_updates += self.n_epochs
        buf = self.adversary_buffers[0]
        if th.is_tensor(buf.values):
            explained_var = explained_variance(
                buf.values.flatten().cpu().numpy(), buf.returns.flatten().cpu().numpy())
        else:
            explained_var = explained_variance(buf.values, buf.returns)
        self.policy.num_adversaries = self.num_adversaries
        self.logger.record("train/value_loss", np.mean(value_losses) if value_losses else 0.0)
        # Minimax-Q phase 0. Absence of these lines means the update never RAN
        # -- which a clean exit does not rule out, since the call site sits in
        # the blended-adversary branch.
        _mm = getattr(self, "_minimax_stats", None)
        if _mm:
            for _k, _v in _mm.items():
                self.logger.record(f"train/minimax_{_k}", _v)
        self._log_leader_metrics(False, entropy_losses, pg_losses, approx_kl_divs, explained_var, clip_range)

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
        # Opt-in blended multi-head adversary update (default off -> the
        # sequential per-head path below is unchanged). Only meaningful with
        # more than one adversary head.
        if (update_adversary and getattr(self, "blend_adversary_heads", False)
                and int(self.num_adversaries) > 1):
            self._train_adversary_blended(clip_range)
            return
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
                    stop_policy_training = False
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
                    
                    if self.use_mirror:
                        side_flag = rollout_data.side_flags
                    else:
                        side_flag = None

                    if update_ego:
                        log_prob, entropy = self.policy.evaluate_ego_actions(rollout_data.observations, actions, side_flag=side_flag)
                        #from stable_baselines3.common.save_util import load_from_zip_file
                        #data, params, pytorch_variables = load_from_zip_file("test_ego_save.pth", device=self.device)
                        #entropy = ego_entropy
                    if update_adversary:
                        log_prob, entropy = self.policy.evaluate_adv_actions(rollout_data.observations, actions, buf_num=[i], side_flag=side_flag)
                        #entropy = adv_entropy
                    # PopArt stats MUST be updated BEFORE the forward -- update_stats
                    # mutates sigma and the final Linear in place, and autograd saves
                    # both for the backward of `values`. `values` is negated below
                    # only when update_adversary, so the EGO-frame target (which is
                    # what mu tracks) is -returns in that case and +returns otherwise.
                    _pa_buf = [i] if update_adversary else [_j for _j in range(self.num_adversaries)]
                    _pa = self.policy.popart_for(_pa_buf)
                    if _pa is not None:
                        _pa.update_stats(-rollout_data.returns if update_adversary
                                         else rollout_data.returns)
                    if update_ego:
                        values = self.policy.evaluate_states(rollout_data.observations, env_indices=rollout_data.env_indices, buf_num=[i for i in range(self.num_adversaries)], side_flag= side_flag)
                    else:
                        values = self.policy.evaluate_states(rollout_data.observations, env_indices=rollout_data.env_indices, buf_num=[i], side_flag=side_flag)
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
                    # SPAR_DEBUG_KL=1 -- diagnose epoch-0 KL. At epoch 0 the policy
                    # has not been updated, so log_prob MUST equal old_log_prob and
                    # approx_kl MUST be 0. The masked-RAM arm reports 0.12-0.15 and
                    # aborts every ego and adv pass before taking a single gradient
                    # step. `ratio.mean()` cannot see this: approx_kl reduces to
                    # -mean(log_ratio) whenever mean(ratio)==1, so a symmetric
                    # spread of ratios looks perfect and is not.
                    if epoch == 0 and os.environ.get("SPAR_DEBUG_LP") and not update_ego:
                        with th.no_grad():
                            _sig = float(sum(
                                pp.detach().double().sum().item()
                                for pp in list(self.policy.mlp_extractor.dstb_net.parameters())
                                + list(self.policy.dstb_action_net.parameters())))
                            _prev = getattr(self, "_dbg_adv_sig", None)
                            # Same obs/actions as the trainer, but with the
                            # COLLECTION-time arguments, to separate "parameters
                            # moved" from "arguments differ".
                            _alt, _ = self.policy.evaluate_adv_actions(
                                rollout_data.observations, actions,
                                buf_num=[0], side_flag=None)
                            _dalt = (_alt - rollout_data.old_log_prob).abs().mean().item()
                            _dcur = (log_prob - rollout_data.old_log_prob).abs().mean().item()
                            print(f"[PARAMDBG] adv param_sig now={_sig:.6f} "
                                  f"at_collect={_prev if _prev is None else f'{_prev:.6f}'} "
                                  f"moved={'YES' if (_prev is not None and abs(_sig-_prev)>1e-9) else 'no'} "
                                  f"| |trainer_lp - stored|={_dcur:.6f} "
                                  f"| |collectargs_lp - stored|={_dalt:.6f}", flush=True)
                    if epoch == 0 and os.environ.get("SPAR_DEBUG_KL"):
                        with th.no_grad():
                            _lr = (log_prob - rollout_data.old_log_prob).reshape(-1)
                            _exact = (_lr == 0).float().mean().item()
                            _kl = th.mean((th.exp(_lr) - 1) - _lr).item()
                            _o = rollout_data.observations
                            print(f"[KLDBG] {'ego' if update_ego else 'adv'} ep0 "
                                  f"n={_lr.numel()} kl={_kl:.5f} "
                                  f"lr mean={_lr.mean().item():+.5f} "
                                  f"std={_lr.std().item():.5f} "
                                  f"min={_lr.min().item():+.4f} max={_lr.max().item():+.4f} "
                                  f"exact0={_exact:.3f} | obs {tuple(_o.shape)} "
                                  f"sum={_o.float().sum().item():.4f} "
                                  f"old_lp mean={rollout_data.old_log_prob.mean().item():+.4f} "
                                  f"new_lp mean={log_prob.mean().item():+.4f}", flush=True)
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
                    # PopArt: `values` is negated above ONLY when update_adversary,
                    # so the ego-frame target is -returns in that case and +returns
                    # otherwise. mu is frame-sensitive and must be fed the ego frame;
                    # sigma is not, and sigma is all that reaches the loss (mu cancels
                    # in a difference of normalized quantities).
                    if _pa is not None:
                        _s = _pa.sigma.detach()
                        value_loss = F.mse_loss(rollout_data.returns / _s, values_pred / _s)
                    else:
                        value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())
                    coef = self.ent_coef if update_ego else self.dstb_ent_coef
                    pl = policy_loss#_ego if update_ego else policy_loss_adv
                    self.ego_params = self.policy.ctrl_optimizer.param_groups[0]['params']
                    # On-policy value update runs here unless V-trace is in pure-offload mode.
                    # In hybrid mode (vtrace + keep_onpolicy) the async worker is parked for all
                    # of train(), so this thread is the sole value writer right now -> safe.
                    do_onpolicy_value = (not self.vtrace_enabled) or self.vtrace_keep_onpolicy_value
                    if do_onpolicy_value:
                        loss = pl + coef * entropy_loss + self.vf_coef * value_loss
                    else:
                        # Value head is owned exclusively by the async V-trace worker.
                        loss = pl + coef * entropy_loss

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
                        #continue_training = False
                        stop_policy_training = True
                        if self.verbose >= 1:
                            print(f"training stopped at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                            print("training: %s" % "ego" if update_ego else "adv")
                        break

                    # Optimization step
                    self.policy.ctrl_optimizer.zero_grad()
                    self.policy.dstb_optimizer.zero_grad()
                    if do_onpolicy_value:
                        self.policy.value_optimizer.zero_grad()
                        if hasattr(self.policy, "ego_value_optimizer"):
                            self.policy.ego_value_optimizer.zero_grad()
                    loss.backward()

                    self.ego_grads_autograd_order.append([self.policy.ctrl_optimizer.param_groups[0]['params'][i].grad for i in range(len(self.policy.ctrl_optimizer.param_groups[0]['params']))])
                    self.policy.adv_grads_autograd_order.append([self.policy.dstb_optimizer.param_groups[0]['params'][i].grad for i in range(len(self.policy.dstb_optimizer.param_groups[0]['params']))])
                    if do_onpolicy_value:
                        self.policy.value_grads_autograd_order.append([self.policy.value_optimizer.param_groups[0]['params'][i].grad for i in range(len(self.policy.value_optimizer.param_groups[0]['params']))])
                    self.policy.value_loss.append(value_loss)
                    # Clip grad norm
                    if do_onpolicy_value:
                        # Worker is parked (pure-offload disabled or paused during train),
                        # so value grads here are this thread's own -> clip all params.
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    else:
                        # Pure offload: value .grad is written by the async worker; clipping
                        # self.policy.parameters() would read/scale it -> race + corruption.
                        # Clip only the params this thread steps (ego + adv).
                        _clip_params = [
                            p
                            for opt in (self.policy.ctrl_optimizer, self.policy.dstb_optimizer)
                            for group in opt.param_groups
                            for p in group["params"]
                        ]
                        th.nn.utils.clip_grad_norm_(_clip_params, self.max_grad_norm)
                    if update_ego and not stop_policy_training: # stop_policy_training is True when the policy is not training anymore
                        self.policy.ctrl_optimizer.step()
                    else:
                        if not stop_policy_training:
                            self.policy.dstb_optimizer.step()
                    
                    # regardless of stop_policy_training, we always update the value optimizer
                    # (unless V-trace owns the value head in pure-offload mode)

                    if do_onpolicy_value:
                        if hasattr(self.policy, "ego_value_optimizer") and update_ego:
                            self.policy.ego_value_optimizer.step()
                        else:
                            self.policy.value_optimizer.step()
                        # SEQUENTIAL path -- this is the LIVE one. The blended call site
                        # earlier is unreachable at num_adversaries == 1, which is every
                        # experiment we run; the first smoke test logged no minimax
                        # metrics at all, which is how that was caught.
                        #
                        # `update_adversary` is exactly the buffer selector used at
                        # the top of this loop (adversary_buffers vs rollout_buffer),
                        # so it is the same discriminator that sets the frame. Read
                        # it from there rather than re-deriving it.
                        self._minimax_q_update(rollout_data, [i],
                                               adv_frame=bool(update_adversary))
                    #self.policy.value_optimizer.step()

                # Trust-region hard stop (parity with the blended path + standard
                # PPO): if the KL early-stop fired on the last batch, break the
                # epoch loop too -- not just the current epoch's remaining batches
                # -- so target_kl bounds the whole per-update policy change, and the
                # ego (sequential path) stops at the same point the blended
                # adversary does. Without this the ego drifts modestly past
                # target_kl and outpaces the adversary near the clamp.
                if stop_policy_training:
                    break
        #self._update_schedulers(step_ego=update_ego, step_adv=(not update_ego), step_val=True, skip=not self.use_lr_annealing)
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
        _mm = getattr(self, "_minimax_stats", None)
        if _mm:
            for _k, _v in _mm.items():
                self.logger.record(f"train/minimax_{_k}", _v)
        if hasattr(self.policy, "ego_value_optimizer") and update_ego:
            self.logger.record("train/ego_value_loss", np.mean(value_losses))
        #self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        if update_adversary:
            self.logger.record("train/adv_clip_fraction", np.mean(clip_fractions))
        else:
            self.logger.record("train/ego_clip_fraction", np.mean(clip_fractions))
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
    
    def env_perturb_params(self, update_ego=True, update_adversary=True, zero_ego_action=False, zero_adv_action=False, random_adv_action=False):
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
        self.collect_rollouts(self.env, self.callback, buf, adv_buf, n_rollout_steps=self.n_steps, run_ego_forward=update_ego, run_adv_forward=update_adversary, use_mirror=self.use_mirror, zero_ego_action=zero_ego_action, zero_adv_action=zero_adv_action, random_adv_action=random_adv_action)
        
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
        excluded.extend(
            ["parallel_updater", "callback", "perturbed_agents", "adversary_buffers",
             # V-trace: worker holds threads/events/CUDA stream (unpicklable), the
             # replay buffers hold multi-GB arrays + a Lock, and vtrace_policy_lock is a
             # bare threading.Lock -- none should ever be serialized into a save.
             "vtrace_trainer", "vtrace_ego_replay", "vtrace_adv_replays", "vtrace_policy_lock"]
        )
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
            if self.device.type == th.device("cpu").type:
                n_workers = 1
            else:
                _, n_workers = get_n_workers()
            self.parallel_updater = ParallelUpdater(n_workers)
            self.first_run = True 

    def _log_leader_metrics(self, ego, entropy_losses, pg_losses, approx_kl_divs, explained_var, clip_range):
        prefix = "ego" if ego else "adv"

        self.logger.record(f"train/{prefix}_entropy_loss", np.mean(entropy_losses))
        self._check_entropy_collapse(prefix, entropy_losses)
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

    @th.no_grad()
    def _maybe_enumerate(self):
        """Enumerate the full 22x22 payoff at the CURRENT env states, if due.

        Runs on the TRAINING envs rather than a second stack: Monitor2P is the
        only thing that breaks under branching, and lbr_pause_monitor suspends
        exactly the three behaviours that break (see retro_wrappers). That
        avoids duplicating the env config, the memory, and the subprocess count.

        Called once per train(), AFTER the rollout is collected, so the branch
        steps cannot contaminate the data the policy update sees. The envs are
        snapshotted first and restored after, so the trajectory continues from
        precisely where it left off.

        Targets are stored, not successor observations: 484 successors x obs
        width per state is gigabytes, while the 22x22 target is 1.9 KB. The cost
        is that V(s') in the stored target ages -- and that is affordable,
        measured: capture loses 0.01 points at realistic critic error and under
        a point at twice the signal.
        """
        if int(getattr(self, "enum_every", 0)) <= 0 or not getattr(self, "minimax_q", False):
            return
        # Log the running totals on EVERY iteration, not only the ones that
        # enumerate. Recording them inside the branch below made the key vanish
        # from later dumps, which reads as "enumeration stopped" -- I misread it
        # that way myself within minutes of the first run. The budget is the
        # number that has to be quotable at any point, so it is always present.
        self.logger.record("train/enum_env_steps", float(self._enum_env_steps))
        self.logger.record("train/enum_states",
                           float(sum(len(o) for o, _ in self._enum_store)))
        # 1 = V_mm leaf (matches option B), 0 = scalar leaf (matches option A).
        # Logged because a MISMATCH here is silent and was worth -4.9 EV.
        self.logger.record("train/enum_leaf_is_vmm",
                           1.0 if str(getattr(self, "minimax_target", "returns")) == "minimax" else 0.0)
        if self.num_timesteps < getattr(self, "_enum_next_at", 0):
            return
        self._enum_next_at = self.num_timesteps + self.enum_every
        venv = self.env
        # Which matchup head V comes from. The enumeration runs on the whole
        # vec-env, and evaluate_states needs a head to route through; head 0 is
        # correct for the single-matchup configs this is run in, and asserting
        # it here is better than silently averaging two matchups' values.
        buf_num = [0]
        na = int(getattr(self, "minimax_n_ego", 22))
        n = venv.num_envs
        KEY = "enum_root"
        WALK = "enum_walk"
        HOLD = "enum_hold"
        try:
            venv.env_method("lbr_pause_monitor", True)
            # Snapshot the TRUE position first. Hunting for contact states means
            # stepping the envs forward, and those steps belong to the rollout
            # collector -- restoring WALK at the end leaves the training
            # trajectory exactly where it was.
            venv.env_method("lbr_snapshot", WALK)
            # PER-ENV hunting. Enumeration is vectorised: the 484 branches cost
            # the same whether 1 or 24 envs carry interaction, so accepting a
            # batch because ANY env has contact -- which my first version did --
            # buys nothing (P(any of 24 at p=0.065) = 80%, so it never walked).
            # Instead lock each env in independently as it finds contact, and
            # enumerate once every env is parked on a contact state.
            locked = np.zeros(n, dtype=bool)
            if self.enum_probe > 0:
                for _ in range(max(1, self.enum_walk)):
                    venv.env_method("lbr_snapshot", KEY,
                                    indices=np.where(~locked)[0].tolist())
                    fresh = self._enum_probe(venv, KEY, na, n, locked)
                    for k in np.where(fresh & ~locked)[0]:
                        venv.env_method("lbr_snapshot", HOLD, indices=[int(k)])
                    locked |= fresh
                    need = int(np.ceil(self.enum_probe_frac * n))
                    if locked.sum() >= need:
                        break
                    idx = np.where(~locked)[0].tolist()
                    venv.env_method("lbr_restore", KEY, indices=idx)
                    a_e, a_a = self._enum_sample(self._last_obs, buf_num)
                    o, _, _, _, _ = venv.step(np.stack([a_e, a_a], axis=1))
                    self._last_obs = o
                    self._enum_env_steps += n
                    self._enum_walked += 1
                self.logger.record("train/enum_walked", float(self._enum_walked))
                self.logger.record("train/enum_locked_frac", float(locked.mean()))
                if not locked.any():
                    venv.env_method("lbr_restore", WALK)
                    venv.env_method("lbr_drop", WALK)
                    return
                # park every locked env back on its own contact state
                li = np.where(locked)[0].tolist()
                venv.env_method("lbr_restore", HOLD, indices=li)
                venv.env_method("lbr_drop", HOLD, indices=li)
            venv.env_method("lbr_snapshot", KEY)
            obs0 = np.array(self._last_obs, copy=True)
            R = np.zeros((na, na, n), dtype=np.float32)
            V1 = np.zeros((na, na, n), dtype=np.float32)
            DN = np.zeros((na, na, n), dtype=bool)
            for i in range(na):
                succ = []
                for j in range(na):
                    venv.env_method("lbr_restore", KEY)
                    o1, r_l, r_r, d, infos = venv.step(self._enum_joint(i, j, n))
                    d = np.asarray(d, dtype=bool)
                    R[i, j] = np.asarray(r_l, dtype=np.float32).reshape(-1)
                    DN[i, j] = d
                    succ.append(self._enum_splice(o1, d, infos))
                    self._enum_env_steps += n
                V1[i] = self._enum_leaf_values(np.concatenate(succ, axis=0),
                                               buf_num).reshape(na, n)
            venv.env_method("lbr_restore", KEY)
            venv.env_method("lbr_drop", KEY)
            venv.env_method("lbr_restore", WALK)
            venv.env_method("lbr_drop", WALK)
        finally:
            venv.env_method("lbr_pause_monitor", False)
        M = (R + float(self.gamma) * V1 * (~DN)).transpose(2, 0, 1)
        if self.enum_contact_only:
            # judged on R, the raw emulator reward -- no critic, so a head that
            # merely predicts variation cannot make a state look like contact.
            Rw = R.transpose(2, 0, 1)
            Rw = Rw - Rw.mean(axis=(1, 2), keepdims=True)
            keep = np.linalg.norm(Rw.reshape(len(Rw), -1), axis=1) > 1e-12
            self.logger.record("train/enum_contact_frac", float(keep.mean()))
            obs0, M = obs0[keep], M[keep]
            if len(M) == 0:
                return
        self._enum_store.append((obs0, M.astype(np.float32)))
        if len(self._enum_store) > self.enum_buffer:
            self._enum_store.pop(0)
        self.logger.record("train/enum_env_steps", float(self._enum_env_steps))
        self.logger.record("train/enum_states", float(sum(len(o) for o, _ in self._enum_store)))

    def _enum_probe(self, venv, key, na, n, locked=None):
        """Which envs show reward variation under a few cheap branches?

        Returns a PER-ENV boolean, not a batch verdict. My first version
        returned `.any()` across envs, which is useless here: enumeration is
        vectorised, so the 484 branches cost the same whether 1 or 24 envs carry
        interaction, and P(any of 24 at p=0.065) = 80% meant it accepted the
        first candidate every time and never walked (enum_walked stayed 0).

        Judged on the raw emulator reward, so nothing the head predicts can make
        a state look like contact.

        KNOWN RISK -- FALSE NEGATIVES. A handful of probe pairs can miss
        interaction confined to specific action combinations, biasing the
        collected set toward states with BROAD interaction. Measurable by
        full-enumerating probe-rejected states and counting missed contact.
        """
        rs = []
        rng = np.random.RandomState(int(self.num_timesteps) & 0xFFFF)
        for _ in range(int(self.enum_probe)):
            i = int(rng.randint(0, na)); j = int(rng.randint(0, na))
            venv.env_method("lbr_restore", key,
                            indices=None if locked is None
                            else np.where(~locked)[0].tolist())
            _, r_l, _, _, _ = venv.step(self._enum_joint(i, j, n))
            rs.append(np.asarray(r_l, dtype=np.float64).reshape(-1))
            self._enum_env_steps += n
        venv.env_method("lbr_restore", key,
                        indices=None if locked is None
                        else np.where(~locked)[0].tolist())
        R = np.stack(rs)                       # (probe, n_envs)
        return (R.max(axis=0) - R.min(axis=0)) > 1e-12

    def _enum_sample(self, obs, buf_num):
        """On-policy (ego, adv) actions for walking between probe candidates.

        Mirrors PolicyOps.ego_probs / adv_probs in local_best_response -- there
        is no public accessor for either distribution, and the adversary path
        needs buf_num and evaluate=True. Written out rather than guessed: an
        earlier version called get_dstb_distribution, which does not exist.
        """
        from stable_baselines3.common.preprocessing import preprocess_obs
        with th.no_grad():
            t = th.as_tensor(obs, device=self.device).float()
            x = preprocess_obs(t, self.policy.observation_space,
                               normalize_images=self.policy.normalize_images)
            fe = self.policy.pi_ctrl_features_extractor(x)
            de = self.policy._get_ego_action_dist_from_latent(
                self.policy.mlp_extractor.ego_forward(fe))
            fa = self.policy.pi_dstb_features_extractor(x)
            da = self.policy._get_adv_action_dist_from_latent(
                self.policy.mlp_extractor.adv_forward(fa, side_flag=None),
                buf_num=buf_num, evaluate=True)[0]
            a_e = de.distribution[0].sample().cpu().numpy().reshape(-1)
            a_a = da.distribution[0].sample().cpu().numpy().reshape(-1)
        return a_e, a_a

    def _enum_joint(self, i, j, n):
        """The vec-env action for 'every env plays joint action (i, j)'."""
        return np.stack([np.full(n, i, dtype=np.int64),
                         np.full(n, j, dtype=np.int64)], axis=1)

    def _enum_splice(self, o1, d, infos):
        """Recover the true post-step obs where the worker auto-reset."""
        o1 = np.array(o1, copy=True)
        for k in range(len(d)):
            if d[k] and isinstance(infos[k], dict) and "terminal_observation" in infos[k]:
                o1[k] = infos[k]["terminal_observation"]
        return o1

    def _enum_leaf_values(self, obs, buf_num, side_flag=None, chunk=2048):
        """The leaf value for the enumerated target, MATCHING the on-policy term.

        THE BUG THIS FIXES. The head is trained on the SUM of two regressions.
        Under --minimax_target minimax the on-policy term regresses onto
        r + gamma*V_mm(s'), where V_mm is the equilibrium of the head's own
        matrix. The enumerated term was built with r + gamma*V_scalar(s') --
        because it reused bootstrap_delta's construction, which is a DIAGNOSTIC
        and rightly uses the scalar critic. Under kappa=1 those are very
        different objects, so every update pulled the head toward two mutually
        inconsistent definitions of Q. Measured consequence: training loss fell
        while held-out EV sat at -4.9, i.e. worse than a constant predictor.

        Under --minimax_target returns the on-policy term regresses onto the
        lambda-returns, which estimate Q^pi, whose one-step form IS
        r + gamma*V^pi(s') = r + gamma*V_scalar(s'). So the scalar critic is the
        CORRECT leaf there, and this returns it unchanged. The two branches are
        not a preference; each matches its own on-policy target.
        """
        if str(getattr(self, "minimax_target", "returns")) != "minimax":
            return self._enum_values(obs, buf_num)
        head = self.policy.minimax_head_for(buf_num)
        frozen = self._minimax_frozen_head(buf_num, head)
        out = []
        for i in range(0, obs.shape[0], chunk):
            t = th.as_tensor(obs[i:i + chunk], device=self.device).float()
            Mn = frozen(self.policy.minimax_latent(t, side_flag=side_flag))
            sol = solve_matrix_game(Mn, iters=getattr(self, "minimax_iters", 1024),
                                    eta=getattr(self, "minimax_eta", 0.5))
            out.append(sol.V.reshape(-1).cpu().numpy())
        return np.concatenate(out)

    def _enum_values(self, obs, buf_num):
        """V_ego over a branch batch.

        NOT predict_values: it calls self.value_net(...), and value_net is a
        ModuleDict in the multi-matchup policy, so it raises "Module [ModuleDict]
        is missing the required forward function". Unit tests cannot catch this
        -- they stub the policy -- and it took a live run to surface.

        evaluate_states divides env_indices by envs_per_matchup unconditionally
        (its =None default is a lie), then with len(buf_num)==1 forces the whole
        batch through that head. So feed indices that map to OUR head, exactly
        as PolicyOps.values_ego does in local_best_response.
        """
        head = int(buf_num[0])
        t = th.as_tensor(obs, device=self.device).float()
        env_idx = th.full((t.shape[0],), head * self.policy.envs_per_matchup,
                          dtype=th.long, device=self.device)
        v = self.policy.evaluate_states(t, buf_num=[head], env_indices=env_idx)
        return v.reshape(-1).cpu().numpy()

    def _enum_aux_loss(self, buf_num, side_flag=None):
        """MSE of the head against the ENUMERATED matrices, all 484 cells.

        enum_k < 484 subsamples the cells, which is the privilege ladder: the
        pilot measured 35/69/59/85/53% of the full-enumeration capture at k=16
        across five checkpoints, i.e. NO stable knee, so the default is the full
        matrix and a cheaper k must be justified per run rather than assumed.
        """
        # Same defensive access as the call site: no _enum_* state on an object
        # unpickled from a pre-feature checkpoint.
        if int(getattr(self, "enum_every", 0)) <= 0 or not getattr(self, "_enum_store", None):
            return None
        obs = np.concatenate([o for o, _ in self._enum_store], axis=0)
        M = np.concatenate([m for _, m in self._enum_store], axis=0)
        obs_t = th.as_tensor(obs, device=self.device).float()
        M_t = th.as_tensor(M, device=self.device).float()

        # HOLDOUT. The buffer is small (enum_buffer x n_envs states) and every
        # minibatch re-fits ALL of it, so the training loss falls even if the
        # head is only memorising those states -- and this codebase has already
        # seen a joint-action head put 81-91% of within-state energy into rank-10
        # NOISE while its loss improved. A descending enum_loss is therefore NOT
        # evidence of anything on its own; enum_loss_holdout is.
        #
        # Split by a fixed stride rather than at random so a state stays on the
        # same side for its whole life in the buffer -- a per-step reshuffle
        # would leak every held-out state into training within a few updates.
        n = len(M_t)
        hold = th.arange(n, device=self.device) % 5 == 0      # 20% held out
        if n < 10 or not bool(hold.any()) or bool(hold.all()):
            hold = th.zeros(n, dtype=th.bool, device=self.device)

        P = self.policy.minimax_matrices(
            obs_t, buf_num=buf_num, side_flag=side_flag,
            stop_grad=getattr(self, "minimax_stop_grad", True))
        na = M_t.shape[-1]
        if self.enum_k >= na * na:
            per_state = ((P - M_t) ** 2).mean(dim=(1, 2))
        else:
            idx = th.randperm(na * na, device=self.device)[:max(1, self.enum_k)]
            per_state = ((P.reshape(n, -1)[:, idx]
                          - M_t.reshape(n, -1)[:, idx]) ** 2).mean(dim=1)
        if bool(hold.any()):
            # EXPLAINED VARIANCE, not just MSE. An MSE is uninterpretable without
            # the target's own variance: a head that learns nothing but the MEAN
            # scores MSE ~ var(M), and this repo has already shipped a run where
            # 2e-04 looked small against target_std 0.017 and meant exactly that.
            # ev <= 0 => no better than predicting the mean, whatever the MSE.
            with th.no_grad():
                Mh = M_t[hold]
                resid = float(per_state[hold].mean())
                var = float(Mh.var(unbiased=False))
                ev = 1.0 - resid / var if var > 1e-30 else float("nan")
            self._minimax_stats = {
                **getattr(self, "_minimax_stats", {}),
                "enum_loss_holdout": resid,
                "enum_ev_holdout": ev,
                "enum_target_var": var,
                "enum_n_train": float((~hold).sum()),
            }
            return per_state[~hold].mean()
        return per_state.mean()

    def _check_entropy_collapse(self, prefix, entropy_losses):
        """Stop the run when a policy saturates to exactly zero entropy.

        entropy_loss is -mean(entropy), so |entropy_loss| ~ 0 means the policy is
        a deterministic point mass. That is ABSORBING, not a transient: no
        probability mass to move -> zero policy gradient -> it stays there. The
        run keeps producing a plausible score curve while no longer being
        self-play at all.

        A streak is required rather than a single hit, so a momentarily-saturated
        update does not kill an otherwise healthy run. For calibration: the arm
        that died had 2761 consecutive zero blocks and the healthy one had 0 of
        4608, so any patience in this range separates them cleanly.
        """
        if not entropy_losses:
            return
        mean_ent = float(np.mean(entropy_losses))
        if not np.isfinite(mean_ent) or abs(mean_ent) >= self.entropy_collapse_tol:
            self._entropy_zero_streak[prefix] = 0
        else:
            self._entropy_zero_streak[prefix] += 1
        streak = self._entropy_zero_streak[prefix]
        self.logger.record(f"train/{prefix}_entropy_zero_streak", streak)
        if streak and streak % max(1, self.entropy_collapse_patience // 4) == 0:
            print(f"[ENTROPY WARNING] {prefix} entropy has been ~0 for {streak} "
                  f"updates (|{mean_ent:.3g}| < {self.entropy_collapse_tol:g}); "
                  f"abort at {self.entropy_collapse_patience}", flush=True)
        if self.entropy_collapse_abort and streak >= self.entropy_collapse_patience:
            raise RuntimeError(
                f"ENTROPY COLLAPSE on the {prefix} policy: |entropy| < "
                f"{self.entropy_collapse_tol:g} for {streak} consecutive updates "
                f"at {self.num_timesteps} steps. This is an absorbing state -- the "
                f"policy gradient is zero and it cannot recover, so the run is no "
                f"longer self-play. Aborting. Set --entropy_collapse_abort False "
                f"to continue anyway (results downstream will not be valid "
                f"self-play measurements).")

    def _minimax_kappa(self):
        """How much of the bootstrap is V_minimax right now. 0.0 = PHASE 0."""
        k = float(getattr(self, "minimax_bootstrap_kappa", 0.0))
        if k == 0.0:
            return 0.0
        w = int(getattr(self, "minimax_bootstrap_warmup", 0))
        if w > 0:
            k *= min(1.0, float(self.num_timesteps) / float(w))
        return k

    @th.no_grad()
    def _minimax_values_for(self, obs, buf_num, side_flag=None, chunk=4096):
        """V_mm(s) = equilibrium value of the head's Q(s,.,.), for a batch."""
        out = []
        for i in range(0, obs.shape[0], chunk):
            M = self.policy.minimax_matrices(obs[i:i + chunk], buf_num=buf_num,
                                             side_flag=side_flag, stop_grad=True)
            out.append(solve_matrix_game(
                M, iters=getattr(self, "minimax_iters", 1024),
                eta=getattr(self, "minimax_eta", 0.5)).V.reshape(-1))
        return th.cat(out)

    def _minimax_bootstrap(self, rollout_buffer, adversary_buffers, last_obs,
                           last_values, side_flag=None):
        """PHASE 1. Replace the bootstrap value with the MINIMAX value.

            V_boot = (1 - kappa) * V_scalar + kappa * V_mm

        overwriting rollout_buffer.values IN PLACE and returning the replacement
        for last_values. Everything downstream -- compute_returns_and_advantage,
        the advantages, the policy loss, the value loss -- then runs UNCHANGED on
        the new values. No new return path, and lambda is whatever --gae_lambda
        says (use 0 -- see LAMBDA below).

        KAPPA == 0 RETURNS BEFORE COMPUTING ANYTHING. The solve does not run and
        the buffer is never written, so Phase 0 is BITWISE identical, not
        approximately so. A test asserts exactly that. This is what keeps the
        diagnostic/feeding switch a runtime flag rather than a code fork.

        LAMBDA. Use --gae_lambda 0. The minimax bootstrap is an OFF-POLICY
        target; a lambda-return mixes ON-POLICY intermediate rewards into it,
        which is unsound without a Retrace-style trace (not built). At lambda 0
        the target is r + gamma*V_mm(s') and the question is clean. Note lambda 0
        ALSO changes the policy's bias/variance tradeoff, so a kappa=0 arm at
        lambda 0 is REQUIRED as the control -- otherwise an observed difference
        cannot be attributed to the operator rather than to lambda.

        WHAT THIS COUPLES, deliberately and worth stating: the value loss now
        regresses value_net onto returns built from V_mm, so V_scalar drifts
        toward V_minimax over training and the two stop being independent. That
        is arguably the point, but it means kappa is not a clean interpolation
        after the first few rollouts.

        SOLVED ONCE PER ROLLOUT over all buffer states (~74 ms for 12,288 states
        at 1024 iters, ~2.5% of env time). Never per minibatch -- that cost 3x
        throughput when the diagnostic path did it.
        """
        kappa = self._minimax_kappa()
        if kappa == 0.0:
            return last_values
        if not getattr(self, "minimax_q", False):
            raise RuntimeError("--minimax_bootstrap_kappa > 0 requires --minimax_q True")
        if getattr(self, "vtrace_enabled", False):
            raise RuntimeError("--minimax_bootstrap_kappa > 0 requires "
                               "--vtrace_enabled False; the V-trace path has its "
                               "own value targets and would disagree about what V is")
        bn = [0]
        obs = rollout_buffer.observations
        T, n_envs = obs.shape[0], obs.shape[1]
        flat = th.as_tensor(obs.reshape((T * n_envs,) + obs.shape[2:])).to(self.device)
        v_mm = self._minimax_values_for(flat, bn, side_flag=side_flag).reshape(T, n_envs)

        v_old = rollout_buffer.values
        was_np = isinstance(v_old, np.ndarray)
        v_old_t = th.as_tensor(v_old).to(self.device).float().reshape(T, n_envs)
        v_new = (1.0 - kappa) * v_old_t + kappa * v_mm
        if not th.isfinite(v_new).all():
            raise RuntimeError("V_minimax bootstrap produced non-finite values; "
                               "the head has diverged -- check train/minimax_q_scale")
        rollout_buffer.values = (v_new.cpu().numpy().astype(np.float32)
                                 if was_np else v_new.to(v_old.dtype))

        # Adversary buffers hold the SAME transitions, sliced by adversary, in
        # the ADV frame -- so they get the negated ego value, exactly as the
        # existing last_values=-values convention does. Recomputing V_mm on
        # their observations would be the same solve twice with a sign.
        for i in range(self.num_adversaries):
            sl = slice(i * self.n_env_per_adv, (i + 1) * self.n_env_per_adv)
            ab = adversary_buffers[i]
            a_np = isinstance(ab.values, np.ndarray)
            neg = (-v_new[:, sl])
            ab.values = (neg.cpu().numpy().astype(np.float32)
                         if a_np else neg.to(ab.values.dtype))

        lo = th.as_tensor(last_obs).to(self.device)
        lv_mm = self._minimax_values_for(lo, bn, side_flag=side_flag)
        lv = (1.0 - kappa) * th.as_tensor(last_values).to(self.device).float().reshape(-1) \
             + kappa * lv_mm
        self._minimax_stats = {
            **(getattr(self, "_minimax_stats", None) or {}),
            "boot_kappa": kappa,
            "boot_v_mm_std": float(v_mm.std()),
            "boot_v_scalar_std": float(v_old_t.std()),
            # >> 1 means V_mm is far wilder than the value it replaces, which is
            # the first thing to look at if the policy destabilises.
            "boot_scale_ratio": float(v_mm.std() / v_old_t.std().clamp_min(1e-12)),
        }
        return lv.reshape(th.as_tensor(last_values).shape)

    def _minimax_frozen_head(self, buf_num, live_head):
        """Snapshot of the minimax head that supplies V_mm(s'), refreshed ONCE
        per rollout.

        WHY A SNAPSHOT AND NOT THE LIVE HEAD. Option B's target is built from
        the head's own Q at the successor, so without freezing, the target moves
        under every minibatch and every epoch -- the network chases a value it
        is simultaneously changing. That is the textbook way to make a
        bootstrapped fixed point diverge, and worse, it makes divergence
        indistinguishable from "Q is simply wrong", which is the exact
        ambiguity that caused option B to be deferred in the first place.

        Keyed on num_timesteps, which advances once per rollout, so the refresh
        happens exactly once no matter how many passes/epochs/minibatches call
        in. That avoids threading a refresh call through two call sites and the
        learn loop, where a missed one would silently mean a stale target.
        """
        stamp = int(self.num_timesteps)
        cache = getattr(self, "_mm_frozen", None)
        if cache is None or cache.get("stamp") != stamp:
            cache = {"stamp": stamp, "heads": {}}
            self._mm_frozen = cache
        key = id(live_head)
        h = cache["heads"].get(key)
        if h is None:
            h = deepcopy(live_head).eval()
            for p in h.parameters():
                p.requires_grad_(False)
            cache["heads"][key] = h
        return h

    def _minimax_q_update(self, rollout_data, buf_num, side_flag=None, *, adv_frame):
        """PHASE 0, option A: regress Q(s, a_ego, a_adv) onto the EXISTING
        lambda-returns. Returns the loss, or None when --minimax_q is off.

        adv_frame is KEYWORD-ONLY AND HAS NO DEFAULT, deliberately. It selects
        the sign of the target, getting it wrong is silent, and a default would
        let a new call site inherit the wrong one by omission -- which is
        exactly how the bug this parameter fixes survived. See FRAME below.

        WHY option A and not the minimax bootstrap: this fits Q^pi -- the value
        of a joint action given both players continue with their current
        policies -- from a target that is DATA and never references Q. It cannot
        diverge, and a failure is unambiguous: if Q^pi is constant plus noise
        across the 22x22 matrix then the observation does not determine
        action-conditional value, and no bootstrap can manufacture that.
        Option B (target = r + gamma * V_minimax(s')) is the real algorithm and
        fits the policy-INDEPENDENT fixed point, but it is self-referential, so
        a failure there cannot be told apart from "the bootstrap did not
        converge". Switch in phase 1 once the gate has answered the first
        question; it is a one-line change of target.

        Fully self-contained: its own optimizer over its own parameters, and the
        shared encoder runs under no_grad. Enabling this cannot move the policy
        -- verified, max|shared - initial| == 0.0 across optimizer steps.
        """
        if not getattr(self, "minimax_q", False):
            return None
        head = self.policy.minimax_head_for(buf_num)
        if head is None:
            return None
        M = self.policy.minimax_matrices(
            rollout_data.observations, buf_num=buf_num, side_flag=side_flag,
            stop_grad=getattr(self, "minimax_stop_grad", True))
        # A joint-action critic needs BOTH seats' actions, and BOTH buffers
        # carry them: AdvRolloutBufferSamples as `dstb_actions`,
        # Q_RolloutBufferSamples (the EGO buffer) as `adv_actions`. So this
        # update runs on BOTH passes and the skip below is dead in every
        # configuration we run -- kept only as a guard for a future buffer type.
        #
        # An earlier comment here asserted the opposite ("the ego path carries
        # only its own action, so this is skipped there"). It was wrong, and
        # because the frame negation below was unconditional, the ego pass was
        # training the head on a SIGN-FLIPPED target for the whole of Phase 0.
        # `skipped_no_adv_actions` never appeared in a single log, which is the
        # evidence that the skip never fired.
        a_adv_t = getattr(rollout_data, "dstb_actions", None)
        if a_adv_t is None:
            a_adv_t = getattr(rollout_data, "adv_actions", None)
        if a_adv_t is None:
            self._minimax_stats = {
                **getattr(self, "_minimax_stats", {}),
                "skipped_no_adv_actions": 1.0,
            }
            return None
        a_ego = rollout_data.actions.long().reshape(-1)
        a_adv = a_adv_t.long().reshape(-1)
        b = th.arange(M.shape[0], device=M.device)
        q_played = M[b, a_ego, a_adv]

        # FRAME. MinimaxHead is EGO-payoff by construction (that convention is
        # what lets the adversary's value be -Q and collapses the six negation
        # sites). The two buffers store returns in DIFFERENT frames:
        #
        #   adversary buffers  ADV frame  -- fed -rewards (see the rollout) and
        #                                 last_values=-values, so negate to
        #                                 reach ego frame.
        #   ego rollout_buffer EGO frame  -- already the head's convention, so
        #                                 DO NOT negate.
        #
        # Negating unconditionally trained the head to predict +G on one pass
        # and -G on the other, i.e. half the updates actively undid the other
        # half. That is why the caller must state the frame explicitly.
        #
        # Getting this wrong is silent and total: measured on the 6.72M
        # checkpoint before the fix, best-fit was G = -0.990*Q + 0.0009,
        # corr(Q,G) = -0.929, EV(Q,G) = -2.74 while EV(-Q,G) = +0.86. The head
        # had learned the target essentially perfectly, inverted, and every
        # downstream probe read it as a broken head.
        #
        # I reasoned about this frame for PopArt and concluded mu cancels in a
        # difference of normalized quantities. True there; NOT true here, where
        # the loss is a direct regression onto returns. Nothing cancels.
        _tgt_gap = None
        _boot = None
        if str(getattr(self, "minimax_target", "returns")) == "minimax":
            # OPTION B -- LITTMAN'S OPERATOR.
            #     target = r + gamma * V_mm(s') * (1 - done)
            # V_mm is the equilibrium value of the head's OWN 22x22 matrix at
            # the successor, solved by the same optimistic-MWU routine. lambda=0
            # by construction: the target is purely local to one transition, so
            # nothing has to survive the minibatch shuffle and no buffer field
            # is needed. lambda>0 would mix ON-POLICY intermediate rewards into
            # an off-policy target and is unsound without a Retrace-style trace.
            #
            # THE ASYMMETRY TO WATCH. One transition trains exactly ONE of the
            # 484 cells (the joint action observed), but the max-min READS all
            # 484 at the successor. The target therefore depends on cells that
            # have never received a gradient, and max preferentially selects the
            # largest -- which the no-op probe measured at 1.39x the entire
            # spread of Q across actions. That is what q_scale/td_error are for.
            nxt = getattr(rollout_data, "next_observations", None)
            if nxt is None:
                raise RuntimeError(
                    "--minimax_target minimax requires next_observations, and "
                    "this buffer's samples do not carry them. Refusing to fall "
                    "back to the returns target: mixing two different targets "
                    "in one head is worse than not running at all.")
            r = rollout_data.rewards.reshape(-1).float()
            d = rollout_data.dones.reshape(-1).float()
            # FRAME. Option A negates the RETURN; here it is the REWARD, because
            # the adversary buffer stores -rewards while V_mm is already
            # ego-frame (the head is ego-payoff). Same parameter, same reason.
            r_ego = -r if adv_frame else r
            frozen = self._minimax_frozen_head(buf_num, head)
            with th.no_grad():
                M_next = frozen(self.policy.minimax_latent(nxt, side_flag=side_flag))
                _sol_t = solve_matrix_game(
                    M_next, iters=getattr(self, "minimax_iters", 1024),
                    eta=getattr(self, "minimax_eta", 0.5))
                _boot = float(self.gamma) * _sol_t.V.reshape(-1) * (1.0 - d)
                target = r_ego + _boot
            _tgt_gap = _sol_t.gap
        else:
            target = rollout_data.returns.reshape(-1)
            if adv_frame:
                target = -target
        loss = F.mse_loss(target, q_played)

        # The enumerated matrices are the ONLY term that sees more than the one
        # played cell. Added rather than substituted so the on-policy signal --
        # which is on-distribution and never stale -- still trains the head.
        # getattr, not a direct call: a model unpickled from a checkpoint written
        # before this feature has no _enum_* state, and the frame/isolation tests
        # drive this method on a minimal stand-in. Matches the defensive access
        # used for every other optional attribute in this file.
        _enum_fn = getattr(self, "_enum_aux_loss", None)
        _enum = _enum_fn(buf_num, side_flag) if _enum_fn is not None else None
        if _enum is not None:
            self._minimax_stats = {**getattr(self, "_minimax_stats", {}),
                                   "enum_loss": float(_enum.detach())}
            loss = loss + self.enum_loss_coef * _enum

        self.policy.minimax_optimizer.zero_grad()
        loss.backward()
        self.policy.minimax_optimizer.step()

        head.note_visits(a_ego, a_adv)

        # The inner solve is DIAGNOSTIC here -- option A's target never uses it.
        # Running it every minibatch at 1024 iters cost ~3x throughput (490
        # steps/s against ~1460 for the same config without minimax). Sample it
        # instead; the gap and V stats move slowly.
        self._mm_calls = getattr(self, "_mm_calls", 0) + 1
        every = max(1, int(getattr(self, "minimax_stat_every", 10)))
        if self._mm_calls % every != 1 and getattr(self, "_minimax_stats", None):
            self._minimax_stats["loss"] = float(loss)
            return loss
        with th.no_grad():
            sol = solve_matrix_game(M.detach(),
                                    iters=getattr(self, "minimax_iters", 1024),
                                    eta=getattr(self, "minimax_eta", 0.5))
        with th.no_grad():
            _t = target.reshape(-1)
            _q = q_played.detach().reshape(-1)
            _tv = float(_t.var())
            _v = head.cell_visits.reshape(-1)
            _vp = th.quantile(_v.float(), th.tensor([0.10, 0.50], device=_v.device))
        _stats = {
            "loss": float(loss),
            # loss alone is UNINTERPRETABLE: MSE ~ target variance is exactly
            # what a head that learned only the mean produces, and that is what
            # 2e-04 against G_std ~0.017 meant for the whole first Phase 0 run.
            # ev makes it a ratio instead of a bare number.
            "target_std": float(_t.std()),
            "ev": float(1.0 - float(((_t - _q) ** 2).mean()) / _tv) if _tv > 0 else float("nan"),
            # COVERAGE IS A GATE PRECONDITION, not a nicety. p_max(ego) ~ 0.94
            # was measured, so sampled joint actions concentrate on a handful of
            # the 484 cells while LBR branches over all 22 adversary actions. If
            # coverage is low, "Q does not discriminate branches" is
            # indistinguishable from "Q never saw those branches" and the gate
            # result means nothing.
            "coverage": head.coverage(),
            # FACTORED HEAD ONLY. These turn the offline experiments that
            # motivated this parameterization into live training metrics:
            #   w_norm       ||W(s)||. ZERO at init by construction, so any
            #                growth is the data asking for interaction. If it
            #                stays ~0, gamma is not earning its place and the
            #                44-output separable head would do.
            #   gamma_share  fraction of WITHIN-state energy in the interaction.
            #                Offline on the true payoff: 0.069.
            #   anti_share   antisymmetric (cyclic / rock-paper-scissors) share
            #                of gamma -- the part that forces mixing. Offline:
            #                0.441. Tracking that means the head is capturing
            #                real structure rather than fitting noise.
            #   noop_emb     ||e_ego(0) - e_ego(9)||, the byte-identical action
            #                pair. Truth is 0. Now 4 params per action instead of
            #                513; if this does not fall, the density argument for
            #                the whole parameterization was wrong.
            **({f"fx_{k}": v for k, v in head.interaction_stats(
                    self.policy.minimax_latent(rollout_data.observations,
                                               side_flag=side_flag)).items()}
               if hasattr(head, "interaction_stats") else {}),
            # coverage (fraction of cells EVER touched) read 1.000 for the whole
            # first run while hiding a 1,400x imbalance underneath. The
            # distribution is what actually matters.
            "visits_min": float(_v.min()),
            "visits_p10": float(_vp[0]),
            "visits_median": float(_vp[1]),
            "visits_max": float(_v.max()),
            # SIGN GUARD. corr(prediction, target) must be POSITIVE. A negative
            # value means the frames have diverged again, and that failure is
            # otherwise invisible -- the loss goes down either way, because a
            # head fitting -G has exactly the same MSE as one fitting +G against
            # the flipped target. This number is the only thing that catches it
            # from the log alone.
            "target_corr": float(th.corrcoef(
                th.stack([q_played.detach().reshape(-1), target.reshape(-1)]))[0, 1])
                if q_played.numel() > 1 else float("nan"),
            # Free convergence certificate for the inner solve. If this is not
            # near zero the solve failed and every V_minimax is meaningless.
            "duality_gap": float(sol.gap.median()),
            # median hides a single unconverged state; one bad solve silently
            # corrupts one GAE bootstrap.
            "duality_gap_max": float(sol.gap.max()),
            "V_minimax_mean": float(sol.V.mean()),
            "V_minimax_std": float(sol.V.std()),
            # Spread of Q ACROSS the matrix at a fixed state -- the quantity the
            # whole bet rests on. V is constant across one-action branches; if
            # this is ~0 then Q is too, and the direction is dead.
            "q_branch_std": float(M.detach().std(dim=(1, 2)).mean()),
        }
        # OPTION B ONLY -- telling DIVERGENCE apart from "Q is simply wrong".
        # That ambiguity is the stated reason option B was deferred, so it gets
        # instrumented rather than argued about:
        #   q_scale        ||Q||. A self-referential bootstrap that is diverging
        #                  blows this up without the loss necessarily rising,
        #                  because the target inflates alongside the prediction.
        #   target_scale   same for the target. q_scale/target_scale drifting
        #                  TOGETHER is divergence; q_scale alone is not.
        #   td_error       |target - Q(s,i,j)|, the quantity actually minimised.
        #   bootstrap_frac share of |target| coming from gamma*V_mm(s') rather
        #                  than r. At gamma 0.94 this should be large; near zero
        #                  means the bootstrap is inert and this is greedy.
        #   target_gap*    duality gap of the solve that BUILT THE TARGET (the
        #                  one above is a sampled diagnostic on s, not s'). An
        #                  unconverged solve here corrupts the target itself.
        #   corr_q_reward  replacement sign guard. target_corr compares Q with
        #                  the on-policy return, which option B deliberately
        #                  abandons, so it stops being a sign check -- but Q
        #                  must still move WITH immediate reward.
        if _boot is not None:
            with th.no_grad():
                _t = target.reshape(-1)
                _r = rollout_data.rewards.reshape(-1).float()
                _r = -_r if adv_frame else _r
                _stats.update({
                    "q_scale": float(q_played.detach().abs().mean()),
                    "target_scale": float(_t.abs().mean()),
                    "td_error": float((_t - q_played.detach()).abs().mean()),
                    "bootstrap_frac": float(_boot.abs().mean()
                                            / _t.abs().mean().clamp_min(1e-12)),
                    "target_gap": float(_tgt_gap.median()),
                    "target_gap_max": float(_tgt_gap.max()),
                    "corr_q_reward": float(th.corrcoef(th.stack(
                        [q_played.detach().reshape(-1), _r]))[0, 1])
                        if q_played.numel() > 1 else float("nan"),
                })
        # PER-FRAME COPIES. This dict is rebuilt on every call, so whichever
        # pass ran last silently overwrote the other's numbers -- which is how a
        # sign error on ONE seat stayed invisible while the aggregate looked
        # merely mediocre (target_corr +0.414 rather than negative). Keep both
        # seats' sign guards alive simultaneously; if either goes negative the
        # frame has diverged again on that seat specifically.
        _prev = getattr(self, "_minimax_stats", None) or {}
        # PHASE 1 keys are written by _minimax_bootstrap at COLLECT time, while
        # this runs at TRAIN time -- two producers, one dict, and this one
        # rebuilds it from scratch. Without carrying them forward the boot_*
        # metrics never reach the logger at all, which is how they were missing
        # for the first 282k steps of the first Phase 1 arm. Same failure mode as
        # the per-seat keys below, and as the overwrite that hid the ego-pass
        # frame bug: whichever writer runs LAST owns the dict.
        for _bk, _bv in _prev.items():
            if _bk.startswith("boot_"):
                _stats[_bk] = _bv
        _sfx = "adv" if adv_frame else "ego"
        _stats[f"target_corr_{_sfx}"] = _stats["target_corr"]
        _stats[f"ev_{_sfx}"] = _stats["ev"]
        for _k in ("target_corr_ego", "ev_ego", "target_corr_adv", "ev_adv"):
            if _k not in _stats and _k in _prev:
                _stats[_k] = _prev[_k]
        self._minimax_stats = _stats
        return loss

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
                        log_prob, entropy = self.policy.evaluate_adv_actions(rollout_data.observations, actions, buf_num=[i], side_flag=side_flag)
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
