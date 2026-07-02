"""V-trace off-policy value correction for the SPAR shared state-value function.

Plugs an asynchronous experience-replay value trainer alongside the existing
on-policy PPO rollout loop. PPO's policy update is untouched; the shared
``value_net`` / ``value_optimizer`` is owned exclusively by the worker here.

Per-side marginal correction:
  - Ego buffer uses ratio  pi_ego(a_ego|s) / mu_ego(a_ego|s)
  - Adv buffer i uses ratio pi_adv_i(a_adv|s) / mu_adv_i(a_adv|s)

All transitions are stored in canonical ego-perspective sign (positive r, V):
the worker's V-trace recursion runs identically for both sides.
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn.functional as F


def compute_vtrace_targets_pt(
    values: th.Tensor,
    bootstrap_value: th.Tensor,
    rewards: th.Tensor,
    dones: th.Tensor,
    log_pi: th.Tensor,
    log_mu: th.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[th.Tensor, Dict[str, float]]:
    """V-trace target v_s for the state-value function (Espeholt et al., 2018).

    Inputs are chunk-shaped (B, T) with B independent sequences of length T.

    Args:
        values:          (B, T)  V_theta(s_t)               -- detached, treated as fixed targets here.
        bootstrap_value: (B,)    V_theta(s_T)               -- bootstrap for the last step.
        rewards:         (B, T)  r_t   in ego-perspective sign.
        dones:           (B, T)  1.0 if transition at t was terminal.
        log_pi:          (B, T)  log pi(a_t | s_t)          under current target policy.
        log_mu:          (B, T)  log mu(a_t | s_t)          under behavior policy at collection.
        gamma:           discount.
        rho_bar:         truncation for the TD-error ratio (sets the fixed point).
        c_bar:           truncation for the trace ratio    (sets variance).

    Returns:
        v_targets: (B, T)  V-trace target v_s for each step. Detached.
        diagnostics: dict of scalar floats for logging.
    """
    assert values.shape == rewards.shape == dones.shape == log_pi.shape == log_mu.shape
    assert bootstrap_value.shape == (values.shape[0],)

    with th.no_grad():
        log_ratio = log_pi - log_mu
        ratio = th.exp(log_ratio)
        rho = th.clamp(ratio, max=rho_bar)
        c = th.clamp(ratio, max=c_bar)

        B, T = values.shape
        next_values = th.cat([values[:, 1:], bootstrap_value.unsqueeze(1)], dim=1)  # V(s_{t+1})
        next_non_terminal = 1.0 - dones

        deltas = rho * (rewards + gamma * next_values * next_non_terminal - values)

        v_minus_V = th.zeros_like(values)
        acc = th.zeros(B, device=values.device, dtype=values.dtype)
        for t in reversed(range(T)):
            acc = deltas[:, t] + gamma * c[:, t] * next_non_terminal[:, t] * acc
            v_minus_V[:, t] = acc

        v_targets = values + v_minus_V

        diagnostics = {
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item(),
            "rho_sat_frac": (ratio >= rho_bar).float().mean().item(),
            "c_sat_frac": (ratio >= c_bar).float().mean().item(),
            "delta_abs_mean": deltas.abs().mean().item(),
        }

    return v_targets, diagnostics


class VTraceReplayBuffer:
    """Per-side off-policy replay buffer for V-trace value updates.

    Storage layout: ring buffer of shape (capacity, n_envs, *) holding the
    state s_t at the start of each transition together with (a_t, r_t, done_t,
    log mu(a_t|s_t)). The next state s_{t+1} is recovered by reading slot t+1
    (because the rollout loop writes ``self._last_obs = new_obs`` between
    steps, so the obs written at slot t+1 is exactly the new_obs of the
    transition at slot t).

    Numpy storage on CPU; sample_chunks materializes a (B, T+1) obs slice
    plus (B, T) per-step fields and moves them to ``device``.
    """

    def __init__(
        self,
        capacity: int,
        n_envs: int,
        obs_shape: Tuple[int, ...],
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: th.device,
        env_index_offset: int = 0,
    ) -> None:
        """env_index_offset shifts the stored env index when this buffer represents
        a slice of the global env layout (e.g., adv buffer i covers global envs
        [i*envs_per_matchup, (i+1)*envs_per_matchup)). Required so the policy's
        matchup routing in evaluate_states gets the right env -> matchup mapping.
        """
        self.capacity = int(capacity)
        self.n_envs = int(n_envs)
        self.obs_shape = tuple(obs_shape)
        self.action_dim = int(action_dim)
        self.device = device
        self.env_index_offset = int(env_index_offset)

        self.obs = np.zeros((self.capacity, self.n_envs, *self.obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((self.capacity, self.n_envs, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.mu_log_probs = np.zeros((self.capacity, self.n_envs), dtype=np.float32)

        self.write_pos = 0  # monotonically increasing global step counter
        self.lock = threading.Lock()

    def __getstate__(self):
        # threading.Lock is unpicklable; drop it (recreated on unpickle).
        state = self.__dict__.copy()
        state.pop("lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        mu_log_prob: np.ndarray,
    ) -> None:
        """Append one timestep across all n_envs. Caller passes ego-perspective reward."""
        with self.lock:
            pos = self.write_pos % self.capacity
            self.obs[pos] = obs
            self.actions[pos] = np.asarray(action).reshape(self.n_envs, self.action_dim)
            self.rewards[pos] = reward
            self.dones[pos] = done
            self.mu_log_probs[pos] = mu_log_prob
            self.write_pos += 1

    def num_steps(self) -> int:
        return min(self.write_pos, self.capacity)

    def can_sample(self, seq_len: int) -> bool:
        # Need at least seq_len + 1 timesteps so that a chunk of T+1 obs is available.
        return self.write_pos >= seq_len + 1

    # Slots within this many steps of the oldest kept index are excluded from
    # sampling so a concurrent writer (which overwrites the oldest slots) cannot
    # clobber a sampled chunk during the unlocked gather below.
    _OVERWRITE_MARGIN = 256

    def sample_chunks(self, batch_size: int, seq_len: int) -> Optional[Dict[str, th.Tensor]]:
        """Return a batch of B contiguous chunks of length (T+1 obs, T transitions).

        Returns None if the buffer doesn't yet hold seq_len+1 timesteps.

        The lock is held only for the O(B) index math; the heavy fancy-index
        gather and host->device copy happen outside the lock so the rollout
        thread's per-step add() is not blocked.
        """
        T = int(seq_len)
        # ---- index selection under the lock (cheap) ----
        with self.lock:
            if not self.can_sample(seq_len):
                return None
            write_pos = self.write_pos
            # Valid global start positions g such that g+T < write_pos (last obs needed
            # is at slot g+T) and g >= oldest still-kept slot. When the ring is full,
            # bump the oldest bound by a margin to survive concurrent overwrites.
            if write_pos > self.capacity:
                # Keep the margin small relative to capacity so it never starves sampling.
                margin = min(self._OVERWRITE_MARGIN, self.capacity // 8)
                oldest = write_pos - self.capacity + margin
            else:
                oldest = 0
            newest_start = write_pos - T - 1
            if oldest > newest_start:
                return None
            starts = np.random.randint(oldest, newest_start + 1, size=batch_size)
            env_idxs = np.random.randint(0, self.n_envs, size=batch_size)

        # ---- gather + transfer outside the lock ----
        pos_T1 = (starts[:, None] + np.arange(T + 1)[None, :]) % self.capacity  # (B, T+1)
        pos_T = pos_T1[:, :T]                                                    # (B, T)
        env_T1 = np.broadcast_to(env_idxs[:, None], (batch_size, T + 1))
        env_T = np.broadcast_to(env_idxs[:, None], (batch_size, T))

        obs_chunk = self.obs[pos_T1, env_T1]
        action_chunk = self.actions[pos_T, env_T]
        reward_chunk = self.rewards[pos_T, env_T]
        done_chunk = self.dones[pos_T, env_T]
        mu_log_prob_chunk = self.mu_log_probs[pos_T, env_T]

        mean_age = float(write_pos - 1 - starts.mean())
        global_env_indices = env_idxs + self.env_index_offset  # (B,) global env idx

        return {
            "obs": th.from_numpy(obs_chunk).to(self.device, non_blocking=True),
            "actions": th.from_numpy(action_chunk).to(self.device, non_blocking=True),
            "rewards": th.from_numpy(reward_chunk).to(self.device, non_blocking=True),
            "dones": th.from_numpy(done_chunk).to(self.device, non_blocking=True),
            "mu_log_probs": th.from_numpy(mu_log_prob_chunk).to(self.device, non_blocking=True),
            "env_indices": th.from_numpy(global_env_indices.astype(np.int64)).to(self.device, non_blocking=True),
            "mean_age": mean_age,
        }


class VTraceValueTrainer:
    """Background thread that runs V-trace value updates on the SPAR shared value head.

    Single worker, single optimizer step site — avoids races on ``value_optimizer``.
    Round-robins across (ego replay, adv replay 0, adv replay 1, ...). On each
    iteration: sample a (B, T+1) chunk, forward V_theta on all T+1 obs (with grad),
    forward log pi(a|s) on T transitions (no grad), compute V-trace targets,
    MSE backprop, step. Metrics are pushed to a thread-safe deque drained by the
    main thread for logger emission.

    Runs on a dedicated CUDA stream so kernel launches can overlap with the
    rollout/main-thread inference on the GPU.
    """

    def __init__(
        self,
        policy: Any,
        value_optimizer: th.optim.Optimizer,
        ego_replay: VTraceReplayBuffer,
        adv_replays: List[VTraceReplayBuffer],
        num_adversaries: int,
        is_discrete_action: bool,
        gamma: float,
        rho_bar: float,
        c_bar: float,
        seq_len: int,
        batch_size: int,
        max_grad_norm: float,
        device: th.device,
        poll_sleep: float = 0.002,
        warmup_transitions: int = 0,
        policy_lock: Any = None,
    ) -> None:
        self.policy = policy
        # Shared with the rollout thread: serializes forward passes through the policy
        # so SB3's stateful, SHARED distribution objects (self.action_dist /
        # self.dstb_action_dist, mutated by proba_distribution()) aren't clobbered mid
        # rollout sample by this worker's log_pi forward. None => no locking.
        self.policy_lock = policy_lock
        self.value_optimizer = value_optimizer
        self.ego_replay = ego_replay
        self.adv_replays = list(adv_replays)
        self.num_adversaries = int(num_adversaries)
        self.is_discrete_action = bool(is_discrete_action)
        self.gamma = float(gamma)
        self.rho_bar = float(rho_bar)
        self.c_bar = float(c_bar)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device
        self.poll_sleep = float(poll_sleep)
        self.warmup_transitions = int(warmup_transitions)

        self.stop_event = threading.Event()
        # Pause handshake: _resume_event set => worker may run; cleared => worker parks.
        # _idle_event is set by the worker once it has parked (and flushed its CUDA
        # stream), so pause() can block until the value head is guaranteed stable.
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._idle_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.stream: Optional[th.cuda.Stream] = (
            th.cuda.Stream(device=device) if (isinstance(device, th.device) and device.type == "cuda")
            else None
        )

        self._rr_idx = 0
        self._param_iter_cache: Optional[List[th.nn.Parameter]] = None

        self.metrics_lock = threading.Lock()
        self.metrics_buffer: Deque[Dict[str, float]] = deque(maxlen=4096)
        self.updates_count = 0

    def __getstate__(self):
        # Threads, Events, Locks and CUDA streams are unpicklable. Drop them; a
        # pickled/deep-copied trainer is inert (it is never restored to run -- the
        # model excludes vtrace_trainer from save and re-creates it in learn()).
        state = self.__dict__.copy()
        for key in ("stop_event", "_resume_event", "_idle_event", "metrics_lock", "thread", "stream"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._idle_event = threading.Event()
        self.metrics_lock = threading.Lock()
        self.thread = None
        self.stream = None

    def _value_params(self) -> List[th.nn.Parameter]:
        if self._param_iter_cache is None:
            params: List[th.nn.Parameter] = []
            for group in self.value_optimizer.param_groups:
                for p in group["params"]:
                    params.append(p)
            self._param_iter_cache = params
        return self._param_iter_cache

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="vtrace-value-worker")
        self.thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        self._resume_event.set()  # release a parked worker so it can observe stop
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            self.thread = None

    def pause(self, timeout: float = 30.0) -> bool:
        """Request the worker to park and block until it has (finishing any in-flight
        update and flushing its CUDA stream). Returns True once parked.

        Safe to call when no worker is running (returns True immediately).
        """
        if self.thread is None or not self.thread.is_alive():
            return True
        self._resume_event.clear()
        # Wait for the worker to reach the park barrier. If it was already parked,
        # _idle_event is already set and this returns immediately.
        deadline_steps = max(1, int(timeout / 0.1))
        for _ in range(deadline_steps):
            if self._idle_event.wait(timeout=0.1):
                return True
            if not self.thread.is_alive():
                return True
        return self._idle_event.is_set()

    def resume(self) -> None:
        # Strict flush: if the main thread just ran an on-policy value update (hybrid
        # mode), its writes were enqueued on the default stream. Sync it here, before
        # un-parking, so the worker's next kernels (on its own stream) observe the
        # updated value params rather than a pre-update snapshot. Called on the main
        # thread, so current_stream() is the default/main stream.
        if self.stream is not None and th.cuda.is_available():
            th.cuda.current_stream().synchronize()
        self._idle_event.clear()
        self._resume_event.set()

    def drain_metrics(self) -> List[Dict[str, float]]:
        with self.metrics_lock:
            out = list(self.metrics_buffer)
            self.metrics_buffer.clear()
        return out

    def _warmed_up(self) -> bool:
        if self.warmup_transitions <= 0:
            return True
        if self.ego_replay.num_steps() < self.warmup_transitions:
            return False
        for buf in self.adv_replays:
            if buf.num_steps() < self.warmup_transitions:
                return False
        return True

    def _loop(self) -> None:
        ctx = th.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()
        n_buffers = 1 + self.num_adversaries
        with ctx:
            while not self.stop_event.is_set():
                # Park barrier: when paused (e.g., during the on-policy ego/adv update),
                # finish nothing new, flush the stream so the value head is stable, and
                # block until resumed. Guarantees no value-param writes during train().
                if not self._resume_event.is_set():
                    if self.stream is not None:
                        self.stream.synchronize()
                    self._idle_event.set()
                    while not self._resume_event.wait(timeout=0.1):
                        if self.stop_event.is_set():
                            return
                    continue

                if not self._warmed_up():
                    time.sleep(self.poll_sleep)
                    continue

                side = self._rr_idx % n_buffers
                self._rr_idx += 1

                if side == 0:
                    replay = self.ego_replay
                    buf_num: Optional[List[int]] = None  # ego
                    is_ego = True
                else:
                    adv_i = side - 1
                    replay = self.adv_replays[adv_i]
                    buf_num = [adv_i]
                    is_ego = False

                try:
                    batch = replay.sample_chunks(self.batch_size, self.seq_len)
                    if batch is None:
                        time.sleep(self.poll_sleep)
                        continue
                    self._update(batch, is_ego=is_ego, buf_num=buf_num)
                except Exception as exc:  # pragma: no cover - keep worker alive
                    with self.metrics_lock:
                        self.metrics_buffer.append({"vtrace_error": 1.0})
                    print(
                        f"[VTraceValueTrainer] update error (is_ego={is_ego}, buf_num={buf_num}): {exc}\n"
                        f"{traceback.format_exc()}",
                        flush=True,
                    )
                    time.sleep(self.poll_sleep)

    def _update(
        self,
        batch: Dict[str, th.Tensor],
        is_ego: bool,
        buf_num: Optional[List[int]],
    ) -> None:
        obs = batch["obs"]                       # (B, T+1, *obs_shape)
        actions = batch["actions"]               # (B, T, action_dim)
        rewards = batch["rewards"].float()        # (B, T)
        dones = batch["dones"].float()            # (B, T)
        mu_log_probs = batch["mu_log_probs"].float()  # (B, T)
        env_indices = batch["env_indices"]        # (B,) global env idx
        mean_age = float(batch.get("mean_age", 0.0))

        B, T_plus_1 = obs.shape[:2]
        T = T_plus_1 - 1
        obs_tail = obs.shape[2:]

        flat_obs_full = obs.reshape((B * T_plus_1,) + obs_tail)
        flat_obs_T = obs[:, :T].reshape((B * T,) + obs_tail)
        flat_actions = actions.reshape((B * T,) + actions.shape[2:])

        # env_indices broadcast: same env per row, but we need per-timestep alignment.
        env_T1 = env_indices.unsqueeze(1).expand(B, T_plus_1).reshape(-1)
        env_T = env_indices.unsqueeze(1).expand(B, T).reshape(-1)

        if is_ego:
            buf_num_for_states = list(range(self.num_adversaries))
        else:
            buf_num_for_states = buf_num  # [adv_i]

        # All policy forward passes (value + log_pi) run under the shared lock so they
        # never overlap the rollout thread's self.policy() sampling. SB3's distribution
        # objects are shared instance attributes mutated by proba_distribution(); a
        # concurrent forward here would clobber them and corrupt the rollout's action
        # sampling. Backward + optimizer step stay OUTSIDE the lock (they touch no
        # shared forward state), so the expensive parts still overlap the rollout.
        _fwd_lock = self.policy_lock if self.policy_lock is not None else contextlib.nullcontext()
        with _fwd_lock:
            # ---- value forward (with grad) ----
            values_flat = self.policy.evaluate_states(
                flat_obs_full,
                buf_num=buf_num_for_states,
                env_indices=env_T1,
                side_flag=None,
            )
            values_flat = values_flat.squeeze(-1)                  # (B*(T+1),)
            values_chunk = values_flat.reshape(B, T_plus_1)
            values_T = values_chunk[:, :T]                          # (B, T)
            bootstrap_value = values_chunk[:, T]                    # (B,)

            # ---- log pi forward (no grad) ----
            with th.no_grad():
                if is_ego:
                    pi_actions = flat_actions.long().flatten() if self.is_discrete_action else flat_actions
                    log_pi_flat, _ = self.policy.evaluate_ego_actions(
                        flat_obs_T,
                        pi_actions,
                        side_flag=None,
                    )
                else:
                    pi_actions = flat_actions.long().flatten() if self.is_discrete_action else flat_actions
                    log_pi_flat, _ = self.policy.evaluate_adv_actions(
                        flat_obs_T,
                        pi_actions,
                        buf_num=buf_num,
                        side_flag=None,
                    )
        log_pi = log_pi_flat.reshape(B, T)

        # ---- V-trace target (no grad through target) ----
        v_target, diagnostics = compute_vtrace_targets_pt(
            values=values_T.detach(),
            bootstrap_value=bootstrap_value.detach(),
            rewards=rewards,
            dones=dones,
            log_pi=log_pi,
            log_mu=mu_log_probs,
            gamma=self.gamma,
            rho_bar=self.rho_bar,
            c_bar=self.c_bar,
        )

        # ---- loss + step ----
        loss = F.mse_loss(values_T, v_target)

        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self._value_params(), self.max_grad_norm)
        self.value_optimizer.step()

        self.updates_count += 1
        with self.metrics_lock:
            self.metrics_buffer.append({
                "vtrace_value_loss": float(loss.item()),
                "vtrace_mean_age": mean_age,
                "vtrace_side": 0.0 if is_ego else float(buf_num[0] + 1),
                "vtrace_v_target_mean": float(v_target.mean().item()),
                "vtrace_v_target_std": float(v_target.std().item()),
                **{f"vtrace_{k}": float(v) for k, v in diagnostics.items()},
            })


