import numpy as np
import os
import csv
from datetime import datetime
from collections import deque
from typing import Callable, Dict, Optional, Tuple


class RatingStagnationTracker:
    """
    Generic stagnation tracker for rating-velocity and entropy signals.

    This tracker is intentionally model-agnostic: callers provide ratings,
    entropy, and (optionally) a callback for LR adjustments.
    """

    def __init__(
        self,
        patience: int,
        tolerance: float,
        rel_tolerance: float,
        ema_beta: float,
        eps: float,
        eval_games: int,
        entropy_weight: float,
        lr_patience: int,
        use_velocity_signal: bool = True,
        use_entropy_signal: bool = True,
        use_slope_early_stop: bool = False,
        slope_window: int = 20,
        slope_tolerance: float = 5e-3,
        min_slope_checks: int = 10,
        entropy_window_size: int = 50,
        entropy_stop_ratio: float = 0.15,
        enable_local_entropy_plot: bool = False,
        local_plot_dir: str = "logs/local_entropy_plots",
        local_plot_prefix: str = "stagnation",
        local_plot_every_checks: int = 1,
        entropy_warmup_checks: int = 100,
        enable_local_reward_plot: bool = False,
        enable_local_kl_plot: bool = False,
        entropy_ratio_only: bool = False,
    ) -> None:
        self.patience = int(patience)
        self.tolerance = float(tolerance)
        self.rel_tolerance = float(rel_tolerance)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self.eval_games = max(1, int(eval_games))
        self.entropy_weight = float(entropy_weight)
        self.lr_patience = int(lr_patience)
        self.use_velocity_signal = bool(use_velocity_signal)
        self.use_entropy_signal = bool(use_entropy_signal)
        self.use_slope_early_stop = bool(use_slope_early_stop)
        self.slope_window = max(2, int(slope_window))
        self.slope_tolerance = float(slope_tolerance)
        self.min_slope_checks = max(1, int(min_slope_checks))
        self.entropy_window_size = max(2, int(entropy_window_size))
        self.entropy_stop_ratio = float(entropy_stop_ratio)
        self.enable_local_entropy_plot = bool(enable_local_entropy_plot)
        self.local_plot_dir = str(local_plot_dir)
        self.local_plot_prefix = str(local_plot_prefix)
        self.local_plot_every_checks = max(1, int(local_plot_every_checks))
        self.entropy_warmup_checks = max(0, int(entropy_warmup_checks))
        self.enable_local_reward_plot = bool(enable_local_reward_plot)
        self.enable_local_kl_plot = bool(enable_local_kl_plot)
        self.entropy_ratio_only = bool(entropy_ratio_only)
        self.local_entropy_csv_path = None
        self.local_entropy_png_path = None
        self.local_reward_csv_path = None
        self.local_reward_png_path = None
        self.local_kl_csv_path = None
        self.local_kl_png_path = None
        self.reset(np.array([], dtype=np.float64))

    def enabled(self) -> bool:
        return self.use_velocity_signal or self.use_entropy_signal

    def reset(self, ratings: np.ndarray) -> None:
        ratings = np.asarray(ratings, dtype=np.float64)
        self.games_since_eval = 0
        self.num_checks = 0
        self.wait_count = 0
        self.best_metric = float("inf")
        self.last_metric = None
        self.last_velocity = None
        self.dynamic_threshold = None
        self.last_eval_ratings = np.copy(ratings)
        self.ema_metric = None
        self.ema_abs_metric = None
        self.ema_reward = None
        self.ema_abs_reward = None
        self.ema_abs_reward_dev = None
        self.metric_history = deque(maxlen=self.slope_window)
        self.last_metric_slope = None
        self.last_metric_slope_normalized = None
        self.last_slope_is_flat = None
        self.entropy_ema = None
        self.entropy_start = None
        self.entropy_smoothed_history = deque(maxlen=self.entropy_window_size)
        self.last_entropy_window_avg = None
        self.last_entropy_window_ratio = None
        self.last_entropy_window_stop = False
        self.ema_abs_entropy = None
        self.ema_abs_entropy_dev = None
        self.last_entropy_tolerance = None
        self.last_entropy_is_stable = None
        self.last_within_entropy_warmup = True
        self.entropy_stable_checks = 0
        self.last_entropy_stability_stop = False
        self.entropy_history = deque(maxlen=self.slope_window)
        self.last_entropy_signal_slope = None
        self.last_entropy_signal_slope_normalized = None
        self.last_entropy_signal_slope_is_flat = None
        self.entropy_timestep_history = []
        self.entropy_raw_history = []
        self.entropy_ema_history = []
        self._plot_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.local_entropy_csv_path = None
        self.local_entropy_png_path = None
        self._entropy_csv_rows_written = 0
        self.reward_timestep_history = []
        self.reward_raw_history = []
        self.reward_ema_history = []
        self.local_reward_csv_path = None
        self.local_reward_png_path = None
        self._reward_csv_rows_written = 0
        # KL divergence (PPO train/approx_kl) -- recorded per check() call when
        # the caller passes current_kl. EMA-smoothed like entropy/reward.
        self.kl_ema = None
        self.kl_timestep_history = []
        self.kl_raw_history = []
        self.kl_ema_history = []
        self.local_kl_csv_path = None
        self.local_kl_png_path = None
        self._kl_csv_rows_written = 0

    def _update_ema_stability_state(
        self,
        current_value: float,
        ema_value: float,
        ema_abs_value: float,
        ema_abs_dev: float,
    ) -> Tuple[float, float, float, float, bool]:
        current_value = float(current_value)
        prev_ema_value = float(ema_value)
        abs_dev = abs(current_value - prev_ema_value)
        updated_ema_value = self.ema_beta * ema_value + (1.0 - self.ema_beta) * current_value
        updated_ema_abs_value = self.ema_beta * ema_abs_value + (1.0 - self.ema_beta) * abs(current_value)
        updated_ema_abs_dev = self.ema_beta * ema_abs_dev + (1.0 - self.ema_beta) * abs_dev
        dynamic_tolerance = max(
            self.tolerance,
            self.rel_tolerance * max(updated_ema_abs_value, self.eps),
        )
        is_stable = updated_ema_abs_dev <= dynamic_tolerance
        return (
            updated_ema_value,
            updated_ema_abs_value,
            updated_ema_abs_dev,
            dynamic_tolerance,
            is_stable,
        )

    def _save_local_entropy_plot(self, force: bool = False) -> None:
        if not self.enable_local_entropy_plot:
            return
        if len(self.entropy_timestep_history) == 0:
            return
        if (not force) and (self.num_checks % self.local_plot_every_checks != 0):
            return
        os.makedirs(self.local_plot_dir, exist_ok=True)
        if self.local_entropy_csv_path is None or self.local_entropy_png_path is None:
            base_name = f"{self.local_plot_prefix}_{self._plot_id}"
            self.local_entropy_csv_path = os.path.join(self.local_plot_dir, f"{base_name}.csv")
            self.local_entropy_png_path = os.path.join(self.local_plot_dir, f"{base_name}.png")

        start = self._entropy_csv_rows_written
        need_header = start == 0
        with open(self.local_entropy_csv_path, "a" if not need_header else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(["timestep", "entropy_raw", "entropy_ema"])
            for t, raw, ema in zip(
                self.entropy_timestep_history[start:],
                self.entropy_raw_history[start:],
                self.entropy_ema_history[start:],
            ):
                writer.writerow([t, raw, ema])
        self._entropy_csv_rows_written = len(self.entropy_timestep_history)

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.plot(self.entropy_timestep_history, self.entropy_raw_history, label="entropy_raw", alpha=0.5)
            plt.plot(self.entropy_timestep_history, self.entropy_ema_history, label="entropy_ema", linewidth=2.0)
            plt.xlabel("exploiter training timestep")
            plt.ylabel("entropy")
            plt.title(f"Entropy vs Timestep ({self.local_plot_prefix})")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(self.local_entropy_png_path, dpi=150)
            plt.close()
        except Exception:
            # Keep training robust even if plotting backend is unavailable.
            pass

    def _save_local_reward_plot(self, force: bool = False) -> None:
        if not self.enable_local_reward_plot:
            return
        if len(self.reward_timestep_history) == 0:
            return
        if (not force) and (self.num_checks % self.local_plot_every_checks != 0):
            return
        os.makedirs(self.local_plot_dir, exist_ok=True)
        if self.local_reward_csv_path is None or self.local_reward_png_path is None:
            base_name = f"{self.local_plot_prefix}_{self._plot_id}_REWARD"
            self.local_reward_csv_path = os.path.join(self.local_plot_dir, f"{base_name}.csv")
            self.local_reward_png_path = os.path.join(self.local_plot_dir, f"{base_name}.png")

        start = self._reward_csv_rows_written
        need_header = start == 0
        with open(self.local_reward_csv_path, "a" if not need_header else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(["timestep", "reward_raw", "reward_ema"])
            for t, raw, ema in zip(
                self.reward_timestep_history[start:],
                self.reward_raw_history[start:],
                self.reward_ema_history[start:],
            ):
                writer.writerow([t, raw, ema])
        self._reward_csv_rows_written = len(self.reward_timestep_history)

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.plot(self.reward_timestep_history, self.reward_raw_history, label="reward_raw", alpha=0.5)
            plt.plot(self.reward_timestep_history, self.reward_ema_history, label="reward_ema", linewidth=2.0)
            plt.xlabel("exploiter training timestep")
            plt.ylabel("reward")
            plt.title(f"REWARD vs Timestep ({self.local_plot_prefix})")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(self.local_reward_png_path, dpi=150)
            plt.close()
        except Exception:
            # Keep training robust even if plotting backend is unavailable.
            pass

    def _save_local_kl_plot(self, force: bool = False) -> None:
        if not self.enable_local_kl_plot:
            return
        if len(self.kl_timestep_history) == 0:
            return
        if (not force) and (self.num_checks % self.local_plot_every_checks != 0):
            return
        os.makedirs(self.local_plot_dir, exist_ok=True)
        if self.local_kl_csv_path is None or self.local_kl_png_path is None:
            base_name = f"{self.local_plot_prefix}_{self._plot_id}_KL"
            self.local_kl_csv_path = os.path.join(self.local_plot_dir, f"{base_name}.csv")
            self.local_kl_png_path = os.path.join(self.local_plot_dir, f"{base_name}.png")

        start = self._kl_csv_rows_written
        need_header = start == 0
        with open(self.local_kl_csv_path, "a" if not need_header else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(["timestep", "kl_raw", "kl_ema"])
            for t, raw, ema in zip(
                self.kl_timestep_history[start:],
                self.kl_raw_history[start:],
                self.kl_ema_history[start:],
            ):
                writer.writerow([t, raw, ema])
        self._kl_csv_rows_written = len(self.kl_timestep_history)

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.plot(self.kl_timestep_history, self.kl_raw_history, label="kl_raw", alpha=0.5)
            plt.plot(self.kl_timestep_history, self.kl_ema_history, label="kl_ema", linewidth=2.0)
            plt.xlabel("exploiter training timestep")
            plt.ylabel("approx_kl (PPO train/approx_kl)")
            plt.title(f"KL divergence vs Timestep ({self.local_plot_prefix})")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(self.local_kl_png_path, dpi=150)
            plt.close()
        except Exception:
            # Keep training robust even if plotting backend is unavailable.
            pass

    def _compute_normalized_slope(self, history: np.ndarray, scale: float) -> Tuple[float, float]:
        x = np.arange(history.size, dtype=np.float64)
        x_centered = x - x.mean()
        y_centered = history - history.mean()
        denom = float(np.dot(x_centered, x_centered))
        slope = float(np.dot(x_centered, y_centered) / max(denom, self.eps))
        normalized_slope = abs(slope) / max(abs(scale), self.eps)
        return slope, normalized_slope

    def register_games(self, n_games: int) -> None:
        self.games_since_eval += int(max(0, n_games))

    def check(
        self,
        ratings: np.ndarray,
        current_entropy: Optional[float],
        lr_adjustment_callback: Optional[Callable[[], None]] = None,
        timestep: Optional[float] = None,
        current_reward: Optional[float] = None,
        current_kl: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        # NOTE: This tracker no longer makes early-stop decisions. All
        # stagnation / EMA-velocity / slope / entropy-window logic was
        # removed. The method now only:
        #   * gates calls by eval_games (preserves caller pacing),
        #   * records entropy/reward/KL into the per-signal histories the
        #     local plot helpers consume,
        #   * triggers the three local plot writers,
        #   * returns (False, logs) so callers' `if should_stop:` branches
        #     are effectively dead code.
        # The `lr_adjustment_callback`, `use_velocity_signal`,
        # `use_entropy_signal`, `entropy_stop_ratio`, `entropy_ratio_only`,
        # `use_slope_early_stop`, `patience`, `slope_*`, `entropy_window_*`,
        # and `entropy_warmup_*` knobs are now no-ops; they remain in
        # __init__ only so existing callers continue to work without
        # signature changes. Manual stop-key (STOP_<key>) files still
        # work — that's a separate mechanism in ManualStopFileCallback.
        ratings = np.asarray(ratings, dtype=np.float64)
        if ratings.size == 0:
            return False, None
        if self.games_since_eval < self.eval_games:
            return False, None

        rating_movements = np.abs(ratings - self.last_eval_ratings)
        mean_velocity = float(np.mean(rating_movements))
        self.num_checks += 1
        self.last_velocity = mean_velocity

        # Entropy: record raw + EMA-smoothed values regardless of
        # use_entropy_signal so the local entropy plot always has data.
        if current_entropy is not None:
            current_entropy_f = float(current_entropy)
            if self.entropy_ema is None:
                self.entropy_ema = current_entropy_f
            else:
                self.entropy_ema = (
                    self.ema_beta * self.entropy_ema
                    + (1.0 - self.ema_beta) * current_entropy_f
                )
            t = float(self.num_checks if timestep is None else timestep)
            self.entropy_timestep_history.append(t)
            self.entropy_raw_history.append(current_entropy_f)
            self.entropy_ema_history.append(float(self.entropy_ema))

        # Reward: same shape, drives the local reward plot.
        if current_reward is not None:
            current_reward_f = float(current_reward)
            if self.ema_reward is None:
                self.ema_reward = current_reward_f
            else:
                self.ema_reward = (
                    self.ema_beta * self.ema_reward
                    + (1.0 - self.ema_beta) * current_reward_f
                )
            reward_t = float(self.num_checks if timestep is None else timestep)
            self.reward_timestep_history.append(reward_t)
            self.reward_raw_history.append(current_reward_f)
            self.reward_ema_history.append(float(self.ema_reward))

        # KL: same shape, drives the local KL plot.
        if current_kl is not None:
            try:
                kl_val = float(current_kl)
            except (TypeError, ValueError):
                kl_val = float("nan")
            if np.isfinite(kl_val):
                if self.kl_ema is None:
                    self.kl_ema = kl_val
                else:
                    self.kl_ema = (
                        self.ema_beta * self.kl_ema
                        + (1.0 - self.ema_beta) * kl_val
                    )
                kl_t = float(self.num_checks if timestep is None else timestep)
                self.kl_timestep_history.append(kl_t)
                self.kl_raw_history.append(kl_val)
                self.kl_ema_history.append(float(self.kl_ema))

        self.games_since_eval = 0
        self.last_eval_ratings = np.copy(ratings)

        logs = {
            "stag/velocity": mean_velocity,
            "stag/entropy": (
                float(current_entropy) if current_entropy is not None else float("nan")
            ),
            "stag/ent_ema": (
                float(self.entropy_ema) if self.entropy_ema is not None else float("nan")
            ),
            "stag/reward_ema": (
                float(self.ema_reward) if self.ema_reward is not None else float("nan")
            ),
            "stag/kl_ema": (
                float(self.kl_ema) if self.kl_ema is not None else float("nan")
            ),
            "stag/num_checks": float(self.num_checks),
        }

        self._save_local_entropy_plot(force=False)
        self._save_local_reward_plot(force=False)
        self._save_local_kl_plot(force=False)
        return False, logs

class BRConvergenceTracker:
    def __init__(
        self,
        patience=10,
        tolerance=1e-4,
        window_size=50,
        ema_beta=0.99,
        rel_tolerance=0.05,
        warmup_checks=100,
        eps=1e-8,
        use_slope_early_stop: bool = False,
        slope_window: int = 20,
        slope_tolerance: float = 5e-3,
        min_slope_checks: int = 10,
        log_prefix="",
    ):
        """
        Args:
            patience: How many 'checks' to wait without improvement before stopping.
            tolerance: The threshold for 'zero' improvement or absolute convergence.
            window_size: Number of steps to average over to smooth out minimax oscillations.
            ema_beta: EMA smoothing factor for reward-stability checks.
            rel_tolerance: Relative tolerance against EMA reward scale.
            warmup_checks: Number of checks before enabling EMA stability decisions.
            eps: Numerical floor for relative tolerance scaling.
        """
        self.patience = patience
        self.tolerance = tolerance
        self.window_size = window_size
        self.ema_beta = ema_beta
        self.rel_tolerance = rel_tolerance
        self.warmup_checks = warmup_checks
        self.eps = eps
        self.use_slope_early_stop = bool(use_slope_early_stop)
        self.slope_window = max(2, int(slope_window))
        self.slope_tolerance = float(slope_tolerance)
        self.min_slope_checks = max(1, int(min_slope_checks))
        self.log_prefix = str(log_prefix)
        
        self.history = deque(maxlen=window_size)
        self.best_metric = float('inf')
        self.wait_count = 0
        self.steps_trained = 0

        # EMA reward-stability state
        self.ema_reward = None
        self.ema_abs_reward = None
        self.ema_abs_dev = None
        self.ema_entropy = None
        self.ema_abs_entropy = None
        self.ema_abs_entropy_dev = None
        self.num_checks = 0
        self.stable_checks = 0
        self.last_reward_tolerance = None
        self.last_entropy_tolerance = None
        self.last_reward_is_stable = None
        self.last_entropy_is_stable = None
        self.last_combined_is_stable = None
        self.last_within_warmup = True
        self.reward_history = deque(maxlen=self.slope_window)
        self.entropy_history = deque(maxlen=self.slope_window)
        self.last_signal_slope = None
        self.last_signal_slope_normalized = None
        self.last_signal_slope_is_flat = None

    def _compute_normalized_slope(self, history: np.ndarray, scale: float) -> Tuple[float, float]:
        x = np.arange(history.size, dtype=np.float64)
        x_centered = x - x.mean()
        y_centered = history - history.mean()
        denom = float(np.dot(x_centered, x_centered))
        slope = float(np.dot(x_centered, y_centered) / max(denom, self.eps))
        normalized_slope = abs(slope) / max(abs(scale), self.eps)
        return slope, normalized_slope

    def _prefixed_message(self, message: str) -> str:
        if self.log_prefix:
            return f"{self.log_prefix} {message}"
        return message

    def _update_ema_stability_state(self, current_value, ema_value, ema_abs_value, ema_abs_dev):
        """
        Apply one EMA update step and return updated state plus a stability flag.

        Returns:
            (
                updated_ema_value,
                updated_ema_abs_value,
                updated_ema_abs_dev,
                dynamic_tolerance,
                is_stable,
            )
        """
        current_value = float(current_value)
        prev_ema_value = ema_value
        abs_dev = abs(current_value - prev_ema_value)
        updated_ema_value = self.ema_beta * ema_value + (1.0 - self.ema_beta) * current_value
        updated_ema_abs_value = self.ema_beta * ema_abs_value + (1.0 - self.ema_beta) * abs(current_value)
        updated_ema_abs_dev = self.ema_beta * ema_abs_dev + (1.0 - self.ema_beta) * abs_dev
        dynamic_tolerance = max(
            self.tolerance,
            self.rel_tolerance * max(updated_ema_abs_value, self.eps),
        )
        is_stable = updated_ema_abs_dev <= dynamic_tolerance
        return (
            updated_ema_value,
            updated_ema_abs_value,
            updated_ema_abs_dev,
            dynamic_tolerance,
            is_stable,
        )

    def check(self, current_exploitability):
        """
        Returns True if the agents have converged.
        Metric can be NashConv, Exploitability, or Sum of Gradient Norms.
        """
        self.history.append(current_exploitability)
        self.steps_trained += 1
        
        # Don't check until we have a full window
        if len(self.history) < self.window_size:
            return False
        print(self._prefixed_message("checking for convergence..."))
        # 1. Calculate exponentially weighted smoothed metric
        # Newer values get larger weights than older values.
        history = np.asarray(self.history, dtype=np.float64)
        decay = np.exp(-1.0 / max(1, self.window_size))
        # Oldest point gets decay^(N-1), newest gets decay^0.
        weights = decay ** np.arange(len(history) - 1, -1, -1, dtype=np.float64)
        weights /= weights.sum()
        smoothed_metric = float(np.dot(weights, history))
        
        # 2. Hard Convergence: Is the error below our absolute tolerance?
        #if smoothed_metric < self.tolerance:
        #    print(f"--- Hard Convergence reached at step {self.steps_trained} ---")
        #    return True
            
        # 3. Stagnation: Has the metric stopped improving?
        if smoothed_metric < (self.best_metric - (self.tolerance * 0.1)):
            self.best_metric = smoothed_metric
            self.wait_count = 0  # Reset patience
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience:
            print(
                self._prefixed_message(
                    f"early stopping triggered at step {self.steps_trained} (stagnation)"
                )
            )
            return True
            
        return False

    def check_reward_stability(
        self,
        current_reward,
        current_entropy=None,
        use_reward_stagnation: bool = True,
        use_entropy_stagnation: bool = True,
    ):
        """
        Returns True when reward (and optionally entropy) have stabilized
        according to EMA absolute deviation.
        """
        self.num_checks += 1
        if not use_reward_stagnation and not use_entropy_stagnation:
            self.last_reward_tolerance = None
            self.last_entropy_tolerance = None
            self.last_reward_is_stable = True
            self.last_entropy_is_stable = True
            self.last_combined_is_stable = False
            self.last_within_warmup = self.num_checks <= self.warmup_checks
            self.stable_checks = 0
            return False

        current_reward = float(current_reward)
        if current_entropy is not None:
            current_entropy = float(current_entropy)
        reward_is_stable = True
        if use_reward_stagnation:
            if self.ema_reward is None:
                self.ema_reward = current_reward
                self.ema_abs_reward = abs(current_reward)
                self.ema_abs_dev = 0.0
                self.last_reward_tolerance = max(
                    self.tolerance,
                    self.rel_tolerance * max(self.ema_abs_reward, self.eps),
                )
                self.last_reward_is_stable = False
                reward_is_stable = False
            else:
                (
                    self.ema_reward,
                    self.ema_abs_reward,
                    self.ema_abs_dev,
                    reward_tolerance,
                    reward_is_stable,
                ) = self._update_ema_stability_state(
                    current_reward, self.ema_reward, self.ema_abs_reward, self.ema_abs_dev
                )
                self.last_reward_tolerance = reward_tolerance
                self.last_reward_is_stable = reward_is_stable
        else:
            self.last_reward_tolerance = None
            self.last_reward_is_stable = True
        
        entropy_is_stable = True
        if use_entropy_stagnation and current_entropy is not None:
            if self.ema_entropy is None:
                self.ema_entropy = current_entropy
                self.ema_abs_entropy = abs(current_entropy)
                self.ema_abs_entropy_dev = 0.0
                self.last_entropy_tolerance = max(
                    self.tolerance,
                    self.rel_tolerance * max(self.ema_abs_entropy, self.eps),
                )
                self.last_entropy_is_stable = False
                entropy_is_stable = False
            else:
                (
                    self.ema_entropy,
                    self.ema_abs_entropy,
                    self.ema_abs_entropy_dev,
                    entropy_tolerance,
                    entropy_is_stable,
                ) = self._update_ema_stability_state(
                    current_entropy,
                    self.ema_entropy,
                    self.ema_abs_entropy,
                    self.ema_abs_entropy_dev,
                )
                self.last_entropy_tolerance = entropy_tolerance
                self.last_entropy_is_stable = entropy_is_stable
        else:
            self.last_entropy_tolerance = None
            self.last_entropy_is_stable = True

        enabled_stability_flags = []
        if use_reward_stagnation:
            enabled_stability_flags.append(reward_is_stable)
        if use_entropy_stagnation:
            enabled_stability_flags.append(entropy_is_stable)
        self.last_combined_is_stable = all(enabled_stability_flags) if enabled_stability_flags else False
        self.last_within_warmup = self.num_checks <= self.warmup_checks

        # Slope checks use EMA-smoothed signals to avoid reacting to noisy rollouts.
        smoothed_reward = float(self.ema_reward if self.ema_reward is not None else current_reward)
        self.reward_history.append(smoothed_reward)
        if current_entropy is not None:
            smoothed_entropy = float(
                self.ema_entropy if self.ema_entropy is not None else current_entropy
            )
            self.entropy_history.append(smoothed_entropy)
        active_signal_values = None
        active_scale = None
        if use_entropy_stagnation and current_entropy is not None:
            active_signal_values = np.asarray(self.entropy_history, dtype=np.float64)
            active_scale = float(self.ema_abs_entropy if self.ema_abs_entropy is not None else abs(current_entropy))
        elif use_reward_stagnation:
            active_signal_values = np.asarray(self.reward_history, dtype=np.float64)
            active_scale = float(self.ema_abs_reward if self.ema_abs_reward is not None else abs(current_reward))

        slope_ready = bool(active_signal_values is not None and active_signal_values.size >= self.slope_window)
        signal_slope = None
        signal_slope_normalized = None
        slope_is_flat = False
        if slope_ready:
            signal_slope, signal_slope_normalized = self._compute_normalized_slope(
                active_signal_values, scale=active_scale
            )
            slope_is_flat = signal_slope_normalized <= self.slope_tolerance

        self.last_signal_slope = signal_slope
        self.last_signal_slope_normalized = signal_slope_normalized
        self.last_signal_slope_is_flat = slope_is_flat if slope_ready else False

        if not self.last_within_warmup:
            if (
                self.use_slope_early_stop
                and slope_ready
                and self.num_checks >= self.min_slope_checks
                and slope_is_flat
            ):
                return True
            if self.last_combined_is_stable:
                self.stable_checks += 1
            else:
                self.stable_checks = 0

            if self.stable_checks >= self.patience:
                return True

        return False

# Example Usage in your Training Loop:
# tracker = BRConvergenceTracker(patience=5, tolerance=1e-5)
# for step in range(MAX_STEPS):
#     update_followers()
#     metric = calculate_nash_conv() # or sum of grad norms
#     if tracker.check(metric):
#         break