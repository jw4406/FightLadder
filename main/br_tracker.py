import numpy as np
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
        self.metric_history = deque(maxlen=self.slope_window)
        self.last_metric_slope = None
        self.last_metric_slope_normalized = None
        self.last_slope_is_flat = None

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
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        ratings = np.asarray(ratings, dtype=np.float64)
        if not self.enabled():
            return False, None
        if ratings.size == 0:
            return False, None
        if self.games_since_eval < self.eval_games:
            return False, None

        rating_movements = np.abs(ratings - self.last_eval_ratings)
        mean_velocity = float(np.mean(rating_movements))
        entropy_loss = float(current_entropy) if current_entropy is not None else 0.0
        velocity_component = mean_velocity if self.use_velocity_signal else 0.0
        entropy_component = entropy_loss * self.entropy_weight if self.use_entropy_signal else 0.0
        metric = velocity_component + entropy_component

        self.num_checks += 1
        self.last_velocity = mean_velocity
        self.last_metric = metric

        if self.ema_metric is None:
            self.ema_metric = metric
            self.ema_abs_metric = abs(metric)
        else:
            beta = self.ema_beta
            self.ema_metric = beta * self.ema_metric + (1.0 - beta) * metric
            self.ema_abs_metric = beta * self.ema_abs_metric + (1.0 - beta) * abs(metric)

        dynamic_improvement_threshold = max(
            self.tolerance,
            self.rel_tolerance * max(self.ema_abs_metric, self.eps),
        )
        self.dynamic_threshold = dynamic_improvement_threshold
        # Use EMA-smoothed metric for slope detection to reduce rollout noise.
        slope_signal_value = float(self.ema_metric if self.ema_metric is not None else metric)
        self.metric_history.append(slope_signal_value)

        slope = None
        normalized_slope = None
        slope_ready = len(self.metric_history) >= self.slope_window
        slope_is_flat = False
        if slope_ready:
            slope, normalized_slope = self._compute_normalized_slope(
                np.asarray(self.metric_history, dtype=np.float64),
                scale=float(self.ema_abs_metric if self.ema_abs_metric is not None else metric),
            )
            slope_is_flat = normalized_slope <= self.slope_tolerance
        self.last_metric_slope = slope
        self.last_metric_slope_normalized = normalized_slope
        self.last_slope_is_flat = slope_is_flat if slope_ready else False

        if metric < (self.best_metric - dynamic_improvement_threshold):
            self.best_metric = metric
            self.wait_count = 0
        else:
            self.wait_count += 1
            if (
                lr_adjustment_callback is not None
                and self.lr_patience > 0
                and self.wait_count % self.lr_patience == 0
            ):
                lr_adjustment_callback()

        self.games_since_eval = 0
        self.last_eval_ratings = np.copy(ratings)

        logs = {
            "elo/stagnation/velocity": mean_velocity,
            "elo/stagnation/velocity_component": velocity_component,
            "elo/stagnation/metric": metric,
            "elo/stagnation/entropy": entropy_loss,
            "elo/stagnation/entropy_component": entropy_component,
            "elo/stagnation/use_velocity_stagnation": float(bool(self.use_velocity_signal)),
            "elo/stagnation/use_entropy_stagnation": float(bool(self.use_entropy_signal)),
            "elo/stagnation/threshold": dynamic_improvement_threshold,
            "elo/stagnation/wait_count": float(self.wait_count),
            "elo/stagnation/patience": float(self.patience),
            "elo/stagnation/num_checks": float(self.num_checks),
            "elo/stagnation/use_slope_early_stop": float(bool(self.use_slope_early_stop)),
            "elo/stagnation/slope_window": float(self.slope_window),
            "elo/stagnation/slope_tolerance": float(self.slope_tolerance),
            "elo/stagnation/min_slope_checks": float(self.min_slope_checks),
            "elo/stagnation/slope": float(slope) if slope is not None else float("nan"),
            "elo/stagnation/slope_normalized": (
                float(normalized_slope) if normalized_slope is not None else float("nan")
            ),
            "elo/stagnation/slope_is_flat": float(bool(self.last_slope_is_flat)),
        }
        if self.use_slope_early_stop:
            should_stop = bool(slope_ready and self.num_checks >= self.min_slope_checks and slope_is_flat)
        else:
            should_stop = self.wait_count >= self.patience
        return should_stop, logs

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