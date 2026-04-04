import numpy as np
from collections import deque

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
        
        self.history = deque(maxlen=window_size)
        self.best_metric = float('inf')
        self.wait_count = 0
        self.steps_trained = 0

        # EMA reward-stability state
        self.ema_reward = None
        self.ema_abs_reward = None
        self.ema_abs_dev = None
        self.num_checks = 0
        self.stable_checks = 0

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
        print("Checking for convergence...")    
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
            print(f"--- Early Stopping triggered at step {self.steps_trained} (Stagnation) ---")
            return True
            
        return False

    def check_reward_stability(self, current_reward):
        """
        Returns True when reward has stabilized according to EMA absolute deviation.
        """
        self.num_checks += 1
        if self.ema_reward is None:
            self.ema_reward = float(current_reward)
            self.ema_abs_reward = abs(float(current_reward))
            self.ema_abs_dev = 0.0
            return False

        self.ema_reward = self.ema_beta * self.ema_reward + (1.0 - self.ema_beta) * float(current_reward)
        self.ema_abs_reward = self.ema_beta * self.ema_abs_reward + (1.0 - self.ema_beta) * abs(float(current_reward))
        abs_dev = abs(float(current_reward) - self.ema_reward)
        self.ema_abs_dev = self.ema_beta * self.ema_abs_dev + (1.0 - self.ema_beta) * abs_dev

        dynamic_tolerance = self.rel_tolerance * max(self.ema_abs_reward, self.eps)
        if self.num_checks > self.warmup_checks:
            if self.ema_abs_dev <= dynamic_tolerance:
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