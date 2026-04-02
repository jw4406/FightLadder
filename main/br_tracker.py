import numpy as np
from collections import deque

class BRConvergenceTracker:
    def __init__(self, patience=10, tolerance=1e-4, window_size=50):
        """
        Args:
            patience: How many 'checks' to wait without improvement before stopping.
            tolerance: The threshold for 'zero' improvement or absolute convergence.
            window_size: Number of steps to average over to smooth out minimax oscillations.
        """
        self.patience = patience
        self.tolerance = tolerance
        self.window_size = window_size
        
        self.history = deque(maxlen=window_size)
        self.best_metric = float('inf')
        self.wait_count = 0
        self.steps_trained = 0

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

# Example Usage in your Training Loop:
# tracker = BRConvergenceTracker(patience=5, tolerance=1e-5)
# for step in range(MAX_STEPS):
#     update_followers()
#     metric = calculate_nash_conv() # or sum of grad norms
#     if tracker.check(metric):
#         break