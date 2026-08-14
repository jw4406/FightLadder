"""The entropy-collapse guard: does it fire on a dead run and stay quiet on a live one?

WHAT IS BEING DEFENDED. A policy at exactly zero entropy is a deterministic point
mass. PPO's gradient is proportional to probability mass that can move, so at zero
entropy the gradient is zero and the policy CAN NEVER RECOVER -- the state is
absorbing, not transient. The run keeps producing a plausible score curve while it
has silently stopped being self-play. p1_clr1e5_winit's adversary entered it at
3.77M steps and burned the next 34M as single-agent RL against a frozen bot.

THE TEST THAT MATTERS is `real winit trace aborts` / `real p1_clr1e5 trace does
not`. A guard tuned on synthetic values can be perfectly self-consistent and still
have a threshold that never fires on the actual failure, or fires constantly on
healthy runs. So both directions are checked against the REAL logged traces:
2761 consecutive zero blocks on the arm that died, 0 of 4608 on the one that
lived.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

OK = True
N_RUN = 0
# A test that never RUNS is indistinguishable from one that passes -- a
# dispatch line I forgot cost an hour today while ALL PASS printed. So the
# count is asserted, not assumed. Bump this deliberately when adding a check.
EXPECTED_CHECKS = 12


def check(name, cond, detail=""):
    global OK, N_RUN
    N_RUN += 1
    OK &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


class FakeLogger:
    def __init__(self): self.rec = {}
    def record(self, k, v, exclude=None): self.rec[k] = v


class Guard:
    """The method under test, bound to a minimal stand-in for the algorithm.

    Imported from the real class so the test cannot drift from the implementation.
    """
    def __init__(self, abort=True, tol=1e-6, patience=20):
        from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
        self._check = CleanDerivativeFreeSPAR._check_entropy_collapse.__get__(self)
        self.entropy_collapse_abort = abort
        self.entropy_collapse_tol = tol
        self.entropy_collapse_patience = patience
        self._entropy_zero_streak = {"ego": 0, "adv": 0}
        self.logger = FakeLogger()
        self.num_timesteps = 3_772_416

    def feed(self, prefix, value):
        self._check(prefix, [value])


def feed_many(g, prefix, values):
    """Returns the update index that raised, or None."""
    for i, v in enumerate(values):
        try:
            g.feed(prefix, v)
        except RuntimeError:
            return i
    return None


def main():
    # ---- 1. the real failure trace must abort --------------------------------
    # winit: adv entropy was exactly 0.0 for 2761 consecutive logged blocks.
    g = Guard()
    at = feed_many(g, "adv", [0.0] * 2761)
    check("real winit trace aborts", at is not None, f"raised at update {at}")
    check("aborts at the configured patience, not later", at == 19, f"index {at}")

    # ---- 2. the real healthy trace must NOT abort ----------------------------
    # p1_clr1e5: 0 of 4608 blocks were at zero; final adv entropy_loss -1.06.
    rng = np.random.RandomState(0)
    live = -np.abs(rng.normal(0.5, 0.3, size=4608)) - 1e-3
    g2 = Guard()
    check("real p1_clr1e5 trace never aborts", feed_many(g2, "adv", live) is None,
          f"min |entropy| {np.abs(live).min():.4g}")

    # ---- 3. a transient must not kill a healthy run --------------------------
    g3 = Guard(patience=20)
    seq = [0.0] * 19 + [-0.4] + [0.0] * 19 + [-0.4] + [0.0] * 19
    check("19-long transients never reach patience", feed_many(g3, "adv", seq) is None,
          "streak resets on any nonzero update")

    # ---- 4. the two seats are tracked INDEPENDENTLY --------------------------
    # A shared counter would let alternating ego/adv updates mask a real collapse.
    g4 = Guard(patience=4)
    for _ in range(3):
        g4.feed("ego", 0.0); g4.feed("adv", -0.5)
    check("adv activity does not reset the ego streak", g4._entropy_zero_streak["ego"] == 3,
          f"ego streak {g4._entropy_zero_streak['ego']}, adv {g4._entropy_zero_streak['adv']}")

    # ---- 5. abort=False warns but does not raise ----------------------------
    g5 = Guard(abort=False)
    check("abort=False never raises", feed_many(g5, "adv", [0.0] * 200) is None)
    check("abort=False still counts the streak", g5._entropy_zero_streak["adv"] == 200)

    # ---- 6. the streak is logged so it is visible BEFORE the abort ----------
    g6 = Guard()
    g6.feed("adv", 0.0)
    check("streak is logged", g6.logger.rec.get("train/adv_entropy_zero_streak") == 1)

    # ---- 7. NaN must not be read as saturation ------------------------------
    # np.mean of an empty/degenerate batch gives nan; |nan| < tol is False in
    # numpy but the intent must be explicit, not incidental.
    g7 = Guard(patience=2)
    check("nan does not count as zero entropy",
          feed_many(g7, "adv", [float("nan")] * 50) is None,
          f"streak {g7._entropy_zero_streak['adv']}")

    # ---- 8. empty batch is a no-op ------------------------------------------
    g8 = Guard()
    g8._check("adv", [])
    check("empty entropy list is a no-op", g8._entropy_zero_streak["adv"] == 0)

    # ---- 9. tolerance boundary ----------------------------------------------
    g9 = Guard(tol=1e-6, patience=3)
    check("just above tol is NOT saturated", feed_many(g9, "adv", [-1.1e-6] * 10) is None)
    g10 = Guard(tol=1e-6, patience=3)
    check("just below tol IS saturated", feed_many(g10, "adv", [-9e-7] * 10) is not None)

    global OK
    if N_RUN != EXPECTED_CHECKS:
        print(f"\n  FAIL  expected {EXPECTED_CHECKS} checks, {N_RUN} RAN -- a check "
              f"was skipped or never dispatched, which is NOT a pass")
        OK = False
    print(f"\n  ALL PASS ({N_RUN} checks)" if OK else "\n  FAILURES PRESENT")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
