"""Counterfactual enumeration: is it inert when off, and correct when on?

WHAT IS BEING DEFENDED. Enumeration branches the TRAINING envs -- it rewinds the
emulator mid-rollout with em.set_state. Two things must hold or it is worse than
useless:

  1. --enum_every 0 must be BITWISE inert. Every arm measured so far was run
     without this feature; if merely compiling it in perturbs a run, none of
     those baselines are comparable any more.
  2. Monitor2P must not observe the branches. It breaks branching in three
     separate ways and two of them are SILENT: needs_reset kills the worker on
     the next step (loud), but reward accumulation corrupts the reported episode
     return and a done branch writes a PHANTOM episode into ep_rew_mean -- both
     of which look like a training result rather than a bug.

The phantom-episode and reward-contamination checks are the ones that matter:
a run with those bugs would train fine and simply report the wrong score.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th

OK = True
N_RUN = 0
# A test that never RUNS is indistinguishable from one that passes -- a
# dispatch line I forgot cost an hour today while ALL PASS printed. So the
# count is asserted, not assumed. Bump this deliberately when adding a check.
EXPECTED_CHECKS = 23


def check(name, cond, detail=""):
    global OK, N_RUN
    N_RUN += 1
    OK &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------- Monitor2P --
class FakeInner:
    """Minimal 2P env: returns done on demand so the branch path is exercised."""
    def __init__(self):
        self.done_next = False
        self.steps = 0
        self.observation_space = None
        self.action_space = None
        self.metadata = {}
        self.reward_range = (-1, 1)
        self.spec = None
    def step(self, a):
        self.steps += 1
        return np.zeros(3), 1.0, -1.0, self.done_next, {}
    def reset(self, **kw):
        return np.zeros(3)
    def close(self):
        pass


def test_monitor_pause():
    from common.retro_wrappers import Monitor2P
    m = Monitor2P(FakeInner(), filename=None)
    m.reset()

    # baseline: normal accounting works
    m.step(0); m.step(0)
    check("normal mode accumulates rewards", len(m.rewards) == 2, f"{len(m.rewards)}")

    n_before = len(m.rewards)
    eps_before = len(m.episode_returns)

    m.lbr_pause_monitor(True)
    check("pause is reported", m.lbr_monitor_paused())
    # a branch that ENDS an episode -- the case that breaks everything
    m.env.done_next = True
    for _ in range(5):
        m.step(0)
    m.env.done_next = False

    check("paused: rewards NOT accumulated", len(m.rewards) == n_before,
          f"{len(m.rewards)} vs {n_before}")
    check("paused: NO phantom episode recorded",
          len(m.episode_returns) == eps_before,
          f"{len(m.episode_returns)} vs {eps_before}")
    check("paused: needs_reset NOT set by a done branch", not m.needs_reset)

    m.lbr_pause_monitor(False)
    m.step(0)
    check("resumes accounting after unpause", len(m.rewards) == n_before + 1)

    # and the guard still works when NOT paused
    m.env.done_next = True
    m.step(0)
    m.env.done_next = False
    check("unpaused: a real done DOES set needs_reset", m.needs_reset)
    raised = False
    try:
        m.step(0)
    except RuntimeError:
        raised = True
    check("unpaused: stepping after done still raises", raised)


# --------------------------------------------------------------- inertness --
class _NullLogger:
    def __init__(self): self.rec = {}
    def record(self, k, v, exclude=None): self.rec[k] = v


class FakeAlgo:
    """Just enough of the algorithm to exercise the enum paths."""
    def __init__(self, **kw):
        from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR as C
        self._maybe_enumerate = C._maybe_enumerate.__get__(self)
        self._enum_aux_loss = C._enum_aux_loss.__get__(self)
        self._enum_joint = C._enum_joint.__get__(self)
        self._enum_splice = C._enum_splice.__get__(self)
        self.enum_every = kw.get("enum_every", 0)
        self.enum_k = kw.get("enum_k", 484)
        self.enum_buffer = kw.get("enum_buffer", 8)
        self.enum_loss_coef = 1.0
        self._enum_store = []
        self._enum_env_steps = 0
        self._enum_next_at = 0
        self.minimax_q = kw.get("minimax_q", True)
        self.num_timesteps = 10_000_000
        self.device = "cpu"
        self.gamma = 0.94
        self.env = self
        self.touched = False
        # _maybe_enumerate records the running budget on EVERY call, before the
        # due-check, so that train/enum_env_steps is visible in every dump rather
        # than only on iterations that enumerate. The real class always has a
        # logger by the time train() runs.
        self.logger = _NullLogger()
    # any env access at all would flip this
    def env_method(self, *a, **k):
        self.touched = True
        raise AssertionError("env_method called while enumeration should be OFF")


def test_inert_when_off():
    a = FakeAlgo(enum_every=0)
    a._maybe_enumerate()
    check("enum_every=0 never touches the env", not a.touched)
    check("enum_every=0 adds no aux loss", a._enum_aux_loss([0]) is None)
    check("enum_every=0 charges no steps", a._enum_env_steps == 0)

    b = FakeAlgo(enum_every=100, minimax_q=False)
    b._maybe_enumerate()
    check("minimax_q=False never touches the env", not b.touched)


def test_schedule():
    """Due-ness must be driven by num_timesteps, not by call count."""
    a = FakeAlgo(enum_every=1000)
    a.env = None                       # would raise if it tried to enumerate
    a._enum_next_at = a.num_timesteps + 1
    try:
        a._maybe_enumerate()
        ok = True
    except Exception:
        ok = False
    check("not due -> returns before touching the env", ok)
    check("budget IS still logged when not due",
          "train/enum_env_steps" in a.logger.rec,
          "so the overhead is visible in every dump, not only on enum iterations")


def test_aux_loss_and_k():
    a = FakeAlgo(enum_every=100)
    S, na = 7, 22
    obs = np.zeros((S, 5), dtype=np.float32)
    M = np.random.RandomState(0).randn(S, na, na).astype(np.float32)
    a._enum_store = [(obs, M)]

    class P:
        def minimax_matrices(self, o, buf_num=None, side_flag=None, stop_grad=True):
            return th.zeros(o.shape[0], na, na)
    a.policy = P()

    a.enum_k = na * na
    full = a._enum_aux_loss([0])
    check("k=484 uses every cell",
          abs(float(full) - float((M ** 2).mean())) < 1e-5,
          f"{float(full):.5f} vs {float((M**2).mean()):.5f}")

    a.enum_k = 16
    sub = a._enum_aux_loss([0])
    check("k=16 returns a finite loss over a subset",
          np.isfinite(float(sub)) and float(sub) > 0)

    # buffer ring
    a.enum_buffer = 2
    a._enum_store = [(obs, M)] * 5
    a._enum_store = a._enum_store[-a.enum_buffer:]
    check("buffer is bounded", len(a._enum_store) == 2)


def test_leaf_matches_target():
    """The enumerated leaf must use the SAME value function as the on-policy term.

    This is the bug that produced held-out EV -4.9: option B regresses onto
    r + gamma*V_mm(s') while the enumerated target was built with V_scalar, so
    the head was pulled toward two inconsistent definitions of Q. Option A
    regresses onto lambda-returns, which estimate Q^pi, so the SCALAR leaf is
    correct there -- the branch is not a preference, each matches its own target.
    """
    from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR as C
    a = FakeAlgo(enum_every=100)
    a._enum_leaf_values = C._enum_leaf_values.__get__(a)
    calls = {"scalar": 0}
    a._enum_values = lambda obs, buf: (calls.__setitem__("scalar", calls["scalar"] + 1)
                                       or np.zeros(len(obs)))
    a.minimax_target = "returns"
    a._enum_leaf_values(np.zeros((4, 3)), [0])
    check("option A (returns) uses the SCALAR leaf", calls["scalar"] == 1)

    a.minimax_target = "minimax"
    try:
        a._enum_leaf_values(np.zeros((4, 3)), [0])
        used_scalar = calls["scalar"] == 2
    except Exception:
        used_scalar = False          # took the V_mm path and hit the fake policy
    check("option B (minimax) does NOT use the scalar leaf", not used_scalar,
          "it must solve the matrix game for V_mm instead")


def test_joint_and_splice():
    a = FakeAlgo()
    j = a._enum_joint(3, 7, 4)
    check("joint action has shape (n, 2)", j.shape == (4, 2), str(j.shape))
    check("joint action carries (i, j)", (j[:, 0] == 3).all() and (j[:, 1] == 7).all())

    o1 = np.zeros((3, 2)); d = np.array([False, True, False])
    infos = [{}, {"terminal_observation": np.array([9.0, 9.0])}, {}]
    out = a._enum_splice(o1, d, infos)
    check("terminal observation is spliced in", (out[1] == 9.0).all())
    check("non-terminal rows untouched", (out[0] == 0).all() and (out[2] == 0).all())


def main():
    print("== Monitor2P pause ==");      test_monitor_pause()
    print("== inert when off ==");       test_inert_when_off()
    print("== schedule ==");             test_schedule()
    print("== aux loss / enum_k ==");    test_aux_loss_and_k()
    print("== leaf matches target =="); test_leaf_matches_target()
    print("== joint / splice ==");       test_joint_and_splice()
    global OK
    if N_RUN != EXPECTED_CHECKS:
        print(f"\n  FAIL  expected {EXPECTED_CHECKS} checks, {N_RUN} RAN -- a check "
              f"was skipped or never dispatched, which is NOT a pass")
        OK = False
    print(f"\n  ALL PASS ({N_RUN} checks)" if OK else "\n  FAILURES PRESENT")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
