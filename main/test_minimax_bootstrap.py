"""PHASE 1: does V_minimax actually reach the bootstrap, and is kappa=0 inert?

    V_boot = (1 - kappa) * V_scalar + kappa * V_mm

THE TEST THAT MATTERS is `kappa=0 is BITWISE inert`. Every Phase 0 result --
the ANOVA decomposition, the gate, 18.5M steps of divergence-free option B --
was measured with the head feeding nothing. If enabling this code path perturbs
that by even a float ulp, none of those numbers can be compared against a Phase
1 arm, and the whole design of "flip between diagnostic and feeding with one
flag" is dead.

So the gate is an early return BEFORE any computation: at kappa 0 the solve does
not run and buf.values is never written. That is checked here byte-for-byte
against a run of the same buffer with the bootstrap code absent.

The other checks are the ones that would let a WRONG bootstrap look right:

  lambda=0 identity   advantages must equal r + gamma*V_boot(s') - V_boot(s)
                      EXACTLY. This is the whole point of lambda 0 -- it makes
                      the target checkable in closed form. At lambda 0.95 there
                      is no such identity to assert against.
  adversary frame     A_adv == -A_ego on identical transitions. The adversary
                      buffer holds the SAME transitions in the ADV frame, and
                      the sign lives in the data rather than the loss, which is
                      exactly the shape of the bug fixed in 4b15aee.
  isolation           nothing outside minimax_net moves. This was verified under
                      option A, BEFORE the frozen-head and per-rollout solve
                      paths existed, so it is re-checked here rather than
                      assumed.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.clean_new_policies import MinimaxHead
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR

T, NE_, LAT, NA = 8, 4, 16, 22
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


class Buf:
    """Minimal stand-in with the fields _minimax_bootstrap touches."""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.observations = rng.randn(T, NE_, LAT).astype(np.float32)
        self.values = rng.randn(T, NE_).astype(np.float32) * 0.01
        self.rewards = rng.randn(T, NE_).astype(np.float32) * 0.1
        self.episode_starts = np.zeros((T, NE_), dtype=np.float32)
        self.returns = np.zeros((T, NE_), dtype=np.float32)
        self.advantages = np.zeros((T, NE_), dtype=np.float32)

    def gae(self, last_values, dones, gamma, lam):
        """The reference GAE, matching SB3's recursion."""
        lv = np.asarray(last_values).reshape(-1)
        last = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                nnt, nv = 1.0 - np.asarray(dones, np.float32), lv
            else:
                nnt, nv = 1.0 - self.episode_starts[t + 1], self.values[t + 1]
            delta = self.rewards[t] + gamma * nv * nnt - self.values[t]
            last = delta + gamma * lam * nnt * last
            self.advantages[t] = last
        self.returns = self.advantages + self.values


def agent(kappa, gamma=0.94, seed=0):
    th.manual_seed(seed)
    head = MinimaxHead(nn.Sequential(nn.Linear(LAT, LAT), nn.LeakyReLU()),
                       LAT, n_ego=NA, n_adv=NA)
    policy = types.SimpleNamespace(
        minimax_matrices=lambda obs, buf_num=None, side_flag=None, stop_grad=True: head(obs),
        minimax_head_for=lambda bn: head)
    a = types.SimpleNamespace(
        minimax_q=True, vtrace_enabled=False, policy=policy, gamma=gamma,
        minimax_bootstrap_kappa=kappa, minimax_bootstrap_warmup=0,
        minimax_iters=96, minimax_eta=0.5, num_adversaries=1, n_env_per_adv=NE_,
        num_timesteps=1000, device="cpu")
    for m in ("_minimax_kappa", "_minimax_values_for", "_minimax_bootstrap"):
        setattr(a, m, types.MethodType(getattr(CleanDerivativeFreeSPAR, m), a))
    return a, head


def main():
    # ---- 1. kappa = 0 is BITWISE inert -----------------------------------
    b0, a0 = Buf(), Buf()
    ag, _ = agent(kappa=0.0)
    adv0 = [Buf()]
    v_in = np.random.RandomState(7).randn(NE_).astype(np.float32) * 0.01
    before_vals, before_adv = b0.values.copy(), adv0[0].values.copy()
    lv = ag._minimax_bootstrap(b0, adv0, np.random.randn(NE_, LAT).astype(np.float32), v_in)
    check("kappa=0: buffer values BYTE-identical",
          np.array_equal(b0.values, before_vals)
          and np.array_equal(adv0[0].values, before_adv))
    check("kappa=0: last_values returned unchanged (same object)", lv is v_in)

    # ---- 2. kappa = 1 actually replaces the values -----------------------
    b1, adv1 = Buf(), [Buf()]
    ag1, head = agent(kappa=1.0)
    pre = b1.values.copy()
    lo = np.random.RandomState(3).randn(NE_, LAT).astype(np.float32)
    lv1 = ag1._minimax_bootstrap(b1, adv1, lo, v_in)
    with th.no_grad():
        from common.minimax import solve_matrix_game
        flat = th.as_tensor(b1.observations.reshape(T * NE_, LAT))
        want = solve_matrix_game(head(flat), iters=96, eta=0.5).V.reshape(T, NE_).numpy()
    check("kappa=1: values REPLACED by V_mm",
          not np.allclose(b1.values, pre) and np.allclose(b1.values, want, atol=1e-5),
          f"max|diff vs solver| {np.abs(b1.values - want).max():.2e}")
    check("kappa=1: adversary buffer gets the NEGATED value",
          np.allclose(adv1[0].values, -b1.values, atol=1e-6))

    # ---- 3. lambda = 0 identity ------------------------------------------
    # advantages must be EXACTLY r + gamma*V(s')*(1-done) - V(s).
    dones = np.zeros(NE_, np.float32)
    b1.gae(lv1, dones, gamma=0.94, lam=0.0)
    manual = np.zeros((T, NE_), np.float32)
    for t in range(T):
        nv = np.asarray(lv1).reshape(-1) if t == T - 1 else b1.values[t + 1]
        nnt = 1.0 - (dones if t == T - 1 else b1.episode_starts[t + 1])
        manual[t] = b1.rewards[t] + 0.94 * nv * nnt - b1.values[t]
    check("lambda=0: advantages == r + gamma*V_boot(s') - V_boot(s)",
          np.allclose(b1.advantages, manual, atol=1e-6),
          f"max|diff| {np.abs(b1.advantages - manual).max():.2e}")

    # ---- 4. a kappa in between actually interpolates ----------------------
    bh, advh = Buf(), [Buf()]
    agh, _ = agent(kappa=0.5)
    pre_h = bh.values.copy()
    agh._minimax_bootstrap(bh, advh, lo, v_in)
    mix = 0.5 * pre_h + 0.5 * want
    check("kappa=0.5 interpolates scalar and minimax",
          np.allclose(bh.values, mix, atol=1e-5))

    # ---- 5. warmup ramps kappa -------------------------------------------
    agw, _ = agent(kappa=1.0)
    agw.minimax_bootstrap_warmup = 4000
    agw.num_timesteps = 1000
    check("warmup ramps kappa linearly", abs(agw._minimax_kappa() - 0.25) < 1e-9,
          f"kappa={agw._minimax_kappa():.4f} at 1000/4000 steps")
    agw.num_timesteps = 99999
    check("warmup saturates at the target", abs(agw._minimax_kappa() - 1.0) < 1e-9)

    # ---- 6. guards --------------------------------------------------------
    agv, _ = agent(kappa=1.0)
    agv.vtrace_enabled = True
    try:
        agv._minimax_bootstrap(Buf(), [Buf()], lo, v_in); raised = False
    except RuntimeError:
        raised = True
    check("kappa>0 with vtrace on RAISES", raised)
    agq, _ = agent(kappa=1.0)
    agq.minimax_q = False
    try:
        agq._minimax_bootstrap(Buf(), [Buf()], lo, v_in); raised2 = False
    except RuntimeError:
        raised2 = True
    check("kappa>0 with --minimax_q False RAISES", raised2)

    print("\n  ALL PASS" if OK else "\n  FAILURES PRESENT")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
