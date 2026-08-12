"""Option B: does the head fit the MINIMAX fixed point, or just something
self-consistent?

    target = r + gamma * V_mm(s') * (1 - done),   V_mm = value of Q(s',.,.)

THE TEST THAT MATTERS is `minimax fixed point, not on-policy`. Every other check
here -- gamma=0, done, frozen target -- would also pass on an implementation
that quietly solved the wrong game, or that averaged over the opponent instead
of minimising over it. So the core case builds a one-state repeated game where
the minimax value and the on-policy value are FAR APART and analytically known,
and asserts the head lands on the minimax one.

THE SUCCESSOR MATRIX MUST BE PINNED. The first version of this test left the
head free and asserted Q converged to r. It did not -- it converged to
r/(1-gamma), which is simply the correct value of an infinitely repeated
self-loop, so the assertion was wrong rather than the code. Worse, that setup
could never have separated the two hypotheses: a free head converges to a
CONSTANT matrix, and max-min of a constant equals its mean, so an implementation
that averaged over the adversary would have passed identically.

So V_mm(s') is pinned to a known matrix chosen with max-min far from the mean:

      [[ 0, 10],        max-min = 0     (adv always takes column 0)
       [ 0, 10]]        mean    = 5.0

      [[ 1,  1],        max-min = 1     (row 0 guarantees 1; row 1 risks 0
       [ 0, 20]]        mean    = 5.5    for a 20 the adversary will never give)

At gamma 0.94 and r 0.10 the two predictions are 0.10 vs 4.80, and 1.04 vs 5.27.
A correct backup and an averaging one are nowhere near each other, and the
second case has a NONZERO value so "converges to zero" cannot pass by accident.

WHY NOT ASSERT ON `target` DIRECTLY. Reaching in to check the local variable
would pass even if the sign were applied to the wrong quantity downstream, which
is exactly the failure mode that produced the ego-pass frame bug. These run the
real _minimax_q_update to convergence and check where Q ends up.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as th
import torch.nn as nn

from stable_baselines3.common.clean_new_policies import MinimaxHead
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR

LATENT, B = 16, 64
OK = True


class FixedMatrix(nn.Module):
    """A stand-in for the frozen target head that emits a KNOWN payoff matrix.

    Pinning the successor is what makes the core test able to fail: left free,
    Q converges to a constant matrix, and max-min of a constant equals its mean,
    so a backup that averaged over the adversary instead of minimising over it
    would be indistinguishable from a correct one.
    """

    def __init__(self, M):
        super().__init__()
        self.register_buffer("M", M)

    def forward(self, latent):
        return self.M.unsqueeze(0).expand(latent.shape[0], *self.M.shape)


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def build(n_ego, n_adv, gamma, target_mode="minimax", seed=0):
    """A minimal agent exposing exactly what _minimax_q_update touches."""
    th.manual_seed(seed)
    head = MinimaxHead(nn.Sequential(nn.Linear(LATENT, LATENT), nn.LeakyReLU()),
                       LATENT, n_ego=n_ego, n_adv=n_adv)
    latent = th.randn(1, LATENT).repeat(B, 1)      # ONE state, repeated

    policy = types.SimpleNamespace(
        minimax_head_for=lambda buf_num: head,
        minimax_matrices=lambda obs, buf_num=None, side_flag=None, stop_grad=True: head(latent),
        minimax_latent=lambda obs, side_flag=None: latent,
        minimax_optimizer=th.optim.Adam(head.parameters(), lr=3e-3),
    )
    agent = types.SimpleNamespace(
        minimax_q=True, policy=policy, minimax_stop_grad=True,
        minimax_stat_every=10**9, minimax_iters=96, minimax_eta=0.5,
        _mm_calls=0, gamma=gamma, minimax_target=target_mode, num_timesteps=0,
        _minimax_frozen_head=None)
    # bind the real frozen-head helper
    agent._minimax_frozen_head = types.MethodType(
        CleanDerivativeFreeSPAR._minimax_frozen_head, agent)
    return agent, head, latent


def run(agent, head, n_ego, n_adv, reward, steps=800, done=0.0, rollout_every=50):
    a_ego = th.randint(0, n_ego, (B,))
    a_adv = th.randint(0, n_adv, (B,))
    data = types.SimpleNamespace(
        observations=th.zeros(B, 1), actions=a_ego, adv_actions=a_adv,
        next_observations=th.zeros(B, 1), returns=th.zeros(B),
        rewards=th.full((B,), reward), dones=th.full((B,), done))
    for t in range(steps):
        # advance num_timesteps periodically so the frozen target refreshes,
        # exactly as it would once per rollout in training
        agent.num_timesteps = (t // rollout_every) + 1
        CleanDerivativeFreeSPAR._minimax_q_update(agent, data, [0], adv_frame=False)
    return data


def main():
    # ---- 1. THE CORE CASE: minimax fixed point, not on-policy ------------
    # PIN the successor's payoff matrix, so V_mm(s') is known analytically and
    # is FAR from every other summary of the same matrix. Without pinning, Q
    # converges to a constant matrix -- and max-min of a constant equals its
    # mean, so the test could not tell a minimax backup from an averaging one.
    gamma, r = 0.94, 0.10
    for name, Mp, v_mm in (
            # adv always takes column 0, so the value is 0 while the mean is 5.0
            ("zero-value", th.tensor([[0.0, 10.0], [0.0, 10.0]]), 0.0),
            # row 0 guarantees 1; row 1 risks 0 for a 20 that adv will never give
            ("nonzero-value", th.tensor([[1.0, 1.0], [0.0, 20.0]]), 1.0)):
        agent, head, latent = build(2, 2, gamma)
        agent._minimax_frozen_head = lambda buf_num, live, _M=Mp: FixedMatrix(_M)
        run(agent, head, 2, 2, reward=r, steps=900)
        with th.no_grad():
            q = float(head(latent)[0].mean())
        want = r + gamma * v_mm
        avg = r + gamma * float(Mp.mean())
        check(f"minimax fixed point, NOT on-policy [{name}]",
              abs(q - want) < 0.15 * abs(avg - want),
              f"Q={q:+.4f}  minimax r+g*V_mm={want:+.4f}  "
              f"an AVERAGING head would sit at {avg:+.4f}")

    # ---- 2. gamma = 0 collapses the target to the immediate reward -------
    agent, head, latent = build(4, 4, gamma=0.0)
    run(agent, head, 4, 4, reward=0.07, steps=700)
    with th.no_grad():
        q0 = float(head(latent)[0].mean())
    check("gamma=0 -> Q converges to r exactly",
          abs(q0 - 0.07) < 0.01, f"Q={q0:+.5f} vs r=+0.07000")

    # ---- 3. done cuts the bootstrap --------------------------------------
    agent, head, latent = build(4, 4, gamma=0.94)
    run(agent, head, 4, 4, reward=0.05, steps=700, done=1.0)
    with th.no_grad():
        qd = float(head(latent)[0].mean())
    check("done=1 cuts the bootstrap -> Q converges to r",
          abs(qd - 0.05) < 0.01, f"Q={qd:+.5f} vs r=+0.05000")

    # ---- 4. the target head is genuinely frozen within a rollout ---------
    # Drive the update DIRECTLY rather than via run(), which rewrites
    # num_timesteps and would refresh the snapshot underneath the assertion.
    agent, head, latent = build(4, 4, gamma=0.94)
    agent.num_timesteps = 7
    f1 = agent._minimax_frozen_head([0], head)
    before = next(iter(f1.parameters())).clone()
    live_before = next(iter(head.parameters())).clone()
    a_ego = th.randint(0, 4, (B,)); a_adv = th.randint(0, 4, (B,))
    d4 = types.SimpleNamespace(
        observations=th.zeros(B, 1), actions=a_ego, adv_actions=a_adv,
        next_observations=th.zeros(B, 1), returns=th.zeros(B),
        rewards=th.full((B,), 0.05), dones=th.zeros(B))
    for _ in range(60):
        CleanDerivativeFreeSPAR._minimax_q_update(agent, d4, [0], adv_frame=False)
    f2 = agent._minimax_frozen_head([0], head)
    after = next(iter(f2.parameters()))
    live_after = next(iter(head.parameters()))
    check("frozen target is stable WHILE the live head moves",
          f1 is f2 and th.allclose(before, after)
          and not th.allclose(live_before, live_after),
          f"frozen unchanged, live moved by "
          f"{float((live_after - live_before).abs().max()):.2e}")
    agent.num_timesteps = 8
    f3 = agent._minimax_frozen_head([0], head)
    check("frozen target REFRESHES on a new rollout", f3 is not f2)

    # ---- 5. missing next_observations must raise, not degrade ------------
    agent, head, _ = build(4, 4, gamma=0.94)
    bad = types.SimpleNamespace(
        observations=th.zeros(B, 1), actions=th.zeros(B, dtype=th.long),
        adv_actions=th.zeros(B, dtype=th.long), returns=th.zeros(B),
        rewards=th.zeros(B), dones=th.zeros(B))
    try:
        CleanDerivativeFreeSPAR._minimax_q_update(agent, bad, [0], adv_frame=False)
        raised = False
    except RuntimeError:
        raised = True
    check("no next_observations -> RuntimeError (no silent fallback)", raised)

    # ---- 6. default is still option A, and it is unchanged ---------------
    agent, head, latent = build(4, 4, gamma=0.94, target_mode="returns")
    a_ego = th.randint(0, 4, (B,)); a_adv = th.randint(0, 4, (B,))
    data = types.SimpleNamespace(
        observations=th.zeros(B, 1), actions=a_ego, adv_actions=a_adv,
        next_observations=th.zeros(B, 1), returns=th.full((B,), 0.09),
        rewards=th.full((B,), -99.0), dones=th.zeros(B))   # reward is a trap
    for _ in range(700):
        CleanDerivativeFreeSPAR._minimax_q_update(agent, data, [0], adv_frame=False)
    with th.no_grad():
        qa = float(head(latent)[0].mean())
    check("target_mode='returns' still fits the RETURN, ignoring r",
          abs(qa - 0.09) < 0.01, f"Q={qa:+.5f} vs returns=+0.09000")

    print("\n  ALL PASS" if OK else "\n  FAILURES PRESENT")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
