"""Tests for the batched zero-sum matrix-game solver.

Run:  python test_minimax.py

The load-bearing test is test_vs_scipy_lp: MWU is approximate, so it is only
trustworthy if it tracks an exact LP on random matrices. Everything else guards
a specific way the solver can be silently wrong.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th

from common.minimax import solve_matrix_game, duality_gap

N = 22  # the real action-space size: 9 directions + 7 attacks + 6 combos


def lp_value(M: np.ndarray) -> float:
    """Exact game value via LP:  max v  s.t.  (M^T p)_o >= v, sum p = 1, p >= 0."""
    from scipy.optimize import linprog
    n, m = M.shape
    c = np.concatenate([np.zeros(n), [-1.0]])            # maximize v
    A_ub = np.concatenate([-M.T, np.ones((m, 1))], axis=1)  # v - (M^T p)_o <= 0
    b_ub = np.zeros(m)
    A_eq = np.concatenate([np.ones((1, n)), [[0.0]]], axis=1)
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n + [(None, None)]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                method="highs")
    assert r.success, r.message
    return float(r.x[-1])


def test_matching_pennies():
    """V=0, both players uniform. The canonical case where the LAST MWU iterate
    cycles forever -- if averaging were off this would fail."""
    M = th.tensor([[1.0, -1.0], [-1.0, 1.0]])
    s = solve_matrix_game(M, iters=512)
    assert abs(float(s.V)) < 1e-2, f"V={float(s.V)}"
    assert th.allclose(s.p, th.full((2,), 0.5), atol=2e-2), s.p
    assert th.allclose(s.q, th.full((2,), 0.5), atol=2e-2), s.q
    print(f"   [1] matching pennies: V={float(s.V):+.4f} p={s.p.tolist()} "
          f"gap={float(s.gap):.2e}")


def test_dominant_row():
    """A row that dominates: ego should play it purely, V = that row's minimum."""
    M = th.tensor([[3.0, 4.0, 5.0],
                   [0.0, 1.0, 2.0],
                   [-1.0, 0.0, 1.0]])
    s = solve_matrix_game(M, iters=512)
    assert abs(float(s.V) - 3.0) < 1e-2, f"V={float(s.V)} != 3.0"
    assert float(s.p[0]) > 0.98, s.p
    print(f"   [2] dominant row: V={float(s.V):.4f} (exact 3.0), p0={float(s.p[0]):.4f}")


def test_vs_scipy_lp(n_games: int = 100):
    """Random 22x22 games against an exact LP. This is the real check.

    Reported as a distribution, not just a max: a single loose game among many
    tight ones means something different from a uniformly mediocre solve, and
    only the percentiles distinguish them.
    """
    rng = np.random.RandomState(0)
    Ms = rng.randn(n_games, N, N).astype(np.float64)
    exact = np.array([lp_value(M) for M in Ms])
    s = solve_matrix_game(th.tensor(Ms), iters=4096, eta=0.5)
    got = s.V.numpy()
    err = np.abs(got - exact)
    gaps = s.gap.numpy()
    pe = np.percentile(err, [50, 90, 99])
    pg = np.percentile(gaps, [50, 90, 99])
    print(f"   [3] vs scipy LP over {n_games} random {N}x{N} games")
    print(f"       |V_mwu - V_lp|  mean {err.mean():.2e}  p50 {pe[0]:.2e}  "
          f"p90 {pe[1]:.2e}  p99 {pe[2]:.2e}  max {err.max():.2e}")
    print(f"       duality gap     mean {gaps.mean():.2e}  p50 {pg[0]:.2e}  "
          f"p90 {pg[1]:.2e}  p99 {pg[2]:.2e}  max {gaps.max():.2e}")
    print(f"       V range [{exact.min():+.3f}, {exact.max():+.3f}]  "
          f"worst game #{int(err.argmax())} (V_lp {exact[err.argmax()]:+.4f}, "
          f"V_mwu {got[err.argmax()]:+.4f})")
    # Absolute tolerances: V is on the reward scale downstream, and these
    # matrices are unit-variance, so 1e-3 is ~0.1% of the payoff range.
    assert err.max() < 1e-3, f"max |V_mwu - V_lp| = {err.max():.4e}"
    assert float(np.median(gaps)) < 5e-3, f"median gap {np.median(gaps):.4e}"
    assert gaps.max() < 5e-2, f"max gap {gaps.max():.4e}"


def test_gap_decreases():
    """More iterations must not make the certificate worse."""
    rng = np.random.RandomState(1)
    M = th.tensor(rng.randn(64, N, N))
    gaps = [float(solve_matrix_game(M, iters=k).gap.median())
            for k in (16, 64, 256, 1024)]
    assert all(b <= a * 1.05 + 1e-6 for a, b in zip(gaps, gaps[1:])), gaps
    assert gaps[-1] < gaps[0] / 2, gaps
    print(f"   [4] gap vs iters (16/64/256/1024): "
          + " -> ".join(f"{g:.2e}" for g in gaps))


def test_scale_invariance_and_the_tau_trap():
    """THE failure this solver is most likely to hit in production.

    Q entries on this task sit near the return scale (G_std ~ 0.0166). Without
    normalization the MWU exponent is ~0 every iteration, the weights never
    leave uniform, and the solver returns a confident-looking wrong answer.
    normalize=True must make V exactly proportional to the scale of M.
    """
    rng = np.random.RandomState(2)
    base = th.tensor(rng.randn(32, N, N))
    for scale in (1.0, 0.0166, 1e-4):          # 1.0, the measured G_std, tiny
        s_on = solve_matrix_game(base * scale, iters=1024, normalize=True)
        s_off = solve_matrix_game(base * scale, iters=1024, normalize=False)
        ref = solve_matrix_game(base, iters=1024, normalize=True).V * scale
        rel_on = float((s_on.V - ref).abs().max() / max(scale, 1e-12))
        # how far from uniform did the un-normalized solve get?
        unif = float((s_off.p - 1.0 / N).abs().max())
        assert rel_on < 5e-2, f"normalize=True broke at scale {scale}: {rel_on}"
        print(f"   [5] scale {scale:<8g} normalize=True rel_err {rel_on:.2e} | "
              f"normalize=False max|p-uniform| {unif:.2e}"
              + ("   <- COLLAPSED TO UNIFORM" if unif < 1e-3 else ""))


def test_batched_matches_singleton():
    """A (B,n,m) solve must equal solving each matrix alone."""
    rng = np.random.RandomState(3)
    Ms = th.tensor(rng.randn(8, N, N))
    batched = solve_matrix_game(Ms, iters=512).V
    singles = th.stack([solve_matrix_game(Ms[i], iters=512).V for i in range(8)])
    assert th.allclose(batched, singles, atol=1e-6), (batched - singles).abs().max()
    print(f"   [6] batched == singleton (max diff "
          f"{float((batched - singles).abs().max()):.2e})")


def test_zero_sum_antisymmetry():
    """Solving -M^T must give -V: the adversary's game is the ego's negated."""
    rng = np.random.RandomState(4)
    M = th.tensor(rng.randn(16, N, N))
    v1 = solve_matrix_game(M, iters=1024).V
    v2 = solve_matrix_game(-M.transpose(1, 2), iters=1024).V
    assert th.allclose(v1, -v2, atol=2e-2), float((v1 + v2).abs().max())
    print(f"   [7] zero-sum antisymmetry V(M) == -V(-M^T) "
          f"(max diff {float((v1 + v2).abs().max()):.2e})")


def test_throughput():
    """A full rollout is 12,288 states, solved once per rollout for GAE.

    Run at the DEFAULT iteration count, not a convenient one -- this number is
    only meaningful if it reflects what production will actually pay. A rollout
    is ~2.7 s of env time at measured throughput, which is the budget to compare
    against.
    """
    import time
    dev = "cuda" if th.cuda.is_available() else "cpu"
    M = th.randn(12288, N, N, device=dev)

    def timed(**kw):
        solve_matrix_game(M, **kw)                       # warm up / autotune
        if dev == "cuda":
            th.cuda.synchronize()
        t0 = time.time()
        s = solve_matrix_game(M, **kw)
        if dev == "cuda":
            th.cuda.synchronize()
        return time.time() - t0, float(s.gap.median())

    dt_def, gap_def = timed()                            # the default
    dt_256, gap_256 = timed(iters=256)
    ROLLOUT_ENV_SECONDS = 2.7
    print(f"   [8] 12,288 x {N}x{N} on {dev}")
    print(f"       iters=256      {dt_256*1000:6.0f} ms  median gap {gap_256:.2e}")
    print(f"       iters=DEFAULT  {dt_def*1000:6.0f} ms  median gap {gap_def:.2e}"
          f"   ({100*dt_def/ROLLOUT_ENV_SECONDS:.1f}% of a rollout's env time)")
    assert dt_def < 0.30, f"{dt_def*1000:.0f} ms per rollout is too slow"
    assert gap_def < 1e-2, f"default median gap {gap_def:.2e} too loose to certify"


def test_lambda_returns():
    """The return recursion, checked against closed forms rather than itself."""
    from common.minimax import minimax_lambda_returns
    T, E, g = 5, 3, 0.9

    # lambda=0 -> the one-step TD target r + gamma*V(s'), which is what
    # textbook minimax-Q uses. This is the case the theory is stated for.
    r = th.randn(T, E)
    V = th.randn(T, E)
    lastV = th.randn(E)
    d = th.zeros(T, E)
    got = minimax_lambda_returns(r, d, V, lastV, gamma=g, gae_lambda=0.0)
    want = r.clone()
    for t in range(T):
        want[t] += g * (lastV if t == T - 1 else V[t + 1])
    assert th.allclose(got, want, atol=1e-6), (got - want).abs().max()

    # lambda=1, no bootstrapping mid-episode -> plain discounted Monte Carlo.
    got = minimax_lambda_returns(r, d, V, lastV, gamma=g, gae_lambda=1.0)
    mc = th.zeros(T, E)
    acc = lastV.clone()
    for t in reversed(range(T)):
        acc = r[t] + g * acc
        mc[t] = acc
    assert th.allclose(got, mc, atol=1e-5), (got - mc).abs().max()

    # A terminal at t must cut the bootstrap: the return there is exactly r_t.
    d2 = th.zeros(T, E); d2[2] = 1.0
    got = minimax_lambda_returns(r, d2, V, lastV, gamma=g, gae_lambda=0.95)
    assert th.allclose(got[2], r[2], atol=1e-6), (got[2] - r[2]).abs().max()
    print("   [9] lambda-returns: lambda=0 == one-step TD, lambda=1 == discounted MC, "
          "done cuts the bootstrap")


if __name__ == "__main__":
    th.manual_seed(0)
    print("minimax solver tests")
    test_matching_pennies()
    test_dominant_row()
    test_vs_scipy_lp()
    test_gap_decreases()
    test_scale_invariance_and_the_tau_trap()
    test_batched_matches_singleton()
    test_zero_sum_antisymmetry()
    test_throughput()
    test_lambda_returns()
    print("\nALL PASSED")
