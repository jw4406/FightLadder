"""Unit tests for PopArtHead (value-target normalization).

Run:  python test_popart.py

The load-bearing property is (1): if a statistics update changes the module's
outputs, every prediction the policy is bootstrapping off takes a step change,
which is exactly what the output-preserving correction exists to prevent.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as th
import torch.nn as nn

from stable_baselines3.common.clean_new_policies import PopArtHead


def _mk(seed=0, beta=0.1):
    th.manual_seed(seed)
    net = nn.Sequential(nn.Linear(16, 32), nn.LeakyReLU(),
                        nn.Linear(32, 32), nn.LeakyReLU(),
                        nn.Linear(32, 1))
    return PopArtHead(net, beta=beta)


def test_output_preserved():
    """A stats update must not move the outputs. This is the whole point."""
    h = _mk()
    x = th.randn(64, 16)
    with th.no_grad():
        v0 = h(x).clone()
    # A target distribution far from the init (mu=0, sigma=1) so the correction
    # has to do real work rather than a no-op rescale.
    for _ in range(50):
        h.update_stats(th.randn(256) * 0.03 + 0.4)
    with th.no_grad():
        v1 = h(x)
    mu, sigma = h.effective_stats()
    assert not (abs(mu) < 1e-6 and abs(sigma - 1.0) < 1e-6), "stats never moved"
    assert th.allclose(v0, v1, rtol=1e-4, atol=1e-6), \
        f"outputs moved: max|d|={float((v0-v1).abs().max()):.3e}"
    print(f"   [1] outputs preserved across stats update "
          f"(mu={mu:+.4f}, sigma={sigma:.4f}, max|d|={float((v0-v1).abs().max()):.2e})")


def test_stats_converge():
    """mu/sigma should track the target distribution, so normalize() whitens."""
    h = _mk(beta=0.05)
    m, s = -0.25, 0.07
    for _ in range(500):
        h.update_stats(th.randn(512) * s + m)
    mu, sigma = h.effective_stats()
    assert abs(mu - m) < 0.02, f"mu {mu} !~ {m}"
    assert abs(sigma - s) < 0.02, f"sigma {sigma} !~ {s}"
    y = th.randn(4096) * s + m
    z = h.normalize(y)
    assert abs(float(z.mean())) < 0.15 and abs(float(z.std()) - 1.0) < 0.15, \
        f"normalize did not whiten: mean={float(z.mean()):.3f} std={float(z.std()):.3f}"
    print(f"   [2] stats converge (mu={mu:+.4f} vs {m}, sigma={sigma:.4f} vs {s}); "
          f"normalized mean={float(z.mean()):+.3f} std={float(z.std()):.3f}")


def test_stats_converge_FAST():
    """REGRESSION: sigma must be right within a FEW updates, not eventually.

    The original code initialized nu to ones while applying a zero-init debias,
    so sigma was ~41x too large and only decayed away over ~1/beta updates. The
    existing convergence test used beta=0.05 with 500 updates -- (1-0.05)^500 is
    7e-12, so the transient was long gone and the bug was invisible. Small beta
    plus few updates is the regime that exposes it.
    """
    s = 0.017                                   # the real target scale here
    for beta, n in ((3e-4, 5), (3e-4, 50), (0.05, 5)):
        h = _mk(beta=beta)
        for _ in range(n):
            h.update_stats(th.randn(4096) * s)
        _, sigma = h.effective_stats()
        assert abs(sigma - s) / s < 0.25, (
            f"beta={beta} n={n}: sigma={sigma:.5f} vs true {s} "
            f"(ratio {sigma/s:.1f}x) -- debias/init mismatch")
    print(f"   [3] sigma correct after as few as 5 updates at beta=3e-4 "
          f"(was 41x high)")


def test_mu_cancels_in_loss():
    """The loss only ever sees sigma -- mu cancels in a difference. This is what
    lets the two call sites normalize without caring about the ego/adv frame."""
    h = _mk()
    for _ in range(50):
        h.update_stats(th.randn(256) * 0.05 + 0.3)
    a, b = th.randn(128) * 0.1, th.randn(128) * 0.1
    lhs = th.nn.functional.mse_loss(h.normalize(a), h.normalize(b))
    rhs = th.nn.functional.mse_loss(a / h.sigma, b / h.sigma)
    assert th.allclose(lhs, rhs, rtol=1e-5), f"{float(lhs)} != {float(rhs)}"
    print(f"   [4] mu cancels in the loss ({float(lhs):.6e} == {float(rhs):.6e})")


def test_state_dict_keys():
    """popart=False must leave the historical key layout untouched, or every
    existing .task stops loading."""
    plain = nn.ModuleDict({"RyuVsSagat": nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))})
    wrapped = nn.ModuleDict({"RyuVsSagat": PopArtHead(
        nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1)))})
    pk, wk = set(plain.state_dict()), set(wrapped.state_dict())
    assert "RyuVsSagat.0.weight" in pk, sorted(pk)
    assert "RyuVsSagat.net.0.weight" in wk, sorted(wk)
    assert not (pk & wk), "layouts must be disjoint -- silent partial loads otherwise"
    assert {"RyuVsSagat.mu", "RyuVsSagat.sigma"} <= wk, sorted(wk)
    print(f"   [5] key layouts disjoint; buffers persisted "
          f"(plain={len(pk)} keys, popart={len(wk)} keys)")


def test_degenerate_targets():
    """Empty / non-finite / zero-variance batches must not poison the stats."""
    h = _mk()
    before = h.effective_stats()
    h.update_stats(th.empty(0))
    h.update_stats(th.tensor([float("nan"), 1.0]))
    assert h.effective_stats() == before, "degenerate batch mutated stats"
    for _ in range(50):
        h.update_stats(th.full((128,), 0.2))     # zero variance
    _, sigma = h.effective_stats()
    assert sigma >= h.sigma_min and th.isfinite(h.sigma).all(), sigma
    x = th.randn(8, 16)
    assert th.isfinite(h(x)).all(), "non-finite output after zero-variance targets"
    print(f"   [6] degenerate batches rejected; sigma floored at {sigma:.2e}")


if __name__ == "__main__":
    print("PopArtHead tests")
    test_output_preserved()
    test_stats_converge()
    test_stats_converge_FAST()
    test_mu_cancels_in_loss()
    test_state_dict_keys()
    test_degenerate_targets()
    print("\nALL PASSED")
