"""The analytic action-embedding basis, and that it survives the trip into the head.

WHAT IS BEING DEFENDED. gamma_head(s) = e_ego W(s) e_adv^T with GLOBAL
embeddings, so the head's interaction is confined to span(e_ego) (x) span(e_adv)
-- r^2 of 441 dims. Measured on p1_clr1e5 @14.4M, the LEARNED embeddings held
4.93% of the true interaction against 3.63% for a RANDOM subspace; the computed
basis reaches 59.24% at rank 4. So the basis is not a tuning detail, it is most
of whether the head can represent gamma at all.

THE TEST THAT MATTERS is `installed basis reproduces the computed capture`.
Everything else here can pass while a transposed or mis-ranked basis is silently
installed -- the head would train without error and simply never represent the
interaction, which is indistinguishable from the failure we are trying to fix.
So the loop is closed end to end: compute a basis, install it in a real head,
measure the head's OWN subspace against the same gamma, and require the two
numbers to agree.

RECOVERY uses synthetic gamma with a KNOWN planted subspace, because on real
data there is no ground truth to check the solver against -- only self-consistency.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th
import torch.nn as nn

from gamma_basis import gammas, capture, solve_basis
from stable_baselines3.common.clean_new_policies import FactoredMinimaxHead

NE = NA = 22
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def planted(S=300, r=3, seed=0, noise=0.0):
    """gamma_s living in a KNOWN r-dim subspace per side, doubly centred."""
    rng = np.random.RandomState(seed)
    U = np.linalg.qr(rng.randn(NE, r))[0]
    V = np.linalg.qr(rng.randn(NA, r))[0]
    U -= U.mean(0); V -= V.mean(0)
    U = np.linalg.qr(U)[0][:, :r]; V = np.linalg.qr(V)[0][:, :r]
    C = rng.randn(S, r, r)
    G = np.einsum("ir,src,jc->sij", U, C, V)
    if noise:
        N = rng.randn(S, NE, NA) * noise
        G = G + gammas(N)                       # keep the noise doubly centred too
    return gammas(G), U, V


def main():
    # ---- 1. recovery of a planted subspace --------------------------------
    G, U, V = planted(r=3)
    Pe, Pa, trace, _, _ = solve_basis(G, 3)
    check("planted rank-3 subspace is recovered", trace[-1][1] > 0.999,
          f"capture {trace[-1][1]:.4%}")
    # principal angles: the recovered basis must SPAN the planted one, not
    # merely capture its energy by accident
    cos = np.linalg.svd(Pe.T @ U, compute_uv=False)
    check("recovered ego basis spans the planted one", cos.min() > 0.999,
          f"min principal cosine {cos.min():.5f}")

    # ---- 2. monotonicity: HOOI never makes it worse -----------------------
    Gn, _, _ = planted(r=4, seed=1, noise=0.5)
    _, _, tr, _, _ = solve_basis(Gn, 4, sweeps=6)
    caps = [c for _, c in tr]
    check("HOOI is monotone non-decreasing",
          all(caps[i + 1] >= caps[i] - 1e-9 for i in range(len(caps) - 1)),
          " -> ".join(f"{c:.3%}" for c in caps[:4]) + " ...")
    check("solved basis beats the random baseline by a wide margin",
          caps[-1] > 3 * (16 / 441),
          f"{caps[-1]:.2%} vs {16/441:.2%} random ({caps[-1]/(16/441):.1f}x)")

    # ---- 3. rank monotonicity ---------------------------------------------
    c4 = solve_basis(Gn, 4)[2][-1][1]
    c8 = solve_basis(Gn, 8)[2][-1][1]
    check("rank 8 captures at least as much as rank 4", c8 >= c4 - 1e-9,
          f"r4 {c4:.2%}  r8 {c8:.2%}")

    # ---- 4. the closure: computed capture == the HEAD's capture -----------
    # Install the basis in a real head and measure ITS subspace against the same
    # gamma. A transpose or rank error shows up here and nowhere else.
    r = 4
    Gr, _, _ = planted(r=6, seed=2, noise=0.3)
    Pe, Pa, tr, _, _ = solve_basis(Gr, r)
    want = tr[-1][1]
    head = FactoredMinimaxHead(nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU()),
                               16, NE, NA, rank=r,
                               embed_init={"e_ego": Pe, "e_adv": Pa})
    ee = head.e_ego.detach().numpy(); ea = head.e_adv.detach().numpy()
    ee = ee - ee.mean(0); ea = ea - ea.mean(0)          # head centres at forward
    got = capture(Gr, np.linalg.qr(ee)[0][:, :r], np.linalg.qr(ea)[0][:, :r])
    check("installed basis reproduces the computed capture", abs(got - want) < 1e-6,
          f"computed {want:.4%}  head {got:.4%}")

    # ---- 5. freeze ---------------------------------------------------------
    head.played(th.randn(8, 16), th.randint(0, NE, (8,)), th.randint(0, NA, (8,))).sum().backward()
    check("freeze_embed=True -> embeddings get NO gradient",
          head.e_ego.grad is None and head.e_adv.grad is None)
    check("freeze_embed=True -> requires_grad is False",
          (not head.e_ego.requires_grad) and (not head.e_adv.requires_grad))
    h2 = FactoredMinimaxHead(nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU()),
                             16, NE, NA, rank=r,
                             embed_init={"e_ego": Pe, "e_adv": Pa}, freeze_embed=False)
    h2.played(th.randn(8, 16), th.randint(0, NE, (8,)), th.randint(0, NA, (8,))).sum().backward()
    check("freeze_embed=False -> embeddings DO get gradient",
          h2.e_ego.grad is not None and float(h2.e_ego.grad.abs().max()) > 0)

    # ---- 6. shape mismatch must RAISE, not broadcast -----------------------
    for bad, why in ((Pe[:, :2], "wrong rank"), (Pe.T, "transposed")):
        try:
            FactoredMinimaxHead(nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU()),
                                16, NE, NA, rank=r,
                                embed_init={"e_ego": bad, "e_adv": Pa})
            raised = False
        except ValueError:
            raised = True
        check(f"{why} basis RAISES", raised)

    # ---- 7. absent embed_init leaves the head untouched -------------------
    th.manual_seed(0)
    a = FactoredMinimaxHead(nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU()), 16, NE, NA, rank=r)
    th.manual_seed(0)
    b = FactoredMinimaxHead(nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU()), 16, NE, NA, rank=r,
                            embed_init=None)
    check("embed_init=None is a no-op (bitwise)", th.equal(a.e_ego, b.e_ego))
    check("embed_init=None leaves embeddings TRAINABLE", a.e_ego.requires_grad)

    print("\n  ALL PASS" if OK else "\n  FAILURES PRESENT")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
