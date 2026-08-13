"""The energy-optimal action embeddings for the factored head's interaction term.

    gamma_head(s) = e_ego W(s) e_adv^T

e_ego and e_adv are GLOBAL (state-independent), so as W(s) ranges over all r x r
the head's gamma is confined to span(e_ego) (x) span(e_adv) -- an r^2-dimensional
slice of the 441-dim doubly-centred space. r=4 gives 16/441 = 3.6%.

MEASURED, p1_clr1e5 @14.4M: only 4.93% of the TRUE interaction lies inside the
LEARNED subspace, against 3.63% for a RANDOM one. The embeddings ended up barely
better than random, so the head could not represent the interaction no matter how
well W was fit. This computes the basis directly instead of hoping SGD finds it.

WHY IT CAN BE COMPUTED. The objective

    maximise   sum_s || P_e^T gamma_s P_a ||_F^2   over orthonormal P_e, P_a

is the Tucker-2 / 2D-PCA problem. The eigenvectors of the row and column
covariances solve each side MARGINALLY and are the standard initialiser; the
JOINT optimum needs alternating refinement (HOOI). Both are 22x22
eigendecompositions -- milliseconds. Reported at three points (random,
eigen-init, post-HOOI) so the refinement's contribution is visible rather than
assumed: a previous estimate of the ceiling used the product of the two top-r
eigenvalue shares, which is a heuristic, not an achievable maximum.

TWO PROPERTIES FALL OUT FREE, and both are asserted rather than trusted:

  energy weighting   C_row = sum_s gamma_s gamma_s^T gives ZERO weight to states
                     with gamma_s = 0. Since gamma vanishes at 77.8% of states
                     and 88.6% of its energy sits in the top 10%, the covariance
                     IS the contact-state filter, weighted by how much
                     interaction each state actually carries. No explicit
                     filtering step is needed or wanted.
  centring           gamma_s is doubly centred, so C_row . 1 = 0 and the
                     all-ones vector is in the null space. Every eigenvector
                     with nonzero eigenvalue is therefore already orthogonal to
                     it -- exactly the constraint the head imposes at forward
                     time.

ORTHONORMAL EMBEDDINGS ARE A DIAGNOSTIC UPGRADE. With P_e, P_a orthonormal,
W(s) is literally the coefficient matrix of gamma_s in that basis and
||W(s)||_F == ||projected gamma_s||_F. train/minimax_fx_w_norm stops being an
arbitrary scale and becomes directly comparable against the emulator's
||gamma_s||.

R OR M. The head trains on r + gamma*V_mm(s'), so strictly the relevant gamma is
M's. But V is the component measured to be actively HARMFUL for branch selection
(adding it to an LBR search turned a +0.139 exploit into -0.229). At 14.4M the
two agreed closely -- gamma 21.80% (R) vs 22.95% (M), both rank ~2 -- so
--key lets you compute both and compare the subspaces before choosing.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def gammas(A):
    """ANOVA interaction term of each (22,22) slice. Doubly centred by construction."""
    import numpy as np
    mu = A.mean(axis=(1, 2))
    al = A.mean(axis=2) - mu[:, None]
    be = A.mean(axis=1) - mu[:, None]
    return A - mu[:, None, None] - al[:, :, None] - be[:, None, :]


def capture(G, Pe, Pa):
    """Fraction of gamma energy inside span(Pe) (x) span(Pa)."""
    import numpy as np
    tot = float((G ** 2).sum())
    if tot <= 0:
        return float("nan")
    return float((np.einsum("ir,sij,jc->src", Pe, G, Pa) ** 2).sum() / tot)


def top_r(C, r):
    import numpy as np
    w, V = np.linalg.eigh(C)               # ascending
    return V[:, ::-1][:, :r], w[::-1]


def solve_basis(G, r, sweeps=5, verbose=True):
    """Eigen-init then HOOI. Returns (Pe, Pa, trace) with capture at each stage."""
    import numpy as np
    Crow = np.einsum("sij,skj->ik", G, G)
    Ccol = np.einsum("sij,sil->jl", G, G)
    Pe, we = top_r(Crow, r)
    Pa, wa = top_r(Ccol, r)
    trace = [("eigen-init", capture(G, Pe, Pa))]
    for it in range(sweeps):
        # Refine each side against the OTHER's current subspace. Marginal
        # eigenvectors are optimal per-side but not jointly; this closes the gap.
        Ge = np.einsum("sij,jc->sic", G, Pa)
        Pe, _ = top_r(np.einsum("sic,skc->ik", Ge, Ge), r)
        Ga = np.einsum("sij,ir->srj", G, Pe)
        Pa, _ = top_r(np.einsum("srj,srl->jl", Ga, Ga), r)
        trace.append((f"hooi-{it+1}", capture(G, Pe, Pa)))
    return Pe, Pa, trace, we, wa


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="raw npz from bootstrap_delta")
    ap.add_argument("--key", default="R", choices=["R", "M"],
                    help="R = exact emulator reward, no critic. M = r + gamma*V(s').")
    ap.add_argument("--rank", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--sweeps", type=int, default=5)
    ap.add_argument("--out", default="", help="save the FIRST rank's basis here")
    a = ap.parse_args(argv)

    import numpy as np
    d = np.load(a.npz)
    if a.key not in d:
        raise SystemExit(f"npz has {list(d.files)}; no '{a.key}'. Recollect with a "
                         f"bootstrap_delta that saves R.")
    A = d[a.key].astype(np.float64)
    G = gammas(A)
    S, ne, na = G.shape
    nz = (np.abs(G).max(axis=(1, 2)) > 1e-12)
    e_state = (G ** 2).sum(axis=(1, 2))
    order = np.sort(e_state)[::-1]

    print("=" * 78)
    print(f"GAMMA BASIS  {os.path.basename(a.npz)}  key={a.key}  {S} states")
    print("=" * 78)
    print(f"  states with gamma == 0        {1 - nz.mean():.1%}")
    print(f"  top 10% of states hold        {order[:max(1, S//10)].sum()/order.sum():.1%} of gamma energy")
    print(f"  EFFECTIVE sample size is far below {S}: the covariance is energy-")
    print(f"  weighted, so only the {nz.sum()} nonzero states inform the basis at all.")

    res = {"npz": os.path.basename(a.npz), "key": a.key, "n_states": S,
           "frac_zero": float(1 - nz.mean()), "ranks": {}}
    first = None
    for r in a.rank:
        Pe, Pa, trace, we, wa = solve_basis(G, r, sweeps=a.sweeps)
        rnd = (r * r) / ((ne - 1) * (na - 1))
        print(f"\n  --- rank {r} ---   subspace {r*r} of {(ne-1)*(na-1)} dims")
        print(f"    {'random baseline':<16} {rnd:>8.2%}")
        for nm, c in trace:
            print(f"    {nm:<16} {c:>8.2%}")
        # the heuristic I previously quoted as the ceiling, for comparison
        prod = float(we[:r].sum() / we.sum() * wa[:r].sum() / wa.sum())
        print(f"    {'(product-of-shares heuristic)':<16} {prod:>8.2%}   <- NOT an achievable bound")
        assert np.allclose(Pe.T @ Pe, np.eye(r), atol=1e-8), "Pe not orthonormal"
        assert np.allclose(Pa.T @ Pa, np.eye(r), atol=1e-8), "Pa not orthonormal"
        # centring should be automatic: gamma is doubly centred => 1 is in the
        # null space of both covariances. Asserted, not assumed.
        assert np.abs(Pe.sum(0)).max() < 1e-6, f"Pe not centred: {np.abs(Pe.sum(0)).max():.2e}"
        assert np.abs(Pa.sum(0)).max() < 1e-6, f"Pa not centred: {np.abs(Pa.sum(0)).max():.2e}"
        res["ranks"][str(r)] = {"random": rnd, "eigen_init": trace[0][1],
                                "hooi": trace[-1][1], "product_heuristic": prod}
        if first is None:
            first = (Pe, Pa, r)

    if a.out:
        Pe, Pa, r = first
        out = os.path.join(REPO_ROOT, a.out)
        np.savez(out, e_ego=Pe.astype(np.float32), e_adv=Pa.astype(np.float32),
                 rank=r, key=a.key, source=os.path.basename(a.npz),
                 capture=res["ranks"][str(r)]["hooi"])
        print(f"\n  wrote basis (rank {r}) -> {out}")
        with open(out.replace(".npz", ".json"), "w") as f:
            json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main()
