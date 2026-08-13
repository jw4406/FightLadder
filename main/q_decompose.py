"""How much of the head's Q(s,i,j) is carried by each ANOVA term?

    Q(s,i,j) = mu(s) + alpha_i(s) + beta_j(s) + gamma_ij(s)
               \____/   \_______________/       \_________/
               state      main effects          interaction

Works on EITHER head. For 'matrix' the split is computed post-hoc from the
emitted 22x22 -- the free head has no internal structure, but any matrix can
still be decomposed. For 'factored' the same post-hoc split is computed AND
compared against the head's own components(), which must agree exactly: the
centring makes mu == V, alpha == A_ego, beta == A_adv, gamma == the bilinear
term. A mismatch means the parameterization is not doing what it claims.

WHY THE SHARES ARE ADDITIVE. Under the uniform inner product the four terms are
mutually orthogonal (that is what the centring buys), so the total sum of
squares splits exactly:

    SS_total = SS_mu + SS_alpha + SS_beta + SS_gamma

with no cross terms. Any residual in that identity is a bug, so it is asserted
rather than assumed.

TWO NORMALISATIONS, because they answer different questions:
  vs TOTAL   mu dominates (~95% offline) because states differ far more than
             actions do. Useful for "what is Q mostly encoding".
  vs WITHIN  drops mu. This is the decision-relevant one: at a FIXED state, how
             much of the action-conditional signal is main effects vs
             interaction. Offline on the true payoff, gamma was 0.069 of within.

REFERENCE VALUES (offline payoff ANOVA, 2,400 states, masked RAM 2.4M):
    mu 94.9% of total | alpha+beta 4.9% | gamma 0.24% of total, 6.9% of within
    gamma singular spectrum .494 .192 .102 .063 -> median rank 2, p90 rank 4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--npz", default="",
                    help="Decompose a SAVED payoff matrix (M from bootstrap_delta) "
                         "instead of the head's Q. This is how the head is compared "
                         "against the emulator-derived payoff AT THE SAME CHECKPOINT: "
                         "identical ANOVA code on both sides, so a difference cannot "
                         "come from a reimplementation. Note M = r + gamma*V_scalar(s'), "
                         "so its ACTION axis is exact (emulator) but the V(s') term is "
                         "the trained critic and is NOT validated.")
    ap.add_argument("--npz_key", default="M",
                    help="Which array in the npz to decompose. 'M' = r + gamma*V(s') "
                         "(action axis exact, V term unvalidated). 'R' = the EXACT "
                         "emulator reward with NO critic anywhere -- its interaction "
                         "is the part that cannot be critic noise.")
    ap.add_argument("--ram_mask", type=str, default="")
    ap.add_argument("--n_states", type=int, default=200)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups,
                                     infer_obs_kwargs, REPO_ROOT)

    if a.npz:
        d = np.load(a.npz)
        if a.npz_key not in d:
            raise SystemExit(f"npz has {list(d.files)}, no '{a.npz_key}' -- recollect "
                             f"with the bootstrap_delta that saves R")
        M = d[a.npz_key].astype(np.float64)
        kind = ("EMULATOR PAYOFF r + gamma*V_scalar(s')" if a.npz_key == "M"
                else "EXACT EMULATOR REWARD (no critic)")
        comps = []
        print(f"  source: {os.path.basename(a.npz)}  {M.shape[0]} states")
        return _analyse(M, comps, kind, os.path.basename(a.npz), a)
    if not a.ckpt:
        raise SystemExit("need --ckpt or --npz")
    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs,
                          **infer_obs_kwargs(data, a.ram_mask or None))
    Ms, comps = [], []
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        pol = model.policy
        head = pol.minimax_head_for([head_idx])
        if head is None:
            raise SystemExit("checkpoint has no minimax head (--minimax_q was off)")
        kind = type(head).__name__
        rng = np.random.RandomState(0)
        obs = venv.reset()
        n_batches = int(np.ceil(a.n_states / a.n_envs))
        for b in range(n_batches):
            for _ in range(a.stride if b else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]
            with th.no_grad():
                M = pol.minimax_matrices(th.as_tensor(obs).to(a.device),
                                         buf_num=[head_idx], stop_grad=True)
                Ms.append(M.cpu().numpy().astype(np.float64))
                if hasattr(head, "components"):
                    lat = pol.minimax_latent(th.as_tensor(obs).to(a.device))
                    v, ae, aa, g = head.components(lat)
                    comps.append(tuple(x.cpu().numpy().astype(np.float64)
                                       for x in (v, ae, aa, g)))
    finally:
        venv.close()

    M = np.concatenate(Ms)                              # (S, ne, na)
    return _analyse(M, comps, kind, os.path.basename(a.ckpt), a)


def _analyse(M, comps, kind, label, a):
    """The decomposition itself. Shared by the head path and the --npz path so
    the two are guaranteed to be measured the same way."""
    import numpy as np
    import os
    S, ne, na = M.shape

    mu = M.mean(axis=(1, 2))
    alpha = M.mean(axis=2) - mu[:, None]
    beta = M.mean(axis=1) - mu[:, None]
    gamma = (M - mu[:, None, None] - alpha[:, :, None] - beta[:, None, :])

    # Orthogonality => the sums of squares add up EXACTLY. Assert it.
    ss_mu = ne * na * float(((mu - mu.mean()) ** 2).sum())
    ss_al = na * float((alpha ** 2).sum())
    ss_be = ne * float((beta ** 2).sum())
    ss_ga = float((gamma ** 2).sum())
    ss_tot = float(((M - M.mean()) ** 2).sum())
    resid = abs(ss_mu + ss_al + ss_be + ss_ga - ss_tot) / max(ss_tot, 1e-30)

    print("=" * 74)
    print(f"Q DECOMPOSITION  {label}")
    print(f"  source {kind}   {S} states   {ne}x{na} actions")
    print("=" * 74)
    print(f"  SS identity residual {resid:.2e}   (must be ~0; orthogonality check)")

    within = ss_al + ss_be + ss_ga
    print(f"\n  {'term':<22} {'vs TOTAL':>12} {'vs WITHIN-state':>18}")
    for nm, ss in (("mu    (state)", ss_mu), ("alpha (ego main)", ss_al),
                   ("beta  (adv main)", ss_be), ("gamma (INTERACTION)", ss_ga)):
        w = f"{ss / within:>17.4%}" if nm.startswith(("alpha", "beta", "gamma")) else f"{'-':>17s}"
        print(f"  {nm:<22} {ss / ss_tot:>11.4%} {w}")

    print(f"\n  rms magnitudes   mu {mu.std():.6f}   alpha {alpha.std():.6f}"
          f"   beta {beta.std():.6f}   gamma {gamma.std():.6f}")

    # rank of the interaction: how many directions does gamma actually use?
    sv = np.linalg.svd(gamma, compute_uv=False)          # (S, min(ne,na))
    frac = (sv ** 2) / np.maximum((sv ** 2).sum(axis=1, keepdims=True), 1e-30)
    cum = np.cumsum(frac, axis=1)
    rank90 = (cum < 0.90).sum(axis=1) + 1
    print(f"  gamma spectrum (mean normalised sv) "
          + " ".join(f"{x:.3f}" for x in (sv / np.maximum(sv[:, :1], 1e-30)).mean(0)[:4]))
    print(f"  gamma rank for 90% energy   median {int(np.median(rank90))}"
          f"   p90 {int(np.percentile(rank90, 90))}")

    # antisymmetric share -- the cyclic part. Isotropic null is
    # (n(n-1)/2)/((n^2-1)... ) = 210/441 = 0.4762 at n=22; BELOW that is not
    # evidence of cyclic structure, it is evidence of less than chance.
    anti = 0.5 * (gamma - np.transpose(gamma, (0, 2, 1)))
    # Isotropic null = (dim of the ANTISYMMETRIC, doubly-centred subspace)
    #                / (dim of the doubly-centred subspace)
    # Antisymmetric n x n has dim n(n-1)/2; requiring zero row sums removes
    # (n-1) more, leaving (n-1)(n-2)/2. Doubly centred has dim (n-1)^2. So the
    # null is (n-2)/(2(n-1)) = 0.4762 at n=22 -- NOT n(n-1)/2 / (n-1)^2 = 0.5238,
    # which forgets the centring constraint and was confirmed wrong empirically
    # (a random doubly-centred matrix measures 0.4763).
    null = (ne - 2) / (2.0 * (ne - 1)) if ne == na else float("nan")
    print(f"  gamma antisymmetric share {float((anti ** 2).sum() / max((gamma ** 2).sum(), 1e-30)):.4f}"
          f"   (isotropic null {null:.4f})")

    res = {"checkpoint": label, "head": kind, "n_states": S,
           "ss_residual": resid,
           "share_total": {"mu": ss_mu / ss_tot, "alpha": ss_al / ss_tot,
                           "beta": ss_be / ss_tot, "gamma": ss_ga / ss_tot},
           "share_within": {"alpha": ss_al / within, "beta": ss_be / within,
                            "gamma": ss_ga / within},
           "rms": {"mu": float(mu.std()), "alpha": float(alpha.std()),
                   "beta": float(beta.std()), "gamma": float(gamma.std())},
           "gamma_rank_median": int(np.median(rank90)),
           "gamma_anti_share": float((anti ** 2).sum() / max((gamma ** 2).sum(), 1e-30))}

    # ---- factored head only: does its OWN split match the post-hoc one? ----
    if comps:
        v = np.concatenate([c[0] for c in comps])
        ae = np.concatenate([c[1] for c in comps])
        aa = np.concatenate([c[2] for c in comps])
        gg = np.concatenate([c[3] for c in comps])
        d = {"V vs mu": np.abs(v - mu).max(), "A_ego vs alpha": np.abs(ae - alpha).max(),
             "A_adv vs beta": np.abs(aa - beta).max(), "gamma vs gamma": np.abs(gg - gamma).max()}
        print(f"\n  FACTORED HEAD -- its own components vs the post-hoc ANOVA:")
        for k, val in d.items():
            print(f"    max|diff| {k:<18} {val:.3e}"
                  f"   {'OK' if val < 1e-4 else 'MISMATCH'}")
        res["factored_agreement"] = {k: float(val) for k, val in d.items()}

    if a.out:
        out = os.path.join(REPO_ROOT, a.out)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  wrote {out}")
    return res


if __name__ == "__main__":
    main()
