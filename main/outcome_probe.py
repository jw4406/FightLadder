"""Is the MATCH OUTCOME predictable, even though the discounted return is not?

D1 established that realized returns are near-unpredictable here: a from-scratch
copy of the critic architecture, trained by clean supervised regression on Monte
Carlo returns, reached test EV 0.056 at gamma=0.99, and frozen-encoder ridge
probes cap at 0.03-0.10 across every gamma.

That leaves three very different explanations, which this separates:

  1  outcome predictable, return not
       -> the environment IS learnable and the RETURN DEFINITION is the culprit.
          At gamma=0.99 the terminal +-1 is discounted by 0.99^435 ~ 0.013 and is
          invisible; at gamma=1.0 it is present but drowned in dense-reward
          variance. Justifies splitting the critic into a dense head (short
          gamma) and an outcome head (gamma=1, per-episode label).
  2  not predictable from PIXELS but predictable from hp_diff
       -> the OBSERVATION ENCODING is the bottleneck, not the value architecture.
          Obs is (3, 100, 128); HP bars are a handful of pixels at that size.
          No critic change would help.
  3  not predictable from anything
       -> states genuinely do not determine outcomes much, and no critic will
          ever help. Consistent with greedy (gamma=0, no critic) beating
          critic-guided LBR at every checkpoint measured.

Why outcome could be learnable where the return is not: the label is computed
once at episode end and broadcast to every timestep of that episode. It is
CONSTANT within an episode and carries ZERO per-step sampling noise. The
discounted return, by contrast, is dominated by the stochasticity of the next
~100 steps of both players' action sampling.

Because the label is per-episode, splits MUST be by episode -- a timestep split
here would be catastrophic, not merely optimistic (train and test would contain
literally identical labels from the same episode).

Predictability is also reported BY PHASE of the episode. Outcome is necessarily
near-determined in the final frames and near-chance at the opening; a single
pooled number hides that, and the useful question is how EARLY it becomes
predictable.
"""
import os
import sys


def _peek(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


_d = _peek(sys.argv[1:])
if _d is not None and str(_d).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import time
import argparse
import numpy as np

from stable_baselines3.common.save_util import load_from_zip_file
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups, REPO_ROOT, _b)
from lbr_head_probe import episode_split
from critic_ceiling import collect_raw, encode, auc, episode_labels  # noqa: F401


def ridge_scores(X, y, tr, te, grid=(1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
                 va=None):
    """Ridge fit on +-1 labels; returns test scores plus the chosen alpha.

    Ridge on a binary target is a linear discriminant up to monotone rescaling,
    which is all AUC needs -- and it keeps this consistent with the other probes
    rather than introducing a differently-regularized logistic fit.
    """
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    A = np.concatenate([(X[tr] - mu) / sd, np.ones((tr.size, 1))], 1)

    def fit(alpha):
        return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y[tr])

    def proj(w, idx):
        B = np.concatenate([(X[idx] - mu) / sd, np.ones((idx.size, 1))], 1)
        return B @ w

    if va is not None and va.size > 4:
        best = max(grid, key=lambda a: auc(proj(fit(a), va), (y[va] > 0).astype(int)))
    else:
        best = grid[len(grid) // 2]
    return proj(fit(best), te), float(best)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Is match outcome predictable?")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval_prot", type=str, default="False")
    ap.add_argument("--max_gb", type=float, default=40.0)
    ap.add_argument("--out", type=str, default="outcome_probe.json")
    args = ap.parse_args(argv)

    data = load_from_zip_file(args.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]

    venv = build_lbr_venv(state, args.n_envs)
    try:
        model, _ = load_checkpoint(args.ckpt, venv, args.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=_b(args.eval_prot))
        t0 = time.time()
        OBS, R, D, V, S, EP = collect_raw(venv, ops, args.steps, args.max_gb)
    finally:
        venv.close()
    policy = ops.p
    print(f"\n   collected {int(D.sum())} episodes in {(time.time()-t0)/60:.1f} min",
          flush=True)

    EPf, Rf = EP.reshape(-1), R.reshape(-1)
    Sf = S.reshape(-1, S.shape[-1])
    Df = D.reshape(-1)

    # Per-episode label, with an independent cross-check. Canonical implementation
    # lives in critic_ceiling (this module already imports from there, so the
    # reverse direction would be circular).
    m, y, ep_m, _meta = episode_labels(EPf, Rf, Sf, Df)
    agree, dist = _meta["label_agreement"], _meta["outcome_dist"]

    # phase within episode, for the by-phase breakdown
    phase = np.zeros(m.sum(), float)
    for e in np.unique(ep_m):
        i = np.nonzero(ep_m == e)[0]
        phase[i] = np.linspace(0.0, 1.0, i.size)

    print("\n   computing frozen encoder features ...", flush=True)
    # The adversary encoder is a CROSS-PERSPECTIVE reference, not just another
    # row: the label here is the EGO's win/loss, so this asks whether the side
    # that is actually winning (rating_gap -558 on this run) has learned a
    # representation that predicts the outcome better than the losing side's.
    FEAT = {"critic_cnn": encode(policy, OBS, "vf", args.device)[m],
            "ego_actor_cnn": encode(policy, OBS, "pi", args.device)[m],
            "adv_actor_cnn": encode(policy, OBS, "dstb", args.device)[m]}

    tr, va, te = episode_split(ep_m, seed=args.seed)
    yb = (y > 0).astype(int)
    res = {"checkpoint": os.path.basename(args.ckpt), "matchup": label,
           "num_timesteps": data.get("num_timesteps"),
           "label_agreement": float(agree), "outcome_dist": dist,
           "n_samples": int(m.sum()),
           "n_episodes": int(np.unique(ep_m).size),
           "n_episodes_test": int(np.unique(ep_m[te]).size), "probes": {}}

    bands = [(0.0, 0.33, "early"), (0.33, 0.66, "mid"), (0.66, 1.0, "late")]
    print(f"\n   {'probe':22s} {'AUC all':>9s} {'early':>8s} {'mid':>8s} "
          f"{'late':>8s} {'alpha':>8s}")
    for nm, X in (("hp_diff only", Sf[m][:, :1]),
                  ("state features", Sf[m]),
                  ("critic CNN (frozen)", FEAT["critic_cnn"]),
                  ("EGO actor CNN", FEAT["ego_actor_cnn"]),
                  ("ADV actor CNN", FEAT["adv_actor_cnn"])):
        sc, alpha = ridge_scores(X, y, tr, te, va=va)
        row = {"auc_all": auc(sc, yb[te]), "alpha": alpha}
        cells = []
        for lo, hi, bn in bands:
            sel = (phase[te] >= lo) & (phase[te] < hi)
            row[f"auc_{bn}"] = auc(sc[sel], yb[te][sel]) if sel.sum() > 10 else float("nan")
            cells.append(row[f"auc_{bn}"])
        res["probes"][nm] = row
        print(f"   {nm:22s} {row['auc_all']:>9.3f} {cells[0]:>8.3f} "
              f"{cells[1]:>8.3f} {cells[2]:>8.3f} {alpha:>8.0e}")

    # The trained critic's own V, as a ranker of eventual outcome.
    Vm = V.reshape(-1)[m]
    res["trained_V_auc"] = auc(Vm[te], yb[te])
    print(f"   {'trained critic V':22s} {res['trained_V_auc']:>9.3f}")

    p = os.path.join(REPO_ROOT, args.out)
    with open(p, "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 74)
    print("READING  (AUC 0.5 = chance, 1.0 = perfect)")
    print("=" * 74)
    best = max(r["auc_all"] for r in res["probes"].values())
    px = max(res["probes"][k]["auc_all"] for k in
             ("critic CNN (frozen)", "EGO actor CNN", "ADV actor CNN"))
    hp = res["probes"]["hp_diff only"]["auc_all"]
    if best < 0.60:
        print("  (3) outcome is NOT predictable from anything -> states do not")
        print("      determine outcomes; no critic will help.")
    elif px < 0.60 <= hp:
        print("  (2) predictable from hp_diff but NOT from pixels -> the")
        print("      OBSERVATION ENCODING is the bottleneck, not the critic.")
    else:
        if hp > px + 0.02:
            print(f"  NOTE: hp_diff alone ({hp:.3f}) beats every PIXEL encoder "
                  f"({px:.3f}).")
            print("      A secondary encoding gap sits on top of the finding below:")
            print("      feeding the state scalars to the critic directly is the")
            print("      cheapest intervention available, and is not architectural.")
        print("  (1) outcome IS predictable while the return is not -> the")
        print("      RETURN DEFINITION is the culprit; splitting the critic into")
        print("      a dense head (short gamma) and an outcome head (gamma=1)")
        print("      is justified.")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
