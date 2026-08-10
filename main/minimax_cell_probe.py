"""Is the minimax Q head accurate on the cells LBR actually branches over?

THE QUESTION THIS ANSWERS. `coverage` in the training log reports the fraction of
the 484 joint-action cells that have EVER received a gradient. It read 1.000 for
the whole Phase 0 run -- which is true and nearly useless, because the checkpoint
shows a 1,400x imbalance underneath it:

    6.72M   484/484 cells, 53.5M updates, per cell: min 1,264  median 7,486  max 10,518,860

LBR branches over ALL 22 adversary actions at every decision point, so it reads Q
in cells the policy almost never plays. If Q is well fit on its dominant cells
and garbage on the rare ones, a null gate result ("minimax ~= minimaxshuffle")
would mean "Q was never shown those branches", not "Q has no branch information"
-- and those imply completely different next steps.

WHY THIS IS EVAL-ONLY. The alternative test is to retrain with random-action
injection and see if the gate changes, which is a multi-hour training run. This
needs no gradient at all: roll BOTH players uniformly at random so every cell is
visited about equally, record the realized return, and compare Q's prediction
against it STRATIFIED BY that cell's visit count from the checkpoint. Uniform
play is the point -- it is the only way to sample the rare cells at all.

READING IT:
  error flat across visit deciles  -> thin cells are fine, a null gate means
                                      what it says, injection is unnecessary
  error blows up on rare cells     -> the gate is confounded; build the
                                      injection run before concluding anything

CAVEAT, stated because it limits the conclusion: uniform play also visits
different STATES than the trained policy does, so a high error on rare cells
could be off-distribution states rather than the cells themselves. The
`--policy_actions` control samples actions from the real policies over the same
states, which separates those: if error is high under random actions but low
under policy actions on the same states, it is the CELLS, not the states.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _peek(argv, flag, default):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return default


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=1200,
                    help="vec-steps of uniform-random play")
    ap.add_argument("--n_envs", type=int, default=13)
    ap.add_argument("--gamma", type=float, default=None,
                    help="discount for the realized return; default = model.gamma")
    ap.add_argument("--policy_actions", action="store_true",
                    help="CONTROL: sample actions from the real policies instead "
                         "of uniformly, over the same states. Separates 'rare "
                         "cells are badly fit' from 'off-distribution states'.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="minimax_cell_probe.json")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, REPO_ROOT)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]

    venv = build_lbr_venv(state, a.n_envs)
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        pol = ops.p
        if not getattr(pol, "minimax_q", False):
            raise SystemExit("checkpoint has no minimax head (--minimax_q was off)")
        head = pol.minimax_net[list(pol.minimax_net.keys())[head_idx]]
        visits = head.cell_visits.detach().cpu().numpy()
        n_ego, n_adv = visits.shape
        gamma = a.gamma if a.gamma is not None else float(getattr(model, "gamma", 0.99))

        rng = np.random.RandomState(0)
        n = venv.num_envs
        OBS, AE, AA, R, D = [], [], [], [], []
        obs = venv.reset()
        for t in range(a.steps):
            if a.policy_actions:
                ae = ops.sample_ego(obs, rng)
                aa = ops.sample_adv(obs, rng)
            else:
                ae = rng.randint(0, n_ego, size=n)
                aa = rng.randint(0, n_adv, size=n)
            OBS.append(obs.copy()); AE.append(ae.copy()); AA.append(aa.copy())
            obs, r_l, r_r, d, infos = venv.step(ops.joint(ae, aa))
            R.append(ops.lbr_reward(r_l, r_r)); D.append(np.asarray(d, bool))
            if (t + 1) % 200 == 0:
                print(f"   {(t+1)*n:,} samples", flush=True)
    finally:
        venv.close()

    R = np.asarray(R); D = np.asarray(D)
    AE = np.asarray(AE); AA = np.asarray(AA)

    # Realized discounted return, valid only where the episode finished.
    T = R.shape[0]
    G = np.zeros_like(R); valid = np.zeros_like(D)
    acc = np.zeros(n); seen = np.zeros(n, bool)
    for t in reversed(range(T)):
        acc = R[t] + gamma * acc * (~D[t])
        seen |= D[t]
        G[t] = acc; valid[t] = seen

    # Q at the joint action actually taken.
    preds = []
    with th.no_grad():
        for t in range(T):
            ob = th.as_tensor(OBS[t]).to(ops.device)
            M = pol.minimax_matrices(ob, buf_num=[head_idx], stop_grad=True)
            b = th.arange(M.shape[0], device=M.device)
            preds.append(M[b, th.as_tensor(AE[t]).to(M.device),
                           th.as_tensor(AA[t]).to(M.device)].cpu().numpy())
    P = np.asarray(preds)

    m = valid.reshape(-1)
    p = P.reshape(-1)[m]; g = G.reshape(-1)[m]
    cell_v = visits[AE.reshape(-1)[m], AA.reshape(-1)[m]]

    def ev(pred, y):
        vy = y.var()
        return float(1.0 - ((y - pred) ** 2).mean() / vy) if vy > 0 else float("nan")

    # Stratify by how well-trained the played cell is.
    order = np.argsort(cell_v)
    deciles = np.array_split(order, 10)
    rows = []
    for i, idx in enumerate(deciles):
        rows.append({
            "decile": i + 1,
            "n": int(idx.size),
            "visits_median": float(np.median(cell_v[idx])),
            "abs_err": float(np.abs(p[idx] - g[idx]).mean()),
            "rmse": float(np.sqrt(((p[idx] - g[idx]) ** 2).mean())),
            "ev": ev(p[idx], g[idx]),
        })

    mode = "policy actions (CONTROL)" if a.policy_actions else "UNIFORM random actions"
    res = {"checkpoint": os.path.basename(a.ckpt), "mode": mode, "gamma": gamma,
           "n_samples": int(m.sum()), "overall_ev": ev(p, g),
           "overall_abs_err": float(np.abs(p - g).mean()),
           "cell_visits": {"min": float(visits.min()), "median": float(np.median(visits)),
                           "max": float(visits.max())},
           "deciles": rows}
    out = os.path.join(REPO_ROOT, a.out)

    print("\n" + "=" * 72)
    print(f"CELL PROBE  {res['checkpoint']}   {mode}")
    print(f"  {res['n_samples']:,} finished samples, gamma={gamma}, "
          f"overall EV {res['overall_ev']:+.4f}")
    print("=" * 72)
    print(f"  {'decile':>7} {'n':>7} {'median visits':>14} {'abs_err':>9} {'rmse':>9} {'EV':>8}")
    for r in rows:
        print(f"  {r['decile']:>7} {r['n']:>7} {r['visits_median']:>14,.0f} "
              f"{r['abs_err']:>9.5f} {r['rmse']:>9.5f} {r['ev']:>+8.4f}")
    # REFUSE to render a verdict on degenerate data. The first smoke run of this
    # script collected 0 finished episodes (60 vec-steps against ~230-650 step
    # episodes), every decile was nan, and the nan comparison fell through to
    # the reassuring branch -- printing "thin cells are NOT the explanation" on
    # no evidence whatsoever. A null result that renders as a confident
    # conclusion is worse than a crash.
    MIN_SAMPLES = 5000
    lo, hi = rows[0], rows[-1]
    ok = (res["n_samples"] >= MIN_SAMPLES
          and np.isfinite(lo["abs_err"]) and np.isfinite(hi["abs_err"])
          and hi["abs_err"] > 0)
    if not ok:
        res["verdict"] = "INSUFFICIENT DATA"
        print(f"\n  INSUFFICIENT DATA: {res['n_samples']:,} finished samples "
              f"(need >= {MIN_SAMPLES:,}). No verdict. Episodes here run "
              f"230-650 steps, so --steps must be large enough for many to "
              f"FINISH -- unfinished episodes have no realized return.")
    else:
        ratio = lo["abs_err"] / hi["abs_err"]
        res["err_ratio_rare_over_common"] = float(ratio)
        print(f"\n  rarest decile vs most-visited: abs_err {lo['abs_err']:.5f} vs "
              f"{hi['abs_err']:.5f}  ({ratio:.2f}x)")
        if ratio > 1.5:
            res["verdict"] = "THIN CELLS WORSE"
            print("  => THIN CELLS ARE WORSE FIT. A null gate result is confounded;")
            print("     random-action injection is worth building before concluding.")
        else:
            res["verdict"] = "FLAT"
            print("  => error is flat across visit counts. Thin cells are NOT the")
            print("     explanation; a null gate result means what it says.")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {out}")
    return res


if __name__ == "__main__":
    main()
