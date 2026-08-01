"""Measure the critic ON- vs OFF-distribution, which is the number that decides
whether LBR's critic tail is salvageable.

LBR scores 22 branches, but 21 of them are successors of actions the policy takes
almost never (measured p_max(adv) ~ 0.98). The critic is trained only on states the
policy actually visits, so those 21 are evaluated off-distribution. Training logs
report explained_variance 0.33-0.70, but that is measured ON the rollout buffer --
it says nothing about the branch states LBR actually cares about.

Method, per probe, all envs in lockstep:
  snapshot -> take the MODAL action -> roll pi to episode end, accumulate the
  discounted return G -> compare against V(s') recorded at the branch.
  restore  -> take a LOW-PROBABILITY action -> same.
Paired by construction: both groups start from the identical state, so the only
difference is which successor the critic is asked to price.

Reports explained variance and RMSE for each group. If off-distribution EV is much
worse than on-distribution, the critic tail is structurally unusable for LBR and no
amount of extra training fixes it -- sharpening the policy makes it worse.
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

import json
import time
import argparse
import numpy as np
import torch as th

from stable_baselines3.common.save_util import load_from_zip_file
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups, splice_terminal,
                                 write_result, REPO_ROOT, ROOT, _b)


def rollout_return(venv, ops, obs, alive, gamma, rng, max_steps):
    """Discounted return from the current state, both sides playing pi."""
    n = venv.num_envs
    G = np.zeros(n)
    disc = np.ones(n)
    for _ in range(max_steps):
        if not alive.any():
            break
        a_ego = ops.sample_ego(obs, rng)
        a_adv = ops.sample_adv(obs, rng)
        lbr_a, pol_a = (a_adv, a_ego) if ops.lbr_is_adv else (a_ego, a_adv)
        obs, r_l, r_r, d, infos = venv.step(ops.joint(lbr_a, pol_a))
        r = ops.lbr_reward(r_l, r_r)
        G += disc * r * alive
        disc *= gamma
        alive = alive & ~np.asarray(d, dtype=bool)
    return G


def probe(venv, ops, args):
    rng = np.random.RandomState(args.seed)
    n = venv.num_envs
    gamma = ops.gamma
    rows = {"modal": {"v": [], "g": [], "f": []},
            "offdist": {"v": [], "g": [], "f": []}}

    obs = venv.reset()
    for _ in range(args.warmup):
        a_e, a_a = ops.sample_ego(obs, rng), ops.sample_adv(obs, rng)
        lbr_a, pol_a = (a_a, a_e) if ops.lbr_is_adv else (a_e, a_a)
        obs, _, _, d, _ = venv.step(ops.joint(lbr_a, pol_a))
        if np.any(d):
            obs = venv.reset()

    for p in range(args.probes):
        # LBR-seat action distribution: modal vs a low-probability action.
        probs = ops.adv_probs(obs) if ops.lbr_is_adv else ops.ego_probs(obs)
        modal = probs.argmax(axis=1)
        # pick the lowest-probability action per env -- maximally off-distribution
        offd = probs.argmin(axis=1)
        opp = ops.sample_ego(obs, rng) if ops.lbr_is_adv else ops.sample_adv(obs, rng)

        venv.env_method("lbr_snapshot", ROOT)
        try:
            for tag, a_lbr in (("modal", modal), ("offdist", offd)):
                venv.env_method("lbr_restore", ROOT)
                o1, r_l, r_r, d, infos = venv.step(ops.joint(a_lbr, opp))
                o1 = splice_terminal(o1, d, infos)
                r0 = ops.lbr_reward(r_l, r_r)
                v = ops.sgn * ops.values_ego(o1)
                alive = ~np.asarray(d, dtype=bool)
                # State features at s', from the LBR seat's point of view.
                # agent_hp is the LEFT player, enemy_hp the RIGHT one, so the
                # sign of the differential flips with which seat LBR holds.
                ahp = np.array([i.get("agent_hp", 0) for i in infos], dtype=np.float64)
                ehp = np.array([i.get("enemy_hp", 0) for i in infos], dtype=np.float64)
                cd = np.array([i.get("round_countdown", 0) for i in infos], dtype=np.float64)
                own, opp_hp = (ehp, ahp) if ops.lbr_is_adv else (ahp, ehp)
                feats = np.stack([own - opp_hp, own, opp_hp, cd], axis=1)
                g_tail = rollout_return(venv, ops, o1, alive, gamma, rng,
                                        args.mc_horizon)
                # V(s') should predict the return FROM s', so compare against the
                # tail only -- r0 is already realized and is not V's job.
                rows[tag]["v"].append(v * alive)
                rows[tag]["g"].append(g_tail)
                rows[tag]["f"].append(feats)
            venv.env_method("lbr_restore", ROOT)
        finally:
            venv.env_method("lbr_drop", ROOT)

        # advance one real step so probes sample distinct states
        a_e, a_a = ops.sample_ego(obs, rng), ops.sample_adv(obs, rng)
        lbr_a, pol_a = (a_a, a_e) if ops.lbr_is_adv else (a_e, a_a)
        obs, _, _, d, _ = venv.step(ops.joint(lbr_a, pol_a))
        if np.any(d):
            obs = venv.reset()
        if (p + 1) % 5 == 0:
            print(f"   probe {p+1}/{args.probes}  ({(p+1)*n*2} paired samples)",
                  flush=True)
    return rows



FEAT_NAMES = ("hp_diff", "own_hp", "opp_hp", "countdown")


def _ev(pred, g):
    var = g.var()
    return float(1.0 - ((pred - g).var() / var)) if var > 1e-12 else float("nan")


def baseline_stats(f, g, seed=0, n_bins=8):
    """Floor and ceiling for how predictable G is from simple state features.

    EV ~ 0 for the critic is only evidence of a BROKEN critic if some other
    predictor does better on the same samples. If a hand-crafted HP model also
    scores ~0, then G is intrinsically unpredictable at this horizon and EV ~ 0
    is the correct answer for any predictor, critic included.

    Everything is scored out-of-sample on a 50/50 split, so a flexible baseline
    cannot win by memorising.
    """
    n = g.size
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    out = {}

    def fit_linear(cols):
        X = np.concatenate([f[:, cols], np.ones((n, 1))], axis=1)
        coef, *_ = np.linalg.lstsq(X[tr], g[tr], rcond=None)
        return _ev(X[te] @ coef, g[te])

    out["hp_diff_linear"] = fit_linear([0])
    out["all_feats_linear"] = fit_linear([0, 1, 2, 3])

    # Non-parametric ceiling: bin by hp_diff, predict the training bin mean.
    # This is the best any function of hp_diff alone can do.
    edges = np.quantile(f[tr, 0], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b_tr = np.digitize(f[tr, 0], edges[1:-1])
    b_te = np.digitize(f[te, 0], edges[1:-1])
    means = np.array([g[tr][b_tr == b].mean() if (b_tr == b).any() else g[tr].mean()
                      for b in range(n_bins)])
    out["hp_diff_binned"] = _ev(means[b_te], g[te])

    # Absolute ceiling on ANY predictor, estimated in-sample: the share of
    # Var(G) that hp_diff bins can account for at all.
    b_all = np.digitize(f[:, 0], edges[1:-1])
    mu = np.array([g[b_all == b].mean() if (b_all == b).any() else g.mean()
                   for b in range(n_bins)])
    out["ceiling_hp_diff_insample"] = float(mu[b_all].var() / g.var()) if g.var() > 1e-12 else float("nan")
    return out


def stats(v, g):
    v = np.concatenate(v); g = np.concatenate(g)
    err = v - g
    var_g = g.var()
    ev = 1.0 - err.var() / var_g if var_g > 1e-12 else float("nan")
    return {"n": int(v.size), "ev": float(ev), "rmse": float(np.sqrt((err**2).mean())),
            "bias": float(err.mean()), "v_mean": float(v.mean()),
            "v_std": float(v.std()), "g_mean": float(g.mean()),
            "g_std": float(g.std())}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Critic on- vs off-distribution calibration")
    ap.add_argument("--main_checkpoint_model_path", type=str, required=True)
    ap.add_argument("--eval_prot", type=str, default="True")
    ap.add_argument("--lbr_matchups", type=str, default="all")
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--probes", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--mc_horizon", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="critic_calibration.json")
    args = ap.parse_args(argv)

    data = load_from_zip_file(args.main_checkpoint_model_path, device="cpu")[0]
    out = {"checkpoint": os.path.basename(args.main_checkpoint_model_path),
           "num_timesteps": data.get("num_timesteps"), "matchups": {}}

    for head_idx, label, state in resolve_matchups(data, args.lbr_matchups):
        print(f"\n=== {label} (head {head_idx}) ===")
        venv = build_lbr_venv(state, args.n_envs)
        try:
            model, _ = load_checkpoint(args.main_checkpoint_model_path, venv, args.device)
            preflight(venv, model)
            ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=_b(args.eval_prot))
            t0 = time.time()
            rows = probe(venv, ops, args)
        finally:
            venv.close()
        m = stats(rows["modal"]["v"], rows["modal"]["g"])
        o = stats(rows["offdist"]["v"], rows["offdist"]["g"])
        base = {}
        for tag in ("modal", "offdist"):
            V = np.concatenate(rows[tag]["v"])
            G = np.concatenate(rows[tag]["g"])
            F = np.concatenate(rows[tag]["f"])
            b = baseline_stats(F, G, seed=args.seed)
            b["critic"] = _ev(V, G)
            # Stratify by time remaining: a state 20 steps from a KO is far more
            # predictable than one 200 steps out, and pooling them hides that.
            med = np.median(F[:, 3])
            for lab, sel in (("late", F[:, 3] <= med), ("early", F[:, 3] > med)):
                if sel.sum() > 40:
                    bb = baseline_stats(F[sel], G[sel], seed=args.seed)
                    b[f"{lab}_critic"] = _ev(V[sel], G[sel])
                    b[f"{lab}_hp_binned"] = bb["hp_diff_binned"]
                    b[f"{lab}_ceiling"] = bb["ceiling_hp_diff_insample"]
            base[tag] = b
        out["matchups"][label] = {"on_distribution": m, "off_distribution": o,
                                  "baselines": base, "gamma": ops.gamma,
                                  "wall_clock_s": round(time.time() - t0, 1)}
        print(f"\n   {'':22s} {'EV':>8s} {'RMSE':>9s} {'bias':>9s} "
              f"{'V std':>9s} {'G std':>9s} {'n':>7s}")
        for tag, s in (("ON-distribution (modal)", m), ("OFF-distribution (min-p)", o)):
            print(f"   {tag:22s} {s['ev']:8.3f} {s['rmse']:9.4f} {s['bias']:+9.4f} "
                  f"{s['v_std']:9.4f} {s['g_std']:9.4f} {s['n']:7d}")
        print(f"   EV degradation off-distribution: "
              f"{m['ev'] - o['ev']:+.3f}   ({args.probes} probes, "
              f"{out['matchups'][label]['wall_clock_s']}s)")
        print()
        print(f"   {'FLOOR/CEILING (out-of-sample EV)':38s} {'on-dist':>9s} {'off-dist':>9s}")
        for k in ("critic", "hp_diff_linear", "all_feats_linear", "hp_diff_binned",
                  "ceiling_hp_diff_insample", "late_critic", "late_hp_binned",
                  "late_ceiling", "early_critic", "early_hp_binned", "early_ceiling"):
            a = base["modal"].get(k); c = base["offdist"].get(k)
            if a is None and c is None:
                continue
            fa = f"{a:9.3f}" if a is not None else f"{'-':>9s}"
            fc = f"{c:9.3f}" if c is not None else f"{'-':>9s}"
            print(f"   {k:38s} {fa} {fc}")
        print("   -> if hp_diff_binned ~ critic ~ 0, G is unpredictable and the")
        print("      critic is not at fault; if hp_diff_binned >> critic, it is.")

    p = os.path.join(REPO_ROOT, args.out)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
