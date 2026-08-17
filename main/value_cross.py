"""CROSS-EVALUATION: score several value heads on the SAME states.

WHY value_gap.py IS NOT ENOUGH. It scores each checkpoint on ITS OWN rollout, so
the Huber head is measured on states the Huber policy visits and the MSE head on
states the MSE policy visits. Those distributions differ (measured: ep_rew -0.082
vs -0.044, ep_len 191 vs 236), which conflates the two things we want separated:

    did the LOSS make V a better predictor?
    or did the REGIME become more predictable?

A more one-sided or shorter-episode regime has lower-variance returns, so its EV
rises for reasons that have nothing to do with the loss. That is structurally the
same error that made `engaged` look 12x better than healthy self-play until the
CONST baseline showed two thirds of it was free.

Here ONE rollout produces ONE set of states and ONE set of MC targets, and every
head is scored against them. The only thing that differs between columns is the
network.

RUN IT IN A 2x2 (plus neutral). If Huber wins on BOTH arms' distributions AND on
a policy-free random one, it is the loss. If it wins only on its own, it is the
regime. --dist_random gives the neutral distribution that is neither arm's home
turf; note that off-distribution scoring penalises BOTH heads, which is why the
comparison is across heads within a column, never across columns.

TARGETS ARE MONTE CARLO, rewards only, completed episodes only. A lambda-return
would bootstrap from one of the heads being scored and hand that head its own
prediction back -- that mistake produced HEAD EV 0.81 against a recorded ~0.023.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist_ckpt", required=True,
                    help="checkpoint whose POLICY generates the rollout")
    ap.add_argument("--score_ckpts", nargs="+", required=True,
                    help="checkpoints whose VALUE HEADS are scored on it")
    ap.add_argument("--dist_random", action="store_true",
                    help="uniform-random actions instead of dist_ckpt's policy -- "
                         "the neutral distribution, neither arm's home turf")
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--gammas", default="0.9,0.94")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="value_cross.json")
    a = ap.parse_args(argv)

    import numpy as np
    from local_best_response import (build_lbr_venv, load_checkpoint, PolicyOps,
                                     resolve_matchups, infer_obs_kwargs)
    from stable_baselines3.common.save_util import load_from_zip_file

    data = load_from_zip_file(a.dist_ckpt, device="cpu")[0]
    head_idx, _lab, st = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(st, a.n_envs, **infer_obs_kwargs(data, a.ram_mask or None))

    def _load(p):
        L = load_checkpoint(p, venv, a.device)
        m = L[0] if isinstance(L, tuple) else L
        return PolicyOps(m, head_idx=head_idx, lbr_is_adv=False)

    ops_d = _load(a.dist_ckpt)
    rng = np.random.RandomState(0)
    na = int(ops_d.n_actions)

    OBS, REW, EP, DONE = [], [], [], []
    obs = venv.reset(); ep_id = np.arange(a.n_envs); nxt = a.n_envs
    for t in range(a.max_steps):
        OBS.append(np.asarray(obs).copy()); EP.append(ep_id.copy())
        if a.dist_random:
            act = ops_d.joint(rng.randint(na, size=a.n_envs),
                              rng.randint(na, size=a.n_envs))
        else:
            act = ops_d.joint(ops_d.sample_ego(obs, rng), ops_d.sample_adv(obs, rng))
        obs, r_l, r_r, d, _ = venv.step(act)
        REW.append(np.asarray(ops_d.lbr_reward(r_l, r_r)).reshape(-1))
        DONE.append(np.asarray(d).reshape(-1).astype(bool))
        for k in np.nonzero(d)[0]:
            ep_id[k] = nxt; nxt += 1
        if nxt - a.n_envs >= a.episodes:
            break
    OBS = np.stack(OBS); REW = np.stack(REW); EP = np.stack(EP); DONE = np.stack(DONE)
    T, N = REW.shape
    n_ep = nxt - a.n_envs
    src = "RANDOM (neutral)" if a.dist_random else os.path.basename(a.dist_ckpt)
    print(f"[cross] distribution = {src}: {T}x{N}, {n_ep} episodes")
    if n_ep < 40:
        raise SystemExit(f"FAILED: only {n_ep} episodes; raise --max_steps.")

    eps = np.unique(EP); rng.shuffle(eps)
    n_te = max(3, len(eps) // 3)
    te_s = set(eps[:n_te]); tr = ~np.isin(EP, list(te_s)); te = np.isin(EP, list(te_s))

    def mc(gamma):
        G = np.zeros((T, N)); ok = np.zeros((T, N), bool)
        for n in range(N):
            run = 0.0; seen = False
            for t in reversed(range(T)):
                if DONE[t, n]: run = 0.0; seen = True
                run = REW[t, n] + gamma * run
                G[t, n] = run; ok[t, n] = seen
        return G, ok

    # Score every head on the SAME stored observations. The value heads never see
    # the env again, so nothing about the rollout differs between columns.
    heads = {}
    for p in a.score_ckpts:
        o = _load(p)
        V = np.concatenate([o.values_ego(OBS.reshape((T * N,) + OBS.shape[2:])[i:i + 8192])
                            for i in range(0, T * N, 8192)]).reshape(T, N)
        heads[os.path.basename(p)] = V
        del o
    venv.close()

    out = {"distribution": src, "n_episodes": int(n_ep), "rows": []}
    print(f"\n  {'gamma':>6} {'head':>44} {'EV':>9} {'std(V)':>9} {'slope':>7}")
    for g in [float(x) for x in a.gammas.split(",")]:
        G, ok = mc(g)
        m_te = te & ok; m_tr = tr & ok
        if m_te.sum() < 300:
            print(f"  {g:6.2f}  too few completed-episode test steps ({m_te.sum()})")
            continue
        ymu = G[m_tr].mean()
        den = ((G[m_te] - ymu) ** 2).mean()
        for name, V in heads.items():
            ev = float(1 - ((G[m_te] - V[m_te]) ** 2).mean() / max(den, 1e-30))
            sl = float(np.cov(G[m_te], V[m_te])[0, 1] / max(np.var(V[m_te]), 1e-30))
            print(f"  {g:6.2f} {name[-44:]:>44} {ev:9.4f} {float(V[m_te].std()):9.5f} {sl:7.3f}")
            out["rows"].append(dict(gamma=g, head=name, ev=ev,
                                    std_v=float(V[m_te].std()), slope=sl,
                                    n_test=int(m_te.sum())))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Compare heads WITHIN a column (same states, same targets). Comparing\n"
          f"  ACROSS distributions is not meaningful -- off-distribution scoring\n"
          f"  penalises every head, which is why the 2x2 is run at all.")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
