"""Is the value head leaving EV on the table its OWN trunk already contains?

THE GAP THIS EXISTS TO MEASURE. critic_ceiling's gamma sweep says a RIDGE on the
critic's frozen features reaches held-out EV ~0.074 at gamma 0.75-0.9 and ~0.065
at the run's 0.94. The trained value head is on record at held-out EV ~0. Nobody
has put those two numbers side by side AT THE SAME GAMMA, ON THE SAME SPLITS,
FROM THE SAME ROLLOUT -- and until that is done, "critic EV ~ 0 is correct given
the ceiling" and "the head is failing to extract what its trunk contains" are
indistinguishable. They call for completely different work: the first is a dead
end, the second is a ~0.065 optimisation target.

Three measurements, one rollout, so they cannot disagree for bookkeeping reasons:

  head    the trained value head's own V(s) against realized returns
  ridge   a ridge on the head's FROZEN features -- the ceiling the head is
          failing (or not) to reach
  const   predict the training-split mean return. EV 0 by construction; both of
          the above must beat it or neither means anything

and two sweeps over the TARGET rather than the predictor:

  gamma   predictability is an INVERTED U (measured: 0.030 0.044 0.060 0.074
          0.072 0.063 0.023 0.002 at gamma 0 .25 .5 .75 .9 .95 .99 1). Longer
          horizons accumulate opponent sampling noise faster than signal
          (G_std 0.0057 -> 0.0418). The run sits at 0.94, past the peak.
  lambda  never swept for value quality -- only discussed for the minimax
          bootstrap. lambda trades bias for variance in the target, and when the
          target is ~93% noise that trade should favour lower lambda.

EPISODE-LEVEL SPLITS, NOT TIMESTEP. Every timestep inside one episode shares a
return; a timestep split leaks and inflated probe scores ~5x in this codebase,
producing two wrong conclusions. A frozen actor-CNN ridge read +0.166 under a
timestep split and +0.005 under an episode split on the same data.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--gammas", default="0.5,0.75,0.9,0.94,0.99")
    ap.add_argument("--lambdas", default="0.3,0.6,0.8,0.95,1.0")
    ap.add_argument("--lams", default="1e-2,1e-1,1,10,100,1e3,1e4,1e5,1e6,1e7")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="value_gap.json")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from local_best_response import (build_lbr_venv, load_checkpoint, PolicyOps,
                                     resolve_matchups, infer_obs_kwargs)
    from stable_baselines3.common.save_util import load_from_zip_file

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    # resolve_matchups returns (head_idx, label, state). Indexing [1] gets the
    # LABEL, which retro.make then fails to open as a state file -- and because
    # that happens inside a SubprocVecEnv2P worker, the parent only sees the pipe
    # close (ConnectionResetError), not the real TypeError. Unpack by name.
    head_idx, _label, st = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(st, a.n_envs, **infer_obs_kwargs(data, a.ram_mask or None))
    loaded = load_checkpoint(a.ckpt, venv, a.device)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    # head_idx from the checkpoint, NOT hardcoded 0. They coincide only while
    # num_adversaries == 1; the same "correct by accident of configuration"
    # hazard that value_forward already carries.
    ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
    rng = np.random.RandomState(0)

    # ---- rollout: obs, reward, value, features, episode id ------------------
    OBS, REW, VAL, EP, DONE = [], [], [], [], []
    obs = venv.reset()
    ep_id = np.arange(a.n_envs)
    nxt = a.n_envs
    for t in range(a.max_steps):
        v = ops.values_ego(obs)
        OBS.append(np.asarray(obs).copy()); VAL.append(np.asarray(v).reshape(-1))
        EP.append(ep_id.copy())
        obs, r_l, r_r, d, _ = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                                  ops.sample_adv(obs, rng)))
        REW.append(np.asarray(ops.lbr_reward(r_l, r_r)).reshape(-1))
        DONE.append(np.asarray(d).reshape(-1).astype(bool))
        for k in np.nonzero(d)[0]:
            ep_id[k] = nxt; nxt += 1
        if nxt - a.n_envs >= a.episodes:
            break
    venv.close()
    OBS = np.stack(OBS); REW = np.stack(REW); VAL = np.stack(VAL)
    EP = np.stack(EP); DONE = np.stack(DONE)
    T, N = REW.shape
    print(f"[gap] {T} steps x {N} envs, {nxt - a.n_envs} completed episodes\n")
    if nxt - a.n_envs < 12:
        raise SystemExit(f"FAILED: only {nxt - a.n_envs} episodes; episode splits "
                         f"need far more. Raise --max_steps.")

    def mc_returns(gamma):
        """MONTE CARLO return from REWARDS ONLY. No bootstrap anywhere.

        The first version of this scored V against a lambda-return that used V as
        its own bootstrap, and duly reported HEAD EV 0.81 at gamma 0.99 against a
        recorded held-out EV of ~0.023. That is the documented Bellman
        self-consistency trap (V_scalar once read EV 0.944 against a one-step
        backup of ITSELF). The lambda-dependence gave it away: 0.81 at lambda 0.3
        decaying to 0.36 at lambda 1.0 as the bootstrap weight fell.

        `valid` marks only steps whose episode COMPLETES inside the rollout --
        a truncated tail would have to be bootstrapped, which is the same
        contamination in a smaller dose.
        """
        G = np.zeros((T, N)); valid = np.zeros((T, N), bool)
        for n in range(N):
            run = 0.0; seen = False
            for t in reversed(range(T)):
                if DONE[t, n]:
                    run = 0.0; seen = True
                run = REW[t, n] + gamma * run
                G[t, n] = run; valid[t, n] = seen
        return G, valid

    dev = th.device(a.device if th.cuda.is_available() else "cpu")
    lams = [float(x) for x in a.lams.split(",") if x.strip()]

    # ---- episode-level split ------------------------------------------------
    eps = np.unique(EP); rng.shuffle(eps)
    n_te = max(3, len(eps) // 4); n_va = max(2, len(eps) // 6)
    te_s, va_s, tr_s = set(eps[:n_te]), set(eps[n_te:n_te+n_va]), set(eps[n_te+n_va:])
    tr = np.isin(EP, list(tr_s)); va = np.isin(EP, list(va_s)); te = np.isin(EP, list(te_s))
    print(f"[gap] episodes train/val/test = {len(tr_s)}/{len(va_s)}/{len(te_s)}  "
          f"samples {tr.sum()}/{va.sum()}/{te.sum()}")

    # Frozen CRITIC features. No public accessor exists, so mirror the value
    # path the same way PolicyOps mirrors ego_forward -- inventing an accessor
    # would silently read the actor's encoder instead and the "ceiling" would be
    # the wrong network's.
    @th.no_grad()
    def value_features(x):
        from stable_baselines3.common.preprocessing import preprocess_obs
        p = model.policy
        out = []
        for i0 in range(0, x.shape[0], 4096):
            t = th.as_tensor(x[i0:i0 + 4096], device=dev)
            z = preprocess_obs(t, p.observation_space,
                               normalize_images=p.normalize_images)
            f = p.vf_features_extractor(z)
            out.append(f.reshape(f.shape[0], -1).double().cpu().numpy())
        return np.concatenate(out)

    try:
        feats = value_features(OBS.reshape((T * N,) + OBS.shape[2:]))
        print(f"[gap] critic features: {feats.shape[1]} dims")
    except Exception as e:
        print(f"[gap] NO critic-feature accessor ({type(e).__name__}: {e}); "
              f"ridge arm will be NaN and the GAP is UNMEASURED", flush=True)
        feats = None

    def ev(pred, targ, m):
        p, y = pred[m], targ[m]
        return float(1 - ((y - p) ** 2).mean() / max(((y - y[..., None].mean()) ** 2).mean(), 1e-30))

    def ridge_ev(X, y, valid=None):
        Xt = th.as_tensor(X, dtype=th.float64, device=dev)
        _m0 = (tr & (np.ones((T, N), bool) if valid is None else valid)).reshape(-1)
        sd = Xt[_m0].std(0); alive = sd > 1e-9
        Xt = (Xt[:, alive] - Xt[_m0][:, alive].mean(0)) / sd[alive]
        yt = th.as_tensor(y.reshape(-1), dtype=th.float64, device=dev)
        vv = np.ones((T, N), bool) if valid is None else valid
        f_tr = (tr & vv).reshape(-1); f_va = (va & vv).reshape(-1)
        f_te = (te & vv).reshape(-1)
        K = Xt[f_tr] @ Xt[f_tr].T
        w, V = th.linalg.eigh(K); w = w.clamp_min(0)
        ymu = yt[f_tr].mean(); VtY = V.T @ (yt[f_tr] - ymu)
        KvV, KtV = (Xt[f_va] @ Xt[f_tr].T) @ V, (Xt[f_te] @ Xt[f_tr].T) @ V
        best = (-1e18, None, None)
        for lam in lams:
            dg = 1.0 / (w + lam * len(K))
            pv = KvV @ (dg[:, None] * VtY[:, None]).reshape(-1) + ymu
            r = float(1 - ((yt[f_va] - pv) ** 2).sum() / ((yt[f_va] - ymu) ** 2).sum())
            if np.isfinite(r) and r > best[0]:
                best = (r, lam, KtV @ (dg[:, None] * VtY[:, None]).reshape(-1) + ymu)
        _, lam, pt = best
        yte = yt[f_te]
        return float(1 - ((yte - pt) ** 2).sum() / ((yte - ymu) ** 2).sum()), lam

    out = {"n_episodes": int(nxt - a.n_envs), "arms": []}
    print(f"\n  MC targets (no bootstrap). LAMBDA IS NOT SWEPT HERE: lambda is a")
    print(f"  property of the TRAINING target, and any offline lambda-return")
    print(f"  reintroduces V into its own target. It needs training arms.\n")
    print(f"  {'gamma':>6} {'n_valid':>8} {'HEAD EV':>9} {'RIDGE EV':>9} {'gap':>8} "
          f"{'std(V)':>8} {'slope':>7} {'|res|p50':>9} {'|res|p95':>9}")
    for g in [float(x) for x in a.gammas.split(",")]:
        G, valid = mc_returns(g)
        m_te = te & valid; m_tr = tr & valid
        if m_te.sum() < 200 or m_tr.sum() < 200:
            print(f"  {g:6.2f}  too few completed-episode steps "
                  f"({m_tr.sum()}/{m_te.sum()})"); continue
        ymu_tr = G[m_tr].mean()
        head = float(1 - ((G[m_te] - VAL[m_te]) ** 2).mean()
                     / max(((G[m_te] - ymu_tr) ** 2).mean(), 1e-30))
        r = float("nan")
        if feats is not None:
            r, _ = ridge_ev(feats, G, valid)
        res = np.abs(G[m_te] - VAL[m_te])
        # SHRINKAGE probe: slope of realized return regressed on V. slope<1 is
        # optimal shrinkage, not miscalibration; a Huber arm shrinking FURTHER
        # would show a lower slope than its MSE control.
        vv = VAL[m_te]; gg = G[m_te]
        slope = float(np.cov(gg, vv)[0, 1] / max(np.var(vv), 1e-30))
        print(f"  {g:6.2f} {int(m_te.sum()):8d} {head:9.4f} {r:9.4f} {r - head:8.4f} "
              f"{float(vv.std()):8.5f} {slope:7.3f} {float(np.percentile(res,50)):9.5f} "
              f"{float(np.percentile(res,95)):9.5f}")
        out["arms"].append(dict(gamma=g, head_ev=head, ridge_ev=r, std_v=float(vv.std()),
                                ret_std=float(gg.std()),
                                slope=slope, res_p50=float(np.percentile(res, 50)),
                                res_p95=float(np.percentile(res, 95)),
                                n_test=int(m_te.sum())))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  HEAD  = the trained value head's own V(s), held-out episodes")
    print(f"  RIDGE = ridge on its FROZEN features -- the ceiling it should reach")
    print(f"  a large positive gap means an OPTIMISATION problem with a concrete")
    print(f"  target; a gap near zero means the head is already at its ceiling and")
    print(f"  the low EV is the horizon/aleatoric limit, not the head.")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
