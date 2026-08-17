"""Capacity, features, or target noise? The three-way test for the value function.

THE GAP TO EXPLAIN. The corrected ceiling says a perfect value function scores
EV 0.59 at gamma 0.94 (value_ceiling.py, 480 roots x 16 genuine replays). The
trained head scores ~0 and a ridge on its frozen features ~0.074. Something is
costing ~0.5 EV and the candidates call for opposite work:

    capacity      the head's model class cannot express E[G|s]
    features      the trunk latent does not CONTAIN E[G|s]
    target noise  both could fit it, but single-sample returns are too noisy
                  to learn from (41% of return variance is within-state)

WHY THE EXISTING NUMBERS CANNOT SEPARATE THESE. Every previous score -- the
head's ~0, the ridge's 0.074 -- was measured against SINGLE-SAMPLE returns, whose
own ceiling is 0.59. So "ridge gets 0.074" conflates "the features lack the
information" with "the ridge was fitted to noise". Those are different claims and
this project has been treating them as one.

THE FIX. value_ceiling replays each root K times, so G.mean(axis=1) is a
LOW-NOISE estimate of E[G|s] -- variance cut by K. Regress on THAT and the
target-noise term is largely removed, leaving capacity and features to be
separated by model class:

    ridge on frozen features  -> ~0.5   features fine; it was TARGET NOISE
    ridge ~0.1, MLP ~0.5      -> CAPACITY / model class is the limit
    both ~0.1                 -> FEATURES are the limit; a bigger head cannot help

CONTROLS. Every arm is also scored against the SINGLE-sample target on the same
split, so the target-noise contribution is visible rather than argued. And a
shuffled-feature arm gives the floor: any score at or below it is not real.

Splits are by ROOT BLOCK, not random: roots are collected along a rollout in
groups of n_envs, so neighbouring roots are near-duplicates and a random split
leaks -- the same mechanism that inflated probe scores ~5x here before.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="*_raw.npz from value_ceiling.py")
    ap.add_argument("--gamma_idx", type=int, default=2, help="index into saved gammas")
    ap.add_argument("--lams", default="1e-3,1e-2,1e-1,1,10,100,1e3,1e4,1e5,1e6,1e7")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--wds", default="1e-4,1e-3,1e-2")
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="value_bottleneck.json")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th

    d = np.load(a.npz)
    G, V, OBS = d["G"], d["V"], d["OBS"]
    gam = float(d["gammas"][a.gamma_idx])
    Gg = G[:, :, a.gamma_idx]
    y_clean = Gg.mean(axis=1)            # low-noise E[G|s], variance cut by K
    y_noisy = Gg[:, 0]                   # one sample: the ordinary target
    ns = len(y_clean)
    print(f"[bn] {ns} roots, K={Gg.shape[1]}, gamma={gam}, obs {OBS.shape[1]}\n")

    # Blocked split: roots arrive in groups of n_envs along one rollout.
    nb = a.blocks
    edges = np.linspace(0, ns, nb + 1).astype(int)
    tr = np.zeros(ns, bool); va = np.zeros(ns, bool); te = np.zeros(ns, bool)
    for b in range(nb):
        sl = slice(edges[b], edges[b + 1])
        (te if b % 4 == 0 else va if b % 4 == 1 else tr)[sl] = True
    print(f"[bn] blocked split train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}")

    dev = th.device(a.device if th.cuda.is_available() else "cpu")
    lams = [float(x) for x in a.lams.split(",")]
    wds = [float(x) for x in a.wds.split(",")]
    X0 = th.as_tensor(OBS, dtype=th.float64, device=dev)
    sd = X0[tr].std(0); alive = sd > 1e-9
    X = (X0[:, alive] - X0[tr][:, alive].mean(0)) / sd[alive]
    rng = np.random.RandomState(0)
    Xs = X[th.as_tensor(rng.permutation(ns), device=dev)]     # shuffled control
    print(f"[bn] {int(alive.sum())}/{OBS.shape[1]} obs bytes vary on train\n")

    def ev(pred, y, m, ymu):
        return float(1 - ((y[m] - pred) ** 2).mean() / max(((y[m] - ymu) ** 2).mean(), 1e-30))

    def ridge(y, Xm, tag):
        yt = th.as_tensor(y, dtype=th.float64, device=dev)
        Xtr = Xm[tr]; ymu = yt[tr].mean()
        K = Xtr @ Xtr.T
        w, Vv = th.linalg.eigh(K); w = w.clamp_min(0)
        VtY = Vv.T @ (yt[tr] - ymu)
        Kv, Kt = (Xm[va] @ Xtr.T) @ Vv, (Xm[te] @ Xtr.T) @ Vv
        best = (-1e18, None, None)
        for lam in lams:
            g = 1.0 / (w + lam * len(K))
            pv = Kv @ (g * VtY) + ymu
            r = float(1 - ((yt[va] - pv) ** 2).sum() / ((yt[va] - ymu) ** 2).sum())
            if np.isfinite(r) and r > best[0]:
                best = (r, lam, Kt @ (g * VtY) + ymu)
        _, lam, pt = best
        if lam == min(lams):
            raise SystemExit(f"FAILED: {tag} picked the MINIMUM lambda; path truncated.")
        return ev(pt, yt, te, ymu), lam

    def mlp(y, Xm, tag):
        yt = th.as_tensor(y, dtype=th.float32, device=dev)
        Xf = Xm.float(); ymu = yt[tr].mean()
        out = []
        for s_ in range(a.seeds):
            best = (-1e18, None, None)
            for wd in wds:
                th.manual_seed(1000 * s_ + 3)
                net = th.nn.Sequential(
                    th.nn.Linear(Xf.shape[1], a.hidden), th.nn.ReLU(),
                    th.nn.Linear(a.hidden, a.hidden), th.nn.ReLU(),
                    th.nn.Linear(a.hidden, 1)).to(dev)
                opt = th.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
                for ep in range(a.epochs):
                    net.train()
                    loss = th.nn.functional.mse_loss(net(Xf[tr]).squeeze(-1), yt[tr] - ymu)
                    opt.zero_grad(); loss.backward(); opt.step()
                    if ep % 10 == 0:
                        net.eval()
                        with th.no_grad():
                            r = ev(net(Xf[va]).squeeze(-1) + ymu, yt, va, ymu)
                        if r > best[0]:
                            with th.no_grad():
                                best = (r, wd, net(Xf[te]).squeeze(-1) + ymu)
            out.append(ev(best[2], yt, te, ymu))
        return float(np.mean(out)), float(np.std(out))

    print(f"  {'target':>18} {'model':>16} {'EV':>9} {'shuffled':>10}")
    res = {}
    for yname, y in (("CLEAN (K-avg)", y_clean), ("NOISY (1 sample)", y_noisy)):
        r_ev, lam = ridge(y, X, f"ridge/{yname}")
        r_sh, _ = ridge(y, Xs, f"ridgeshuf/{yname}")
        print(f"  {yname:>18} {'ridge':>16} {r_ev:9.4f} {r_sh:10.4f}")
        m_ev, m_sd = mlp(y, X, f"mlp/{yname}")
        m_sh, _ = mlp(y, Xs, f"mlpshuf/{yname}")
        print(f"  {yname:>18} {'MLP':>16} {m_ev:9.4f} {m_sh:10.4f}   (+-{m_sd:.3f} over seeds)")
        res[yname] = dict(ridge=r_ev, ridge_shuf=r_sh, mlp=m_ev, mlp_shuf=m_sh, mlp_sd=m_sd)

    # the trained head, on the same split, against both targets
    for yname, y in (("CLEAN (K-avg)", y_clean), ("NOISY (1 sample)", y_noisy)):
        yt = th.as_tensor(y, dtype=th.float64, device=dev)
        ymu = yt[tr].mean()
        h = ev(th.as_tensor(V, dtype=th.float64, device=dev)[te], yt, te, ymu)
        print(f"  {yname:>18} {'TRAINED HEAD':>16} {h:9.4f}")
        res[yname]["trained_head"] = h

    c = res["CLEAN (K-avg)"]
    print(f"\n  VERDICT (clean target, gamma={gam}):")
    if c["ridge"] > 0.35:
        print("    ridge on frozen features already recovers most of the ceiling")
        print("    => FEATURES ARE FINE. The limit was TARGET NOISE, not capacity.")
    elif c["mlp"] > c["ridge"] + 0.15:
        print("    MLP >> ridge on the same features")
        print("    => MODEL CLASS / CAPACITY is the limit; a bigger head helps.")
    else:
        print("    neither model class recovers it from these features")
        print("    => FEATURES are the limit; head capacity cannot help.")
    with open(a.out, "w") as f:
        json.dump({"gamma": gam, "n_roots": int(ns), "res": res}, f, indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
