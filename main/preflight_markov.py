"""PHASE -1b: is a single masked RAM frame a SUFFICIENT STATISTIC for the dynamics?

Phase -1 answered DISTINGUISHABILITY -- can the observation tell the 484
successors of one state apart. It can: 99.95% of the achievable ceiling from a
single frame. But distinguishability is NECESSARY, NOT SUFFICIENT. A byte can be
present and discriminative while some byte the mask DROPPED still drives the
dynamics, in which case the observation is still a POMDP and history helps.

That is what this measures, and it is the operational form of the MDP claim:

    If obs_t is Markov, then (obs_t, a_t) determines obs_{t+1}, and CONDITIONING
    ON HISTORY CANNOT IMPROVE THE PREDICTION. Any improvement from adding past
    frames is exactly the non-Markovianity that stacking buys.

So: predict the next masked RAM frame from (window, action), sweeping the window
depth. Policy-free -- uniform random actions, one rollout, re-sliced per arm, so
no checkpoint and no training confound.

CONTROLS, because "more features scored better" is the oldest artifact in this
programme and has already fooled us three times:

  shuffled   the k=12 window with its history frames taken from a RANDOM OTHER
             timestep. Same feature count, same marginal distribution, temporal
             link destroyed. If ordered k=12 does not beat SHUFFLED k=12, the
             gain is CAPACITY, not history, and it is not evidence of a POMDP.
  const      predict each byte's training mean. The floor every EV must clear.

Episode-level train/val/test splits. Timestep splits inflated probe scores ~5x
in this codebase and produced two wrong conclusions; consecutive frames of an
8-frame agent step are about as correlated as two samples can be.

READING IT. The headline is (ordered k=12) - (k=1) on HELD-OUT episodes, with
(ordered k=12) - (shuffled k=12) next to it. If both are ~0, a single frame is
Markov for the dynamics and stacking cannot help. If the first is large and the
second is ~0, the win is parameters, not memory. Only both being large is
evidence for stacking.

Reported per byte as well as in aggregate: counters increment deterministically
and are trivially predictable from one frame, so a mean over 2105 bytes hides
whatever is happening on the handful that matter. The byte-level table is the
diagnostic -- if history helps, it should help on a SPECIFIC, nameable set of
addresses, not diffusely.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WINDOW = 12          # emulator frames retained; stride 1 => contiguous


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="Champion.Level1.RyuVsRyu.2Player")
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--ep_len", type=int, default=200, help="agent steps per episode")
    ap.add_argument("--ks", default="1,2,4,12")
    ap.add_argument("--lams", default="1e-6,1e-5,1e-4,1e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100,300,1e3,3e3,1e4,3e4,1e5,1e6")
    ap.add_argument("--min_std", type=float, default=1.0,
                    help="a target byte must move at least this many RAM UNITS "
                         "(obs is /255) on BOTH the train and test splits. The "
                         "first pass used >1e-12 variance, which admitted bytes "
                         "whose test variance was ~0 and whose per-byte EV then "
                         "read -359164. Aggregates were fine; the byte-level "
                         "table was not.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="preflight_markov.json")
    a = ap.parse_args(argv)

    global np
    import numpy as np
    import torch as th
    from local_best_response import make_lbr_env

    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    lams = [float(x) for x in a.lams.split(",") if x.strip()]
    if max(ks) > WINDOW:
        raise SystemExit(f"window is {WINDOW} frames; k={max(ks)} does not fit")

    mask = np.load(a.ram_mask)
    rng = np.random.RandomState(a.seed)
    env = make_lbr_env(a.state, obs_type="ram", ram_mask=mask,
                       ram_stack=WINDOW, ram_stride=1, seed=a.seed)()
    na = env.lbr_config()["n_actions"]
    nb = int(mask.size)
    print(f"[markov] {na} actions/player, {nb} masked bytes, "
          f"{a.episodes} eps x {a.ep_len} steps", flush=True)

    # ---- rollout -----------------------------------------------------------
    # W[t] : the WINDOW contiguous emulator frames ending at agent step t
    # A[t] : the joint action taken AT step t
    # target for sample t is the newest frame of W[t+1]
    W, A, EP = [], [], []
    for ep in range(a.episodes):
        env.reset()
        env.step(np.array([rng.randint(na), rng.randint(na)]))   # fill the tap
        for t in range(a.ep_len):
            W.append(env.ram_tail(WINDOW).copy())
            act = np.array([rng.randint(na), rng.randint(na)])
            A.append(act)
            EP.append(ep)
            env.step(act)
        if (ep + 1) % 6 == 0:
            print(f"[markov] {ep+1}/{a.episodes} episodes", flush=True)
    env.close()

    W = np.stack(W)                       # (N, WINDOW, nb), oldest..newest
    A = np.stack(A)
    EP = np.asarray(EP)
    # drop the last step of each episode: it has no successor inside the episode
    keep = np.concatenate([EP[1:] == EP[:-1], [False]])
    Y = W[1:, -1, :]                                   # next newest frame
    Y = np.concatenate([Y, np.zeros((1, nb), np.float32)])[keep]
    Wk, Ak, EPk = W[keep], A[keep], EP[keep]
    N = len(Wk)
    print(f"[markov] {N} usable transitions", flush=True)

    # ---- splits, BY EPISODE ------------------------------------------------
    eps = np.unique(EPk); rng.shuffle(eps)
    n_te = max(2, len(eps) // 4); n_va = max(2, len(eps) // 6)
    te, va, tr = set(eps[:n_te]), set(eps[n_te:n_te+n_va]), set(eps[n_te+n_va:])
    m_tr = np.isin(EPk, list(tr)); m_va = np.isin(EPk, list(va)); m_te = np.isin(EPk, list(te))
    print(f"[markov] episodes train/val/test = {len(tr)}/{len(va)}/{len(te)}  "
          f"samples {m_tr.sum()}/{m_va.sum()}/{m_te.sum()}", flush=True)
    # A degenerate split silently produces NaN EVs that look like a null result.
    if len(tr) < 6 or len(va) < 2 or len(te) < 3 or m_tr.sum() < 200:
        raise SystemExit(f"FAILED: degenerate episode split "
                         f"(train/val/test = {len(tr)}/{len(va)}/{len(te)}, "
                         f"{m_tr.sum()} train samples). Raise --episodes; this "
                         f"needs >=12 to split three ways.")

    onehot = np.zeros((N, 2 * na), np.float32)
    onehot[np.arange(N), Ak[:, 0]] = 1.0
    onehot[np.arange(N), na + Ak[:, 1]] = 1.0

    # Only bytes that actually VARY on the training split are scorable; a
    # constant byte has zero variance and its "EV" is 0/0.
    thr = a.min_std / 255.0
    live = (Y[m_tr].std(axis=0) > thr) & (Y[m_te].std(axis=0) > thr)
    print(f"[markov] {int(live.sum())}/{nb} target bytes move >{a.min_std} RAM "
          f"unit on BOTH train and test", flush=True)
    if live.sum() < 10:
        raise SystemExit(f"FAILED: only {int(live.sum())} target bytes qualify at "
                         f"--min_std {a.min_std}; there is nothing to predict.")

    dev = th.device(a.device if th.cuda.is_available() else "cpu")
    Yt = th.as_tensor(Y[:, live], dtype=th.float64, device=dev)

    def features(k, shuffled=False):
        """k frames sampled at stride 1 back from the newest, plus the action."""
        idx = [WINDOW - 1 - i for i in range(k)]        # newest first
        parts = [Wk[:, idx[0], :]]
        for i in idx[1:]:
            f = Wk[:, i, :]
            if shuffled:
                # history from a random OTHER timestep: same marginal, no link
                f = f[rng.permutation(N)]
            parts.append(f)
        return np.concatenate(parts + [onehot], axis=1)

    def ridge_ev(X, name):
        """Dual ridge -- d is up to 25k, n is a few thousand, so solve in sample
        space.

        TWO THINGS THAT BROKE THE FIRST VERSION OF THIS, both of which made
        MORE HISTORY LOOK CATASTROPHICALLY WORSE and would have been reported as
        "history does not help":

        1. Standardising with sd.clamp_min(1e-8) does not tame a column that is
           constant on train and moves on test -- it divides by 1e-8 and turns it
           into an enormous feature. Dead columns are DROPPED, not clamped.
        2. Frames one emulator tick apart are ~identical, so k=12 supplies 12
           near-duplicate copies of the same 2105 columns. The gram is nearly
           singular and the ridge path needs to reach far higher lambda than a
           grid topping out at 10. The old grid selected its own MAXIMUM for
           every stacked arm -- the tell that the path was truncated. The
           selected lambda is now checked against the grid boundary and the run
           FAILS if it lands there.
        """
        Xt = th.as_tensor(X, dtype=th.float32, device=dev)
        sd = Xt[m_tr].std(0)
        alive = sd > 1e-6
        if int(alive.sum()) == 0:
            raise SystemExit(f"FAILED: every feature is constant on train for {name}")
        Xt = (Xt[:, alive] - Xt[m_tr][:, alive].mean(0)) / sd[alive]
        Xtr, Xva, Xte = Xt[m_tr], Xt[m_va], Xt[m_te]
        Ytr = Yt[m_tr]; ymu = Ytr.mean(0)
        # gram in fp32 (25k-dim features), eigen-solve in fp64 (conditioning).
        # One eigendecomposition serves the whole lambda path.
        Ktr = (Xtr @ Xtr.T).double()
        Kva = (Xva @ Xtr.T).double()
        Kte = (Xte @ Xtr.T).double()
        w, V = th.linalg.eigh(Ktr)
        w = w.clamp_min(0)
        Yc = Ytr - ymu
        VtY = V.T @ Yc
        KvaV, KteV = Kva @ V, Kte @ V
        n_tr = len(Ktr)
        best = (-1e9, None, None)
        for lam in lams:
            d = 1.0 / (w + lam * n_tr)
            pv = KvaV @ (d[:, None] * VtY) + ymu
            ev_va = float(1 - ((Yt[m_va] - pv) ** 2).mean() /
                          ((Yt[m_va] - ymu) ** 2).mean())
            if np.isfinite(ev_va) and ev_va > best[0]:
                pt = KteV @ (d[:, None] * VtY) + ymu
                best = (ev_va, lam, pt)
        if best[2] is None:
            raise SystemExit(f"FAILED: no lambda produced a finite "
                             f"validation EV for {name}.")
        _, lam, pt = best
        if lam in (min(lams), max(lams)):
            raise SystemExit(
                f"FAILED: {name} selected lambda={lam:g}, which is the "
                f"{'MIN' if lam == min(lams) else 'MAX'} of the grid "
                f"[{min(lams):g}, {max(lams):g}]. The regularisation path is "
                f"truncated and the held-out EV is not the ridge's best -- "
                f"widen --lams. This exact failure made k=12 read EV -7487.")
        n_drop = int((~alive).sum())
        Yte = Yt[m_te]
        num = ((Yte - pt) ** 2).mean(0)
        den = ((Yte - ymu) ** 2).mean(0).clamp_min(1e-30)
        per_byte = (1 - num / den).cpu().numpy()
        pooled = float(1 - num.mean() / den.mean())
        # Pooled and MEDIAN only. The MEAN over bytes is dominated by bytes with
        # near-zero variance, whose EV denominator is ~0 -- that is what produced
        # "-339" for the arm that is actually the best one here.
        print(f"  {name:>16}  lam={lam:<8g} feat={int(alive.sum())}"
              f"(-{n_drop} dead)  pooled_EV={pooled:.4f}  "
              f"median_byte={float(np.median(per_byte)):.4f}  "
              f"p25={float(np.percentile(per_byte,25)):.4f}", flush=True)
        return dict(pooled=pooled, mean_byte=float(np.median(per_byte)),
                    median_byte=float(np.median(per_byte)), lam=lam,
                    per_byte=per_byte)

    print(f"\n  held-out EV predicting the NEXT masked RAM frame\n")
    res = {}
    for k in ks:
        res[f"k{k}"] = ridge_ev(features(k), f"k={k}")
    res["k12_shuffled"] = ridge_ev(features(max(ks), shuffled=True),
                                   f"k={max(ks)} SHUFFLED")

    kmax = f"k{max(ks)}"
    d_hist = res[kmax]["mean_byte"] - res["k1"]["mean_byte"]
    d_ctrl = res[kmax]["mean_byte"] - res["k12_shuffled"]["mean_byte"]
    pb1, pbk = res["k1"]["per_byte"], res[kmax]["per_byte"]
    gain = pbk - pb1
    big = np.where(gain > 0.05)[0]
    addr = np.asarray(mask)[live][big]
    order = np.argsort(-gain[big])

    print(f"\n  (all EVs are MEDIAN-over-bytes on held-out episodes)")
    print(f"  HISTORY GAIN   k={max(ks)} - k=1          : {d_hist:+.4f}")
    print(f"  CONTROL        k={max(ks)} - k=12 shuffled : {d_ctrl:+.4f}")
    print(f"  bytes improved by >0.05 EV        : {len(big)} / {int(live.sum())}")
    if len(big):
        print(f"  top addresses (RAM index, k=1 EV -> k={max(ks)} EV):")
        for i in order[:12]:
            print(f"      0x{addr[i]:04X}   {pb1[big[i]]:.3f} -> {pbk[big[i]]:.3f}")

    out = {k: {kk: vv for kk, vv in v.items() if kk != "per_byte"}
           for k, v in res.items()}
    out["history_gain"] = d_hist
    out["control_gain"] = d_ctrl
    out["n_bytes_improved"] = int(len(big))
    out["improved_addresses"] = [int(x) for x in addr[order[:64]]]
    out["config"] = vars(a)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
