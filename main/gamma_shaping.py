"""Can a STATUS-PAIR reward term raise gamma's SHARE? Offline, on data already collected.

WHY THIS SHAPE AND NOT THE PREVIOUS THREE. gamma's share of the action-dependent
energy is gamma / (alpha + beta + gamma). Counter-hit, pressure and trade
weighting all ADDED a term carrying main-effect content, so the denominator grew
faster than the numerator and all three DILUTED the share. For pressure that was
provable in advance: beta*(p_a - p_e) is a difference of PER-PLAYER indicators,
i.e. literally an alpha term plus a beta term.

There are only two ways to raise the share: add a term that is PURE interaction,
or remove main-effect content. This tests the first. A term built on the PAIR of
statuses, antisymmetrised and DOUBLE-CENTRED, has zero row and column means by
construction, so it lands in gamma rather than in alpha or beta.

TWO QUESTIONS, IN ORDER. The second is only worth asking if the first says yes.

  A. CEILING. Is the true interaction gamma_ij(s) a function of the status pair
     AT ALL? Fit f[s_a, s_e] by least squares across every state and branch and
     report held-out R^2. If status pairs cannot predict gamma, then NO reward
     term built from statuses can amplify it, and status-based shaping is closed
     regardless of how the term is constructed.

  B. THE VARIANT. With W = the fitted f, antisymmetrised and double-centred,
     sweep kappa and score gamma's share exactly as contact_density.py does.

CONTROLS, because "the number went up when I added a fitted term" is the oldest
artifact here and this fits W to the data:
  random    a random double-centred antisymmetric W of the same norm. Isolates
            "double-centred terms raise the share mechanically" from "THIS W
            carries real structure".
  heldout   f is fitted on half the states and scored on the other half. A W
            fitted and scored on the same states would report its own overfit.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIELDS = ("agent_hp", "enemy_hp", "agent_x", "enemy_x",
          "agent_status", "enemy_status", "round_countdown")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--kappas", default="0,0.25,0.5,1,2,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gamma_shaping.json")
    a = ap.parse_args(argv)

    import numpy as np
    rng = np.random.RandomState(a.seed)
    F = {f: i for i, f in enumerate(FIELDS)}

    ROOT, POST, RL, DONE = [], [], [], []
    for f in a.npz:
        d = np.load(f, allow_pickle=True)
        ROOT.append(d["ROOT"]); POST.append(d["POST"])
        RL.append(d["RL"]); DONE.append(d["DONE"])
    ROOT = np.concatenate(ROOT); POST = np.concatenate(POST)
    RL = np.concatenate(RL); DONE = np.concatenate(DONE)
    ns, na = RL.shape[0], RL.shape[1]

    ra = ROOT[:, None, None, F["agent_hp"]]; re = ROOT[:, None, None, F["enemy_hp"]]
    pa = POST[..., F["agent_hp"]]; pe = POST[..., F["enemy_hp"]]
    live = (~DONE) & (pa > 0) & (pe > 0) & (pa <= ra) & (pe <= re)
    base = np.where(live, 0.001 * ((re - pe) - (ra - pa)), RL)
    print(f"[shape] {ns} roots x {na*na} branches, {live.mean():.0%} in-fight\n")

    def anova(M):
        mu = M.mean(axis=(1, 2), keepdims=True)
        al = M.mean(axis=2, keepdims=True) - mu
        be = M.mean(axis=1, keepdims=True) - mu
        return mu, al, be, M - mu - al - be

    def score(M, label):
        mu, al, be, G = anova(M)
        wn = ((M - mu) ** 2).sum(axis=(1, 2))
        gn = (G ** 2).sum(axis=(1, 2))
        act = gn > 1e-18
        share = float(gn[act].sum() / max(wn[act].sum(), 1e-30))
        print(f"    {label:>26}  contact={float(act.mean()):6.1%}  "
              f"gamma_share={share:7.2%}  |gamma|={float(np.sqrt(gn.mean())):.5f}")
        return dict(contact=float(act.mean()), gamma_share=share,
                    gamma_mag=float(np.sqrt(gn.mean())))

    # ---- status indexing ---------------------------------------------------
    sa = POST[..., F["agent_status"]].astype(np.int64)
    se = POST[..., F["enemy_status"]].astype(np.int64)
    vals = np.unique(np.concatenate([sa.ravel(), se.ravel()]))
    idx = {v: k for k, v in enumerate(vals)}
    nS = len(vals)
    ia = np.vectorize(idx.get)(sa); ie = np.vectorize(idx.get)(se)
    print(f"  {nS} distinct statuses: {list(vals)}")

    _, _, _, G0 = anova(base)

    # ---- A. CEILING: can the status pair predict the true gamma at all? ----
    # Episode-analogue split: fit on half the STATES, score on the other half.
    perm = rng.permutation(ns)
    tr, te = perm[:ns // 2], perm[ns // 2:]
    def cellmean(states):
        num = np.zeros((nS, nS)); den = np.zeros((nS, nS))
        np.add.at(num, (ia[states].ravel(), ie[states].ravel()), G0[states].ravel())
        np.add.at(den, (ia[states].ravel(), ie[states].ravel()), 1.0)
        return num / np.maximum(den, 1.0), den
    f_tr, cnt = cellmean(tr)
    pred = f_tr[ia[te], ie[te]]
    ss_res = float(((G0[te] - pred) ** 2).sum())
    ss_tot = float((G0[te] ** 2).sum())          # gamma is already zero-mean
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    print(f"\n  A. CEILING -- held-out R^2 of gamma predicted from the status pair")
    print(f"     R^2 = {r2:.4f}   (fit on {len(tr)} states, scored on {len(te)})")
    print(f"     status-pair cells with >=100 samples: "
          f"{int((cnt >= 100).sum())}/{nS*nS}")
    if r2 <= 0.01:
        print(f"     => status pairs carry essentially NONE of the interaction.")
        print(f"        No reward term built from statuses can amplify gamma,")
        print(f"        however it is constructed. Status-based shaping is CLOSED.")

    # ---- B. the variant ----------------------------------------------------
    f_all, _ = cellmean(np.arange(ns))
    W = 0.5 * (f_all - f_all.T)                       # antisymmetric
    W = W - W.mean(axis=1, keepdims=True) - W.mean(axis=0, keepdims=True) + W.mean()
    W = 0.5 * (W - W.T)                               # re-antisymmetrise after centring
    print(f"\n     W double-centring residual: row {np.abs(W.mean(1)).max():.2e}, "
          f"col {np.abs(W.mean(0)).max():.2e}, antisym {np.abs(W + W.T).max():.2e}")

    Wr = rng.randn(nS, nS); Wr = 0.5 * (Wr - Wr.T)
    Wr = Wr - Wr.mean(axis=1, keepdims=True) - Wr.mean(axis=0, keepdims=True) + Wr.mean()
    Wr = 0.5 * (Wr - Wr.T)

    def tau(Wm):
        t = Wm[ia, ie]
        s = float(np.sqrt((t ** 2).mean()))
        return t / max(s, 1e-30)                      # unit RMS, so kappa is in
    t_fit, t_rnd = tau(W), tau(Wr)                    # units of the base RMS
    scale = float(np.sqrt((base ** 2).mean()))
    print(f"     base RMS = {scale:.5f}; kappa is in units of that\n")

    # How much of the ADDED term is itself interaction? If this is not ~100%,
    # double-centring over STATUS did not survive the mapping into ACTION space.
    for nm, t in (("fitted", t_fit), ("random", t_rnd)):
        mu, al, be, Gt = anova(t)
        wn = ((t - mu) ** 2).sum(); gn = (Gt ** 2).sum()
        print(f"     tau[{nm}] is {gn/max(wn,1e-30):.1%} interaction in the ACTION basis")

    print(f"\n  B. VARIANT -- gamma share vs kappa")
    out = {"r2_status_pair": r2, "n_status": nS, "arms": {}}
    for nm, t in (("fitted W", t_fit), ("random W (control)", t_rnd)):
        out["arms"][nm] = []
        for k in [float(x) for x in a.kappas.split(",")]:
            M = np.where(live, base + k * scale * t, RL)
            r = score(M, f"{nm} kappa={k:g}")
            r["kappa"] = k
            out["arms"][nm].append(r)
    out["baseline"] = score(base, "baseline")

    b = out["baseline"]["gamma_share"]
    best = max((r["gamma_share"], nm, r["kappa"])
               for nm, rs in out["arms"].items() for r in rs if r["kappa"] > 0)
    print(f"\n  baseline gamma_share = {b:.2%}")
    print(f"  best variant         = {best[1]} kappa={best[2]:g} -> {best[0]:.2%} "
          f"({'BEATS' if best[0] > b else 'LOSES TO'} baseline)")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
