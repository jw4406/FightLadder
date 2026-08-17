"""THE CEILING. Is the interaction term gamma LEARNABLE from the observation at all?

This is the gate every other critic idea is capped by, and the one this project
has never run. Everything measured so far asks how much gamma a particular head
CAPTURED. This asks how much is capturable AT ALL: regress the true enumerated
gamma_ij(s) on the observation the network actually sees, and report held-out
R^2. No head can beat that number.

WHY IT REPLACES THE METRICS WE HAVE BEEN USING. Every previous gate was
inflatable without skill:
  corrW(R)      a CONSTANT matrix scores 0.412 of the 0.605 a head scored.
  gamma_share   a RANDOM double-centred W raises it 10.98% -> 35.13% with zero
                information -- it rewards interaction-shaped NOISE.
  in-batch EV   read 0.23->0.66 on a head whose held-out corrW was -0.012.
Held-out R^2 against a shuffled control cannot be faked by any of those.

THE DECOMPOSITION IS THE POINT. mu, alpha, beta and gamma are scored SEPARATELY:

    M(s)_ij = mu(s) + alpha_i(s) + beta_j(s) + gamma_ij(s)

If alpha and beta are learnable and gamma is not, that single fact explains
every negative result in this program -- the separable part of the payoff is
predictable from the state and the joint part is not, so a joint-action critic
has nothing to learn that two per-player heads would not already have. If NOTHING
is learnable, the problem is the observation or the trunk, not the head. If gamma
IS learnable at, say, R^2 0.3, then the measured ~12% capture is leaving a lot on
the table and architecture work is justified.

CONTROLS:
  const     predict the TRAIN-mean matrix. This is the CONST baseline that
            retracted the corrW result; held-out it should sit near 0.
  shuffled  observations permuted across states, destroying the state-target
            pairing while preserving every marginal. Any R^2 above this is the
            only part that is real.

SPLITS ARE CONTIGUOUS BLOCKS, not random states. Roots are collected along a
rollout, so neighbouring states are near-duplicates; a random split leaks and
inflates R^2 -- the same mechanism that inflated probe scores ~5x in this
codebase when timestep splits were used instead of episode splits.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", required=True,
                    help="npz with OBS and R (contact_density --mode collect, or "
                         "bootstrap_delta --save_obs)")
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1,1,3,10,30,100,300,1e3,3e3,1e4,1e5,1e6")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--mlp_hidden", type=int, default=512)
    ap.add_argument("--mlp_epochs", type=int, default=300)
    ap.add_argument("--mlp_seeds", type=int, default=3)
    ap.add_argument("--mlp_wds", default="1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--mlp_lrs", default="3e-4,1e-3")
    ap.add_argument("--contact_only", action="store_true",
                    help="score only states where gamma != 0. Pooling in the 91% "
                         "of states whose gamma is identically zero makes the "
                         "target trivially predictable (predict zero) and would "
                         "report a high R^2 that means nothing.")
    ap.add_argument("--blocks", type=int, default=20,
                    help="contiguous blocks per file, dealt round-robin "
                         "to test/val/train/train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="gamma_learnability.json")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th

    OBS, R = [], []
    for f in a.npz:
        d = np.load(f, allow_pickle=True)
        if "OBS" not in d.files:
            raise SystemExit(f"FAILED: {f} has no OBS; re-collect with obs saving")
        # Blocks stay per-file so a contiguous split never straddles two rollouts.
        OBS.append(d["OBS"].astype(np.float64))
        R.append((d["R"] if "R" in d.files else d["RL"]).astype(np.float64))
    n_per = [len(x) for x in OBS]
    OBS = np.concatenate(OBS); R = np.concatenate(R)
    ns, na = R.shape[0], R.shape[1]
    print(f"[ceil] {ns} states from {len(a.npz)} files {n_per}, "
          f"obs {OBS.shape[1]}, {na}x{na} payoff\n")

    mu = R.mean(axis=(1, 2))
    al = R.mean(axis=2) - mu[:, None]
    be = R.mean(axis=1) - mu[:, None]
    G = R - mu[:, None, None] - al[:, :, None] - be[:, None, :]

    keep = np.ones(ns, bool)
    if a.contact_only:
        keep = (G ** 2).sum(axis=(1, 2)) > 1e-18
        print(f"[ceil] --contact_only: {keep.sum()}/{ns} states have gamma != 0")
        if keep.sum() < 60:
            raise SystemExit(f"FAILED: only {keep.sum()} contact states; too few "
                             f"to split three ways.")

    # INTERLEAVED CONTIGUOUS BLOCKS, not one chunk each. A single contiguous
    # 60/15/25 split puts train early in the rollout and test late, and the
    # rollout drifts (hp falls, the round advances), so val and test end up in
    # different regimes: the first version of this selected lambda=10 on val and
    # scored R^2 = -0.897 on test. Chopping each file into many blocks and
    # dealing them round-robin keeps neighbouring near-duplicate states out of
    # different splits while making the three splits distributionally matched.
    nb = max(8, a.blocks)
    tr = np.zeros(ns, bool); va = np.zeros(ns, bool); te = np.zeros(ns, bool)
    o = 0
    for n in n_per:
        edges = np.linspace(0, n, nb + 1).astype(int)
        for b in range(nb):
            sl = slice(o + edges[b], o + edges[b + 1])
            (te if b % 4 == 0 else va if b % 4 == 1 else tr)[sl] = True
        o += n
    tr &= keep; va &= keep; te &= keep
    print(f"[ceil] contiguous blocks: train {tr.sum()} / val {va.sum()} / test {te.sum()}")
    if min(tr.sum(), va.sum(), te.sum()) < 20:
        raise SystemExit("FAILED: a split has <20 states; collect more roots.")

    dev = th.device(a.device if th.cuda.is_available() else "cpu")
    lams = [float(x) for x in a.lams.split(",") if x.strip()]

    X = th.as_tensor(OBS, dtype=th.float64, device=dev)
    sd = X[tr].std(0)
    alive = sd > 1e-9                      # DROP dead columns, never clamp: a
    X = (X[:, alive] - X[tr][:, alive].mean(0)) / sd[alive]   # column constant on
    print(f"[ceil] {int(alive.sum())}/{OBS.shape[1]} obs bytes vary on train\n")

    def ridge(Y, name, Xm):
        Yt = th.as_tensor(Y.reshape(ns, -1), dtype=th.float64, device=dev)
        Xtr, Ytr = Xm[tr], Yt[tr]
        ymu = Ytr.mean(0)
        K = (Xtr @ Xtr.T)
        w, V = th.linalg.eigh(K); w = w.clamp_min(0)
        VtY = V.T @ (Ytr - ymu)
        KvV, KtV = (Xm[va] @ Xtr.T) @ V, (Xm[te] @ Xtr.T) @ V
        best = (-1e18, None, None)
        for lam in lams:
            dg = 1.0 / (w + lam * len(K))
            pv = KvV @ (dg[:, None] * VtY) + ymu
            r2 = float(1 - ((Yt[va] - pv) ** 2).sum() / ((Yt[va] - ymu) ** 2).sum())
            if np.isfinite(r2) and r2 > best[0]:
                best = (r2, lam, KtV @ (dg[:, None] * VtY) + ymu)
        r2_val, lam, pt = best
        # Boundary handling has to distinguish two very different cases.
        # MIN boundary: under-regularised, the path is genuinely truncated.
        # MAX boundary: at huge lambda the solution CONVERGES to the constant
        # predictor. If validation R^2 is ~0 there, "constant wins" is the
        # answer -- the target is not predictable -- not a truncated sweep.
        if lam == min(lams):
            raise SystemExit(f"FAILED: {name} selected the MINIMUM lambda "
                             f"({lam:g}); the path is truncated. Widen --lams.")
        if lam == max(lams) and r2_val > 0.01:
            raise SystemExit(f"FAILED: {name} selected the MAXIMUM lambda "
                             f"({lam:g}) while still scoring val R^2 {r2_val:.3f} "
                             f"> 0; the path is truncated. Widen --lams.")
        Yte = Yt[te]
        ss_res = float(((Yte - pt) ** 2).sum())
        ss_tot = float(((Yte - ymu) ** 2).sum())          # CONST baseline
        r2 = 1 - ss_res / max(ss_tot, 1e-30)
        return r2, lam, (lam == max(lams)), r2_val

    rng = np.random.RandomState(0)
    Xshuf = X[th.as_tensor(rng.permutation(ns), device=dev)]

    def run(Y, name):
        r2, lam, atmax, r2v = ridge(Y, name, X)
        # SHUFFLED control: same features, same target, pairing destroyed. The
        # permutation is fixed once so every term is scored against the same
        # noise floor.
        r2s, _, _, _ = ridge(Y, name + " shuffled", Xshuf)
        tag = "  [CONSTANT wins -- not predictable]" if atmax else ""
        # A large val/test gap means the splits are not the same distribution
        # and the number is not interpretable, whatever it says.
        warn = "  <-- VAL/TEST MISMATCH" if abs(r2v - r2) > 0.15 else ""
        print(f"  {name:>26}  R2 = {r2:+.4f} (val {r2v:+.4f})  shuffled = {r2s:+.4f}   "
              f"REAL = {r2 - r2s:+.4f}   (lam {lam:g}){tag}{warn}")
        return dict(r2=r2, r2_val=r2v, r2_shuffled=r2s, real=r2 - r2s,
                    lam=lam, constant_wins=bool(atmax))

    print("  held-out R^2 predicting each ANOVA term from the observation")
    print("  (CONST baseline is R^2 = 0 by construction; shuffled is the noise floor)\n")
    out = {"n_states": int(ns), "contact_only": bool(a.contact_only)}
    out["mu"] = run(mu[:, None], "mu  (state value)")
    out["alpha"] = run(al, "alpha (ego main effect)")
    out["beta"] = run(be, "beta  (adv main effect)")
    out["gamma"] = run(G, "GAMMA (interaction)")
    out["within"] = run(R - mu[:, None, None], "within-state (a+b+g)")

    g, aa, bb = out["gamma"]["real"], out["alpha"]["real"], out["beta"]["real"]
    print(f"\n  READING:")
    print(f"    alpha/beta learnable at {aa:+.3f}/{bb:+.3f}, gamma at {g:+.3f}")
    if g <= 0.02 and max(aa, bb) > 0.10:
        print(f"    => the SEPARABLE part is predictable and the JOINT part is NOT.")
        print(f"       A joint-action critic has nothing to learn here that two")
        print(f"       per-player heads would not already have. This caps every")
        print(f"       architecture idea on the list.")
    elif g <= 0.02:
        print(f"    => NOTHING is learnable. The bottleneck is the observation or")
        print(f"       the trunk, not the head.")
    else:
        print(f"    => gamma IS learnable at R^2 {g:.3f}. Measured head capture is")
        print(f"       ~12%, so architecture work has real headroom.")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
