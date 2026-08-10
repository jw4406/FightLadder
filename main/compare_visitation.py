"""Does V-trace change WHICH STATES self-play visits?

READS the .npz files written by state_visitation.py, named

    vis_<arm>_<ckpt>_s<seed>.npz

THE ONLY THING THAT MAKES A DISTANCE MEAN ANYTHING IS THE NULL. Two rollouts of
the SAME checkpoint with different seeds do not produce the same empirical
distribution, so a cross-arm distance is uninterpretable on its own -- it has to
be read against how far apart two seeds of one arm land. Every row below reports

    cross  = mean distance between the two ARMS   (all seed pairs)
    null   = distance between the two SEEDS of one arm (worst arm)
    ratio  = cross / null

and a feature only counts as different when ratio >= RATIO_THRESH. This project
has produced three separate false conclusions from statistics with no null or a
fail-open comparison (a nan fall-through that printed "thin cells are NOT the
explanation" on zero samples; a gate script that printed a verdict on two probes
that had crashed). Requiring >=2 seeds is how that class of error is prevented
here, so a missing seed is a hard error rather than a skipped null.

Continuous features use Wasserstein-1 normalized by the POOLED std at that
checkpoint, so the number reads in standard-deviation units and is comparable
across features. Categorical features (the status codes) and the action
histograms use total variation, which is already unit-free.

CONFOUND REPORTED, NOT HIDDEN: ep_len is printed for every arm. Arms with
different episode lengths visit a different mix of early/late states, which
shifts every marginal on its own.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FNAME_RE = re.compile(r"vis_(?P<arm>.+?)_(?P<ckpt>\d+)_s(?P<seed>\d+)\.npz$")
RATIO_THRESH = 2.0
CATEGORICAL = ("agent_status", "enemy_status")


def _tv(x, y, n_bins):
    import numpy as np
    p = np.bincount(x.astype(np.int64), minlength=n_bins).astype(np.float64)
    q = np.bincount(y.astype(np.int64), minlength=n_bins).astype(np.float64)
    p /= max(p.sum(), 1); q /= max(q.sum(), 1)
    return 0.5 * float(np.abs(p - q).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of vis_*.npz")
    ap.add_argument("--arms", nargs=2, required=True,
                    help="the two arms to contrast, e.g. vton vtoff")
    ap.add_argument("--plot", type=str, default=None, help="optional PNG path")
    a = ap.parse_args(argv)

    import numpy as np
    from scipy.stats import wasserstein_distance

    runs = defaultdict(dict)          # (arm, ckpt) -> {seed: npz}
    for p in sorted(glob.glob(os.path.join(a.dir, "vis_*.npz"))):
        m = FNAME_RE.search(os.path.basename(p))
        if not m:
            continue
        runs[(m["arm"], int(m["ckpt"]))][int(m["seed"])] = np.load(p, allow_pickle=True)
    if not runs:
        raise SystemExit(f"no vis_*.npz in {a.dir}")

    A, B = a.arms
    ckpts = sorted({c for (arm, c) in runs if arm in (A, B)})
    shared = [c for c in ckpts if (A, c) in runs and (B, c) in runs]
    if not shared:
        raise SystemExit(f"no checkpoint has BOTH {A} and {B}; matched steps are "
                         f"the only place the arms can be compared")

    def feats(z):
        """Named 1-D arrays: raw emulator vars plus the derived ones."""
        keys = [str(k) for k in z["keys"]]
        X = z["X"]
        d = {k: X[:, i] for i, k in enumerate(keys)}
        if "agent_x" in d and "enemy_x" in d:
            d["separation"] = np.abs(d["agent_x"] - d["enemy_x"])
        if "agent_hp" in d and "enemy_hp" in d:
            d["hp_diff"] = d["agent_hp"] - d["enemy_hp"]
        return d

    print("=" * 78)
    print(f"STATE VISITATION   {A}  vs  {B}")
    print("=" * 78)

    any_diff = False
    for c in shared:
        ra, rb = runs[(A, c)], runs[(B, c)]
        for nm, r in ((A, ra), (B, rb)):
            if len(r) < 2:
                raise SystemExit(
                    f"arm {nm} at {c} has {len(r)} seed(s); >=2 are REQUIRED -- "
                    f"without a null the cross-arm distance cannot be read")
        fa = {s: feats(z) for s, z in ra.items()}
        fb = {s: feats(z) for s, z in rb.items()}
        sa, sb = sorted(fa), sorted(fb)
        names = [k for k in fa[sa[0]] if all(k in f for f in
                 list(fa.values()) + list(fb.values()))]

        la = float(np.mean([ra[s]["ep_len"] for s in sa]))
        lb = float(np.mean([rb[s]["ep_len"] for s in sb]))
        print(f"\n  ckpt {c:,}    ep_len  {A} {la:.0f}   {B} {lb:.0f}"
              f"    ({len(sa)}+{len(sb)} seeds)")
        print(f"    {'feature':<16} {'cross':>9} {'null':>9} {'ratio':>7}")

        for k in names:
            cat = k in CATEGORICAL
            if cat:
                nb = int(max(max(f[k].max() for f in fa.values()),
                             max(f[k].max() for f in fb.values()))) + 1
                dist = lambda x, y: _tv(x, y, nb)
            else:
                pool = np.concatenate([f[k] for f in
                                       list(fa.values()) + list(fb.values())])
                sd = float(pool.std()) or 1.0
                dist = lambda x, y, sd=sd: wasserstein_distance(x, y) / sd
            null = max(dist(fa[sa[0]][k], fa[sa[1]][k]),
                       dist(fb[sb[0]][k], fb[sb[1]][k]))
            cross = float(np.mean([dist(fa[i][k], fb[j][k]) for i in sa for j in sb]))
            ratio = cross / null if null > 0 else float("inf")
            flag = "  <== DIFFERS" if ratio >= RATIO_THRESH else ""
            any_diff |= ratio >= RATIO_THRESH
            print(f"    {k:<16} {cross:>9.4f} {null:>9.4f} {ratio:>7.2f}{flag}")

        # Action histograms: not a state, but the mechanism by which visitation
        # would change at all.
        n_act = int(max(max(z["a_ego"].max() for z in list(ra.values()) + list(rb.values())),
                        max(z["a_adv"].max() for z in list(ra.values()) + list(rb.values())))) + 1
        for side in ("a_ego", "a_adv"):
            null = max(_tv(ra[sa[0]][side], ra[sa[1]][side], n_act),
                       _tv(rb[sb[0]][side], rb[sb[1]][side], n_act))
            cross = float(np.mean([_tv(ra[i][side], rb[j][side], n_act)
                                   for i in sa for j in sb]))
            ratio = cross / null if null > 0 else float("inf")
            flag = "  <== DIFFERS" if ratio >= RATIO_THRESH else ""
            any_diff |= ratio >= RATIO_THRESH
            print(f"    {side+' hist':<16} {cross:>9.4f} {null:>9.4f} {ratio:>7.2f}{flag}")

    print("\n" + "=" * 78)
    if any_diff:
        print(f"  VERDICT: at least one feature separates the arms by "
              f">={RATIO_THRESH}x the seed null.")
        print(f"  V-trace changes the visited state distribution, not only the")
        print(f"  value estimate. Read the flagged rows above for WHICH states.")
    else:
        print(f"  VERDICT: every feature is within {RATIO_THRESH}x of the seed null.")
        print(f"  On this evidence the two arms visit the SAME states; V-trace is")
        print(f"  changing the value estimate without moving the occupancy.")
    print("  (matched steps only -- these are the checkpoints both arms reached)")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = [k for k in ("separation", "agent_y", "hp_diff", "agent_hp") ]
        fig, axes = plt.subplots(len(shared), len(show),
                                 figsize=(4 * len(show), 3 * len(shared)),
                                 squeeze=False)
        for i, c in enumerate(shared):
            for j, k in enumerate(show):
                ax = axes[i][j]
                for nm, r in ((A, runs[(A, c)]), (B, runs[(B, c)])):
                    v = np.concatenate([feats(z)[k] for z in r.values()
                                        if k in feats(z)])
                    ax.hist(v, bins=60, density=True, histtype="step", label=nm)
                ax.set_title(f"{k} @ {c/1e6:.2f}M", fontsize=9)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)
        print(f"\n  wrote {a.plot}")


if __name__ == "__main__":
    main()
