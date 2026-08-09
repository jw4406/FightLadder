"""Plot exploitability (eps) directly, per seat and per LBR mode.

WHY THIS EXISTS: aggregate_local_eval_data.py plots RETURNS with a selfplay
overlay, so eps has to be read as a vertical gap between two moving lines -- and
the selfplay line moves a lot (it swung -0.257 to -0.069 on arm B). That makes
the sign of eps hard to see and the greedy-vs-lbr comparison nearly impossible.

Here eps = lbr_return - selfplay_return, both taken from the sidecar in the LBR
SEAT's own frame, so positive always means "the exploiter beat the incumbent" and
zero is a fixed, meaningful reference.

NEGATIVE eps IS NOT A SMALL BOUND -- it is a FAILED MEASUREMENT. The exploiter
lost to the incumbent, so the true eps is unknown-but->=0. Those points are drawn
hollow below a shaded floor and are EXCLUDED from the NashConv lower bound, which
sums max(eps, 0). Summing them raw makes a dominated run look like an
equilibrium: on the neu run that turned NashConv >= 0.437 into an apparent 0.011.

Only the highest-priority mode present owns the sidecar JSON (HEADLINE_PRIORITY
in local_best_response.py), but the sidecar records EVERY mode's return, so a
single JSON per (checkpoint, seat) yields all modes.

Usage:
    python plot_eps_curves.py --subdir spar_Ry_Sa_armB
"""
import argparse
import glob
import json
import os
import re
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODE_KEY = {"greedy": "greedy_return_mean",
            "lbr": "lbr_return_mean",
            "shuffle": "shuffle_return_mean"}
STYLE = {"greedy": ("tab:green", "o", "-"),
         "lbr": ("tab:blue", "s", "-"),
         "shuffle": ("tab:gray", "^", "--")}


def collect(subdir):
    """-> {(seat, mode): [(steps, eps), ...]} sorted by steps."""
    pat = os.path.join(HERE, "br_rewards", subdir, "spar_*_lbr*_br0_.json")
    files = glob.glob(pat)
    if not files:
        raise SystemExit(f"no sidecars under {pat}")
    out = collections.defaultdict(list)
    for f in files:
        d = json.load(open(f))
        st = int(re.search(r"spar_(\d+)_", os.path.basename(f)).group(1))
        sp = d.get("selfplay_return_mean")
        seat = d.get("lbr_seat")
        if sp is None or seat is None:
            continue
        for mode, key in MODE_KEY.items():
            v = d.get(key)
            if v is None:
                continue
            out[(seat, mode)].append((st, v - sp))
    for k in out:
        out[k].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", required=True, help="folder under main/br_rewards/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = collect(a.subdir)
    seats = [s for s in ("ego", "adv") if any(k[0] == s for k in data)]
    fig, axes = plt.subplots(1, len(seats), figsize=(7.0 * len(seats), 5.2),
                             squeeze=False)

    for ax, seat in zip(axes[0], seats):
        lo = 0.0
        for (s, mode), pts in sorted(data.items()):
            if s != seat:
                continue
            c, m, ls = STYLE.get(mode, ("k", "x", ":"))
            xs = [p[0] / 1e6 for p in pts]
            ys = [p[1] for p in pts]
            lo = min(lo, min(ys))
            ax.plot(xs, ys, color=c, linestyle=ls, label=mode, zorder=3)
            # solid = a real bound; hollow = the exploiter LOST, so eps is
            # unknown-but->=0 and the point must not be read as a small bound.
            for x, y in zip(xs, ys):
                ax.plot([x], [y], marker=m, color=c, zorder=4,
                        markerfacecolor=(c if y > 0 else "white"),
                        markeredgecolor=c)
        ax.axhline(0, color="k", lw=1.0, zorder=2)
        ax.axhspan(min(lo * 1.15, -0.02), 0, color="red", alpha=0.06, zorder=1)
        ax.text(0.015, 0.03, "VACUOUS: exploiter lost to the incumbent;\n"
                             "true eps is unknown-but->=0",
                transform=ax.transAxes, fontsize=8, color="darkred", va="bottom")
        # eps_X = (exploiter sitting in seat X) - (incumbent in seat X), both
        # against the SAME opponent. So it measures how far seat X's OWN policy
        # is from a best response -- NOT how exploitable the other seat is.
        other = "adversary" if seat == "ego" else "ego"
        ax.set_title(f"eps_{seat}   (how far the {seat} is from BR vs the incumbent {other})")
        ax.set_xlabel("Timestep (millions)")
        ax.set_ylabel("eps  =  LBR return  -  selfplay return")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"Exploitability lower bound: {a.subdir}", fontsize=13)
    fig.tight_layout()
    out = a.out or os.path.join(HERE, "local_eval_plots", a.subdir, "eps_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print(f"\n{'steps':>10} {'seat':>5} {'mode':>8} {'eps':>8}")
    for (seat, mode), pts in sorted(data.items()):
        for st, e in pts:
            print(f"{st:>10} {seat:>5} {mode:>8} {e:>8.3f}"
                  + ("   <- VACUOUS" if e <= 0 else ""))


if __name__ == "__main__":
    main()
