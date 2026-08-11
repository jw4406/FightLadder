"""Exploitability GAP across checkpoints: adversary-exploiter minus ego-exploiter.

    A = LBR return at the EGO seat   -> how exploitable the ADVERSARY is
    B = LBR return at the ADV seat   -> how exploitable the EGO is
    GAP = B - A

Deliberately NOT eps. eps subtracts the selfplay return, and that baseline is
untrustworthy on this arm: between 2.4M and 10.08M the entire rise in eps_ego
(0.232 -> 0.277) came from selfplay falling (-0.057 -> -0.110) while the
exploiter's absolute return was FLAT (+0.175 -> +0.167). Differencing two
exploiter returns measured the same way removes that baseline entirely.

THE HEADLINE IS A + B, THE DUALITY GAP:

    A + B = max_s v_ego(s, pi_adv) - min_t v_ego(pi_ego, t)

which is large whenever EITHER player is far from best-responding and reaches 0
only at equilibrium. Two equally-bad players show a LARGE A+B -- they are both
massively exploitable -- which is the whole point of tracking it.

B - A is only the ASYMMETRY (which side is exploited more) and is ~0 for two
equally-bad players, so it must not be read as progress. Printed as a secondary
column.

A+B ALSO REMOVES THE UNTRUSTED SELFPLAY BASELINE, by cancellation rather than by
omission: NashConv = (A - sp_ego) + (B - sp_adv), and sp_ego + sp_adv = 0 in a
zero-sum game -- measured here as -0.00095 + 0.00095 exactly. So A+B equals
NashConv without depending on selfplay at all (verified: 6.72M greedy gives
A+B = 0.16031 against the code's NashConv 0.16030).

Not exactly zero-sum: a draw pays BOTH players +1 (retro_wrappers.py:281), so
the cancellation carries a small draw-rate term. Magnitude ~0.002 against a
reward scale of ~0.176.

Scrapes the LBR logs rather than the .txt sidecars so it works on runs whose
output_subdir differs.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEAT_RE = re.compile(r"LBR plays the (EGO|ADV) seat")
MODE_RE = re.compile(r"^\[LBR\]\s+(greedy|lbr|shuffle|minimax|minimaxshuffle)\s+"
                     r"return=([-+0-9.]+)")
CKPT_RE = re.compile(r"checkpoint spar_\w*?_(\d+)_steps")


def scrape(paths):
    """{step: {seat: {mode: return}}} from LBR stdout logs."""
    out = {}
    for p in paths:
        step, seat = None, None
        for line in open(p, errors="ignore"):
            m = CKPT_RE.search(line)
            if m:
                step, seat = int(m.group(1)), None
                continue
            m = SEAT_RE.search(line)
            if m:
                seat = m.group(1)
                continue
            m = MODE_RE.match(line.strip())
            if m and step is not None and seat is not None:
                out.setdefault(step, {}).setdefault(seat, {})[m.group(1)] = float(m.group(2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+",
                    default=["logs/lbr_masked_ego.log", "logs/eps_series.log",
                             "logs/gap_series.log"],
                    help="LBR stdout logs to scrape")
    ap.add_argument("--mode", default="greedy",
                    help="which exploiter defines the gap. greedy is the tightest "
                         "measured so far and never consults the critic, so it is "
                         "the most stable yardstick across checkpoints.")
    a = ap.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = []
    for pat in a.logs:
        paths += glob.glob(pat if os.path.isabs(pat) else os.path.join(repo, pat))
    data = scrape(paths)
    if not data:
        raise SystemExit(f"no LBR results found in {paths}")

    print(f"  exploiter = {a.mode}   (A = ego seat, B = adv seat)")
    print(f"  {'step':>11} {'A ego-seat':>11} {'B adv-seat':>11} "
          f"{'A+B DUALITY':>12} {'B-A asym':>10}")
    for s in sorted(data):
        A = data[s].get("EGO", {}).get(a.mode)
        B = data[s].get("ADV", {}).get(a.mode)
        fa = f"{A:>+11.5f}" if A is not None else f"{'--':>11}"
        fb = f"{B:>+11.5f}" if B is not None else f"{'--':>11}"
        if A is not None and B is not None:
            print(f"  {s:>11,} {fa} {fb} {A+B:>+12.5f} {B-A:>+10.5f}")
        else:
            miss = "adv seat" if B is None else "ego seat"
            print(f"  {s:>11,} {fa} {fb} {'--':>12} {'--':>10}   ({miss} not run)")

    done = [s for s in sorted(data)
            if data[s].get("EGO", {}).get(a.mode) is not None
            and data[s].get("ADV", {}).get(a.mode) is not None]
    if len(done) >= 2:
        d0 = data[done[0]]["ADV"][a.mode] + data[done[0]]["EGO"][a.mode]
        d1 = data[done[-1]]["ADV"][a.mode] + data[done[-1]]["EGO"][a.mode]
        print(f"\n  DUALITY GAP (A+B) {done[0]:,} -> {done[-1]:,}: "
              f"{d0:+.5f} -> {d1:+.5f}"
              f"   ({'SHRINKING -- converging' if d1 < d0 else 'WIDENING -- diverging'})")
    else:
        print(f"\n  need BOTH seats at >=2 checkpoints for a trend; have {len(done)}")


if __name__ == "__main__":
    main()
