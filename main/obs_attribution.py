"""WHICH downsampling destroys the branch information?

RAM distinguishes 441 = 21^2 successors at a median decision point -- every
genuinely distinct joint action (0 and 9 are byte-identical no-ops, so 21 real
actions per side). The agent's observation distinguishes ONE. `_get_obs` applies
three independent reductions, and this attributes the collapse among them:

    current       frames[::4] (indices 0,4,8 of 12), o[::2,::2], channel i%3
    recent        frames[3::4] (indices 3,7,11) -- SAME SHAPE, SAME COST, only
                  sampled to include the NEWEST frame. An action's effect shows
                  up at the END of the 8-frame window, so if the newest frames
                  are where the information is, this is a free fix.
    all_frames    all 12 frames        (temporal reduction removed)
    full_spatial  200x256              (spatial reduction removed)
    all_channels  all 3 RGB per frame  (channel reduction removed)
    everything    all three removed    (upper bound for a pixel observation)

SCOPE: ego actions only, with the adversary held at action 0. That is a 22-way
test, not 484-way -- 22x cheaper and sufficient, since the question is whether
the observation resolves ACTIONS at all. The ceiling is 21 distinct (0 and 9
coincide), and RAM is measured alongside as ground truth.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPAND_ROOT = "attrib_root"
VARIANTS = ("current", "recent", "all_frames", "full_spatial",
            "all_channels", "everything")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_states", type=int, default=120)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="obs_attribution.json")
    a = ap.parse_args(argv)

    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, REPO_ROOT)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs)
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        n = venv.num_envs
        na = ops.n_actions
        rng = np.random.RandomState(0)

        per_variant = defaultdict(list)          # variant -> [distinct per state]
        ram_counts = []
        obs = venv.reset()
        n_exp = int(np.ceil(a.n_states / n))
        for e in range(n_exp):
            for _ in range(a.stride if e else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]

            venv.env_method("lbr_snapshot", EXPAND_ROOT)
            digs = {v: [[] for _ in range(n)] for v in VARIANTS}
            rams = [[] for _ in range(n)]
            for i in range(na):
                venv.env_method("lbr_restore", EXPAND_ROOT)
                venv.step(ops.joint(np.full(n, i), np.zeros(n, dtype=int)))
                vres = venv.env_method("lbr_obs_variants")
                rres = venv.env_method("lbr_fingerprint")
                for k in range(n):
                    for v in VARIANTS:
                        digs[v][k].append(vres[k][v])
                    rams[k].append(rres[k])
            venv.env_method("lbr_restore", EXPAND_ROOT)
            venv.env_method("lbr_drop", EXPAND_ROOT)

            for k in range(n):
                for v in VARIANTS:
                    per_variant[v].append(len(set(digs[v][k])))
                ram_counts.append(len(set(rams[k])))
            print(f"   expansion {e+1}/{n_exp}", flush=True)
    finally:
        venv.close()

    ram = np.array(ram_counts)
    S = ram.size
    print("\n" + "=" * 74)
    print(f"OBSERVATION ATTRIBUTION  {os.path.basename(a.ckpt)}   {S} states")
    print(f"  distinct successors over {na} EGO actions (adv fixed); "
          f"ceiling is {na-1} (0 and 9 coincide)")
    print("=" * 74)
    print(f"  {'variant':<14} {'median':>7} {'p25':>6} {'p75':>6} {'max':>6} "
          f"{'% of RAM':>9}")
    res = {"checkpoint": os.path.basename(a.ckpt), "n_states": S,
           "ram_median": float(np.median(ram))}
    print(f"  {'RAM (truth)':<14} {np.median(ram):>7.0f} "
          f"{np.percentile(ram,25):>6.0f} {np.percentile(ram,75):>6.0f} "
          f"{ram.max():>6.0f} {100.0:>8.1f}%")
    for v in VARIANTS:
        d = np.array(per_variant[v])
        pct = 100.0 * np.median(d) / max(np.median(ram), 1)
        res[v] = {"median": float(np.median(d)), "max": int(d.max()),
                  "pct_of_ram": float(pct)}
        print(f"  {v:<14} {np.median(d):>7.0f} {np.percentile(d,25):>6.0f} "
              f"{np.percentile(d,75):>6.0f} {d.max():>6.0f} {pct:>8.1f}%")

    cur = res["current"]["median"]
    best_single = max(("recent", "all_frames", "full_spatial", "all_channels"),
                      key=lambda v: res[v]["median"])
    print("\n" + "=" * 74)
    print(f"  current resolves {cur:.0f} of {np.median(ram):.0f} game-distinct "
          f"successors ({res['current']['pct_of_ram']:.1f}%)")
    print(f"  best SINGLE fix: `{best_single}` -> {res[best_single]['median']:.0f} "
          f"({res[best_single]['pct_of_ram']:.1f}%)")
    print(f"  all three removed: {res['everything']['median']:.0f} "
          f"({res['everything']['pct_of_ram']:.1f}%)")
    if res["recent"]["median"] >= 0.8 * res["everything"]["median"] and \
       res["recent"]["median"] > cur:
        res["verdict"] = "TEMPORAL PHASE -- free fix available"
        print("\n  => `recent` recovers most of it at IDENTICAL tensor shape and")
        print("     compute. The loss is WHICH frames are sampled, not how many")
        print("     or at what resolution. Changing frames[::4] to frames[3::4]")
        print("     is a one-line change that does not touch the CNN.")
    elif res["everything"]["median"] <= 2 * cur:
        res["verdict"] = "NOT THE OBSERVATION"
        print("\n  => even with every reduction removed the observation barely")
        print("     resolves more than today. A pixel observation at this frame")
        print("     rate cannot express the distinction; the input is not fixable")
        print("     by resampling alone.")
    else:
        res["verdict"] = f"DOMINATED BY {best_single}"
        print(f"\n  => `{best_single}` is the dominant loss. Removing it costs")
        print(f"     real compute (larger CNN input), unlike `recent`.")
    with open(os.path.join(REPO_ROOT, a.out), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {os.path.join(REPO_ROOT, a.out)}")
    return res


if __name__ == "__main__":
    main()
