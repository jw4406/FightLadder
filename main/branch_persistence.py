"""Do action-distinct branches STAY distinct, or reconverge?

RAM resolves 21/21 action-distinct successors one step after a decision; pixels
and all 14 curated info variables resolve 1. So a full-RAM observation would
hand the critic a distinction nothing else can see. Before paying for that --
it invalidates every existing checkpoint and every analysis tool -- one question
has to be answered:

    IS THE DISTINCTION PREDICTIVE, OR TRANSIENT?

Two possibilities with opposite consequences:

  RECONVERGE  the differing bytes are a consumed input buffer / a counter that
              resets during hitstun-blockstun lockout. The branches collapse
              back to one state, the future is identical, and there is NOTHING
              to predict. Worse, Q(s,a,o) already conditions on the action, so a
              byte that merely records "this button was pressed" tells the critic
              something it already knows. Full RAM would buy nothing.
  DIVERGE     the difference propagates into genuinely different futures --
              visible in pixels a few steps later. Then the information IS
              predictive, it is simply DELAYED beyond the one-step horizon the
              observation covers, and full RAM delivers it immediately.

METHOD. Branch on all 22 ego actions, then roll every branch forward with the
IDENTICAL continuation (both players no-op) and count distinct states at
increasing horizons. Any divergence that appears under a common continuation is
caused by the branch action alone.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPAND_ROOT = "persist_root"
HORIZONS = (1, 2, 4, 8, 16)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_states", type=int, default=60)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="branch_persistence.json")
    ap.add_argument("--ram_mask", type=str, default="",
                    help="RAM byte-index .npy, required when the checkpoint was "
                         "trained with a MASKED ram observation: the checkpoint "
                         "records the WIDTH, not which bytes.")
    a = ap.parse_args(argv)

    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, infer_obs_kwargs, REPO_ROOT)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs, **infer_obs_kwargs(data, (getattr(a, 'ram_mask', '') or None)))
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        n = venv.num_envs
        na = ops.n_actions
        maxh = max(HORIZONS)
        rng = np.random.RandomState(0)
        noop = np.zeros(n, dtype=int)

        ram_d = {h: [] for h in HORIZONS}
        obs_d = {h: [] for h in HORIZONS}
        obs = venv.reset()
        n_exp = int(np.ceil(a.n_states / n))
        for e in range(n_exp):
            for _ in range(a.stride if e else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]

            venv.env_method("lbr_snapshot", EXPAND_ROOT)
            # [horizon][env] -> list over the 22 branch actions
            rh = {h: [[] for _ in range(n)] for h in HORIZONS}
            oh = {h: [[] for _ in range(n)] for h in HORIZONS}
            for i in range(na):
                venv.env_method("lbr_restore", EXPAND_ROOT)
                o1 = venv.step(ops.joint(np.full(n, i), noop))[0]
                for t in range(1, maxh + 1):
                    if t > 1:
                        # IDENTICAL continuation for every branch, so any
                        # divergence is attributable to the branch action alone.
                        o1 = venv.step(ops.joint(noop, noop))[0]
                    if t in HORIZONS:
                        res = venv.env_method("lbr_state_variants")
                        for k in range(n):
                            rh[t][k].append(res[k]["ram"])
                            oh[t][k].append(hash(o1[k].tobytes()))
            venv.env_method("lbr_restore", EXPAND_ROOT)
            venv.env_method("lbr_drop", EXPAND_ROOT)
            for h in HORIZONS:
                for k in range(n):
                    ram_d[h].append(len(set(rh[h][k])))
                    obs_d[h].append(len(set(oh[h][k])))
            print(f"   expansion {e+1}/{n_exp}", flush=True)
    finally:
        venv.close()

    S = len(ram_d[HORIZONS[0]])
    ceiling = na - 1
    print("\n" + "=" * 74)
    print(f"BRANCH PERSISTENCE  {os.path.basename(a.ckpt)}   {S} states")
    print(f"  distinct states over {na} ego actions after h steps of an IDENTICAL")
    print(f"  no-op continuation; ceiling {ceiling}")
    print("=" * 74)
    print(f"  {'horizon':>8} {'RAM median':>12} {'PIXEL median':>14} {'PIXEL max':>11}")
    res = {"checkpoint": os.path.basename(a.ckpt), "n_states": S,
           "ceiling": ceiling, "horizons": {}}
    for h in HORIZONS:
        r = float(np.median(ram_d[h])); o = float(np.median(obs_d[h]))
        res["horizons"][str(h)] = {"ram_median": r, "obs_median": o,
                                   "obs_max": int(np.max(obs_d[h]))}
        print(f"  {h:>8} {r:>12.0f} {o:>14.0f} {np.max(obs_d[h]):>11.0f}")

    r1 = np.median(ram_d[1]); rmax = np.median(ram_d[max(HORIZONS)])
    omax = np.median(obs_d[max(HORIZONS)])
    print("\n" + "=" * 74)
    if rmax <= 1.5:
        res["verdict"] = "RECONVERGE -- full RAM buys nothing"
        print("  => the branches RECONVERGE. The one-step RAM difference is")
        print("     transient (a consumed input buffer or a counter), the futures")
        print("     are identical, and there is nothing to predict. A full-RAM")
        print("     observation would NOT help. Do not build it.")
    elif omax > 1.5:
        res["verdict"] = "DIVERGE -- visible later; RAM gives it early"
        print("  => the branches DIVERGE and the divergence becomes VISIBLE in")
        print("     pixels by the longer horizons. The information is real and")
        print("     merely DELAYED past the one-step window. Full RAM delivers it")
        print("     immediately -- the observation change is justified.")
    else:
        res["verdict"] = "RAM-ONLY DIVERGENCE"
        print("  => branches stay distinct in RAM but NEVER become visible in")
        print("     pixels. The distinction is real but unobservable to any")
        print("     pixel agent; full RAM is the only way to access it.")
    with open(os.path.join(REPO_ROOT, a.out), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {os.path.join(REPO_ROOT, a.out)}")
    return res


if __name__ == "__main__":
    main()
