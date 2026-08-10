"""STEP 0: which observation actually resolves action-distinct successors?

Measured so far, at a median decision point over the 22 ego actions (adversary
fixed; the ceiling is 21 because actions 0 and 9 are byte-identical no-ops):

    RAM + frames + attrs   21 / 21     lbr_fingerprint -- full discrimination
    pixels                  1 / 21     at ANY resolution, frame count, or
                                       channel set. The differing state is not
                                       rendered at all.

So a replacement observation is required. Two candidates, and they differ by
three orders of magnitude in size:

    info    the ~14 curated retro variables already wrapped by InfoObsWrapper
            (`--obs_type info`), including agent_status / enemy_status
    ram     the full Genesis RAM array (~64KB of mostly-irrelevant bytes)

This measures both BEFORE either is wired into the SPAR policy, because the
answer decides how much work is justified:

    info == 21   -> the existing wrapper suffices; 14 features beat 64KB on
                    learnability and the change is small
    info << 21   -> the discriminating byte is not among the curated variables
                    and full RAM is genuinely necessary

Per-key distinctness is reported too, so if a single variable carries the signal
that variable can be added rather than the whole RAM.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPAND_ROOT = "cand_root"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_states", type=int, default=120)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="obs_candidate_probe.json")
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

        d_ram, d_info, d_obs = [], [], []
        per_key = defaultdict(list)
        keys = None
        ram_bytes = None
        obs = venv.reset()
        n_exp = int(np.ceil(a.n_states / n))
        for e in range(n_exp):
            for _ in range(a.stride if e else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]

            venv.env_method("lbr_snapshot", EXPAND_ROOT)
            rams = [[] for _ in range(n)]
            infos = [[] for _ in range(n)]
            obss = [[] for _ in range(n)]
            for i in range(na):
                venv.env_method("lbr_restore", EXPAND_ROOT)
                o1 = venv.step(ops.joint(np.full(n, i), np.zeros(n, dtype=int)))[0]
                res = venv.env_method("lbr_state_variants")
                for k in range(n):
                    if keys is None:
                        keys, ram_bytes = res[k]["keys"], res[k]["ram_bytes"]
                    rams[k].append(res[k]["ram"])
                    infos[k].append(tuple(res[k]["vals"]))
                    obss[k].append(hash(o1[k].tobytes()))
            venv.env_method("lbr_restore", EXPAND_ROOT)
            venv.env_method("lbr_drop", EXPAND_ROOT)

            for k in range(n):
                d_ram.append(len(set(rams[k])))
                d_info.append(len(set(infos[k])))
                d_obs.append(len(set(obss[k])))
                arr = np.array(infos[k])                      # (na, n_keys)
                for c, key in enumerate(keys):
                    per_key[key].append(len(set(arr[:, c].tolist())))
            print(f"   expansion {e+1}/{n_exp}", flush=True)
    finally:
        venv.close()

    d_ram = np.array(d_ram); d_info = np.array(d_info); d_obs = np.array(d_obs)
    S = d_ram.size
    ceiling = na - 1
    print("\n" + "=" * 74)
    print(f"OBSERVATION CANDIDATES  {os.path.basename(a.ckpt)}   {S} states")
    print(f"  distinct successors over {na} ego actions; ceiling {ceiling} "
          f"(0 and 9 are identical);  RAM is {ram_bytes:,} bytes")
    print("=" * 74)
    print(f"  {'candidate':<20} {'median':>7} {'p25':>6} {'p75':>6} {'max':>6}")
    for nm, d in (("pixels (current)", d_obs), ("info (14 vars)", d_info),
                  ("ram (full)", d_ram)):
        print(f"  {nm:<20} {np.median(d):>7.0f} {np.percentile(d,25):>6.0f} "
              f"{np.percentile(d,75):>6.0f} {d.max():>6.0f}")

    print(f"\n  per-variable distinctness (median over states):")
    order = sorted(keys, key=lambda k: -np.median(per_key[k]))
    for key in order:
        m = np.median(per_key[key])
        bar = "#" * int(round(20 * m / max(ceiling, 1)))
        print(f"    {key:<20} {m:>5.0f}  {bar}")

    res = {"checkpoint": os.path.basename(a.ckpt), "n_states": S,
           "ceiling": ceiling, "ram_bytes": ram_bytes,
           "pixels_median": float(np.median(d_obs)),
           "info_median": float(np.median(d_info)),
           "ram_median": float(np.median(d_ram)),
           "per_key_median": {k: float(np.median(per_key[k])) for k in keys}}
    print("\n" + "=" * 74)
    if np.median(d_info) >= 0.9 * np.median(d_ram) and np.median(d_info) > 1:
        res["verdict"] = "INFO SUFFICES"
        print("  => the ~14 curated variables resolve essentially what full RAM")
        print("     does. Use --obs_type info: far smaller, far more learnable,")
        print("     and the wrapper already exists. Full RAM is unnecessary.")
    elif np.median(d_ram) > 1:
        res["verdict"] = "FULL RAM NEEDED"
        print("  => the curated variables MISS the discriminating state; only")
        print("     full RAM resolves it. Build the RAM observation, and use the")
        print("     per-variable table above to see what is missing.")
    else:
        res["verdict"] = "NEITHER"
        print("  => neither candidate resolves branches. Re-examine before")
        print("     changing the observation at all.")
    with open(os.path.join(REPO_ROOT, a.out), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {os.path.join(REPO_ROOT, a.out)}")
    return res


if __name__ == "__main__":
    main()
