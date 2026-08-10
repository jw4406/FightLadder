"""Is the branch degeneracy REAL, or manufactured by the observation pipeline?

A4 in payoff_structure.py found a median of 1 distinct successor OBSERVATION out
of 484 joint actions. That was measured by hashing the agent's observation --
and the observation is downsampled three separate ways:

    temporal  FrameStack keeps 12 frames; _get_obs takes indices 0, 4, 8 only.
              The NEWEST THREE FRAMES (9,10,11) are never shown to the network,
              so the freshest content is already 3 emulator frames stale.
    spatial   o[::2, ::2] -> 100x128 from 200x256.
    channel   frame 0 contributes only its RED plane, frame 4 only GREEN,
              frame 8 only BLUE (`i % 3`). Time and colour are entangled.

So "median 1 distinct successor" may mean either

  (a) the GAME genuinely does not distinguish these actions at this timescale
      (hitstun / blockstun / animation lockout), or
  (b) the game DOES distinguish them and the observation pipeline throws the
      distinction away.

These have opposite implications. Under (a) no critic can help and the
joint-action direction is dead on its merits. Under (b) the information exists,
the agent simply cannot perceive it, and every negative result this week is a
statement about the INPUT rather than about the algorithm.

This separates them by hashing BOTH, at the same branch, in the same run:

    RAM fingerprint   lbr_fingerprint() -- md5 of emulator RAM. Ground truth,
                      no downsampling of any kind.
    observation hash  exactly what A4 used, and exactly what the critic sees.

    RAM ~ obs         -> (a). The game is genuinely forced. Conclusion stands.
    RAM >> obs        -> (b). The observation is the binding constraint, and the
                      degeneracy is manufactured.

NOTE ON WHAT IS ALREADY SAFE: the payoff ANOVA's interaction term (gamma ~0.7%)
was computed from REWARDS, which come from agent_hp/enemy_hp in RAM and never
pass through the observation pipeline. That result is unaffected by whatever
this script finds.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPAND_ROOT = "degen_root"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_states", type=int, default=120)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="obs_vs_state_degeneracy.json")
    a = ap.parse_args(argv)

    import json
    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, splice_terminal,
                                     REPO_ROOT)

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

        OBS_H, RAM_H = [], []
        obs = venv.reset()
        n_exp = int(np.ceil(a.n_states / n))
        for e in range(n_exp):
            for _ in range(a.stride if e else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]

            venv.env_method("lbr_snapshot", EXPAND_ROOT)
            oh = [[None] * na for _ in range(na)]
            rh = [[None] * na for _ in range(na)]
            for i in range(na):
                for j in range(na):
                    venv.env_method("lbr_restore", EXPAND_ROOT)
                    o1, r_l, r_r, d, infos = venv.step(
                        ops.joint(np.full(n, i), np.full(n, j)))
                    o1 = splice_terminal(o1, np.asarray(d, bool), infos)
                    oh[i][j] = [hash(o1[k].tobytes()) for k in range(n)]
                    # RAM digest of the SUCCESSOR, per env. Pipe-cheap (a str).
                    rh[i][j] = list(venv.env_method("lbr_fingerprint"))
            venv.env_method("lbr_restore", EXPAND_ROOT)
            venv.env_method("lbr_drop", EXPAND_ROOT)

            for k in range(n):
                OBS_H.append([oh[i][j][k] for i in range(na) for j in range(na)])
                RAM_H.append([rh[i][j][k] for i in range(na) for j in range(na)])
            print(f"   expansion {e+1}/{n_exp}", flush=True)
    finally:
        venv.close()

    d_obs = np.array([len(set(x)) for x in OBS_H])
    d_ram = np.array([len(set(x)) for x in RAM_H])
    S = d_obs.size

    def q(v):
        return (int(np.min(v)), int(np.percentile(v, 25)), int(np.median(v)),
                int(np.percentile(v, 75)), int(np.max(v)))

    print("\n" + "=" * 74)
    print(f"OBSERVATION vs GAME-STATE DEGENERACY   {os.path.basename(a.ckpt)}   "
          f"{S} states")
    print("=" * 74)
    print(f"  distinct successors out of {na*na} branches")
    print(f"  {'':18} {'min':>6} {'p25':>6} {'med':>6} {'p75':>6} {'max':>6}")
    print(f"  {'OBSERVATION':18} " + " ".join(f"{x:>6}" for x in q(d_obs)))
    print(f"  {'RAM (ground truth)':18} " + " ".join(f"{x:>6}" for x in q(d_ram)))
    ratio = float(np.median(d_ram) / max(np.median(d_obs), 1))
    print(f"\n  median RAM / median OBS = {ratio:.1f}x")
    print(f"  states forced in OBS but NOT in RAM: "
          f"{int(((d_obs == 1) & (d_ram > 1)).sum())}/{S} "
          f"({((d_obs == 1) & (d_ram > 1)).mean():.1%})")
    print(f"  states genuinely forced in RAM:      "
          f"{int((d_ram == 1).sum())}/{S} ({(d_ram == 1).mean():.1%})")

    res = {"checkpoint": os.path.basename(a.ckpt), "n_states": S,
           "obs_median": float(np.median(d_obs)), "ram_median": float(np.median(d_ram)),
           "ratio": ratio,
           "forced_in_obs_only": float(((d_obs == 1) & (d_ram > 1)).mean()),
           "forced_in_ram": float((d_ram == 1).mean())}
    print("\n" + "=" * 74)
    if ratio >= 3:
        res["verdict"] = "OBSERVATION-MANUFACTURED"
        print("  VERDICT: the degeneracy is largely MANUFACTURED by the")
        print("  observation. The game distinguishes these branches; the network")
        print("  cannot see the difference. Every branch-level negative this week")
        print("  is a statement about the INPUT, not about the algorithm.")
    else:
        res["verdict"] = "GENUINE"
        print("  VERDICT: the game itself does not distinguish these branches at")
        print("  this timescale. Downsampling is NOT the binding constraint and")
        print("  the joint-action conclusion stands on its merits.")
    with open(os.path.join(REPO_ROOT, a.out), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {os.path.join(REPO_ROOT, a.out)}")
    return res


if __name__ == "__main__":
    main()
