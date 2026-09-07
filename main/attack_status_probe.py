#!/usr/bin/env python
"""Derive ATTACK_STATUSES empirically: which agent_status codes is the agent in
when it DEALS damage. Rolls out the policy, attributes each enemy-HP drop to the
agent's status over a short lookback window (attack startup->connect), and ranks
statuses by total damage dealt. The minimal set covering ~90% of damage = the
attack states. Also reports spacing at the moment of the hit (for PRESSURE_RANGE).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict
from stable_baselines3.common.save_util import load_from_zip_file
from local_best_response import (build_lbr_venv, resolve_matchups, PolicyOps,
    _extract_left_right_names_from_state, infer_obs_kwargs, preflight)
from behavior_probe import _lax_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--lookback", type=int, default=2, help="attribute a hit to statuses in the K steps up to it")
    a = ap.parse_args()

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    hi, lab, state = resolve_matchups(data, "all")[0]
    if a.state:
        state = a.state
    label = a.label or lab
    lc, rc = _extract_left_right_names_from_state(state)
    dt = dict(decision_timing="joint", actionable_statuses=(512, 514, 520), dwell_frames=4, max_skip_frames=90)
    venv = build_lbr_venv(state, a.n_envs, ego_char=lc, left_char=lc, right_char=rc,
                          charge_preserving_skip=True, **dt, **infer_obs_kwargs(data, None))
    n = a.n_envs
    dmg_by_status = defaultdict(float)   # total enemy-hp lost attributed to this agent status
    cnt_by_status = defaultdict(int)     # frames spent in status (for rate)
    hit_spacing = []                     # |agent_x-enemy_x| at the moment a hit lands
    total_dmg = 0.0
    try:
        m, _ = _lax_load(a.ckpt, venv, "cuda"); preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        rng = np.random.RandomState(0)
        hist = [[] for _ in range(n)]        # recent (status) per env, for lookback
        prev_ehp = [None] * n
        obs = venv.reset()
        for _ in range(a.steps):
            ea = ops.sample_ego(obs, rng); aa = ops.sample_adv(obs, rng)
            obs, rl, rr, dn, infos = venv.step(ops.joint(aa, ea))
            for i, inf in enumerate(infos):
                s = inf.get("agent_status"); ehp = inf.get("enemy_hp")
                ax, ex = inf.get("agent_x"), inf.get("enemy_x")
                if s is None or ehp is None:
                    continue
                s = int(s)
                cnt_by_status[s] += 1
                hist[i].append(s); hist[i] = hist[i][-a.lookback:]
                if prev_ehp[i] is not None:
                    d = prev_ehp[i] - float(ehp)
                    if 0 < d < 60:           # a hit (ignore round-reset hp jumps)
                        total_dmg += d
                        for st in set(hist[i]):
                            dmg_by_status[st] += d / len(set(hist[i]))
                        if ax is not None and ex is not None:
                            hit_spacing.append(abs(float(ax) - float(ex)))
                prev_ehp[i] = float(ehp)
    finally:
        venv.close()

    ranked = sorted(dmg_by_status.items(), key=lambda kv: -kv[1])
    cum = 0.0; keep = []
    for st, d in ranked:
        keep.append(st); cum += d
        if total_dmg > 0 and cum / total_dmg >= 0.90:
            break
    # CLEAN set: drop the actionable/neutral statuses (they only rank high because joint-timing
    # settles the per-step status to neutral, misattributing damage), then take the non-actionable
    # statuses covering >=90% of the NON-actionable damage. This is the usable attack_statuses.
    ACTIONABLE = {512, 514, 520}
    na = [(st, d) for st, d in ranked if st not in ACTIONABLE]
    na_total = sum(d for _, d in na)
    clean = []; c = 0.0
    for st, d in na:
        clean.append(st); c += d
        if na_total > 0 and c / na_total >= 0.90:
            break
    sp = np.array(hit_spacing) if hit_spacing else np.array([0.0])
    print(f"  {label}  total_dmg={total_dmg:.0f}  hits={len(hit_spacing)}")
    print("  status : damage_share  (frames)")
    for st, d in ranked[:12]:
        print(f"    {st:5d} : {d/max(total_dmg,1):.3f}        ({cnt_by_status[st]})")
    print(f"  >> raw ATTACK_STATUSES (>=90% of damage): {','.join(str(s) for s in sorted(keep))}")
    print(f"  >> CLEAN attack_statuses (non-actionable): {','.join(str(s) for s in sorted(clean))}")
    print(f"  >> hit spacing: median {np.median(sp):.0f}px  p75 {np.percentile(sp,75):.0f}px  p90 {np.percentile(sp,90):.0f}px")
    print(f"MACHINE {label} attack_statuses={'|'.join(str(s) for s in sorted(keep))} "
          f"clean_attack_statuses={'|'.join(str(s) for s in sorted(clean))} "
          f"hit_spacing_med={np.median(sp):.0f} hit_spacing_p90={np.percentile(sp,90):.0f}")


if __name__ == "__main__":
    main()
