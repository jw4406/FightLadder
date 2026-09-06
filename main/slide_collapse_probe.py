#!/usr/bin/env python
"""Slide-collapse probe: how concentrated is the ego's action distribution?

Rolls out the ego policy (training-matched env: joint timing, cps=True) and records
every ego action. Reports:
  - normalized action entropy (low => collapsed onto a few actions),
  - top-1 / top-3 action share,
  - the fraction that are the Vega crouching-HK SLIDE {17,52,59} (down / down-fwd /
    down-back + heavy kick), the documented cheese attractor.
A high slide fraction and/or low entropy = the ego has collapsed onto the slide.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stable_baselines3.common.save_util import load_from_zip_file
from local_best_response import (build_lbr_venv, resolve_matchups, PolicyOps,
    _extract_left_right_names_from_state, infer_obs_kwargs, preflight)
from behavior_probe import _lax_load

SLIDE = [17, 52, 59]   # Vega crouching-HK slide variants (down/down-fwd/down-back + HK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n_envs", type=int, default=8)
    a = ap.parse_args()

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    hi, lab, state = resolve_matchups(data, "all")[0]
    if a.state:
        state = a.state
    label = a.label or lab
    lc, rc = _extract_left_right_names_from_state(state)
    dt = dict(decision_timing="joint", actionable_statuses=(512, 514, 520),
              dwell_frames=4, max_skip_frames=90)
    venv = build_lbr_venv(state, a.n_envs, ego_char=lc, left_char=lc, right_char=rc,
                          charge_preserving_skip=True, **dt, **infer_obs_kwargs(data, None))
    acts = []
    try:
        m, _ = _lax_load(a.ckpt, venv, "cuda")
        preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        rng = np.random.RandomState(0)
        obs = venv.reset()
        for _ in range(a.steps):
            ea = np.asarray(ops.sample_ego(obs, rng)).reshape(-1)
            aa = ops.sample_adv(obs, rng)
            acts.extend(ea.tolist())
            obs, rl, rr, dn, infos = venv.step(ops.joint(aa, ea))
        nA = int(ops.n_actions)
    finally:
        venv.close()

    acts = np.array(acts, dtype=int); N = len(acts)
    vals, counts = np.unique(acts, return_counts=True)
    p = counts / N
    order = np.argsort(-counts)
    H = float(-(p * np.log(p)).sum()); Hmax = float(np.log(nA))
    top1 = float(counts[order[0]] / N)
    top3 = float(counts[order[:3]].sum() / N)
    slide = float(np.isin(acts, SLIDE).mean())
    topstr = " ".join(f"{int(vals[i])}:{counts[i]/N:.3f}" for i in order[:10])
    print(f"  {label}  ego_actions={N}  action_space={nA}")
    print(f"    norm_entropy={H/Hmax:.3f}  (low => collapsed)   top1={top1:.3f}  top3={top3:.3f}")
    print(f"    slide{{17,52,59}}={slide:.3f}")
    print(f"    top10 actions (idx:frac): {topstr}")
    print(f"MACHINE {label} N={N} Hnorm={H/Hmax:.4f} top1={top1:.4f} top3={top3:.4f} "
          f"slide={slide:.4f} nA={nA}")


if __name__ == "__main__":
    main()
