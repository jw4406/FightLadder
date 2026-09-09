#!/usr/bin/env python
"""Measure charge-special FIRING RATE of a trained policy in self-play.

Reads info['p1_special_active']/['p2_special_active'] (retro_wrappers, the VALIDATED
sustained raw-RAM flags [32770]==12 / [33410]==12) and counts rising edges per seat
= number of special-move fires. P1 = left = ego (Vega); P2 = right = adv (opponent).
Env matches training: joint decision-timing, dwell 4, charge_preserving_skip=True.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stable_baselines3.common.save_util import load_from_zip_file
from local_best_response import (build_lbr_venv, resolve_matchups, PolicyOps,
    _extract_left_right_names_from_state, infer_obs_kwargs, preflight)
from behavior_probe import _lax_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", default="", help="override matchup state")
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
                          charge_preserving_skip=True, **dt,
                          **infer_obs_kwargs(data, None))
    try:
        m, _ = _lax_load(a.ckpt, venv, "cuda")
        preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        rng = np.random.RandomState(0)
        n = a.n_envs
        p1_prev = np.zeros(n, bool); p2_prev = np.zeros(n, bool)
        p1_fires = p2_fires = p1_active = p2_active = nsteps = 0
        obs = venv.reset()
        for _ in range(a.steps):
            ea = ops.sample_ego(obs, rng); aa = ops.sample_adv(obs, rng)
            obs, rl, rr, dn, infos = venv.step(ops.joint(aa, ea))
            for i, inf in enumerate(infos):
                p1 = bool(inf.get('p1_special_active', 0))
                p2 = bool(inf.get('p2_special_active', 0))
                if p1 and not p1_prev[i]: p1_fires += 1
                if p2 and not p2_prev[i]: p2_fires += 1
                p1_prev[i] = p1; p2_prev[i] = p2
                p1_active += p1; p2_active += p2; nsteps += 1
    finally:
        venv.close()

    r1 = p1_active / max(nsteps, 1); r2 = p2_active / max(nsteps, 1)
    print(f"  {label}  steps={nsteps}")
    print(f"    P1 Vega(ego)   fires={p1_fires:4d}  active_frac={r1:.4f}")
    print(f"    P2 {rc}(adv)    fires={p2_fires:4d}  active_frac={r2:.4f}")
    print(f"MACHINE {label} p1_fires={p1_fires} p1_active={r1:.4f} "
          f"p2_fires={p2_fires} p2_active={r2:.4f} steps={nsteps}")


if __name__ == "__main__":
    main()
