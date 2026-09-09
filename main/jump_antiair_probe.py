#!/usr/bin/env python
"""Diagnose the mutual-jumping equilibrium.

Measures, per seat, in the training-matched env (joint timing, cps=True):
  - jump_frac : fraction of frames the fighter is AIRBORNE (agent_y off the grounded level)
  - antiair_rate : fraction of that fighter's JUMPS that get PUNISHED -- the fighter takes
                   damage while airborne or within `land_window` frames of landing.
LOW antiair_rate => jumping is an unpunished free local optimum (the root cause).
HIGH antiair_rate => jumps are being answered; jumping is entropy/mechanic-driven, not free.
Airborne = |y - grounded_median| > thr. Reports y-range so the threshold can be sanity-checked.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stable_baselines3.common.save_util import load_from_zip_file
from local_best_response import (build_lbr_venv, resolve_matchups, PolicyOps,
    _extract_left_right_names_from_state, infer_obs_kwargs, preflight)
from behavior_probe import _lax_load


def jumps(y, hp, thr, land_window):
    """One env's y/hp series -> (n_jumps, n_punished, air_frac). y,hp are 1-D arrays."""
    y = np.asarray(y, float); hp = np.asarray(hp, float)
    ground = np.median(y)
    air = np.abs(y - ground) > thr
    starts = np.where((~air[:-1]) & air[1:])[0] + 1          # grounded->airborne edges
    nj = np_ = 0
    for s in starts:
        e = s
        while e < len(air) and air[e]:
            e += 1
        w = min(len(hp), e + land_window)
        took = hp[s] - hp[s:w].min() if w > s else 0.0
        nj += 1
        if 0 < took < 60:                                    # took damage (ignore round-reset jumps)
            np_ += 1
    return nj, np_, float(air.mean()), ground


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--thr", type=float, default=6.0, help="airborne threshold (|y-ground| px)")
    ap.add_argument("--land_window", type=int, default=3, help="frames after landing still counted as the jump")
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
    AY = [[] for _ in range(n)]; AH = [[] for _ in range(n)]
    EY = [[] for _ in range(n)]; EH = [[] for _ in range(n)]
    try:
        m, _ = _lax_load(a.ckpt, venv, "cuda"); preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        rng = np.random.RandomState(0)
        obs = venv.reset()
        for _ in range(a.steps):
            ea = ops.sample_ego(obs, rng); aa = ops.sample_adv(obs, rng)
            obs, rl, rr, dn, infos = venv.step(ops.joint(aa, ea))
            for i, inf in enumerate(infos):
                if inf.get("agent_y") is None:
                    continue
                AY[i].append(float(inf["agent_y"])); AH[i].append(float(inf["agent_hp"]))
                EY[i].append(float(inf["enemy_y"])); EH[i].append(float(inf["enemy_hp"]))
    finally:
        venv.close()

    def agg(Ys, Hs):
        nj = npu = 0; af = []; yr = []
        for y, h in zip(Ys, Hs):
            if len(y) < 3:
                continue
            j, p, fr, g = jumps(y, h, a.thr, a.land_window)
            nj += j; npu += p; af.append(fr); yr += [min(y), max(y)]
        return nj, npu, (np.mean(af) if af else 0.0), (min(yr) if yr else 0), (max(yr) if yr else 0)

    # agent (ego=ChunLi/left) jumps -- punished by the ADVERSARY's anti-air
    a_nj, a_np, a_af, a_ymin, a_ymax = agg(AY, AH)
    # enemy (adv/right) jumps -- punished by the AGENT's anti-air
    e_nj, e_np, e_af, e_ymin, e_ymax = agg(EY, EH)
    a_rate = a_np / max(a_nj, 1); e_rate = e_np / max(e_nj, 1)
    print(f"  {label}")
    print(f"    EGO  jump_frac={a_af:.3f}  jumps={a_nj}  antiaired={a_rate:.3f}  (y {a_ymin:.0f}..{a_ymax:.0f})")
    print(f"    ADV  jump_frac={e_af:.3f}  jumps={e_nj}  antiaired={e_rate:.3f}  (y {e_ymin:.0f}..{e_ymax:.0f})")
    print(f"MACHINE {label} ego_jump_frac={a_af:.4f} ego_antiaired={a_rate:.4f} ego_jumps={a_nj} "
          f"adv_jump_frac={e_af:.4f} adv_antiaired={e_rate:.4f} adv_jumps={e_nj}")


if __name__ == "__main__":
    main()
