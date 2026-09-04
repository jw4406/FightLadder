"""Quantify a policy checkpoint's SPACING and CONTACT behavior -- zone vs melee.

On-policy self-play rollout (ego head + adv head from the checkpoint). Collects
per agent-step |agent_x - enemy_x| (SPACING) and HP deltas (CONTACT). A zoner
sits at large spacing and trades little contact until a projectile lands; a melee
policy closes distance and trades HP up close. Compare a matchup against a known
zoner (Ryu/Sagat) to make "engages closer than the zoner" a number, not a claim.

Rolled fixed-clock (decision_timing off) so two checkpoints are compared on the
SAME env dynamics; the decision-timing-trained policy still exhibits its spacing
strategy (locked-frame actions are ignored by the game either way).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3.common.save_util import load_from_zip_file
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups, REPO_ROOT,
                                 _extract_left_right_names_from_state)


def _lax_load(ckpt, venv, device):
    try:
        return load_checkpoint(ckpt, venv, device)
    except RuntimeError as e:
        if "q_value_net" not in str(e):
            raise
        import stable_baselines3.common.base_class as _bc
        _orig = _bc.BaseAlgorithm.set_parameters
        _bc.BaseAlgorithm.set_parameters = lambda self, d, exact_match=True, device="auto": \
            _orig(self, d, exact_match=False, device=device)
        try:
            return load_checkpoint(ckpt, venv, device)
        finally:
            _bc.BaseAlgorithm.set_parameters = _orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", default="", help="override matchup state")
    ap.add_argument("--label", default="")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--close_px", type=float, default=60.0, help="melee range threshold")
    ap.add_argument("--far_px", type=float, default=120.0, help="zone range threshold")
    ap.add_argument("--out", default="")
    # Decision-timing: default "off" preserves the historical fixed-clock behavior
    # (same-clock cross-checkpoint comparison). Pass "joint"/"ego" to run the policy
    # IN-DISTRIBUTION for decision-timing-trained checkpoints.
    ap.add_argument("--decision_timing", choices=["off", "ego", "joint"], default="off")
    ap.add_argument("--actionable_statuses", type=str, default="512,514,520")
    ap.add_argument("--dwell_frames", type=int, default=4)
    ap.add_argument("--max_skip_frames", type=int, default=90)
    a = ap.parse_args()

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    hi, lab, state = resolve_matchups(data, "all")[0]
    if a.state:
        state = a.state
    label = a.label or lab
    _dt_kw = {}
    if a.decision_timing != "off":
        _dt_kw = dict(decision_timing=a.decision_timing,
                      actionable_statuses=tuple(int(x) for x in
                          a.actionable_statuses.split(",") if x.strip()),
                      dwell_frames=a.dwell_frames, max_skip_frames=a.max_skip_frames)
    # Build the env with the CHECKPOINT's characters so the combo table (hence the
    # action space) matches. Without this a charge char like Vega (63+2=65) gets the
    # default motion-char table (63+6=69) and the checkpoint fails to load. ego is the
    # left/protagonist char in these matchup states.
    _lc, _rc = _extract_left_right_names_from_state(state)
    venv = build_lbr_venv(state, a.n_envs, ego_char=_lc, left_char=_lc, right_char=_rc, **_dt_kw)
    try:
        m, _ = _lax_load(a.ckpt, venv, "cuda"); preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        rng = np.random.RandomState(0)
        SP, DHP, AST = [], [], []       # spacing, |hp delta|, agent_status
        prev = None
        obs = venv.reset()
        for _ in range(a.steps):
            ea = ops.sample_ego(obs, rng); aa = ops.sample_adv(obs, rng)
            # joint(lbr_actions, pol_actions): lbr=adv(Guile)=aa, pol=ego(Vega)=ea.
            # Order MUST be (aa, ea) so ego lands on LEFT / adv on RIGHT (matches
            # duel.py's [left=ego, right=adv]); (ea, aa) swaps seats -> off-distribution.
            obs, rl, rr, dn, infos = venv.step(ops.joint(aa, ea))
            for i in infos:
                ax, ex = i.get("agent_x"), i.get("enemy_x")
                if ax is None or ex is None:
                    continue
                SP.append(abs(float(ax) - float(ex)))
                AST.append(int(i.get("agent_status", -1)))
            hp = np.array([[float(i.get("agent_hp", 0)), float(i.get("enemy_hp", 0))]
                           for i in infos])
            if prev is not None:
                d = np.abs(hp - prev)
                # ignore round-reset jumps (hp jumps back to full ~176)
                DHP.extend(d[(d < 100).all(axis=1)].sum(axis=1).tolist())
            prev = hp
    finally:
        venv.close()

    SP = np.array(SP); DHP = np.array(DHP)
    res = {
        "label": label, "ckpt": os.path.basename(a.ckpt), "n_samples": int(SP.size),
        "spacing_median": float(np.median(SP)),
        "spacing_p10": float(np.percentile(SP, 10)),
        "spacing_p90": float(np.percentile(SP, 90)),
        "spacing_mean": float(SP.mean()),
        "frac_close_lt%d" % int(a.close_px): float((SP < a.close_px).mean()),
        "frac_far_gt%d" % int(a.far_px): float((SP > a.far_px).mean()),
        "contact_rate": float((DHP > 0).mean()) if DHP.size else float("nan"),
        "mean_dmg_per_step": float(DHP.mean()) if DHP.size else float("nan"),
    }
    st, cnt = np.unique(np.array(AST), return_counts=True)
    order = np.argsort(-cnt)[:5]
    res["top_statuses"] = {int(st[i]): round(float(cnt[i] / cnt.sum()), 3) for i in order}

    print(f"\n  {label}   ckpt={res['ckpt']}   n={res['n_samples']}")
    print(f"  SPACING |agent_x-enemy_x|:  median {res['spacing_median']:.0f}px  "
          f"(p10 {res['spacing_p10']:.0f} / p90 {res['spacing_p90']:.0f})")
    print(f"    frac CLOSE (<{int(a.close_px)}px, melee): {res['frac_close_lt%d'%int(a.close_px)]:.1%}"
          f"   frac FAR (>{int(a.far_px)}px, zone): {res['frac_far_gt%d'%int(a.far_px)]:.1%}")
    print(f"  CONTACT: {res['contact_rate']:.1%} of steps trade damage   "
          f"mean dmg/step {res['mean_dmg_per_step']:.3f}")
    print(f"  top agent_status: {res['top_statuses']}")
    if a.out:
        with open(os.path.join(REPO_ROOT, a.out), "w") as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {a.out}")
    print(f"MACHINE {label} spacing_med={res['spacing_median']:.1f} "
          f"close={res['frac_close_lt%d'%int(a.close_px)]:.3f} "
          f"far={res['frac_far_gt%d'%int(a.far_px)]:.3f} contact={res['contact_rate']:.3f}")


if __name__ == "__main__":
    main()
