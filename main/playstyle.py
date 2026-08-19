"""Characterize ego vs adversary PLAYSTYLE from self-play, by instrumenting the
actual actions + positions rather than eyeballing video.

Action space (transform_action, 22 discrete):
  0-8  directions:  0 neutral,1 UP(jump),2 DOWN(crouch/block),3 LEFT,4 RIGHT,
                    5 UP-LEFT,6 UP-RIGHT,7 DOWN-LEFT,8 DOWN-RIGHT
  9-15 attacks:     9 = no press; 10-15 = the 6 attack buttons
  16-21 specials:   the 6 SF combos (motion moves: fireball/uppercut/etc)
Ego is LEFT (opponent to the RIGHT -> RIGHT=approach); adversary is RIGHT
(opponent to the LEFT -> LEFT=approach). Reported per seat: the action mix,
aggression (attack+special rate), net advance (approach-retreat), spacing, and
damage dealt.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RIGHTWARD = {4, 6, 8}
LEFTWARD = {3, 5, 7}
UP = {1, 5, 6}
DOWN = {2, 7, 8}


def categorize(a, ego):
    """-> (bucket, is_attack, advance) for one action index and seat."""
    if a >= 16:
        return "special", True, 0
    if a >= 10:
        return "attack", True, 0
    if a == 9 or a == 0:
        return "neutral", False, 0
    if a == 1:
        return "jump", False, 0
    if a == 2:
        return "crouch/block", False, 0
    # remaining are directional (3-8): resolve advance vs retreat per seat
    rightward = a in RIGHTWARD
    approach = rightward if ego else (not rightward)  # ego opp is right
    bucket = ("jump" if a in UP else "crouch/block" if a in DOWN else
              ("approach" if approach else "retreat"))
    return bucket, False, (1 if approach else -1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--max_steps", type=int, default=2500)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="rs1")
    ap.add_argument("--out", default="playstyle.json")
    a = ap.parse_args(argv)

    import numpy as np
    from local_best_response import (build_lbr_venv, load_checkpoint, PolicyOps,
                                     resolve_matchups, infer_obs_kwargs)
    from stable_baselines3.common.save_util import load_from_zip_file

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, _lab, st = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(st, a.n_envs, **infer_obs_kwargs(data, a.ram_mask or None))
    L = load_checkpoint(a.ckpt, venv, a.device)
    model = L[0] if isinstance(L, tuple) else L
    ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
    rng = np.random.RandomState(a.seed)

    BUCKETS = ["neutral", "approach", "retreat", "jump", "crouch/block", "attack", "special"]
    ego_ct = {b: 0 for b in BUCKETS}
    adv_ct = {b: 0 for b in BUCKETS}
    ego_adv_sum = adv_adv_sum = 0     # net advance
    dists, ego_xs, adv_xs = [], [], []
    ego_dmg = adv_dmg = 0.0           # damage each seat DEALS
    prev_ah = np.full(a.n_envs, np.nan)   # PER-ENV previous hp (scalar prev = cross-env bug)
    prev_eh = np.full(a.n_envs, np.nan)
    n = 0

    obs = venv.reset()
    for t in range(a.max_steps):
        ego_a = np.asarray(ops.sample_ego(obs, rng)).reshape(-1)
        adv_a = np.asarray(ops.sample_adv(obs, rng)).reshape(-1)
        obs, r_l, r_r, d, infos = venv.step(ops.joint(ego_a, adv_a))
        for e in range(a.n_envs):
            info = infos[e]
            be, ae_atk, adv1 = categorize(int(ego_a[e]), ego=True)
            bd, ad_atk, adv2 = categorize(int(adv_a[e]), ego=False)
            ego_ct[be] += 1; adv_ct[bd] += 1
            ego_adv_sum += adv1; adv_adv_sum += adv2
            ax, ex = info.get("agent_x"), info.get("enemy_x")
            ah, eh = info.get("agent_hp"), info.get("enemy_hp")
            if ax is not None and ex is not None:
                dists.append(abs(float(ax) - float(ex))); ego_xs.append(float(ax)); adv_xs.append(float(ex))
            # damage dealt = opponent hp DROP this step, PER ENV (ignore resets)
            if ah is not None and eh is not None and not d[e]:
                if not np.isnan(prev_eh[e]) and 0 < (prev_eh[e] - eh) < 90:
                    ego_dmg += (prev_eh[e] - eh)     # ego dealt to enemy
                if not np.isnan(prev_ah[e]) and 0 < (prev_ah[e] - ah) < 90:
                    adv_dmg += (prev_ah[e] - ah)     # adv dealt to ego
            prev_ah[e] = ah if (ah is not None and not d[e]) else np.nan
            prev_eh[e] = eh if (eh is not None and not d[e]) else np.nan
            n += 1

    venv.close()

    def pct(ct):
        tot = max(sum(ct.values()), 1)
        return {b: 100.0 * ct[b] / tot for b in BUCKETS}
    ep, apd = pct(ego_ct), pct(adv_ct)
    dists = np.array(dists) if dists else np.zeros(1)
    ego_xs = np.array(ego_xs) if len(ego_xs) else np.zeros(1)
    adv_xs = np.array(adv_xs) if len(adv_xs) else np.zeros(1)

    print(f"\n  PLAYSTYLE  {os.path.basename(a.ckpt)}   ({n} step-samples, {a.n_envs} envs)")
    print(f"  {'bucket':>13} | {'EGO (Ryu,L)':>12} | {'ADV (Sagat,R)':>13}")
    for b in BUCKETS:
        print(f"  {b:>13} | {ep[b]:11.1f}% | {apd[b]:12.1f}%")
    ego_aggr = ep["attack"] + ep["special"]; adv_aggr = apd["attack"] + apd["special"]
    print(f"  {'AGGRESSION':>13} | {ego_aggr:11.1f}% | {adv_aggr:12.1f}%   (attack+special)")
    print(f"  {'net advance':>13} | {100.0*ego_adv_sum/max(n,1):11.1f}% | {100.0*adv_adv_sum/max(n,1):12.1f}%   (approach-retreat)")
    print(f"\n  spacing: distance mean {dists.mean():.1f} +/- {dists.std():.1f}  "
          f"(p10 {np.percentile(dists,10):.0f} / p50 {np.percentile(dists,50):.0f} / p90 {np.percentile(dists,90):.0f})")
    print(f"  mean pos: ego_x {ego_xs.mean():.0f}  adv_x {adv_xs.mean():.0f}   "
          f"(ego range {ego_xs.min():.0f}-{ego_xs.max():.0f}, adv range {adv_xs.min():.0f}-{adv_xs.max():.0f})")
    print(f"  damage DEALT: ego {ego_dmg:.0f}  adv {adv_dmg:.0f}   "
          f"(ratio {ego_dmg/max(adv_dmg,1):.2f}x)")

    out = dict(ckpt=os.path.basename(a.ckpt), n=n, ego=ep, adv=apd,
               ego_aggression=ego_aggr, adv_aggression=adv_aggr,
               ego_net_advance=100.0*ego_adv_sum/max(n,1),
               adv_net_advance=100.0*adv_adv_sum/max(n,1),
               dist_mean=float(dists.mean()), dist_p50=float(np.percentile(dists,50)),
               ego_x_mean=float(ego_xs.mean()), adv_x_mean=float(adv_xs.mean()),
               ego_damage=float(ego_dmg), adv_damage=float(adv_dmg))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
