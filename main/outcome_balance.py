"""Who is actually WINNING? Ego vs adversary round outcomes in self-play.

ep_rew_mean is MISLEADING at aggresive_coeff != 1: the game is positive-sum, so
an even trade pays BOTH players +2d and ep_rew is positive in a balanced brawl.
This counts actual round outcomes instead -- the env sets info['outcome'] to
'win'/'lose'/'draw' from the AGENT(=ego, left) perspective at each done -- and
also reports the end-of-round hp differential (agent_hp - enemy_hp), a continuous
measure that does not depend on the reward at all.

Run on an a=1 checkpoint AND an a=3 checkpoint: if ego win% is ~50% in both, the
"ego is winning because the value head is ego-perspective" hypothesis is dead. If
ego dominates specifically at a=3, that localises it to the a!=1 value convention.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outcome_balance.json")
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
    rng = np.random.RandomState(0)

    win = lose = draw = 0
    hp_diffs = []
    obs = venv.reset()
    done_ct = 0
    for t in range(a.max_steps):
        obs, r_l, r_r, d, infos = venv.step(
            ops.joint(ops.sample_ego(obs, rng), ops.sample_adv(obs, rng)))
        for e, di in enumerate(np.asarray(d).reshape(-1)):
            if not di:
                continue
            info = infos[e]
            oc = info.get("outcome")
            if oc == "win":
                win += 1
            elif oc == "lose":
                lose += 1
            elif oc == "draw":
                draw += 1
            else:
                # fall back to hp comparison if 'outcome' absent this step
                ah, eh = info.get("agent_hp", 0), info.get("enemy_hp", 0)
                win += ah > eh; lose += ah < eh; draw += ah == eh
            hp_diffs.append(float(info.get("agent_hp", 0) - info.get("enemy_hp", 0)))
            done_ct += 1
        if done_ct >= a.episodes:
            break
    venv.close()

    tot = max(win + lose + draw, 1)
    hd = np.array(hp_diffs) if hp_diffs else np.zeros(1)
    out = dict(ckpt=os.path.basename(a.ckpt), n=int(tot),
               ego_win=win / tot, ego_lose=lose / tot, draw=draw / tot,
               hp_diff_mean=float(hd.mean()), hp_diff_se=float(hd.std() / max(len(hd) ** 0.5, 1)))
    print(f"\n  {os.path.basename(a.ckpt)}  ({tot} rounds)")
    print(f"    ego win  {out['ego_win']:6.1%}   ego lose {out['ego_lose']:6.1%}   "
          f"draw {out['draw']:6.1%}")
    print(f"    end-of-round hp diff (agent - enemy) = {out['hp_diff_mean']:+.2f} "
          f"+/- {out['hp_diff_se']:.2f}")
    # honest read
    edge = out['ego_win'] - out['ego_lose']
    if abs(edge) < 0.08:
        print(f"    => BALANCED (|win-lose| = {abs(edge):.1%} < 8%). Neither seat dominates.")
    else:
        who = "EGO" if edge > 0 else "ADVERSARY"
        print(f"    => {who} dominates by {abs(edge):.1%}.")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"    wrote {a.out}")


if __name__ == "__main__":
    main()
