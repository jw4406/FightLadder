"""How do rounds END: timeout (clock expires, both alive) vs KO (a health bar hits
zero)? Cross-tabulated against the ego outcome (win/lose/draw).

Purpose: the ac=1 unscaled arm (rs1) has the ego winning 65% with a NARROW hp
lead (+18.6) and ep_len grown to the timer cap (517). That pattern says the ego
wins by STALLING to timeout with a slight lead, not by KO. This tool confirms or
kills that: classify each round end as KO (min(agent_hp, enemy_hp) < 0) or
TIMEOUT (both >= 0), and report the 3x2 table plus the winner/loser hp margins.

Mirrors outcome_balance.py's env + policy setup exactly so the two are directly
comparable; the ONLY addition is the timeout/KO split and the hp margin summary.
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
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ko_thresh", type=float, default=0.0,
                    help="hp <= this counts as a KO. Env KO branch uses hp < 0; "
                         "0.0 catches clamped-to-zero KOs too.")
    ap.add_argument("--out", default="round_end_mode.json")
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

    # cross-tab counters: outcome in {win,lose,draw} x mode in {ko,timeout}
    tab = {o: {"ko": 0, "timeout": 0} for o in ("win", "lose", "draw")}
    win_ah, win_eh = [], []      # winner-side hp margins on ego WINS
    all_ah, all_eh = [], []
    done_ct = 0
    obs = venv.reset()
    for t in range(a.max_steps):
        obs, r_l, r_r, d, infos = venv.step(
            ops.joint(ops.sample_ego(obs, rng), ops.sample_adv(obs, rng)))
        for e, di in enumerate(np.asarray(d).reshape(-1)):
            if not di:
                continue
            info = infos[e]
            ah = float(info.get("agent_hp", 0.0))
            eh = float(info.get("enemy_hp", 0.0))
            # outcome from the env if present, else hp comparison
            oc = info.get("outcome")
            if oc not in ("win", "lose", "draw"):
                oc = "win" if ah > eh else ("lose" if ah < eh else "draw")
            # mode: a KO means a bar hit zero; else the clock ran out
            mode = "ko" if (ah <= a.ko_thresh or eh <= a.ko_thresh) else "timeout"
            tab[oc][mode] += 1
            all_ah.append(ah); all_eh.append(eh)
            if oc == "win":
                win_ah.append(ah); win_eh.append(eh)
            done_ct += 1
        if done_ct >= a.episodes:
            break
    venv.close()

    # FAIL LOUD if we did not actually collect the rounds we asked for
    if done_ct < a.episodes:
        raise RuntimeError(f"collected only {done_ct}/{a.episodes} rounds in "
                           f"{a.max_steps} steps -- raise --max_steps; a partial "
                           f"count is indistinguishable from a real result")

    tot = sum(tab[o][m] for o in tab for m in tab[o])
    def pct(x): return f"{100.0 * x / max(tot, 1):5.1f}%"

    print(f"\n  {os.path.basename(a.ckpt)}   ({tot} rounds)")
    print(f"  {'outcome':>8} | {'KO':>10} {'TIMEOUT':>10} | {'row tot':>8}")
    for o in ("win", "lose", "draw"):
        ko, to = tab[o]["ko"], tab[o]["timeout"]
        print(f"  {o:>8} | {pct(ko):>10} {pct(to):>10} | {pct(ko+to):>8}")
    to_all = sum(tab[o]["timeout"] for o in tab)
    ko_all = sum(tab[o]["ko"] for o in tab)
    print(f"  {'TOTAL':>8} | {pct(ko_all):>10} {pct(to_all):>10} |")
    if win_ah:
        wa, we = np.array(win_ah), np.array(win_eh)
        print(f"\n  on ego WINS (n={len(wa)}): agent_hp {wa.mean():6.1f}+/-{wa.std():.1f}"
              f"   enemy_hp {we.mean():6.1f}+/-{we.std():.1f}"
              f"   margin {(wa-we).mean():+6.1f}")
        print(f"    read: high both-hp + timeout => stall-to-timeout; "
              f"low enemy_hp + KO => decisive win")

    out = dict(ckpt=os.path.basename(a.ckpt), n=int(tot),
               table=tab, timeout_frac=to_all / max(tot, 1),
               ko_frac=ko_all / max(tot, 1),
               win_agent_hp_mean=float(np.mean(win_ah)) if win_ah else None,
               win_enemy_hp_mean=float(np.mean(win_eh)) if win_eh else None)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
