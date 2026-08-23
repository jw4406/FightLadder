"""WHICH downsampling destroys the branch information?

RAM distinguishes 441 = 21^2 successors at a median decision point -- every
genuinely distinct joint action (0 and 9 are byte-identical no-ops, so 21 real
actions per side). The agent's observation distinguishes ONE. `_get_obs` applies
three independent reductions, and this attributes the collapse among them:

    current       frames[::4] (indices 0,4,8 of 12), o[::2,::2], channel i%3
    recent        frames[3::4] (indices 3,7,11) -- SAME SHAPE, SAME COST, only
                  sampled to include the NEWEST frame. An action's effect shows
                  up at the END of the 8-frame window, so if the newest frames
                  are where the information is, this is a free fix.
    all_frames    all 12 frames        (temporal reduction removed)
    full_spatial  200x256              (spatial reduction removed)
    all_channels  all 3 RGB per frame  (channel reduction removed)
    everything    all three removed    (upper bound for a pixel observation)

SCOPE: ego actions only, with the adversary held at action 0. That is a 22-way
test, not 484-way -- 22x cheaper and sufficient, since the question is whether
the observation resolves ACTIONS at all. The ceiling is 21 distinct (0 and 9
coincide), and RAM is measured alongside as ground truth.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPAND_ROOT = "attrib_root"
VARIANTS = ("current", "recent", "all_frames", "full_spatial",
            "all_channels", "everything")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_states", type=int, default=120)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="",
                    help="output JSON; default obs_attribution_nsf{nsf}_{sampler}.json")
    # FRAMESKIP SWEEP. The checkpoint's CNN is fixed at K=3 input channels, and
    # K = len(range(0, num_stack, num_step_frames//2)). To shorten the frameskip
    # WITHOUT breaking the CNN, num_stack must be scaled with num_step_frames so
    # K stays 3: nsf/ns = 4/6, 8/12, 12/18, 16/24 all give K=3. num_step_frames
    # must be divisible by 4 (build_sf_combos asserts it -- a 4-input motion
    # cannot be stretched unevenly), so nsf in {4,8,12,16}, not 2 or 6.
    ap.add_argument("--num_step_frames", type=int, default=8)
    ap.add_argument("--num_stack", type=int, default=12)
    ap.add_argument("--random_sampler", action="store_true",
                    help="reach decision points with RANDOM actions instead of "
                         "the policy -- decouples the state distribution from the "
                         "OOD policy when num_step_frames != the trained 8.")
    ap.add_argument("--by_status", action="store_true",
                    help="also bucket per-root action-distinctness by the ego's "
                         "agent_status at the root -- tests whether the 441->1 "
                         "collapse is driven by NON-ACTIONABLE states (locked in "
                         "an animation the game ignores input during).")
    ap.add_argument("--decision_timing", choices=["off", "ego", "joint"], default="off",
                    help="gate env steps on actionability -- the FIX under test. "
                         "With it on, roots are reached at actionable frames, so "
                         "the median distinctness should jump 1 -> ~11.")
    ap.add_argument("--actionable_statuses", type=str, default="512,514,520",
                    help="ego agent_status values that are actionable (from "
                         "--by_status). Default is the empirical high-distinctness set.")
    ap.add_argument("--max_skip_frames", type=int, default=90)
    ap.add_argument("--dwell_frames", type=int, default=1,
                    help="require the ego to be actionable for this many CONSECUTIVE "
                         "frames before returning -- skips past the recovery-settle "
                         "the raw status byte returns on. 1 = first actionable frame.")
    ap.add_argument("--ram_mask", type=str, default="",
                    help="RAM byte-index .npy, required when the checkpoint was "
                         "trained with a MASKED ram observation: the checkpoint "
                         "records the WIDTH, not which bytes.")
    a = ap.parse_args(argv)

    # Fail loudly if the sweep config would not produce a CNN-compatible obs.
    if a.num_step_frames % 4 != 0:
        raise SystemExit(f"--num_step_frames={a.num_step_frames} not divisible by 4; "
                         "build_sf_combos would raise. Legal: 4, 8, 12, 16.")
    K = len(range(0, a.num_stack, a.num_step_frames // 2))
    if K != 3:
        raise SystemExit(
            f"num_stack={a.num_stack}, num_step_frames={a.num_step_frames} gives "
            f"K={K} channels; the checkpoint's CNN needs K=3. Use ns/nsf in "
            "{6/4, 12/8, 18/12, 24/16}.")
    if not a.out:
        sampler = "random" if a.random_sampler else "policy"
        a.out = f"obs_attribution_nsf{a.num_step_frames}_{sampler}.json"

    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, infer_obs_kwargs, REPO_ROOT)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    _dt_kw = {}
    if a.decision_timing != "off":
        _dt_kw = dict(decision_timing=a.decision_timing,
                      actionable_statuses=tuple(int(x) for x in
                          a.actionable_statuses.split(",") if x.strip()),
                      max_skip_frames=a.max_skip_frames,
                      dwell_frames=a.dwell_frames)
    venv = build_lbr_venv(state, a.n_envs,
                          num_step_frames=a.num_step_frames, num_stack=a.num_stack,
                          **_dt_kw,
                          **infer_obs_kwargs(data, (getattr(a, 'ram_mask', '') or None)))
    try:
        try:
            model, _ = load_checkpoint(a.ckpt, venv, a.device)
        except RuntimeError as e:
            # This branch replaced the recurrent (LSTM) minimax-Q q_value_net with
            # the factored head, so pre-change SPAR checkpoints fail a STRICT load
            # on q_value_net keys alone. This diagnostic never touches q_value_net
            # (it uses pi for sampling only), so a non-strict load of everything
            # ELSE is correct -- but it must stay LOUD: force non-strict, then
            # assert every non-q_value_net tensor loaded EXACTLY, else re-raise.
            if "q_value_net" not in str(e):
                raise
            import torch as th
            import stable_baselines3.common.base_class as _bc
            _orig = _bc.BaseAlgorithm.set_parameters
            def _lax(self, d, exact_match=True, device="auto"):
                return _orig(self, d, exact_match=False, device=device)
            _bc.BaseAlgorithm.set_parameters = _lax
            try:
                model, _ = load_checkpoint(a.ckpt, venv, a.device)
            finally:
                _bc.BaseAlgorithm.set_parameters = _orig
            live = model.policy.state_dict()
            saved = load_from_zip_file(a.ckpt, device="cpu")[1]["policy"]
            bad = []
            for k, v in saved.items():
                if k.startswith("q_value_net"):
                    continue
                if k not in live or tuple(live[k].shape) != tuple(v.shape) \
                        or not th.equal(live[k].cpu(), v.cpu()):
                    bad.append(k)
            if bad:
                raise SystemExit(
                    f"Non-strict load left {len(bad)} NON-q_value_net tensors "
                    f"unloaded/mismatched: {bad[:8]}{' ...' if len(bad) > 8 else ''}. "
                    "pi/vf are what this diagnostic uses, so this is a hard failure.")
            print(f"[load] q_value_net (recurrent, unused here) skipped; all "
                  f"{len(saved) - sum(k.startswith('q_value_net') for k in saved)} "
                  f"pi/vf/action tensors verified byte-exact.", flush=True)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        n = venv.num_envs
        na = ops.n_actions
        rng = np.random.RandomState(0)

        per_variant = defaultdict(list)          # variant -> [distinct per state]
        ram_counts = []
        root_status = []                         # ego agent_status at each root
        obs = venv.reset()
        n_exp = int(np.ceil(a.n_states / n))
        for e in range(n_exp):
            for _ in range(a.stride if e else 5):
                if a.random_sampler:
                    ego_a = rng.randint(0, na, size=n)
                    adv_a = rng.randint(0, na, size=n)
                else:
                    ego_a = ops.sample_ego(obs, rng)
                    adv_a = ops.sample_adv(obs, rng)
                obs = venv.step(ops.joint(ego_a, adv_a))[0]

            venv.env_method("lbr_snapshot", EXPAND_ROOT)
            # ego agent_status AT THE ROOT (before any action is stepped)
            if a.by_status:
                sv = venv.env_method("lbr_state_variants")
                root_stat = [int(round(s["vals"][s["keys"].index("agent_status")]))
                             for s in sv]
            digs = {v: [[] for _ in range(n)] for v in VARIANTS}
            rams = [[] for _ in range(n)]
            for i in range(na):
                venv.env_method("lbr_restore", EXPAND_ROOT)
                venv.step(ops.joint(np.full(n, i), np.zeros(n, dtype=int)))
                vres = venv.env_method("lbr_obs_variants")
                rres = venv.env_method("lbr_fingerprint")
                for k in range(n):
                    for v in VARIANTS:
                        digs[v][k].append(vres[k][v])
                    rams[k].append(rres[k])
            venv.env_method("lbr_restore", EXPAND_ROOT)
            venv.env_method("lbr_drop", EXPAND_ROOT)

            for k in range(n):
                for v in VARIANTS:
                    per_variant[v].append(len(set(digs[v][k])))
                ram_counts.append(len(set(rams[k])))
                if a.by_status:
                    root_status.append(root_stat[k])
            print(f"   expansion {e+1}/{n_exp}", flush=True)
    finally:
        venv.close()

    ram = np.array(ram_counts)
    S = ram.size
    print("\n" + "=" * 74)
    print(f"OBSERVATION ATTRIBUTION  {os.path.basename(a.ckpt)}   {S} states")
    print(f"  distinct successors over {na} EGO actions (adv fixed); "
          f"ceiling is {na-1} (0 and 9 coincide)")
    print("=" * 74)
    print(f"  {'variant':<14} {'median':>7} {'p25':>6} {'p75':>6} {'max':>6} "
          f"{'% of RAM':>9}")
    res = {"checkpoint": os.path.basename(a.ckpt), "n_states": S,
           "num_step_frames": a.num_step_frames, "num_stack": a.num_stack,
           "sampler": "random" if a.random_sampler else "policy",
           "ram_median": float(np.median(ram))}
    print(f"  {'RAM (truth)':<14} {np.median(ram):>7.0f} "
          f"{np.percentile(ram,25):>6.0f} {np.percentile(ram,75):>6.0f} "
          f"{ram.max():>6.0f} {100.0:>8.1f}%")
    for v in VARIANTS:
        d = np.array(per_variant[v])
        pct = 100.0 * np.median(d) / max(np.median(ram), 1)
        res[v] = {"median": float(np.median(d)), "max": int(d.max()),
                  "pct_of_ram": float(pct)}
        print(f"  {v:<14} {np.median(d):>7.0f} {np.percentile(d,25):>6.0f} "
              f"{np.percentile(d,75):>6.0f} {d.max():>6.0f} {pct:>8.1f}%")

    cur = res["current"]["median"]
    best_single = max(("recent", "all_frames", "full_spatial", "all_channels"),
                      key=lambda v: res[v]["median"])
    print("\n" + "=" * 74)
    print(f"  current resolves {cur:.0f} of {np.median(ram):.0f} game-distinct "
          f"successors ({res['current']['pct_of_ram']:.1f}%)")
    print(f"  best SINGLE fix: `{best_single}` -> {res[best_single]['median']:.0f} "
          f"({res[best_single]['pct_of_ram']:.1f}%)")
    print(f"  all three removed: {res['everything']['median']:.0f} "
          f"({res['everything']['pct_of_ram']:.1f}%)")
    if res["recent"]["median"] >= 0.8 * res["everything"]["median"] and \
       res["recent"]["median"] > cur:
        res["verdict"] = "TEMPORAL PHASE -- free fix available"
        print("\n  => `recent` recovers most of it at IDENTICAL tensor shape and")
        print("     compute. The loss is WHICH frames are sampled, not how many")
        print("     or at what resolution. Changing frames[::4] to frames[3::4]")
        print("     is a one-line change that does not touch the CNN.")
    elif res["everything"]["median"] <= 2 * cur:
        res["verdict"] = "NOT THE OBSERVATION"
        print("\n  => even with every reduction removed the observation barely")
        print("     resolves more than today. A pixel observation at this frame")
        print("     rate cannot express the distinction; the input is not fixable")
        print("     by resampling alone.")
    else:
        res["verdict"] = f"DOMINATED BY {best_single}"
        print(f"\n  => `{best_single}` is the dominant loss. Removing it costs")
        print(f"     real compute (larger CNN input), unlike `recent`.")
    if a.by_status:
        st = np.array(root_status)
        cur = np.array(per_variant["current"])
        eve = np.array(per_variant["everything"])
        rm = ram
        print("\n" + "=" * 74)
        print("  ACTION-DISTINCTNESS CONDITIONED ON EGO agent_status AT THE ROOT")
        print("  hypothesis: distinctness collapses to 1 in NON-ACTIONABLE states")
        print("=" * 74)
        print(f"  {'status':>7} {'n_roots':>8} {'RAM_med':>8} {'cur_med':>8} "
              f"{'eve_med':>8} {'eve_mean':>9}")
        res["by_status"] = {}
        order = sorted(set(st.tolist()), key=lambda s: -(st == s).sum())
        for s in order:
            m = st == s
            row = {"n": int(m.sum()), "ram_med": float(np.median(rm[m])),
                   "current_med": float(np.median(cur[m])),
                   "everything_med": float(np.median(eve[m])),
                   "everything_mean": float(eve[m].mean())}
            res["by_status"][int(s)] = row
            print(f"  {s:>7} {row['n']:>8} {row['ram_med']:>8.0f} "
                  f"{row['current_med']:>8.0f} {row['everything_med']:>8.0f} "
                  f"{row['everything_mean']:>9.2f}")
        # Empirical actionable split: a status is "actionable" if its median
        # ceiling (everything) distinctness is > 2. No a-priori status semantics.
        act = np.array([np.median(eve[st == s]) > 2 for s in st])
        print("-" * 74)
        print(f"  ACTIONABLE roots (eve_med>2 status): {act.sum():>4}/{len(st)} "
              f"({100*act.mean():.0f}%)  everything median distinct = "
              f"{np.median(eve[act]) if act.any() else float('nan'):.0f}")
        print(f"  NON-ACTIONABLE roots:               {(~act).sum():>4}/{len(st)} "
              f"({100*(~act).mean():.0f}%)  everything median distinct = "
              f"{np.median(eve[~act]) if (~act).any() else float('nan'):.0f}")
        res["by_status_summary"] = {
            "actionable_frac": float(act.mean()),
            "everything_med_actionable": float(np.median(eve[act])) if act.any() else None,
            "everything_med_nonactionable": float(np.median(eve[~act])) if (~act).any() else None}

    with open(os.path.join(REPO_ROOT, a.out), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {os.path.join(REPO_ROOT, a.out)}")
    return res


if __name__ == "__main__":
    main()
