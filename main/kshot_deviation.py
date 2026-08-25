"""k-SHOT deviation search: can a k-step LOCAL combo defeat the frozen policy?

The one-shot deviation gap (bootstrap_delta.expand_root -> local_nashconv) is
blind to COMBOS: no single move beats a solid defense, but a 3-hit link can. A
combo's value lives in CONTINUING through states pi never visits, where the
one-shot gap is never measured. This tool closes that blind spot.

WHAT IT MEASURES. For the exploiter seat (frozen opponent), at each on-policy
root s it searches over exploiter action sequences of length UP TO k. Each
sequence is scored by the realized MC return of:

    play the combo,  THEN revert to pi (both seats),  roll to the horizon.

    gap_<=k(s) = max over combos of length j<=k  E[ return(combo, then pi) ] - V^pi(s)

VARIABLE LENGTH (the "revert early" option) is encoded with a PI sentinel: a
length-j combo is padded to length k with PI, and PI positions SAMPLE pi. So the
length-1..k combos are pooled into ONE candidate set, gap_<=k is MONOTONE in k
by construction, and -- crucially -- a SINGLE winner's-curse reeval de-biases it:
select the best candidate (any length) on one seed set, RE-EVALUATE the winner
on FRESH seeds. Taking a per-state max over separate per-k runs would be a second
winner's curse (max of noisy estimates); this avoids it.

The FROZEN OPPONENT is SAMPLED from its policy throughout (matches the V^pi
baseline; no argmax confound). NO critic anywhere. Common random numbers: every
rollout draws u_lbr then u_opp each step (even when u_lbr is unused) so streams
align, and the gap is the PER-PATH DIFFERENCE combo_p - baseline_p at a shared
seed. The baseline is the all-PI candidate, rolled the SAME k+H steps.

PRUNED (--topM): a cheap pass-1 heuristic (in-window combo reward, argmax
opponent, deterministic) ranks all pooled candidates; each root's top-M are
unioned and only those get the full CRN MC. gap becomes a LOWER BOUND; the
pruning ratio and winners' heuristic ranks are printed.

LOUD: full search asserts the whole pooled set ran; the CRN gap must not be
many-sigma negative (a combo can mimic pi). Also reports the WINNER COMBO-LENGTH
distribution -- if winners are length 1, longer combos add nothing.
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = "kshot_root"


def _sample_u(probs, u):
    import numpy as np
    c = np.cumsum(probs, axis=1)
    return (u[:, None] < c).argmax(axis=1)


def _pooled_candidates(na, k):
    """All combos of length 1..k, PI-padded to length k. PI == na."""
    PI = na
    cands = []
    for j in range(1, k + 1):
        for combo in itertools.product(range(na), repeat=j):
            cands.append(tuple(combo) + (PI,) * (k - j))
    return cands, PI


def _combo_len(seq_col, PI):
    """True combo length of a padded column = position of first PI (or k)."""
    import numpy as np
    k = len(seq_col)
    for t in range(k):
        if seq_col[t] == PI:
            return t
    return k


def _rollout(venv, ops, gamma, horizon, seq, obs0, seed, PI):
    """One CRN rollout from ROOT (caller restored ROOT). Rolls k + horizon steps.

    Fixed draw protocol: u_lbr then u_opp every step. Window (t < k): exploiter
    plays seq[t], EXCEPT PI positions sample pi via u_lbr (the revert-early
    option). After the window: exploiter samples pi. Opponent always samples its
    frozen pi via u_opp. seq[t] may be an int or an (n,) array; PI == na.
    """
    import numpy as np
    from local_best_response import splice_terminal
    rng = np.random.RandomState(seed)
    n = venv.num_envs
    ret = np.zeros(n); alive = np.ones(n, bool); disc = 1.0
    o = obs0
    k = len(seq)
    adv = ops.lbr_is_adv
    t = 0
    while alive.any() and t < k + horizon:
        u_lbr = rng.random_sample(n); u_opp = rng.random_sample(n)
        opp = _sample_u(ops.ego_probs(o) if adv else ops.adv_probs(o), u_opp)
        lbr_pi = _sample_u(ops.adv_probs(o) if adv else ops.ego_probs(o), u_lbr)
        if t < k:
            s = seq[t]
            s_arr = np.full(n, s) if np.isscalar(s) else np.asarray(s)
            lbr = np.where(s_arr == PI, lbr_pi, s_arr)
        else:
            lbr = lbr_pi
        o2, rl, rr, d, inf = venv.step(ops.joint(lbr, opp))
        d = np.asarray(d, bool)
        ret += disc * ops.lbr_reward(rl, rr) * alive
        alive &= ~d; disc *= gamma
        o = splice_terminal(o2, d, inf)
        t += 1
    return ret


def _window_reward(venv, ops, gamma, seq, obs0, PI):
    """pass-1 heuristic: in-window discounted reward, argmax opponent, PI
    positions play the exploiter's argmax; deterministic, NO tail. (n,)."""
    import numpy as np
    from local_best_response import splice_terminal
    n = venv.num_envs
    ret = np.zeros(n); alive = np.ones(n, bool); disc = 1.0
    o = obs0
    adv = ops.lbr_is_adv
    for t in range(len(seq)):
        opp = (ops.ego_probs(o) if adv else ops.adv_probs(o)).argmax(axis=1)
        if seq[t] == PI:
            lbr = (ops.adv_probs(o) if adv else ops.ego_probs(o)).argmax(axis=1)
        else:
            lbr = np.full(n, seq[t])
        o2, rl, rr, d, inf = venv.step(ops.joint(lbr, opp))
        d = np.asarray(d, bool)
        ret += disc * ops.lbr_reward(rl, rr) * alive
        alive &= ~d; disc *= gamma
        o = splice_terminal(o2, d, inf)
    return ret


def _crn_select(venv, ops, gamma, horizon, candidates, n_paths, obs, sel_seed, PI):
    import numpy as np
    n = venv.num_envs
    best = np.full(n, -np.inf)
    k = len(candidates[0]) if candidates else 0
    best_seq = np.full((k, n), PI, dtype=np.int64)
    counted = 0
    for seq in candidates:
        vals = np.zeros((n_paths, n))
        for p in range(n_paths):
            venv.env_method("lbr_restore", ROOT)
            vals[p] = _rollout(venv, ops, gamma, horizon, list(seq), obs, sel_seed + p, PI)
        m = vals.mean(axis=0)
        better = m > best
        best = np.where(better, m, best)
        for t in range(k):
            best_seq[t] = np.where(better, seq[t], best_seq[t])
        counted += 1
    return best, best_seq, counted


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="")
    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--num_step_frames", type=int, default=8)
    ap.add_argument("--exploiter", choices=["ego", "adv"], default="adv")
    ap.add_argument("--k", type=int, default=2, help="MAX combo length (pools 1..k)")
    ap.add_argument("--topM", type=int, default=0,
                    help="0 = full pooled enumeration. >0 = keep each root's top-M by "
                         "the pass-1 heuristic (gap becomes a LOWER BOUND).")
    ap.add_argument("--sel_paths", type=int, default=8, help="MC paths for SELECTION")
    ap.add_argument("--reeval_paths", type=int, default=32, help="MC paths for the unbiased REEVAL")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--n_states", type=int, default=24)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--n_envs", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, infer_obs_kwargs)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, _, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs, num_step_frames=a.num_step_frames,
                          reward_scale=a.reward_scale,
                          **infer_obs_kwargs(data, a.ram_mask or None))
    GAP, GAP_SE, SEL, BIAS, VPI, WLEN, CANDN, RANKS = [], [], [], [], [], [], [], []
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=(a.exploiter == "adv"))
        gamma = ops.gamma
        n, na = venv.num_envs, ops.n_actions
        rng = np.random.RandomState(0)
        all_cands, PI = _pooled_candidates(na, a.k)
        pool = len(all_cands)
        pruning = a.topM > 0 and pool > a.topM
        print(f"[kshot] exploiter={a.exploiter} k<={a.k} na={na} -> {pool} pooled "
              f"candidates (lengths 1..{a.k}) | sel_paths={a.sel_paths} "
              f"reeval_paths={a.reeval_paths} horizon={a.horizon} gamma={gamma} | "
              f"{'PRUNED topM=' + str(a.topM) if pruning else 'FULL'}")

        obs = venv.reset()
        n_batches = int(np.ceil(a.n_states / n))
        for b in range(n_batches):
            for _ in range(a.stride if b else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]
            venv.env_method("lbr_snapshot", ROOT)
            sel_seed = 1000 + b * 100000
            re_seed = 50000 + b * 100000

            scores = None
            if pruning:
                scores = np.zeros((pool, n))
                for si, seq in enumerate(all_cands):
                    venv.env_method("lbr_restore", ROOT)
                    scores[si] = _window_reward(venv, ops, gamma, seq, obs, PI)
                top_idx = np.argsort(-scores, axis=0)[:a.topM, :]
                cand_idx = np.unique(top_idx)
                candidates = [all_cands[i] for i in cand_idx]
            else:
                candidates = all_cands

            best, best_seq, counted = _crn_select(venv, ops, gamma, a.horizon,
                                                  candidates, a.sel_paths, obs, sel_seed, PI)
            if not pruning and counted != pool:
                raise SystemExit(f"LOUD FAIL: ran {counted} != pooled {pool} -- "
                                 f"silently under-searched, gap is not a max")

            if pruning:
                # winners' heuristic rank (how deep in top-M the winner fell)
                win_code = np.zeros(n, dtype=np.int64)
                base = na + 1  # alphabet includes PI
                for t in range(a.k):
                    win_code = win_code * base + best_seq[t]
                code_of = {}
                for i, seq in enumerate(all_cands):
                    c = 0
                    for t in range(a.k):
                        c = c * base + seq[t]
                    code_of[c] = i
                for r in range(n):
                    wi = code_of.get(int(win_code[r]), None)
                    if wi is not None:
                        RANKS.append(int((scores[:, r] > scores[wi, r]).sum()))
                CANDN.append(len(candidates))

            # REEVAL winner AND all-PI baseline on the SAME fresh seeds -> CRN gap.
            base_seq = [np.full(n, PI) for _ in range(a.k)]
            win_seq = [best_seq[t] for t in range(a.k)]
            combo_r = np.zeros((a.reeval_paths, n)); base_r = np.zeros((a.reeval_paths, n))
            for p in range(a.reeval_paths):
                venv.env_method("lbr_restore", ROOT)
                combo_r[p] = _rollout(venv, ops, gamma, a.horizon, win_seq, obs, re_seed + p, PI)
                venv.env_method("lbr_restore", ROOT)
                base_r[p] = _rollout(venv, ops, gamma, a.horizon, base_seq, obs, re_seed + p, PI)
            venv.env_method("lbr_restore", ROOT)
            venv.env_method("lbr_drop", ROOT)

            vpi = base_r.mean(axis=0); combo_mean = combo_r.mean(axis=0)
            gap_paths = combo_r - base_r
            gap = gap_paths.mean(axis=0)
            gap_se = (gap_paths.std(axis=0, ddof=1) / np.sqrt(a.reeval_paths)
                      if a.reeval_paths > 1 else np.zeros(n))
            for r in range(n):
                WLEN.append(_combo_len(best_seq[:, r], PI))
            GAP.append(gap); GAP_SE.append(gap_se)
            SEL.append(best - vpi); BIAS.append(best - combo_mean); VPI.append(vpi)
            print(f"   batch {b+1}/{n_batches}  gap<=k {gap.mean():+.4f} +/- {gap_se.mean():.4f}"
                  f"  winners-curse {(best-combo_mean).mean():+.4f}"
                  f"  win-len {np.mean(WLEN[-n:]):.2f}", flush=True)
    finally:
        venv.close()

    gap = np.concatenate(GAP); gap_se = np.concatenate(GAP_SE)
    sel = np.concatenate(SEL); bias = np.concatenate(BIAS); vpi = np.concatenate(VPI)
    wlen = np.array(WLEN); S = gap.shape[0]
    zmin = float((gap / (gap_se + 1e-12)).min())
    if zmin < -6.0:
        raise SystemExit(f"LOUD FAIL: min gap z {zmin:.1f} << 0 -- impossible (combo mimics pi)")
    tag = f"PRUNED topM={a.topM}" if (a.topM > 0 and len(all_cands) > a.topM) else "FULL"
    print("\n" + "=" * 76)
    print(f"k-SHOT DEVIATION GAP (CRN, unbiased, {tag})   {os.path.basename(a.ckpt)}   "
          f"exploiter={a.exploiter}   k<={a.k}   {S} states")
    print("=" * 76)
    print(f"  V^pi                        mean {vpi.mean():+.5f}")
    print(f"  gap_<=k (REEVAL, unbiased)  mean {gap.mean():+.5f}  median {np.median(gap):+.5f}"
          f"  meanSE {gap_se.mean():.4f}")
    print(f"  gap_<=k (SELECTION, biased) mean {sel.mean():+.5f}   [do NOT quote]")
    print(f"  winner's-curse bias         mean {bias.mean():+.5f}")
    print(f"  fraction of states gap>2SE:  {float((gap > 2*gap_se).mean()):.1%}")
    print(f"  WINNER COMBO-LENGTH  mean {wlen.mean():.2f}  "
          + "  ".join(f"len{j}={int((wlen==j).sum())}" for j in range(1, a.k + 1)))
    print(f"    (winners of length>1 = states where a genuine MULTI-STEP combo wins)")
    multi = wlen > 1
    if multi.any():
        print(f"    among length>1 winners: gap {gap[multi].mean():+.4f} "
              f"+/- {gap_se[multi].mean():.4f}  ({int(multi.sum())} states)")
    # THE clean validator: a genuine multi-step combo is a SIGNIFICANT win that is
    # ALSO multi-step. Length alone is confounded by the na^2 >> na candidate
    # count imbalance (a len-2 wins the max by noise far more often).
    real = (gap > 2 * gap_se) & (wlen > 1)
    print(f"  REAL multi-step combos (gap>2SE AND len>1): {int(real.sum())}/{S} states"
          + (f"   mean gap {gap[real].mean():+.4f}" if real.any() else "  -> none"))
    if RANKS:
        rk = np.array(RANKS)
        print(f"  PRUNING: cand/state {int(np.mean(CANDN))}/{len(all_cands)} "
              f"({np.mean(CANDN)/len(all_cands):.1%}); winner rank med {int(np.median(rk))}/topM={a.topM}")
        if np.median(rk) > 0.8 * a.topM:
            print(f"  !! winners near rank M -- topM too small, gap under-reports")
    if a.out:
        np.savez_compressed(os.path.join(REPO_ROOT, a.out), gap=gap, gap_se=gap_se,
                            sel_gap=sel, bias=bias, vpi=vpi, wlen=wlen, k=a.k,
                            exploiter=a.exploiter, topM=a.topM)
        print(f"  saved -> {a.out}")
    return {"k": a.k, "gap_mean": float(gap.mean()), "gap_median": float(np.median(gap)),
            "gap_se_mean": float(gap_se.mean()), "bias_mean": float(bias.mean()),
            "vpi_mean": float(vpi.mean()), "win_len_mean": float(wlen.mean()),
            "n_states": S}


if __name__ == "__main__":
    main()
