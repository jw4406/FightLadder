"""Would a minimax bootstrap change the leaf value at all?

THE DECISION THIS INFORMS. SPAR consumes only V(s) -- GAE never queries Q -- and
LBR gets its action-conditioning from the SIMULATOR (snapshot, step all 88
branches, read r + gamma*V(s')), not from a head. So an action-conditional
critic has no consumer today. The only thing it would buy is a MINIMAX
BOOTSTRAP: replacing the on-policy leaf value with the equilibrium value of the
local matrix game, on the argument that V^pi is the wrong target in a
non-stationary self-play game.

That argument is testable WITHOUT building any head, from the same 22x22
matrices payoff_structure.py already collects. Three leaf values per state:

    V_pi        = sum_ij pi_ego(i) pi_adv(j) M_ij      what is used now
    V_minimax   = value of the matrix game M           what a joint critic would
                                                       give (full solve, MWU)
    V_additive  = c + max_i alpha_i + min_j beta_j     what the SEPARABLE
                                                       44-output head could
                                                       represent; an additive
                                                       matrix has a PURE
                                                       equilibrium, so this is a
                                                       max and a min, no solver

TWO QUESTIONS, ANSWERED AT ONCE:

  |V_minimax - V_pi|        the SIZE of the bootstrap effect. If ~0, changing the
                            bootstrap is inert and NO head -- separable, bilinear
                            or full -- will do anything. Stop.
  |V_minimax - V_additive|  whether the cheap head can CAPTURE that effect. Small
                            means the 44-output head suffices; large means the
                            effect lives in the interaction term, which the
                            payoff ANOVA measured at gamma ~0.1%.

Scale for judging "small": the return std on this task is ~0.0166, and the
one-step reward scale is ~0.176.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPAND_ROOT = "boot_root"


def expand_root(venv, ops, na, n, gamma, n_paths, horizon, bootstrap, rng,
                snapshot_key=EXPAND_ROOT):
    """Enumerate all na*na joint deviations from the venv's CURRENT states and
    score each by the mean of `n_paths` on-policy Monte-Carlo rollouts.

    Single deviation at t=0 (the joint action (i,j)), then BOTH seats follow
    their policies for up to `horizon` steps, restoring to the snapshot before
    every path so the MC noise is averaged rather than inherited. Returns four
    (na, na, n) arrays in the ego frame:

      R   exact one-step deviation reward. The deviation is deterministic, so
          this is bit-exact and R[0]==R[9] (actions 0 and 9 are byte-identical).
      M   r0 + sum_t gamma^t r_t  averaged over n_paths, plus gamma^H*V(s_H) for
          still-alive envs iff `bootstrap`. This is the leaf the one-shot
          deviation gap consumes. With `bootstrap` and horizon==0 it collapses
          to the old r + gamma*V(s')*(1-done) matrix, bit-for-bit.
      SE  per-cell MC standard error of M across the n_paths (0 if n_paths==1).
      DN  done flag at the deviation step.
    """
    import numpy as np
    from local_best_response import splice_terminal

    R = np.zeros((na, na, n)); M = np.zeros((na, na, n))
    SE = np.zeros((na, na, n)); DN = np.zeros((na, na, n), bool)
    venv.env_method("lbr_snapshot", snapshot_key)
    try:
        for i in range(na):
            for j in range(na):
                gp = np.zeros((n_paths, n))
                r0 = d0 = None
                for p in range(n_paths):
                    venv.env_method("lbr_restore", snapshot_key)
                    o1, r_l, r_r, d1, inf1 = venv.step(
                        ops.joint(np.full(n, i), np.full(n, j)))     # the deviation
                    d1 = np.asarray(d1, bool)
                    rew0 = np.asarray(ops.lbr_reward(r_l, r_r), np.float64)
                    if p == 0:
                        r0, d0 = rew0.copy(), d1.copy()   # deterministic across paths
                    ret = rew0.copy()                     # r0 at discount 1
                    alive = ~d1
                    disc = gamma
                    o = splice_terminal(o1, d1, inf1)
                    t = 0
                    while alive.any() and t < horizon:     # on-policy tail
                        a_e = ops.sample_ego(o, rng); a_a = ops.sample_adv(o, rng)
                        o2, rl2, rr2, d2, inf2 = venv.step(ops.joint(a_e, a_a))
                        d2 = np.asarray(d2, bool)
                        ret += disc * ops.lbr_reward(rl2, rr2) * alive
                        alive &= ~d2
                        disc *= gamma
                        o = splice_terminal(o2, d2, inf2)
                        t += 1
                    if bootstrap:                          # gamma^(t+1)*V(s_leaf)
                        ret += disc * np.asarray(ops.values_ego(o), np.float64) * alive
                    gp[p] = ret
                R[i, j], DN[i, j] = r0, d0
                M[i, j] = gp.mean(axis=0)
                SE[i, j] = (gp.std(axis=0, ddof=1) / np.sqrt(n_paths)
                            if n_paths > 1 else 0.0)
        venv.env_method("lbr_restore", snapshot_key)
    finally:
        venv.env_method("lbr_drop", snapshot_key)
    return R, M, SE, DN


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", type=str, default="")
    ap.add_argument("--n_states", type=int, default=180)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--n_envs", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="bootstrap_delta.json")
    ap.add_argument("--reward_scale", type=float, default=0.001,
                    help="MUST match the checkpoint's training scale. An UNSCALED "
                         "head (reward_scale=1.0) enumerated in a 0.001-scale env "
                         "compares a scale-1 Q against scale-0.001 payoffs -- evW "
                         "goes to garbage (-1e6). corrW is scale-invariant and "
                         "survives, but evW/headroom need this to be right.")
    ap.add_argument("--num_step_frames", type=int, default=8,
                    help="MUST match the checkpoint. An nsf=16 policy enumerated "
                         "in an nsf=8 env is evaluated on a different game, and "
                         "nothing would raise -- the obs width is identical.")
    ap.add_argument("--save_obs", action="store_true",
                    help="Also save the ROOT observation for every state. Needed to "
                         "use the enumerated matrices as TRAINING data for the head "
                         "-- without it the matrices can only be a diagnostic. Adds "
                         "roughly obs_width*4 bytes per state (~5 MB for 600 states "
                         "of masked RAM), so it is opt-in.")
    # k-step gold standard: score each joint deviation by REALIZED discounted
    # return, averaged over n_paths on-policy Monte-Carlo rollouts, instead of
    # the one-step r + gamma*V(s') leaf. Default has NO critic anywhere.
    ap.add_argument("--n_paths", type=int, default=8,
                    help="K MC rollouts averaged per deviated action. Each is an "
                         "independent on-policy tail; the mean is the leaf and the "
                         "spread gives a per-cell MC standard error (saved as SE).")
    ap.add_argument("--horizon", type=int, default=120,
                    help="Max tail steps per path (~7x the gamma horizon, so "
                         "truncation is negligible at gamma~0.94). Rolls to episode "
                         "end or this cap, whichever first.")
    ap.add_argument("--bootstrap", action="store_true",
                    help="Add the gamma^t*V(s_t) leaf at truncation for still-alive "
                         "envs. DEFAULT OFF: no critic in the leaf, M is the pure "
                         "realized return. `--bootstrap --horizon 0` reproduces the "
                         "old r + gamma*V(s') matrix bit-for-bit.")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, splice_terminal,
                                     infer_obs_kwargs, REPO_ROOT)
    from common.minimax import solve_matrix_game

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs,
                          num_step_frames=a.num_step_frames,
                          reward_scale=a.reward_scale,
                          **infer_obs_kwargs(data, a.ram_mask or None))
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        gamma = ops.gamma
        n, na = venv.num_envs, ops.n_actions
        rng = np.random.RandomState(0)

        M_all, PE_all, PA_all, V0_all, R_all, SE_all, OBS_all = (
            [], [], [], [], [], [], [])
        obs = venv.reset()
        n_exp = int(np.ceil(a.n_states / n))
        for e in range(n_exp):
            for _ in range(a.stride if e else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]
            # Policies and V at the ROOT, before any branching.
            # OBS is what lets the enumerated matrices be used as TRAINING data
            # rather than only as a diagnostic: without the root observation
            # there is no way to ask the real head to predict M(s). Off by
            # default because it is the only field that scales with obs width.
            if a.save_obs:
                OBS_all.append(np.asarray(obs).copy())
            PE_all.append(ops.ego_probs(obs))
            PA_all.append(ops.adv_probs(obs))
            V0_all.append(ops.values_ego(obs))

            # k-step gold standard: each joint deviation scored by the mean of
            # a.n_paths on-policy MC rollouts to a.horizon. Default leaf has NO
            # critic -- M IS the realized return. R (exact one-step deviation
            # reward) is kept separately: it has no critic, so its ANOVA is the
            # part of the interaction that is unarguably real.
            R, M_mc, SE, DN = expand_root(venv, ops, na, n, gamma,
                                          a.n_paths, a.horizon, a.bootstrap, rng,
                                          snapshot_key=EXPAND_ROOT)
            M_all.append(M_mc.transpose(2, 0, 1))
            R_all.append(R.transpose(2, 0, 1))
            SE_all.append(SE.transpose(2, 0, 1))
            print(f"   expansion {e+1}/{n_exp}  (paths={a.n_paths} "
                  f"horizon={a.horizon} bootstrap={'on' if a.bootstrap else 'OFF'})",
                  flush=True)
    finally:
        venv.close()

    M = np.concatenate(M_all)                       # (S, na, na) ego-frame payoff
    PE = np.concatenate(PE_all); PA = np.concatenate(PA_all)
    V0 = np.concatenate(V0_all)
    R = np.concatenate(R_all); SE = np.concatenate(SE_all)
    S = M.shape[0]

    # LOUD action-axis check. Actions 0 and 9 are byte-identical, so the
    # DEVIATION REWARD R must match bit-for-bit on both axes; M is MC-stochastic
    # and need only agree within the averaged noise. Either failing is a bug --
    # a test that does not run is indistinguishable from one that passes.
    if M.shape[1] > 9 and M.shape[2] > 9:
        r_ax = float(max(np.abs(R[:, 0, :] - R[:, 9, :]).max(),
                         np.abs(R[:, :, 0] - R[:, :, 9]).max()))
        if r_ax != 0.0:
            raise SystemExit(f"LOUD FAIL: deviation reward not action-exact "
                             f"(R rows 0 vs 9 differ by {r_ax:.3e}) -- the joint "
                             f"action axis is wrong, every M is suspect")
        dM = np.abs(M[:, 0, :] - M[:, 9, :])
        pooled = np.sqrt(SE[:, 0, :] ** 2 + SE[:, 9, :] ** 2) + 1e-12
        zmed = float(np.median(dM / pooled))
        if a.n_paths > 1 and zmed > 6.0:
            raise SystemExit(f"LOUD FAIL: M rows 0 vs 9 disagree at median "
                             f"{zmed:.1f}sigma of MC noise -- tail machinery bug")
        print(f"  action-axis OK: R exact (0.0), M(0 vs 9) median {zmed:.2f} "
              f"sigma of MC noise")
    print(f"  M leaf: {'gamma^t*V bootstrap' if a.bootstrap else 'PURE MC (no critic)'}"
          f"   paths={a.n_paths}  horizon={a.horizon}"
          f"   mean MC SE {float(SE.mean()):.4e}")

    # V_pi: the on-policy expectation, what the bootstrap uses today.
    V_pi = np.einsum("si,sij,sj->s", PE, M, PA)

    # V_minimax: full equilibrium of each 22x22 game, via the same solver the
    # minimax head would have used.
    sol = solve_matrix_game(th.as_tensor(M, dtype=th.float32), iters=1024, eta=0.5)
    V_mm = sol.V.cpu().numpy().reshape(-1)
    gap = sol.gap.cpu().numpy().reshape(-1)

    # V_additive: what the separable 44-output head could represent. An additive
    # matrix has a PURE equilibrium, so this is a max and a min -- no solver.
    mu = M.mean(axis=(1, 2))
    alpha = M.mean(axis=2) - mu[:, None]            # (S, na) ego main effect
    beta = M.mean(axis=1) - mu[:, None]             # (S, na) adv main effect
    V_add = mu + alpha.max(axis=1) + beta.min(axis=1)

    d_boot = V_mm - V_pi
    d_head = V_mm - V_add
    sd = float(M.std())

    def stat(x):
        return (float(np.abs(x).mean()), float(np.abs(x).std()),
                float(np.percentile(np.abs(x), 90)), float(np.abs(x).max()))

    print("\n" + "=" * 76)
    print(f"BOOTSTRAP DELTA  {os.path.basename(a.ckpt)}   {S} states")
    print("=" * 76)
    print(f"  payoff std across all cells   {sd:.6f}")
    print(f"  solver duality gap (mean)     {gap.mean():.3e}  "
          f"(sanity: must be << payoff std)")
    print(f"\n  {'quantity':<34} {'mean|.|':>10} {'sd':>10} {'p90':>10} {'max':>10}")
    for nm, x in (("V_minimax - V_pi   (EFFECT SIZE)", d_boot),
                  ("V_minimax - V_additive (HEAD GAP)", d_head)):
        m, s_, p, mx = stat(x)
        print(f"  {nm:<34} {m:>10.6f} {s_:>10.6f} {p:>10.6f} {mx:>10.6f}")
    # Within-state spread: the scale on which actions differ AT a state. The
    # total std is ~95% cross-state (mu), so normalising by it understates the
    # decision-relevant size of the effect. Computed, not estimated.
    sd_within = float((M - M.mean(axis=(1, 2), keepdims=True)).std())
    rel_boot = float(np.abs(d_boot).mean() / max(sd, 1e-12))
    rel_head = float(np.abs(d_head).mean() / max(sd, 1e-12))
    relw_boot = float(np.abs(d_boot).mean() / max(sd_within, 1e-12))
    relw_head = float(np.abs(d_head).mean() / max(sd_within, 1e-12))
    print(f"\n  payoff std  total {sd:.6f}   WITHIN-state {sd_within:.6f}"
          f"   (within/total {sd_within/max(sd,1e-12):.3f})")
    print(f"  vs TOTAL  std:  bootstrap effect {rel_boot:>7.2%}   head shortfall {rel_head:>7.2%}")
    print(f"  vs WITHIN std:  bootstrap effect {relw_boot:>7.2%}   head shortfall {relw_head:>7.2%}")

    # Free check that was collected and unused last time: how far is the critic's
    # OWN V(s) from the Bellman-consistent value of its own matrices?
    d_v0 = V0.reshape(-1) - V_pi
    print(f"\n  |V_critic(s) - V_pi|  mean {np.abs(d_v0).mean():.6f}"
          f"   = {np.abs(d_v0).mean()/max(sd,1e-12):.2%} of total std"
          f"   (Bellman inconsistency of the critic itself)")

    raw = os.path.join(REPO_ROOT, a.out.replace(".json", "_raw.npz"))
    _extra = {"OBS": np.concatenate(OBS_all)} if OBS_all else {}
    np.savez_compressed(raw, M=M, PE=PE, PA=PA, V0=V0, R=R, SE=SE, **_extra)
    print(f"\n  raw matrices -> {raw}")

    res = {"checkpoint": os.path.basename(a.ckpt), "n_states": S,
           "payoff_std": sd, "duality_gap_mean": float(gap.mean()),
           "boot_effect_mean_abs": float(np.abs(d_boot).mean()),
           "head_gap_mean_abs": float(np.abs(d_head).mean()),
           "boot_effect_rel": rel_boot, "head_gap_rel": rel_head,
           "payoff_std_within": sd_within,
           "boot_effect_rel_within": relw_boot, "head_gap_rel_within": relw_head,
           "critic_bellman_gap_mean_abs": float(np.abs(d_v0).mean())}

    print("\n" + "=" * 76)
    if rel_boot < 0.05:
        res["verdict"] = "BOOTSTRAP INERT"
        print("  => the minimax bootstrap barely moves the leaf value. Changing it")
        print("     is inert, and NO action-conditional head -- separable, bilinear")
        print("     or full 484 -- can matter. V is sufficient; spend the effort on")
        print("     the dynamics instead.")
    elif rel_head < 0.25 * rel_boot:
        res["verdict"] = "EFFECT REAL, SEPARABLE HEAD SUFFICES"
        print("  => the bootstrap moves the leaf value materially, and the ADDITIVE")
        print("     form captures nearly all of it. Build the 44-output separable")
        print("     head; the 484-cell matrix and the MWU solve are unnecessary.")
    else:
        res["verdict"] = "EFFECT REAL, NEEDS THE INTERACTION"
        print("  => the bootstrap matters AND the additive form misses much of it,")
        print("     so the effect lives in the interaction term. That contradicts")
        print("     the payoff ANOVA (gamma ~0.1%) and both should be re-examined")
        print("     before building anything.")
    out = os.path.join(REPO_ROOT, a.out)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {out}")
    return res


if __name__ == "__main__":
    main()
