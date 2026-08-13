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
                          **infer_obs_kwargs(data, a.ram_mask or None))
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        gamma = ops.gamma
        n, na = venv.num_envs, ops.n_actions
        rng = np.random.RandomState(0)

        M_all, PE_all, PA_all, V0_all, R_all = [], [], [], [], []
        obs = venv.reset()
        n_exp = int(np.ceil(a.n_states / n))
        for e in range(n_exp):
            for _ in range(a.stride if e else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]
            # Policies and V at the ROOT, before any branching.
            PE_all.append(ops.ego_probs(obs))
            PA_all.append(ops.adv_probs(obs))
            V0_all.append(ops.values_ego(obs))

            venv.env_method("lbr_snapshot", EXPAND_ROOT)
            R = np.zeros((na, na, n)); V1 = np.zeros((na, na, n))
            DN = np.zeros((na, na, n), bool)
            for i in range(na):
                succ = []
                for j in range(na):
                    venv.env_method("lbr_restore", EXPAND_ROOT)
                    o1, r_l, r_r, d, infos = venv.step(
                        ops.joint(np.full(n, i), np.full(n, j)))
                    d = np.asarray(d, bool)
                    R[i, j] = ops.lbr_reward(r_l, r_r)
                    DN[i, j] = d
                    succ.append(splice_terminal(o1, d, infos))
                V1[i] = ops.values_ego(np.concatenate(succ, axis=0)).reshape(na, n)
            venv.env_method("lbr_restore", EXPAND_ROOT)
            venv.env_method("lbr_drop", EXPAND_ROOT)
            M_all.append((R + gamma * V1 * (~DN)).transpose(2, 0, 1))
            # R SAVED SEPARATELY. M mixes the exact emulator reward with
            # gamma*V_scalar(s'), and V is a TRAINED critic -- so an interaction
            # measured on M cannot be told apart from critic error that happens
            # to vary across the 484 successors. R alone has NO critic in it, so
            # its ANOVA is the part of the interaction that is unarguably real.
            R_all.append(R.transpose(2, 0, 1))
            print(f"   expansion {e+1}/{n_exp}", flush=True)
    finally:
        venv.close()

    M = np.concatenate(M_all)                       # (S, na, na) ego-frame payoff
    PE = np.concatenate(PE_all); PA = np.concatenate(PA_all)
    V0 = np.concatenate(V0_all)
    S = M.shape[0]

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
    np.savez_compressed(raw, M=M, PE=PE, PA=PA, V0=V0,
                        R=np.concatenate(R_all))
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
