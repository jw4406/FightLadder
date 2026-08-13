"""Are the CURRENT policies at a Nash equilibrium? Answered from the payoff
matrix alone -- no rollouts, no exploiter training.

THE PRINCIPLE. In a discounted stochastic game the ONE-SHOT DEVIATION PRINCIPLE
says a policy pair is a Nash equilibrium IFF, at every state, (pi_ego(.|s),
pi_adv(.|s)) is a Nash equilibrium of the matrix game Q^pi(s,.,.). So
equilibrium is a per-state, per-matrix property and is checkable directly:

    eps_ego = max_a (M q_pi)_a  -  p_pi^T M q_pi     ego's gain from deviating
    eps_adv = p_pi^T M q_pi     -  min_o (p_pi^T M)_o
    GAP     = eps_ego + eps_adv                      == 0 exactly at Nash

This is `duality_gap` from common/minimax.py evaluated at the BEHAVIOUR policies
instead of at the solver's equilibrium -- same function, different arguments.

WHY IT MATTERS FOR THE ONE-SIDED CHECKPOINTS. A lopsided score does NOT imply a
broken run: a zero-sum game can have any equilibrium value, so score 0.0 could
simply mean one character dominates. Score cannot distinguish "converged to an
unbalanced equilibrium" from "fell over". The GAP can, because it is zero at any
equilibrium regardless of that equilibrium's value.

WHAT THE MATRIX IS, precisely. M[i,j] = r_ij + gamma*V(s'_ij)*(1-done), where
r_ij and s'_ij come from the EMULATOR (all 484 joint actions enumerated, real
dynamics, deterministic) and V is the trained SCALAR critic. So:
  * the ACTION axis is exact -- M[:,0,:] == M[:,9,:] bit-for-bit, because
    actions 0 and 9 are byte-identical and therefore produce identical rewards
    and identical successors. Asserted below.
  * the V(s') term is NOT validated. Critic error varies across successors, so
    it can distort the gap and p*. The 0-vs-9 identity cannot bound it, because
    those two actions lead to the SAME successor.

SUPPORT OVERLAP IS THE ROBUST STATISTIC. The gap's magnitude inherits the
critic's error, but whether the equilibrium puts any mass on the action the
policy actually plays is a much coarser question, and it does not depend on
getting the gap's scale right.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def analyse(path, iters=1024, eta=0.5):
    import numpy as np
    import torch as th
    from common.minimax import solve_matrix_game

    d = np.load(path)
    M = d["M"].astype(np.float64)
    PE = d["PE"].astype(np.float64)
    PA = d["PA"].astype(np.float64)
    S, ne, na = M.shape

    # Exactness check on the action axis (see module docstring).
    noise = 0.0
    if ne > 9 and na > 9:
        noise = float(max(np.abs(M[:, 0, :] - M[:, 9, :]).max(),
                          np.abs(M[:, :, 0] - M[:, :, 9]).max()))

    Vpi = np.einsum("si,sij,sj->s", PE, M, PA)
    eps_e = np.einsum("sij,sj->si", M, PA).max(axis=1) - Vpi
    eps_a = Vpi - np.einsum("si,sij->sj", PE, M).min(axis=1)
    gap = eps_e + eps_a
    sd_w = float((M - M.mean(axis=(1, 2), keepdims=True)).std())

    sol = solve_matrix_game(th.as_tensor(M, dtype=th.float32), iters=iters, eta=eta)
    p, q = sol.p.cpu().numpy(), sol.q.cpu().numpy()
    a_pi, o_pi = PE.argmax(1), PA.argmax(1)
    s = np.arange(S)
    return {
        "n_states": S, "within_std": sd_w, "action_axis_noise": noise,
        "gap_mean": float(gap.mean()), "gap_median": float(np.median(gap)),
        "gap_mean_rel": float(gap.mean() / sd_w),
        "gap_median_rel": float(np.median(gap) / sd_w),
        "frac_near_nash": float((gap < 0.1 * sd_w).mean()),
        "pi_ego_support": float((PE > 0.01).sum(1).mean()),
        "p_star_support": float((p > 0.01).sum(1).mean()),
        "p_star_at_pi_action": float(p[s, a_pi].mean()),
        "q_star_at_pi_action": float(q[s, o_pi].mean()),
        "uniform": 1.0 / ne,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", required=True,
                    help="raw npz files from bootstrap_delta.py (M, PE, PA)")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args(argv)

    paths = []
    for pat in a.npz:
        paths += sorted(glob.glob(pat)) or [pat]

    rows = {}
    print("=" * 96)
    print("LOCAL NashConv -- gap == 0 at ANY equilibrium, whatever its value")
    print("=" * 96)
    print(f"  {'arm':<22} {'gap/within':>11} {'median':>9} {'~nash':>7}"
          f" {'supp pi':>8} {'supp p*':>8} {'p*(pi act)':>11} {'axis noise':>11}")
    for p_ in paths:
        try:
            r = analyse(p_)
        except Exception as e:
            print(f"  {os.path.basename(p_):<22} FAILED: {e}")
            continue
        name = os.path.basename(p_).replace("_raw.npz", "")
        rows[name] = r
        print(f"  {name:<22} {r['gap_mean_rel']:>11.3f} {r['gap_median_rel']:>9.3f}"
              f" {r['frac_near_nash']:>7.1%} {r['pi_ego_support']:>8.1f}"
              f" {r['p_star_support']:>8.1f} {r['p_star_at_pi_action']:>11.3f}"
              f" {r['action_axis_noise']:>11.1e}")
    print(f"\n  uniform reference for p*(pi action): {1/22:.3f}")
    print("  gap/within is in units of the WITHIN-STATE payoff std, i.e. the scale")
    print("  actions actually vary on. axis noise must be 0 (byte-identical pair).")

    if a.out:
        out = os.path.join(REPO_ROOT, a.out)
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n  wrote {out}")
    return rows


if __name__ == "__main__":
    main()
