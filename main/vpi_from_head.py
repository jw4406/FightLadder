"""Would V_pi computed from the joint head beat the scalar critic as a GAE leaf?

THE IDEA. Enumeration gives an accurate Q(s,i,j). The on-policy leaf value is
then an ANALYTIC expectation rather than a learned regression:

    V_pi(s) = sum_i sum_j  pi_ego(i) pi_adv(j) Q(s,i,j)

This changes NO fixed point -- V_pi is what the scalar critic already estimates,
so GAE with it is still ordinary actor-critic. The only possible gain is variance
reduction: an exact expectation over known action distributions instead of a
value fit to noisy returns.

So the question is purely empirical: is V_scalar currently a BAD estimate of
V_pi? If it is already accurate, this buys nothing and costs an enumeration
budget. If it is poor, the head is a strictly better leaf with no minimax
operator, no equilibrium solve, and no self-referential bootstrap.

GROUND TRUTH is the enumerated matrix under the actual policy probabilities:
    V_pi_true = einsum(PE, M, PA)
which bootstrap_delta already computes and saves (PE, PA, M, V0).

Also reported: V_br, the ONE-STEP IMPROVEMENT where the ego best-responds to the
adversary's current distribution. It sits strictly between V_pi and V_minimax and
needs no equilibrium solve -- and greedy one-step lookahead is the only LBR
variant that has produced a non-vacuous bound on this project.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--npz_glob", required=True)
    ap.add_argument("--ram_mask", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     resolve_matchups, infer_obs_kwargs)

    def ev(pred, tgt):
        return 1.0 - float(((pred - tgt) ** 2).mean()) / max(float(tgt.var()), 1e-30)

    print(f"{'ckpt':>9} {'n':>5} {'EV V_scalar':>12} {'EV V_pi(head)':>14} "
          f"{'corr scalar':>12} {'corr head':>10} | {'V_br-V_pi':>10} {'V_mm-V_pi':>10}")
    for f in sorted(glob.glob(a.npz_glob)):
        steps = int(re.search(r"_(\d+)_raw", f).group(1))
        ck = os.path.join(a.run_dir, f"spar_Ry_Sa_{steps}_steps.task")
        if not os.path.exists(ck):
            continue
        d = np.load(f)
        if "OBS" not in d:
            continue
        OBS = d["OBS"]; M = d["M"].astype(np.float64)
        PE = d["PE"].astype(np.float64); PA = d["PA"].astype(np.float64)
        V0 = d["V0"].astype(np.float64).reshape(-1)          # the scalar critic

        # GROUND TRUTH one-step on-policy value, from the enumerated matrix.
        V_pi = np.einsum("si,sij,sj->s", PE, M, PA)
        # one-step improvement: ego best-responds to the adversary's CURRENT mix
        V_br = np.einsum("sij,sj->si", M, PA).max(axis=1)

        data = load_from_zip_file(ck, device="cpu")[0]
        hi, _, st = resolve_matchups(data, "all")[0]
        venv = build_lbr_venv(st, 2, **infer_obs_kwargs(data, a.ram_mask or None))
        try:
            m, _ = load_checkpoint(ck, venv, a.device)
            preflight(venv, m)
            with th.no_grad():
                Q = m.policy.minimax_matrices(
                    th.as_tensor(OBS, dtype=th.float32, device=a.device),
                    buf_num=[hi]).cpu().numpy().astype(np.float64)
                from common.minimax import solve_matrix_game
                V_mm = solve_matrix_game(
                    th.as_tensor(M, dtype=th.float32, device=a.device),
                    iters=1024, eta=0.5).V.reshape(-1).cpu().numpy()
        finally:
            venv.close()

        V_pi_head = np.einsum("si,sij,sj->s", PE, Q, PA)
        print(f"{steps:9,} {len(V_pi):5d} {ev(V0, V_pi):12.3f} {ev(V_pi_head, V_pi):14.3f} "
              f"{np.corrcoef(V0, V_pi)[0,1]:12.3f} {np.corrcoef(V_pi_head, V_pi)[0,1]:10.3f} | "
              f"{np.abs(V_br - V_pi).mean():10.5f} {np.abs(V_mm - V_pi).mean():10.5f}")

    print("\n  EV/corr are against V_pi computed from the ENUMERATED matrix (truth).")
    print("  V_scalar already accurate  -> V_pi(head) buys nothing.")
    print("  V_br-V_pi and V_mm-V_pi show how far the one-step-improvement and")
    print("  equilibrium leaves move the bootstrap AT ALL -- if ~0, no operator matters.")


if __name__ == "__main__":
    main()
