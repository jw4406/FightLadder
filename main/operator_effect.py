"""How much does the minimax operator move the leaf value AT ALL?

THE QUESTION UNDERNEATH THE WHOLE PROGRAM. kappa>0 replaces the on-policy leaf
V_pi(s) with V_minimax(s) -- the equilibrium value of the 22x22 matrix. If those
two numbers barely differ, then NO head, however good, changes the bootstrap,
and the entire joint-critic direction is inert regardless of corrW(R).

An earlier reading gave |V_mm - V_pi| = 0.00138 against a within-state std of
~0.008, but that was on the frozen-ego timer-out regime where returns are
near-zero and everything is small. This measures it where episodes RESOLVE.

THREE LEAVES, all computed from the SAME enumerated matrix so the comparison is
about the OPERATOR, not about head quality:

    V_pi  = sum_ij pi_ego(i) M_ij pi_adv(j)      what SPAR uses now
    V_br  = max_i sum_j pi_adv(j) M_ij           ego best-responds one step
    V_mm  = equilibrium value of M               Littman's operator

Scale references, because a difference is meaningless without one:
  * within-state std of the matrix -- the spread the operator selects over
  * std of V_pi ACROSS states -- what the critic already has to resolve
  * (1-gamma) * delta -- the per-step advantage a changed leaf actually confers

CEILING, NOT ESTIMATE. M here is the TRUE enumerated matrix, so these are the
differences a PERFECT head would produce. A real head with headroom +0.19 gets
some fraction of it.
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
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--gamma", type=float, default=0.94)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from common.minimax import solve_matrix_game

    print(f"{'collection':>26} {'states':>6} {'|Vmm-Vpi|':>10} {'|Vbr-Vpi|':>10} "
          f"{'within std':>11} {'std(Vpi)':>9} {'mm/withn':>9} {'mm/Vpi':>7}")
    for f in a.npz:
        d = np.load(f)
        M = d["M"].astype(np.float64)
        PE = d["PE"].astype(np.float64); PA = d["PA"].astype(np.float64)

        V_pi = np.einsum("si,sij,sj->s", PE, M, PA)
        V_br = np.einsum("sij,sj->si", M, PA).max(axis=1)
        sol = solve_matrix_game(th.as_tensor(M, dtype=th.float32, device=a.device),
                                iters=1024, eta=0.5)
        V_mm = sol.V.reshape(-1).cpu().numpy()

        W = M - M.mean(axis=(1, 2), keepdims=True)
        wstd = float(W.std())
        dmm = float(np.abs(V_mm - V_pi).mean())
        dbr = float(np.abs(V_br - V_pi).mean())
        spi = float(V_pi.std())
        print(f"{os.path.basename(f)[:26]:>26} {len(M):6d} {dmm:10.5f} {dbr:10.5f} "
              f"{wstd:11.5f} {spi:9.5f} {dmm/max(wstd,1e-30):9.2f} "
              f"{dmm/max(spi,1e-30):7.2f}")

    print(f"\n  mm/withn : leaf shift as a fraction of the matrix spread the")
    print(f"             operator is selecting over -- if <<1 the max-min is")
    print(f"             picking something barely different from the mean.")
    print(f"  mm/Vpi   : leaf shift against the ACROSS-state variation the critic")
    print(f"             already resolves. <<1 means the change is in the noise.")
    print(f"  per-step advantage ~ (1-gamma)*delta = {1-a.gamma:.2f} * delta")
    print(f"  M is the TRUE enumerated matrix, so these are what a PERFECT head")
    print(f"  would deliver. Real headroom is +0.19 at best.")


if __name__ == "__main__":
    main()
