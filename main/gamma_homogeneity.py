"""Is corrW(R) hard to earn, or does predicting the MEAN gamma matrix suffice?

THE MISSING BASELINE. corrW(R) is pooled over every cell of every state. If the
states in a regime all have SIMILAR gamma matrices, a head that outputs one
fixed matrix scores well without knowing anything state-specific. That is a
state-INDEPENDENT predictor, and it is not what a joint-action critic is for --
the minimax operator reads one state's matrix at a time, so a head that ignores
the state cannot rank branches at that state.

I have never measured this, and it is the obvious candidate artifact for the one
regime that "worked": engaged is 517-step timer-out stalemates, where both
fighters stand in range doing very little, so its states may be near-identical.
Every failing arm has resolving 137-step episodes, which are heterogeneous.

Reported per collection:
  mean_gamma_corr   average pairwise correlation between states' gamma matrices.
                    High => the regime is homogeneous and the task is easy.
  const_corrW       what the MEAN gamma matrix scores as a predictor. This is
                    the floor corrW(R) must beat to mean anything.
  headroom          the head's own corrW(R) minus const_corrW.
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
    a = ap.parse_args(argv)

    import numpy as np
    from gamma_basis import gammas

    print(f"{'collection':>26} {'states':>7} {'contact':>8} "
          f"{'mean pairwise corr':>19} {'const corrW(R)':>15}")
    for f in a.npz:
        d = np.load(f)
        R = d["R"].astype(np.float64)
        G = gammas(R)
        n = np.linalg.norm(G.reshape(len(G), -1), axis=1)
        m = n > 1e-12
        if m.sum() < 3:
            print(f"{os.path.basename(f)[:26]:>26}  too few contact states")
            continue
        Gc = G[m].reshape(int(m.sum()), -1)
        # pairwise correlation between states' interaction matrices
        Z = Gc / np.linalg.norm(Gc, axis=1, keepdims=True)
        C = Z @ Z.T
        iu = np.triu_indices(len(C), k=1)
        pair = float(C[iu].mean())

        # the CONSTANT predictor: every state gets the mean gamma matrix.
        # Scored exactly as head_quality scores the head -- state-centred,
        # pooled over all cells of all states.
        Rw = R - R.mean(axis=(1, 2), keepdims=True)
        mean_g = G[m].mean(axis=0)
        P = np.broadcast_to(mean_g, R.shape)
        Pw = P - P.mean(axis=(1, 2), keepdims=True)
        const = float(np.corrcoef(Pw.ravel(), Rw.ravel())[0, 1])
        print(f"{os.path.basename(f)[:26]:>26} {len(R):7d} {m.mean():8.1%} "
              f"{pair:19.3f} {const:15.3f}")

    print("\n  A head only demonstrates state-CONDITIONAL knowledge by beating")
    print("  the const column. If const is already high, the regime is")
    print("  homogeneous and corrW(R) is cheap there.")


if __name__ == "__main__":
    main()
