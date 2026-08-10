"""Is the payoff JOINT, or merely additive? The decomposition that decides
whether a minimax critic is needed at all.

Rank alone does not answer the question. A rank-1 matrix u v^T is highly
action-conditional; a rank-1 matrix c * 11^T carries nothing. Both read
"per-state rank 1". So decompose each payoff matrix the standard two-way way:

    M(s)_ij = mu(s) + alpha_i(s) + beta_j(s) + gamma_ij(s)

    mu     grand mean          -> pure STATE value. This is V(s). No action info.
    alpha  ego main effect     -> "this ego action is good here", regardless of
                                  what the opponent does
    beta   adversary main eff. -> same for the opponent
    gamma  INTERACTION         -> the payoff depends on the COMBINATION. THIS is
                                  the only term that requires a joint-action
                                  critic; everything else is representable by
                                  two independent action-value heads.

READING IT:
  mu dominates                 -> Q(s,i,j) = V(s). Joint-action direction dead.
  mu + alpha/beta dominate     -> actions matter but SEPARABLY. A cheap
                                  Q_ego(s,a) + Q_adv(s,o) head suffices; the
                                  484-cell matrix and the minimax solve are
                                  unnecessary machinery.
  gamma is material            -> genuinely joint. minimax-Q is the right object
                                  and the effort is justified.

Also reports how much of the leading singular direction is the ALL-ONES
direction, which separates "rank 1 and trivial" from "rank 1 and meaningful".

Reads the raw .npz written by payoff_structure.py -- no re-collection.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="*_raw.npz from payoff_structure.py")
    a = ap.parse_args(argv)

    import numpy as np
    z = np.load(a.raw)
    R, Q, H = z["R"], z["Q"], z["H"]
    S, na, _ = Q.shape
    distinct = np.array([len(np.unique(H[s])) for s in range(S)])
    free = distinct > 1

    def anova(M, name):
        if M.shape[0] < 20:
            print(f"\n  [{name}] only {M.shape[0]} states -- skipped")
            return
        mu = M.mean(axis=(1, 2), keepdims=True)                 # (S,1,1)
        alpha = M.mean(axis=2, keepdims=True) - mu              # (S,na,1)
        beta = M.mean(axis=1, keepdims=True) - mu               # (S,1,na)
        gamma = M - mu - alpha - beta                           # (S,na,na)

        # Total variance of the payoff ACROSS everything (states and cells).
        tot = float(((M - M.mean()) ** 2).sum())
        v_mu = float(((mu - M.mean()) ** 2).sum() * na * na)
        v_al = float((alpha ** 2).sum() * na)
        v_be = float((beta ** 2).sum() * na)
        v_ga = float((gamma ** 2).sum())
        print(f"\n  [{name}]  {M.shape[0]} states")
        print(f"    {'mu    (state value V)':<28} {100*v_mu/tot:>7.3f}%")
        print(f"    {'alpha (ego main effect)':<28} {100*v_al/tot:>7.3f}%")
        print(f"    {'beta  (adv main effect)':<28} {100*v_be/tot:>7.3f}%")
        print(f"    {'gamma (INTERACTION)':<28} {100*v_ga/tot:>7.3f}%   <- needs minimax")
        act = (v_al + v_be + v_ga) / tot
        print(f"    {'any action effect':<28} {100*act:>7.3f}%")

        # Is the leading singular direction just all-ones?
        D = M - M.mean(0)
        ones = np.ones((na, na)) / na
        proj = (D * ones).sum(axis=(1, 2)) / max(np.linalg.norm(ones), 1e-30)
        e_ones = float((proj ** 2).sum() / max((D ** 2).sum(), 1e-30))
        print(f"    {'cross-state var on ALL-ONES':<28} {100*e_ones:>7.3f}%")
        return {"mu": v_mu / tot, "alpha": v_al / tot, "beta": v_be / tot,
                "gamma": v_ga / tot, "ones": e_ones}

    print("=" * 72)
    print(f"PAYOFF ANOVA  {os.path.basename(a.raw)}   {S} states, "
          f"{free.sum()} non-forced")
    print("=" * 72)
    anova(Q, "r + gamma*V(s')  ALL states")
    g = anova(Q[free], "r + gamma*V(s')  NON-FORCED")
    liveR = np.linalg.norm(R.reshape(S, -1), axis=1) > 0
    anova(R[liveR], "one-step reward r  (live states)")
    anova(R[free & liveR], "one-step reward r  NON-FORCED & live")

    if g:
        print("\n" + "=" * 72)
        if g["gamma"] < 0.02:
            print("  VERDICT: the INTERACTION term is negligible. Whatever action")
            print("  effect exists is SEPARABLE -- two independent action-value")
            print("  heads would capture it. A 484-cell joint matrix and a minimax")
            print("  solve are machinery with nothing to do.")
        elif g["mu"] > 0.95:
            print("  VERDICT: payoff is ~ pure state value. Q(s,i,j) = V(s).")
        else:
            print("  VERDICT: the interaction term is material -- a joint-action")
            print("  critic is the right object here.")


if __name__ == "__main__":
    main()
