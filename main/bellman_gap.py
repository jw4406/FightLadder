"""Is the critic's Bellman residual a DEFECT, or the arithmetic of correct shrinkage?

THE NUMBER THIS INTERROGATES. bootstrap_delta.py collects, as a free side-effect,

    Delta(s) = V_critic(s) - V_pi(s),   V_pi(s) = sum_ij pi_e(i) pi_a(j) M_ij

i.e. how far the critic's own V is from the one-step Bellman backup of ITSELF
under the current policies. Measured 0.002824 at 2.4M = 18.24% of the payoff std,
which is 2.2x the ENTIRE effect of swapping the bootstrap to minimax
(|V_minimax - V_pi| = 0.001265). That ordering is why this is being looked at
before Phase 1: changing an operator by less than the operator is already wrong
optimizes the wrong term.

TWO PROPERTIES MAKE THIS CLEAN, and neither holds for most Bellman-residual
measurements. The transition is deterministic given the joint action, and all
22x22 = 484 branches are enumerated with the real emulator. So the expectation
over actions is EXACT and the transition contributes no variance: Delta is pure
function-approximation error with ZERO Monte Carlo noise. 0.002824 is not an
estimate of a residual, it IS the residual.

THE HYPOTHESIS. [[slope-is-optimal-shrinkage-not-miscalibration]] established
that the critic's affine slope < 1 is the CORRECT response to being a noisy
predictor (measured within +/-0.03 of the MSE-optimal slope at every checkpoint
over 82M steps). If the critic is a shrunk copy of the truth, V_hat = a + b V*
with b < 1, then the Bellman residual is FORCED:

    V_pi   = sum_pi[r + gamma(a + b V*(s'))] = R_bar + gamma a + gamma b sum_pi V*(s')
    V*(s)  = R_bar + gamma sum_pi V*(s')          (definition of V*)
    =>  V_pi = gamma a + b V*(s) + (1-b) R_bar
    =>  Delta = V_hat(s) - V_pi = a(1-gamma) - (1-b) R_bar(s)

So under pure affine shrinkage Delta is AFFINE IN THE EXPECTED IMMEDIATE REWARD,
with slope -(1-b) and offset a(1-gamma) -- both parameters already measured
elsewhere, gamma known. A falsifiable prediction with no free parameters.

If it holds, "fix the Bellman gap" means "undo the shrinkage", which was measured
to INCREASE MSE. The gap is then not a defect and the diagnostic closes itself.

WHAT THIS FILE DOES (phase A) vs WHAT IT CANNOT (phase A5). The sharp test needs
R_bar, and the saved npz stores only M = R + gamma*V1*(~done) -- R and V1 are
collapsed into one array and cannot be separated after the fact. So phase A runs
the tests that need only (M, PE, PA, V0):

  bias vs dispersion   |mean Delta| / mean|Delta|. A residual that is mostly
                       dispersion averages out along a trajectory and barely
                       moves the fixed point; a systematic one shifts it. This is
                       the same cut that settled operator-vs-Bellman-error, where
                       the Bellman error came out 98.6% dispersion.
  corr(Delta, V0)      the WEAK form of the prediction. Under shrinkage Delta
                       depends on R_bar ALONE, so if R_bar is not itself strongly
                       tied to the value level, Delta should be near-uncorrelated
                       with V0. A strong correlation falsifies the model here,
                       for free, without collecting anything.
  constant-critic null the floor. mean|Delta| for the best CONSTANT critic. If V
                       is not much better than a constant at satisfying its own
                       Bellman equation, 18.24% is not the interesting number --
                       the whole scale is.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _fit(x, y):
    """Least-squares y ~ c0 + c1 x. Returns (c1, c0, r2)."""
    import numpy as np
    x = np.asarray(x, np.float64).reshape(-1)
    y = np.asarray(y, np.float64).reshape(-1)
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[1]), float(coef[0]), r2, pred


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=str,
                    default=os.path.join(REPO_ROOT, "bootstrap_delta_2400000_big_raw.npz"),
                    help="raw matrices from bootstrap_delta.py (M, PE, PA, V0)")
    ap.add_argument("--gamma", type=float, default=0.94,
                    help="discount the checkpoint was TRAINED with. Not recoverable "
                         "from the npz; run_minimax_phase0.sh sets 0.94. Only used "
                         "to report the predicted offset a(1-gamma), never to "
                         "recompute M.")
    ap.add_argument("--out", type=str, default="bellman_gap.json")
    a = ap.parse_args(argv)

    import numpy as np

    d = np.load(a.npz)
    M = d["M"].astype(np.float64)                  # (S, na, na)
    PE = d["PE"].astype(np.float64)                # (S, na)
    PA = d["PA"].astype(np.float64)                # (S, na)
    V0 = d["V0"].astype(np.float64).reshape(-1)    # (S,)
    S = M.shape[0]

    V_pi = np.einsum("si,sij,sj->s", PE, M, PA)
    delta = V0 - V_pi

    sd_tot = float(M.std())
    sd_within = float((M - M.mean(axis=(1, 2), keepdims=True)).std())
    mad = float(np.abs(delta).mean())

    print("=" * 78)
    print(f"BELLMAN GAP   {os.path.basename(a.npz)}   {S} states   gamma={a.gamma}")
    print("=" * 78)
    print(f"  payoff std   total {sd_tot:.6f}   within-state {sd_within:.6f}")
    print(f"  mean|Delta|  {mad:.6f}   = {mad/sd_tot:.2%} of total std"
          f"   (reproduces bootstrap_delta's free check)")

    # ---- 1. bias vs dispersion -------------------------------------------
    # A zero-mean residual averages out along a trajectory: the lambda-return
    # telescopes and errors of alternating sign cancel. A systematic one does
    # not -- it shifts the fixed point the critic converges to.
    mean_d = float(delta.mean())
    bias_frac = abs(mean_d) / mad if mad > 0 else float("nan")
    print(f"\n  --- 1. BIAS vs DISPERSION ---")
    print(f"  mean Delta            {mean_d:+.6f}")
    print(f"  std  Delta            {float(delta.std()):.6f}")
    print(f"  |mean| / mean|.|      {bias_frac:.4f}"
          f"   ({'SYSTEMATIC' if bias_frac > 0.5 else 'mostly DISPERSION'})")

    # ---- 2. the constant-critic null -------------------------------------
    # Floor: what residual does the BEST CONSTANT critic get? V_hat = c makes
    # Delta = c - V_pi, minimised in mean|.| by the MEDIAN of V_pi.
    c_star = float(np.median(V_pi))
    mad_null = float(np.abs(c_star - V_pi).mean())
    print(f"\n  --- 2. CONSTANT-CRITIC NULL ---")
    print(f"  best constant c*      {c_star:+.6f}")
    print(f"  mean|Delta| at c*     {mad_null:.6f}   = {mad_null/sd_tot:.2%} of total std")
    print(f"  critic vs null        {mad/mad_null:.3f}x"
          f"   ({'BETTER' if mad < mad_null else 'NO BETTER'} than a constant)")

    # ---- 3. the weak form of the shrinkage prediction --------------------
    # Delta = a(1-gamma) - (1-b) R_bar depends on R_bar ALONE. R_bar is the
    # expected immediate reward, which is a local quantity; the value LEVEL V0
    # integrates the whole future. If they are not strongly tied, shrinkage
    # predicts Delta is near-uncorrelated with V0. Strong correlation falsifies.
    r_dv0 = float(np.corrcoef(delta, V0)[0, 1])
    r_dvpi = float(np.corrcoef(delta, V_pi)[0, 1])
    sl_v0, ic_v0, r2_v0, pred_v0 = _fit(V0, delta)
    resid_v0 = float(np.abs(delta - pred_v0).mean())
    print(f"\n  --- 3. WEAK SHRINKAGE TEST ---")
    print(f"  corr(Delta, V0)       {r_dv0:+.4f}")
    print(f"  corr(Delta, V_pi)     {r_dvpi:+.4f}")
    print(f"  Delta ~ V0 fit        slope {sl_v0:+.4f}  intercept {ic_v0:+.6f}  R2 {r2_v0:.4f}")
    print(f"  mean|Delta| after     {resid_v0:.6f}"
          f"   ({1 - resid_v0/mad:.1%} of the residual removed by V0 alone)")

    # V0 against V_pi directly: if the critic were Bellman-consistent this is
    # the identity. Deviation from slope 1 is the visible face of the same thing.
    sl_c, ic_c, r2_c, _ = _fit(V_pi, V0)
    print(f"  V0 ~ V_pi fit         slope {sl_c:+.4f}  intercept {ic_c:+.6f}  R2 {r2_c:.4f}")

    # ---- 4. the comparison the DECISION actually rests on -----------------
    # Raw magnitude is the wrong yardstick for "which error should I fix
    # first". A zero-mean residual telescopes through the lambda-return
    # recursion -- errors of alternating sign cancel along a trajectory -- so it
    # inflates variance without moving the fixed point. A SYSTEMATIC one moves
    # the fixed point. Compare the two candidate interventions on the same
    # states, by the same statistic, rather than by mean|.| alone.
    import torch as th
    from common.minimax import solve_matrix_game
    sol = solve_matrix_game(th.as_tensor(M, dtype=th.float32), iters=1024, eta=0.5)
    d_op = sol.V.cpu().numpy().reshape(-1).astype(np.float64) - V_pi
    op_bias = abs(float(d_op.mean())) / float(np.abs(d_op).mean())
    print(f"\n  --- 4. BELLMAN ERROR vs OPERATOR CHANGE (same states) ---")
    print(f"  {'quantity':<28} {'mean|.|':>10} {'|mean|':>10} {'bias frac':>10}")
    print(f"  {'Delta  (Bellman residual)':<28} {mad:>10.6f} "
          f"{abs(mean_d):>10.6f} {bias_frac:>10.4f}")
    print(f"  {'V_minimax - V_pi (operator)':<28} {float(np.abs(d_op).mean()):>10.6f} "
          f"{abs(float(d_op.mean())):>10.6f} {op_bias:>10.4f}")
    print(f"  operator is {op_bias/bias_frac:.1f}x more SYSTEMATIC despite being "
          f"{float(np.abs(d_op).mean())/mad:.2f}x the magnitude")

    # ---- verdict / gate ---------------------------------------------------
    print("\n" + "=" * 78)
    if bias_frac > 0.5:
        verdict = "SYSTEMATIC -- shrinkage model does NOT fit, localize it"
        print("  => the residual is mostly a SYSTEMATIC offset, not dispersion.")
        print("     Affine shrinkage predicts a term in R_bar, which varies state")
        print("     to state; a constant offset is a different mechanism. Go to")
        print("     phase B (conditioning / multi-step / cross-checkpoint).")
    elif abs(r_dv0) > 0.5:
        verdict = "CORRELATED WITH V0 -- shrinkage model falsified"
        print("  => Delta tracks the value LEVEL, which the shrinkage model says")
        print("     it should not (Delta depends on R_bar alone). The residual is")
        print("     something other than correct shrinkage. Go to phase B.")
    else:
        verdict = "CONSISTENT WITH SHRINKAGE -- run the sharp test (A5)"
        print("  => dispersion-dominated and not tied to the value level, which is")
        print("     what affine shrinkage predicts. This is NECESSARY but not")
        print("     sufficient: the sharp test needs R_bar. Add R to")
        print("     bootstrap_delta.py's savez and fit Delta = c0 + c1*R_bar,")
        print("     checking c1 ~ -(1-b) and c0 ~ a(1-gamma) against the")
        print("     PRE-MEASURED slope and intercept.")

    res = {
        "npz": os.path.basename(a.npz), "n_states": S, "gamma": a.gamma,
        "payoff_std": sd_tot, "payoff_std_within": sd_within,
        "mean_abs_delta": mad, "mean_abs_delta_rel": mad / sd_tot,
        "mean_delta": mean_d, "std_delta": float(delta.std()),
        "bias_fraction": bias_frac,
        "const_null_mad": mad_null, "critic_vs_null": mad / mad_null,
        "corr_delta_v0": r_dv0, "corr_delta_vpi": r_dvpi,
        "fit_delta_v0": {"slope": sl_v0, "intercept": ic_v0, "r2": r2_v0,
                         "residual_mad": resid_v0},
        "fit_v0_vpi": {"slope": sl_c, "intercept": ic_c, "r2": r2_c},
        "operator_mad": float(np.abs(d_op).mean()),
        "operator_bias_fraction": op_bias,
        "systematic_ratio_operator_over_bellman": op_bias / bias_frac,
        "verdict": verdict,
    }
    out = os.path.join(REPO_ROOT, a.out)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {out}")
    return res


if __name__ == "__main__":
    main()
