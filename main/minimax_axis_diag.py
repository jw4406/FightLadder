"""Why does the minimax head help on the EGO seat but not the ADV seat?

THE OBSERVATION (2026-08-10, minimax_phase0_vton, LBR minimax vs minimaxshuffle):

                    minimax    shuffle     diff
    6.72M   ADV    +0.01225   +0.01563   -0.0034    indistinguishable
   12.48M   ADV    -0.12304   -0.11314   -0.0099    indistinguishable
    6.72M   EGO    -0.05992   -0.16668   +0.1068    minimax MUCH better
   12.48M   EGO    -0.19098   -0.28001   +0.0890    minimax MUCH better

At ~0.025 SE the EGO gaps are 3.6-4.3 SE and replicate. This is backwards from
the naive expectation: `_minimax_q_update` SKIPS the ego path entirely (an ego
rollout carries no `dstb_actions`), so the head trains ONLY on adversary-path
data -- yet it helps only on the EGO seat.

LBR's ego seat enumerates ROWS of Q (a_ego); the adv seat enumerates COLUMNS
(a_adv). So the first question is whether Q simply has more usable structure
along one axis than the other. Three measurements, no rollout beyond collecting
states:

  D1 SPREAD   std of Q across rows vs across columns, at fixed state. This is
              the raw variation each seat has to order by.
  D2 NOISE    ASSUMPTION-FREE noise floor. Actions 0 and 9 are BYTE-IDENTICAL --
              DIRECTIONS_BUTTONS[0] == ATTACKS_BUTTONS[0] == [], both
              hold-nothing-for-8-frames (const.py). Truth therefore satisfies
              Q(0,j) == Q(9,j) and Q(i,0) == Q(i,9) exactly, so any measured
              difference is pure head noise -- no model, no assumption.
  D3 COVERAGE cell_visits marginals per axis. Ego policy entropy is -0.26
              against uniform -3.09, so rows may be trained far more unevenly
              than columns.

D1/D2 combine into a per-axis SIGNAL-TO-NOISE ratio, which is the number that
actually decides whether a seat can order its branches. Spread alone is not
signal -- a head can be arbitrarily spread and still uncorrelated with anything,
which is exactly what the first Phase 0 run did.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NOOP_A, NOOP_B = 0, 9   # byte-identical no-ops; see const.py DIRECTIONS/ATTACKS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=400,
                    help="vec-steps of state collection. These are per-state "
                         "matrix statistics, not return regressions, so a few "
                         "thousand states is plenty.")
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="minimax_axis_diag.json")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, REPO_ROOT)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]

    venv = build_lbr_venv(state, a.n_envs)
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        pol = ops.p
        if not getattr(pol, "minimax_q", False):
            raise SystemExit("checkpoint has no minimax head")
        head = pol.minimax_net[list(pol.minimax_net.keys())[head_idx]]
        visits = head.cell_visits.detach().cpu().numpy()

        rng = np.random.RandomState(0)
        Ms = []
        obs = venv.reset()
        for t in range(a.steps):
            ob = th.as_tensor(obs).to(ops.device)
            with th.no_grad():
                Ms.append(pol.minimax_matrices(ob, buf_num=[head_idx],
                                               stop_grad=True).cpu().numpy())
            obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                      ops.sample_adv(obs, rng)))[0]
    finally:
        venv.close()

    M = np.concatenate(Ms)                       # (N, n_ego, n_adv)
    N, n_ego, n_adv = M.shape
    print(f"\n   {N:,} states, matrix {n_ego}x{n_adv}")

    # D1 -- spread along each axis, at FIXED state.
    # ego axis: vary a_ego with a_adv held -> std over axis 1
    spread_ego = float(M.std(axis=1).mean())
    spread_adv = float(M.std(axis=2).mean())

    # D2 -- noise floor from the duplicated no-op. Truth: Q(0,j)==Q(9,j).
    noise_ego = float(np.abs(M[:, NOOP_A, :] - M[:, NOOP_B, :]).mean())
    noise_adv = float(np.abs(M[:, :, NOOP_A] - M[:, :, NOOP_B]).mean())
    snr_ego = spread_ego / noise_ego if noise_ego > 0 else float("inf")
    snr_adv = spread_adv / noise_adv if noise_adv > 0 else float("inf")

    # D3 -- coverage marginals.
    row_v, col_v = visits.sum(1), visits.sum(0)
    def imbal(v):
        v = np.sort(v)[::-1]
        return float(v[0] / max(v[-1], 1e-9))

    print("\n" + "=" * 74)
    print(f"AXIS DIAGNOSTIC  {os.path.basename(a.ckpt)}")
    print("=" * 74)
    print(f"  {'':22s} {'EGO axis (rows)':>18} {'ADV axis (cols)':>18}")
    print(f"  {'D1 spread of Q':22s} {spread_ego:>18.6f} {spread_adv:>18.6f}")
    print(f"  {'D2 noise (no-op 0 vs 9)':22s} {noise_ego:>18.6f} {noise_adv:>18.6f}")
    print(f"  {'   SNR = spread/noise':22s} {snr_ego:>18.2f} {snr_adv:>18.2f}")
    print(f"  {'D3 visits total':22s} {row_v.sum():>18,.0f} {col_v.sum():>18,.0f}")
    print(f"  {'   max/min imbalance':22s} {imbal(row_v):>18,.1f}x {imbal(col_v):>17,.1f}x")
    print(f"  {'   min marginal':22s} {row_v.min():>18,.0f} {col_v.min():>18,.0f}")

    res = {"checkpoint": os.path.basename(a.ckpt), "n_states": N,
           "spread_ego": spread_ego, "spread_adv": spread_adv,
           "noise_ego": noise_ego, "noise_adv": noise_adv,
           "snr_ego": snr_ego, "snr_adv": snr_adv,
           "visits_row_imbalance": imbal(row_v), "visits_col_imbalance": imbal(col_v),
           "visits_row_min": float(row_v.min()), "visits_col_min": float(col_v.min())}

    ratio = snr_ego / snr_adv if snr_adv > 0 else float("inf")
    res["snr_ratio_ego_over_adv"] = float(ratio)
    print(f"\n  SNR RATIO ego/adv = {ratio:.2f}")
    if ratio >= 1.5:
        res["verdict"] = "EGO AXIS BETTER"
        print("  => the ego axis carries more orderable signal per unit noise.")
        print("     That is the seat LBR-as-ego enumerates, so this ALONE could")
        print("     explain minimax >> shuffle on ego and ~= on adv.")
    elif ratio <= 1 / 1.5:
        res["verdict"] = "ADV AXIS BETTER"
        print("  => the adv axis is cleaner, which is the OPPOSITE of the LBR")
        print("     result. The asymmetry is then NOT explained by the matrix and")
        print("     the harness control (V-based lbr vs shuffle per seat) is the")
        print("     next test.")
    else:
        res["verdict"] = "AXES COMPARABLE"
        print("  => both axes carry comparable signal. The matrix does NOT explain")
        print("     the seat asymmetry; suspect the LBR harness or the frame/sign")
        print("     convention, and run V-based lbr vs shuffle in BOTH seats.")
    out = os.path.join(REPO_ROOT, a.out)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {out}")
    return res


if __name__ == "__main__":
    main()
