"""Is the head's Q good on states it has NOT been fit to, cell by cell?

WHY THIS AND NOT train/minimax_ev. That metric is computed on the TRAINING batch
and on the ONE played cell per transition. It has read ~0.5 while the held-out
value was ~0 before, and it read 0.23 -> 0.66 through an entire run whose head
turned out to predict held-out matrices worse than a constant. So it cannot
answer the question the bootstrap actually asks.

WHAT THE BOOTSTRAP ASKS. kappa>0 replaces the scalar leaf value with

    V_minimax(s') = equilibrium value of the head's FULL 22x22 matrix at s'

evaluated at SUCCESSOR states during rollout. The head received gradient on one
cell of each such state, never the other 483. So the operative question is: given
a state from the policy's own distribution, how good is the WHOLE matrix?

THE DECOMPOSITION THAT MATTERS. Raw EV over all cells is dominated by mu(s),
which is just V(s) -- a scalar critic gets that right and it tells us nothing
about joint-action value. Subtracting each state's own mean leaves the
action-conditional part, which is the only thing a joint critic exists to
provide:

    ev_within = 1 - Var(pred - target | state-centred) / Var(target | state-centred)

A SCALAR CRITIC SCORES EXACTLY 0 HERE by construction. Below 0 means the head is
worse than having no action-conditioning at all, and V_minimax built on it is
noise.

Needs collections made with `bootstrap_delta.py --save_obs`.
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
    ap.add_argument("--run_dir", required=True, help="trained_models/tasks/todo of the arm")
    ap.add_argument("--npz_glob", required=True, help="enum_*_raw.npz with OBS")
    ap.add_argument("--ram_mask", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=2000,
                    help="Bootstrap resamples OVER STATES for a 95%% CI on corrW(R). "
                         "With no uncertainty attached, 0.62 against -0.09 is not "
                         "obviously outside noise at 300 states -- the interval is "
                         "the claim, not the point estimate.")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     resolve_matchups, infer_obs_kwargs)
    from gamma_basis import gammas as _gam

    def ev(pred, tgt):
        resid = float(((pred - tgt) ** 2).mean())
        var = float(tgt.var())
        return 1.0 - resid / var if var > 1e-30 else float("nan")

    print(f"{'ckpt':>11} {'states':>7} {'ev_all':>8} {'evW(M)':>10} "
          f"{'corrW(M)':>12} {'evW(R)':>9} {'corrW(R)':>9} {'95% CI':>17}"
          f" {'CONST':>8} {'headroom':>8}")
    print(f"{'':>11} {'':>7} {'':>8} {'<-- scalar critic = 0.000':>10}")
    print(f"{'':>11}  CONST = a single fixed matrix for every state; HEADROOM = "
          f"corrW(R) - CONST is the only part that is state-CONDITIONAL")
    for f in sorted(glob.glob(a.npz_glob)):
        steps = int(re.search(r"_(\d+)_raw", f).group(1))
        ck = os.path.join(a.run_dir, f"spar_Ry_Sa_{steps}_steps.task")
        if not os.path.exists(ck):
            print(f"{steps:11,}  NO CHECKPOINT")
            continue
        d = np.load(f)
        if "OBS" not in d:
            print(f"{steps:11,}  npz has no OBS")
            continue
        OBS = d["OBS"]; M = d["M"].astype(np.float32)
        Rr = d["R"].astype(np.float64)   # emulator reward, NO critic anywhere

        data = load_from_zip_file(ck, device="cpu")[0]
        head_idx, _, state = resolve_matchups(data, "all")[0]
        venv = build_lbr_venv(state, 2, **infer_obs_kwargs(data, a.ram_mask or None))
        try:
            model, _ = load_checkpoint(ck, venv, a.device)
            preflight(venv, model)
            policy = model.policy
            with th.no_grad():
                P = policy.minimax_matrices(
                    th.as_tensor(OBS, dtype=th.float32, device=a.device),
                    buf_num=[head_idx]).cpu().numpy().astype(np.float64)
        finally:
            venv.close()

        T = M.astype(np.float64)
        # state-centred: remove each state's own mean from BOTH, leaving only the
        # action-conditional structure the joint critic is supposed to supply.
        Pw = P - P.mean(axis=(1, 2), keepdims=True)
        Tw = T - T.mean(axis=(1, 2), keepdims=True)
        cw = float(np.corrcoef(Pw.ravel(), Tw.ravel())[0, 1])
        # Same against R. The head is trained on r + gamma*V_mm(s') while M uses
        # V_scalar(s'), so a correct head could score badly on M from target
        # mismatch alone. R has no critic in it at all, so agreement with R's
        # action-conditional structure is mismatch-free evidence either way.
        Rw = Rr - Rr.mean(axis=(1, 2), keepdims=True)
        cr = float(np.corrcoef(Pw.ravel(), Rw.ravel())[0, 1])
        # THE BASELINE THAT MUST BE BEATEN. corrW pools over every cell of every
        # state, so if a regime's gamma matrices are all similar, a head that
        # emits ONE FIXED matrix scores well while knowing nothing
        # state-specific -- and the minimax operator reads one state at a time,
        # so such a head cannot rank branches. Measured: the frozen-ego regime
        # (517-step timer-out stalemates, mean pairwise gamma corr 0.36) gives a
        # CONSTANT predictor corrW(R) = 0.412, so its celebrated head score of
        # 0.605 was mostly free. Healthy self-play scores ~0.02 here. Reporting
        # corrW without this column overstated a result by ~2/3 for a full day.
        _gm = _gam(Rr)
        _live = np.linalg.norm(_gm.reshape(len(_gm), -1), axis=1) > 1e-12
        if _live.sum() >= 3:
            _P = np.broadcast_to(_gm[_live].mean(axis=0), Rr.shape)
            _Pw = _P - _P.mean(axis=(1, 2), keepdims=True)
            _const = float(np.corrcoef(_Pw.ravel(), Rw.ravel())[0, 1])
        else:
            _const = float("nan")
        # resample STATES, not cells: the 484 cells within a state are highly
        # dependent, so a cell-level bootstrap would understate the interval.
        _rng = np.random.RandomState(0); _n = len(Pw); _cs = []
        for _ in range(a.boot):
            _i = _rng.randint(0, _n, _n)
            _cs.append(np.corrcoef(Pw[_i].ravel(), Rw[_i].ravel())[0, 1])
        _lo, _hi = np.percentile(_cs, 2.5), np.percentile(_cs, 97.5)
        print(f"{steps:11,} {len(T):7d} {ev(P, T):8.3f} {ev(Pw, Tw):10.3f} "
              f"{cw:12.3f} {ev(Pw, Rw):9.3f} {cr:9.3f} [{_lo:+.3f},{_hi:+.3f}]"
              f" {_const:8.3f} {cr - _const:+8.3f}")


if __name__ == "__main__":
    main()
