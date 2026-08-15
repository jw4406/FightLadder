"""Is corrW(R) broad, or carried by a minority of states? And is W under-scaled?

WHY THE AGGREGATE IS NOT ENOUGH. corr(Pw, Rw) pooled over every cell of every
state can be produced two ways: most states weakly right, or a few states very
right and the rest noise. Those imply different things about whether the head is
usable as a leaf evaluator -- the minimax operator reads ONE state's matrix at a
time, so a head that is excellent on 10% of states and noise on 90% is not a
usable bootstrap even at a high pooled correlation.

THE SCALE QUESTION. pred_std < tgt_std (0.0059 vs 0.0083 at 384k) while
fx_w_norm fell 3.8x during the same run. If the head has the right DIRECTIONS
but too small a MAGNITUDE, the regression slope of truth on prediction exceeds
1 and explained variance is lost purely to scale -- recoverable by rescaling,
without retraining. That is testable: fit R ~ a * P per state and read `a`.
A slope near 1 means the scale is right and the loss is genuine error; a slope
well above 1 means W is shrunk too far.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ram_mask", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     resolve_matchups, infer_obs_kwargs)

    d = np.load(a.npz)
    OBS = d["OBS"]; Rr = d["R"].astype(np.float64)
    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    hi, _, st = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(st, 2, **infer_obs_kwargs(data, a.ram_mask or None))
    try:
        m, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, m)
        with th.no_grad():
            P = m.policy.minimax_matrices(
                th.as_tensor(OBS, dtype=th.float32, device=a.device),
                buf_num=[hi]).cpu().numpy().astype(np.float64)
    finally:
        venv.close()

    Pw = P - P.mean(axis=(1, 2), keepdims=True)
    Rw = Rr - Rr.mean(axis=(1, 2), keepdims=True)
    # only states that HAVE interaction can be scored; a gamma==0 state has no
    # within-state structure to correlate against and would inject 0/0 noise.
    norm = np.linalg.norm(Rw.reshape(len(Rw), -1), axis=1)
    live = norm > 1e-12
    pc, slopes = [], []
    for s in np.where(live)[0]:
        p, r = Pw[s].ravel(), Rw[s].ravel()
        if p.std() < 1e-12:
            continue
        pc.append(np.corrcoef(p, r)[0, 1])
        slopes.append(float(np.dot(p, r) / np.dot(p, p)))   # r ~ a*p
    pc = np.array(pc); slopes = np.array(slopes)

    print(f"states {len(Rw)}   with interaction {int(live.sum())} ({live.mean():.1%})   scored {len(pc)}")
    print(f"\nPER-STATE corr(pred, R):")
    print(f"  mean {pc.mean():+.3f}   median {np.median(pc):+.3f}")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<2} {np.percentile(pc, q):+.3f}")
    print(f"  fraction of states with corr > 0    : {(pc > 0).mean():.1%}")
    print(f"  fraction with corr > 0.3            : {(pc > 0.3).mean():.1%}")
    print(f"  BROAD if most states are positive; NARROW if a tail carries it")

    print(f"\nSCALE  (slope a in  R ~ a * pred, per state):")
    print(f"  median slope {np.median(slopes):.3f}")
    print(f"  a > 1 means the head UNDER-predicts magnitude -- W shrunk too far")
    print(f"  pooled pred_std {Pw.std():.5f}   tgt_std {Rw.std():.5f}   "
          f"ratio {Rw.std()/max(Pw.std(),1e-30):.3f}")
    # what would evW(R) become if the ONLY defect were scale?
    k = float(np.dot(Pw.ravel(), Rw.ravel()) / np.dot(Pw.ravel(), Pw.ravel()))
    ev_now = 1 - ((Pw - Rw) ** 2).mean() / Rw.var()
    ev_scaled = 1 - ((k * Pw - Rw) ** 2).mean() / Rw.var()
    print(f"\n  evW(R) as-is        {ev_now:+.3f}")
    print(f"  evW(R) if rescaled  {ev_scaled:+.3f}   (optimal global k = {k:.3f})")
    print(f"  the gap is what a pure SCALE fix would recover")


if __name__ == "__main__":
    main()
