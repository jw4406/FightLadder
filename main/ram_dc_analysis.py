"""How much of the trunk's DC dominance does the mask actually remove?

MEASURED PROBLEM (spar_Ry_Sa_2880000, --obs_type ram):

    DC / fluctuation      minimax trunk  201.06     value trunk  24.56
    (image baseline)                       4.33                   1.08

For a first layer h = W x + b with random W, the ratio the trunk inherits is
set by the INPUT:

    ||h_bar|| / ||delta_h||  ~  ||x_bar|| / ||delta_x||

because W maps both isotropically. So the fix is whatever shrinks ||x_bar||
without shrinking ||delta_x||. Two independent contributions:

  DEAD COORDINATES  63,431 of 65,536 RAM bytes never change. They add to
                    ||x_bar|| and contribute exactly nothing to ||delta_x||.
                    Masking deletes them.
  PER-BYTE OFFSET   a live byte cycling 100..110 still has mean 105 and std ~3,
                    so it carries its own DC even after masking. Only
                    STANDARDIZING (subtract per-byte mean, divide by per-byte
                    std) removes this, and the mask cannot.

This measures all three regimes on real rollout data so the decision is made on
numbers rather than on the theory above.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=str,
                    default="two_player/Ryu_left/Champion.Level1.RyuVsSagat.2Player.state")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n_envs", type=int, default=6)
    ap.add_argument("--mask", type=str, default="ram_mask.npy")
    a = ap.parse_args(argv)

    import numpy as np
    from local_best_response import build_lbr_venv, REPO_ROOT

    mask = np.load(a.mask if os.path.isabs(a.mask)
                   else os.path.join(REPO_ROOT, a.mask))
    venv = build_lbr_venv(a.state, a.n_envs)
    try:
        n = venv.num_envs
        rng = np.random.RandomState(0)
        R = []
        venv.reset()
        for t in range(a.steps):
            venv.step(np.stack([rng.randint(0, 22, size=n),
                                rng.randint(0, 22, size=n)], axis=-1))
            R.append(np.stack([np.asarray(r) for r in venv.env_method("lbr_ram")]))
    finally:
        venv.close()

    RAW = np.concatenate(R)                              # (N, 65536) uint8
    X = RAW.astype(np.float64) / 255.0                   # as fed to the network
    N = X.shape[0]
    xbar = X.mean(0)
    dX = X - xbar

    def dc(x_bar, d_x, name, dim):
        nb = float(np.linalg.norm(x_bar))
        nd = float(np.linalg.norm(d_x, axis=1).mean())
        r = nb / max(nd, 1e-12)
        print(f"  {name:<34} {dim:>8,} {nb:>10.4f} {nd:>10.4f} {r:>12.2f}")
        return r

    print("\n" + "=" * 78)
    print(f"RAM INPUT GEOMETRY   {N:,} states   mask = {mask.size:,} bytes")
    print("=" * 78)
    print(f"  {'regime':<34} {'dim':>8} {'||xbar||':>10} {'||dx||':>10} "
          f"{'DC/fluct':>12}")
    r_full = dc(xbar, dX, "full RAM (what is training now)", X.shape[1])
    r_mask = dc(xbar[mask], dX[:, mask], "masked", mask.size)
    # Standardize the masked input: per-byte mean 0, unit std.
    sd = dX[:, mask].std(0) + 1e-8
    Xs = (X[:, mask] - xbar[mask]) / sd
    # Standardizing sets the mean to zero BY CONSTRUCTION, so its DC/fluct is 0
    # trivially and the "improvement factor" against it is meaningless. Reported
    # for completeness as the floor, not as a ratio.
    r_std = dc(Xs.mean(0), Xs - Xs.mean(0), "masked + standardized (floor)", mask.size)

    # Count liveness on the RAW uint8. Doing it on the float64 copy counts
    # round-off as variation: summing N identical v/255 values and dividing
    # returns a mean off by an ulp, so std > 0 for CONSTANT bytes and the count
    # came out 18,542 instead of the true 2,045.
    live = (RAW.max(0) > RAW.min(0)).sum()
    print(f"\n  live bytes in this sample: {live:,} / {X.shape[1]:,} "
          f"({live/X.shape[1]:.2%})")
    print(f"\n  improvement from masking alone      {r_full/max(r_mask,1e-12):>8.1f}x")
    print(f"  standardizing removes the DC term entirely (ratio 0 by construction)")
    print("\n  predicted trunk DC/fluct (current value trunk reads 24.56):")
    print(f"    masked            ~{24.56 * r_mask / max(r_full,1e-12):>7.2f}")
    print(f"    masked+standard    ~0     (DC removed by construction; "
          f"image baseline 1.08)")


if __name__ == "__main__":
    main()
