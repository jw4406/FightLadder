"""LOUD tests for the k-step MC branch scoring in bootstrap_delta.expand_root.

Runs against a REAL checkpoint + emulator -- lbr_snapshot cannot be mocked, so
there is no unit-test shortcut. Three checks, each raises SystemExit on failure
(a test that does not run is indistinguishable from one that passes):

  1. REGRESSION  `--bootstrap --horizon 0` reproduces the OLD leaf
     r0 + gamma*V(s')*(1-done) bit-for-bit. Proves the k-step code collapses to
     the previous computation at its boundary.
  2. ACTION-EXACT  the deviation reward R is byte-exact on the action axis:
     R[0]==R[9] and R[:,0]==R[:,9] (actions 0 and 9 are byte-identical).
  3. AVERAGING  the MC standard error shrinks ~1/sqrt(K): SE(K=8)/SE(K=32) ~ 2,
     and the two path-averaged means agree within the pooled noise. Proves the
     paths are independent and the mean/SE are wired correctly.

Expects N_TESTS checks to run; if fewer run it is a FAILURE, not a pass.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_TESTS = 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="")
    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--num_step_frames", type=int, default=8)
    ap.add_argument("--n_envs", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=6, help="steps to reach a root")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, splice_terminal,
                                     infer_obs_kwargs)
    from bootstrap_delta import expand_root

    ran = 0
    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, _, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs, num_step_frames=a.num_step_frames,
                          reward_scale=a.reward_scale,
                          **infer_obs_kwargs(data, a.ram_mask or None))
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        gamma = ops.gamma
        n, na = venv.num_envs, ops.n_actions
        rng = np.random.RandomState(0)

        obs = venv.reset()
        for _ in range(a.warmup):
            obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                      ops.sample_adv(obs, rng)))[0]

        # ---- OLD-style leaf at this exact root: r0 + gamma*V(s')*(1-done) ----
        venv.env_method("lbr_snapshot", "test_ref")
        R_old = np.zeros((na, na, n)); V1 = np.zeros((na, na, n))
        DN_old = np.zeros((na, na, n), bool)
        for i in range(na):
            succ = []
            for j in range(na):
                venv.env_method("lbr_restore", "test_ref")
                o1, r_l, r_r, d, infos = venv.step(ops.joint(np.full(n, i),
                                                             np.full(n, j)))
                d = np.asarray(d, bool)
                R_old[i, j] = ops.lbr_reward(r_l, r_r)
                DN_old[i, j] = d
                succ.append(splice_terminal(o1, d, infos))
            V1[i] = ops.values_ego(np.concatenate(succ, axis=0)).reshape(na, n)
        venv.env_method("lbr_restore", "test_ref")
        venv.env_method("lbr_drop", "test_ref")
        M_old = R_old + gamma * V1 * (~DN_old)

        # ---- TEST 1: regression, bootstrap+horizon0 == old leaf, bit-for-bit --
        R1, M1, SE1, DN1 = expand_root(venv, ops, na, n, gamma,
                                       n_paths=1, horizon=0, bootstrap=True, rng=rng)
        d_reg = float(np.abs(M1 - M_old).max())
        d_r = float(np.abs(R1 - R_old).max())
        # R (deviation reward) must be BIT-EXACT. M may differ only by float32
        # batching roundoff on the V(s') term: old batches na*n successors into
        # one critic forward, expand_root calls V per branch on n rows, and
        # batch-size-dependent BLAS kernels round differently on identical input.
        # A real k=1 logic error (wrong discount/mask) would be O(V)~1e-2..1, so
        # ATOL=1e-5 separates roundoff from any genuine break by >3 orders.
        ATOL = 1e-5
        if d_r != 0.0:
            raise SystemExit(f"TEST 1 FAIL (regression): deviation reward differs by "
                             f"{d_r:.3e}, expected BIT-EXACT -- k=1 boundary broken")
        if d_reg > ATOL:
            raise SystemExit(f"TEST 1 FAIL (regression): |M_kstep - M_old|={d_reg:.3e} "
                             f"> {ATOL:.0e} -- too large for batching roundoff, real break")
        ran += 1
        print(f"  TEST 1 PASS  bootstrap+horizon0 == old r+gamma*V(s')  "
              f"(R bit-exact; M diff {d_reg:.1e} = FP roundoff, < {ATOL:.0e})")

        # ---- TEST 2: deviation reward is action-exact (0 vs 9) -----------------
        if na > 9:
            e_ax = float(max(np.abs(R1[0] - R1[9]).max(),
                             np.abs(R1[:, 0] - R1[:, 9]).max()))
            if e_ax != 0.0:
                raise SystemExit(f"TEST 2 FAIL (action-exact): R 0 vs 9 differ by "
                                 f"{e_ax:.3e}, expected 0 -- joint action axis wrong")
            ran += 1
            print(f"  TEST 2 PASS  R action-exact on both axes  (0 vs 9 diff {e_ax:.1e})")
        else:
            print(f"  TEST 2 SKIP  na={na} <= 9, no byte-identical pair")

        # ---- TEST 3: MC averaging -- SE ~ 1/sqrt(K), means agree ---------------
        rngA = np.random.RandomState(1); rngB = np.random.RandomState(2)
        _, M8, SE8, _ = expand_root(venv, ops, na, n, gamma,
                                    n_paths=8, horizon=40, bootstrap=False, rng=rngA)
        _, M32, SE32, _ = expand_root(venv, ops, na, n, gamma,
                                      n_paths=32, horizon=40, bootstrap=False, rng=rngB)
        # SE ratio: per-path std ~equal, so SE8/SE32 ~ sqrt(32/8) = 2. Compare on
        # cells with resolvable noise (SE32 > 0) to avoid 0/0.
        msk = SE32 > 1e-9
        if msk.sum() < 10:
            raise SystemExit(f"TEST 3 FAIL: only {int(msk.sum())} cells have MC noise "
                             f"-- horizon too short or rewards all zero, cannot test")
        ratio = float(np.median(SE8[msk] / SE32[msk]))
        # means must agree within the pooled SE (median z well under a few sigma)
        pooled = np.sqrt(SE8 ** 2 + SE32 ** 2) + 1e-12
        zmed = float(np.median(np.abs(M8 - M32) / pooled))
        if not (1.5 <= ratio <= 2.7):
            raise SystemExit(f"TEST 3 FAIL (SE scaling): median SE8/SE32={ratio:.2f}, "
                             f"expected ~2.0 (sqrt(32/8)) -- paths not independent or "
                             f"SE mis-wired")
        if zmed > 4.0:
            raise SystemExit(f"TEST 3 FAIL (mean agreement): K=8 vs K=32 means differ "
                             f"at median {zmed:.1f} sigma -- averaging biased")
        ran += 1
        print(f"  TEST 3 PASS  SE8/SE32={ratio:.2f} (~2.0), mean agree {zmed:.2f} sigma")
    finally:
        venv.close()

    expected = N_TESTS if na > 9 else N_TESTS - 1
    if ran != expected:
        raise SystemExit(f"LOUD FAIL: {ran}/{expected} tests ran -- a missing test "
                         f"is a silent failure")
    print(f"\nALL {ran}/{expected} TESTS PASSED")


if __name__ == "__main__":
    main()
