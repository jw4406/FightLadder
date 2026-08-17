"""Score V_scalar and the head's V_pi against REALIZED returns, not a Bellman backup.

WHY THIS EXISTS. Comparing V_scalar(s) to sum_pi pi [R + gamma V_scalar(s')] is a
comparison of the critic with a ONE-STEP BACKUP OF ITSELF. Any converged critic
satisfies that approximately whether or not it predicts anything real -- it
scored EV 0.944 that way, while this project has separately measured the critic's
actual return-prediction EV at ~0.05. Self-consistency is not accuracy. The same
error produced an enum_ev_holdout of +0.85 on a head whose corrW(R) was -0.012.

So the target here is INDEPENDENT of every critic: discounted returns actually
realized by rolling the current policies forward from each state.

    V_mc(s) = mean over K rollouts of  sum_t gamma^t r_t

Both candidate leaves are then scored against it:
    V_scalar(s)                       the critic in use today
    V_pi_head(s) = sum_ij PE_i Q_ij PA_j   the analytic expectation from the head

K rollouts per state, restoring to the same snapshot each time, so the MC noise
is averaged rather than inherited. With gamma=0.94 the effective horizon is
~1/(1-gamma) = 17 steps, so H well past that captures essentially the whole
return.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MC_ROOT = "mc_root"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="")
    ap.add_argument("--n_states", type=int, default=192)
    ap.add_argument("--rollouts", type=int, default=8, help="K, averaged per state")
    ap.add_argument("--horizon", type=int, default=120, help="H; ~7x the gamma horizon")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--n_envs", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, infer_obs_kwargs)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, _, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs, **infer_obs_kwargs(data, a.ram_mask or None))
    OBS, VS, VH, VMC, MCSE = [], [], [], [], []
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        gamma = ops.gamma
        n = venv.num_envs
        rng = np.random.RandomState(0)
        has_head = bool(getattr(model.policy, "minimax_q", False))
        print(f"[mc] gamma {gamma}  K={a.rollouts}  H={a.horizon}  "
              f"head={'yes' if has_head else 'NO'}")

        obs = venv.reset()
        n_batches = int(np.ceil(a.n_states / n))
        for b in range(n_batches):
            for _ in range(a.stride if b else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]
            OBS.append(np.asarray(obs).copy())
            VS.append(ops.values_ego(obs))
            if has_head:
                with th.no_grad():
                    Q = model.policy.minimax_matrices(
                        th.as_tensor(obs, dtype=th.float32, device=a.device),
                        buf_num=[head_idx]).cpu().numpy().astype(np.float64)
                VH.append(np.einsum("si,sij,sj->s", ops.ego_probs(obs), Q,
                                    ops.adv_probs(obs)))

            # K independent rollouts from THIS state, restoring each time.
            # NO lbr_pause_monitor here: build_lbr_venv is deliberately
            # Monitor2P-FREE (that is why the LBR driver exists), so there is no
            # monitor to suspend and the call falls through to RetroEnv and
            # raises. Pausing is only needed when branching the TRAINING envs.
            venv.env_method("lbr_snapshot", MC_ROOT)
            g = np.zeros((a.rollouts, n))
            for k in range(a.rollouts):
                venv.env_method("lbr_restore", MC_ROOT)
                o = obs
                alive = np.ones(n, bool)
                disc = 1.0
                for t in range(a.horizon):
                    o, r_l, r_r, d, _ = venv.step(
                        ops.joint(ops.sample_ego(o, rng), ops.sample_adv(o, rng)))
                    g[k] += disc * ops.lbr_reward(r_l, r_r) * alive
                    alive &= ~np.asarray(d, bool)
                    disc *= gamma
                    if not alive.any():
                        break
            venv.env_method("lbr_restore", MC_ROOT)
            venv.env_method("lbr_drop", MC_ROOT)
            VMC.append(g.mean(axis=0))
            # per-state MC standard error, so the EV can be corrected for
            # target noise. Without it an EV against a noisy target is a LOWER
            # BOUND of unknown tightness -- which is not a quotable number.
            MCSE.append(g.std(axis=0, ddof=1) / np.sqrt(a.rollouts))
            print(f"   batch {b+1}/{n_batches}", flush=True)
    finally:
        venv.close()

    V_s = np.concatenate(VS); V_mc = np.concatenate(VMC)
    se = np.concatenate(MCSE)
    # Var(V_mc) = Var(true V_pi) + E[se^2].  Subtract the noise floor to get the
    # variance that is actually EXPLAINABLE, and report the noise-corrected EV.
    noise = float((se ** 2).mean()); tot = float(V_mc.var())
    signal = max(tot - noise, 1e-30)
    def ev(p, t):
        return 1.0 - float(((p - t) ** 2).mean()) / max(float(t.var()), 1e-30)

    print(f"\n{len(V_mc)} states, target = REALIZED discounted return "
          f"(mean of {a.rollouts} rollouts)")
    print(f"  V_mc      mean {V_mc.mean():+.5f}  std {V_mc.std():.5f}")
    print(f"  target variance: total {tot:.3e} = signal {signal:.3e} + MC noise {noise:.3e}"
          f"   ({noise/tot:.1%} noise)")
    print(f"\n{'leaf':>16} {'EV':>8} {'EV(denoised)':>10} {'corr':>8} {'slope':>8}")
    rows = [("V_scalar", V_s)]
    if VH:
        rows.append(("V_pi(head)", np.concatenate(VH)))
    for lab, v in rows:
        sl = float(np.polyfit(v, V_mc, 1)[0])
        e = ev(v, V_mc)
        # noise-corrected: residual also contains the MC noise, so remove it
        e_c = 1.0 - (float(((v - V_mc) ** 2).mean()) - noise) / signal
        print(f"{lab:>16} {e:8.3f} {e_c:10.3f} {np.corrcoef(v, V_mc)[0,1]:8.3f} {sl:8.3f}")
    print(f"\n  EV is against an INDEPENDENT target -- no critic appears in V_mc.")
    print(f"  A constant predictor scores 0. Prior record: the critic's")
    print(f"  return-prediction EV has been measured at ~0.05 on this task.")
    if a.out:
        np.savez_compressed(a.out, V_scalar=V_s, V_mc=V_mc,
                            **({"V_head": np.concatenate(VH)} if VH else {}),
                            OBS=np.concatenate(OBS))
        print(f"  saved -> {a.out}")


if __name__ == "__main__":
    main()
