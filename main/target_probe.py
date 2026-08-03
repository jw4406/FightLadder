"""Is the V-TRACE TARGET itself predictive of realized returns?

If the target is good and the net can't fit it -> the network is at fault.
If the TARGET doesn't track realized returns -> the net is doing its job and the
target generation is the disease. Every architectural fix would be treating a
symptom.

With ratios ~ 1 (log shows vtrace_ratio_mean = 0.998, rho_sat_frac = 0) the
V-trace target telescopes to the T-step return plus a bootstrap:
    v_t = sum_k gamma^k r_{t+k}  +  gamma^T V(x_{t+T})
At T=64, gamma=0.99 the bootstrap carries weight gamma^T = 0.527 -- more than
half the target is the network's own estimate.

Reported, all against the SAME realized returns G on the same samples:
    V(x_t)          what the net predicts now
    v_target        what it is being trained toward
    reward_part     the target with the bootstrap term removed
    bootstrap_part  the gamma^T V(x_{t+T}) term alone
"""
import sys, argparse, numpy as np, torch as th
sys.path.insert(0, "/home/jw4406/codebase/FightLadder/main")
from stable_baselines3.common.save_util import load_from_zip_file
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups)


def ev(p, y):
    return float(1 - (p - y).var() / y.var()) if y.var() > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--ego_zero", action="store_true",
                    help="Force the ego to action 0, matching --ego_style zero_action. "
                         "Without this the probe rolls a SAMPLED ego, i.e. a state "
                         "distribution the critic never trained on.")
    a = ap.parse_args()
    d = load_from_zip_file(a.ckpt, device="cpu")[0]
    T = a.T if a.T else int(d.get("vtrace_seq_len") or 64)
    hi, lab, state = resolve_matchups(d, "all")[0]
    venv = build_lbr_venv(state, a.n_envs)
    try:
        m, _ = load_checkpoint(a.ckpt, venv, "cuda"); preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        g = ops.gamma
        rng = np.random.RandomState(0); V, R, D = [], [], []
        obs = venv.reset()
        for _ in range(a.steps):
            V.append(ops.values_ego(obs) * ops.sgn)
            ae = (np.zeros(obs.shape[0], dtype=np.int64) if a.ego_zero
                  else ops.sample_ego(obs, rng))
            ad = ops.sample_adv(obs, rng)
            obs, rl, rr, dn, _ = venv.step(ops.joint(ad, ae))
            R.append(ops.lbr_reward(rl, rr)); D.append(np.asarray(dn, bool))
    finally:
        venv.close()
    V = np.array(V); R = np.array(R); D = np.array(D)      # (S, n)
    S, n = R.shape

    # realized return to episode end (only samples whose episode completed)
    G = np.zeros_like(R); valid = np.zeros_like(D)
    acc = np.zeros(n); seen = np.zeros(n, bool)
    for t in reversed(range(S)):
        acc = R[t] + g * acc * (~D[t]); seen |= D[t]; G[t] = acc; valid[t] = seen

    # T-step reward sum + bootstrap, with episode-boundary masking
    rew = np.zeros_like(R); boot = np.zeros_like(R); ok = np.zeros_like(D)
    for t in range(S - T):
        alive = np.ones(n, bool); disc = np.ones(n); s = np.zeros(n)
        for k in range(T):
            s += disc * R[t + k] * alive
            disc *= g
            alive &= ~D[t + k]
        rew[t] = s
        boot[t] = (g ** T) * V[t + T] * alive
        ok[t] = True
    msk = (valid & ok).reshape(-1)
    Vf, Gf = V.reshape(-1)[msk], G.reshape(-1)[msk]
    rf, bf = rew.reshape(-1)[msk], boot.reshape(-1)[msk]
    tgt = rf + bf

    print(f"\n  {lab}  steps={d.get('num_timesteps')}  T={T}  gamma={g}  gamma^T={g**T:.3f}  n={msk.sum()}")
    print(f"\n  {'quantity':28s} {'EV vs realized G':>17s} {'std':>9s}")
    for nm, x in (("V(x_t)  (net output)", Vf), ("v_target (training target)", tgt),
                  ("  reward part only", rf), ("  bootstrap part only", bf),
                  ("realized G", Gf)):
        print(f"  {nm:28s} {ev(x, Gf):>17.3f} {x.std():>9.4f}")
    # THE overfitting measure: how well does V predict its OWN TRAINING TARGET
    # on freshly-collected states? Compare against the in-batch value from the
    # log (1 - vtrace_value_loss / v_target_std^2) to separate generalization
    # from the choice of target.
    print(f"\n  EV(V, v_target) on FRESH states = {ev(Vf, tgt):.3f}   <- vs in-batch from log")
    print(f"  EV(V, G)        on FRESH states = {ev(Vf, Gf):.3f}")
    print(f"  bootstrap share of target variance: {bf.var()/max(tgt.var(),1e-30):.2f}")
    print(f"MACHINE {'egozero' if a.ego_zero else 'egosample'} {d.get('num_timesteps')} {ev(Vf,tgt):.4f} {ev(Vf,Gf):.4f} {ev(tgt,Gf):.4f} {int(msk.sum())}")


if __name__ == "__main__":
    main()
