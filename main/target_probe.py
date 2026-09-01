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
    ap.add_argument("--reward_scale", type=float, default=0.001,
                    help="MUST match the checkpoint's training reward_scale. The env "
                         "rewards (and thus realized G) scale by this, while V was "
                         "trained at the checkpoint's scale -- a mismatch makes V and G "
                         "differ by that ratio and EV(V,G) blows up (e.g. -6e5 when a "
                         "reward_scale=1.0 checkpoint is probed at the 0.001 default). "
                         "reward_scale is NOT stored in the checkpoint, so pass it.")
    ap.add_argument("--actionable_statuses", type=str, default="512,514,520",
                    help="split held-out EV by whether the ego was ACTIONABLE at the "
                         "state -- CONTROLS the decision-timing confound: if a "
                         "fixed-clock critic already scores much higher on actionable "
                         "states, decision-timing's EV lead is partly the easier eval "
                         "distribution, not a better critic.")
    a = ap.parse_args()
    _ACT = frozenset(int(x) for x in a.actionable_statuses.split(",") if x.strip())
    d = load_from_zip_file(a.ckpt, device="cpu")[0]
    T = a.T if a.T else int(d.get("vtrace_seq_len") or 64)
    hi, lab, state = resolve_matchups(d, "all")[0]
    venv = build_lbr_venv(state, a.n_envs, reward_scale=a.reward_scale)
    try:
        m, _ = load_checkpoint(a.ckpt, venv, "cuda"); preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        g = ops.gamma
        rng = np.random.RandomState(0); V, R, D = [], [], []
        obs = venv.reset()
        St = []; cur_stat = None            # ego agent_status ALIGNED with V[t]
        for _ in range(a.steps):
            V.append(ops.values_ego(obs) * ops.sgn)
            St.append(cur_stat if cur_stat is not None
                      else np.full(obs.shape[0], -1))
            ae = (np.zeros(obs.shape[0], dtype=np.int64) if a.ego_zero
                  else ops.sample_ego(obs, rng))
            ad = ops.sample_adv(obs, rng)
            obs, rl, rr, dn, infos = venv.step(ops.joint(ad, ae))
            cur_stat = np.array([int(i.get('agent_status', -1)) for i in infos])
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
    St = np.array(St)                                    # (S, n)
    msk = (valid & ok).reshape(-1)
    Vf, Gf = V.reshape(-1)[msk], G.reshape(-1)[msk]
    rf, bf = rew.reshape(-1)[msk], boot.reshape(-1)[msk]
    tgt = rf + bf
    Sf = St.reshape(-1)[msk]
    act = np.array([int(s) in _ACT for s in Sf])         # actionable at the state

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
    # CONFOUND CONTROL: same critic, same return def, split by actionability.
    na_, a_ = (~act).sum(), act.sum()
    print(f"\n  --- EV(V,G) split by actionability (controls the DT confound) ---")
    print(f"  ACTIONABLE  states: n={a_:>6}  EV={ev(Vf[act], Gf[act]) if a_>10 else float('nan'):+.3f}"
          f"  Gstd={Gf[act].std() if a_>0 else float('nan'):.2f}")
    print(f"  NON-action  states: n={na_:>6}  EV={ev(Vf[~act], Gf[~act]) if na_>10 else float('nan'):+.3f}"
          f"  Gstd={Gf[~act].std() if na_>0 else float('nan'):.2f}")
    print(f"  ALL         states: n={len(act):>6}  EV={ev(Vf, Gf):+.3f}")
    print(f"MACHINE_SPLIT act_n={int(a_)} act_ev={ev(Vf[act],Gf[act]) if a_>10 else float('nan'):.4f} "
          f"nonact_n={int(na_)} nonact_ev={ev(Vf[~act],Gf[~act]) if na_>10 else float('nan'):.4f} "
          f"all_ev={ev(Vf,Gf):.4f}")
    print(f"MACHINE {'egozero' if a.ego_zero else 'egosample'} {d.get('num_timesteps')} {ev(Vf,tgt):.4f} {ev(Vf,Gf):.4f} {ev(tgt,Gf):.4f} {int(msk.sum())}")


if __name__ == "__main__":
    main()
