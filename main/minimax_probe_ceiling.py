"""Can ANYTHING predict return-at-joint-action from the frozen critic latent?

WHY THIS EXISTS. On the 6.72M Phase 0 checkpoint, measured on the same states and
the same target:

    value_net  (V)   EV  +0.939     <- inflated by stalling, but it FITS
    minimax_net (Q)  EV  -2.648     <- worse than predicting the mean

Q is not hitting a task ceiling; it is failing where another head on the SAME
encoder succeeds. Two explanations remain and they need different fixes:

  (a) TRAINING BUG -- the representation supports it and the head's optimisation
      is broken (one cell per sample, a bad target/prediction pairing, a frame
      error in _minimax_q_update).
  (b) REPRESENTATION -- latent_vf, shaped entirely by V's objective and frozen
      under --minimax_stop_grad, genuinely cannot express action-conditional
      value, so no amount of head training would help.

A ridge on the frozen latent separates them. Same logic as the D1 supervised
ceiling: if a LINEAR probe beats the trained head, the information is present and
the head's training is at fault. If the probe also fails, the frozen
representation is the ceiling and stop_grad is the thing to reconsider.

Probes, in increasing capability:
    latent                    V-style: state only, no action information at all.
                              A ceiling on what ANY state-value head could do.
    latent + onehot(a_ego,a_adv)
                              adds an additive per-cell offset -- can it beat
                              `latent` at all? if not, the target carries no
                              action-conditional signal on this data.
    latent x onehot(joint)    per-cell slope: the full linear analogue of what
                              the matrix head is trying to learn.
Splits are BY EPISODE. A timestep split leaks: consecutive steps share almost
all of their return, and that inflated earlier probe scores ~5x in this project.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=9000,
                    help="vec-steps. Episodes here run ~500 steps, so this must "
                         "be large enough for >=100 to FINISH -- 3000 gave only "
                         "72 and tripped the underpowered guard.")
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="minimax_probe_ceiling.json")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, REPO_ROOT)
    from critic_ceiling import ridge_fit, episode_split
    from stable_baselines3.common.preprocessing import preprocess_obs

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]

    venv = build_lbr_venv(state, a.n_envs)
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        pol = ops.p
        gamma = a.gamma if a.gamma is not None else float(getattr(model, "gamma", 0.99))
        rng = np.random.RandomState(a.seed)
        n = venv.num_envs

        LAT, AE, AA, R, D, EP, QP, VP = [], [], [], [], [], [], [], []
        ep_id = np.arange(n); nxt = n
        obs = venv.reset()
        for t in range(a.steps):
            ob = th.as_tensor(obs).to(ops.device)
            with th.no_grad():
                x = preprocess_obs(ob, pol.observation_space,
                                   normalize_images=pol.normalize_images)
                lat = pol.mlp_extractor.forward_critic(pol.vf_features_extractor(x))
                LAT.append(lat.cpu().numpy())
                VP.append(ops.values_ego(obs))
            ae = ops.sample_ego(obs, rng); aa = ops.sample_adv(obs, rng)
            with th.no_grad():
                M = pol.minimax_matrices(ob, buf_num=[head_idx], stop_grad=True)
                b = th.arange(M.shape[0], device=M.device)
                QP.append(M[b, th.as_tensor(ae).to(M.device),
                            th.as_tensor(aa).to(M.device)].cpu().numpy())
            AE.append(ae.copy()); AA.append(aa.copy()); EP.append(ep_id.copy())
            obs, r_l, r_r, d, infos = venv.step(ops.joint(ae, aa))
            d = np.asarray(d, bool)
            R.append(ops.lbr_reward(r_l, r_r)); D.append(d)
            for j in np.nonzero(d)[0]:
                ep_id[j] = nxt; nxt += 1
            if (t + 1) % 500 == 0:
                print(f"   {(t+1)*n:,} samples, {int(np.asarray(D).sum())} episodes",
                      flush=True)
    finally:
        venv.close()

    R = np.asarray(R); D = np.asarray(D)
    T = R.shape[0]
    G = np.zeros_like(R); valid = np.zeros_like(D)
    acc = np.zeros(n); seen = np.zeros(n, bool)
    for t in reversed(range(T)):
        acc = R[t] + gamma * acc * (~D[t]); seen |= D[t]
        G[t] = acc; valid[t] = seen

    m = valid.reshape(-1)
    LAT = np.concatenate(LAT)[m]
    ae = np.concatenate(AE)[m]; aa = np.concatenate(AA)[m]
    g = G.reshape(-1)[m]; ep = np.concatenate(EP)[m]
    qp = np.concatenate(QP)[m]; vp = np.concatenate(VP)[m]
    n_ep = int(np.unique(ep).size)
    print(f"\n   {m.sum():,} samples from {n_ep} finished episodes")
    if n_ep < 100:
        raise SystemExit(f"only {n_ep} episodes (<100): underpowered, raise --steps")

    n_a = int(max(ae.max(), aa.max())) + 1
    joint = ae * n_a + aa
    oh_j = np.zeros((g.size, n_a * n_a), np.float32); oh_j[np.arange(g.size), joint] = 1
    oh_a = np.zeros((g.size, 2 * n_a), np.float32)
    oh_a[np.arange(g.size), ae] = 1; oh_a[np.arange(g.size), n_a + aa] = 1

    tr, va, te = episode_split(ep, seed=a.seed)
    def ev(pred, y):
        return float(1.0 - ((y - pred) ** 2).mean() / y.var())

    rows = {}
    for name, X in (("latent (state only)", LAT),
                    ("latent + onehot(a_ego,a_adv)", np.hstack([LAT, oh_a])),
                    ("latent + onehot(joint cell)", np.hstack([LAT, oh_j]))):
        sc, alpha = ridge_fit(X, g, tr, te, val=va)
        rows[name] = {"ev": sc, "alpha": alpha}
        print(f"   {name:<32} EV {sc:+.4f}  (alpha {alpha:g})")

    trained = {"trained value_net V": ev(vp[te], g[te]),
               "trained minimax Q": ev(qp[te], g[te])}
    res = {"checkpoint": os.path.basename(a.ckpt), "gamma": gamma,
           "n_samples": int(m.sum()), "n_episodes": n_ep,
           "steps_per_episode": float(D.size / max(1, int(D.sum()))),
           "probes": rows, "trained": trained}
    out = os.path.join(REPO_ROOT, a.out)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 72)
    print(f"PROBE CEILING  {res['checkpoint']}   {n_ep} episodes, "
          f"{res['steps_per_episode']:.0f} steps/ep")
    print("=" * 72)
    for k, v in rows.items():
        print(f"  {k:<32} {v['ev']:+.4f}")
    for k, v in trained.items():
        print(f"  {k:<32} {v:+.4f}")
    lin = rows["latent + onehot(joint cell)"]["ev"]
    if lin > trained["trained minimax Q"] + 0.05:
        print(f"\n  => a LINEAR probe on the frozen latent beats the trained Q head")
        print(f"     ({lin:+.4f} vs {trained['trained minimax Q']:+.4f}). The")
        print(f"     information is THERE; the head's training is at fault.")
    else:
        print(f"\n  => the linear probe does no better than the head. The frozen")
        print(f"     latent cannot express action-conditional value -- stop_grad")
        print(f"     (not the head's optimiser) is the thing to reconsider.")
    print(f"\n  wrote {out}")
    return res


if __name__ == "__main__":
    main()
