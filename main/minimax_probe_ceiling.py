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


def _extract(pol, x, site, head_idx=0):
    """Representation at one of six depths, all on the SAME preprocessed obs.

    The actor and critic do NOT share a CNN here (share_features_extractor=False),
    so conv_pi is a genuinely different network from conv_vf -- trained by the
    policy gradient, never by V's scalar objective. That is the whole reason to
    look at it.

    minimax_trunk / value_trunk are the two HEAD trunks, both fed the identical
    latent_vf. They answer whether a trunk PRESERVES or DESTROYS the state
    information it was handed: latent_vf scores EV ~0.29 on returns, so a trunk
    scoring near 0.29 is passing it through and a trunk scoring ~0 is throwing it
    away. The minimax trunk was measured to be constant-dominated (DC/fluct 4.33
    vs the value trunk's 1.08) with its variance smeared over 299 weak units
    instead of concentrated in 10, so this is the direct test of whether that
    costs it the signal.
    """
    import torch as th
    if site == "conv_vf":
        return pol.vf_features_extractor(x)
    if site == "conv_pi":
        return pol.pi_features_extractor(x)
    if site == "latent_pi":
        return pol.mlp_extractor.ego_forward(pol.pi_features_extractor(x))
    lat = pol.mlp_extractor.forward_critic(pol.vf_features_extractor(x))
    if site == "minimax_trunk":
        return pol.minimax_net[list(pol.minimax_net.keys())[head_idx]].trunk(lat)
    if site == "value_trunk":
        vh = pol.value_net[list(pol.value_net.keys())[head_idx]]
        mods = list(vh.children())
        while mods and not isinstance(mods[-1], th.nn.Linear):
            mods.pop()
        return th.nn.Sequential(*mods[:-1])(lat)   # drop the scalar output layer
    return lat


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
    ap.add_argument("--target", choices=("return", "reward"), default="return",
                    help="'return' = discounted return (the critic's actual "
                         "target). 'reward' = the ONE-STEP reward r(a,o), which "
                         "ISOLATES the action effect. At gamma=0.94 the horizon "
                         "is ~16.7 steps, so a single action's direct "
                         "contribution is ~1/17th of the return's variance -- "
                         "diluted before it can be measured. greedy proves r is "
                         "action-dependent and decision-relevant (it beats "
                         "V-based lbr 3/3 using r alone), so if latent+action "
                         "cannot predict r either, that is a REPRESENTATION "
                         "failure, not dilution.")
    ap.add_argument("--uniform_actions", action="store_true",
                    help="Sample BOTH seats uniformly instead of from the "
                         "policies. The policy-sampled series is underpowered on "
                         "action diversity at late checkpoints -- if the ego "
                         "plays one action most of the time, the ridge sees few "
                         "examples of the others and a real action effect could "
                         "hide. Uniform sampling removes that limitation. It also "
                         "shifts the STATE distribution off-policy, which is the "
                         "cost.")
    ap.add_argument("--probe_site",
                    choices=("latent_vf", "conv_vf", "conv_pi", "latent_pi",
                             "minimax_trunk", "value_trunk"),
                    default="latent_vf",
                    help="WHERE to read the representation. The null result so "
                         "far is a fact about latent_vf ONLY, and that is the "
                         "worst possible place to look: under "
                         "--minimax_stop_grad the critic encoder receives "
                         "gradient solely from V's objective, which predicts a "
                         "SCALAR, so action-conditional structure has no reason "
                         "to survive to it. MinimaxHead reads latent_vf and "
                         "nothing else, so it can only see what got through.\n"
                         "  conv_vf   critic CNN output, before the value MLP\n"
                         "  conv_pi   actor CNN output -- a SEPARATE network "
                         "(share_features_extractor=False), trained by the "
                         "policy gradient rather than by V\n"
                         "  latent_pi actor latent; the policy must represent "
                         "action-relevant structure to have a policy at all, so "
                         "this is where the information is most likely to be\n"
                         "Gain at conv/latent_pi but not latent_vf means the "
                         "information EXISTS and the architecture discards it "
                         "before the head can use it -- which is an input "
                         "problem, not a capacity problem.")
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
                lat = _extract(pol, x, a.probe_site, head_idx)
                LAT.append(lat.reshape(lat.shape[0], -1).cpu().numpy())
                VP.append(ops.values_ego(obs))
            if a.uniform_actions:
                ae = rng.randint(0, ops.n_actions, size=n)
                aa = rng.randint(0, ops.n_actions, size=n)
            else:
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

    if a.target == "reward":
        # One-step reward. Valid everywhere -- no episode needs to finish, since
        # r is observed immediately. That also multiplies the usable sample count.
        G = R.copy()
        valid = np.ones_like(D)
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
    res = {"checkpoint": os.path.basename(a.ckpt), "target": a.target,
           "probe_site": a.probe_site, "latent_dim": int(LAT.shape[1]),
           "uniform_actions": bool(a.uniform_actions), "gamma": gamma,
           "n_samples": int(m.sum()), "n_episodes": n_ep,
           "steps_per_episode": float(D.size / max(1, int(D.sum()))),
           "probes": rows, "trained": trained}
    out = os.path.join(REPO_ROOT, a.out)
    print("\n" + "=" * 72)
    print(f"PROBE CEILING  {res['checkpoint']}   target={a.target}   "
          f"site={a.probe_site} (dim {LAT.shape[1]})"
          f"{'  UNIFORM-ACTIONS' if a.uniform_actions else '  policy-actions'}   "
          f"{n_ep} episodes, {res['steps_per_episode']:.0f} steps/ep")
    print("=" * 72)
    for k, v in rows.items():
        print(f"  {k:<32} {v['ev']:+.4f}")
    for k, v in trained.items():
        print(f"  {k:<32} {v:+.4f}")
    # THE comparison is state-only vs state+action. An earlier version of this
    # verdict compared the linear probe against the TRAINED head, which answered
    # "is the head's training at fault" -- the right question when the head was
    # the suspect, and the wrong one now. Handing the action to a ridge
    # explicitly is the most favourable possible test of whether
    # action-conditional value exists in this representation at all.
    base = rows["latent (state only)"]["ev"]
    lin = rows["latent + onehot(joint cell)"]["ev"]
    add = rows["latent + onehot(a_ego,a_adv)"]["ev"]
    gain = lin - base
    res["action_gain"] = float(gain)
    print(f"\n  ACTION GAIN  (state+joint) - (state only) = {gain:+.4f}")
    if gain > 0.02:
        res["verdict"] = "ACTION HELPS"
        print(f"     => action-conditional value EXISTS in this representation.")
        print(f"        The joint-action hypothesis is live; the gate is worth running.")
    else:
        res["verdict"] = "ACTION ADDS NOTHING"
        print(f"     => handing the action to a ridge EXPLICITLY buys nothing")
        print(f"        ({base:+.4f} -> {add:+.4f} -> {lin:+.4f}). Q(s,a,o) cannot")
        print(f"        beat V(s) at what it is fit to, so the gate would be")
        print(f"        measuring a mechanism that is not there.")
    print(f"\n  (for reference: trained V {trained['trained value_net V']:+.4f}, "
          f"trained Q {trained['trained minimax Q']:+.4f})")
    # Dump LAST. This used to run before action_gain/verdict were assigned, so
    # every sidecar ever written by this script lacked the two fields the whole
    # experiment turns on, and any downstream reader got a KeyError.
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {out}")
    return res


if __name__ == "__main__":
    main()
