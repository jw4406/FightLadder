"""Is the value HEAD discarding information its own features already contain?

Established so far:
  - the critic's encoder DOES learn: vf_features decode hp_diff at R^2 0.53-0.66
  - the trained value head still scores EV ~ 0 against realized returns

So the failure is downstream of the features. This probe fits a linear readout
from the critic's OWN features to the realized return, on the same samples the
trained head is scored on. If the linear readout beats the trained head, the head
(or the target it was fit to) is the problem -- not the representation.

Returns are computed the cheap way: roll pi continuously, record everything, then
accumulate G_t = r_t + gamma*(1-done_t)*G_{t+1} backwards per env. No snapshots,
no per-sample rollouts -- every timestep becomes a training sample. Only samples
whose episode actually terminated inside the recorded window are kept, so no
truncated returns contaminate the fit.

Compared, all on identical samples and an identical train/test split:
  trained_value_head   what the critic actually predicts
  ridge(vf_features)   best linear readout of the CRITIC's own features
  ridge(pi_features)   same for the ACTOR's features (upper reference)
  hp_diff_linear       one hand-crafted scalar
  state_linear         hp/x/countdown together
"""
import os
import sys


def _peek(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


_d = _peek(sys.argv[1:])
if _d is not None and str(_d).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import time
import argparse
import numpy as np
import torch as th

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.preprocessing import preprocess_obs
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups, REPO_ROOT, _b)


def collect(venv, ops, n_steps, seed):
    rng = np.random.RandomState(seed)
    n = venv.num_envs
    VF, LV, PI, V, R, D, S = [], [], [], [], [], [], []
    # Episode id per env slot, so the train/test split can be made BY EPISODE.
    # Without this the probe splits by timestep and leaks -- see episode_split().
    EP = []
    ep_id = np.arange(n)
    next_ep = n
    obs = venv.reset()
    for t in range(n_steps):
        EP.append(ep_id.copy())
        with th.no_grad():
            x = preprocess_obs(th.as_tensor(obs).to(ops.device),
                               ops.p.observation_space,
                               normalize_images=ops.p.normalize_images)
            vf = ops.p.vf_features_extractor(x)          # stage [1] CNN  -> 512
            VF.append(vf.cpu().numpy())
            # stage [2] mlp_extractor critic trunk -> 256 ("latent_vf").
            # Probing here separates the trunk from the per-matchup ModuleDict
            # head [3], which value_forward applies next.
            LV.append(ops.p.mlp_extractor.forward_critic(vf).cpu().numpy())
            PI.append(ops.p.pi_ctrl_features_extractor(x).cpu().numpy())
        V.append(ops.values_ego(obs) * ops.sgn)

        a_e = ops.sample_ego(obs, rng)
        a_a = ops.sample_adv(obs, rng)
        lbr_a, pol_a = (a_a, a_e) if ops.lbr_is_adv else (a_e, a_a)
        obs, r_l, r_r, d, infos = venv.step(ops.joint(lbr_a, pol_a))

        R.append(ops.lbr_reward(r_l, r_r))
        d = np.asarray(d, dtype=bool)
        D.append(d)
        for j in np.nonzero(d)[0]:
            ep_id[j] = next_ep
            next_ep += 1
        ahp = np.array([i.get("agent_hp", 0) for i in infos], float)
        ehp = np.array([i.get("enemy_hp", 0) for i in infos], float)
        ax = np.array([i.get("agent_x", 0) for i in infos], float)
        ex = np.array([i.get("enemy_x", 0) for i in infos], float)
        cd = np.array([i.get("round_countdown", 0) for i in infos], float)
        own, opp = (ehp, ahp) if ops.lbr_is_adv else (ahp, ehp)
        S.append(np.stack([own - opp, own, opp, ax - ex, cd], axis=1))
        if (t + 1) % 100 == 0:
            print(f"   {(t+1)*n} steps", flush=True)

    R = np.array(R); D = np.array(D)                     # (T, n)
    T = R.shape[0]
    G = np.zeros_like(R)
    valid = np.zeros_like(D)
    # Backward accumulation per env; `seen_done` marks samples whose episode
    # actually finished inside the window, so no truncated return is used.
    acc = np.zeros(R.shape[1]); seen = np.zeros(R.shape[1], dtype=bool)
    for t in reversed(range(T)):
        acc = R[t] + ops.gamma * acc * (~D[t])
        seen = seen | D[t]
        G[t] = acc
        valid[t] = seen
    flat = lambda A: np.concatenate(A, axis=0)
    m = valid.reshape(-1)
    return (flat(VF)[m], flat(LV)[m], flat(PI)[m], np.array(V).reshape(-1)[m],
            G.reshape(-1)[m], flat(S)[m], np.array(EP).reshape(-1)[m])


def _ev(pred, y):
    return float(1.0 - (pred - y).var() / y.var()) if y.var() > 1e-12 else float("nan")


def episode_split(ep, seed=0, fracs=(0.5, 0.25, 0.25)):
    """Split sample indices by EPISODE id. Returns (train, val, test).

    Canonical implementation -- critic_diagnostics.py and critic_ceiling.py both
    use this one, so the three scripts cannot drift on a detail this subtle.

    A random TIMESTEP split leaks badly against return targets. Every timestep
    inside one episode shares nearly the same discounted return, so a timestep
    split puts near-duplicate targets on both sides and any probe can recover G
    just by recognising which episode a state came from. Measured here on
    spar_Ry_Sa_2880000 with a frozen actor-CNN ridge: +0.166 under a timestep
    split versus +0.005 under an episode split -- ~95% of the apparent signal
    was leakage, and it produced two wrong architectural conclusions before it
    was caught.

    val exists so a ridge alpha can be picked without touching test.
    """
    uniq = np.unique(ep)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n1 = int(len(uniq) * fracs[0])
    n2 = n1 + int(len(uniq) * fracs[1])
    groups = (set(uniq[:n1]), set(uniq[n1:n2]), set(uniq[n2:]))
    return tuple(np.nonzero(np.isin(ep, list(g)))[0] for g in groups)


def ridge_ev(X, y, tr, te, ridge):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    A = np.concatenate([(X[tr] - mu) / sd, np.ones((tr.size, 1))], 1)
    B = np.concatenate([(X[te] - mu) / sd, np.ones((te.size, 1))], 1)
    w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y[tr])
    return _ev(B @ w, y[te])


def mlp_ev(X, y, tr, te, seed=0, hidden=256, epochs=400, lr=1e-3, wd=1e-3,
           device="cuda", batch=512, report=False):
    """Non-linear probe with capacity comparable to value_net[matchup] (4x256).

    A low ridge score does NOT prove information was destroyed -- only that it is
    not in the LINEAR span. If an MLP of comparable capacity recovers it, the
    stage merely rotated the signal and the "information lost" framing is wrong.
    """
    th.manual_seed(seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr = th.tensor((X[tr] - mu) / sd, dtype=th.float32, device=device)
    Xte = th.tensor((X[te] - mu) / sd, dtype=th.float32, device=device)
    ym, ys = y[tr].mean(), y[tr].std() + 1e-12
    ytr = th.tensor((y[tr] - ym) / ys, dtype=th.float32, device=device).unsqueeze(1)
    net = th.nn.Sequential(
        th.nn.Linear(X.shape[1], hidden), th.nn.ReLU(),
        th.nn.Linear(hidden, hidden), th.nn.ReLU(),
        th.nn.Linear(hidden, hidden), th.nn.ReLU(),
        th.nn.Linear(hidden, 1)).to(device)
    opt = th.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    N = Xtr.shape[0]
    g = th.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = th.randperm(N, generator=g).to(Xtr.device)
        for i in range(0, N, batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            th.nn.functional.mse_loss(net(Xtr[j]), ytr[j]).backward()
            opt.step()
        if report and (ep + 1) % max(1, epochs // 4) == 0:
            with th.no_grad():
                tr_ev = 1 - (net(Xtr).squeeze(1) - ytr.squeeze(1)).var().item()
            print(f"      ep{ep+1:5d}  train_EV(std units)={tr_ev:+.3f}", flush=True)
    with th.no_grad():
        pred = net(Xte).squeeze(1).cpu().numpy() * ys + ym
    return _ev(pred, y[te])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Value-head vs its own features")
    ap.add_argument("--main_checkpoint_model_path", type=str, required=True)
    ap.add_argument("--eval_prot", type=str, default="True")
    ap.add_argument("--lbr_matchups", type=str, default="all")
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="head_probe.json")
    args = ap.parse_args(argv)

    data = load_from_zip_file(args.main_checkpoint_model_path, device="cpu")[0]
    out = {"checkpoint": os.path.basename(args.main_checkpoint_model_path),
           "num_timesteps": data.get("num_timesteps"), "matchups": {}}

    for head_idx, label, state in resolve_matchups(data, args.lbr_matchups):
        print(f"\n=== {label} (head {head_idx}) ===")
        venv = build_lbr_venv(state, args.n_envs)
        try:
            model, _ = load_checkpoint(args.main_checkpoint_model_path, venv, args.device)
            preflight(venv, model)
            ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=_b(args.eval_prot))
            t0 = time.time()
            VF, LV, PI, V, G, S, EP = collect(venv, ops, args.steps, args.seed)
        finally:
            venv.close()

        n = G.size
        # Split BY EPISODE, not by timestep. fracs=(0.5, 0.0, 0.5) keeps this
        # script's original 50/50 train/test proportions; the leak, not the
        # proportion, was the bug. (val is unused here because alpha is a fixed
        # --ridge argument rather than cross-validated.)
        tr, _unused_val, te = episode_split(EP, seed=args.seed, fracs=(0.5, 0.0, 0.5))
        n_ep_tr = int(np.unique(EP[tr]).size) if tr.size else 0
        n_ep_te = int(np.unique(EP[te]).size) if te.size else 0
        res = {
            "s1_cnn_512": ridge_ev(VF, G, tr, te, args.ridge),
            "s2_mlp_trunk_256": ridge_ev(LV, G, tr, te, args.ridge),
            "s3_full_trained_V": _ev(V[te], G[te]),
            "trained_value_head": _ev(V[te], G[te]),
            "ridge_vf_features": ridge_ev(VF, G, tr, te, args.ridge),
            "ridge_pi_features": ridge_ev(PI, G, tr, te, args.ridge),
            "hp_diff_linear": ridge_ev(S[:, :1], G, tr, te, args.ridge),
            "state_linear": ridge_ev(S, G, tr, te, args.ridge),
            "s1_cnn_mlp": mlp_ev(VF, G, tr, te, args.seed, device=args.device),
            "s2_trunk_mlp": mlp_ev(LV, G, tr, te, args.seed, device=args.device),
            "n": int(n), "G_std": float(G.std()), "V_std": float(V.std()),
            "split": "by-episode",
            "n_episodes_train": n_ep_tr, "n_episodes_test": n_ep_te,
            "wall_clock_s": round(time.time() - t0, 1),
        }
        out["matchups"][label] = res
        print(f"\n   n={n} samples with complete returns,  G std={G.std():.4f},  V std={V.std():.4f}")
        print(f"   split BY EPISODE: {n_ep_tr} train / {n_ep_te} test episodes")
        print(f"   {'stage (linear readout of its output)':38s} {'EV':>9s}   {'delta':>8s}")
        chain = [("[1] vf_features_extractor CNN 512", res["s1_cnn_512"]),
                 ("[2] + mlp_extractor trunk    256", res["s2_mlp_trunk_256"]),
                 ("[3] + value_net[matchup] -> V     ", res["s3_full_trained_V"])]
        prev = None
        for nm, val in chain:
            d = "" if prev is None else f"{val - prev:+8.3f}"
            print(f"   {nm:38s} {val:9.3f}   {d:>8s}")
            prev = val
        print()
        print(f"   {'NON-LINEAR probe (4x256 MLP, same capacity as value_net)':52s}")
        print(f"   {'[1] CNN 512 -> MLP':38s} {res['s1_cnn_mlp']:9.3f}   "
              f"(ridge {res['s1_cnn_512']:+.3f})")
        print(f"   {'[2] trunk 256 -> MLP':38s} {res['s2_trunk_mlp']:9.3f}   "
              f"(ridge {res['s2_mlp_trunk_256']:+.3f})")
        print()
        print(f"   {'-- reference: actor CNN':38s} {res['ridge_pi_features']:9.3f}")
        print(f"   {'-- reference: hand state features':38s} {res['state_linear']:9.3f}")
        d12 = res["s1_cnn_512"] - res["s2_mlp_trunk_256"]
        d23 = res["s2_mlp_trunk_256"] - res["s3_full_trained_V"]
        print()
        print(f"   information lost in mlp_extractor trunk [2]: {d12:+.3f}")
        print(f"   information lost in value_net head    [3]: {d23:+.3f}")
        print(f"   -> culprit: {'[2] mlp_extractor trunk' if d12 > d23 + 0.05 else ('[3] value_net ModuleDict head' if d23 > d12 + 0.05 else 'both / neither dominant')}")

    p = os.path.join(REPO_ROOT, args.out)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
