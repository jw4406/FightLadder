"""Linear-probe the actor's and critic's CNN encoders for game state.

Motivation: the critic measures explained_variance ~ 0 against realized returns,
while a single hand-crafted scalar (HP differential) reaches EV ~ +0.22 on the
same samples. So the information exists -- the question is whether the CRITIC'S
encoder extracts it.

SPAR sets share_features_extractor=False (clean_new_policies.py:53), flipping the
SB3 1.7 default of True. So `vf_features_extractor` is a separate NatureCNN whose
only gradient comes from the value loss, while `pi_ctrl_features_extractor` is an
alias of the shared trunk that also receives the (much larger) policy gradient.

This probe freezes the checkpoint, pushes the same observations through both
encoders, and fits a ridge regression from each feature vector to game-state
targets read out of `info`. Nothing is trained beyond the linear head, so the
result is purely "is this information linearly decodable from these features".

Reading the result:
  actor decodes HP, critic doesn't  -> the critic's encoder never learned to see
  neither decodes HP                -> the observation encoding is inadequate
  both decode HP well              -> features are fine; failure is head/target
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

TARGETS = ("hp_diff", "own_hp", "opp_hp", "agent_x", "enemy_x", "x_gap", "countdown")


def collect(venv, ops, n_steps, seed):
    """Roll pi and record (actor features, critic features, state targets)."""
    rng = np.random.RandomState(seed)
    A, C, Y = [], [], []
    obs = venv.reset()
    for t in range(n_steps):
        a_e = ops.sample_ego(obs, rng)
        a_a = ops.sample_adv(obs, rng)
        lbr_a, pol_a = (a_a, a_e) if ops.lbr_is_adv else (a_e, a_a)
        obs2, r_l, r_r, d, infos = venv.step(ops.joint(lbr_a, pol_a))

        with th.no_grad():
            x = preprocess_obs(th.as_tensor(obs).to(ops.device),
                               ops.p.observation_space,
                               normalize_images=ops.p.normalize_images)
            f_actor = ops.p.pi_ctrl_features_extractor(x).cpu().numpy()
            f_critic = ops.p.vf_features_extractor(x).cpu().numpy()

        ahp = np.array([i.get("agent_hp", 0) for i in infos], float)
        ehp = np.array([i.get("enemy_hp", 0) for i in infos], float)
        ax = np.array([i.get("agent_x", 0) for i in infos], float)
        ex = np.array([i.get("enemy_x", 0) for i in infos], float)
        cd = np.array([i.get("round_countdown", 0) for i in infos], float)
        own, opp = (ehp, ahp) if ops.lbr_is_adv else (ahp, ehp)
        Y.append(np.stack([own - opp, own, opp, ax, ex, ax - ex, cd], axis=1))
        A.append(f_actor); C.append(f_critic)

        obs = obs2
        if np.any(d):
            obs = venv.reset()
        if (t + 1) % 50 == 0:
            print(f"   collected {(t+1)*venv.num_envs} samples", flush=True)
    return np.concatenate(A), np.concatenate(C), np.concatenate(Y)


def probe(F, Y, seed=0, ridge=1.0):
    """Ridge regression from features to each target; out-of-sample R^2."""
    n = F.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    # standardize features on train
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    Xtr = np.concatenate([(F[tr] - mu) / sd, np.ones((tr.size, 1))], 1)
    Xte = np.concatenate([(F[te] - mu) / sd, np.ones((te.size, 1))], 1)
    d = Xtr.shape[1]
    G = Xtr.T @ Xtr + ridge * np.eye(d)
    out = {}
    for j, name in enumerate(TARGETS):
        y = Y[:, j]
        if y.std() < 1e-9:
            out[name] = float("nan"); continue
        w = np.linalg.solve(G, Xtr.T @ y[tr])
        pred = Xte @ w
        err = pred - y[te]
        out[name] = float(1.0 - err.var() / y[te].var()) if y[te].var() > 1e-12 else float("nan")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Linear probe of actor vs critic encoders")
    ap.add_argument("--main_checkpoint_model_path", type=str, required=True)
    ap.add_argument("--eval_prot", type=str, default="True")
    ap.add_argument("--lbr_matchups", type=str, default="all")
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="feature_probe.json")
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
            shared = ops.p.vf_features_extractor is ops.p.pi_ctrl_features_extractor
            print(f"   share_features_extractor: {shared}"
                  f"   (SB3 1.7 default is True; SPAR sets False)")
            t0 = time.time()
            A, C, Y = collect(venv, ops, args.steps, args.seed)
        finally:
            venv.close()

        pa = probe(A, Y, args.seed, args.ridge)
        pc = probe(C, Y, args.seed, args.ridge)
        out["matchups"][label] = {"actor": pa, "critic": pc, "n": int(A.shape[0]),
                                  "feat_dim": int(A.shape[1]), "shared": bool(shared),
                                  "wall_clock_s": round(time.time() - t0, 1)}
        print(f"\n   n={A.shape[0]} samples, feature dim={A.shape[1]}")
        print(f"   {'target':12s} {'ACTOR cnn':>11s} {'CRITIC cnn':>11s} {'gap':>9s}")
        for k in TARGETS:
            g = pa[k] - pc[k]
            print(f"   {k:12s} {pa[k]:11.3f} {pc[k]:11.3f} {g:+9.3f}")
        print("   (out-of-sample R^2 of a ridge readout; 0 = no better than the mean)")

    p = os.path.join(REPO_ROOT, args.out)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
