"""Control for the 'trunk bottleneck' claim: is the TRAINED 512->256 map worse
than a RANDOM one of identical shape? Without this, d12<0 may be nothing but the
cost of halving the dimension."""
import sys, argparse, numpy as np, torch as th
sys.path.insert(0, "/home/jw4406/codebase/FightLadder/main")
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.preprocessing import preprocess_obs
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups, _b)
from lbr_head_probe import ridge_ev, mlp_ev, _ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    hi, lab, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs)
    try:
        m, _ = load_checkpoint(a.ckpt, venv, "cuda"); preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        rng = np.random.RandomState(a.seed)
        VF, TR, V, R, D = [], [], [], [], []
        obs = venv.reset()
        for t in range(a.steps):
            with th.no_grad():
                x = preprocess_obs(th.as_tensor(obs).to(ops.device), ops.p.observation_space,
                                   normalize_images=ops.p.normalize_images)
                vf = ops.p.vf_features_extractor(x)
                VF.append(vf.cpu().numpy())
                TR.append(ops.p.mlp_extractor.forward_critic(vf).cpu().numpy())
            ae, ad = ops.sample_ego(obs, rng), ops.sample_adv(obs, rng)
            la, pa = (ad, ae) if ops.lbr_is_adv else (ae, ad)
            obs, rl, rr, d, infos = venv.step(ops.joint(la, pa))
            R.append(ops.lbr_reward(rl, rr)); D.append(np.asarray(d, bool))
    finally:
        venv.close()

    R = np.array(R); D = np.array(D); G = np.zeros_like(R); valid = np.zeros_like(D)
    acc = np.zeros(R.shape[1]); seen = np.zeros(R.shape[1], bool)
    for t in reversed(range(R.shape[0])):
        acc = R[t] + ops.gamma * acc * (~D[t]); seen |= D[t]; G[t] = acc; valid[t] = seen
    msk = valid.reshape(-1)
    VF = np.concatenate(VF)[msk]; TR = np.concatenate(TR)[msk]; G = G.reshape(-1)[msk]

    # CONTROL: random 512->256 map, orthonormal rows, same output dim as the trunk
    rs = np.random.RandomState(1234)
    W = rs.randn(VF.shape[1], TR.shape[1]).astype(np.float32) / np.sqrt(VF.shape[1])
    RP = np.tanh(VF @ W)          # random projection + same nonlinearity family

    n = G.size; idx = np.random.RandomState(a.seed).permutation(n)
    tr, te = idx[:n//2], idx[n//2:]
    print(f"\n  {lab}  n={n}")
    print(f"  {'representation':26s} {'ridge':>8s} {'MLP':>8s}")
    out = {}
    for nm, X in (("[1] vf_features 512", VF), ("[2] TRAINED trunk 256", TR),
                  ("[2c] RANDOM proj 256", RP)):
        r = ridge_ev(X, G, tr, te, 10.0); mm = mlp_ev(X, G, tr, te, a.seed, device="cuda")
        out[nm] = (r, mm); print(f"  {nm:26s} {r:8.3f} {mm:8.3f}")
    d_tr = out["[2] TRAINED trunk 256"][0] - out["[1] vf_features 512"][0]
    d_rp = out["[2c] RANDOM proj 256"][0] - out["[1] vf_features 512"][0]
    print(f"\n  d12 trained trunk : {d_tr:+.3f}")
    print(f"  d12 random proj   : {d_rp:+.3f}")
    print(f"  trained - random  : {d_tr - d_rp:+.3f}  -> "
          f"{'training made it WORSE than random' if d_tr < d_rp - 0.03 else ('training HELPED vs random' if d_tr > d_rp + 0.03 else 'trained ~ random: drop is just dimensionality')}")


if __name__ == "__main__":
    main()
