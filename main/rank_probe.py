"""Why would a TRAINED 512->512 map lose signal a RANDOM one preserves?

Not expressivity (it can represent identity) and not the activation (LeakyReLU is
sign-preserving). The remaining mechanical candidate is representational
COLLAPSE: the trunk is trained to minimise value loss against a bootstrapped
target, not to preserve return-predictive information. If that target is poor,
the trunk is free to discard everything the head doesn't use -- which shows up as
a drop in effective rank.
"""
import sys, argparse, numpy as np, torch as th
sys.path.insert(0, "/home/jw4406/codebase/FightLadder/main")
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.preprocessing import preprocess_obs
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups)


def eff_rank(X):
    """Participation ratio of covariance eigenvalues: (sum l)^2 / sum(l^2).
    = number of directions actually carrying variance."""
    C = np.cov(X - X.mean(0), rowvar=False)
    l = np.linalg.eigvalsh(C); l = np.clip(l, 0, None)
    pr = (l.sum() ** 2) / max((l ** 2).sum(), 1e-30)
    tot = l.sum(); cs = np.cumsum(np.sort(l)[::-1]) / max(tot, 1e-30)
    return pr, int(np.searchsorted(cs, 0.95) + 1), int(np.searchsorted(cs, 0.99) + 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--n_envs", type=int, default=8)
    a = ap.parse_args()
    d = load_from_zip_file(a.ckpt, device="cpu")[0]
    hi, lab, state = resolve_matchups(d, "all")[0]
    venv = build_lbr_venv(state, a.n_envs)
    try:
        m, _ = load_checkpoint(a.ckpt, venv, "cuda"); preflight(venv, m)
        ops = PolicyOps(m, head_idx=hi, lbr_is_adv=True)
        rng = np.random.RandomState(0); VF, LV = [], []
        obs = venv.reset()
        for _ in range(a.steps):
            with th.no_grad():
                x = preprocess_obs(th.as_tensor(obs).to(ops.device), ops.p.observation_space,
                                   normalize_images=ops.p.normalize_images)
                vf = ops.p.vf_features_extractor(x)
                VF.append(vf.cpu().numpy())
                LV.append(ops.p.mlp_extractor.forward_critic(vf).cpu().numpy())
            ae, ad = ops.sample_ego(obs, rng), ops.sample_adv(obs, rng)
            obs, *_rest = venv.step(ops.joint(ad, ae))
            if np.any(_rest[2]): obs = venv.reset()
    finally:
        venv.close()
    VF = np.concatenate(VF); LV = np.concatenate(LV)
    rs = np.random.RandomState(1234)
    W = rs.randn(VF.shape[1], LV.shape[1]).astype(np.float32) / np.sqrt(VF.shape[1])
    RP = np.maximum(VF @ W, 0.01 * (VF @ W))     # LeakyReLU, matched to the trunk

    print(f"\n  {lab}   n={VF.shape[0]}   steps={d.get('num_timesteps')}")
    print(f"  {'representation':22s} {'dim':>5s} {'dead%':>7s} {'std':>8s} {'effRank':>8s} {'d95':>5s} {'d99':>5s}")
    for nm, X in (("vf_features (CNN)", VF), ("latent_vf (TRAINED)", LV), ("random proj (ctrl)", RP)):
        dead = 100.0 * (X.std(0) < 1e-6).mean()
        pr, d95, d99 = eff_rank(X)
        print(f"  {nm:22s} {X.shape[1]:>5d} {dead:>7.1f} {X.std():>8.4f} {pr:>8.1f} {d95:>5d} {d99:>5d}")


if __name__ == "__main__":
    main()
