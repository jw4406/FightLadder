"""WHY does Linear(512,484) produce a state-INDEPENDENT offset between two
functionally identical actions?

The head is  Q(s,i,j) = w_ij . h(s) + b_ij  where h(s) is the 512-d trunk output.
So for the byte-identical no-op pair (0 vs 9):

    delta_j(s) = (w_0j - w_9j) . h(s) + (b_0j - b_9j)
               = dw_j . h(s) + db_j

Split over states, with h(s) = hbar + dh(s):

    mean_s delta_j = dw_j . hbar + db_j        <- the CONSTANT component
    std_s  delta_j = std_s( dw_j . dh(s) )     <- the STATE-VARYING component

That algebra makes the measured 99%-constant result mechanical rather than
mysterious, IF the activations are DC-dominated: post-LeakyReLU features have a
large common mean hbar and comparatively small fluctuation dh. Then ANY weight
difference dw is amplified by ||hbar|| into a big constant, while contributing to
the varying part only through the much smaller ||dh||.

This script measures that ratio directly, and splits the constant component into
its weight term (dw.hbar) and its bias term (db) -- which says whether the offset
lives in the weights or the biases.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NOOP_A, NOOP_B = 0, 9


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--n_envs", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ram_mask", type=str, default="",
                    help="RAM byte-index .npy, required when the checkpoint was "
                         "trained with a MASKED ram observation: the checkpoint "
                         "records the WIDTH, not which bytes.")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from stable_baselines3.common.preprocessing import preprocess_obs
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, infer_obs_kwargs)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs, **infer_obs_kwargs(data, (getattr(a, 'ram_mask', '') or None)))
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        pol = ops.p
        head = pol.minimax_net[list(pol.minimax_net.keys())[head_idx]]

        rng = np.random.RandomState(0)
        H, AE = [], []
        obs = venv.reset()
        for t in range(a.steps):
            ob = th.as_tensor(obs).to(ops.device)
            with th.no_grad():
                x = preprocess_obs(ob, pol.observation_space,
                                   normalize_images=pol.normalize_images)
                lat = pol.mlp_extractor.forward_critic(pol.vf_features_extractor(x))
                H.append(head.trunk(lat).cpu().numpy())      # trunk output h(s)
            ae = ops.sample_ego(obs, rng)
            AE.append(ae.copy())
            obs = venv.step(ops.joint(ae, ops.sample_adv(obs, rng)))[0]
    finally:
        venv.close()

    H = np.concatenate(H)                                    # (N, 512)
    AE = np.concatenate(AE)
    hbar = H.mean(0)
    dH = H - hbar
    n_adv = head.n_adv

    W = head.out.weight.detach().cpu().numpy()               # (484, 512)
    B = head.out.bias.detach().cpu().numpy()                 # (484,)
    W = W.reshape(head.n_ego, n_adv, -1)
    B = B.reshape(head.n_ego, n_adv)

    dw = W[NOOP_A] - W[NOOP_B]                               # (n_adv, 512)
    db = B[NOOP_A] - B[NOOP_B]                               # (n_adv,)

    const_w = dw @ hbar                                      # (n_adv,)
    const_total = const_w + db
    vary = (dH @ dw.T).std(axis=0)                           # (n_adv,)

    print("\n" + "=" * 72)
    print(f"BIAS MECHANISM  {os.path.basename(a.ckpt)}   {H.shape[0]:,} states")
    print("=" * 72)
    print("  --- activation geometry (trunk output h(s), 512-d) ---")
    print(f"    ||hbar||                        {np.linalg.norm(hbar):>12.4f}")
    print(f"    mean_s ||h(s) - hbar||          {np.linalg.norm(dH, axis=1).mean():>12.4f}")
    print(f"    DC / fluctuation ratio          "
          f"{np.linalg.norm(hbar) / max(np.linalg.norm(dH, axis=1).mean(), 1e-12):>12.2f}")
    print(f"    frac of h coords with mean>0    {(hbar > 0).mean():>12.2%}")
    print("\n  --- decomposition of the no-op offset (mean over the 22 columns) ---")
    print(f"    |dw . hbar|   weight term       {np.abs(const_w).mean():>12.6f}")
    print(f"    |db|          bias term         {np.abs(db).mean():>12.6f}")
    print(f"    |constant total|                {np.abs(const_total).mean():>12.6f}")
    print(f"    state-varying std               {vary.mean():>12.6f}")
    print(f"\n    ||dw|| (mean over columns)      {np.linalg.norm(dw, axis=1).mean():>12.6f}")
    print(f"    ||w_0j|| (mean)                 {np.linalg.norm(W[NOOP_A], axis=1).mean():>12.6f}")
    print(f"    relative weight difference      "
          f"{np.linalg.norm(dw,axis=1).mean()/max(np.linalg.norm(W[NOOP_A],axis=1).mean(),1e-12):>12.2%}")

    # How different are the STATES in which each no-op is actually played?
    # If the policy picks them in different situations, each row is fit on a
    # different conditional distribution -- a selection effect, not drift.
    m0, m9 = AE == NOOP_A, AE == NOOP_B
    print("\n  --- selection: where does the policy PLAY each no-op? ---")
    print(f"    action 0 played {m0.sum():,} / {AE.size:,}   "
          f"action 9 played {m9.sum():,}")
    if m0.sum() > 30 and m9.sum() > 30:
        c0, c9 = H[m0].mean(0), H[m9].mean(0)
        sep = np.linalg.norm(c0 - c9)
        print(f"    ||E[h | a=0] - E[h | a=9]||     {sep:>12.4f}")
        print(f"    as frac of mean fluctuation     "
              f"{sep / max(np.linalg.norm(dH,axis=1).mean(),1e-12):>12.2%}")
        print(f"    predicted offset from selection "
              f"{np.abs(dw @ (c0 - c9)).mean():>12.6f}")
    else:
        print("    (one no-op is too rare under this policy to compare -- which is")
        print("     itself the point: the two rows see very different data)")


if __name__ == "__main__":
    main()
