"""CONTROL: is the dying trunk specific to the 484-output minimax head, or a
property of this whole stack?

The minimax head's trunk output has 99.8% of its 512 units at negative mean by
12.48M (LeakyReLU 0.01 region, attenuated 100x), so Q(s,i,j) collapses toward a
per-cell bias with almost no state dependence. That is the explanation for every
null we have measured -- but only if it is CAUSED by the joint-action design.

The value head is the perfect control. Architecturally IDENTICAL trunk
(3 x [Linear(512,512) + LeakyReLU]), reads the SAME latent_vf, trained on the
SAME rollouts by the same optimizer class -- and differs in exactly one respect:
1 output instead of 484. It also demonstrably works (it reaches its supervised
ceiling; a ridge on its latent scores EV 0.29).

  V's trunk ALSO dead  -> general property of the stack (tiny targets ~0.012,
                          AdamW decay 0.01). Says little about the minimax
                          design; the fix is optimizer/normalization.
  ONLY minimax dead    -> caused by the 484-output structure: 484 conflicting
                          gradients on one shared representation with a single
                          cell active per transition. The fix is the
                          parameterization (shared action embeddings / low rank).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n_envs", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="",
                    help="optional JSON path, so a sweep can compare checkpoints "
                         "machine-readably instead of regexing stdout")
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
        key = list(pol.minimax_net.keys())[head_idx]
        mm_trunk = pol.minimax_net[key].trunk

        # Value head: same Sequential shape, drop the final Linear(512,1) so the
        # comparison is trunk-output vs trunk-output.
        vhead = pol.value_net[list(pol.value_net.keys())[head_idx]]
        vmods = list(vhead.children())
        while vmods and not isinstance(vmods[-1], th.nn.Linear):
            vmods.pop()
        vmods = vmods[:-1]                       # drop the scalar output Linear
        v_trunk = th.nn.Sequential(*vmods)

        HM, HV = [], []
        rng = np.random.RandomState(0)
        obs = venv.reset()
        for t in range(a.steps):
            ob = th.as_tensor(obs).to(ops.device)
            with th.no_grad():
                x = preprocess_obs(ob, pol.observation_space,
                                   normalize_images=pol.normalize_images)
                lat = pol.mlp_extractor.forward_critic(pol.vf_features_extractor(x))
                HM.append(mm_trunk(lat).cpu().numpy())
                HV.append(v_trunk(lat).cpu().numpy())
                if t == 0:
                    print(f"   latent_vf {tuple(lat.shape)}  "
                          f"minimax trunk out {tuple(mm_trunk(lat).shape)}  "
                          f"value trunk out {tuple(v_trunk(lat).shape)}")
            obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                      ops.sample_adv(obs, rng)))[0]
    finally:
        venv.close()

    def stats(H, name):
        hbar = H.mean(0)
        dH = H - hbar
        fluct = np.linalg.norm(dH, axis=1).mean()
        return {
            "name": name, "dim": H.shape[1],
            "norm_hbar": float(np.linalg.norm(hbar)),
            "fluct": float(fluct),
            "dc_ratio": float(np.linalg.norm(hbar) / max(fluct, 1e-12)),
            "frac_mean_pos": float((hbar > 0).mean()),
            "frac_act_pos": float((H > 0).mean()),
            "per_unit_std_med": float(np.median(H.std(0))),
            # A healthy representation may carry its signal in a FEW strongly
            # varying directions rather than uniformly. Median per-unit std
            # cannot see that; variance concentration can.
            "per_unit_std_max": float(H.std(0).max()),
            "var_top10_share": float(np.sort(H.var(0))[::-1][:10].sum()
                                     / max(H.var(0).sum(), 1e-30)),
            "n_units_std_gt_1pct": int((H.std(0) > 0.01 * max(np.linalg.norm(hbar), 1e-12)).sum()),
        }

    HM = np.concatenate(HM); HV = np.concatenate(HV)
    sm, sv = stats(HM, "minimax trunk (484 out)"), stats(HV, "value trunk (1 out)")

    print("\n" + "=" * 76)
    print(f"TRUNK CONTROL  {os.path.basename(a.ckpt)}   {HM.shape[0]:,} states")
    print("=" * 76)
    print(f"  {'':30s} {'MINIMAX (484)':>18} {'VALUE (1)':>18}")
    for k, lbl in (("dim", "output dim"),
                   ("norm_hbar", "||hbar||"),
                   ("fluct", "mean ||h - hbar||"),
                   ("dc_ratio", "DC / fluctuation"),
                   ("frac_mean_pos", "% units w/ mean > 0"),
                   ("frac_act_pos", "% activations > 0"),
                   ("per_unit_std_med", "median per-unit std"),
                   ("per_unit_std_max", "MAX per-unit std"),
                   ("var_top10_share", "var share of top-10 units"),
                   ("n_units_std_gt_1pct", "units w/ std > 1% of |hbar|")):
        f = ("{:>18.2%}" if (k.startswith("frac") or k.endswith("share")) else
             ("{:>18d}" if (k == "dim" or k.startswith("n_units")) else "{:>18.5f}"))
        print(f"  {lbl:30s} " + f.format(sm[k]) + " " + f.format(sv[k]))

    if a.out:
        import json
        from local_best_response import REPO_ROOT
        _o = os.path.join(REPO_ROOT, a.out) if not os.path.isabs(a.out) else a.out
        with open(_o, "w") as _f:
            json.dump({"checkpoint": os.path.basename(a.ckpt),
                       "minimax": sm, "value": sv}, _f, indent=2)
        print(f"  wrote {_o}")
    print()
    dead_mm = sm["dc_ratio"] > 2.5
    dead_v = sv["dc_ratio"] > 2.5
    if dead_mm and not dead_v:
        print("  => ONLY THE MINIMAX TRUNK IS DEAD. The value head, identical in")
        print("     shape and fed the identical latent, stays active. The cause is")
        print("     the 484-output structure -- 484 conflicting gradients on one")
        print("     shared representation, one cell active per transition.")
        print("     FIX = parameterization (shared action embeddings / low rank),")
        print("     not the optimizer.")
    elif dead_mm and dead_v:
        print("  => BOTH trunks are constant-dominated. This is a property of the whole stack")
        print("     (target std ~0.012, AdamW decay 0.01), not of the joint-action")
        print("     design. FIX = optimizer / normalization, and the minimax")
        print("     parameterization is not the culprit.")
    else:
        print("  => the minimax trunk is NOT dead by this threshold; the collapse")
        print("     story does not hold at this checkpoint. Re-examine.")


if __name__ == "__main__":
    main()
