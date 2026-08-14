"""Is the ~53% ceiling the W(s) MAP, or the trunk LATENT itself?

THE FORK. Enumerated matrices lift the real head's capture of the true
interaction from ~5% to ~53%, and it SATURATES there -- flat from gamma-share
54.3% to 80.7%. The same objective with a FREE per-state W_s reaches 70.19%. So
roughly 17 points are lost somewhere between "the latent" and "W(s)".

    W(s) = Linear(latent -> r*r)          what ships today
    W(s) = MLP(latent -> h -> r*r)        this test
    W_s   free per state                  the 70.19% proxy ceiling

If the MLP reaches ~70%, the trunk LATENT already carries the interaction and a
one-layer map was the bottleneck -- a cheap architectural fix.
If it stays at ~53%, the latent does not carry it, and the fix is training the
TRUNK, which is far more invasive and would change the Phase-0 isolation
guarantee (minimax_matrices detaches the shared latent by design).

FAIRNESS. Every arm re-initialises w_out from scratch, including the linear
control -- otherwise the linear arm would start from trained weights and the MLP
arms from noise, and the comparison would measure initialisation, not capacity.
Everything upstream of w_out (trunk, embeddings, v/a heads) keeps its checkpoint
state and is shared identically across arms.

Nothing in stable_baselines3 is modified: w_out is invoked as a module, so an
nn.Sequential drops in.
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True, help="from bootstrap_delta --save_obs")
    ap.add_argument("--ram_mask", type=str, default="")
    ap.add_argument("--target", default="M", choices=["M", "R"])
    ap.add_argument("--hidden", type=int, nargs="+", default=[0, 128, 512],
                    help="0 = the current single Linear; >0 = one hidden layer")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_trunk", action="store_true",
                    help="Also train the shared value encoder (stop_grad=False). "
                         "Answers whether the LATENT can carry the interaction at "
                         "all. NOTE this breaks the Phase-0 isolation guarantee and "
                         "would change V, so it is a CEILING measurement only.")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     resolve_matchups, infer_obs_kwargs)
    from gamma_basis import gammas, capture

    d = np.load(a.npz)
    if "OBS" not in d:
        raise SystemExit("npz has no OBS -- recollect with --save_obs")
    OBS = d["OBS"]; TGT = d[a.target].astype(np.float32)
    S = TGT.shape[0]
    Gtrue = gammas(d["R"].astype(np.float64))
    ntr = int(S * (1 - a.holdout))
    tr, te = np.arange(ntr), np.arange(ntr, S)
    print(f"[wcap] {S} states  train {len(tr)} holdout {len(te)}  target={a.target}")

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, _, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, 2, **infer_obs_kwargs(data, a.ram_mask or None))
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        policy = model.policy
        head = policy.minimax_head_for([head_idx])
        head0 = copy.deepcopy(head).state_dict()
        w_in = head.w_out.in_features
        r = head.rank
        print(f"[wcap] head rank {r}, w_out {w_in} -> {r*r}")

        obs_t = th.as_tensor(OBS, dtype=th.float32, device=a.device)
        tgt_t = th.as_tensor(TGT, dtype=th.float32, device=a.device)

        def cap_now():
            ee = head.e_ego.detach().cpu().numpy(); ea = head.e_adv.detach().cpu().numpy()
            ee = ee - ee.mean(0); ea = ea - ea.mean(0)
            return capture(Gtrue, np.linalg.qr(ee)[0][:, :r], np.linalg.qr(ea)[0][:, :r])

        orig_w = copy.deepcopy(head.w_out)
        print(f"\n{'W(s) map':<26} {'params':>9} {'capture':>9} {'holdout MSE':>13}")
        for h in a.hidden:
            # restore the ORIGINAL w_out shape first: head0 carries Linear keys,
            # so loading it while an MLP is installed raises on missing keys.
            head.w_out = copy.deepcopy(orig_w).to(a.device)
            head.load_state_dict(head0)
            th.manual_seed(0)
            if h == 0:
                new_w = nn.Linear(w_in, r * r)
            else:
                new_w = nn.Sequential(nn.Linear(w_in, h), nn.LeakyReLU(),
                                      nn.Linear(h, r * r))
            head.w_out = new_w.to(a.device)
            nparam = sum(p.numel() for p in head.w_out.parameters())
            params = [p for p in head.parameters() if p.requires_grad]
            if a.train_trunk:
                params += [p for p in policy.vf_features_extractor.parameters()
                           if p.requires_grad]
                params += [p for p in policy.mlp_extractor.parameters()
                           if p.requires_grad]
            opt = th.optim.AdamW(params, lr=a.lr)
            otr, ttr = obs_t[tr], tgt_t[tr]
            for _ in range(a.steps):
                P = policy.minimax_matrices(otr, buf_num=[head_idx],
                                            stop_grad=not a.train_trunk)
                loss = ((P - ttr) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with th.no_grad():
                Pte = policy.minimax_matrices(obs_t[te], buf_num=[head_idx])
                mse = float(((Pte - tgt_t[te]) ** 2).mean())
            lab = ("Linear (ships today)" if h == 0 else f"MLP hidden={h}")
            if a.train_trunk:
                lab += " +trunk"
            print(f"{lab:<26} {nparam:>9,} {cap_now():>9.2%} {mse:>13.3e}")

        head.w_out = orig_w
        head.load_state_dict(head0)
        print(f"\n[wcap] free-W_s proxy ceiling on this parameterisation: 70.19%")
        print(f"[wcap] if the MLP arms sit at ~53%, the LATENT is the limit, not the map.")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
