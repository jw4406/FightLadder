"""PILOT: does the REAL head recover the interaction when given enumerated matrices?

WHAT THIS DE-RISKS. Two offline pre-tests said the head's failure is
observability: fitting global embeddings with a FREE per-state W_s reached the
closed-form optimum (70.19%) from full 484-cell matrices and collapsed to 0.95%
-- below the random floor -- from one cell per state. But both used a free W_s.
The real head must produce W(s) from the TRUNK, and a trunk that cannot express
the right W(s) would make the whole enumeration pipeline worthless.

So: load a real checkpoint, train ONLY the head on enumerated matrices, and ask
whether its interaction moves. An afternoon here decides whether the pipeline is
worth building.

THE COMPARISON. Same head, same states, same optimiser, same number of updates --
only the OBSERVABILITY differs:

    ONE cell/state   what training sees today
    k cells/state    the ablation ladder, --enum_k
    all 484          full enumeration

READ IT AS A LOWER BOUND on what the pipeline would achieve: this trains the head
against a FIXED trunk on a FIXED state set, whereas in training the trunk keeps
improving and states keep refreshing.

Requires a collection made with `bootstrap_delta.py --save_obs`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True, help="from bootstrap_delta --save_obs")
    ap.add_argument("--ram_mask", type=str, default="")
    ap.add_argument("--target", default="M", choices=["M", "R"],
                    help="M = r + gamma*V(s') (what training would bootstrap); "
                         "R = exact emulator reward (no critic error)")
    ap.add_argument("--enum_k", type=int, nargs="+", default=[1, 4, 16, 64, 484],
                    help="cells observed per state; the privilege ladder")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json_out", default="", help="machine-readable results, for "
                                                  "cross-checkpoint aggregation")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     resolve_matchups, infer_obs_kwargs)
    from gamma_basis import gammas, capture

    d = np.load(a.npz)
    if "OBS" not in d:
        raise SystemExit(f"{a.npz} has no OBS -- recollect with "
                         f"`bootstrap_delta.py --save_obs`")
    OBS = d["OBS"]; TGT = d[a.target].astype(np.float32)
    S, NE, NA = TGT.shape
    Gtrue = gammas(d["R"].astype(np.float64))     # truth is ALWAYS the critic-free R
    print(f"[pilot] {S} states, obs {OBS.shape[1:]}, target={a.target}")

    # episode-level split is not available here, but the states are strided far
    # apart by construction, so a contiguous split is close to independent.
    ntr = int(S * (1 - a.holdout))
    tr, te = np.arange(ntr), np.arange(ntr, S)
    print(f"[pilot] train {len(tr)}  holdout {len(te)}")

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, _, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, 2, **infer_obs_kwargs(data, a.ram_mask or None))
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        policy = model.policy
        if not getattr(policy, "minimax_q", False):
            raise SystemExit("checkpoint has no minimax head (--minimax_q was off)")

        obs_t = th.as_tensor(OBS, dtype=th.float32, device=a.device)
        tgt_t = th.as_tensor(TGT, dtype=th.float32, device=a.device)

        import copy
        # policy.minimax_head is the CONFIG STRING ("factored"/"matrix"); the
        # module lives in the minimax_net ModuleDict, keyed by matchup.
        # minimax_head_for() is the accessor that resolves it the same way
        # minimax_matrices() does, so the two can never disagree.
        head = policy.minimax_head_for([head_idx])
        if head is None:
            raise SystemExit("no minimax head on this checkpoint")
        head0 = copy.deepcopy(head).state_dict()

        def head_capture():
            """Capture of the TRUE gamma by the head's current embeddings."""
            h = head
            if not hasattr(h, "e_ego"):
                return float("nan")          # matrix head has no embeddings
            ee = h.e_ego.detach().cpu().numpy(); ea = h.e_adv.detach().cpu().numpy()
            ee = ee - ee.mean(0); ea = ea - ea.mean(0)
            r = ee.shape[1]
            return capture(Gtrue, np.linalg.qr(ee)[0][:, :r],
                           np.linalg.qr(ea)[0][:, :r])

        def head_gamma_share(idx):
            """gamma as a share of the head's own within-state energy, held out."""
            with th.no_grad():
                P = policy.minimax_matrices(obs_t[idx], buf_num=[head_idx]).cpu().numpy()
            g = gammas(P.astype(np.float64))
            w = P - P.mean(axis=(1, 2), keepdims=True)
            return float((g ** 2).sum() / max((w ** 2).sum(), 1e-30))

        base_cap = head_capture()
        print(f"\n[pilot] BEFORE any training: head capture of true gamma "
              f"{base_cap:.2%}   head gamma-share (holdout) {head_gamma_share(te):.2%}")

        rng = np.random.RandomState(0)
        out = {"ckpt": os.path.basename(a.ckpt), "npz": os.path.basename(a.npz),
               "target": a.target, "n_states": int(S),
               "steps": int(data.get("num_timesteps") or 0),
               "true_gamma_share": float((Gtrue ** 2).sum() /
                   max(((d["R"] - d["R"].mean(axis=(1, 2), keepdims=True)) ** 2).sum(), 1e-30)),
               "base_capture": float(base_cap),
               "base_gamma_share": float(head_gamma_share(te)), "arms": {}}
        print(f"\n{'enum_k':>7} {'capture':>9} {'gamma-share':>12} {'holdout MSE':>13}")
        for k in a.enum_k:
            head.load_state_dict(head0)                     # same start every arm
            params = [p for p in head.parameters() if p.requires_grad]
            opt = th.optim.AdamW(params, lr=a.lr)
            # WHICH cells are observed is fixed per state for the whole run --
            # resampling every step would hand the learner far more information
            # than on-policy training ever gets.
            mask = np.zeros((len(tr), NE, NA), bool)
            for s in range(len(tr)):
                sel = rng.choice(NE * NA, size=min(k, NE * NA), replace=False)
                mask[s].flat[sel] = True
            mask_t = th.as_tensor(mask, device=a.device)
            otr, ttr = obs_t[tr], tgt_t[tr]

            for _ in range(a.steps):
                P = policy.minimax_matrices(otr, buf_num=[head_idx], stop_grad=True)
                loss = (((P - ttr) ** 2) * mask_t).sum() / mask_t.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with th.no_grad():
                Pte = policy.minimax_matrices(obs_t[te], buf_num=[head_idx])
                mse = float(((Pte - tgt_t[te]) ** 2).mean())
            cap_k, gs_k = head_capture(), head_gamma_share(te)
            out["arms"][str(k)] = {"capture": float(cap_k),
                                   "gamma_share": float(gs_k), "holdout_mse": mse}
            print(f"{k:>7} {cap_k:>9.2%} {gs_k:>12.2%} {mse:>13.3e}")

        head.load_state_dict(head0)
        if a.json_out:
            import json
            with open(a.json_out, "w") as f:
                json.dump(out, f, indent=1)
            print(f"[pilot] results -> {a.json_out}")
        print(f"\n[pilot] baseline for reference: k=1 is what training sees today.")
        print(f"[pilot] offline proxy predicted 0.95% (1 cell) -> 70.19% (484).")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
