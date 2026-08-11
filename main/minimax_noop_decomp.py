"""WHERE does the no-op disagreement come from?

Actions 0 and 9 are byte-identical (DIRECTIONS_BUTTONS[0] == ATTACKS_BUTTONS[0]
== [], both hold-nothing-for-8-frames), so truth requires Q(0,j) == Q(9,j) for
every state and every j. `minimax_axis_diag.py` measured that gap at ~0.008 --
LARGER than the 0.0069 spread of Q across all 22 actions. This asks why.

The head is DETERMINISTIC, so this is not stochastic noise; it is systematic
approximation error. Four candidate sources, and they imply different fixes:

  A INIT RESIDUE      cells (0,j) and (9,j) have completely independent rows in
                      the output Linear(512,484). Nothing ties them. They start
                      at different random values and only DATA can pull them
                      together. -> gap should be visible in a FRESH head and
                      shrink with training.
  B TARGET SAMPLING   each cell fits a small-sample mean of high-variance
                      returns. -> gap should scale like sigma_G / sqrt(visits),
                      so it should CORRELATE with visit counts.
  C TRUNK DRIFT       the 3-layer trunk is shared by all 484 outputs, and the
                      target is non-stationary (self-play + GAE bootstrapping
                      through a moving V). Every other cell's gradient perturbs
                      the shared features, so the two rows chase a moving
                      representation. -> gap is STATE-DEPENDENT.
  D CONSTANT BIAS     a fixed offset between the two output rows. -> gap is
                      STATE-INDEPENDENT.

THE DECISIVE SPLIT is C vs D, and it is exact:

    delta_j(s) = Q(s,0,j) - Q(s,9,j)
    |mean_s delta_j|   -> the CONSTANT component (D: output-layer bias)
    std_s(delta_j)     -> the STATE-VARYING component (C: trunk/readout interaction)

A constant offset is a trivially fixable parameterization defect. A
state-varying one means the shared representation genuinely responds differently
to two identical actions, which no amount of output-layer tying would fix.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NOOP_A, NOOP_B = 0, 9


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n_envs", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="minimax_noop_decomp.json")
    ap.add_argument("--ram_mask", type=str, default="",
                    help="RAM byte-index .npy, required when the checkpoint was "
                         "trained with a MASKED ram observation: the checkpoint "
                         "records the WIDTH, not which bytes.")
    a = ap.parse_args(argv)

    import numpy as np
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, infer_obs_kwargs, REPO_ROOT)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs, **infer_obs_kwargs(data, (getattr(a, 'ram_mask', '') or None)))
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        pol = ops.p
        head = pol.minimax_net[list(pol.minimax_net.keys())[head_idx]]
        visits = head.cell_visits.detach().cpu().numpy()

        # --- A: what does a FRESH head of the same shape produce? --------------
        # Same class, same init path, untrained. Its no-op gap is the residue
        # that training has to remove.
        import copy
        fresh = copy.deepcopy(head)
        for m in fresh.modules():
            if isinstance(m, th.nn.Linear):
                m.reset_parameters()

        rng = np.random.RandomState(0)
        Ms, Fs = [], []
        obs = venv.reset()
        for t in range(a.steps):
            ob = th.as_tensor(obs).to(ops.device)
            with th.no_grad():
                Ms.append(pol.minimax_matrices(ob, buf_num=[head_idx],
                                               stop_grad=True).cpu().numpy())
                # same frozen latent, untrained head
                from stable_baselines3.common.preprocessing import preprocess_obs
                x = preprocess_obs(ob, pol.observation_space,
                                   normalize_images=pol.normalize_images)
                lat = pol.mlp_extractor.forward_critic(pol.vf_features_extractor(x))
                Fs.append(fresh(lat).cpu().numpy())
            obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                      ops.sample_adv(obs, rng)))[0]
    finally:
        venv.close()

    M = np.concatenate(Ms)                      # (N, 22, 22) trained
    F = np.concatenate(Fs)                      # (N, 22, 22) fresh
    N = M.shape[0]

    def decomp(X):
        d = X[:, NOOP_A, :] - X[:, NOOP_B, :]   # (N, 22) delta per column
        const = np.abs(d.mean(axis=0))          # per-j constant component
        vary = d.std(axis=0)                    # per-j state-varying component
        return float(np.abs(d).mean()), float(const.mean()), float(vary.mean()), d

    gap_t, const_t, vary_t, d_t = decomp(M)
    gap_f, const_f, vary_f, _ = decomp(F)
    spread_t = float(M.std(axis=1).mean())
    scale_t = float(np.abs(M - M.mean()).mean())

    # --- B: does the gap track visit counts? ------------------------------------
    v0, v9 = visits[NOOP_A, :], visits[NOOP_B, :]
    per_j = np.abs(d_t).mean(axis=0)
    vmin = np.minimum(v0, v9)
    ok = vmin > 0
    corr = float(np.corrcoef(np.log10(vmin[ok] + 1), per_j[ok])[0, 1]) if ok.sum() > 2 else float("nan")

    print("\n" + "=" * 74)
    print(f"NO-OP DECOMPOSITION  {os.path.basename(a.ckpt)}   {N:,} states")
    print("=" * 74)
    print(f"  {'':32s} {'TRAINED':>12} {'FRESH INIT':>12}")
    print(f"  {'|Q(0,j) - Q(9,j)|  (total gap)':32s} {gap_t:>12.6f} {gap_f:>12.6f}")
    print(f"  {'  constant component':32s} {const_t:>12.6f} {const_f:>12.6f}")
    print(f"  {'  state-varying component':32s} {vary_t:>12.6f} {vary_f:>12.6f}")
    print(f"\n  {'spread of Q across actions':32s} {spread_t:>12.6f}")
    print(f"  {'mean |Q - mean(Q)|':32s} {scale_t:>12.6f}")
    print(f"\n  gap / spread                     {gap_t / max(spread_t,1e-12):>12.2f}")
    print(f"  varying / constant               {vary_t / max(const_t,1e-12):>12.2f}")
    print(f"  corr(log10 visits, per-j gap)    {corr:>12.3f}   "
          f"(B predicts strongly NEGATIVE)")
    print(f"  visits row0 {v0.sum():,.0f}   row9 {v9.sum():,.0f}")

    res = {"checkpoint": os.path.basename(a.ckpt), "n_states": N,
           "gap_trained": gap_t, "const_trained": const_t, "vary_trained": vary_t,
           "gap_fresh": gap_f, "const_fresh": const_f, "vary_fresh": vary_f,
           "spread": spread_t, "scale": scale_t, "visit_corr": corr,
           "visits_row0": float(v0.sum()), "visits_row9": float(v9.sum())}
    print()
    if vary_t > 2 * const_t:
        res["verdict"] = "STATE-VARYING (trunk)"
        print("  => the gap is mostly STATE-DEPENDENT. The shared representation")
        print("     responds differently to two identical actions, so tying the")
        print("     output rows would NOT fix it. Consistent with C (trunk")
        print("     drift under a non-stationary target).")
    elif const_t > 2 * vary_t:
        res["verdict"] = "CONSTANT (output-layer bias)"
        print("  => the gap is mostly a FIXED OFFSET between two output rows --")
        print("     a parameterization defect, exactly what shared action")
        print("     embeddings would remove by construction.")
    else:
        res["verdict"] = "MIXED"
        print("  => constant and state-varying components are comparable; both")
        print("     the parameterization and the representation contribute.")
    out = os.path.join(REPO_ROOT, a.out)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {out}")
    return res


if __name__ == "__main__":
    main()
