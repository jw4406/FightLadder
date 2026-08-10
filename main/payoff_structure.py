"""Does the joint-action payoff matrix vary across states, and at what rank?

WHY. The minimax head converged to Q(s,i,j) ~ b_ij -- a 484-entry lookup table,
constant in state, varying across cells. The trunk preserves 67% of the state
signal and Linear(512,484) destroys 96% of what is left. The proposed fix is a
factorized readout,

    Q(s,i,j) = e_ego(i)^T W(s) e_adv(j)          E fixed, W(s) small (r x r)

whose value rests on an assumption nobody has tested: that the TRUE payoff
matrix is state-dependent and low-rank. This measures that assumption before
anything is built.

WHAT IS MEASURED. At a sampled state, snapshot the emulator and step ALL 484
joint actions one agent-step each, restoring between them. That gives the exact
one-step payoff matrix M(s) at a real state -- not an estimate, and not the
top-k marginalized 88-branch slice LBR normally uses.

    A5  VALIDITY   actions 0 and 9 are BYTE-IDENTICAL (DIRECTIONS_BUTTONS[0] ==
                   ATTACKS_BUTTONS[0] == []) and the emulator is DETERMINISTIC
                   given a restored state, so r(s,0,j) must EXACTLY equal
                   r(s,9,j), and likewise for columns. Runs FIRST and hard-fails:
                   a broken pipeline that still prints plausible spectra is the
                   worst outcome available.
    A4  DEGENERACY distinct successor OBSERVATIONS per decision. Measured before
                   the spectra because it CONDITIONS them: a decision point where
                   every action yields the same successor cannot express action
                   structure, so pooling those in guarantees a null that says
                   nothing about states where the player actually has a choice.
    A1  VARIATION  M(s) = Mbar + D(s); mean||D||_F / ||Mbar||_F.
    A2  PER-STATE  SVD of each D(s): rank for 90% energy.
        RANK
    A3  SUBSPACE   SVD of the stacked (n_states x 484) matrix. Bilinear with a
                   FIXED E spans only an r^2-dimensional subspace of R^484, so
                   this is the number that constrains the design.

Payoffs are recorded twice: the raw one-step reward r (assumption-free, but zero
at ~94% of states since damage is sparse in time) and r + gamma*V(s') (what LBR
actually consumes, and therefore contaminated by V).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NOOP_A, NOOP_B = 0, 9
EXPAND_ROOT = "payoff_root"
MIN_ANALYSE = 20


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_states", type=int, default=600)
    ap.add_argument("--stride", type=int, default=60,
                    help="policy steps between expansions, to decorrelate states")
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", type=str, default="payoff_structure.json")
    ap.add_argument("--plot", type=str, default=None)
    a = ap.parse_args(argv)

    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups, splice_terminal,
                                     REPO_ROOT)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(state, a.n_envs)
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        gamma = a.gamma if a.gamma is not None else ops.gamma
        n = venv.num_envs
        na = ops.n_actions
        rng = np.random.RandomState(0)

        R_all, Q_all, H_all = [], [], []
        obs = venv.reset()
        n_exp = int(np.ceil(a.n_states / n))
        for e in range(n_exp):
            for _ in range(a.stride if e else 5):
                obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                          ops.sample_adv(obs, rng)))[0]

            venv.env_method("lbr_snapshot", EXPAND_ROOT)
            R = np.zeros((na, na, n)); V1 = np.zeros((na, na, n))
            DN = np.zeros((na, na, n), bool)
            HS = np.zeros((na, na, n), dtype=np.int64)
            for i in range(na):
                succ_row = []
                for j in range(na):
                    venv.env_method("lbr_restore", EXPAND_ROOT)
                    o1, r_l, r_r, d, infos = venv.step(
                        ops.joint(np.full(n, i), np.full(n, j)))
                    d = np.asarray(d, bool)
                    R[i, j] = ops.lbr_reward(r_l, r_r)
                    DN[i, j] = d
                    o1 = splice_terminal(o1, d, infos)
                    succ_row.append(o1)
                    for k in range(n):
                        HS[i, j, k] = hash(o1[k].tobytes())
                # One batched forward per row: 22*n rows instead of holding
                # 484*n successor frames in memory.
                V1[i] = ops.values_ego(np.concatenate(succ_row, axis=0)).reshape(na, n)
            venv.env_method("lbr_restore", EXPAND_ROOT)
            venv.env_method("lbr_drop", EXPAND_ROOT)

            R_all.append(R.transpose(2, 0, 1))
            Q_all.append((R + gamma * V1 * (~DN)).transpose(2, 0, 1))
            H_all.append(HS.transpose(2, 0, 1))
            print(f"   expansion {e+1}/{n_exp}  ({(e+1)*n} states)", flush=True)
    finally:
        venv.close()

    R = np.concatenate(R_all)
    Q = np.concatenate(Q_all)
    H = np.concatenate(H_all)
    S = R.shape[0]
    out = {"checkpoint": os.path.basename(a.ckpt), "n_states": S,
           "n_actions": na, "gamma": float(gamma)}

    # Persist the RAW matrices BEFORE any analysis. An analysis-side KeyError
    # previously killed the run after 8 minutes of branching and took the whole
    # collection with it.
    raw = os.path.join(REPO_ROOT, a.out.replace(".json", "_raw.npz"))
    np.savez_compressed(raw, R=R, Q=Q, H=H)
    print(f"\n   {S} states x {na}x{na};  raw -> {raw}")

    # ---- A5 VALIDITY --------------------------------------------------------
    row_err = float(np.abs(R[:, NOOP_A, :] - R[:, NOOP_B, :]).max())
    col_err = float(np.abs(R[:, :, NOOP_A] - R[:, :, NOOP_B]).max())
    out["a5_row_err"], out["a5_col_err"] = row_err, col_err
    print("\n" + "=" * 74)
    print(f"A5 VALIDITY  (actions {NOOP_A} and {NOOP_B} are byte-identical no-ops)")
    print("=" * 74)
    print(f"   max |r(s,0,j) - r(s,9,j)|   {row_err:.3e}")
    print(f"   max |r(s,i,0) - r(s,i,9)|   {col_err:.3e}")
    if max(row_err, col_err) > a.tol:
        out["verdict"] = "INVALID"
        with open(os.path.join(REPO_ROOT, a.out), "w") as f:
            json.dump(out, f, indent=2)
        raise SystemExit("\n   A5 FAILED: two byte-identical actions gave "
                         "DIFFERENT rewards from the same restored state, but "
                         "the emulator is deterministic. The branch pipeline is "
                         "broken; refusing to report a spectrum.")
    print("   PASS -- identical actions give identical results from restore")

    # ---- A4 DEGENERACY, before the spectra: it conditions them ---------------
    distinct = np.array([len(np.unique(H[s])) for s in range(S)])
    free = distinct > 1
    out["a4"] = {"median": float(np.median(distinct)), "min": int(distinct.min()),
                 "max": int(distinct.max()), "of": na * na,
                 "frac_non_forced": float(free.mean())}
    print("\n" + "=" * 74)
    print("A4 BRANCH DEGENERACY  (distinct successor observations per decision)")
    print("=" * 74)
    print(f"   median {np.median(distinct):.0f} / {na*na}   "
          f"min {distinct.min()}   max {distinct.max()}")
    print(f"   NON-FORCED (>1 distinct): {free.sum()}/{S} ({free.mean():.1%})")
    if np.median(distinct) <= 3:
        print("   => branches are FORCED at most decision points. Q cannot")
        print("      distinguish what the emulator does not distinguish.")

    def report(M, name, restrict_nonzero=False):
        """A1/A2/A3 on one matrix set. Returns (dict, stacked spectrum or None)."""
        live = np.linalg.norm(M.reshape(M.shape[0], -1), axis=1) > 0
        frac = float(live.mean())
        print("\n" + "=" * 74)
        print(f"[{name}]")
        print("=" * 74)
        if restrict_nonzero:
            print(f"   {live.sum()}/{M.shape[0]} states have a NONZERO payoff "
                  f"matrix ({frac:.1%}); analysing those only")
            M = M[live]
        if M.shape[0] < MIN_ANALYSE:
            print(f"   INSUFFICIENT: {M.shape[0]} states (< {MIN_ANALYSE}). No verdict.")
            return {"insufficient": True, "frac_live": frac,
                    "n_analysed": int(M.shape[0])}, None
        Sx = M.shape[0]
        Mbar = M.mean(0)
        D = M - Mbar
        nb = float(np.linalg.norm(Mbar))
        nd = float(np.linalg.norm(D.reshape(Sx, -1), axis=1).mean())
        print(f"   A1  ||Mbar||_F {nb:.6f}   mean||D||_F {nd:.6f}   "
              f"ratio {nd/max(nb,1e-12):.4f}")
        # REFUSE a spectrum on degenerate data: with D == 0 every cumulative bin
        # reads < 0.90 and rank90 comes out as na+1, a value that cannot exist.
        # The first smoke run printed exactly that ("23 / 22").
        if nd <= 1e-12:
            print("   NO cross-state variation (mean||D|| ~ 0). Either the states "
                  "are not decorrelated (raise --stride/--n_states) or the payoff "
                  "is constant. No rank is reportable.")
            return {"degenerate": True, "norm_mean": nb, "norm_var": nd,
                    "ratio": 0.0, "frac_live": frac, "n_analysed": int(Sx)}, None
        sv = np.linalg.svd(D, compute_uv=False)
        en = np.cumsum(sv ** 2, axis=1) / np.clip((sv ** 2).sum(1, keepdims=True),
                                                  1e-30, None)
        r90 = np.minimum((en < 0.90).sum(1) + 1, na)
        s3 = np.linalg.svd(D.reshape(Sx, -1), compute_uv=False)
        e3 = np.cumsum(s3 ** 2) / max((s3 ** 2).sum(), 1e-30)
        d90 = int((e3 < 0.90).sum() + 1)
        print(f"   A2  per-state rank for 90% energy (median) {np.median(r90):.0f} / {na}")
        print("       mean singular values  " +
              "  ".join(f"{x:.4f}" for x in sv.mean(0)[:6]))
        print(f"   A3  subspace dim for 90% of cross-state variation {d90} / {na*na}"
              f"   -> r ~ {np.sqrt(d90):.1f}")
        return {"norm_mean": nb, "norm_var": nd, "ratio": nd / max(nb, 1e-12),
                "per_state_rank90_median": float(np.median(r90)),
                "subspace_dim90": d90, "frac_live": frac, "n_analysed": int(Sx),
                "sv_mean": [float(x) for x in sv.mean(0)[:8]]}, s3

    out["reward"], s3r = report(R, "one-step reward r (assumption-free)",
                                restrict_nonzero=True)
    out["q"], s3q = report(Q, "r + gamma*V(s')  (what LBR consumes)")
    s3f = None
    if free.sum() >= MIN_ANALYSE:
        out["q_nonforced"], s3f = report(Q[free],
                                         "r + gamma*V(s')  NON-FORCED states only")
        out["reward_nonforced"], _ = report(R[free],
                                            "one-step reward r  NON-FORCED only",
                                            restrict_nonzero=True)
    else:
        print(f"\n   too few non-forced states ({free.sum()}) to analyse "
              f"separately -- raise --n_states")

    # ---- verdict ------------------------------------------------------------
    # Prefer the NON-FORCED result: it is the only one that separates "no action
    # structure exists" from "we averaged over states where no action could
    # possibly matter".
    src = None
    for cand in ("q_nonforced", "reward_nonforced", "reward", "q"):
        if out.get(cand, {}).get("subspace_dim90"):
            src = cand
            break
    print("\n" + "=" * 74)
    if src is None:
        out["verdict"] = "NO VERDICT -- degenerate or insufficient data"
        print("  VERDICT: withheld. No analysis set yielded enough live,")
        print("  decorrelated states. Raise --n_states / --stride.")
    else:
        blob = out[src]
        dim, ratio = blob["subspace_dim90"], blob["ratio"]
        r_eff = float(np.sqrt(dim))
        out["verdict_source"], out["r_suggested"] = src, r_eff
        print(f"  VERDICT (from `{src}`, {blob['n_analysed']} states)   "
              f"ratio {ratio:.4f}   subspace dim {dim}  ->  r ~ {r_eff:.1f}")
        if dim <= 2:
            out["verdict"] = "NO JOINT-ACTION STRUCTURE -- direction dead"
            print("  The payoff matrix is ~ c(s) * ones: a per-state SCALAR, i.e.")
            print("  Q(s,i,j) = V(s). A factorized head would recover exactly V")
            print("  and nothing more. The joint-action direction is FALSIFIED.")
        elif ratio < 0.05:
            out["verdict"] = "STATE-INDEPENDENT -- direction dead"
            print("  Payoff is essentially state-INDEPENDENT; no readout")
            print("  parameterization can help.")
        else:
            out["verdict"] = "STATE-DEPENDENT, LOW RANK -- factorization justified"
            print(f"  Structure exists and is low-rank. Build bilinear with "
                  f"r = {max(2, int(np.ceil(r_eff)))}.")

    with open(os.path.join(REPO_ROOT, a.out), "w") as f:
        json.dump(out, f, indent=2)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for sp, lbl in ((s3q, "all states"), (s3f, "non-forced"), (s3r, "reward")):
            if sp is None:
                continue
            e = np.cumsum(sp ** 2) / max((sp ** 2).sum(), 1e-30)
            ax.plot(np.arange(1, len(e) + 1), e, label=lbl)
        ax.axhline(0.9, ls="--", c="r", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("subspace dimension")
        ax.set_ylabel("cumulative energy"); ax.legend(fontsize=8)
        ax.set_title(f"cross-state payoff variation, {os.path.basename(a.ckpt)}")
        fig.tight_layout(); fig.savefig(a.plot, dpi=120)
        print(f"\n  wrote {a.plot}")
    print(f"  wrote {os.path.join(REPO_ROOT, a.out)}")
    return out


if __name__ == "__main__":
    main()
