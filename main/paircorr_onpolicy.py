"""On-policy gamma cross-state correlation: does THIS policy visit states where
the joint-action interaction is CORRELATED across states (i.e. learnable)?

paircorr has sat pinned at ~0.005 under uniform-random play across every offline
lever (frameskip, reward variants, aggresive_coeff). The one regime that ever
showed structure (engaged, 0.359) was a policy-CONCENTRATED stalemate. So the
open question is whether a TRAINED policy's state distribution has cross-state
gamma structure that random play does not. This rolls the checkpoint's own
policy, snapshots the states it actually visits, enumerates all 484 joint
actions at each, and computes paircorr under the ZERO-SUM (a=1) reward -- so the
number is comparable to every other paircorr in the programme.

CONFOUND CONTROLLED: gamma is scored under a=1 regardless of the policy's
training aggresive_coeff, so this isolates the STATE-DISTRIBUTION effect from the
reward-function effect (which the offline sweep already showed is inert).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = "pc_root"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--n_states", type=int, default=200)
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="paircorr_onpolicy.json")
    a = ap.parse_args(argv)

    import numpy as np
    from local_best_response import (build_lbr_venv, load_checkpoint, PolicyOps,
                                     resolve_matchups, infer_obs_kwargs)
    from stable_baselines3.common.save_util import load_from_zip_file

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, _lab, st = resolve_matchups(data, "all")[0]
    venv = build_lbr_venv(st, a.n_envs, **infer_obs_kwargs(data, a.ram_mask or None))
    L = load_checkpoint(a.ckpt, venv, a.device)
    model = L[0] if isinstance(L, tuple) else L
    ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
    rng = np.random.RandomState(0)
    n = a.n_envs
    na = int(ops.n_actions)

    obs = venv.reset()
    for _ in range(20):
        obs = venv.step(ops.joint(ops.sample_ego(obs, rng), ops.sample_adv(obs, rng)))[0]

    def hp(info_key, inf):
        return np.array([i.get(info_key, 0) for i in inf], dtype=np.float64)

    n_roots = a.n_states
    De = np.zeros((n_roots, na, na)); Da = np.zeros((n_roots, na, na))
    LIVE = np.zeros((n_roots, na, na), bool)
    got = 0
    while got < n_roots:
        # root hp for the CURRENT states (one per env); use env 0..n as roots
        # in batches of n. Read root hp from a no-op step is unavailable, so read
        # from info after enumerating (POST) and the root from a snapshot marker:
        # simplest -- snapshot, enumerate, the root hp = max over branches that
        # did no damage is unreliable, so read root hp from the emulator via a
        # branch that holds still (action 0,0 -> often zero damage). Instead we
        # capture root hp by stepping a hold action once BEFORE snapshot.
        hold = ops.joint(np.zeros(n, int), np.zeros(n, int))
        _, _, _, _, inf0 = venv.step(hold)
        ra, re = hp("agent_hp", inf0), hp("enemy_hp", inf0)
        venv.env_method("lbr_snapshot", ROOT)
        take = min(n, n_roots - got)
        for i in range(na):
            for j in range(na):
                venv.env_method("lbr_restore", ROOT)
                _, _, _, d, inf = venv.step(ops.joint(np.full(n, i), np.full(n, j)))
                pa = hp("agent_hp", inf); pe = hp("enemy_hp", inf)
                dn = np.asarray(d).reshape(-1).astype(bool)
                for e in range(take):
                    De[got + e, i, j] = re[e] - pe[e]
                    Da[got + e, i, j] = ra[e] - pa[e]
                    LIVE[got + e, i, j] = (not dn[e]) and pa[e] > 0 and pe[e] > 0 \
                        and pa[e] <= ra[e] and pe[e] <= re[e]
        venv.env_method("lbr_restore", ROOT); venv.env_method("lbr_drop", ROOT)
        got += take
        for _ in range(a.gap):
            obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                      ops.sample_adv(obs, rng)))[0]
        if got % 64 < n:
            print(f"[pc] {got}/{n_roots} roots", flush=True)
    venv.close()

    # a=1 zero-sum reward matrix; gamma via ANOVA; paircorr over active states.
    M = np.where(LIVE, De - Da, 0.0)
    mu = M.mean(axis=(1, 2), keepdims=True)
    al = M.mean(axis=2, keepdims=True) - mu; be = M.mean(axis=1, keepdims=True) - mu
    G = M - mu - al - be
    gn = (G ** 2).sum(axis=(1, 2)); wn = ((M - mu) ** 2).sum(axis=(1, 2))
    act = gn > 1e-18
    contact = float(act.mean())
    share = float(gn[act].sum() / max(wn[act].sum(), 1e-30)) if act.any() else float("nan")
    pair = float("nan")
    if act.sum() > 1:
        Z = G[act].reshape(int(act.sum()), -1)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        C = Z @ Z.T; iu = np.triu_indices(len(C), k=1); pair = float(C[iu].mean())
    out = dict(ckpt=os.path.basename(a.ckpt), n_roots=int(n_roots),
               n_active=int(act.sum()), contact=contact, gamma_share=share,
               gamma_mag=float(np.sqrt(gn.mean())), paircorr=pair)
    print(f"\n  ON-POLICY, a=1 scoring: contact={contact:.1%}  active={int(act.sum())}  "
          f"gamma_share={share:.2%}  |gamma|={out['gamma_mag']:.4f}  paircorr={pair:+.4f}")
    print(f"  baseline (random play) paircorr ~ 0.005. engaged stalemate was 0.359.")
    print(f"  paircorr >> 0.005 => this policy's states have LEARNABLE joint structure.")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
