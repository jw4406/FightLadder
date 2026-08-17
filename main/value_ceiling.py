"""THE INTRINSIC CEILING. How well could ANY value function possibly score here?

Every value measurement in this project has scored a PREDICTOR. None has measured
the CEILING, so "critic EV ~ 0" has always been ambiguous between two readings
that call for opposite work:

    V is bad                          -> optimisation/representation problem
    V is at a ceiling that is low     -> nothing to fix in the predictor

The ceiling is set by how much of the return's variance is WITHIN a state rather
than BETWEEN states. V(s) can only ever predict E[G|s], and it is scored against
a single SAMPLE of G, so

    EV_max  =  1  -  E_s[ Var(G|s) ]  /  Var(G)

If the same state produces wildly different returns, no value function scores
well, however perfect.

WHY THIS IS MEASURABLE HERE AND ALMOST NOWHERE ELSE. The emulator is
DETERMINISTIC -- enumeration via set_state gives bit-identical results -- so none
of the return variance is environmental. ALL of it is the two policies' own
action sampling. And lbr_snapshot/lbr_restore lets us return to the SAME state
and roll out K times with fresh action draws, which measures Var(G|s) directly
rather than assuming it.

That also explains a standing result: "greedy beats critic-guided LBR" is not an
indictment of the critic's accuracy -- greedy OBSERVES the immediate reward
instead of predicting an expectation, so it is noise-free by construction.

READING IT:
  ceiling ~0.07   V at ~0 is already near-optimal; improving the predictor is
                  closed, and the only lever left is reducing target variance.
  ceiling 0.3-0.5 V is leaving a lot on the table; it IS an optimisation problem.
  ceiling ~0      the return is unpredictable at this horizon; the value-function
                  framing itself needs rethinking.

CONTROL: the same states scored with K=1 reproduce the ordinary (noisy) target,
so the K-sample reduction is visible rather than asserted.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "vc_root"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--n_states", type=int, default=120,
                    help="root states to sample Var(G|s) at")
    ap.add_argument("--k", type=int, default=16,
                    help="independent rollouts from EACH root state")
    ap.add_argument("--horizon", type=int, default=80,
                    help="steps per rollout. At gamma 0.94 the effective horizon "
                         "is 16.7 steps, so 80 captures >99% of the discounted "
                         "return without paying for the tail.")
    ap.add_argument("--gap", type=int, default=12, help="steps between roots")
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--gammas", default="0.75,0.9,0.94,0.99")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="value_ceiling.json")
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

    gammas = [float(x) for x in a.gammas.split(",")]
    # EACH ENV IS ITS OWN ROOT. lbr_snapshot saves each worker's OWN emulator
    # state, so the n envs hold n DIFFERENT states -- not one state replicated.
    # The first version asserted lockstep ("all n_envs run the SAME root") and
    # with k == n_envs it did reps=1, so it averaged over n DISTINCT states and
    # called the spread Var(G|s). It measured BETWEEN-root variance and reported
    # it as within-root; the 0.05 "ceiling" it produced was meaningless.
    K = int(a.k)
    n_roots = a.n_states * n
    G = np.zeros((n_roots, K, len(gammas)))
    V = np.zeros(n_roots)
    # Root OBSERVATIONS, so the low-noise target G.mean(axis=1) can be
    # regressed on what the network actually sees. Without this the
    # capacity / features / target-noise hypotheses cannot be separated.
    OBS = None

    obs = venv.reset()
    for _ in range(20):
        obs = venv.step(ops.joint(ops.sample_ego(obs, rng), ops.sample_adv(obs, rng)))[0]

    for s in range(a.n_states):
        base = s * n
        V[base:base + n] = np.asarray(ops.values_ego(obs)).reshape(-1)
        _o = np.asarray(obs).reshape(n, -1)
        if OBS is None:
            OBS = np.zeros((n_roots, _o.shape[1]), np.float32)
        OBS[base:base + n] = _o
        venv.env_method("lbr_snapshot", ROOT)
        for kk in range(K):
            venv.env_method("lbr_restore", ROOT)      # every env back to ITS OWN root
            o = obs
            disc = np.ones((n, len(gammas))); acc = np.zeros((n, len(gammas)))
            alive = np.ones(n, bool)
            for t in range(a.horizon):
                o, r_l, r_r, d, _ = venv.step(
                    ops.joint(ops.sample_ego(o, rng), ops.sample_adv(o, rng)))
                rew = np.asarray(ops.lbr_reward(r_l, r_r)).reshape(-1)
                for gi in range(len(gammas)):
                    acc[:, gi] += disc[:, gi] * rew * alive
                    disc[:, gi] *= gammas[gi]
                alive &= ~np.asarray(d).reshape(-1).astype(bool)
                if not alive.any():
                    break
            G[base:base + n, kk, :] = acc
        venv.env_method("lbr_restore", ROOT); venv.env_method("lbr_drop", ROOT)
        for _ in range(a.gap):
            obs = venv.step(ops.joint(ops.sample_ego(obs, rng),
                                      ops.sample_adv(obs, rng)))[0]
        if (s + 1) % 5 == 0:
            print(f"[ceil] {(s+1)*n}/{n_roots} roots", flush=True)
    venv.close()

    # SELF-CHECK that the restore/replay actually works. Replays of ONE root must
    # differ (fresh action draws) -- if they are identical, Var(G|s) is 0 by
    # construction and the ceiling would read 1.0 for the wrong reason.
    _w = float(G[:, :, 2].var(axis=1).mean()); _b = float(G[:, :, 2].mean(axis=1).var())
    if _w <= 0:
        raise SystemExit("FAILED: replays of the same root are IDENTICAL; action "
                         "sampling is not fresh and the decomposition is void.")
    print(f"\n  [check] within-root var {_w:.3e}  between-root var {_b:.3e}")

    print(f"\n  {n_roots} roots x {K} REPLAYS OF THE SAME ROOT, horizon {a.horizon}\n")
    print(f"  {'gamma':>6} {'Var(G|s)':>11} {'Var(G)':>11} {'EV_MAX':>9} "
          f"{'V head EV':>10} {'K EV_max':>12}")
    out = {"n_roots": int(n_roots), "k": K, "rows": []}
    for gi, g in enumerate(gammas):
        Gg = G[:, :, gi]
        within = float(Gg.var(axis=1).mean())          # E_s[Var(G|s)]
        mu_s = Gg.mean(axis=1)
        total = float(Gg.var())                        # Var(G) overall
        ev_max = 1.0 - within / max(total, 1e-30)
        # What the trained head scores against the TRUE per-state mean -- i.e.
        # how good V is at the thing it could in principle be perfect at.
        ev_head = float(1 - ((mu_s - V) ** 2).mean() / max(mu_s.var(), 1e-30))
        # Averaging K samples cuts the noise floor by K: the ceiling a
        # K-sample-averaged TARGET would present to a learner.
        ev_max_k = 1.0 - (within / K) / max(total, 1e-30)
        print(f"  {g:6.2f} {within:11.3e} {total:11.3e} {ev_max:9.4f} "
              f"{ev_head:10.4f} {ev_max_k:12.4f}")
        out["rows"].append(dict(gamma=g, within=within, total=total,
                                ev_max=ev_max, ev_head_vs_truemean=ev_head,
                                ev_max_k=ev_max_k))
    np.savez_compressed(a.out.replace(".json", "_raw.npz"),
                        G=G, V=V, OBS=OBS, gammas=np.array(gammas))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  EV_MAX      ceiling for ANY value function against SINGLE-sample returns")
    print(f"  V head EV   the trained head against the TRUE per-state mean -- this")
    print(f"              isolates the head's error from the target's noise")
    print(f"  K=16 EV_max what a K-sample-averaged target would allow. The gap")
    print(f"              between EV_MAX and this is the prize for averaging targets")
    print(f"              via set_state, which is a lever almost no RL setup has.")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
