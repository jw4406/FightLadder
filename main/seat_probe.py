"""Does the colour-cycling cost STATE-identification information?

obs_attribution.py proved the i%3 colour-cycling costs ZERO ACTION-distinctness
(all_channels == current, 8/8 cells). But that test holds the matchup fixed and
never asks what colour is FOR: telling the two sprites apart (seat ID -- in a
MIRROR match palette is the ONLY cue) or telling characters apart. This probes
exactly that, at the pixel-information level (no trained net -- random rollout,
fresh probe), comparing `current` (colour-cycled K=3) vs `all_channels` (full
RGB) observations.

SEAT: on a mirror match (RyuVsRyu) the sprites are identical, so recovering
  agent_x / seat-side / agent_status from pixels REQUIRES palette. Control:
  RyuVsSagat (distinct silhouettes) -- seat is recoverable from SHAPE there, so
  current ~ all_channels validates the probe and gives the "shape suffices"
  reference. The mirror-vs-control contrast is the result.
CHARID: hold the agent as Ryu-left, vary the opponent across characters, probe
  enemy_character from pixels.

Guards against my own logged traps: EPISODE-LEVEL train/test split (timestep
splits inflate ~5x), held-out metrics only, a shuffle-target FLOOR and a
majority/mean CHANCE baseline per cell so nothing is baseline-less, and PCA to a
matched n_components so all_channels' higher dim cannot win on overfit alone.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_RYU = "two_player/Ryu_left/Champion.Level1.RyuVs{}.2Player.state"
CHARID_OPPONENTS = ["Ryu", "Guile", "Sagat", "MBison", "Dhalsim", "Zangief",
                    "Blanka", "ChunLi"]


def collect(state, n_envs, n_samples, reset_every, warmup, ds, seed, close_range=0.0):
    """Random-rollout collection of probe rows for one state.

    Returns X_current (M,Dc), X_all (M,Da), targets dict, and episode ids
    (state,env,block) -- disjoint trajectory segments for an episode-level split.
    """
    from common.utils import SubprocVecEnv2P, VecTransposeImage2P
    from local_best_response import make_lbr_env
    rng = np.random.RandomState(seed)
    venv = VecTransposeImage2P(SubprocVecEnv2P(
        [make_lbr_env(state, side="both", reset_type="round",
                      transform_action=True, obs_type="image", seed=seed + i,
                      reset_close_range=close_range, close_max_steps=150)
         for i in range(n_envs)]))
    try:
        na = int(venv.env_method("lbr_config")[0]["n_actions"])
        cur, allc, tgt, eps = [], [], {}, []
        per_block = reset_every - warmup
        n_blocks = int(np.ceil(n_samples / (n_envs * per_block)))
        for b in range(n_blocks):
            venv.reset()
            for w in range(warmup):
                venv.step(rng.randint(0, na, size=(n_envs, 2)))
            for _ in range(per_block):
                venv.step(rng.randint(0, na, size=(n_envs, 2)))
                rows = venv.env_method("lbr_probe_sample", ds)
                for e, r in enumerate(rows):
                    cur.append(r["current"].reshape(-1))
                    allc.append(r["all_channels"].reshape(-1))
                    for k, v in r["targets"].items():
                        tgt.setdefault(k, []).append(v)
                    eps.append((state, e, b))
    finally:
        venv.close()
    X_cur = np.asarray(cur, dtype=np.float32)
    X_all = np.asarray(allc, dtype=np.float32)
    tgt = {k: np.asarray(v) for k, v in tgt.items()}
    # stable integer episode id
    uniq = {e: i for i, e in enumerate(dict.fromkeys(eps))}
    ep_id = np.array([uniq[e] for e in eps])
    return X_cur, X_all, tgt, ep_id


def _split(ep_id, seed, frac=0.30):
    rng = np.random.RandomState(seed)
    u = np.unique(ep_id)
    rng.shuffle(u)
    n_test = max(1, int(round(len(u) * frac)))
    test_eps = set(u[:n_test].tolist())
    test = np.array([e in test_eps for e in ep_id])
    return ~test, test


def _fit_eval(X, y, tr, te, n_pca, task, seed):
    """Held-out metric + shuffle floor + chance, with matched-capacity PCA."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score

    def pipe(Xtr, Xte):
        sc = StandardScaler().fit(Xtr)
        k = min(n_pca, Xtr.shape[1], Xtr.shape[0] - 1)
        pca = PCA(n_components=k, random_state=seed).fit(sc.transform(Xtr))
        return pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))

    Ztr, Zte = pipe(X[tr], X[te])
    ytr, yte = y[tr], y[te]

    def run(yt):
        if task == "reg":
            m = Ridge(alpha=10.0).fit(Ztr, yt)
            return float(r2_score(yte, m.predict(Zte)))
        m = LogisticRegression(max_iter=2000, C=1.0,
                               multi_class="auto").fit(Ztr, yt)
        return float(balanced_accuracy_score(yte, m.predict(Zte)))

    rng = np.random.RandomState(seed)
    real = run(ytr)
    shuf = run(rng.permutation(ytr))
    if task == "reg":
        chance = float(r2_score(yte, np.full_like(yte, ytr.mean(), dtype=float)))
        extra = {"test_std": float(np.std(yte))}
    else:
        vals, cnt = np.unique(yte, return_counts=True)
        chance = float(cnt.max() / cnt.sum())          # majority-class acc
        extra = {"n_classes": int(len(np.unique(ytr))),
                 "majority_frac": chance}
    return {"heldout": real, "shuffle_floor": shuf, "chance": chance, **extra}


def seat_target(tgt, name):
    if name == "agent_x":
        return tgt["agent_x"].astype(float), "reg"
    if name == "seat_side":                       # 1 = agent RIGHT of enemy
        return (tgt["agent_x"] > tgt["enemy_x"]).astype(int), "clf"
    if name == "agent_status":
        return tgt["agent_status"].astype(int), "clf"
    raise ValueError(name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["seat", "charid", "both"], default="both")
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--samples_per_state", type=int, default=3000)
    ap.add_argument("--reset_every", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--close_range", type=float, default=0.0,
                    help="seat mode: walk fighters within this many px at reset so "
                         "crossovers are frequent AND spread across episodes -- "
                         "without it the agent is always P1/left and seat is "
                         "recoverable by POSITION, not palette (confounded).")
    ap.add_argument("--charid_states", type=str, default=",".join(CHARID_OPPONENTS))
    ap.add_argument("--out", type=str, default="seat_probe.json")
    a = ap.parse_args(argv)

    res = {"config": vars(a), "seat": {}, "charid": {}}

    def line(tag, r):
        d = r["heldout"] - r["shuffle_floor"]
        extra = (f"std={r['test_std']:.1f}" if "test_std" in r
                 else f"cls={r['n_classes']} maj={r['majority_frac']:.2f}")
        print(f"    {tag:<26} heldout={r['heldout']:+.3f}  "
              f"shuffle={r['shuffle_floor']:+.3f}  chance={r['chance']:+.3f}  "
              f"(signal={d:+.3f})  {extra}")

    if a.mode in ("seat", "both"):
        print("=" * 78)
        print("SEAT / agent-identification probe   (current vs all_channels)")
        print("  MIRROR RyuVsRyu: palette is the ONLY seat cue.  "
              "CONTROL RyuVsSagat: shape suffices.")
        print("=" * 78)
        for label, opp in [("MIRROR_RyuVsRyu", "Ryu"), ("CONTROL_RyuVsSagat", "Sagat")]:
            state = _RYU.format(opp)
            Xc, Xa, tgt, ep = collect(state, a.n_envs, a.samples_per_state,
                                      a.reset_every, a.warmup, a.ds, a.seed,
                                      close_range=a.close_range)
            tr, te = _split(ep, a.seed)
            print(f"\n  [{label}]  M={len(ep)}  episodes={len(np.unique(ep))}  "
                  f"train/test={tr.sum()}/{te.sum()}  Dc={Xc.shape[1]} Da={Xa.shape[1]}")
            cross = float((tgt["agent_x"] > tgt["enemy_x"]).mean())
            print(f"  CROSSOVER fraction (agent right of enemy) = {cross:.3f}  "
                  f"-- palette is only NEEDED where position does NOT identify "
                  f"the agent, i.e. on crossovers")
            res["seat"][label] = {"crossover_frac": cross}
            for tname in ["agent_x", "seat_side", "agent_status"]:
                y, task = seat_target(tgt, tname)
                task = "reg" if tname == "agent_x" else "clf"
                # LOUD guard: a clf target with <2 classes in train OR test is
                # degenerate (e.g. no crossovers) and cannot be probed -- report
                # it rather than fake a number.
                if task == "clf" and (len(np.unique(y[tr])) < 2
                                      or len(np.unique(y[te])) < 2):
                    print(f"  target={tname}: SKIPPED -- <2 classes present "
                          f"(train {len(np.unique(y[tr]))}, test "
                          f"{len(np.unique(y[te]))}); degenerate/underpowered")
                    res["seat"][label][tname] = {"skipped": "degenerate"}
                    continue
                print(f"  target={tname}:")
                cell = {}
                for vname, X in [("current", Xc), ("all_channels", Xa)]:
                    r = _fit_eval(X, y, tr, te, a.pca, task, a.seed)
                    line(f"{vname}", r)
                    cell[vname] = r
                res["seat"][label][tname] = cell

    if a.mode in ("charid", "both"):
        print("\n" + "=" * 78)
        print("CHARACTER-ID probe   enemy_character from pixels (agent=Ryu_left fixed)")
        print("=" * 78)
        opps = [s.strip() for s in a.charid_states.split(",") if s.strip()]
        per = max(400, a.samples_per_state // len(opps))
        Xc_all, Xa_all, y_all, ep_all = [], [], [], []
        base = 0
        for opp in opps:
            state = _RYU.format(opp)
            Xc, Xa, tgt, ep = collect(state, a.n_envs, per,
                                      a.reset_every, a.warmup, a.ds, a.seed)
            Xc_all.append(Xc); Xa_all.append(Xa)
            y_all.append(tgt["enemy_character"].astype(int))
            ep_all.append(ep + base); base += ep.max() + 1
            print(f"  collected {len(ep):5d}  RyuVs{opp}  "
                  f"enemy_character={int(tgt['enemy_character'][0])}")
        Xc = np.concatenate(Xc_all); Xa = np.concatenate(Xa_all)
        y = np.concatenate(y_all); ep = np.concatenate(ep_all)
        tr, te = _split(ep, a.seed)
        print(f"\n  pooled M={len(ep)}  chars={len(np.unique(y))}  "
              f"episodes={len(np.unique(ep))}  train/test={tr.sum()}/{te.sum()}")
        print("  target=enemy_character:")
        for vname, X in [("current", Xc), ("all_channels", Xa)]:
            r = _fit_eval(X, y, tr, te, a.pca, "clf", a.seed)
            line(vname, r)
            res["charid"][vname] = r

    from local_best_response import REPO_ROOT
    with open(os.path.join(REPO_ROOT, a.out), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {os.path.join(REPO_ROOT, a.out)}")
    return res


if __name__ == "__main__":
    main()
