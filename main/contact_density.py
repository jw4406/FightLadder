"""Does the DENSE REWARD, not the game, cause gamma to be zero on 94% of states?

THE CLAIM UNDER TEST. The in-fight dense reward is

    r1 = dense * (aggr * (prev_enemy_hp - enemy_hp) - (prev_agent_hp - agent_hp))
       = dense * (aggr * D_e - D_a)

so it is IDENTICALLY ZERO whenever no damage lands -- for all 484 joint actions
at once. On such a state Q(s,i,j) is constant in (i,j): mu, alpha, beta AND
gamma are all exactly zero. Damage is the only channel through which the payoff
depends on actions at all. If that is right, the measured "6% contact rate" is a
property of THIS REWARD FUNCTION, not of Street Fighter, and it is fixable.

THREE INTERVENTIONS, ALL MEASURED FROM ONE COLLECTION. The trick is to record
the raw COMPONENTS of each branch (hp deltas, statuses, positions) rather than
the reward itself, so every reward variant is recomputed offline from the same
484 enumerated branches at the same states. No re-collection, no confound
between conditions, and the weights sweep continuously instead of at two points.

  (5) start-state distribution   contact and gamma as a function of the ROOT
                                 separation |agent_x - enemy_x|. One rollout
                                 gives the whole curve, which is the scaling law
                                 the whole contact-density argument rests on.
  (1) counter-hit weighting      r1 = D_e*(1 + kappa*[enemy was attacking])
                                    - D_a*(1 + kappa*[agent was attacking]).
                                 Targets gamma SPECIFICALLY: a counter-hit is
                                 the purest joint event in the game, since
                                 neither action alone predicts it.
  (2) pressure shaping           r1 += beta*(p_agent - p_enemy), p = in range
                                 AND attacking. Raises contact directly.

ALL THREE ARE ANTISYMMETRIC (r2 = -r1) so the game stays ZERO-SUM, which
minimax-Q requires. Note that the existing `aggresive_coeff` knob is NOT:
r1 + r2 = (a-1)(D_e + D_a), zero only at a = 1. It is the obvious dial, it would
raise contact, and it would invalidate the operator being evaluated.

POLICY-FREE. Uniform random actions, no checkpoint. A trained checkpoint would
confound the measurement with whatever pathology that checkpoint carries, which
is the objection that killed the first version of the Phase -1 plan.

NOTHING HERE IS DERIVED FROM A HARDCODED CONSTANT. Which status values mean
"attacking", and what counts as "in range", are both read off the collected data
by asking which statuses and which separations actually co-occur with damage.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT_KEY = "cd_root"
FIELDS = ("agent_hp", "enemy_hp", "agent_x", "enemy_x",
          "agent_status", "enemy_status", "round_countdown")


def collect(a):
    import numpy as np
    from local_best_response import make_lbr_env

    mask = np.load(a.ram_mask)
    rng = np.random.RandomState(a.seed)
    env = make_lbr_env(a.state, obs_type="ram", ram_mask=mask, seed=a.seed)()
    na = env.lbr_config()["n_actions"]
    print(f"[cd] {na}x{na} branches, {a.n_roots} roots, state={a.state}", flush=True)

    def rnd():
        return np.array([rng.randint(na), rng.randint(na)])

    env.reset()
    info = None
    for _ in range(a.warmup):
        info = env.step(rnd())[-1]
    missing = [f for f in FIELDS if f not in info]
    if missing:
        raise SystemExit(f"FAILED: info is missing {missing}; got {sorted(info)}")

    ROOT = np.zeros((a.n_roots, len(FIELDS)), np.float64)
    POST = np.zeros((a.n_roots, na, na, len(FIELDS)), np.float64)
    RL = np.zeros((a.n_roots, na, na), np.float64)
    RR = np.zeros((a.n_roots, na, na), np.float64)
    DONE = np.zeros((a.n_roots, na, na), bool)

    n_skipped = 0
    for root in range(a.n_roots):
        ROOT[root] = [info[f] for f in FIELDS]
        env.lbr_snapshot(ROOT_KEY)
        for i in range(na):
            for j in range(na):
                env.lbr_restore(ROOT_KEY)
                _, rl, rr, d, inf = env.step(np.array([i, j]))
                POST[root, i, j] = [inf[f] for f in FIELDS]
                RL[root, i, j] = rl
                RR[root, i, j] = rr
                DONE[root, i, j] = bool(d)
        env.lbr_restore(ROOT_KEY)
        env.lbr_drop(ROOT_KEY)
        # SFWrapper.update_status asserts on the round/match state machine, and
        # a random walk can legitimately drive it through a double-KO or timeout
        # edge that trips the assert. Recover by resetting, and COUNT it -- a
        # silently-skipped root would quietly bias the state distribution.
        try:
            for _ in range(a.gap):
                info = env.step(rnd())[-1]
        except AssertionError:
            n_skipped += 1
            env.reset()
            for _ in range(a.warmup):
                info = env.step(rnd())[-1]
        if (root + 1) % 25 == 0:
            print(f"[cd] {root+1}/{a.n_roots} roots", flush=True)
    env.close()

    np.savez_compressed(a.out, ROOT=ROOT, POST=POST, RL=RL, RR=RR, DONE=DONE,
                        fields=np.array(FIELDS), na=na, n_skipped=n_skipped)
    frac = n_skipped / max(a.n_roots, 1)
    print(f"[cd] wrote {a.out}  ({n_skipped} round-boundary resets, {frac:.1%})",
          flush=True)
    if frac > 0.25:
        raise SystemExit(f"FAILED: {frac:.0%} of roots hit a round-boundary reset; "
                         f"the state distribution is dominated by recoveries, not "
                         f"by play.")


# ---------------------------------------------------------------------------


def analyze(a):
    import numpy as np

    F = {f: i for i, f in enumerate(FIELDS)}
    ROOT, POST, RL, DONE = [], [], [], []
    for f in a.npz:
        d = np.load(f, allow_pickle=True)
        ROOT.append(d["ROOT"]); POST.append(d["POST"])
        RL.append(d["RL"]); DONE.append(d["DONE"])
    ROOT = np.concatenate(ROOT); POST = np.concatenate(POST)
    RL = np.concatenate(RL); DONE = np.concatenate(DONE)
    ns, na = RL.shape[0], RL.shape[1]
    print(f"[cd] {ns} roots x {na*na} branches\n")

    # damage per branch, relative to the ROOT hp (which is what prev_*_hp holds)
    D_e = ROOT[:, None, None, F["enemy_hp"]] - POST[..., F["enemy_hp"]]
    D_a = ROOT[:, None, None, F["agent_hp"]] - POST[..., F["agent_hp"]]
    # A branch that ends a round takes the terminal reward path, where the hp
    # decomposition does not apply. Everything below is the in-fight reward.
    live = ~DONE & (POST[..., F["agent_hp"]] >= 0) & (POST[..., F["enemy_hp"]] >= 0)

    # SELF-CHECK. If the offline reconstruction of the CURRENT reward does not
    # reproduce the reward the emulator actually returned, every variant built
    # on the same components is void.
    base = 0.001 * (D_e - D_a)
    err = float(np.abs(base - RL)[live].max()) if live.any() else float("inf")
    if err > 1e-9:
        raise SystemExit(
            f"FAILED: offline reconstruction of the CURRENT dense reward differs "
            f"from the emulator's by {err:.3g} on non-terminal branches. The "
            f"component decomposition is wrong and every variant below is void.")
    print(f"  [ok] offline reconstruction matches the emulator exactly "
          f"(max err {err:.1e} over {int(live.sum())} live branches)")

    # ---- data-derived semantics -------------------------------------------
    st_a = POST[..., F["agent_status"]].astype(int)
    st_e = POST[..., F["enemy_status"]].astype(int)
    dealt = (D_e > 0) & live          # agent dealt damage
    took = (D_a > 0) & live
    # A status is "attacking" if being in it is strongly associated with having
    # just dealt damage. Read off the data, not hardcoded.
    atk = {}
    for s in np.unique(st_a):
        m = (st_a == s) & live
        atk[int(s)] = float(dealt[m].mean()) if m.sum() > 50 else 0.0
    ATK = sorted([s for s, p in atk.items() if p >= a.atk_thresh])
    print(f"  attack statuses (P(dealt damage | status) >= {a.atk_thresh}): {ATK}")
    print(f"    per-status P(dealt): " +
          ", ".join(f"{s}:{p:.2f}" for s, p in sorted(atk.items())))
    if not ATK:
        raise SystemExit("FAILED: no status qualifies as attacking; lower --atk_thresh")

    dx_root = np.abs(ROOT[:, F["agent_x"]] - ROOT[:, F["enemy_x"]])
    dx_post = np.abs(POST[..., F["agent_x"]] - POST[..., F["enemy_x"]])
    hit_dx = dx_post[(dealt | took)]
    RANGE = float(np.percentile(hit_dx, 90)) if hit_dx.size else float("nan")
    print(f"  in-range threshold = 90th pct of |dx| when damage lands = {RANGE:.0f} px "
          f"(median {np.median(hit_dx):.0f}, n={hit_dx.size})")

    c_a = np.isin(st_a, ATK)          # agent is in an attack state
    c_e = np.isin(st_e, ATK)
    in_rng = dx_post <= RANGE

    def gammas(M):
        mu = M.mean(axis=(1, 2), keepdims=True)
        al = M.mean(axis=2, keepdims=True) - mu
        be = M.mean(axis=1, keepdims=True) - mu
        return M - mu - al - be

    def score(M, label):
        W = M - M.mean(axis=(1, 2), keepdims=True)     # within-state
        G = gammas(M)
        wn = (W ** 2).sum(axis=(1, 2))
        gn = (G ** 2).sum(axis=(1, 2))
        active = wn > 1e-18
        share = float(gn[active].sum() / max(wn[active].sum(), 1e-30))
        contact = float((gn > 1e-18).mean())
        # pooled |gamma| relative to the reward scale, so a "share" of a tiny
        # payoff is not mistaken for a big effect
        mag = float(np.sqrt(gn.mean()))
        print(f"    {label:>28}  contact={contact:7.1%}  gamma_share={share:7.2%}  "
              f"|gamma|={mag:.5f}  within={float(np.sqrt(wn.mean())):.5f}")
        return dict(contact=contact, gamma_share=share, gamma_mag=mag,
                    within=float(np.sqrt(wn.mean())))

    out = {"n_roots": ns, "attack_statuses": ATK, "range_px": RANGE}

    # ---- TEST 5: contact and gamma vs ROOT separation ---------------------
    print(f"\n  TEST 5 -- start-state distribution: is contact a function of range?")
    qs = np.percentile(dx_root, [0, 20, 40, 60, 80, 100])
    out["test5"] = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (dx_root >= lo) & (dx_root <= hi)
        if m.sum() < 5:
            continue
        r = score(base[m], f"|dx| {lo:.0f}-{hi:.0f} (n={int(m.sum())})")
        r.update(dx_lo=float(lo), dx_hi=float(hi), n=int(m.sum()))
        out["test5"].append(r)

    # ---- TEST 1: counter-hit weighting ------------------------------------
    print(f"\n  TEST 1 -- counter-hit weight kappa (zero-sum preserved)")
    out["test1"] = []
    for k in [float(x) for x in a.kappas.split(",")]:
        M = 0.001 * (D_e * (1 + k * c_e) - D_a * (1 + k * c_a))
        r = score(M, f"kappa={k:g}")
        r["kappa"] = k
        out["test1"].append(r)

    # ---- TEST 2: pressure shaping -----------------------------------------
    print(f"\n  TEST 2 -- pressure weight beta, p = in range AND attacking")
    out["test2"] = []
    p_a = (c_a & in_rng).astype(np.float64)
    p_e = (c_e & in_rng).astype(np.float64)
    print(f"    p_agent fires on {p_a.mean():.1%} of branches, "
          f"p_enemy {p_e.mean():.1%}")
    for b in [float(x) for x in a.betas.split(",")]:
        M = base + b * (p_a - p_e)
        r = score(M, f"beta={b:g}")
        r["beta"] = b
        out["test2"].append(r)

    b0 = out["test1"][0] if out["test1"] and out["test1"][0]["kappa"] == 0 else None
    print(f"\n  BASELINE (current reward, all roots):")
    out["baseline"] = score(base, "current dense reward")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["collect", "analyze"], required=True)
    ap.add_argument("--state", default="Champion.Level1.RyuVsRyu.2Player")
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--n_roots", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npz", nargs="*", default=[])
    ap.add_argument("--kappas", default="0,0.5,1,2,4,8")
    ap.add_argument("--betas", default="0,0.001,0.003,0.01,0.03")
    ap.add_argument("--atk_thresh", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    (collect if a.mode == "collect" else analyze)(a)


if __name__ == "__main__":
    main()
