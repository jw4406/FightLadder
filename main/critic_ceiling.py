"""D1 -- what is the CEILING on predicting returns in this environment?

The critic scores held-out EV ~ 0 and its lookahead loses to a critic-free greedy
player. Two very different explanations remain, and every proposed fix depends on
which is true:

  (a) the network/training loop is broken -- fixable by architecture or optimizer
  (b) the RETURN ITSELF is close to unpredictable at gamma=0.99 -- in which case
      no architectural fix helps and the target definition has to change

This measures (b) directly. Freeze BOTH policies, roll many complete episodes,
then fit a FRESH value network end-to-end from pixels on realized Monte-Carlo
returns -- clean supervised regression, no bootstrapping, no replay, no moving
opponent, no V-trace. If a from-scratch supervised fit on exactly the same
architecture cannot beat ~0.2 EV, the ceiling is low and (b) is the answer.

Why gamma matters here: at gamma=0.99 the effective horizon is 1/(1-gamma) = 100
steps while episodes run 230-436. The terminal +-1 win/loss is discounted by
0.99^435 ~ 0.013 -- essentially invisible. So G is almost entirely dense HP-delta
shaping over the next ~100 steps, which depends on the opponent's stochastic
action choices and may be irreducibly noisy. The gamma sweep prices that.

SPLITS ARE BY EPISODE, NOT TIMESTEP. Every timestep inside one episode shares
nearly the same return, so a random timestep split puts near-duplicate targets on
both sides and lets any probe recover G by recognising which episode a state came
from. That inflates scores badly -- measured here, a frozen actor-CNN ridge read
+0.166 under a timestep split and +0.005 under an episode split. lbr_head_probe.py
and critic_diagnostics.py were both fixed to split by episode; any result from
those scripts predating that fix is inflated and should be regenerated.
"""
import os
import sys


def _peek(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


_d = _peek(sys.argv[1:])
if _d is not None and str(_d).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import copy
import json
import time
import argparse
import numpy as np
import torch as th

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.preprocessing import preprocess_obs
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups, REPO_ROOT, _b)
from lbr_head_probe import _ev
from critic_diagnostics import episode_split as _canonical_episode_split
from common.utils import reset_child_params


def collect_raw(venv, ops, n_steps, max_gb, progress=200):
    """Roll frozen policies, storing RAW observations so a fresh CNN can train.

    obs is (3, 100, 128) uint8 = 38.4 KB, so ~175k samples is 6.7 GB -- fine in
    RAM on this box, and it is what lets this test train from pixels rather than
    from a frozen encoder (which would only bound the HEAD, not the whole net).
    """
    n = venv.num_envs
    cap = n_steps * n
    per = int(np.prod(venv.observation_space.shape))
    need_gb = cap * per / 1e9
    if need_gb > max_gb:
        raise SystemExit(f"would need {need_gb:.1f} GB > --max_gb {max_gb}; "
                         f"lower --steps or --n_envs")
    print(f"   allocating {need_gb:.1f} GB for {cap:,} observations", flush=True)

    OBS = np.empty((cap,) + venv.observation_space.shape, dtype=np.uint8)
    R = np.zeros((n_steps, n), np.float32)
    D = np.zeros((n_steps, n), bool)
    V = np.zeros((n_steps, n), np.float32)
    S = np.zeros((n_steps, n, 5), np.float32)
    EP = np.zeros((n_steps, n), np.int64)

    rng = np.random.RandomState(0)
    ep_id = np.arange(n)
    nxt = n
    obs = venv.reset()
    for t in range(n_steps):
        OBS[t * n:(t + 1) * n] = obs
        V[t] = ops.values_ego(obs) * ops.sgn
        EP[t] = ep_id

        a_e = ops.sample_ego(obs, rng)
        a_a = ops.sample_adv(obs, rng)
        lbr_a, pol_a = (a_a, a_e) if ops.lbr_is_adv else (a_e, a_a)
        obs, r_l, r_r, d, infos = venv.step(ops.joint(lbr_a, pol_a))

        R[t] = ops.lbr_reward(r_l, r_r)
        d = np.asarray(d, bool)
        D[t] = d
        ahp = np.array([i.get("agent_hp", 0) for i in infos], float)
        ehp = np.array([i.get("enemy_hp", 0) for i in infos], float)
        ax = np.array([i.get("agent_x", 0) for i in infos], float)
        ex = np.array([i.get("enemy_x", 0) for i in infos], float)
        cd = np.array([i.get("round_countdown", 0) for i in infos], float)
        own, opp = (ehp, ahp) if ops.lbr_is_adv else (ahp, ehp)
        S[t] = np.stack([own - opp, own, opp, ax - ex, cd], 1)

        for j in np.nonzero(d)[0]:
            ep_id[j] = nxt
            nxt += 1
        if progress and (t + 1) % progress == 0:
            print(f"   {(t+1)*n:,} steps, {int(D.sum())} episodes", flush=True)
    return OBS, R, D, V, S, EP


def returns_at(R, D, gamma):
    """Realized discounted return, valid only where the episode completed."""
    S_len, n = R.shape
    G = np.zeros_like(R)
    valid = np.zeros_like(D)
    acc = np.zeros(n)
    seen = np.zeros(n, bool)
    for t in reversed(range(S_len)):
        acc = R[t] + gamma * acc * (~D[t])
        seen |= D[t]
        G[t] = acc
        valid[t] = seen
    return G, valid


def episode_split(ep, seed=0, fracs=(0.6, 0.2, 0.2)):
    """Split by EPISODE id. See the module docstring -- a timestep split leaks.

    Delegates to the canonical implementation in critic_diagnostics so the two
    scripts cannot drift; only the default train/val/test proportions differ
    (this one trains a network and wants more train data).
    """
    return _canonical_episode_split(ep, seed=seed, fracs=fracs)


def auc(scores, labels):
    """Rank-based AUC for binary labels in {0,1}. No sklearn dependency.

    Canonical copy: outcome_probe.py imports this. It lives HERE and not there
    because outcome_probe already imports collect_raw/encode from this module,
    so the reverse import would be circular.
    """
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, float)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks over ties so a constant predictor scores exactly 0.5
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_p, n_n = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def episode_labels(EPf, Rf, Sf, Df, verbose=True):
    """Per-episode win/loss label, with an independent cross-check.

    Two constructions: the sign of the final HP difference, and the sign of the
    reward on the terminal step (the +-1 outcome bonus dwarfs the ~0.176-scale
    dense term there). They agreed 99.6%/100% on the two checkpoints measured; if
    they ever diverge the label is untrustworthy and everything downstream is void.

    Unfinished episodes are dropped (no label exists yet) and draws are excluded
    from the binary task and reported separately.

    Returns (m, y, ep_m, meta) where `m` is a boolean mask over the FLAT sample
    axis, `y` is +-1 per kept sample, `ep_m` the matching episode ids.
    """
    uniq = np.unique(EPf)
    lab_hp, lab_rew, keep = {}, {}, []
    for e in uniq:
        idx = np.nonzero(EPf == e)[0]
        if not Df[idx].any():          # episode never finished in the window
            continue
        last = idx[-1]
        lab_hp[e] = np.sign(Sf[last, 0])
        lab_rew[e] = np.sign(Rf[last])
        keep.append(e)
    keep = np.array(keep)
    agree = np.mean([lab_hp[e] == lab_rew[e] for e in keep]) if keep.size else 0.0
    if verbose:
        print(f"   label cross-check: hp-sign vs terminal-reward-sign agree "
              f"{agree:.1%} over {keep.size} finished episodes", flush=True)

    lab = lab_rew                       # terminal reward is the more direct signal
    dist = {int(v): int(sum(1 for e in keep if lab[e] == v)) for v in (-1, 0, 1)}
    if verbose:
        print(f"   outcome distribution (loss/draw/win): {dist}", flush=True)

    m = np.isin(EPf, keep[[lab[e] != 0 for e in keep]]) if keep.size else np.zeros_like(Df)
    y = np.array([lab[e] for e in EPf[m]], float)
    ep_m = EPf[m]
    if verbose:
        print(f"   {m.sum():,} samples from {np.unique(ep_m).size} decisive episodes",
              flush=True)
    return m, y, ep_m, {"label_agreement": float(agree), "outcome_dist": dist,
                        "n_finished_episodes": int(keep.size),
                        "n_decisive_episodes": int(np.unique(ep_m).size)}


def _auc_score(pred, y):
    """score_fn adapter: ridge fits on +-1 targets, AUC wants {0,1} labels."""
    return auc(pred, (y > 0).astype(int))


def ridge_fit(X, y, tr, te, grid=(1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6), val=None,
              score_fn=None):
    """Ridge with alpha picked on `val` (never on the reported test split).

    `score_fn(pred, y_true)` defaults to explained variance; pass `_auc_score` to
    select and report on AUC instead (higher is better either way, so the
    argmax over the grid is unchanged).
    """
    score_fn = _ev if score_fn is None else score_fn
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    A = np.concatenate([(X[tr] - mu) / sd, np.ones((tr.size, 1))], 1)

    def score(alpha, idx):
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y[tr])
        B = np.concatenate([(X[idx] - mu) / sd, np.ones((idx.size, 1))], 1)
        return score_fn(B @ w, y[idx])

    sel = val if val is not None and val.size > 2 else te
    best = max(grid, key=lambda a: (score(a, sel) if np.isfinite(score(a, sel))
                                    else -np.inf))
    return score(best, te), float(best)


# The policy carries FIVE CNN encoders. These are the three that are actually
# trained and distinct under SPAR: the critic's, the ego actor's, and the
# adversary actor's. (`features_extractor` and `pi_features_extractor` also
# exist but are not on the SPAR forward path.) "pi" means the EGO actor
# specifically -- every "actor CNN" number in this repo refers to pi_ctrl.
ENCODERS = {
    "vf": "vf_features_extractor",
    "pi": "pi_ctrl_features_extractor",
    "dstb": "pi_dstb_features_extractor",
}


@th.no_grad()
def encode(policy, OBS, which, device, batch=512):
    """Run a frozen encoder over stored uint8 observations, in batches."""
    try:
        ext = getattr(policy, ENCODERS[which])
    except KeyError:
        raise KeyError(f"unknown encoder {which!r}; expected one of {sorted(ENCODERS)}")
    out = []
    for i in range(0, OBS.shape[0], batch):
        x = preprocess_obs(th.as_tensor(OBS[i:i + batch]).to(device),
                           policy.observation_space,
                           normalize_images=policy.normalize_images)
        out.append(ext(x).cpu().numpy())
    return np.concatenate(out)


def train_fresh_cnn(policy, head_key, OBS, G, tr, va, te, device,
                    epochs=40, batch=256, lr=3e-4, wd=1e-4, patience=6, seed=0,
                    task="return"):
    """Train a from-scratch copy of the EXACT critic architecture on realized G.

    Architecture is deep-copied from the loaded policy and then re-initialized, so
    this is the same encoder/trunk/head the run uses -- only the training signal
    differs (clean supervised MC returns instead of bootstrapped V-trace on stale
    replay against a moving opponent).

    task="return"  : G is the discounted return; MSE on standardized targets,
                     scored by explained variance.
    task="outcome" : G is a 0/1 win label; BCE-with-logits, scored by AUC. Targets
                     are NOT standardized (they are already 0/1) and the logit is
                     read raw, so the reported AUC is rank-based and unaffected by
                     any monotone miscalibration of the head.
    """
    if task not in ("return", "outcome"):
        raise ValueError(f"task must be 'return' or 'outcome', got {task!r}")
    th.manual_seed(seed)
    enc = copy.deepcopy(policy.vf_features_extractor).to(device)
    trunk = copy.deepcopy(policy.mlp_extractor).to(device)
    head = copy.deepcopy(policy.value_net[head_key]).to(device)
    # PopArt: a wrapped head would carry the trained run's (mu, sigma) into what
    # is supposed to be a FROM-SCRATCH ceiling, leaking the target scale.
    # reset_child_params does not touch buffers, so strip the wrapper and train
    # the bare Sequential -- this measures architecture capacity, not PopArt.
    if hasattr(head, "net") and isinstance(head.net, th.nn.Module):
        head = head.net
    for m in (enc, trunk, head):
        reset_child_params(m)
        m.train()

    # Standardizing a 0/1 label would rescale the BCE logit into a meaningless
    # range, so the outcome task keeps its targets raw.
    ym, ys = (0.0, 1.0) if task == "outcome" else (G[tr].mean(), G[tr].std() + 1e-12)
    params = list(enc.parameters()) + list(trunk.parameters()) + list(head.parameters())
    opt = th.optim.AdamW(params, lr=lr, weight_decay=wd)

    def forward(idx):
        x = preprocess_obs(th.as_tensor(OBS[idx]).to(device),
                           policy.observation_space,
                           normalize_images=policy.normalize_images)
        return head(trunk.forward_critic(enc(x))).squeeze(-1)

    @th.no_grad()
    def evaluate(idx):
        for m in (enc, trunk, head):
            m.eval()
        preds = []
        for i in range(0, idx.size, 512):
            preds.append(forward(idx[i:i + 512]).cpu().numpy())
        for m in (enc, trunk, head):
            m.train()
        p = np.concatenate(preds)
        if task == "outcome":
            return auc(p, G[idx])
        return _ev(p * ys + ym, G[idx])

    best_val, best_state, bad = -np.inf, None, 0
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, perm.size, batch):
            j = perm[i:i + batch]
            yt = th.as_tensor((G[j] - ym) / ys, dtype=th.float32, device=device)
            opt.zero_grad()
            if task == "outcome":
                loss = th.nn.functional.binary_cross_entropy_with_logits(forward(j), yt)
            else:
                loss = th.nn.functional.mse_loss(forward(j), yt)
            loss.backward()
            opt.step()
            tot += float(loss) * j.size
        v = evaluate(va)
        _ln, _mn = ("bce", "val_AUC") if task == "outcome" else ("mse", "val_EV")
        print(f"      epoch {ep+1:3d}  train_{_ln} {tot/perm.size:.5f}  {_mn} {v:+.4f}",
              flush=True)
        if v > best_val + 1e-4:
            best_val, bad = v, 0
            best_state = tuple(copy.deepcopy(m.state_dict()) for m in (enc, trunk, head))
        else:
            bad += 1
            if bad >= patience:
                print(f"      early stop (no val gain in {patience})", flush=True)
                break
    if best_state is not None:
        for m, s in zip((enc, trunk, head), best_state):
            m.load_state_dict(s)
    return evaluate(te), best_val


def main(argv=None):
    ap = argparse.ArgumentParser(description="D1: supervised ceiling on return prediction")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval_prot", type=str, default="False",
                    help="False = EGO perspective (matches the ego-perspective critic)")
    # Extended DOWN to 0. The original grid started at 0.9 and missed where the
    # dense signal actually lives: greedy (gamma=0, one-step reward, no critic)
    # beats critic-guided LBR at every checkpoint measured, and predictability
    # roughly doubled from gamma=0.99 (0.045) to 0.9 (0.104) at the balanced
    # checkpoint. gamma=0 makes G the immediate reward, which is the quantity
    # greedy actually uses -- so it anchors the low end of the curve.
    ap.add_argument("--gammas", type=str,
                    default="0.0,0.25,0.5,0.75,0.9,0.95,0.99,1.0")
    ap.add_argument("--max_gb", type=float, default=40.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--skip_cnn", action="store_true",
                    help="frozen-feature ceilings only (no end-to-end training)")
    ap.add_argument("--out", type=str, default="critic_ceiling.json")
    # GATE for the win-prediction head. "return" is the original D1 behaviour and
    # stays the default. "outcome" swaps the target for the per-episode win label
    # (MSE->BCE, EV->AUC) and answers the only question that decides whether the
    # head is worth building: is the AUC 0.72 from a ridge on FROZEN encoders a
    # floor with headroom above it, or is it already the aleatoric ceiling?
    #
    # Go is deterministic, so AlphaGo's value net kept paying for more capacity.
    # Here the opponent is genuinely mixed (advH ~ -0.76), so a PERFECT predictor
    # still cannot resolve their next action. If a freshly trained CNN cannot beat
    # the frozen-ridge baseline computed in the SAME run on the SAME split, the
    # head is capped at what a ridge already gives and phase 1 is not worth it.
    ap.add_argument("--target", type=str, default="return",
                    choices=("return", "outcome"))
    args = ap.parse_args(argv)

    data = load_from_zip_file(args.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]
    primary = float(data.get("gamma") or 0.99)

    venv = build_lbr_venv(state, args.n_envs)
    try:
        model, _ = load_checkpoint(args.ckpt, venv, args.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=_b(args.eval_prot))
        t0 = time.time()
        OBS, R, D, V, S, EP = collect_raw(venv, ops, args.steps, args.max_gb)
    finally:
        venv.close()
    collect_s = time.time() - t0
    policy = ops.p
    head_key = f"{label}_{head_idx}"

    n_eps = int(D.sum())
    print(f"\n   collected {n_eps} episodes in {collect_s/60:.1f} min", flush=True)

    EPf = EP.reshape(-1)
    Sf = S.reshape(-1, S.shape[-1])
    Vf = V.reshape(-1)

    res = {"checkpoint": os.path.basename(args.ckpt), "matchup": label,
           "num_timesteps": data.get("num_timesteps"), "episodes": n_eps,
           "primary_gamma": primary, "collect_s": round(collect_s, 1),
           "eval_prot": args.eval_prot, "by_gamma": {}}

    print("\n   computing frozen encoder features ...", flush=True)
    FEAT = {"critic_cnn": encode(policy, OBS, "vf", args.device),
            "ego_actor_cnn": encode(policy, OBS, "pi", args.device),
            "adv_actor_cnn": encode(policy, OBS, "dstb", args.device)}

    if args.target == "outcome":
        Rf, Df = R.reshape(-1), D.reshape(-1)
        m, y, ep_m, meta = episode_labels(EPf, Rf, Sf, Df)
        if np.unique(ep_m).size < 40:
            raise SystemExit(f"only {np.unique(ep_m).size} decisive episodes; "
                             f"AUC would be noise. Raise --steps.")
        idx_all = np.nonzero(m)[0]
        tr, va, te = episode_split(ep_m, seed=args.seed)
        res.update(meta)
        res["target"] = "outcome"
        res.pop("by_gamma", None)

        # WIN RATE belongs next to every AUC. A collapsed regime makes outcome
        # trivially predictable (AUC 0.93 at 10.08M where the ego lost 96% of
        # episodes) -- a headline AUC read without it is an artifact.
        win_rate = float((y > 0).mean())
        res["win_rate"] = win_rate
        res["balanced"] = bool(0.3 <= win_rate <= 0.7)
        print(f"   win rate {win_rate:.3f} "
              f"({'BALANCED' if res['balanced'] else 'IMBALANCED -- AUC not comparable'})",
              flush=True)

        _S = Sf[m]
        _Sz = (_S - _S[tr].mean(0)) / (_S[tr].std(0) + 1e-8)
        _VF = FEAT["critic_cnn"][m]
        _cat = np.concatenate([_VF, _Sz.astype(_VF.dtype)], axis=1)
        probes = {}
        for nm, X in (("hp_diff_linear", _S[:, :1]),
                      ("state_linear", _S),
                      ("frozen_critic_cnn", _VF),
                      ("concat_cnn_plus_state", _cat),
                      ("frozen_ego_actor_cnn", FEAT["ego_actor_cnn"][m]),
                      ("frozen_adv_actor_cnn", FEAT["adv_actor_cnn"][m])):
            sc, alpha = ridge_fit(X, y, tr, te, val=va, score_fn=_auc_score)
            probes[nm] = {"auc": sc, "alpha": alpha}
            print(f"   {nm:24s} AUC {sc:>7.3f}  (alpha {alpha:g})", flush=True)
        probes["trained_critic_V"] = {"auc": _auc_score(Vf[m][te], y[te]),
                                      "alpha": None}
        print(f"   {'trained_critic_V':24s} AUC "
              f"{probes['trained_critic_V']['auc']:>7.3f}", flush=True)
        res["probes"] = probes
        frozen_best = max(v["auc"] for k, v in probes.items()
                          if k != "trained_critic_V" and np.isfinite(v["auc"]))
        res["frozen_ridge_best_auc"] = frozen_best

        if not args.skip_cnn:
            print(f"\n   training FRESH critic end-to-end on the OUTCOME label "
                  f"({tr.size:,} train / {va.size:,} val / {te.size:,} test)",
                  flush=True)
            t1 = time.time()
            y01 = (y > 0).astype(np.float32)
            test_auc, val_auc = train_fresh_cnn(
                policy, head_key, OBS[idx_all], y01, tr, va, te, args.device,
                epochs=args.epochs, seed=args.seed, task="outcome")
            res["fresh_cnn"] = {"test_auc": test_auc, "best_val_auc": val_auc,
                                "train_s": round(time.time() - t1, 1)}
            res["headroom_vs_frozen_ridge"] = test_auc - frozen_best
            print(f"\n   FRESH supervised outcome head: test AUC = {test_auc:.4f} "
                  f"(best val {val_auc:.4f})", flush=True)
            print(f"   best frozen ridge          : {frozen_best:.4f}")
            print(f"   HEADROOM                   : "
                  f"{res['headroom_vs_frozen_ridge']:+.4f}")
            print("\n" + "=" * 74)
            if not res["balanced"]:
                print("VERDICT: UNUSABLE -- imbalanced regime, AUC is an artifact.")
            elif res["headroom_vs_frozen_ridge"] >= 0.05:
                print("VERDICT: BUILD -- end-to-end training beats the frozen ridge.")
            else:
                print("VERDICT: STOP -- no headroom over a ridge on frozen features.")
            print("=" * 74)

        p = os.path.join(REPO_ROOT, args.out)
        with open(p, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n   wrote {p}", flush=True)
        return res

    for g in [float(x) for x in args.gammas.split(",")]:
        G, valid = returns_at(R, D, g)
        m = valid.reshape(-1)
        Gm, epm = G.reshape(-1)[m], EPf[m]
        tr, va, te = episode_split(epm, seed=args.seed)
        row = {"n": int(m.sum()), "n_episodes": int(np.unique(epm).size),
               "G_std": float(Gm.std()),
               "trained_critic_V": _ev(Vf[m][te], Gm[te])}
        # CONCAT PROBE: does appending the 5 hand state scalars to the critic's
        # CNN features improve return prediction? This is the offline kill-switch
        # for the proposed --critic_state_features change: if [VF || S] does not
        # beat VF on held-out EPISODES, wiring the concat into the training loop
        # cannot help and the work is pointless.
        #
        # Scalars are standardized to the feature scale FIRST. hp and countdown
        # are raw game units (hundreds) while vf_features are post-activation
        # (order 1); concatenating unnormalized lets the scalars dominate the
        # first layer by magnitude alone, which would make any gain
        # uninterpretable. ridge_fit standardizes per-column anyway, but doing it
        # here keeps the reported input faithful to what training would see.
        _S = Sf[m]
        _Sz = (_S - _S[tr].mean(0)) / (_S[tr].std(0) + 1e-8)
        _VF = FEAT["critic_cnn"][m]
        _cat = np.concatenate([_VF, _Sz.astype(_VF.dtype)], axis=1)

        for nm, X in (("hp_diff_linear", Sf[m][:, :1]),
                      ("state_linear", Sf[m]),
                      ("frozen_critic_cnn", _VF),
                      ("concat_cnn_plus_state", _cat),
                      ("frozen_ego_actor_cnn", FEAT["ego_actor_cnn"][m]),
                      ("frozen_adv_actor_cnn", FEAT["adv_actor_cnn"][m])):
            sc, alpha = ridge_fit(X, Gm, tr, te, val=va)
            row[nm] = sc
            row[nm + "_alpha"] = alpha
        res["by_gamma"][f"{g}"] = row
        print(f"   gamma={g:<6} n={row['n']:>7,} eps={row['n_episodes']:>4} "
              f"G_std={row['G_std']:.4f}  trainedV={row['trained_critic_V']:+.3f}  "
              f"criticCNN={row['frozen_critic_cnn']:+.3f}  "
              f"CONCAT={row['concat_cnn_plus_state']:+.3f} "
              f"(d={row['concat_cnn_plus_state']-row['frozen_critic_cnn']:+.3f})  "
              f"egoCNN={row['frozen_ego_actor_cnn']:+.3f}  "
              f"state={row['state_linear']:+.3f}", flush=True)

    if not args.skip_cnn:
        G, valid = returns_at(R, D, primary)
        m = valid.reshape(-1)
        idx_all = np.nonzero(m)[0]
        Gm, epm = G.reshape(-1)[m], EPf[m]
        tr, va, te = episode_split(epm, seed=args.seed)
        print(f"\n   training FRESH critic end-to-end at gamma={primary} "
              f"({tr.size:,} train / {va.size:,} val / {te.size:,} test)", flush=True)
        t1 = time.time()
        test_ev, val_ev = train_fresh_cnn(
            policy, head_key, OBS[idx_all], Gm, tr, va, te, args.device,
            epochs=args.epochs, seed=args.seed)
        res["fresh_cnn"] = {"test_ev": test_ev, "best_val_ev": val_ev,
                            "gamma": primary,
                            "train_s": round(time.time() - t1, 1)}
        print(f"\n   FRESH supervised critic: test EV = {test_ev:+.4f} "
              f"(best val {val_ev:+.4f})", flush=True)

    p = os.path.join(REPO_ROOT, args.out)
    with open(p, "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 74)
    print(f"CEILING  {res['checkpoint']}   {n_eps} episodes   splits BY EPISODE")
    print("=" * 74)
    print(f"  {'gamma':>7} {'G_std':>8} {'trainedV':>9} {'state':>8} "
          f"{'criticCNN':>10} {'CONCAT':>8} {'delta':>7} {'egoCNN':>8}")
    for g, r in res["by_gamma"].items():
        print(f"  {g:>7} {r['G_std']:>8.4f} {r['trained_critic_V']:>+9.3f} "
              f"{r['state_linear']:>+8.3f} {r['frozen_critic_cnn']:>+10.3f} "
              f"{r['concat_cnn_plus_state']:>+8.3f} "
              f"{r['concat_cnn_plus_state']-r['frozen_critic_cnn']:>+7.3f} "
              f"{r['frozen_ego_actor_cnn']:>+8.3f}")
    if "fresh_cnn" in res:
        fc = res["fresh_cnn"]
        print(f"\n  FRESH end-to-end supervised critic @ gamma={fc['gamma']}: "
              f"test EV {fc['test_ev']:+.4f}")
        print("\n  Reading:")
        print("    fresh EV high  -> the ARCHITECTURE is fine; the training loop"
              " (v-trace / replay / moving opponent) is the problem")
        print("    fresh EV low   -> the RETURN is near-unpredictable at this"
              " gamma; architectural fixes will not help and the target must change")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
