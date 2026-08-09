"""Critic diagnostic suite: is the value function of a checkpoint any good?

This session established that the two statistics people reach for first are both
misleading on their own:

  * the `ego_explained_variance` in the training log is IN-BATCH -- it read 0.50
    while held-out EV on fresh states was 0.05.
  * a held-out EV measured on a short rollout is NOISE. EV(V, G) regresses against
    EPISODE RETURNS, and every timestep inside one episode shares a return, so
    `--steps 800 --n_envs 8` is ~15-22 effective samples, not the ~5000 it reports.
    The same run scattered over -0.25..+0.29 at that setting and flattened to
    ~0.01/-0.06/0.16 when re-run with 10x the episodes.

So this suite (a) always reports the EPISODE count and refuses to render a verdict
below a floor, (b) attaches a bootstrap CI resampled over EPISODES rather than
timesteps, and (c) never reports a single number where a control exists.

Four tiers, cheap to expensive:

  0  static      weights only, no env. Catches an untrained/random-init head.
  1  prediction  does V predict its own target, and realized returns?
  2  representation  WHERE in [CNN -> trunk -> head] the signal dies.
  3  behavioral  does the critic improve DECISIONS? (LBR vs shuffled vs greedy)

Tier 3 is the one that actually settled the question on spar_Ry_Sa: LBR with the
real critic scored indistinguishably from LBR with a randomly PERMUTED critic,
while greedy (gamma=0, critic unused) beat both. No regression statistic carries
that much weight -- it is measured from realized emulator outcomes.

Tiers 1 and 2 share ONE rollout. The five predecessor scripts (target_probe,
lbr_head_probe, lbr_feature_probe, ctrl_probe, rank_probe) each collected their
own, which is the dominant cost and pure waste.
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


def _peek_int(argv, flag, default):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
    return default


# BLAS threading must be pinned BEFORE numpy is imported. Left alone, the ridge
# solves grab ~14 cores and starve any concurrent LBR sweep -- measured load 44
# on a 32-core box. The ridge problems here are 513x513; extra threads buy
# nothing and cost the neighbours a lot.
_bt = str(_peek_int(sys.argv[1:], "--blas_threads", 4))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _bt)

import re
import glob
import json
import time
import argparse
import subprocess
import numpy as np
import torch as th

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.preprocessing import preprocess_obs
from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                 PolicyOps, resolve_matchups, REPO_ROOT, _b)
# episode_split lives in lbr_head_probe (the lowest-level of these modules, and
# the one the others already depend on) so there is exactly one implementation.
# Re-exported here because critic_ceiling.py imports it from this module.
from lbr_head_probe import ridge_ev, mlp_ev, _ev, episode_split
from rank_probe import eff_rank

# Below this many COMPLETED episodes, every EV number in tier 1 is noise and the
# suite reports UNDERPOWERED instead of a verdict. 15-22 episodes produced a
# +-0.27 swing on this codebase; 150 flattened it.
MIN_EPISODES_FOR_VERDICT = 100
# Agent steps/episode above which the game is treated as STALLING (episodes
# running to the round timer rather than ending in a knockout). Calibrated on
# measured values, not guessed: healthy decisive play sits at 161-320, LBR
# timeout episodes hit 515-538, and 538 x 8 frames / 60 fps = 72 s against rounds
# of ~27 s. 400 sits between the two regimes with margin on both sides.
STALL_STEPS_PER_EPISODE = 400

# A FIXED ridge cannot be used here. At ridge=1.0 with 512 features and 600 rows
# the CNN stage scored negative and was misread as "the encoder learned nothing";
# ridge=10 flipped it to +0.525. And ridge=10 on 967 training rows produces
# EV = -60 -- catastrophic overfit -- while the same alpha is fine at 5000 rows.
# So the strength is SELECTED per stage on a validation split, and the chosen
# alpha is reported alongside every score. Cross-stage comparisons are only
# meaningful when each stage got its own best-case regularization.
DEFAULT_RIDGE = 10.0
RIDGE_GRID = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def ridge_ev_sel(X, y, tr, va, te, grid=RIDGE_GRID):
    """Ridge EV with alpha selected on `va`, then refit on tr+va, scored on te.

    All three index sets must come from episode_split -- selecting alpha on a
    LEAKY validation split picks an alpha that is far too small, which is how a
    fixed ridge=10 produced EV = -60 on 967 rows earlier.
    """
    if tr.size < 4 or te.size < 4:
        return float("nan"), float("nan")
    if va.size < 4:
        return ridge_ev(X, y, tr, te, grid[len(grid) // 2]), grid[len(grid) // 2]
    best_a, best_s = None, -np.inf
    for a in grid:
        s = ridge_ev(X, y, tr, va, a)
        if np.isfinite(s) and s > best_s:
            best_a, best_s = a, s
    if best_a is None:
        best_a = grid[-1]
    full = np.concatenate([tr, va])
    return ridge_ev(X, y, full, te, best_a), float(best_a)


def _ridge_ev_centered(X, y, tr, te, alpha):
    """Ridge WITHOUT per-feature standardization -- centering only.

    lbr_head_probe.ridge_ev divides each column by its own std. That is the
    right default for heterogeneous features, but it is actively harmful on a
    near-collinear representation: at effRank 1.8 most of the 512 columns are
    noise, and rescaling each to unit variance amplifies that noise to parity
    with the one direction that matters. Centering only preserves the relative
    scale, so the useful direction survives the shrinkage.
    """
    mu = X[tr].mean(0)
    sd = X[tr].std() + 1e-8            # ONE global scale, not per-feature
    A = np.concatenate([(X[tr] - mu) / sd, np.ones((tr.size, 1))], 1)
    B = np.concatenate([(X[te] - mu) / sd, np.ones((te.size, 1))], 1)
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y[tr])
    return _ev(B @ w, y[te])


def best_linear_ev(X, y, tr, va, te, grid=RIDGE_GRID):
    """Tightest linear readout across both scalings, alpha selected on val.

    A linear readout's score is a LOWER BOUND on how much linearly-decodable
    signal the representation holds, so taking the max over several valid fits
    is the correct estimator -- whichever fit is worse is simply a looser bound,
    not evidence about the representation.

    Returns (score, alpha, which).
    """
    if tr.size < 4 or te.size < 4:
        return float("nan"), float("nan"), "none"
    best = (-np.inf, float("nan"), "none")
    for name, fn in (("std", ridge_ev), ("centered", _ridge_ev_centered)):
        for a in grid:
            v = fn(X, y, tr, va, a)
            if not np.isfinite(v):
                continue
            s = fn(X, y, np.concatenate([tr, va]), te, a)
            if np.isfinite(s) and v > best[0]:
                best = (v, a, name)
    if best[2] == "none":
        return float("nan"), float("nan"), "none"
    fn = ridge_ev if best[2] == "std" else _ridge_ev_centered
    return fn(X, y, np.concatenate([tr, va]), te, best[1]), float(best[1]), best[2]


# --------------------------------------------------------------------------
# tier 0 -- static
# --------------------------------------------------------------------------
def tier0_static(ckpt_path, data):
    """Weight-level checks. No env, no rollout, runs in about a second."""
    out = {}
    params = load_from_zip_file(ckpt_path, device="cpu")[1]
    policy = params.get("policy", {})

    groups = {
        "vf_features_extractor": [],
        "mlp_extractor.value_net": [],
        "value_net": [],
        "pi_ctrl_features_extractor": [],
    }
    for k, v in policy.items():
        for g in groups:
            if k.startswith(g):
                groups[g].append((k, v))

    out["modules"] = {}
    for g, items in groups.items():
        if not items:
            out["modules"][g] = {"present": False}
            continue
        norms, zero_rows, total_rows, n_params = [], 0, 0, 0
        for k, v in items:
            t = v.detach().float()
            n_params += t.numel()
            norms.append(float(t.norm()))
            if t.dim() >= 2:
                flat = t.reshape(t.shape[0], -1)
                zero_rows += int((flat.abs().max(dim=1).values < 1e-8).sum())
                total_rows += flat.shape[0]
        out["modules"][g] = {
            "present": True,
            "n_tensors": len(items),
            "n_params": int(n_params),
            "weight_norm_total": float(np.sqrt(sum(n * n for n in norms))),
            "dead_row_frac": (zero_rows / total_rows) if total_rows else 0.0,
        }

    # An LSTM value head cannot be loaded by this path at all -- fail fast with a
    # readable message rather than a traceback deep inside SB3's load.
    #
    # Must anchor on "value_net." exactly. A substring test for "value" also hits
    # q_value_net.<matchup>.0.weight_hh_l0 -- a SEPARATE LSTM Q head that every
    # single-opponent SPAR checkpoint carries -- and would falsely condemn every
    # loadable checkpoint in the repo.
    out["has_lstm_value_head"] = any(
        k.startswith("value_net.") and ("lstm" in k.lower() or "weight_hh" in k)
        for k in policy
    )
    out["has_q_value_net"] = any(k.startswith("q_value_net.") for k in policy)
    out["n_policy_tensors"] = len(policy)
    out["num_timesteps"] = data.get("num_timesteps")
    return out


# --------------------------------------------------------------------------
# shared rollout -- serves tiers 1 and 2
# --------------------------------------------------------------------------
def head_taps(head):
    """Post-activation tap points INSIDE value_net[matchup].

    The head is Sequential(Linear 512->512, act, Linear 512->512, act,
    Linear 512->512, act, Linear 512->1). Every comparison made before this
    jumped straight from the trunk output (512-d) to V (scalar), across three
    unmeasured layers -- so "the head destroys information" could not be
    distinguished from "the final 512->1 projection points the wrong way",
    and those imply completely different fixes.

    Tapping the activations (not the Linears) gives the representation actually
    handed to the next layer, and excludes the final scalar automatically since
    no activation follows it.

    PopArt: when --popart is on, value_net[matchup] is a PopArtHead wrapping the
    Sequential as `.net`, so the children here would be (net,) -- one non-Linear
    child, silently yielding a single bogus tap instead of h1/h2/h3. Descend
    first. Detected structurally rather than by isinstance so this file does not
    have to import from the vendored SB3 tree.
    """
    inner = getattr(head, "net", None)
    if inner is not None and isinstance(inner, th.nn.Module):
        head = inner
    return [(f"h{i}", m) for i, m in enumerate(
        (c for c in head.children() if not isinstance(c, th.nn.Linear)), start=1)]



def collect_all(venv, ops, n_steps, seed, progress_every=200, head=None):
    """One pass. Returns per-timestep features, critic output, rewards, dones.

    Episode ids are tracked per env so the bootstrap can resample whole EPISODES;
    resampling timesteps would badly understate the CI, since returns are near
    constant within an episode.
    """
    rng = np.random.RandomState(seed)
    n = venv.num_envs
    VF, LV, PI, PD, V, R, D, S, EP = [], [], [], [], [], [], [], [], []

    # Taps inside the value head. These fire during ops.values_ego() below, so
    # they capture exactly the activations the trained head produces on its own
    # forward pass -- no re-implementation of the head that could drift from it.
    HEAD, handles = {}, []
    if head is not None:
        for nm, mod in head_taps(head):
            HEAD[nm] = []
            handles.append(mod.register_forward_hook(
                lambda _m, _i, o, _k=nm: HEAD[_k].append(o.detach().cpu().numpy())))
    ep_id = np.arange(n)          # unique episode id per env slot
    next_ep = n
    obs = venv.reset()
    for t in range(n_steps):
        with th.no_grad():
            x = preprocess_obs(th.as_tensor(obs).to(ops.device),
                               ops.p.observation_space,
                               normalize_images=ops.p.normalize_images)
            vf = ops.p.vf_features_extractor(x)
            VF.append(vf.cpu().numpy())
            LV.append(ops.p.mlp_extractor.forward_critic(vf).cpu().numpy())
            # PI is the EGO actor's encoder; PD the ADVERSARY's. Both are probed
            # as references -- on this run the adversary is the side that is
            # actually winning, so its representation is the more interesting
            # comparison of the two.
            PI.append(ops.p.pi_ctrl_features_extractor(x).cpu().numpy())
            PD.append(ops.p.pi_dstb_features_extractor(x).cpu().numpy())
        V.append(ops.values_ego(obs) * ops.sgn)
        EP.append(ep_id.copy())

        a_e = ops.sample_ego(obs, rng)
        a_a = ops.sample_adv(obs, rng)
        lbr_a, pol_a = (a_a, a_e) if ops.lbr_is_adv else (a_e, a_a)
        obs, r_l, r_r, d, infos = venv.step(ops.joint(lbr_a, pol_a))

        R.append(ops.lbr_reward(r_l, r_r))
        d = np.asarray(d, dtype=bool)
        D.append(d)
        ahp = np.array([i.get("agent_hp", 0) for i in infos], float)
        ehp = np.array([i.get("enemy_hp", 0) for i in infos], float)
        ax = np.array([i.get("agent_x", 0) for i in infos], float)
        ex = np.array([i.get("enemy_x", 0) for i in infos], float)
        cd = np.array([i.get("round_countdown", 0) for i in infos], float)
        own, opp = (ehp, ahp) if ops.lbr_is_adv else (ahp, ehp)
        S.append(np.stack([own - opp, own, opp, ax - ex, cd], axis=1))

        for j in np.nonzero(d)[0]:
            ep_id[j] = next_ep
            next_ep += 1
        if progress_every and (t + 1) % progress_every == 0:
            print(f"   collect {(t+1)*n} steps, {int(np.array(D).sum())} episodes",
                  flush=True)
    for h in handles:
        h.remove()

    out = dict(VF=np.array(VF), LV=np.array(LV), PI=np.array(PI),
               PD=np.array(PD), V=np.array(V), R=np.array(R), D=np.array(D),
               S=np.array(S), EP=np.array(EP))
    # One tap capture per vec-step, shaped (n, 512) -- same layout as VF/LV. A
    # mismatch means values_ego() chunked the batch differently and the rows
    # would not line up with the targets, so fail loudly rather than silently
    # correlate misaligned arrays.
    for nm, chunks in HEAD.items():
        arr = np.array(chunks)
        if arr.shape[:2] != out["VF"].shape[:2]:
            raise RuntimeError(f"head tap {nm} shape {arr.shape} does not align "
                               f"with VF {out['VF'].shape}")
        out[f"HEAD_{nm}"] = arr
    return out


def derive_targets(raw, gamma, T):
    """Realized returns G and the V-trace target, with episode-boundary masking.

    Mirrors target_probe.py exactly: with ratios ~1 the V-trace target telescopes
    to a T-step discounted reward sum plus gamma^T * V(x_{t+T}).
    """
    R, D, V = raw["R"], raw["D"], raw["V"]
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

    rew = np.zeros_like(R)
    boot = np.zeros_like(R)
    ok = np.zeros_like(D)
    for t in range(S_len - T):
        alive = np.ones(n, bool)
        disc = np.ones(n)
        s = np.zeros(n)
        for k in range(T):
            s += disc * R[t + k] * alive
            disc *= gamma
            alive &= ~D[t + k]
        rew[t] = s
        boot[t] = (gamma ** T) * V[t + T] * alive
        ok[t] = True

    return dict(G=G, tgt=rew + boot, rew=rew, boot=boot,
                mask_G=valid, mask_tgt=(valid & ok))


# --------------------------------------------------------------------------
# tier 1 -- prediction quality
# --------------------------------------------------------------------------
def _affine_r2(pred, y, tr, te):
    """R^2 of the best affine rescaling a*pred + b, fit on train, scored on test.

    EV is a variance ratio, so with vtrace_v_target_std ~ 0.03 a pure SCALE or
    BIAS error drives EV to ~0 even when V ranks states perfectly. Comparing this
    against raw EV separates a calibration failure from an information failure.
    """
    A = np.stack([pred[tr], np.ones(tr.size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    fit = coef[0] * pred[te] + coef[1]
    return _ev(fit, y[te]), float(coef[0]), float(coef[1])


def _bootstrap_ev(pred, y, ep, n_boot=200, seed=0):
    """Percentile CI for EV, resampling whole EPISODES with replacement."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(ep)
    if uniq.size < 3:
        return (float("nan"), float("nan"))
    by_ep = {e: np.nonzero(ep == e)[0] for e in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([by_ep[e] for e in pick])
        vals.append(_ev(pred[idx], y[idx]))
    return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))


def tier1_prediction(raw, der, ridge, seed):
    m_t = der["mask_tgt"].reshape(-1)
    m_g = der["mask_G"].reshape(-1)
    Vf = raw["V"].reshape(-1)
    Gf = der["G"].reshape(-1)
    tf = der["tgt"].reshape(-1)
    bf = der["boot"].reshape(-1)
    ep = raw["EP"].reshape(-1)
    Sf = raw["S"].reshape(raw["S"].shape[0] * raw["S"].shape[1], -1)

    n_episodes = int(raw["D"].sum())
    # Mean AGENT steps per episode. The cheapest degeneracy detector we have:
    # when both policies stop engaging, episodes stop ending in knockouts and run
    # to the round timer instead. Measured on the popart run, this went 161
    # (9.6M, decisive) -> 515 (34.56M, timing out) while EV climbed to 0.957 --
    # which was the GAME becoming deterministic, not the critic improving.
    steps_per_episode = (float(raw["D"].size) / n_episodes) if n_episodes else float("inf")
    out = {"n_episodes": n_episodes,
           "n_samples_G": int(m_g.sum()),
           "n_samples_tgt": int(m_t.sum()),
           "steps_per_episode": steps_per_episode,
           "underpowered": n_episodes < MIN_EPISODES_FOR_VERDICT}

    V_g, G_g, ep_g = Vf[m_g], Gf[m_g], ep[m_g]
    V_t, T_t, G_t = Vf[m_t], tf[m_t], Gf[m_t]

    out["ev_V_vs_G"] = _ev(V_g, G_g)
    out["ev_V_vs_target"] = _ev(V_t, T_t)
    out["ev_target_vs_G"] = _ev(T_t, G_t)
    out["bootstrap_share_of_target_var"] = float(
        bf[m_t].var() / max(tf[m_t].var(), 1e-30))
    out["ev_V_vs_G_ci90"] = _bootstrap_ev(V_g, G_g, ep_g, seed=seed)

    # shape vs scale
    if V_g.std() > 1e-12 and G_g.std() > 1e-12:
        out["corr_V_G"] = float(np.corrcoef(V_g, G_g)[0, 1])
    else:
        out["corr_V_G"] = float("nan")
    # Episode-level split -- see episode_split(). Everything FITTED below
    # (calibration, baselines) is affected by the leak, not just tier 2.
    tr, va, te = episode_split(ep_g, seed=seed)
    out["n_episodes_train"] = int(np.unique(ep_g[tr]).size) if tr.size else 0
    out["n_episodes_test"] = int(np.unique(ep_g[te]).size) if te.size else 0
    r2, slope, icept = _affine_r2(V_g, G_g, np.concatenate([tr, va]), te)
    out["affine_r2_V_G"] = r2
    out["calib_slope"] = slope
    out["calib_intercept"] = icept

    out["std_V"] = float(V_g.std())
    out["std_G"] = float(G_g.std())
    out["std_target"] = float(tf[m_t].std())

    # baselines the critic has to beat. constant-mean is EV=0 by construction.
    Sg = Sf[m_g]
    out["baseline_constant"] = 0.0
    out["baseline_hp_diff_linear"], _ = ridge_ev_sel(Sg[:, :1], G_g, tr, va, te)
    out["baseline_state_linear"], _ = ridge_ev_sel(Sg, G_g, tr, va, te)
    return out


def read_inbatch_ev_from_log(log_path, num_timesteps):
    """Pull the training-time (in-batch) EV for the block nearest this checkpoint.

    In-batch EV genuinely cannot be reconstructed from a checkpoint -- it is
    measured on the replay samples the critic was fit to, and the buffer is not
    saved. Reading it from the log is the only faithful source, so the
    in-batch/held-out GAP is only reported when a log is supplied.
    """
    if not log_path or not os.path.isfile(log_path):
        return None
    best, cur = None, {}
    for line in open(log_path, errors="ignore"):
        if not line.startswith("|"):
            continue
        m = re.match(r"\|\s+(\S+)\s+\|\s+(\S+)\s+\|", line)
        if not m:
            continue
        k = m.group(1).rstrip("/").split("/")[-1]
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        if k in ("ego_explained_variance", "adv_explained_variance", "total_timesteps"):
            cur[k] = v
        if k == "total_timesteps" and "ego_explained_variance" in cur:
            d = abs(cur["total_timesteps"] - (num_timesteps or 0))
            if best is None or d < best[0]:
                best = (d, dict(cur))
            cur = {}
    if best is None:
        return None
    return {"in_batch_ego_ev": best[1].get("ego_explained_variance"),
            "log_timesteps": best[1].get("total_timesteps"),
            "timestep_delta": best[0]}


# --------------------------------------------------------------------------
# tier 2 -- where the signal dies
# --------------------------------------------------------------------------
def tier2_representation(raw, der, ridge, seed, device, run_mlp=True, affine_ref=None):
    m_g = der["mask_G"].reshape(-1)
    G = der["G"].reshape(-1)[m_g]
    V = raw["V"].reshape(-1)[m_g]

    def flat(key):
        a = raw[key]
        return a.reshape(a.shape[0] * a.shape[1], -1)[m_g]

    VF, LV, PI, PD = flat("VF"), flat("LV"), flat("PI"), flat("PD")
    n = G.size
    # Episode-level split, same reason as tier 1. Under the old timestep
    # permutation these probes read ~0.09-0.17 where the honest value is ~0.03.
    ep = raw["EP"].reshape(-1)[m_g]
    tr, va, te = episode_split(ep, seed=seed)

    # Random projection of the SAME shape as the trained trunk. Without it,
    # "the trunk preserves information" is unfalsifiable -- a random map of the
    # same width preserves a lot, so the trained trunk must be scored against it.
    rs = np.random.RandomState(1234)
    W = rs.randn(VF.shape[1], LV.shape[1]).astype(np.float32) / np.sqrt(VF.shape[1])
    proj = VF @ W
    RP = np.maximum(proj, 0.01 * proj)      # LeakyReLU, matched to the trunk

    out = {"n": int(n), "split": "by-episode",
           "n_episodes_train": int(np.unique(ep[tr]).size) if tr.size else 0,
           "n_episodes_test": int(np.unique(ep[te]).size) if te.size else 0,
           "ridge_selection": "alpha on episode-held-out val", "alphas": {}}
    for key, X in (("s1_cnn_ridge", VF), ("s2_trunk_ridge", LV),
                   ("ctrl_random_proj_ridge", RP), ("ref_ego_actor_cnn_ridge", PI),
                   ("ref_adv_actor_cnn_ridge", PD)):
        score, alpha = ridge_ev_sel(X, G, tr, va, te)
        out[key] = score
        out["alphas"][key] = alpha
    out["s3_trained_V"] = _ev(V[te], G[te])
    if run_mlp:
        fit = np.concatenate([tr, va])
        out["s1_cnn_mlp"] = mlp_ev(VF, G, fit, te, seed, device=device)
        out["s2_trunk_mlp"] = mlp_ev(LV, G, fit, te, seed, device=device)

    for nm, X in (("vf_features", VF), ("latent_vf", LV), ("random_proj", RP)):
        pr, d95, d99 = eff_rank(X)
        out[f"rank_{nm}"] = {
            "dim": int(X.shape[1]),
            "dead_frac": float((X.std(0) < 1e-6).mean()),
            "std": float(X.std()),
            "eff_rank": float(pr), "d95": d95, "d99": d99,
        }

    # Readouts INSIDE the head, on the same target/split/alpha procedure as the
    # trunk. This is the only comparison that can separate "the head's layers
    # destroy information" from "the final 512->1 projection points the wrong
    # way" -- every earlier comparison jumped trunk -> scalar across three
    # unmeasured layers, absorbing both effects into one number.
    head_keys = sorted(k for k in raw if k.startswith("HEAD_"))
    out["head_layers"] = {}
    for k in head_keys:
        A = raw[k]
        X = A.reshape(A.shape[0] * A.shape[1], -1)[m_g]
        score, alpha, which = best_linear_ev(X, G, tr, va, te)
        pr, d95, _d99 = eff_rank(X)
        out["head_layers"][k[5:]] = {"ridge_ev": score, "alpha": alpha,
                                     "scaling": which,
                                     "eff_rank": float(pr), "d95": d95}

    # CONSISTENCY GUARD. V is a LINEAR function of the last tap (that tap feeds
    # the final Linear directly), so a linear readout of it CANNOT legitimately
    # score below affine-rescaled V. If it does, the readout is underfitting --
    # not the representation failing -- and reporting the difference as
    # "information destroyed by the head" is exactly the wrong conclusion.
    # This fired for real: ridge on h3 read 0.006 against affine-V 0.029 at
    # effRank 1.8, and printed a confident "<- destruction" label beside it.
    out["probe_artifact"] = None
    if out["head_layers"] and affine_ref is not None and np.isfinite(affine_ref):
        lk = sorted(out["head_layers"])[-1]
        last = out["head_layers"][lk]["ridge_ev"]
        if np.isfinite(last) and last < affine_ref - 0.005:
            out["probe_artifact"] = {
                "layer": lk, "measured": last, "affine_V_floor": affine_ref,
                "note": "linear readout scored below affine-rescaled V, which is "
                        "impossible; readout is underfitting a low-rank "
                        "representation. Using affine_V_floor instead.",
            }
            out["head_layers"][lk]["ridge_ev_raw"] = last
            out["head_layers"][lk]["ridge_ev"] = affine_ref

    out["lost_in_trunk"] = out["s1_cnn_ridge"] - out["s2_trunk_ridge"]
    out["lost_in_head"] = out["s2_trunk_ridge"] - out["s3_trained_V"]
    if out["head_layers"]:
        last = out["head_layers"][sorted(out["head_layers"])[-1]]["ridge_ev"]
        # Split the old lumped "lost_in_head" into its two distinguishable parts.
        out["lost_in_head_layers"] = out["s2_trunk_ridge"] - last
        out["lost_in_final_projection"] = last - out["s3_trained_V"]
    return out


# --------------------------------------------------------------------------
# tier 3 -- behavioral
# --------------------------------------------------------------------------
def tier3_behavioral(data, num_timesteps, screening, args):
    """Does the critic improve DECISIONS?

    Prefers existing LBR sidecars -- a full sweep already produces exactly these
    numbers, and re-running costs ~3 h per checkpoint. Only shells out when no
    sidecar exists for this checkpoint.
    """
    pat = os.path.join(REPO_ROOT, "br_rewards", "**",
                       f"*_{num_timesteps}_main_*_lbr_br0_*.json")
    found = sorted(glob.glob(pat, recursive=True))
    out = {"source": "sidecar" if found else None, "seats": {}}

    if not found and screening:
        cmd = [sys.executable, "-u",
               os.path.join(REPO_ROOT, "local_best_response.py"),
               "--main_checkpoint_model_path", args.ckpt,
               "--eval_prot", "both",
               "--lbr_episodes", str(args.tier3_episodes),
               "--lbr_n_envs", str(args.n_envs),
               "--lbr_controls", "True",
               "--device", args.device]
        print(f"   [tier3] no sidecar found; running LBR screen "
              f"({args.tier3_episodes} episodes)", flush=True)
        subprocess.run(cmd, cwd=os.path.join(REPO_ROOT, "main"), check=False)
        found = sorted(glob.glob(pat, recursive=True))
        out["source"] = "screen" if found else None

    for f in found:
        d = json.load(open(f))
        sp = d["selfplay_return_mean"]
        seat = "eps_adv" if "_main_left_" in os.path.basename(f) else "eps_ego"
        out["seats"][seat] = {
            "selfplay": sp,
            "eps_lbr": d["lbr_return_mean"] - sp,
            "eps_greedy": d["greedy_return_mean"] - sp,
            "eps_shuffle": d["shuffle_return_mean"] - sp,
            "episodes": d.get("episodes"),
        }
    if len(out["seats"]) == 2:
        out["nashconv_lbr"] = sum(v["eps_lbr"] for v in out["seats"].values())
        out["nashconv_greedy"] = sum(v["eps_greedy"] for v in out["seats"].values())
    return out


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------
def verdicts(res):
    """Turn floats into statements. Each entry is (name, status, detail)."""
    v = []
    t0, t1 = res.get("tier0"), res.get("tier1")
    t2, t3 = res.get("tier2"), res.get("tier3")

    if t0:
        if t0.get("has_lstm_value_head"):
            v.append(("value_head_arch", "FAIL",
                      "LSTM value head -- not loadable by this path"))
        for g, mod in t0["modules"].items():
            if mod.get("present") and mod.get("dead_row_frac", 0) > 0.10:
                v.append((f"dead_rows:{g}", "WARN",
                          f"{mod['dead_row_frac']:.1%} of rows are all-zero"))

    if t1:
        if t1["underpowered"]:
            v.append(("power", "UNDERPOWERED",
                      f"{t1['n_episodes']} episodes < {MIN_EPISODES_FOR_VERDICT}; "
                      "tier-1 EV numbers are not interpretable"))
        else:
            lo, hi = t1["ev_V_vs_G_ci90"]
            if hi < 0.10:
                v.append(("held_out_ev", "FAIL",
                          f"EV(V,G)={t1['ev_V_vs_G']:.3f} CI90 [{lo:.2f},{hi:.2f}] "
                          "-- critic does not predict returns on fresh states"))
            elif t1["ev_V_vs_G"] < 0.30:
                v.append(("held_out_ev", "WARN",
                          f"EV(V,G)={t1['ev_V_vs_G']:.3f} is weak"))
            else:
                v.append(("held_out_ev", "PASS", f"EV(V,G)={t1['ev_V_vs_G']:.3f}"))

            if t1["affine_r2_V_G"] - t1["ev_V_vs_G"] > 0.15:
                v.append(("calibration", "FAIL",
                          f"affine-R2 {t1['affine_r2_V_G']:.3f} >> EV "
                          f"{t1['ev_V_vs_G']:.3f} (slope {t1['calib_slope']:.2f}) "
                          "-- ranks states but the SCALE is wrong"))
            if t1["ev_V_vs_G"] < t1["baseline_hp_diff_linear"]:
                v.append(("vs_baseline", "FAIL",
                          f"loses to a linear readout of hp_diff "
                          f"({t1['baseline_hp_diff_linear']:.3f})"))
            if t1["std_G"] > 0 and t1["std_V"] / t1["std_G"] < 0.25:
                v.append(("dispersion", "WARN",
                          f"std(V)/std(G)={t1['std_V']/t1['std_G']:.2f} "
                          "-- V has collapsed toward the mean"))
        # STALLING. Read this BEFORE any EV number on the same checkpoint.
        #
        # When both policies stop forcing a decision, episodes run to the round
        # timer. The return then becomes a near-deterministic function of where
        # you are in a repeating animation loop, which the pixels identify
        # exactly -- so EV rockets while the critic learns nothing. Measured on
        # the popart run at 34.56M: 515 steps/episode, EV 0.957, and a RANDOM
        # PROJECTION scored 0.949, i.e. the trained encoder added 0.008.
        # hp_diff simultaneously fell to -0.000 because nobody was landing hits.
        #
        # This also explains why greedy LBR went vacuous on the neu run's late
        # checkpoints (dec/ep ~530): a greedy exploiter needs an immediate reward
        # to act on, and against a turtling opponent there is none.
        spe = t1.get("steps_per_episode")
        if spe is not None and spe > STALL_STEPS_PER_EPISODE:
            corroborated = bool(
                t2 and t1.get("ev_V_vs_G", 0) > 0.50
                and t2.get("s1_cnn_ridge") is not None
                and t2.get("ctrl_random_proj_ridge") is not None
                and t2["s1_cnn_ridge"] - t2["ctrl_random_proj_ridge"] < 0.05)
            v.append(("stalling", "FAIL" if corroborated else "WARN",
                      f"{spe:.0f} agent steps/episode (> {STALL_STEPS_PER_EPISODE}) "
                      f"-- episodes are running to the round timer, not ending in "
                      f"knockouts. EV on this checkpoint is NOT interpretable as "
                      f"critic quality" +
                      ("; confirmed: high EV with a random projection within 0.05 "
                       "of the trained encoder" if corroborated else "")))
        if t1.get("bootstrap_share_of_target_var", 1.0) < 0.30:
            v.append(("target_tautology", "NOTE",
                      f"bootstrap is only "
                      f"{t1['bootstrap_share_of_target_var']:.0%} of target "
                      "variance; EV(target,G) is near-tautological, do not "
                      "read it as target quality"))

    # Tier 2 shares tier 1's rollout, so it inherits the same power problem. A
    # ridge fit on a few hundred effective samples produces numbers that look
    # decisive and are not; suppress the verdicts rather than publish them.
    if t2 and t1 and t1["underpowered"]:
        v.append(("tier2_power", "UNDERPOWERED",
                  "representation verdicts suppressed -- same rollout as tier 1"))
    elif t2:
        if t2["s2_trunk_ridge"] < t2["ctrl_random_proj_ridge"] - 0.05:
            v.append(("trunk_vs_random", "FAIL",
                      f"trained trunk ({t2['s2_trunk_ridge']:.3f}) is WORSE than a "
                      f"random projection ({t2['ctrl_random_proj_ridge']:.3f})"))
        if t2.get("probe_artifact"):
            pa = t2["probe_artifact"]
            v.append(("probe_artifact", "WARN",
                      f"linear readout of [{pa['layer']}] measured "
                      f"{pa['measured']:+.3f}, below affine-rescaled V "
                      f"({pa['affine_V_floor']:+.3f}) which is impossible -- "
                      f"readout underfit a low-rank layer; floored, do NOT read "
                      f"the difference as information loss"))
        if t2["lost_in_head"] > t2["lost_in_trunk"] + 0.05:
            v.append(("stage_culprit", "NOTE", "signal is lost in value_net head [3]"))
        elif t2["lost_in_trunk"] > t2["lost_in_head"] + 0.05:
            v.append(("stage_culprit", "NOTE", "signal is lost in mlp_extractor trunk [2]"))

    if t3 and t3.get("seats"):
        for seat, s in t3["seats"].items():
            if s["eps_lbr"] <= 0:
                v.append((f"lbr_bound:{seat}", "FAIL",
                          f"eps_lbr={s['eps_lbr']:+.4f} <= 0 -- vacuous bound"))
            if abs(s["eps_lbr"] - s["eps_shuffle"]) < 0.02:
                v.append((f"critic_vs_shuffle:{seat}", "FAIL",
                          f"lbr {s['eps_lbr']:+.4f} ~= shuffled critic "
                          f"{s['eps_shuffle']:+.4f} -- critic carries no "
                          "branch-discriminating signal"))
            if s["eps_greedy"] > s["eps_lbr"] + 0.02:
                v.append((f"critic_harmful:{seat}", "FAIL",
                          f"greedy {s['eps_greedy']:+.4f} beats critic-guided "
                          f"{s['eps_lbr']:+.4f} -- the critic makes decisions WORSE"))
    return v


def print_report(res):
    t0, t1 = res.get("tier0"), res.get("tier1")
    t2, t3 = res.get("tier2"), res.get("tier3")
    print("\n" + "=" * 78)
    print(f"CRITIC DIAGNOSTICS  {res['checkpoint']}  ({res.get('num_timesteps')} steps)")
    print("=" * 78)

    if t0:
        print("\n[tier 0] static")
        print(f"  {'module':30s} {'params':>12s} {'|W|':>10s} {'dead rows':>10s}")
        for g, m in t0["modules"].items():
            if not m.get("present"):
                print(f"  {g:30s} {'-- absent --':>12s}")
                continue
            print(f"  {g:30s} {m['n_params']:>12,d} {m['weight_norm_total']:>10.2f} "
                  f"{m['dead_row_frac']:>9.1%}")

    if t1:
        _spe = t1.get("steps_per_episode")
        _ep_txt = ""
        if _spe is not None and np.isfinite(_spe):
            _ep_txt = f"  steps/ep={_spe:.0f}"
            if _spe > STALL_STEPS_PER_EPISODE:
                _ep_txt += " STALLING"
        print(f"\n[tier 1] prediction   episodes={t1['n_episodes']}  "
              f"samples={t1['n_samples_G']}{_ep_txt}")
        lo, hi = t1["ev_V_vs_G_ci90"]
        print(f"  {'EV(V, realized G)':32s} {t1['ev_V_vs_G']:>8.3f}   "
              f"CI90 [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {'EV(V, v_target)':32s} {t1['ev_V_vs_target']:>8.3f}")
        print(f"  {'EV(v_target, G)':32s} {t1['ev_target_vs_G']:>8.3f}   "
              f"(bootstrap share {t1['bootstrap_share_of_target_var']:.2f})")
        if res.get("in_batch"):
            ib = res["in_batch"]["in_batch_ego_ev"]
            if ib is not None:
                print(f"  {'in-batch EV (from log)':32s} {ib:>8.3f}   "
                      f"GAP = {ib - t1['ev_V_vs_G']:+.3f}")
        print(f"  {'corr(V, G)':32s} {t1['corr_V_G']:>8.3f}")
        print(f"  {'affine-rescaled R2':32s} {t1['affine_r2_V_G']:>8.3f}   "
              f"(slope {t1['calib_slope']:+.3f}, intercept {t1['calib_intercept']:+.4f})")
        print(f"  {'std V / std G':32s} "
              f"{t1['std_V']/max(t1['std_G'],1e-12):>8.3f}")
        print(f"  {'baseline hp_diff linear':32s} {t1['baseline_hp_diff_linear']:>8.3f}")
        print(f"  {'baseline state linear':32s} {t1['baseline_state_linear']:>8.3f}")

    if t2:
        al = t2.get("alphas", {})
        print(f"\n[tier 2] representation   n={t2['n']}  "
              f"ridge alpha selected per stage on a validation split")
        for lab, key in (("[1] vf_features CNN -> ridge", "s1_cnn_ridge"),
                         ("[2] + mlp trunk     -> ridge", "s2_trunk_ridge")):
            print(f"  {lab:32s} {t2[key]:>8.3f}   (alpha {al.get(key, float('nan')):.0e})")
        print(f"  {'[3] + value head    -> V':32s} {t2['s3_trained_V']:>8.3f}")
        for lab, key in (("CONTROL random proj -> ridge", "ctrl_random_proj_ridge"),
                         ("REFERENCE ego actor CNN", "ref_ego_actor_cnn_ridge"),
                         ("REFERENCE adv actor CNN", "ref_adv_actor_cnn_ridge")):
            print(f"  {lab:32s} {t2[key]:>8.3f}   (alpha {al.get(key, float('nan')):.0e})")
        if "s1_cnn_mlp" in t2:
            print(f"  {'[1] CNN   -> MLP (nonlinear)':32s} {t2['s1_cnn_mlp']:>8.3f}")
            print(f"  {'[2] trunk -> MLP (nonlinear)':32s} {t2['s2_trunk_mlp']:>8.3f}")
        if t2.get("head_layers"):
            print(f"\n  {'inside value head (same target/split)':32s} "
                  f"{'ridge EV':>9s} {'effRank':>9s}")
            for nm in sorted(t2["head_layers"]):
                h = t2["head_layers"][nm]
                flag = "  [floor: see probe_artifact]" if "ridge_ev_raw" in h else ""
                print(f"  {('  [' + nm + '] post-activation'):32s} "
                      f"{h['ridge_ev']:>9.3f} {h['eff_rank']:>9.1f}   "
                      f"(alpha {h['alpha']:.0e}, {h.get('scaling','std')}){flag}")
            print(f"  {'  -> V (final 512->1)':32s} {t2['s3_trained_V']:>9.3f}")
            print(f"\n  lost in trunk        {t2['lost_in_trunk']:+.3f}")
            print(f"  lost in head LAYERS  {t2['lost_in_head_layers']:+.3f}"
                  f"   <- destruction")
            print(f"  lost in FINAL proj   {t2['lost_in_final_projection']:+.3f}"
                  f"   <- scale / direction")
        else:
            print(f"  lost in trunk {t2['lost_in_trunk']:+.3f}   "
                  f"lost in head {t2['lost_in_head']:+.3f}")
        print(f"\n  {'representation':18s} {'dim':>5s} {'dead%':>7s} {'effRank':>8s} "
              f"{'d95':>5s}")
        for nm in ("vf_features", "latent_vf", "random_proj"):
            r = t2[f"rank_{nm}"]
            print(f"  {nm:18s} {r['dim']:>5d} {r['dead_frac']*100:>7.1f} "
                  f"{r['eff_rank']:>8.1f} {r['d95']:>5d}")

    if t3 and t3.get("seats"):
        print(f"\n[tier 3] behavioral   source={t3.get('source')}")
        print(f"  {'seat':10s} {'selfplay':>10s} {'eps_lbr':>10s} {'eps_greedy':>11s} "
              f"{'eps_shuffle':>12s}")
        for seat, s in sorted(t3["seats"].items()):
            print(f"  {seat:10s} {s['selfplay']:>10.4f} {s['eps_lbr']:>+10.4f} "
                  f"{s['eps_greedy']:>+11.4f} {s['eps_shuffle']:>+12.4f}")
        if "nashconv_greedy" in t3:
            print(f"  NashConv lower bound: lbr {t3['nashconv_lbr']:+.4f}   "
                  f"greedy {t3['nashconv_greedy']:+.4f}")

    print("\n" + "-" * 78)
    print("VERDICTS")
    print("-" * 78)
    vs = res.get("verdicts", [])
    if not vs:
        print("  (none triggered)")
    for name, status, detail in vs:
        print(f"  [{status:13s}] {name:26s} {detail}")
    print()


# --------------------------------------------------------------------------
# series mode
# --------------------------------------------------------------------------
def plot_series(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [r for r in results if r.get("tier1")]
    if not results:
        print("  [series] nothing with tier-1 data to plot")
        return None
    results.sort(key=lambda r: r.get("num_timesteps") or 0)
    xs = [r["num_timesteps"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax = axes[0]
    ev = [r["tier1"]["ev_V_vs_G"] for r in results]
    lo = [r["tier1"]["ev_V_vs_G_ci90"][0] for r in results]
    hi = [r["tier1"]["ev_V_vs_G_ci90"][1] for r in results]
    ax.plot(xs, ev, marker="o", color="#1f77b4", label="held-out EV(V, G)")
    ax.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.20,
                    label="CI90 (episode bootstrap)")
    ib = [(r.get("in_batch") or {}).get("in_batch_ego_ev") for r in results]
    if any(x is not None for x in ib):
        ax.plot([x for x, y in zip(xs, ib) if y is not None],
                [y for y in ib if y is not None],
                marker="s", linestyle="--", color="#d62728", label="in-batch EV (log)")
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.set_ylabel("explained variance")
    ax.set_title("Critic prediction quality across training")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    got_t3 = [r for r in results if (r.get("tier3") or {}).get("seats")]
    if got_t3:
        gx = [r["num_timesteps"] for r in got_t3]
        for key, color, lab in (("eps_lbr", "#1f77b4", "eps (critic LBR)"),
                                ("eps_greedy", "#2ca02c", "eps (greedy, no critic)"),
                                ("eps_shuffle", "#ff7f0e", "eps (shuffled critic)")):
            for seat, ls in (("eps_adv", "-"), ("eps_ego", "--")):
                ys = [r["tier3"]["seats"].get(seat, {}).get(key) for r in got_t3]
                if all(y is None for y in ys):
                    continue
                ax.plot(gx, ys, marker="o", markersize=4, linestyle=ls,
                        color=color, alpha=0.9, label=f"{lab} [{seat}]")
        ax.axhline(0.0, color="#888888", linewidth=0.8)
        ax.set_ylabel("eps (exploitability lower bound)")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "no tier-3 data", ha="center", transform=ax.transAxes)
    ax.set_xlabel("training timestep")
    ax.grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
def run_one(ckpt, args):
    data = load_from_zip_file(ckpt, device="cpu")[0]
    num_timesteps = data.get("num_timesteps")
    res = {"checkpoint": os.path.basename(ckpt), "num_timesteps": num_timesteps,
           "args": {"steps": args.steps, "n_envs": args.n_envs, "seed": args.seed,
                    "ridge": args.ridge, "tiers": args.tiers}}
    tiers = set(int(t) for t in args.tiers.split(",") if t.strip())

    if 0 in tiers:
        res["tier0"] = tier0_static(ckpt, data)
        if res["tier0"]["has_lstm_value_head"]:
            print("  [tier0] LSTM value head -- skipping rollout tiers")
            tiers -= {1, 2}

    if tiers & {1, 2}:
        T = args.T if args.T else int(data.get("vtrace_seq_len") or 64)
        head_idx, label, state = resolve_matchups(data, args.lbr_matchups)[0]
        res["matchup"] = label
        venv = build_lbr_venv(state, args.n_envs)
        try:
            model, _ = load_checkpoint(ckpt, venv, args.device)
            preflight(venv, model)
            ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=_b(args.eval_prot))
            t0 = time.time()
            head_mod = model.policy.value_net[f"{label}_{head_idx}"]
            raw = collect_all(venv, ops, args.steps, args.seed, head=head_mod)
            res["collect_s"] = round(time.time() - t0, 1)
            gamma = ops.gamma
        finally:
            venv.close()
        der = derive_targets(raw, gamma, T)
        res["gamma"], res["T"] = gamma, T
        if 1 in tiers:
            res["tier1"] = tier1_prediction(raw, der, args.ridge, args.seed)
            ib = read_inbatch_ev_from_log(args.log, num_timesteps)
            if ib:
                res["in_batch"] = ib
        if 2 in tiers:
            res["tier2"] = tier2_representation(
                raw, der, args.ridge, args.seed, args.device,
                run_mlp=not args.no_mlp,
                affine_ref=(res.get("tier1") or {}).get("affine_r2_V_G"))

    if 3 in tiers:
        res["tier3"] = tier3_behavioral(data, num_timesteps,
                                        screening=args.tier3_run, args=args)

    res["verdicts"] = verdicts(res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Critic diagnostic suite")
    ap.add_argument("--ckpt", type=str, default=None, help="single checkpoint")
    ap.add_argument("--series", type=str, default=None,
                    help="glob over checkpoints, e.g. 'trained_models/.../spar_Ry_Sa_*_steps.task'")
    ap.add_argument("--ckpts", type=str, default=None,
                    help="comma-separated explicit checkpoint list; use when the "
                         "wanted subset is not expressible as one glob")
    ap.add_argument("--tiers", type=str, default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=5000,
                    help="vec-steps to collect. Needs to yield >=100 EPISODES; "
                         "800 gives ~15-22 and is pure noise.")
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    ap.add_argument("--T", type=int, default=None, help="v-trace seq len override")
    ap.add_argument("--eval_prot", type=str, default="True")
    ap.add_argument("--lbr_matchups", type=str, default="all")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log", type=str, default=None,
                    help="training log, to recover the IN-BATCH EV for the gap")
    ap.add_argument("--no_mlp", action="store_true",
                    help="skip the nonlinear tier-2 probes (much faster)")
    ap.add_argument("--tier3_run", action="store_true",
                    help="run an LBR screen when no sidecar exists (expensive)")
    ap.add_argument("--tier3_episodes", type=int, default=20)
    ap.add_argument("--out", type=str, default="critic_diagnostics")
    ap.add_argument("--blas_threads", type=int, default=4,
                    help="cap BLAS threads so a concurrent LBR sweep is not "
                         "starved; read before numpy is imported")
    args = ap.parse_args(argv)

    if not (args.ckpt or args.series or args.ckpts):
        ap.error("need --ckpt, --ckpts or --series")

    def _step_of(p):
        m = re.search(r"_(\d+)_steps", p)
        return int(m.group(1)) if m else 0

    if args.ckpt:
        ckpts = [args.ckpt]
    elif args.ckpts:
        ckpts = sorted((c.strip() for c in args.ckpts.split(",") if c.strip()),
                       key=_step_of)
    else:
        ckpts = sorted(glob.glob(args.series), key=_step_of)
    if not ckpts:
        print(f"no checkpoints matched: {args.series}")
        return 1

    # REPO_ROOT is local_best_response.py's own directory, i.e. <repo>/main --
    # not the repository root, despite the name.
    out_dir = os.path.join(REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, c in enumerate(ckpts):
        print(f"\n########## [{i+1}/{len(ckpts)}] {os.path.basename(c)} ##########",
              flush=True)
        try:
            r = run_one(c, args)
        except Exception as e:                       # keep a series going
            print(f"  [error] {type(e).__name__}: {e}")
            continue
        results.append(r)
        print_report(r)
        p = os.path.join(out_dir, f"{os.path.basename(c).replace('.task','')}.json")
        with open(p, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  wrote {p}")

    if len(results) > 1:
        pp = plot_series(results, os.path.join(out_dir, "series.png"))
        if pp:
            print(f"\nwrote {pp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
