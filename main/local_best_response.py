"""Local Best Response (LBR) evaluation of a checkpoint.

This is Lisy & Bowling (2017) Local Best Response. It is NOT `local_br_eval.py`,
where "local" means "in-process eval of an already-trained BR exploiter".

At each of its decision points the LBR player enumerates every action, takes the
one that looks best under a one-step lookahead, then reverts to the policy. The
critic only *chooses*; the number we report is the realized match outcome from the
emulator. A miscalibrated critic therefore makes the bound looser, never invalid:
LBR is a strategy, and the value of any strategy against pi lower-bounds the
best-response value against pi.

Cost is ~100x below training a real best response (10M-150M env steps), so this
can run on every checkpoint instead of a handful.

Two things this file is deliberately careful about:

  * The branch requires committing an ego action. Using the *sampled* one would be
    clairvoyance in a simultaneous-move game and would not be a lower bound, so we
    marginalize over ego's top-k actions weighted by pi_ego (--lbr_ego_topk).
  * Values come from `evaluate_states(buf_num=[i], env_indices=...)`, never
    `value_forward`. The latter slices the batch by num_adversaries assuming
    canonical env order; on a branch batch it silently applies the wrong matchup
    head, and under mirror (num_adversaries > batch) it returns all zeros with no
    exception, which would reduce LBR to greedy damage via a silent bug.

Controls shipped alongside every run (--lbr_controls): a greedy-damage player
(gamma=0, no critic) and a critic-shuffled player. If LBR does not beat them, the
lookahead machinery is contributing nothing and that is the finding.
"""
import os
import sys


def _peek_torch_device_argv(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


# Must run before `import torch`, matching local_br_eval.py:5-14.
_lbr_dev = _peek_torch_device_argv(sys.argv[1:])
if _lbr_dev is not None and str(_lbr_dev).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
import json
import time
import argparse
import contextlib
import collections
import numpy as np
import torch as th

import retro
from stable_baselines3.common.save_util import load_from_zip_file

from common.minimax import solve_matrix_game
from common.const import sf_game
from common.retro_wrappers import SFWrapper, RamObsWrapper, InfoObsWrapper
from common.utils import SubprocVecEnv2P, VecTransposeImage2P
from utils import select_matchup_env
from local_br_eval import _resolve_cds_family_class, _extract_left_right_names_from_state
from new_br_worker import _sanitize_for_filename, _derive_spar_run_subdir


# The actor guard in local_br_eval.py:40 covers only the policy. LBR's argmax is
# driven by the critic, so a silently random-initialized value head would degrade
# LBR to greedy damage without any error.
_CRITIC_KEY_PREFIXES = ("value_net", "vf_features_extractor", "mlp_extractor.value_net")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _obs_is_image(data):
    """3-D observation space => image => a real (parameterised) features extractor."""
    return len(tuple(getattr(data.get("observation_space"), "shape", ()) or ())) == 3


# Phase-level wall-clock accounting. Cheap enough to leave always on (one
# perf_counter pair per phase, against phases that cost milliseconds).
PROF_T = collections.defaultdict(float)
PROF_N = collections.defaultdict(int)


@contextlib.contextmanager
def _prof(key):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        PROF_T[key] += time.perf_counter() - t0
        PROF_N[key] += 1


def prof_reset():
    PROF_T.clear()
    PROF_N.clear()


def prof_report(total_s, label=""):
    rows = sorted(PROF_T.items(), key=lambda kv: -kv[1])
    acct = sum(PROF_T[k] for k in PROF_T if not k.startswith("_"))
    print(f"   --- profile {label} ---")
    print(f"   {'phase':26s} {'total_s':>9s} {'calls':>8s} {'ms/call':>9s} {'% wall':>7s}")
    for k, v in rows:
        n = PROF_N[k]
        print(f"   {k:26s} {v:9.2f} {n:8d} {1000*v/max(n,1):9.3f} "
              f"{100*v/max(total_s,1e-9):7.1f}")
    print(f"   {'ACCOUNTED':26s} {acct:9.2f} {'':8s} {'':9s} "
          f"{100*acct/max(total_s,1e-9):7.1f}")
    print(f"   {'WALL':26s} {total_s:9.2f}")
    return {k: {"total_s": round(v, 3), "calls": PROF_N[k],
                "ms_per_call": round(1000 * v / max(PROF_N[k], 1), 4)}
            for k, v in rows}


# --------------------------------------------------------------------------- env


def make_lbr_env(state, side="both", reset_type="round", enable_combo=True,
                 null_combo=False, transform_action=True, seed=0,
                 obs_type="image", ram_mask=None):
    """Env factory for LBR. Deliberately omits Monitor2P.

    Monitor2P is stateful and fails loudly under branching: it raises
    "Tried to step environment that needs reset" the moment a branch probe hits
    done, and appends a phantom episode record for each one. Nothing in the LBR
    path needs it -- the driver accumulates returns itself.
    """
    def _init():
        env = retro.make(
            game=sf_game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=2,
        )
        env = SFWrapper(env, side=side, rendering=False, reset_type=reset_type,
                        init_level=1, state_dir=None, verbose=False,
                        enable_combo=enable_combo, null_combo=null_combo,
                        transform_action=transform_action)
        # Must match the CHECKPOINT's observation space. Building an image env
        # for a --obs_type ram policy fails deep inside the first forward with an
        # unreadable 65,536-element space dump; infer_obs_kwargs() reads the
        # right setting off the checkpoint so callers cannot get this wrong.
        if obs_type == "ram":
            env = RamObsWrapper(env, mask=ram_mask)
        elif obs_type == "info":
            env = InfoObsWrapper(env)
        env.seed(seed)
        return env
    return _init


def build_lbr_venv(state, n_envs, **kw):
    fns = [make_lbr_env(state, seed=i, **kw) for i in range(n_envs)]
    venv = SubprocVecEnv2P(fns)
    # VecTransposeImage2P is for images only; a 1-D Box must not go through it.
    if kw.get("obs_type", "image") == "image":
        return VecTransposeImage2P(venv)
    return venv


def infer_obs_kwargs(data, ram_mask=None):
    """obs_type/ram_mask implied by a checkpoint's observation_space.

    A checkpoint records the space it was trained with, so the eval env can be
    built to match without the caller having to remember. A 1-D Box means
    --obs_type ram (or info); anything 3-D is an image.

    `ram_mask` is required when the space is 1-D and SMALLER than full RAM: the
    checkpoint stores the WIDTH but not WHICH bytes, so the mask file has to be
    supplied and is validated against the width here rather than failing later.
    """
    space = data.get("observation_space")
    shape = tuple(getattr(space, "shape", ()) or ())
    if len(shape) != 1:
        return {"obs_type": "image", "ram_mask": None}
    n = int(shape[0])
    if n == 65536:
        return {"obs_type": "ram", "ram_mask": None}
    if n <= 64:
        return {"obs_type": "info", "ram_mask": None}
    if ram_mask is None:
        raise SystemExit(
            f"checkpoint observation is {n} wide, i.e. a MASKED ram obs, but no "
            f"--ram_mask was given. The checkpoint records the width, not which "
            f"bytes; pass the .npy used for training.")
    m = np.load(ram_mask) if isinstance(ram_mask, str) else np.asarray(ram_mask)
    if m.size != n:
        raise SystemExit(f"--ram_mask has {m.size} indices but the checkpoint "
                         f"observation is {n} wide; wrong mask file.")
    return {"obs_type": "ram", "ram_mask": m}


def preflight(venv, model):
    """Assert the env and model are in a configuration LBR can actually evaluate."""
    cfg = venv.env_method("lbr_config")[0]
    if not cfg["transform_action"]:
        raise SystemExit(
            "LBR requires --transform_action True. Got action_space="
            f"{cfg['action_space']}, which for MultiBinary(15) means 32768 joint "
            "actions -- enumeration is intractable. Note br_worker.py defaults it off."
        )
    if cfg["n_actions"] is None:
        raise SystemExit(f"LBR needs a MultiDiscrete action space; got {cfg['action_space']}")
    if cfg["reset_type"] != "round":
        raise SystemExit(
            f"LBR requires --reset round (got {cfg['reset_type']}). Under other reset "
            "types a branch can advance `level` into SF_BONUS_LEVEL, whose skip loop at "
            "retro_wrappers.py:187 is an unbounded `while` with no iteration cap and "
            "would hang the worker."
        )
    if getattr(model, "vtrace_trainer", None) is not None:
        raise SystemExit(
            "Loaded model still has a live V-trace worker thread. The policy's "
            "distribution objects are shared mutable state; concurrent forwards would "
            "race. Expected None after .load()."
        )
    return cfg


# ------------------------------------------------------------------------- model


def load_checkpoint(path, venv, device):
    cls, detection = _resolve_cds_family_class(path)
    model = cls.load(path, env=venv, num_perturbed=1, device=device)
    model.policy.set_training_mode(False)

    # Critic smoke test: a dead or random-init value head makes LBR silently
    # degenerate into greedy damage, which is exactly the failure mode the
    # controls exist to detect -- so catch it here instead.
    _data, _params, _ = load_from_zip_file(path, device="cpu")
    saved = _params["policy"]
    # FlattenExtractor is PARAMETERLESS, so a legitimate --obs_type ram
    # checkpoint has no vf_features_extractor.* keys at all
    # (clean_derivative_free_spar.py:581 selects it for non-image spaces). Absence
    # there is not evidence of an untrained critic; only the modules that always
    # carry weights are worth asserting on.
    _required = [k for k in _CRITIC_KEY_PREFIXES
                 if k != "vf_features_extractor" or _obs_is_image(_data)]
    missing_critic = [k for k in _required
                      if not any(name.startswith(k) for name in saved)]
    if missing_critic:
        raise SystemExit(
            f"Checkpoint has no weights for {missing_critic}. LBR's argmax is driven "
            "by the critic, so this would run an untrained value head."
        )
    return model, detection


class PolicyOps:
    """Thin, side-aware wrapper over the SPAR policy for LBR.

    ego_side is the *checkpoint's* ego (left in the live configs). `lbr_is_adv`
    says whether the LBR player is driving the adversary head or the ego head.
    """

    def __init__(self, model, head_idx=0, lbr_is_adv=True, gamma_override=None,
                 minimax_iters=1024):
        self.model = model
        self.p = model.policy
        self.head = head_idx
        self.lbr_is_adv = lbr_is_adv
        self.device = model.device
        self.minimax_iters = int(minimax_iters)
        # gamma weights the critic against the immediate reward in Q = r + gamma*V.
        # Overriding it sweeps between greedy (gamma=0, critic ignored) and the
        # trained value (gamma=model.gamma), which is the direct test of whether
        # the critic is bad or merely weighted far too heavily.
        self.gamma = float(model.gamma) if gamma_override is None else float(gamma_override)
        self.n_actions = int(model.action_space.nvec[0])
        # Multi-matchup checkpoints can carry ego_side=None (not just a missing
        # attribute), so a plain getattr default is not enough -- it would leave
        # ego_side None, make the `== "left"` test False, and seat LBR on the wrong
        # side. The two_player states are all <Ego>_left/..., so left is the default.
        self.ego_side = getattr(model, "ego_side", None) or "left"
        # V_adv = -V_ego: the value head is ego-perspective only, negated for the
        # adversary at six sites in the training code.
        self.sgn = -1.0 if lbr_is_adv else 1.0

    def _t(self, obs):
        return th.as_tensor(obs).to(self.device)

    @th.no_grad()
    def minimax_values(self, obs):
        """V_ego from the MINIMAX value of the joint-action matrix Q(s,.,.).

        The gate's leaf evaluator. Where values_ego returns V^pi(s) -- which is
        constant across LBR branches until the successor states diverge, and
        measured indistinguishable from a permuted copy (lbr vs shuffle:
        -0.1419 vs -0.1407 over 42 runs) -- this returns the value of the
        22x22 matrix game at s, which is a function of the whole action space.

        Raises if the checkpoint has no matrix head, rather than silently
        falling back to V: a mode called `minimax` that quietly evaluated V
        would produce a number that looks like a result and is not one.
        """
        if not getattr(self.p, "minimax_q", False) or not len(getattr(self.p, "minimax_net", {})):
            raise RuntimeError(
                "minimax mode requires a checkpoint trained with --minimax_q True; "
                "this policy has no minimax_net")
        t = self._t(obs)
        M = self.p.minimax_matrices(t, buf_num=[self.head], stop_grad=True)
        V = solve_matrix_game(M, iters=self.minimax_iters).V
        # evaluate_states returns EGO-perspective values and callers apply
        # ops.sgn afterwards; Q is ego-payoff too, so the same convention holds.
        return V.reshape(-1).cpu().numpy()

    @th.no_grad()
    def values_ego(self, obs):
        """V_ego over a batch of arbitrary size.

        evaluate_states divides env_indices by envs_per_matchup unconditionally
        (the =None default is a lie), and with len(buf_num)==1 it then forces the
        whole batch through that one head. Feed it indices that map to our head.
        """
        t = self._t(obs)
        n = t.shape[0]
        env_idx = th.full((n,), self.head * self.p.envs_per_matchup,
                          dtype=th.long, device=self.device)
        v = self.p.evaluate_states(t, buf_num=[self.head], env_indices=env_idx)
        return v.reshape(-1).cpu().numpy()

    @th.no_grad()
    def ego_probs(self, obs):
        """pi_ego(.|s) as a full (N, n_actions) array. No public accessor exists,
        so mirror ego_forward's path and read the categorical off the distribution."""
        from stable_baselines3.common.preprocessing import preprocess_obs
        t = self._t(obs)
        x = preprocess_obs(t, self.p.observation_space,
                           normalize_images=self.p.normalize_images)
        f = self.p.pi_ctrl_features_extractor(x)
        latent = self.p.mlp_extractor.ego_forward(f)
        dist = self.p._get_ego_action_dist_from_latent(latent)
        return dist.distribution[0].probs.cpu().numpy()

    @th.no_grad()
    def adv_probs(self, obs):
        from stable_baselines3.common.preprocessing import preprocess_obs
        t = self._t(obs)
        x = preprocess_obs(t, self.p.observation_space,
                           normalize_images=self.p.normalize_images)
        f = self.p.pi_dstb_features_extractor(x)
        latent = self.p.mlp_extractor.adv_forward(f, side_flag=None)
        dist = self.p._get_adv_action_dist_from_latent(latent, buf_num=[self.head],
                                                       evaluate=True)[0]
        return dist.distribution[0].probs.cpu().numpy()

    @th.no_grad()
    def sample_ego(self, obs, rng):
        p = self.ego_probs(obs)
        return _sample_rows(p, rng)

    @th.no_grad()
    def sample_adv(self, obs, rng):
        p = self.adv_probs(obs)
        return _sample_rows(p, rng)

    def joint(self, lbr_actions, pol_actions):
        """Assemble the (N, 2) [left, right] action array.

        The LBR player occupies the adversary's side when lbr_is_adv. With
        ego_side='left' that means LBR is the right column.
        """
        lbr_on_left = (self.ego_side == "left") == (not self.lbr_is_adv)
        left = lbr_actions if lbr_on_left else pol_actions
        right = pol_actions if lbr_on_left else lbr_actions
        return np.stack([np.asarray(left).reshape(-1),
                         np.asarray(right).reshape(-1)], axis=-1)

    def lbr_reward(self, r_left, r_right):
        """Reward from the LBR player's seat.

        Read the correct slot rather than negating: on a draw both players get +1
        (retro_wrappers.py:281-283), so the game is not exactly zero-sum and
        -r_other would be wrong there.
        """
        lbr_on_left = (self.ego_side == "left") == (not self.lbr_is_adv)
        return np.asarray(r_left if lbr_on_left else r_right, dtype=np.float64)

    def policy_reward(self, r_left, r_right):
        lbr_on_left = (self.ego_side == "left") == (not self.lbr_is_adv)
        return np.asarray(r_right if lbr_on_left else r_left, dtype=np.float64)


def _sample_rows(probs, rng):
    """Vectorized categorical sample, one draw per row."""
    c = probs.cumsum(axis=1)
    c[:, -1] = 1.0
    u = rng.random_sample((probs.shape[0], 1))
    return (u < c).argmax(axis=1)


def splice_terminal(obs, dones, infos):
    """SubprocVecEnv2P auto-resets on done and overwrites the returned observation
    (utils.py:89-92). Recover the true post-step obs so the bootstrap sees the real
    successor. VecTransposeImage2P has already transposed terminal_observation."""
    if not np.any(dones):
        return obs
    out = np.array(obs, copy=True)
    for i, d in enumerate(dones):
        if d and "terminal_observation" in infos[i]:
            out[i] = infos[i]["terminal_observation"]
    return out


# ---------------------------------------------------------------------- schedule

ROOT = "lbr_root"


def lbr_decide(venv, ops, obs, topk, mode="lbr", rng=None, shuffle_rng=None,
               infer_chunk=512):
    """One LBR decision point, in vec-step lockstep across all envs.

    Every env runs the identical schedule on its own state, so each policy/critic
    call stays a single batched forward.

    mode:
      lbr        Q = r + gamma * V   (the real thing)
      greedy     Q = r               (gamma=0, critic never consulted)
      shuffle    Q = r + gamma * V, with V permuted across the branch axis
    """
    n = obs.shape[0]
    na = ops.n_actions

    # The LBR side enumerates all na actions; the OPPONENT is marginalized over
    # its own policy's top-k. Which head that is depends on which seat LBR holds --
    # getting this backwards makes --eval_prot False compute nothing meaningful.
    with _prof("branch/opp_probs"):
        opp_p = ops.ego_probs(obs) if ops.lbr_is_adv else ops.adv_probs(obs)
    k = min(topk, na)
    idx = np.argpartition(-opp_p, k - 1, axis=1)[:, :k]          # (N, k)
    w = np.take_along_axis(opp_p, idx, axis=1)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)   # renormalize

    need_v = (mode != "greedy")
    use_minimax = mode.startswith("minimax")
    do_shuffle = mode.endswith("shuffle")
    rewards = np.zeros((na, k, n), dtype=np.float64)
    dones = np.zeros((na, k, n), dtype=bool)
    succ = [] if need_v else None

    with _prof("branch/snapshot"):
        venv.env_method("lbr_snapshot", ROOT)
    try:
        # NOTE: the profile shows branch/restore at ~29% of wall, which looks like
        # an obvious win for folding restore+step into one env_method call. It is
        # not -- that was tried and measured. At N=16 a no-op env_method round-trip
        # is 0.117 ms, i.e. 1% of a 9.5 ms branch, so there is no round-trip cost to
        # recover; the merged call measured 1.04x (and 0.93x at N=8), and dropping
        # the observation from the payload changed nothing. The real cost is the
        # BARRIER: SubprocVecEnv2P's serial recv loop waits on the slowest of N
        # workers, and single-env lbr_restore is 107 us against 2410 us at N=16.
        # The lever that would work is fewer barriers, not smaller ones -- e.g.
        # running all na*k branches inside each worker for one barrier per decision
        # (~3x). Do not re-attempt the merge.
        for a_lbr in range(na):
            for j in range(k):
                with _prof("branch/restore"):
                    venv.env_method("lbr_restore", ROOT)
                with _prof("branch/env_step"):
                    o1, r_l, r_r, d, infos = venv.step(
                        ops.joint(np.full(n, a_lbr), idx[:, j]))
                rewards[a_lbr, j] = ops.lbr_reward(r_l, r_r)
                dones[a_lbr, j] = np.asarray(d, dtype=bool)
                if need_v:
                    # Defer the critic: one batched forward over all na*k*n
                    # successors beats na*k forwards of n rows, which is dominated
                    # by per-call overhead (measured ~4x on a 12-env run).
                    with _prof("branch/collect_obs"):
                        succ.append(splice_terminal(o1, d, infos))
        with _prof("branch/restore"):
            venv.env_method("lbr_restore", ROOT)
    finally:
        # Never rely on catching the registry-cap RuntimeError: it is raised inside
        # the worker, which kills it and surfaces as an opaque EOFError.
        with _prof("branch/drop"):
            venv.env_method("lbr_drop", ROOT)

    if need_v:
        with _prof("branch/critic_forward"):
            allobs = np.concatenate(succ, axis=0)           # (na*k*n, C, H, W)
            _vfn = ops.minimax_values if use_minimax else ops.values_ego
            v = np.concatenate([_vfn(allobs[i:i + infer_chunk])
                                for i in range(0, allobs.shape[0], infer_chunk)])
        v = (ops.sgn * v).reshape(na, k, n)
        if do_shuffle:
            # Permute the BRANCH axis (0), independently per env, the same
            # permutation across the k marginalized opponent actions.
            #
            # THIS USED TO BE `v[:, :, shuffle_rng.permutation(n)]` -- axis 2,
            # the ENV axis. The decision is `Q.argmax(axis=0)`, so permuting
            # envs left the branch ordering that LBR actually selects over
            # completely intact within each env; it only swapped whole
            # 22-branch bundles between envs. Since all envs sit in
            # near-identical states just after a reset, that is a much weaker
            # ablation than intended and is BIASED TOWARD A NULL. Every
            # lbr-vs-shuffle and minimax-vs-minimaxshuffle number produced
            # before this fix is contaminated, including the 42-run
            # -0.1419 vs -0.1407 baseline.
            #
            # Same permutation across k, not independent per k: what is being
            # destroyed is the identity of the BRANCH, and a branch's k
            # successors belong to it jointly.
            perm = np.stack([shuffle_rng.permutation(na) for _ in range(n)],
                            axis=1)                      # (na, n)
            v = np.take_along_axis(v, perm[:, None, :], axis=0)
        q = rewards + ops.gamma * v * (~dones)
    else:
        q = rewards

    # Marginalize over ego's top-k, weighted by pi_ego. w is (n, k) -> (k, n).
    Q = (q * w.T[None, :, :]).sum(axis=1)
    return Q.argmax(axis=0), Q


def run_lbr(venv, ops, episodes, topk, stride, mode, seed, max_steps=100000,
            infer_chunk=512, verbose=True):
    """Play the LBR side against the policy and return realized episode returns."""
    rng = np.random.RandomState(seed)
    shuffle_rng = np.random.RandomState(seed + 1)
    n = venv.num_envs

    obs = venv.reset()
    ep_lbr = [[] for _ in range(n)]
    ep_pol = [[] for _ in range(n)]
    done_lbr, done_pol = [], []
    cur_lbr = np.zeros(n)
    cur_pol = np.zeros(n)
    n_decisions = 0
    gaps = []
    argmax_hist = np.zeros(ops.n_actions, dtype=np.int64)
    prof_reset()
    t0 = time.time()
    step = 0

    while len(done_lbr) < episodes and step < max_steps:
        if step % stride == 0:
            a_star, Q = lbr_decide(venv, ops, obs, topk, mode=mode,
                                   rng=rng, shuffle_rng=shuffle_rng,
                                   infer_chunk=infer_chunk)
            n_decisions += 1
            argmax_hist += np.bincount(a_star, minlength=ops.n_actions)
            # Diagnostic only: this is NOT a bound. It is measured on the trajectory
            # we branched from rather than the LBR player's own occupancy, and it is
            # upward-biased by max-over-noisy-estimates.
            a_pol = ops.sample_adv(obs, rng) if ops.lbr_is_adv else ops.sample_ego(obs, rng)
            gaps.append(float(np.mean(Q.max(axis=0) - Q[a_pol, np.arange(n)])))
            lbr_a = a_star
        else:
            with _prof("play/policy_forward"):
                lbr_a = ops.sample_adv(obs, rng) if ops.lbr_is_adv else ops.sample_ego(obs, rng)

        with _prof("play/policy_forward"):
            pol_a = ops.sample_ego(obs, rng) if ops.lbr_is_adv else ops.sample_adv(obs, rng)
        with _prof("play/env_step"):
            obs, r_l, r_r, d, infos = venv.step(ops.joint(lbr_a, pol_a))
        cur_lbr += ops.lbr_reward(r_l, r_r)
        cur_pol += ops.policy_reward(r_l, r_r)
        for i, di in enumerate(d):
            if di:
                done_lbr.append(cur_lbr[i]); done_pol.append(cur_pol[i])
                cur_lbr[i] = 0.0; cur_pol[i] = 0.0
        step += 1
        if verbose and step % 25 == 0:
            print(f"   [{mode}] step {step:5d}  episodes {len(done_lbr):3d}/{episodes}  "
                  f"decisions {n_decisions:5d}  {time.time()-t0:6.1f}s", flush=True)

    return {
        "mode": mode,
        "lbr_return_mean": float(np.mean(done_lbr[:episodes])) if done_lbr else float("nan"),
        "policy_return_mean": float(np.mean(done_pol[:episodes])) if done_pol else float("nan"),
        "episodes": int(min(len(done_lbr), episodes)),
        "n_decision_points": n_decisions,
        "vec_steps": step,
        "one_shot_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
        "one_shot_gap_median": float(np.median(gaps)) if gaps else float("nan"),
        "argmax_action_histogram": argmax_hist.tolist(),
        "wall_clock_s": round(time.time() - t0, 2),
        "profile": prof_report(time.time() - t0, label=mode),
    }


def run_selfplay(venv, ops, episodes, seed, max_steps=100000):
    """Both sides play pi. The baseline the LBR number must beat."""
    rng = np.random.RandomState(seed + 99)
    n = venv.num_envs
    obs = venv.reset()
    cur = np.zeros(n)
    out = []
    step = 0
    while len(out) < episodes and step < max_steps:
        a_ego = ops.sample_ego(obs, rng)
        a_adv = ops.sample_adv(obs, rng)
        lbr_a, pol_a = (a_adv, a_ego) if ops.lbr_is_adv else (a_ego, a_adv)
        obs, r_l, r_r, d, infos = venv.step(ops.joint(lbr_a, pol_a))
        cur += ops.lbr_reward(r_l, r_r)
        for i, di in enumerate(d):
            if di:
                out.append(cur[i]); cur[i] = 0.0
        step += 1
    return float(np.mean(out[:episodes])) if out else float("nan")


# --------------------------------------------------------------------- reporting


def write_result(out_dir, subdir, fname, value):
    d = os.path.join(REPO_ROOT, out_dir, subdir) if subdir else os.path.join(REPO_ROOT, out_dir)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, fname)
    with open(p, "w") as f:
        f.write(str(value))
    return p


def resolve_matchups(data, selector="all"):
    """[(head_idx, label, state), ...] for a checkpoint.

    Envs are laid out in contiguous per-matchup blocks: env e belongs to head
    e // envs_per_matchup, and the head's ModuleDict key is f"{label}_{head_idx}"
    (select_matchup_env, utils.py:86). So head i is fully described by the block
    starting at i * envs_per_matchup.
    """
    epm = int(data["envs_per_matchup"])
    n_heads = int(data["num_adversaries"])
    states = data["state_list"]
    matchups = data.get("matchups") or []
    out = []
    for i in range(n_heads):
        j = i * epm
        label = matchups[j] if j < len(matchups) else f"matchup{i}"
        out.append((i, label, states[j]))
    if selector and str(selector).lower() != "all":
        want = {s.strip() for s in str(selector).split(",")}
        out = [m for m in out if str(m[0]) in want or m[1] in want]
        if not out:
            raise SystemExit(f"--lbr_matchups {selector!r} matched nothing. "
                             f"Available: {[(i, l) for i, l, _ in resolve_matchups(data)]}")
    return out


def build_filename(style, steps, main_side, main_char, exp_side, exp_char,
                   exp_type, br_idx, suffix=""):
    style = _sanitize_for_filename(style) if style else ""
    head = f"{style}_" if style else ""
    tail = f"{suffix}_" if suffix else ""
    return (f"{head}{steps}_main_{main_side}_{main_char}_"
            f"exploiter_{exp_side}_{exp_char}_{exp_type}_br{br_idx}_{tail}.txt")


# -------------------------------------------------------------------------- main


def build_parser():
    p = argparse.ArgumentParser(description="Local Best Response evaluation")
    p.add_argument("--main_checkpoint_model_path", type=str, required=True)
    p.add_argument("--state", type=str, default=None,
                   help="Override the state. Default: each matchup's own state from "
                        "the checkpoint's state_list.")
    p.add_argument("--lbr_matchups", type=str, default="all",
                   help="'all', or a comma-separated list of head indices or matchup "
                        "labels (e.g. '0,2' or 'GuileVsRyu,GuileVsSagat').")
    p.add_argument("--eval_prot", type=str, default="both",
                   help="True: LBR replaces the adversary (measures ego exploitability). "
                        "False: LBR replaces the ego. 'both': run both and report the "
                        "duality gap (NashConv = eps_ego + eps_adv).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_subdir", type=str, default=None)
    p.add_argument("--training_style", type=str, default="spar")
    p.add_argument("--filename_suffix", type=str, default="")
    p.add_argument("--br_index", type=int, default=0)
    p.add_argument("--lbr_ego_topk", type=int, default=4)
    p.add_argument("--lbr_stride", type=int, default=1)
    p.add_argument("--lbr_episodes", type=int, default=50)
    p.add_argument("--lbr_n_envs", type=int, default=16)
    p.add_argument("--lbr_seed", type=int, default=0)
    p.add_argument("--lbr_controls", type=str, default="True",
                   help="legacy all-or-nothing switch: True adds greedy+shuffle "
                        "to lbr. Ignored when --lbr_modes is given.")
    p.add_argument("--ram_mask", type=str, default="",
                   help="RAM byte-index .npy, required only when the checkpoint "
                        "was trained with a MASKED ram observation (the checkpoint "
                        "records the width, not which bytes).")
    p.add_argument("--lbr_modes", type=str, default="",
                   help="comma-separated subset of {lbr,greedy,shuffle}; overrides "
                        "--lbr_controls. e.g. 'greedy' runs ONLY the greedy-damage "
                        "player, which is 3x cheaper and is the only variant "
                        "producing a non-vacuous bound on the runs measured so far. "
                        "The highest-priority mode present owns the sidecar JSON "
                        "and the selfplay_rewards/ file.")
    p.add_argument("--lbr_infer_chunk", type=int, default=512,
                   help="Max rows per batched critic forward.")
    p.add_argument("--lbr_gamma", type=float, default=-1.0,
                   help="Override the discount used in Q = r + gamma*V. <0 uses the "
                        "checkpoint's own model.gamma. Set 0 to ignore the critic "
                        "entirely (equivalent to the greedy control). Sweeping this "
                        "separates 'the critic is bad' from 'the critic is weighted "
                        "far too heavily'.")
    p.add_argument("--lbr_max_steps", type=int, default=100000)
    return p


def _b(x):
    return str(x).lower() in ("1", "true", "yes", "y")


TAG = {"lbr": "lbr", "greedy": "lbrgreedy", "shuffle": "lbrshuffle",
       "minimax": "lbrminimax", "minimaxshuffle": "lbrminimaxshuffle"}

# Which mode owns the sidecar JSON and the selfplay_rewards/ file. Exactly one
# mode per (checkpoint, direction) writes them, because they describe the RUN,
# not the variant -- selfplay is computed once per direction regardless of how
# many modes are evaluated.
#
# This used to be hardcoded to "lbr". With --lbr_modes greedy that produced bare
# .txt files with no selfplay reference and no sidecar, so eps was uncomputable
# and tier 3 of critic_diagnostics had nothing to read. Now it is the
# highest-priority mode actually present, which keeps the old behaviour whenever
# lbr is in the set.
HEADLINE_PRIORITY = ("minimax", "lbr", "greedy", "shuffle", "minimaxshuffle")


def resolve_modes(args):
    """Mode list for this run, honouring --lbr_modes over legacy --lbr_controls."""
    raw = (getattr(args, "lbr_modes", "") or "").strip()
    if raw:
        modes = [m.strip() for m in raw.split(",") if m.strip()]
        bad = [m for m in modes if m not in TAG]
        if bad:
            raise SystemExit(f"--lbr_modes: unknown {bad}; valid are {sorted(TAG)}")
        if not modes:
            raise SystemExit("--lbr_modes resolved to an empty set")
        return modes
    return ["lbr"] + (["greedy", "shuffle"] if _b(args.lbr_controls) else [])


def headline_mode(modes):
    for m in HEADLINE_PRIORITY:
        if m in modes:
            return m
    return modes[0]


def run_matchup(ckpt, state, head_idx, label, directions, args):
    """Evaluate one matchup in one or both directions.

    The venv and the loaded model are built once per matchup and shared across
    directions -- only PolicyOps (seat, sign, which head enumerates) changes.
    """
    print(f"\n[LBR] ===== matchup {head_idx}: {label} =====")
    print(f"[LBR] state {state}")
    venv = build_lbr_venv(state, args.lbr_n_envs,
                          **infer_obs_kwargs(load_from_zip_file(ckpt, device="cpu")[0],
                                             getattr(args, "ram_mask", None) or None))
    out = {}
    try:
        model, detection = load_checkpoint(ckpt, venv, args.device)
        cfg = preflight(venv, model)
        modes = resolve_modes(args)
        headline = headline_mode(modes)
        for lbr_is_adv in directions:
            ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=lbr_is_adv,
                            gamma_override=(None if args.lbr_gamma < 0
                                            else args.lbr_gamma))
            seat = "adv" if lbr_is_adv else "ego"
            print(f"[LBR] -- direction: LBR plays the {seat.upper()} seat "
                  f"(ego_side={ops.ego_side}, sgn={ops.sgn:+.0f}, gamma={ops.gamma})")
            res = {}
            for mode in modes:
                res[mode] = run_lbr(venv, ops, args.lbr_episodes, args.lbr_ego_topk,
                                    args.lbr_stride, mode, args.lbr_seed,
                                    max_steps=args.lbr_max_steps,
                                    infer_chunk=args.lbr_infer_chunk)
                print(f"[LBR]    {mode:8s} return={res[mode]['lbr_return_mean']:+.5f} "
                      f"({res[mode]['episodes']} eps, {res[mode]['wall_clock_s']}s)")
            sp = run_selfplay(venv, ops, args.lbr_episodes, args.lbr_seed,
                              max_steps=args.lbr_max_steps)
            print(f"[LBR]    selfplay return={sp:+.5f}")
            out[seat] = {"modes": res, "selfplay": sp, "cfg": cfg,
                         "headline": headline,
                         "detection": detection, "ego_side": ops.ego_side,
                         "sgn": ops.sgn, "gamma": ops.gamma}
    finally:
        venv.close()
    return out


def write_matchup_results(ckpt, data, state, label, per_seat, args, subdir):
    """One .txt per (direction, variant) plus a sidecar on the ego-exploiting run."""
    steps = data.get("num_timesteps", 0)
    left_char, right_char = _extract_left_right_names_from_state(state)
    written = []
    for seat, blob in per_seat.items():
        lbr_is_adv = (seat == "adv")
        # `main_*` is the side being exploited; `exploiter_*` is the LBR seat.
        if lbr_is_adv:
            main_side, main_char, exp_side, exp_char = "left", left_char, "right", right_char
        else:
            main_side, main_char, exp_side, exp_char = "right", right_char, "left", left_char
        for mode, res in blob["modes"].items():
            fn = build_filename(args.training_style, steps, main_side, main_char,
                                exp_side, exp_char, TAG[mode], args.br_index,
                                args.filename_suffix)
            written.append(write_result("br_rewards", subdir, fn,
                                        res["lbr_return_mean"]))
            if mode == blob["headline"]:
                write_result("selfplay_rewards", subdir, fn, blob["selfplay"])
                sidecar = dict(res)
                sidecar.update({
                    "checkpoint": os.path.basename(ckpt), "state": state,
                    "matchup": label, "lbr_seat": seat,
                    "selfplay_return_mean": blob["selfplay"],
                    "greedy_return_mean": blob["modes"].get("greedy", {}).get("lbr_return_mean"),
                    "shuffle_return_mean": blob["modes"].get("shuffle", {}).get("lbr_return_mean"),
                    "gamma": blob["gamma"], "sgn": blob["sgn"],
                    "ego_side": blob["ego_side"], "ego_topk": args.lbr_ego_topk,
                    "stride": args.lbr_stride, "n_envs": args.lbr_n_envs,
                    "env_cfg": blob["cfg"],
                    "caveats": [
                        "Headline is the realized match outcome: a valid but possibly "
                        "loose lower bound on exploitability within the 22-macro action "
                        "space, for a single round.",
                        "one_shot_gap_* is a diagnostic, NOT a bound.",
                        "NashConv here sums two LBR lower bounds, so it is itself only "
                        "a lower bound on the true duality gap.",
                        "Not exactly zero-sum: a draw pays both players +1 "
                        "(retro_wrappers.py:281).",
                    ],
                })
                with open(os.path.join(REPO_ROOT, "br_rewards", subdir,
                                       fn.replace(".txt", ".json")), "w") as f:
                    json.dump(sidecar, f, indent=2)
    return written


def main(argv=None):
    args = build_parser().parse_args(argv)
    ckpt = args.main_checkpoint_model_path
    data = load_from_zip_file(ckpt, device="cpu")[0]

    ep = str(args.eval_prot).lower()
    directions = [True, False] if ep == "both" else [_b(args.eval_prot)]

    matchups = resolve_matchups(data, args.lbr_matchups)
    if args.state:
        matchups = [(h, l, args.state) for h, l, _ in matchups]

    print(f"[LBR] checkpoint {os.path.basename(ckpt)}  ({data.get('num_timesteps')} steps)")
    print(f"[LBR] {len(matchups)} matchup(s), {len(directions)} direction(s)")
    for h, l, s in matchups:
        print(f"[LBR]   head {h}: {l:22s} {s}")

    subdir = args.output_subdir or _derive_spar_run_subdir(os.path.basename(ckpt))
    summary = {}
    for head_idx, label, state in matchups:
        per_seat = run_matchup(ckpt, state, head_idx, label, directions, args)
        write_matchup_results(ckpt, data, state, label, per_seat, args, subdir)
        summary[label] = per_seat

    # ---- duality gap ------------------------------------------------------
    print()
    print("=" * 78)
    print("DUALITY GAP (NashConv) per matchup -- LOWER BOUNDS")
    print("=" * 78)
    # Report eps for every mode actually run, not a hardcoded pair.
    modes = [m for m in HEADLINE_PRIORITY if m in resolve_modes(args)]
    for label, per_seat in summary.items():
        print(f"\n  {label}")
        for mode in modes:
            parts, total, complete = [], 0.0, True
            for seat in ("ego", "adv"):
                blob = per_seat.get(seat)
                if blob is None:
                    complete = False
                    continue
                eps = blob["modes"][mode]["lbr_return_mean"] - blob["selfplay"]
                # Each epsilon is a deviation GAIN and must be >= 0 in theory; a
                # negative value means LBR failed to beat the incumbent policy in
                # that seat, i.e. the bound is vacuous there.
                parts.append(f"eps_{seat}={eps:+.5f}" + ("" if eps >= 0 else " (!)"))
                total += eps
            tag = "NashConv" if complete else "PARTIAL"
            print(f"    {mode:8s} " + "  ".join(parts) +
                  f"   ->  {tag}={total:+.5f}" +
                  ("" if complete else "   [run --eval_prot both for the full gap]"))
    print()
    print("  Note: NashConv = eps_ego + eps_adv (a SUM of both players' deviation")
    print("  gains). Each term is an LBR lower bound, so the total is a lower bound")
    print("  on the true duality gap. eps < 0 means that seat's bound is vacuous.")


if __name__ == "__main__":
    main()
