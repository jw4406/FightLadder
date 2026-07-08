import os
import sys


def _peek_torch_device_argv(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


_br_eval_dev = _peek_torch_device_argv(sys.argv[1:])
if _br_eval_dev is not None and str(_br_eval_dev).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import ast
import re

from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.justin.clean_derivative_free_spar_ippo import CleanDerivativeFreeSPARIPPO
from common.algorithms import Exploiter
from ippo import env_generator
import json
import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
import argparse
from gymnasium.spaces import Box
from new_br_worker import (
    _FixedMatchupPolicyAdapter,
    _dedupe_preserve_order,
    _sanitize_for_filename,
    load_league_model,
)


# Actor-side policy submodules that MUST be present for a valid eval. If any of
# these are missing after a strict=False load, the actor is random-initialized
# and eval results are meaningless -> we fail loud instead.
_ACTOR_KEY_PREFIXES = (
    "action_net",
    "dstb_action_net",
    "pi_ctrl_features_extractor",
    "pi_dstb_features_extractor",
)


def _resolve_cds_family_class(path):
    """Pick the algorithm class for a (non-league) SPAR-family checkpoint.

    Taxonomy mirrors duel.py's MODEL_TYPES:
      - spar               -> CleanDerivativeFreeSPAR
      - ippo / 2timescale  -> CleanDerivativeFreeSPARIPPO  (dedicated ego value head)
      - league (FSP/PSRO)  -> handled by the separate load_league_model path,
                              NOT here (their checkpoints are LeaguePPO/agent_dict
                              torch-saves, not SB3 zips). FSP vs PSRO is a
                              training-time meta-solver distinction only, so a
                              single "league" path covers both.

    ippo/2timescale are distinguished from spar by the IPPO-only top-level
    ego value-feature extractor in the *saved policy weights*
    (ego_vf_features_extractor.*), which CleanIPPOActorActorCriticPolicy always
    creates and the base CleanActorActorCriticPolicy never does. Keying on the
    state_dict is architecture ground-truth and does not depend on policy_class
    metadata being populated.

    NB: do NOT key on "ego_value_net" -- the base MlpExtractorAdv builds an
    ``ego_value_net`` submodule, so CDS checkpoints legitimately contain
    ``mlp_extractor.ego_value_net.*`` keys. ``ego_vf_features_extractor`` is the
    only unambiguous top-level, IPPO-exclusive module name.

    Redundancy check: models saved after ippo.py sets
    ``finetune_model.model_arch_type`` carry that string in the checkpoint's
    ``data``. It is cross-checked against the weight-based detection; the weights
    stay authoritative for the spar-vs-ippo class choice (they determine what
    actually loads), and a spar/ippo disagreement is flagged loudly as a
    stale/incorrect-metadata signal. A non-SPAR-family declaration (``league``,
    or any unrecognized type, which defaults to league/PSRO) raises and directs
    to the league path -- these are not SB3 SPAR-family checkpoints.

    Raises FileNotFoundError if `path` is missing (so callers' done-checkpoint
    fallback still works), and ValueError if `path` is not a readable SB3 zip
    (e.g. a league/PSRO checkpoint reached this path without --is_league).
    """
    from stable_baselines3.common.save_util import load_from_zip_file
    try:
        data, _params, _ = load_from_zip_file(path, device="cpu")
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
        raise ValueError(
            f"[local_br_eval] {path!r} is not a readable SB3 checkpoint zip. "
            f"If this is a league/PSRO checkpoint, pass --is_league True (the "
            f"league eval path handles LeaguePPO/agent_dict files). "
            f"Underlying error: {exc}"
        ) from exc
    policy_sd = (_params or {}).get("policy", {}) or {}
    has_ego = any(k.startswith("ego_vf_features_extractor") for k in policy_sd.keys())
    cls = CleanDerivativeFreeSPARIPPO if has_ego else CleanDerivativeFreeSPAR

    # Redundancy check against the training-declared arch type (absent on older
    # checkpoints). Weights are ground truth; a mismatch means bad metadata.
    arch = (data or {}).get("model_arch_type")
    note = "model_arch_type absent (older checkpoint); used weights"
    if isinstance(arch, str):
        if arch in ("spar", "ippo", "2timescale"):
            if (arch in ("ippo", "2timescale") and not has_ego) or (arch == "spar" and has_ego):
                print(
                    f"[local_br_eval] WARNING: checkpoint model_arch_type={arch!r} disagrees "
                    f"with the saved policy weights (ego value head "
                    f"{'present' if has_ego else 'absent'}). Trusting the weights and loading "
                    f"as {cls.__name__} -- checkpoint metadata is stale/incorrect; investigate.",
                    flush=True,
                )
                note = f"model_arch_type={arch!r} MISMATCH with weights -- trusted weights"
            else:
                note = f"model_arch_type={arch!r} agrees with weights"
        else:
            # Non-SPAR-family declaration (league/PSRO, or an unrecognized type we
            # default to league/PSRO). These are not SB3 SPAR-family checkpoints,
            # so route to the league path instead of mis-loading here.
            raise ValueError(
                f"[local_br_eval] checkpoint model_arch_type={arch!r} indicates a "
                f"league/PSRO (non-SPAR-family) model. Load it with --is_league True "
                f"via the league eval path (load_league_model), not the SPAR-family "
                f"path."
            )
    info = {
        "resolved_class": cls.__name__,
        "model_arch_type": arch,
        "ego_head_in_weights": has_ego,
        "detection_path": (
            f"{cls.__name__} (ego value head "
            f"{'present' if has_ego else 'absent'} in weights); {note}"
        ),
    }
    return cls, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_prot", type=str, required=True)
    parser.add_argument("--main_checkpoint_model_path", type=str, required=True)
    parser.add_argument("--done_model_checkpoint_path", type=str, required=True)
    parser.add_argument("--br_checkpoint_model_path", type=str, required=True)
    parser.add_argument("--full_state_list", type=str, required=True)
    parser.add_argument("--state_list", type=str, required=True)
    parser.add_argument("--use_mirror", type=str, required=True)
    parser.add_argument("--dedicated_exploiter", type=str, required=True)
    parser.add_argument("--br_index", type=int, required=True)
    parser.add_argument("--game_args", type=str, required=True)
    # When True, main_checkpoint_model_path is a LeaguePPO checkpoint and
    # the eval path uses model.policy / model.policy_other instead of CDS
    # heads. BR model still must be an Exploiter (dedicated mode only).
    parser.add_argument("--is_league", type=str, default="False")
    # Per-training-process segregation. Outputs go to
    #   br_rewards/<output_subdir>/...     and
    #   selfplay_rewards/<output_subdir>/...
    # when set. Empty string preserves the legacy unsegregated layout.
    # Computed by the launcher in new_br_worker.py from the source dir
    # (league) or the .task filename prefix (SPAR).
    parser.add_argument("--output_subdir", type=str, default="")
    # Training style label ("league", "ippo", "spar", "2timescale", ...).
    # Prepended to the output .txt filename so aggregate_local_eval_data.py
    # can surface it in plot filenames. Empty string preserves the legacy
    # unprefixed filename format.
    parser.add_argument("--training_style", type=str, default="")
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="",
        help="If set, this string is sanitized and injected into the "
             "reward .txt filename just before the trailing '_.txt'. "
             "Used by the BR worker's periodic-eval callback to keep "
             "mid-training snapshots (every N env-steps) from "
             "overwriting each other or the final eval. Format the "
             "caller uses: 'brstep<N>_<YYYYMMDDTHHMMSS>'. Empty "
             "(default) preserves the legacy filename — the final "
             "post-learn() eval call leaves this empty so it lands at "
             "the canonical filename the aggregator parses.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for loaded models (e.g. cpu, cuda:0)",
    )
    return parser


def _extract_step_outputs(step_output):
    """Handle both 2-player VecEnv step signatures and standard Gym-style signatures."""
    if len(step_output) == 5:
        obs, reward, _, done, info = step_output
    else:
        obs, reward, done, info = step_output
    reward = np.asarray(reward).reshape(-1)
    done = np.asarray(done).reshape(-1).astype(bool)
    return obs, reward, done, info


def _collect_episode_returns(model, target_episodes, action_fn):
    """Collect per-episode returns from vectorized envs that finish asynchronously."""
    obs = model.env.reset()
    n_envs = model.env.num_envs
    running_returns = np.zeros(n_envs, dtype=np.float32)
    finished_returns = []

    while len(finished_returns) < target_episodes:
        clipped_action = action_fn(obs)
        obs, reward, done, info = _extract_step_outputs(model.env.step(clipped_action))
        running_returns += reward

        done_indices = np.where(done)[0]
        for idx in done_indices:
            finished_returns.append(float(running_returns[idx]))
            running_returns[idx] = 0.0
            print(f"Episode {len(finished_returns)} completed", flush=True)
            if len(finished_returns) >= target_episodes:
                break

    return finished_returns


def _extract_left_right_names_from_state(state):
    if not state:
        return "unknown_left", "unknown_right"
    basename = os.path.basename(str(state))
    matchup_match = re.search(r"\.([A-Za-z]+)Vs([A-Za-z]+)\.2Player\.state$", basename)
    if matchup_match:
        return matchup_match.group(1), matchup_match.group(2)
    return "unknown_left", "unknown_right"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data_dict = json.loads(args.game_args)
    args.game_args = argparse.Namespace(**data_dict)
    args.state_list = ast.literal_eval(args.state_list)
    args.full_state_list = ast.literal_eval(args.full_state_list)
    args.dedicated_exploiter = args.dedicated_exploiter == "True"
    args.eval_prot = args.eval_prot == "True"
    args.is_league = args.is_league == "True"
    # Sanitize the subdir name (defensive: launcher already sanitizes, but
    # we re-apply to guard against direct CLI invocations).
    args.output_subdir = _sanitize_for_filename(args.output_subdir or "")
    args.training_style = _sanitize_for_filename(args.training_style or "")
    args.filename_suffix = _sanitize_for_filename(args.filename_suffix or "")

    main_checkpoint_model_path = args.main_checkpoint_model_path
    done_model_checkpoint_path = args.done_model_checkpoint_path
    br_model_path = args.br_checkpoint_model_path

    # Per-training-process segregation. When --output_subdir is non-empty,
    # outputs nest under it so different main training runs don't collide.
    # Empty string preserves the legacy unsegregated layout.
    _workdir = os.environ.get("WORKDIR")
    _main_training_dir = os.environ.get("MAIN_TRAINING_DIR")
    if _workdir and _main_training_dir:
        base_dir = os.path.join(_workdir, _main_training_dir, "FightLadder", "main")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    br_rewards_folder = os.path.join(base_dir, "br_rewards")
    selfplay_rewards_folder = os.path.join(base_dir, "selfplay_rewards")
    if args.output_subdir:
        br_rewards_folder = os.path.join(br_rewards_folder, args.output_subdir)
        selfplay_rewards_folder = os.path.join(selfplay_rewards_folder, args.output_subdir)
    os.makedirs(br_rewards_folder, exist_ok=True)
    os.makedirs(selfplay_rewards_folder, exist_ok=True)

    env = env_generator(args.game_args, STATE=args.state_list)
    full_env = env_generator(args.game_args, STATE=args.full_state_list)
    # ENV_ID = args.env_id
    # Which load path / arch each model went down, for the end-of-run printout.
    model_detection = None
    br_detection = None
    if args.is_league:
        # League path: reuse load_league_model so the unpickling /
        # league_constructor logic stays in one place. The function builds
        # an internal single-side league env we don't need; we override
        # model.env with the SPAR-style 2-player VecEnv from env_generator
        # so _collect_episode_returns can step joint [ego, adv] actions.
        try:
            model = load_league_model(
                vars(args.game_args),
                main_checkpoint_model_path,
                league_matchup_states=args.full_state_list,
                n_envs=env.num_envs,
                device=args.device,
                use_wandb=False,
            )
        except FileNotFoundError:
            model = load_league_model(
                vars(args.game_args),
                done_model_checkpoint_path,
                league_matchup_states=args.full_state_list,
                n_envs=env.num_envs,
                device=args.device,
                use_wandb=False,
            )
        model.env = env
        model_detection = {
            "resolved_class": type(model).__name__,
            "model_arch_type": getattr(model, "model_arch_type", None),
            "ego_head_in_weights": None,
            "detection_path": "league path (--is_league) via load_league_model",
        }
    else:
        # Auto-select spar vs ippo/2timescale from the checkpoint's policy weights
        # (resolver lets FileNotFoundError through so the done-checkpoint fallback works).
        try:
            _mcls, model_detection = _resolve_cds_family_class(main_checkpoint_model_path)
            model = _mcls.load(
                main_checkpoint_model_path, env=env, num_perturbed=1, device=args.device
            )
        except FileNotFoundError:
            _mcls, model_detection = _resolve_cds_family_class(done_model_checkpoint_path)
            model = _mcls.load(
                done_model_checkpoint_path, env=env, num_perturbed=1, device=args.device
            )

# if args.eval_prot is True: # we're training an optimal adversary
#     dstb_action_space = Box(low=model.dstb_action_space.low, high=model.dstb_action_space.high, shape=model.dstb_action_space.shape)
#     env.action_space = dstb_action_space
# else:
#     assert args.eval_prot is False 
#     # we're training an optimal ego against the current adversary
#     ego_action_space = Box(low=model.action_space.low, high=model.action_space.high, shape=model.action_space.shape)
#     env.action_space = ego_action_space

# if args.exploiter_is_cds:
#     pass
# else:
#     env.action_space = model.dstb_action_space

    if not args.dedicated_exploiter:
        if args.is_league:
            # Continue-mode league BR is a LeaguePPO file (saved by
            # _run_league_continue_exploiter), not an Exploiter zip, and
            # new_br_worker.py's launcher does not populate br_model_path
            # for that path. Refuse here rather than silently loading the
            # wrong format.
            raise NotImplementedError(
                "local_br_eval does not support league + continue-exploiter "
                "mode. Run with --dedicated_exploiter True (BR is an "
                "Exploiter checkpoint), or eval league outputs via the "
                "league-native eval path."
            )
        # Auto-select spar vs ippo/2timescale; resolved before the try so it is
        # also used by the strict=False reconstruction fallback below.
        _br_cls, br_detection = _resolve_cds_family_class(br_model_path)
        try:
            br_model = _br_cls.load(
                br_model_path, env=env, num_perturbed=1, device=args.device
            )
        except RuntimeError as exc:
            # Per-matchup override in load_br_checkpoints.load_and_continue
            # aliases policy heads (value_net, dstb_action_net) without
            # pruning the original ModuleDict entries, and never rebuilds
            # the optimizers to match the post-override num_adversaries=1
            # geometry. Two failure modes hit a normal SB3 load:
            #   1) RuntimeError "Unexpected key(s) in state_dict" -- dormant
            #      heads from other matchups (GuileVsRyu_*, etc.).
            #   2) ValueError "parameter group that doesn't match the size
            #      of optimizer's group" -- optimizer state was saved with
            #      N param groups (pre-override geometry) but freshly built
            #      with 1 group from the saved num_adversaries=1 metadata.
            # SB3's BaseAlgorithm.load hard-codes exact_match=True and only
            # has a retry path for SB3<1.7 missing-pi_features_extractor
            # cases, so we bypass set_parameters entirely on retry: load
            # the zip directly, rebuild the model, and load ONLY the policy
            # weights with strict=False. Optimizers are skipped because
            # eval never calls .learn().
            if "Unexpected key(s) in state_dict" not in str(exc):
                raise
            print(
                f"[local_br_eval] strict checkpoint load failed; retrying "
                f"with policy-only strict=False load (bypassing optimizers) "
                f"to tolerate per-matchup override dormant heads + optimizer "
                f"geometry mismatch. Error: {exc}",
                flush=True,
            )
            from stable_baselines3.common.save_util import load_from_zip_file
            data, params, _ = load_from_zip_file(
                br_model_path, device=args.device
            )
            # Mirror SB3's BaseAlgorithm.load preflight: strip stored device
            # from policy_kwargs so the freshly-constructed policy uses ours.
            if (
                "policy_kwargs" in data
                and isinstance(data["policy_kwargs"], dict)
                and "device" in data["policy_kwargs"]
            ):
                del data["policy_kwargs"]["device"]
            # `n_envs` must reflect the current env, not the saved one.
            if env is not None and hasattr(env, "num_envs"):
                data["n_envs"] = env.num_envs
            br_model = _br_cls(
                policy=data["policy_class"],
                env=env,
                device=args.device,
                _init_setup_model=False,
            )
            br_model.__dict__.update(data)
            br_model._setup_model()
            # Eval-only: load policy weights, accept missing/unexpected keys.
            # Missing keys (e.g., q_value_net.<matchup>_0 if the override
            # didn't alias q_value_net) stay at their fresh init values.
            missing, unexpected = br_model.policy.load_state_dict(
                params["policy"], strict=False,
            )
            print(
                f"[local_br_eval] policy.load_state_dict(strict=False): "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys.",
                flush=True,
            )
            # Missing value/q-value heads are harmless for eval (value_forward=False),
            # but missing ACTOR weights mean a random-init actor -> invalid eval. Fail
            # loud rather than silently reporting garbage win rates.
            missing_actor = [k for k in missing if k.startswith(_ACTOR_KEY_PREFIXES)]
            if missing_actor:
                raise RuntimeError(
                    f"[local_br_eval] policy load is missing ACTOR weights "
                    f"{missing_actor[:6]}{'...' if len(missing_actor) > 6 else ''} -- "
                    f"eval would run a random-initialized actor. Aborting. This usually "
                    f"means the checkpoint's architecture does not match the resolved "
                    f"class ({_br_cls.__name__})."
                )
            if missing:
                print(
                    f"[local_br_eval] WARNING: missing keys include: "
                    f"{missing[:6]}{'...' if len(missing) > 6 else ''} -- "
                    f"these will use random initialization (may affect eval "
                    f"if the forward path exercises them).",
                    flush=True,
                )
    else:
        # if args.eval_prot is True:  # we're training an optimal adversary
        #     dstb_action_space = Box(
        #         low=model.dstb_action_space.low,
        #         high=model.dstb_action_space.high,
        #         shape=model.dstb_action_space.shape,
        #     )
        #     env.action_space = dstb_action_space
        # else:
        #     assert args.eval_prot is False
        #     # we're training an optimal ego against the current adversary
        #     ego_action_space = Box(
        #         low=model.action_space.low,
        #         high=model.action_space.high,
        #         shape=model.action_space.shape,
        #     )
        #     env.action_space = ego_action_space
        # print("#$%*&^%$EVAL PROT: %s$%^&*", args.eval_prot)
        # print("$@#$%^&*()(*&^%$#@)%s$#%^&*()(*&^%$#@", args.exploiter_is_cds)
        br_model = Exploiter.load(br_model_path, env=env, n_envs=1, device=args.device)
        br_detection = {
            "resolved_class": type(br_model).__name__,
            "model_arch_type": getattr(br_model, "model_arch_type", None),
            "ego_head_in_weights": None,
            "detection_path": "dedicated Exploiter (Exploiter.load)",
        }

    nr = 50
    exploiter_rewards, selfplay_rewards = [], []
    model_policy_for_eval = model.policy
    use_fixed_matchup_adapter = False
    ego_is_left = True
    if args.dedicated_exploiter and not args.is_league:
        # CDS-only: dedicated runs pass a repeated singleton state list;
        # intercept policy forward calls on the exploited CDS model through
        # one fixed matchup head. League models have no matchup heads
        # (standard SB3 policy + policy_other), so this adapter does not
        # apply.
        eval_unique_states = _dedupe_preserve_order(args.state_list)
        if len(eval_unique_states) == 1:
            dedicated_state = eval_unique_states[0]
            model_unique_states = _dedupe_preserve_order(getattr(model, "state_list", []))
            if dedicated_state in model_unique_states:
                fixed_matchup_idx = model_unique_states.index(dedicated_state)
                model_policy_for_eval = _FixedMatchupPolicyAdapter(
                    model.policy, fixed_matchup_idx=fixed_matchup_idx
                )
                use_fixed_matchup_adapter = True
                halfway = len(model_unique_states) // 2
                ego_is_left = fixed_matchup_idx < halfway
                print(
                    "Configured fixed-matchup eval adapter: "
                    f"state={dedicated_state}, fixed_matchup_idx={fixed_matchup_idx}, "
                    f"ego_is_left={ego_is_left}",
                    flush=True,
                )

    #use_mirror = getattr(args.game_args, 'use_mirror', False)

    def _build_side_flags(n_envs, reduced_state_list, full_state_list, use_mirror):
        if len(set(reduced_state_list)) ==1:
            index = full_state_list.index(reduced_state_list[0])
            ego_is_left = index < len(full_state_list) // 2
            vals = np.zeros(n_envs, dtype=np.float32) if ego_is_left else np.ones(n_envs, dtype=np.float32)
            return th.tensor(vals, device=model.device).unsqueeze(1), 1.0 - th.tensor(vals, device=model.device).unsqueeze(1)
        if use_mirror:
            halfway = len(full_state_list) // 2
            vals = np.array([0.0 if i < halfway else 1.0 for i in range(n_envs)], dtype=np.float32)
            #vals = np.ones(n_envs, dtype=np.float32)
        else:
            vals = np.zeros(n_envs, dtype=np.float32)
        ego_sf = th.tensor(vals, device=model.device).unsqueeze(1)
        return ego_sf, 1.0 - ego_sf

    exp_ego_sf, exp_adv_sf = _build_side_flags(env.num_envs, args.state_list, args.full_state_list, args.use_mirror)
    full_ego_sf, full_adv_sf = _build_side_flags(full_env.num_envs, args.full_state_list, args.full_state_list, args.use_mirror)

    def exploiter_action_fn(obs):
        with th.no_grad():
            obs_model_tensor = obs_as_tensor(obs, model.device)
            if args.is_league:
                # League frozen-side selection mirrors the new_br_worker.py
                # league branch:
                #   eval_prot=True  -> exploit ego, frozen=left  -> model.policy
                #   eval_prot=False -> exploit adv, frozen=right -> model.policy_other
                if args.eval_prot:
                    action, _, _ = model.policy(obs_model_tensor)
                else:
                    action, _, _ = model.policy_other(obs_model_tensor)
            elif args.dedicated_exploiter and use_fixed_matchup_adapter:
                #ego_sf_val = 0.0 if ego_is_left else 1.0
                #ego_sf = th.full((obs_model_tensor.shape[0], 1), ego_sf_val, device=model.device)
                #adv_sf = 1.0 - ego_sf
                action, _ = model_policy_for_eval(
                    obs_model_tensor,
                    deterministic=False,
                    ego_forward=args.eval_prot,
                    adv_forward=not args.eval_prot,
                    ego_side_flag=exp_ego_sf,
                    adv_side_flag=exp_adv_sf,
                )
            else:
                if args.eval_prot:
                    action, _, _, _, _, _ = model.policy(
                        obs_model_tensor, ego_side_flag=exp_ego_sf, adv_side_flag=exp_adv_sf,
                    )
                else:
                    _, _, action, _, _, _ = model.policy(
                        obs_model_tensor, ego_side_flag=exp_ego_sf, adv_side_flag=exp_adv_sf,
                    )
            # BR is always an Exploiter for league (continue+league guarded
            # above), so the dedicated branch covers it.
            if args.dedicated_exploiter:
                action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
            else:
                if args.eval_prot:
                    _, _, action_br, _, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
                else:
                    action_br, _, _, _, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
        action = action.cpu().numpy()
        action_br = action_br.cpu().numpy()
        # eval_prot=True → exploiting="ego"; mirrors Exploiter's
        # (self.exploiting=="ego") == self.ego_is_left logic.
        exploited_on_left = args.eval_prot == ego_is_left
        if exploited_on_left:
            return np.hstack([action, action_br])
        return np.hstack([action_br, action])
# for i in range(nr):
#     curr_reward = 0
#     obs = model.env.reset()
#     obs = np.expand_dims(obs, 0)
#     done = False
#     while not done:
#         with th.no_grad():
#             if args.exploiter_is_cds:
#                 ego_actions, ego_log_probs, action_br, adv_log_probs, values, q_values = br_model.policy(obs_as_tensor(obs, br_model.device), deterministic=False, ego_forward=True, adv_forward=True, zero_ego_action=False, zero_adv_action=True)
#             else:
#                 action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
#             action, _, adv_action, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
#         action = action.cpu().numpy()

#         clipped_action = np.hstack([action_br, adv_action])
#         obs, reward, done, info = model.env.step(clipped_action)
#         curr_reward += reward
#     exploiting_adv_rewards.append(curr_reward)
#     print(f"Episode {i+1} completed")

    def selfplay_action_fn(obs):
        with th.no_grad():
            obs_tensor = obs_as_tensor(obs, model.device)
            if args.is_league:
                # LeaguePPO stores left and right policies separately;
                # query both for the joint [left, right] action vector.
                action, _, _ = model.policy(obs_tensor)
                adv_action, _, _ = model.policy_other(obs_tensor)
            # elif args.use_mirror:
            #     #ego_sf_val = 0.0 if ego_is_left else 1.0
            #     #ego_sf = th.full((obs_tensor.shape[0], 1), ego_sf_val, device=model.device)
            #     #adv_sf = 1.0 - ego_sf
            #     action, adv_action = model_policy_for_eval(
            #         obs_tensor,
            #         deterministic=False,
            #         ego_forward=True,
            #         adv_forward=True,
            #         ego_side_flag=full_ego_sf,
            #         adv_side_flag=full_adv_sf,
            #     )
            else:
                action, _, adv_action, _, _, _ = model.policy(
                    obs_tensor,
                    ego_side_flag=full_ego_sf,
                    adv_side_flag=full_adv_sf,
                    value_forward=False,
                    q_value_forward=False
                )
        action = action.cpu().numpy()
        adv_action = adv_action.cpu().numpy()
        return np.hstack([action, adv_action])

    exploiter_rewards = _collect_episode_returns(model, nr, exploiter_action_fn)
    model.env = full_env
    selfplay_rewards = _collect_episode_returns(model, nr, selfplay_action_fn)

    # TODO: write out to a file and then aggregate the results and plot
    # os.makedirs(rewards_folder, exist_ok=True)
    main_side = "left" if args.eval_prot else "right"
    exploiter_side = "left" if not args.eval_prot else "right"
    tested_states = _dedupe_preserve_order(args.state_list)
    tested_state = tested_states[0] if tested_states else ""
    left_name, right_name = _extract_left_right_names_from_state(tested_state)
    main_name = left_name if main_side == "left" else right_name
    exploiter_name = right_name if exploiter_side == "right" else left_name
    # Optional training-style prefix. Aggregate parses it out of the
    # filename via FILENAME_RE; legacy unprefixed files still match because
    # the style group in that regex is optional.
    style_prefix = f"{args.training_style}_" if args.training_style else ""
    # Disambiguator suffix: encodes exploiter type (continue vs dedicated)
    # and br_index so multiple exploiters of the same side and matchup
    # don't overwrite each other's reward files. Aggregate parses these
    # to plot continue and dedicated as separate series and to keep
    # replicates as separate scatter points.
    exp_type = "dedicated" if args.dedicated_exploiter else "continue"
    # Optional periodic-eval suffix: when the BR worker's periodic
    # callback invokes us mid-training, it passes a sanitized string
    # like "brstep5000000_20260616T143055" which gets inserted between
    # the br index and the trailing "_.txt". Empty for the final eval
    # so canonical filenames remain unchanged (and the aggregator
    # regex keeps matching).
    suffix_part = f"{args.filename_suffix}_" if args.filename_suffix else ""
    filename = (
        f"{style_prefix}{model.num_timesteps}_main_{main_side}_{main_name}_"
        f"exploiter_{exploiter_side}_{exploiter_name}_"
        f"{exp_type}_br{args.br_index}_{suffix_part}.txt"
    )
    with open(os.path.join(br_rewards_folder, filename), "w") as f:
        f.write(str(np.mean(exploiter_rewards)))
    with open(os.path.join(selfplay_rewards_folder, filename), "w") as f:
        f.write(str(np.mean(selfplay_rewards)))

    eval_target = "ego" if args.eval_prot else "adv"
    tested_state_for_print = tested_state if len(tested_states) == 1 else tested_states
    run_summary = {
        "checkpoint_num_timesteps": int(model.num_timesteps),
        "br_index": args.br_index,
        "tested_state": tested_state_for_print,
        "left_name": left_name,
        "right_name": right_name,
        "output_filename": filename,
        "eval_target": eval_target,
        "dedicated_exploiter": args.dedicated_exploiter,
        "device": args.device,
        "main_checkpoint_model_path": args.main_checkpoint_model_path,
        "done_model_checkpoint_path": args.done_model_checkpoint_path,
        "br_checkpoint_model_path": args.br_checkpoint_model_path,
        "full_state_list": args.full_state_list,
        "state_list": args.state_list,
        "model_detection": model_detection,
        "br_detection": br_detection,
    }
    print(
        f"local br eval complete for checkpoint {model.num_timesteps} | "
        f"br_index={args.br_index} | state={tested_state_for_print} | eval_target={eval_target}",
        flush=True,
    )

    def _fmt_detection(d):
        if not d:
            return "unknown"
        return f"{d.get('detection_path', '?')} | model_arch_type={d.get('model_arch_type')!r}"

    print(f"[local_br_eval] main model detection: {_fmt_detection(model_detection)}", flush=True)
    print(f"[local_br_eval] br model detection:   {_fmt_detection(br_detection)}", flush=True)
    print("local br eval args:", flush=True)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)

    
if __name__ == "__main__":
    main()
    print("local br eval complete")