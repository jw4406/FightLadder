"""
Loader for BR checkpoint directories.

Given a path containing BR model checkpoints (zip files saved by
ExploiterCheckpointCallback during continue-mode BR training), this module:

  1. Parses filenames to identify unique BR models and their latest checkpoint.
  2. Loads each checkpoint's metadata to determine:
     - CDS architecture type ("ippo" or "spar")
     - The state list used during training
     - Which side was being updated (ego or adversary)
  3. Provides a function to continue learning from any checkpoint.

Usage:
    python load_br_checkpoints.py --br_dir <path> --game_args_json <json> [--continue_training] [--device cuda]

This script lives in FightLadder/main and should be run from there
(or with cwd set to FightLadder/main) so that sibling modules like
ippo, utils, common.*, etc. are importable.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Ensure the custom SB3 fork (sibling of this directory) is importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SB3_DIR = os.path.join(os.path.dirname(_THIS_DIR), "stable_baselines3")
if _SB3_DIR not in sys.path:
    sys.path.insert(0, _SB3_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_FILENAME_RE = re.compile(
    r"^(?P<prefix>.+?\.zip)_(?P<steps>\d+)_steps\.zip$"
)

_EXPLOIT_SIDE_RE = re.compile(r"_exploiting_(?P<side>ego|adv)_")


def parse_br_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Extract name_prefix and step count from a BR checkpoint filename."""
    m = _FILENAME_RE.match(filename)
    if m is None:
        return None
    prefix = m.group("prefix")
    steps = int(m.group("steps"))

    side_m = _EXPLOIT_SIDE_RE.search(prefix)
    exploit_side = side_m.group("side") if side_m else None

    return {
        "prefix": prefix,
        "steps": steps,
        "exploit_side": exploit_side,
        "filename": filename,
    }


def scan_br_directory(br_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan a directory of BR checkpoints and return the latest checkpoint
    for each unique BR model (identified by filename prefix).

    Returns:
        dict mapping prefix -> {
            "prefix": str,
            "steps": int,
            "exploit_side": "ego" | "adv",
            "filename": str,
            "path": str,
        }
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for fname in os.listdir(br_dir):
        if not fname.endswith(".zip"):
            continue
        parsed = parse_br_filename(fname)
        if parsed is None:
            continue
        prefix = parsed["prefix"]
        if prefix not in groups or parsed["steps"] > groups[prefix]["steps"]:
            parsed["path"] = os.path.join(br_dir, fname)
            groups[prefix] = parsed
    return groups


def _read_raw_json_from_zip(path: str) -> Dict[str, Any]:
    """
    Read the ``data`` JSON blob from an SB3 zip checkpoint and return
    the raw JSON dict *without* deserializing cloudpickle objects.

    Pickled entries (dicts with a `:serialized:` key) are left as-is
    so we never trigger module imports (e.g. ippo → utils) that may
    not be on sys.path.
    """
    import json as _json
    import zipfile
    with zipfile.ZipFile(path, "r") as zf:
        raw = zf.read("data").decode()
    return _json.loads(raw)


def _safe_get(raw: Dict, key: str, default=None):
    """Get a value from the raw JSON dict, skipping serialized blobs."""
    val = raw.get(key, default)
    if isinstance(val, dict) and ":serialized:" in val:
        return default
    return val


def inspect_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a BR checkpoint's metadata (data dict only, no model reconstruction).

    Uses raw JSON parsing to avoid cloudpickle deserialization, so this
    works without heavy dependencies (torch, retro, ippo, utils, etc.).

    Returns a dict with:
        - arch: "ippo" or "spar"
        - unique_states: list of unique state strings
        - state_list: full state list (with repeats)
        - exploit_side: "ego" or "adv" (from data or filename)
        - training_br: bool
        - num_adversaries: int
        - envs_per_matchup: int
        - matchups: list
        - checkpoint_basename: str (the base CDS checkpoint this BR was derived from)
        - num_timesteps: int (timesteps completed so far)
        - total_timesteps: int (original total target)
    """
    raw = _read_raw_json_from_zip(path)

    # --- Architecture inference (mirrors br_preflight.infer_cds_architecture) ---
    explicit_arch = _safe_get(raw, "model_arch_type")
    if isinstance(explicit_arch, str) and explicit_arch.strip().lower() in ("ippo", "spar"):
        arch = explicit_arch.strip().lower()
    else:
        arch = _safe_get(raw, "_worker_cds_arch", "spar")
        if isinstance(arch, str) and arch.strip().lower() in ("ippo", "spar"):
            arch = arch.strip().lower()
        else:
            policy_cls = _safe_get(raw, "policy_class")
            if isinstance(policy_cls, str) and "IPPO" in policy_cls:
                arch = "ippo"
            else:
                basename = os.path.basename(path).lower()
                arch = "ippo" if "ippo" in basename else "spar"

    unique_states = _safe_get(raw, "_worker_unique_states", [])
    state_list = _safe_get(raw, "state_list", [])
    if not unique_states and state_list:
        unique_states = list(dict.fromkeys(state_list))

    stop_key = _safe_get(raw, "br_manual_stop_key", "")
    exploit_side = None
    if "_ego" in str(stop_key):
        exploit_side = "ego"
    elif "_adv" in str(stop_key):
        exploit_side = "adv"

    game_args_raw = _safe_get(raw, "game_args")
    if isinstance(game_args_raw, dict) and ":serialized:" not in game_args_raw:
        game_args_dict = game_args_raw
    else:
        game_args_dict = None

    return {
        "arch": arch,
        "unique_states": unique_states,
        "state_list": state_list,
        "exploit_side": exploit_side,
        "training_br": _safe_get(raw, "training_br", False),
        "num_adversaries": _safe_get(raw, "num_adversaries", 1),
        "envs_per_matchup": _safe_get(raw, "envs_per_matchup", 1),
        "matchups": _safe_get(raw, "matchups", []),
        "checkpoint_basename": _safe_get(raw, "_checkpoint_basename", ""),
        "num_timesteps": _safe_get(raw, "_num_timesteps_at_start", 0),
        "total_timesteps": _safe_get(raw, "_total_timesteps", 10_000_000),
        "policy_class": _safe_get(raw, "policy_class"),
        "game_args": game_args_dict,
    }


def load_and_continue(
    checkpoint_path: str,
    game_args: Optional[dict],
    exploit_side: str,
    arch: str,
    unique_states: List[str],
    n_envs: int = 2,
    device: str = "cuda",
    total_timesteps: int = 10_000_000,
    use_wandb: bool = False,
    callback=None,
) -> None:
    """
    Load a CDS model from a BR checkpoint and continue training
    whichever side was being updated.

    Args:
        checkpoint_path: Path to the BR checkpoint zip.
        game_args: Dict of game arguments (will be wrapped in argparse.Namespace).
            If None, game_args are extracted from the checkpoint itself.
        exploit_side: "ego" or "adv" — which side was being exploited.
            "ego" → adversary was learning (update_adversary=True).
            "adv" → ego was learning (update_ego=True).
        arch: "ippo" or "spar".
        unique_states: Unique state strings from the checkpoint
            (_worker_unique_states — the full pre-override set).
        n_envs: Number of environments per matchup.
        device: Torch device.
        total_timesteps: Total timesteps for the continued training run.
        use_wandb: Whether to enable wandb logging.
        callback: Optional SB3 callback for the learn() call.
    """
    from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
    from common.justin.clean_derivative_free_spar_ippo import CleanDerivativeFreeSPARIPPO
    from stable_baselines3.common.save_util import load_from_zip_file
    from ippo import env_generator
    from utils import state2matchup
    import numpy as np

    raw = _read_raw_json_from_zip(checkpoint_path)
    ckpt_matchups = _safe_get(raw, "matchups", [])
    ckpt_unique_matchups = list(dict.fromkeys(ckpt_matchups))
    ckpt_num_adv = _safe_get(raw, "num_adversaries", None)

    if game_args is None:
        data_tmp, _, _ = load_from_zip_file(checkpoint_path, device="cpu")
        ga_raw = data_tmp.get("game_args")
        if ga_raw is None:
            raise ValueError(
                f"No game_args found in checkpoint {checkpoint_path} and none provided."
            )
        game_args = vars(ga_raw) if hasattr(ga_raw, "__dict__") else dict(ga_raw)

    cds_cls = CleanDerivativeFreeSPARIPPO if arch == "ippo" else CleanDerivativeFreeSPAR

    FULL_STATE = [s for s in unique_states for _ in range(n_envs)]
    ns_game_args = argparse.Namespace(**game_args)
    env = env_generator(ns_game_args, STATE=FULL_STATE)

    try:
        ftm = cds_cls.load(
            path=checkpoint_path, env=env, game_args=ns_game_args,
            num_perturbed=1, device=device,
        )
    except (RuntimeError, ValueError):
        data, params, _ = load_from_zip_file(checkpoint_path, device=device)
        ftm = cds_cls(
            "AACCnnPolicy", env, device=device, verbose=2,
            n_steps=256, batch_size=512, n_epochs=1,
            state_list=FULL_STATE, envs_per_matchup=n_envs,
            env_generator_func=env_generator,
            num_adversaries=len(unique_states), n_env_per_adv=n_envs, seed=0,
            target_kl=0.025, use_mirror=False, use_wandb=use_wandb,
        )
        if "policy" in params:
            missing, unexpected = ftm.policy.load_state_dict(
                params["policy"], strict=False,
            )
            print(
                f"[load_br_checkpoints] Policy loaded (strict=False): "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys",
                flush=True,
            )
        elo_r = data.get("elo_adversary_ratings")
        elo_g = data.get("elo_games_played")
        if elo_r is not None:
            ftm.elo_adversary_ratings = np.array(elo_r, dtype=np.float64)
        if elo_g is not None:
            ftm.elo_games_played = np.array(elo_g, dtype=np.int64)

    env.close()
    ftm.env = None

    per_matchup = (
        len(unique_states) > 1
        and (
            len(ckpt_unique_matchups) == 1
            or (ckpt_num_adv is not None and ckpt_num_adv < len(unique_states))
        )
    )
    if per_matchup:
        dedicated_matchup = ckpt_unique_matchups[0] if ckpt_unique_matchups else state2matchup(unique_states[0])
        dedicated_state = next(
            (s for s in unique_states if state2matchup(s) == dedicated_matchup),
            unique_states[0],
        )
        full_matchup_labels = [state2matchup(s) for s in unique_states]
        fixed_idx = (
            full_matchup_labels.index(dedicated_matchup)
            if dedicated_matchup in full_matchup_labels
            else 0
        )
        old_key = f"{dedicated_matchup}_{fixed_idx}"
        new_key = f"{dedicated_matchup}_0"
        aliased = []
        for attr in ("value_net", "dstb_action_net"):
            hd = getattr(ftm.policy, attr, None)
            if hd is not None and old_key in hd and new_key not in hd:
                hd[new_key] = hd[old_key]
                aliased.append(attr)

        if hasattr(ftm, "elo_adversary_ratings") and len(ftm.elo_adversary_ratings) > fixed_idx:
            ftm.elo_adversary_ratings = ftm.elo_adversary_ratings[
                fixed_idx : fixed_idx + 1
            ].copy()
        if hasattr(ftm, "elo_games_played") and len(ftm.elo_games_played) > fixed_idx:
            ftm.elo_games_played = ftm.elo_games_played[
                fixed_idx : fixed_idx + 1
            ].copy()

        ftm.matchups = [dedicated_matchup] * n_envs
        ftm.state_list = [dedicated_state] * n_envs
        ftm.envs_per_matchup = n_envs
        ftm.num_adversaries = 1
        for pattr, val in [
            ("matchups", ftm.matchups),
            ("envs_per_matchup", n_envs),
            ("num_adversaries", 1),
            ("num_env_per_adv", n_envs),
        ]:
            if hasattr(ftm.policy, pattr):
                setattr(ftm.policy, pattr, val)

        effective_state_list = [dedicated_state] * n_envs
        print(
            f"[load_br_checkpoints] Per-matchup override: "
            f"matchup={dedicated_matchup}, idx={fixed_idx}, "
            f"alias {old_key}->{new_key} on {aliased}",
            flush=True,
        )
    else:
        effective_state_list = FULL_STATE

    eval_prot = (exploit_side == "ego")
    update_ego = not eval_prot
    update_adversary = eval_prot

    env = env_generator(ns_game_args, STATE=effective_state_list, n_envs=n_envs)
    ftm.env = env

    actual_n_envs = ftm.env.num_envs
    if ftm.n_envs != actual_n_envs:
        from stable_baselines3.common.buffers import DictRolloutBuffer, Q_RolloutBuffer
        from gymnasium import spaces
        ftm.n_envs = actual_n_envs
        buffer_cls = (
            DictRolloutBuffer
            if isinstance(ftm.observation_space, spaces.Dict)
            else Q_RolloutBuffer
        )
        ftm.rollout_buffer = buffer_cls(
            ftm.n_steps, ftm.observation_space, ftm.action_space,
            device=ftm.device, gamma=ftm.gamma, gae_lambda=ftm.gae_lambda,
            n_envs=actual_n_envs,
        )
        if hasattr(ftm, "adversary_buffers") and ftm.adversary_buffers is not None:
            adversary_buffers = []
            for _ in range(int(ftm.num_adversaries)):
                adversary_buffers.append(
                    buffer_cls(
                        ftm.n_steps, ftm.observation_space, ftm.action_space,
                        device=ftm.device, gamma=ftm.gamma,
                        gae_lambda=ftm.gae_lambda,
                        n_envs=ftm.envs_per_matchup,
                    )
                )
            ftm.adversary_buffers = adversary_buffers

    if hasattr(ftm.policy, "num_env_per_adv"):
        ftm.policy.num_env_per_adv = ftm.envs_per_matchup
    if hasattr(ftm.policy, "envs_per_matchup"):
        ftm.policy.envs_per_matchup = ftm.envs_per_matchup

    ftm.exploited = None
    ftm.training_br = True
    ftm.use_wandb = use_wandb
    ftm.use_lr_annealing = False
    ftm.c_learning_rate = 1e-4
    ftm.d_learning_rate = 2e-4
    ftm.v_learning_rate = 5e-4
    ftm.policy.ctrl_optimizer.param_groups[0]["lr"] = 1e-4
    ftm.policy.dstb_optimizer.param_groups[0]["lr"] = 1e-4
    ftm.policy.value_optimizer.param_groups[0]["lr"] = 2e-4

    learning_side = "adversary" if eval_prot else "ego"
    elo_str = ""
    if hasattr(ftm, "elo_adversary_ratings"):
        elo_str = f"\n  elo_ratings={ftm.elo_adversary_ratings.tolist()}"
    print(
        f"[load_br_checkpoints] Continuing training from {os.path.basename(checkpoint_path)}\n"
        f"  arch={arch}, exploit_side={exploit_side}, learning_side={learning_side}\n"
        f"  update_ego={update_ego}, update_adversary={update_adversary}\n"
        f"  unique_states={unique_states}\n"
        f"  effective_states={list(dict.fromkeys(effective_state_list))}\n"
        f"  total_timesteps={total_timesteps}, device={device}{elo_str}",
        flush=True,
    )

    ftm.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        update_ego=update_ego,
        update_adversary=update_adversary,
    )
    return ftm


def summarize_br_models(br_dir: str) -> List[Dict[str, Any]]:
    """
    Scan a BR checkpoint directory and return a summary of the latest
    checkpoint for each unique BR model, including metadata from the
    checkpoint data dict.

    Returns a list of dicts sorted by prefix, each containing:
        - prefix, steps, exploit_side, filename, path (from filename parsing)
        - arch, unique_states, state_list, training_br, etc. (from checkpoint data)
    """
    latest = scan_br_directory(br_dir)
    results = []
    for prefix in sorted(latest.keys()):
        entry = latest[prefix]
        try:
            meta = inspect_checkpoint(entry["path"])
        except Exception as e:
            meta = {"error": str(e)}
        if meta.get("exploit_side") and entry.get("exploit_side") is None:
            entry["exploit_side"] = meta["exploit_side"]
        elif entry.get("exploit_side") and not meta.get("exploit_side"):
            meta["exploit_side"] = entry["exploit_side"]
        entry["meta"] = meta
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Scan BR checkpoint directory, report latest checkpoints, "
                    "and optionally continue training."
    )
    parser.add_argument(
        "--br_dir", type=str, required=True,
        help="Path to directory containing BR checkpoint zip files.",
    )
    parser.add_argument(
        "--game_args_json", type=str, default=None,
        help="JSON string or path to JSON file with game_args dict. "
             "Required when --continue_training is set.",
    )
    parser.add_argument(
        "--continue_training", action="store_true",
        help="If set, continue training from each latest checkpoint.",
    )
    parser.add_argument(
        "--filter_prefix", type=str, default=None,
        help="Only process BR models whose prefix contains this substring.",
    )
    parser.add_argument(
        "--n_envs", type=int, default=2,
        help="Number of environments per matchup.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for training.",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=10_000_000,
        help="Total timesteps for continued training.",
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable wandb logging during continued training.",
    )
    parser.add_argument(
        "--array_index", type=int, default=None,
        help="If set, train only the model at this index in the sorted/filtered "
             "list (0-based). Used by SLURM array jobs. Exits cleanly if the "
             "index is out of range.",
    )
    parser.add_argument(
        "--count_only", action="store_true",
        help="Print the number of trainable models and exit. "
             "Useful for setting SLURM --array upper bound.",
    )
    args = parser.parse_args()

    print(f"Scanning BR checkpoint directory: {args.br_dir}")
    results = summarize_br_models(args.br_dir)
    print(f"\nFound {len(results)} unique BR models.\n")

    if not args.count_only:
        for r in results:
            meta = r.get("meta", {})
            err = meta.get("error")
            if err:
                print(f"  {r['prefix']}")
                print(f"    latest step: {r['steps']}")
                print(f"    ERROR loading metadata: {err}\n")
                continue

            learning_side = "adversary" if r["exploit_side"] == "ego" else "ego"
            print(f"  {r['prefix']}")
            print(f"    latest step:    {r['steps']}")
            print(f"    arch:           {meta['arch']}")
            print(f"    exploit_side:   {r['exploit_side']} (learning: {learning_side})")
            print(f"    unique_states:  {meta['unique_states']}")
            print(f"    base_ckpt:      {meta['checkpoint_basename']}")
            print(f"    matchups:       {meta['matchups']}")
            print()

    if not args.continue_training:
        print("Pass --continue_training to resume training from these checkpoints.")
        return

    game_args = None
    if args.game_args_json is not None:
        if os.path.isfile(args.game_args_json):
            with open(args.game_args_json, "r") as f:
                game_args = json.load(f)
        else:
            game_args = json.loads(args.game_args_json)
    else:
        print("No --game_args_json provided; will extract game_args from each checkpoint.\n")

    filtered = results
    if args.filter_prefix:
        filtered = [r for r in results if args.filter_prefix in r["prefix"]]
        print(f"Filtered to {len(filtered)} models matching '{args.filter_prefix}'.\n")

    trainable = [
        r for r in filtered
        if not r.get("meta", {}).get("error") and r.get("exploit_side") is not None
    ]
    print(f"Trainable models: {len(trainable)}")

    if args.count_only:
        return

    if args.array_index is not None:
        if args.array_index >= len(trainable):
            print(
                f"Array index {args.array_index} >= {len(trainable)} trainable models. "
                f"Nothing to do.",
            )
            return
        trainable = [trainable[args.array_index]]
        print(f"Array index {args.array_index}: {trainable[0]['prefix']}\n")

    for r in trainable:
        meta = r.get("meta", {})
        load_and_continue(
            checkpoint_path=r["path"],
            game_args=game_args,
            exploit_side=r["exploit_side"],
            arch=meta["arch"],
            unique_states=meta["unique_states"],
            n_envs=args.n_envs,
            device=args.device,
            total_timesteps=args.total_timesteps,
            use_wandb=args.use_wandb,
        )


if __name__ == "__main__":
    main()
