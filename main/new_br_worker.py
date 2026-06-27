"""
BR worker. Parse --device before importing torch so ``--device cpu`` can hide GPUs from PyTorch.
"""
import os
import re
import sys
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from copy import deepcopy
import socket
socket.setdefaulttimeout(None)
def _peek_torch_device_argv(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


_peeked_dev = _peek_torch_device_argv(sys.argv[1:])
if _peeked_dev is not None and str(_peeked_dev).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _ensure_cpu_only_env(device: str) -> None:
    """Multiprocessing ``spawn`` may not replay argv peek; call with the same ``device`` as training."""
    if str(device).lower().startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

from common.algorithms import Exploiter, LeaguePPO
import argparse
import time
import json
import random
from pprint import pformat
import numpy as np
import torch
import wandb
import shutil
import subprocess
import multiprocessing as mp
from stable_baselines3.common.preprocessing import is_image_space
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.justin.clean_derivative_free_spar_ippo import CleanDerivativeFreeSPARIPPO
from stable_baselines3.common.save_util import load_from_zip_file
from ippo import env_generator
from train_ma import constructor as league_constructor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ExploiterCheckpointCallback
from gymnasium.spaces import Box
from utils import state2matchup
from br_preflight import (
    build_dedicated_job_specs,
    dedupe_preserve_order,
    extract_unique_states_from_checkpoint_data,
    infer_cds_architecture,
    sanitize_for_filename,
)
# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
_workdir = os.environ.get("WORKDIR")
_main_training_dir = os.environ.get("MAIN_TRAINING_DIR")
if _workdir and _main_training_dir:
    _base_dir = os.path.join(_workdir, _main_training_dir, "FightLadder", "main")
else:
    _base_dir = current_dir
TASK_DIR = os.path.join(_base_dir, "trained_models/tasks")
DONE_DIR = os.path.join(_base_dir, "trained_models/tasks/done")
BR_MODEL_DIR = os.path.join(_base_dir, "trained_models/tasks/br_models")
#WR_STATS_DIR = os.path.join(current_dir, "main/trained_models/wr_stats")
#MEAN_REW_STATS_DIR = os.path.join(current_dir, "main/trained_models/mean_rew_stats")
os.makedirs(BR_MODEL_DIR, exist_ok=True)
os.makedirs(TASK_DIR, exist_ok=True)
#os.makedirs(PROCESSING_DIR, exist_ok=True)
#os.makedirs(DONE_DIR, exist_ok=True)
#os.makedirs(WR_STATS_DIR, exist_ok=True)
#os.makedirs(MEAN_REW_STATS_DIR, exist_ok=True)
#os.makedirs(ERROR_DIR, exist_ok=True)

if not os.listdir(TASK_DIR):
    print("Warning: The TASK_DIR is empty. Please run ippo.py --player PLAYER to generate a task file.")

POLL_INTERVAL = 5  # Seconds to wait before checking for new tasks
# BR_TRAINING_STEPS is now plumbed end-to-end from the launcher via
# --br_training_steps (orchestrator argparse) -> shared_config_json
# -> br_single_*.py -> run_br_for_task_in_subprocess(br_training_steps=...)
# -> _run_br_for_task. The module-level constant was deleted so any
# unplumbed code path fails loudly instead of silently using 10M.


def _reap_finished(active: List[mp.Process]) -> List[mp.Process]:
    """Join finished processes and return those still running."""
    still_active = []
    for p in active:
        if p.is_alive():
            still_active.append(p)
        else:
            p.join()
    return still_active


def _wait_for_slot(active: List[mp.Process], max_concurrent: int) -> List[mp.Process]:
    """Block until at least one slot is free, then return the updated active list."""
    while len(active) >= max_concurrent:
        active = _reap_finished(active)
        if len(active) >= max_concurrent:
            time.sleep(2)
    return active


class ManualStopFileCallback(BaseCallback):
    """
    Graceful manual-stop callback.

    When stop_file appears, return False from _on_step so learn() exits cleanly,
    allowing post-training logic (e.g., local_br_eval launch) to still run.
    """

    def __init__(self, stop_file: str, stop_key: str, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.stop_file = str(stop_file)
        self.stop_key = str(stop_key)
        self._already_triggered = False
        self._terminated_stop_file = None

    def _prepend_terminated(self, path: Optional[str]) -> Optional[str]:
        if path is None or str(path) == "":
            return path
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        if basename.startswith("TERMINATED_"):
            return path
        return os.path.join(dirname, f"TERMINATED_{basename}")

    def _rename_if_exists(self, src: Optional[str]) -> Optional[str]:
        if src is None or str(src) == "":
            return src
        dst = self._prepend_terminated(src)
        if dst is None:
            return src
        if src == dst:
            return src
        if os.path.exists(src):
            os.rename(src, dst)
            return dst
        return src

    def _get_tracker(self):
        # Dedicated exploiter path.
        tracker = getattr(self.model, "br_convergence_tracker", None)
        if tracker is not None:
            return tracker
        # Continue exploiter path.
        return getattr(self.model, "stagnation_tracker", None)

    def _mark_tracker_outputs_terminated(self) -> None:
        tracker = self._get_tracker()
        if tracker is None:
            return
        tracker.local_plot_prefix = (
            tracker.local_plot_prefix
            if str(getattr(tracker, "local_plot_prefix", "")).startswith("TERMINATED_")
            else f"TERMINATED_{getattr(tracker, 'local_plot_prefix', self.stop_key)}"
        )
        tracker.local_entropy_csv_path = self._rename_if_exists(
            getattr(tracker, "local_entropy_csv_path", None)
        )
        tracker.local_entropy_png_path = self._rename_if_exists(
            getattr(tracker, "local_entropy_png_path", None)
        )
        tracker.local_reward_csv_path = self._rename_if_exists(
            getattr(tracker, "local_reward_csv_path", None)
        )
        tracker.local_reward_png_path = self._rename_if_exists(
            getattr(tracker, "local_reward_png_path", None)
        )

    def _on_step(self) -> bool:
        if os.path.exists(self.stop_file):
            if not self._already_triggered:
                terminated_stop_file = self._rename_if_exists(self.stop_file)
                self._terminated_stop_file = terminated_stop_file
                self._mark_tracker_outputs_terminated()
                print(
                    f"MANUAL STOP [{self.stop_key}] detected at {self.stop_file}. "
                    f"Renamed marker to {self._terminated_stop_file}. "
                    "Ending current learn() call gracefully.",
                    flush=True,
                )
                self._already_triggered = True
            return False
        return True


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    """Backward-compatible wrapper around br_preflight helper."""
    return dedupe_preserve_order(values)


def _sanitize_for_filename(value: str) -> str:
    """Backward-compatible wrapper around br_preflight helper."""
    return sanitize_for_filename(value)


# SPAR .task filenames follow `<run_prefix>_<step>_steps.task` (see the
# timestep parsing at line ~341 in load_spar_model). The run prefix is
# stable across all checkpoints from the same training run, so it serves as
# the natural per-run identifier for output segregation.
_SPAR_RUN_PREFIX_RE = re.compile(r"^(?P<run>.+)_\d+_steps\.task$")


def _derive_spar_run_subdir(task_filename: str) -> str:
    """
    Extract the per-run identifier from a SPAR .task filename.

    Returns the run prefix (everything before `_<step>_steps.task`) so that
    different ippo training runs land in different br_rewards / selfplay_rewards
    subfolders. Falls back to the filename stem if the suffix doesn't match
    (defensive — caller should still get *some* segregation rather than
    silently sharing an unsegregated bucket).
    """
    m = _SPAR_RUN_PREFIX_RE.match(task_filename)
    if m is not None:
        return _sanitize_for_filename(m.group("run"))
    stem, _ = os.path.splitext(task_filename)
    return _sanitize_for_filename(stem) or "unknown_run"


def _extract_unique_states_from_task(task_file_path: str, device: str = "cuda") -> List[str]:
    """
    Read checkpoint/task metadata and return unique state strings.

    Why this helper exists:
    - Dedicated exploiter mode now schedules one BR job per unique matchup state.
    - We need to inspect the task checkpoint *before* launching subprocesses.
    - This helper centralizes metadata parsing so scheduling logic stays readable.
    """
    data, _, _ = load_from_zip_file(task_file_path, device=device)
    return extract_unique_states_from_checkpoint_data(data=data, task_file_path=task_file_path)


def _infer_cds_architecture(data: Dict[str, Any], task_file_path: str) -> str:
    """
    Infer checkpoint architecture family for BR loading.

    Returns:
        "ippo" if checkpoint appears to use IPPO CDS policy/value heads
        "spar" otherwise (default/fallback)
    """
    return infer_cds_architecture(data=data, task_file_path=task_file_path)


def _build_dedicated_job_specs(
    unique_states: List[str],
    replicates_per_matchup: int,
    run_eval_prot: bool,
    run_eval_adv: bool,
    launch_local_br_eval: bool,
) -> List[Dict[str, Any]]:
    """
    Build dedicated BR job specs:
      - one entry per (state, side, replicate)
      - side is controlled by eval_prot/eval_adv flags

    Notes:
    - "replicates_per_matchup" means if set to K, each matchup+side gets K
      independent subprocess jobs (different BR indices/checkpoints).
    - We keep a monotonic "job_index" for easy logging/debugging.
    """
    return build_dedicated_job_specs(
        unique_states=unique_states,
        replicates_per_matchup=replicates_per_matchup,
        run_eval_prot=run_eval_prot,
        run_eval_adv=run_eval_adv,
        launch_local_br_eval=launch_local_br_eval,
        state_to_matchup=state2matchup,
    )


class _FixedMatchupPolicyAdapter:
    """
    Lightweight policy adapter that forces CDS forward passes to a single matchup head.

    Why this exists:
    - `CleanDerivativeFreeSPAR` is architected around a full matchup topology.
    - In dedicated BR runs we intentionally train on one matchup subset.
    - We do NOT want to modify CDS internals, so we adapt behavior at the worker layer.

    Behavior:
    - Delegates all unknown attributes to the wrapped policy.
    - Intercepts `__call__` and injects `network_keys=[fixed_idx]*batch_size`.
    - Preserves `.to(device)` semantics so existing device-transfer code keeps working.
    """

    def __init__(self, base_policy, fixed_matchup_idx: int):
        self._base_policy = base_policy
        self._fixed_matchup_idx = int(fixed_matchup_idx)

    def __call__(self, obs_tensor, *args, **kwargs):
        # Always route every sample in this mini-batch to the same matchup head.
        # This enforces "single-matchup dedicated evaluation/training" semantics
        # even when the original CDS model contains multiple matchup heads.
        batch_size = int(obs_tensor.shape[0])
        #kwargs = dict(kwargs)
        #kwargs["network_keys"] = [self._fixed_matchup_idx] * batch_size

        ego_forward = kwargs['ego_forward']
        adv_forward = kwargs['adv_forward']

        if ego_forward is True and adv_forward is True:
            raise ValueError("cannot have both ego and adv forward -- we can only exploit one at a time")

        if ego_forward:
            side_flag = kwargs.get('ego_side_flag', None)
            exploited_actions, exploited_log_probs = self._base_policy.ego_forward(obs_tensor, side_flag=side_flag)
        elif adv_forward:
            side_flag = kwargs.get('adv_side_flag', None)
            exploited_actions, exploited_log_probs = self._base_policy.adv_forward(obs_tensor, buf_num=[self._fixed_matchup_idx], side_flag=side_flag)
        else:
            raise ValueError(f"Invalid forward flag: {ego_forward} or {adv_forward}")

        return exploited_actions, exploited_log_probs

    def to(self, device):
        # Keep adapter identity stable while moving underlying policy modules.
        self._base_policy = self._base_policy.to(device)
        return self

    def __getattr__(self, name):
        # Delegate all other attributes/methods transparently.
        return getattr(self._base_policy, name)


def _is_ego_left_for_state(loaded_model, dedicated_state: str) -> bool:
    """In mirror-mode models, the first half of unique states have ego on the left (P1)
    and the second half have ego on the right (P2). Returns True if ego is P1/left."""
    unique_states = getattr(loaded_model, "_worker_unique_states", None)
    if not isinstance(unique_states, list) or len(unique_states) == 0:
        return True
    halfway = len(unique_states) // 2
    try:
        idx = unique_states.index(dedicated_state)
    except ValueError:
        return True
    return idx < halfway


def _resolve_matchup_index_for_state(loaded_model, dedicated_state: str) -> int:
    """
    Map a dedicated state string to a CDS matchup-head index.

    We prefer an exact state-index lookup using the checkpoint's unique state order,
    then fall back to matchup-label lookup if needed.
    """
    unique_states = getattr(loaded_model, "_worker_unique_states", None)
    if isinstance(unique_states, list) and dedicated_state in unique_states:
        return unique_states.index(dedicated_state)

    target_matchup = state2matchup(dedicated_state)
    envs_per_matchup = int(max(1, getattr(loaded_model, "envs_per_matchup", 1)))
    matchups = getattr(loaded_model, "matchups", None) or []
    for i in range(0, len(matchups), envs_per_matchup):
        if matchups[i] == target_matchup:
            return i // envs_per_matchup
    raise ValueError(
        f"Could not resolve matchup index for dedicated state '{dedicated_state}'."
    )


def load_spar_model(
    game_args: dict,
    task_file_path: str,
    n_envs: int = 2,
    device: str = "cuda",
    state_subset: Optional[List[str]] = None,
    use_wandb: bool = True,
):
    worker_id = os.getpid()
    print(f"WORKER [{worker_id}]: Processing task: {os.path.basename(task_file_path)}")

    #try:
    # The task file IS the model checkpoint file, just renamed.
    checkpoint_path = task_file_path

    # Extract timestep from the checkpoint filename for wandb logging
    try:
        basename = os.path.basename(checkpoint_path)
        # Assumes format like '..._12345_steps.task'
        timestep_str = basename.replace('.task', '').split('_')[-2]
        ego_timestep = int(timestep_str)
    except (IndexError, ValueError):
        print(f"WORKER [{worker_id}]: Could not parse timestep from filename: {basename}. BR win rate will not be logged against a specific step.")
        ego_timestep = None

    
    # Default load_from_zip_file(..., device="auto") maps to CUDA and allocates GPU weights
    # even though we only need ``data`` here; always respect the worker device.
    data, _, _ = load_from_zip_file(checkpoint_path, device=device)
    cds_arch = _infer_cds_architecture(data, checkpoint_path)
    cds_cls = CleanDerivativeFreeSPARIPPO if cds_arch == "ippo" else CleanDerivativeFreeSPAR
    print(
        f"WORKER [{worker_id}]: Detected checkpoint architecture={cds_arch} "
        f"(loader={cds_cls.__name__})"
    )
    # IMPORTANT:
    # Keep CDS model topology in its original full-matchup form. Even in
    # dedicated runs we should NOT shrink the loaded CDS state/matchup metadata,
    # because internal forward/update logic assumes the full topology.
    #
    # Dedicated isolation is handled later by:
    #   1) dedicated BR env construction (subset env for the exploiter), and
    #   2) `_FixedMatchupPolicyAdapter` (pins exploited CDS head selection).
    uniques = _dedupe_preserve_order(data["state_list"])
    # Build the concrete STATE list passed to env/model, repeating each unique
    # state n_envs times to match existing worker conventions.
    STATE = [state for state in uniques for _ in range(n_envs)]
    game_args = argparse.Namespace(**game_args)
    

    # argument to STATE should be the master list of states, including repeats for repeated matchups.
    # we should always follow this convention -- when there is a discrepancy
    # we should default to STATE.

    


    env = env_generator(game_args, STATE=STATE)
    #env.num_envs = 1 # HACKY FOR NOW!
    try:
        ftm = cds_cls.load(
            path=checkpoint_path, env=env, game_args=game_args, num_perturbed=1, device=device
        )
        #if ftm.policy.num_env_per_adv is None:
        #    ftm.policy.num_env_per_adv = ftm.envs_per_matchup
    except Exception as e:
        data, params, pytorch_variables = load_from_zip_file(
            checkpoint_path, device=device)
        ftm = cds_cls(
            "AACCnnPolicy",
            env,
            device=device,
            verbose=2,
            n_steps=256,
            batch_size=512,
            n_epochs=1,
            state_list=STATE,
            envs_per_matchup=1,
            env_generator_func=env_generator,
            num_adversaries=1,
            n_env_per_adv=1,
            seed= 0,
            target_kl=0.025,
            use_mirror=False,
            use_wandb=use_wandb,
        )
        ftm.set_parameters(params, exact_match=True, device=ftm.device)
    # Keep a stable copy of the unique checkpoint state ordering for dedicated
    # matchup-index resolution. This is worker metadata only (no CDS changes).
    ftm._worker_cds_arch = cds_arch
    ftm._worker_unique_states = list(uniques)
    ftm._worker_full_state_list = list(STATE)
    # env was only needed to reconstruct the model — close its 2N emulator
    # subprocesses before the caller creates the real training env.
    env.close()
    ftm.env = None
    return ftm


class _LeaguePolicyAdapter:
    """
    Adapter that makes a LeaguePPO model conform to the CDS-style
    ``self.exploited.policy(obs, ego_forward=..., adv_forward=...)`` interface
    expected by ``Exploiter.collect_rollouts``.

    LeaguePPO stores two standard SB3 policies: ``policy`` (left) and
    ``policy_other`` (right). This adapter selects the appropriate one
    based on ``side`` and ignores the CDS-specific keyword arguments.
    """

    def __init__(self, league_model, side="left"):
        # Capture the underlying policies *before* the caller assigns
        # ``league_model.policy = adapter``. Once that assignment happens,
        # ``league_model.policy`` is the adapter itself, so any later read of
        # ``self._league_model.policy`` would resolve back to ``self`` and
        # cause infinite recursion in __call__/to/__getattr__.
        self._league_model = league_model
        self._side = side
        self._left_policy = league_model.policy
        self._right_policy = getattr(league_model, "policy_other", None)

    def __call__(self, obs_tensor, *args, **kwargs):
        if self._side == "left":
            policy = self._left_policy
        else:
            policy = self._right_policy
        actions, _, _ = policy(obs_tensor)
        return actions, None, None

    def to(self, device):
        self._left_policy = self._left_policy.to(device)
        if self._right_policy is not None:
            self._right_policy = self._right_policy.to(device)
        return self

    def __getattr__(self, name):
        # Delegate to the captured left policy. Reading
        # ``self._league_model.policy`` here would recurse because that
        # attribute now points to this adapter.
        return getattr(self._left_policy, name)


# Sibling-snapshot regexes accept both `.pt` (live training output) and
# `.task` (renamed copies queued for the BR worker). The two extensions are
# byte-identical torch.save dumps; only the suffix differs based on whether
# the snapshot is at-rest or in the queue.
_LEAGUE_RIGHT_RE = re.compile(
    r"^MA\d+_right_m_\d+_(?P<left_char>[a-z0-9]+)_vs_(?P<right_char>[a-z0-9]+)_\d+_\d{8}_\d{6}\.(?:pt|task)$"
)
_BACKUP_LEAGUE_RIGHT_RE = re.compile(
    r"^LE\d+_right_m_\d+_(?P<left_char>[a-z0-9]+)_vs_(?P<right_char>[a-z0-9]+)_\d+_\d{8}_\d{6}\.(?:pt|task)$"
)

def _infer_league_matchup_states_from_dir(task_file_path: str) -> List[str]:
    """
    Scan sibling MA*_right* .pt files to infer matchup states for a MA*_left task.

    Right-model filenames follow the pattern produced by ``league._agent_name``:
        ``MA0_right_m_00_<right_char>_vs_<left_char>_<step>.pt``

    We extract character pairs, build retro state strings in the same format
    used by ``train_ma._build_states_from_roster``, and return unique states
    in matchup-index order.
    """
    model_dir = os.path.dirname(task_file_path)
    seen = set()
    states: List[str] = []
    for fname in sorted(os.listdir(model_dir)):
        m = _LEAGUE_RIGHT_RE.match(fname) or _BACKUP_LEAGUE_RIGHT_RE.match(fname)
        if m is None:
            continue
        right_char = m.group("right_char")
        left_char = m.group("left_char")
        key = (left_char, right_char)
        if key in seen:
            continue
        seen.add(key)
        left_title = left_char.capitalize()
        right_title = right_char.capitalize()
        state = (
            f"two_player/{left_title}_left/"
            f"Champion.Level1.{left_title}Vs{right_title}.2Player.state"
        )
        states.append(state)

    if not states:
        raise FileNotFoundError(
            f"Could not infer league matchup states: no MA*_right*.pt files "
            f"found in {model_dir}"
        )
    return states


def _load_league_checkpoint(path: str, device: str):
    """
    ``torch.load`` a league checkpoint while resolving the pickled
    ``constructor`` function reference.

    League checkpoints are saved by ``train_ma`` (running as ``__main__``), so
    the pickled ``constructor`` callable is bound to ``__main__.constructor``.
    When we load from a different entry-point the module name won't match.
    We temporarily inject the symbol into both ``__main__`` and ``__mp_main__``
    so ``torch.load``'s unpickler can resolve it.
    """
    import __main__
    patched_modules = [__main__]
    mp_main = sys.modules.get("__mp_main__")
    if mp_main is not None:
        patched_modules.append(mp_main)

    originals = [(mod, getattr(mod, "constructor", None)) for mod in patched_modules]
    for mod in patched_modules:
        mod.constructor = league_constructor
    try:
        return torch.load(path, map_location=device)
    finally:
        for mod, orig in originals:
            if orig is None:
                if hasattr(mod, "constructor"):
                    delattr(mod, "constructor")
            else:
                mod.constructor = orig


def load_league_model(
    game_args: dict,
    task_file_path: str,
    league_matchup_states: List[str],
    n_envs: int = 2,
    device: str = "cuda",
    use_wandb: bool = True,
):
    """
    Load a league (LeaguePPO) checkpoint saved by ``Payoff.add_player``.

    The .task file is a renamed .pt produced by ``torch.save({"cls_name": ..., "kwargs": ...})``.
    We reconstruct the model via ``train_ma.constructor`` (the same function used
    during league training) so that the env wrappers, hyperparameters, and policy
    architecture exactly match the original training run.  Weights are restored
    with ``set_parameters(agent_dict)``.

    Worker-metadata attributes are attached so downstream scheduling
    (``_build_dedicated_job_specs``, etc.) works unchanged.
    """
    worker_id = os.getpid()
    print(f"WORKER [{worker_id}]: Processing league task: {os.path.basename(task_file_path)}")

    saved = _load_league_checkpoint(task_file_path, device)
    saved_kwargs = saved["kwargs"]
    agent_dict = saved_kwargs["agent_dict"]

    # Capture the original side from the checkpoint *before* we flip it. The
    # flipped value drives league_constructor below (training-side
    # convention), but downstream BR scheduling needs to know which side the
    # checkpoint actually came from so it can pick the right frozen-side
    # adapter / continue-mode helper.
    loaded_side = saved_kwargs.get("side")
    if loaded_side == "left":
        side = "right"
    elif loaded_side == "right":
        side = "left"
    else:
        # Strict mapping: any other value (None, "both", typos, future enum
        # additions) is a bug or schema drift. Fail loudly instead of
        # silently coercing to one side.
        raise ValueError(
            f"Saved league checkpoint has invalid side={loaded_side!r}; "
            "expected 'left' or 'right'."
        )

    # experimental
    saved_args = saved_kwargs.get("args")

    constructor_args = argparse.Namespace(**game_args)
    if saved_args is not None:
        sa = saved_args if isinstance(saved_args, dict) else vars(saved_args)
        for key in ("num_env", "log_dir"):
            if key not in vars(constructor_args) and key in sa:
                setattr(constructor_args, key, sa[key])
    if not hasattr(constructor_args, "num_env"):
        constructor_args.num_env = n_envs
    if not hasattr(constructor_args, "log_dir"):
        constructor_args.log_dir = "logs/ma"

    first_state = league_matchup_states[0]
    league_model = league_constructor(
        constructor_args,
        side,
        log_name=None,
        single_env=False,
        state_name=first_state,
        matchup_key="left_vs_all",
    )
    league_model.set_parameters(agent_dict)

    checkpoint_step = saved_kwargs.get("checkpoint_step", 0)
    if checkpoint_step:
        league_model.set_steps(checkpoint_step)

    league_model._worker_cds_arch = "league"
    league_model._worker_unique_states = list(league_matchup_states)
    league_model._worker_full_state_list = list(league_matchup_states)
    # Original (pre-flip) side and matchup_key from the saved checkpoint.
    # Continue-mode (`_run_league_continue_exploiter`) reads these to decide
    # which side to freeze and which matchup's right model to override with
    # the loaded `agent_dict`.
    league_model._worker_loaded_side = loaded_side
    league_model._worker_loaded_matchup_key = saved_kwargs.get("matchup_key")
    league_model.state_list = list(league_matchup_states)
    league_model.matchups = [state2matchup(s) for s in league_matchup_states]
    league_model.envs_per_matchup = 1
    league_model.use_wandb = use_wandb

    print(
        f"WORKER [{worker_id}]: League model loaded: side={side}, "
        f"matchup_states={league_matchup_states}"
    )
    return league_model


def _continue_train_loaded_league_model(
    ftm: LeaguePPO,
    eval_prot: bool,
    total_timesteps: int,
    callback,
    rollout_opponent_num: int,
) -> None:
    """
    Continue training a loaded league checkpoint using the same LeaguePPO learner path
    as train_ma/league (side-aware rollouts + one-side policy updates).

    Side update rule:
    - eval_prot=True  -> train right-side policy (freeze left)
    - eval_prot=False -> train left-side policy (freeze right)
    """
    from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
    from gymnasium import spaces

    train_side = "right" if eval_prot else "left"
    ftm.side = train_side

    # Sync n_envs and rollout buffers with the (possibly different) env that was
    # assigned to ftm before this call.  load_league_model builds the model with
    # num_env from the constructor, but train_best_response may swap in a new env
    # with a different n_envs.  Without this, the buffer shapes won't match and
    # numpy will raise a reshape error.
    actual_n_envs = ftm.env.num_envs
    if ftm.n_envs != actual_n_envs:
        ftm.n_envs = actual_n_envs
        buffer_cls = DictRolloutBuffer if isinstance(ftm.observation_space, spaces.Dict) else RolloutBuffer
        ftm.rollout_buffer = buffer_cls(
            ftm.n_steps,
            ftm.observation_space,
            ftm.action_space,
            device=ftm.device,
            gamma=ftm.gamma,
            gae_lambda=ftm.gae_lambda,
            n_envs=actual_n_envs,
        )
        ftm.rollout_buffer_other = buffer_cls(
            ftm.n_steps,
            ftm.observation_space,
            ftm.action_space,
            device=ftm.device,
            gamma=ftm.gamma,
            gae_lambda=ftm.gae_lambda,
            n_envs=actual_n_envs,
        )

    # Freeze the opposite-side policy by supplying a detached copy as rollout
    # opponent policy. LeaguePPO.train() only updates the side selected by ftm.side.
    frozen_left_policy = deepcopy(ftm.policy)
    frozen_right_policy = deepcopy(ftm.policy_other)
    frozen_left_policy.set_training_mode(False)
    frozen_right_policy.set_training_mode(False)

    def _get_kwargs():
        kwargs: Dict[str, Any] = {
            "coordinate_fn": lambda _outcome: None,
            "sync_fn": lambda: None,
        }
        if train_side == "right":
            kwargs["policy"] = frozen_left_policy
        else:
            kwargs["policy_other"] = frozen_right_policy

        return kwargs

    ftm.learn(
        total_timesteps=total_timesteps,
        rollout_opponent_num=max(1, int(rollout_opponent_num)),
        callback=callback,
        log_interval=1,
        tb_log_name="Learner",
        reset_num_timesteps=False,
        progress_bar=False,
        get_kwargs_fn=_get_kwargs,
    )


def _league_worker(idx, learner, total_steps, rollout_opponent_num):
    """Worker target for league continue-exploiter subprocesses."""
    import torch as _torch
    print(f"league_worker {learner.player.name} start (pid={os.getpid()})")
    device_count = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
    with _torch.cuda.device(idx % device_count):
        learner.player.construct_agent()
        learner.run(
            total_timesteps=total_steps,
            rollout_opponent_num=rollout_opponent_num,
        )


_LEAGUE_RIGHT_ANY_RE = re.compile(
    r"^(?P<role>[A-Z]+\d+)_right_m_(?P<midx>\d+)_(?P<left_char>[a-z0-9]+)_vs_(?P<right_char>[a-z0-9]+)_(?P<step>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.(?:pt|task)$"
)

# Left-side league checkpoints. The league only ever has one left model
# (matchup_key = "m_00_left_vs_all" — see common.league._matchup_key /
# League.__init__), so this regex is intentionally narrower than the right-side
# one: a fixed "left_vs_all" matchup, no per-character pair.
_LEAGUE_LEFT_ANY_RE = re.compile(
    r"^(?P<role>[A-Z]+\d+)_left_m_(?P<midx>\d+)_left_vs_all_(?P<step>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.(?:pt|task)$"
)


def _load_right_side_checkpoints(
    model_dir: str,
    league_matchup_states: List[str],
    device: str = "cpu",
) -> Dict[str, dict]:
    """
    Scan *model_dir* for right-side .pt files and return the highest-step
    ``agent_dict`` for each matchup key (e.g. ``ryu_vs_guile``).

    Only MA checkpoints are considered (not ME/LE/historical) because the
    League will create its own ME/LE players initialized from the MA weights.
    """
    from train_ma import _sanitize_matchup_token

    best: Dict[str, tuple] = {}
    for fname in os.listdir(model_dir):
        m = _LEAGUE_RIGHT_ANY_RE.match(fname)
        if m is None:
            continue
        role = m.group("role")
        if not role.startswith("MA"):
            continue
        left_char = m.group("left_char")
        right_char = m.group("right_char")
        step = int(m.group("step"))
        key = f"{_sanitize_matchup_token(left_char)}_vs_{_sanitize_matchup_token(right_char)}"
        if key not in best or step > best[key][0]:
            best[key] = (step, os.path.join(model_dir, fname))

    result: Dict[str, dict] = {}
    for key, (step, path) in best.items():
        saved = _load_league_checkpoint(path, device)
        agent_dict = saved["kwargs"]["agent_dict"]
        checkpoint_step = saved["kwargs"].get("checkpoint_step", step)
        result[key] = {"agent_dict": agent_dict, "checkpoint_step": checkpoint_step}
        print(f"[league_continue] Loaded right checkpoint: {key} step={checkpoint_step} from {os.path.basename(path)}")

    return result


def _load_left_side_checkpoint(
    model_dir: str,
    device: str = "cpu",
) -> Optional[dict]:
    """
    Scan *model_dir* for the highest-step MA left-side .pt and return its
    ``agent_dict`` + ``checkpoint_step``.

    The league only contains a single left-side MA agent (matchup_key
    ``m_00_left_vs_all``), so unlike the right-side scanner this returns a
    single dict (or ``None`` if no MA left file exists).

    Used by ``_run_league_continue_exploiter`` when the loaded task is a
    right-side checkpoint and we need the matching left protagonist to
    continue-train against frozen right.
    """
    best: Optional[tuple] = None  # (step, path)
    for fname in os.listdir(model_dir):
        m = _LEAGUE_LEFT_ANY_RE.match(fname)
        if m is None:
            continue
        # Mirror the right-side helper: only consider main-agent (MA) snapshots,
        # not exploiter snapshots (ME/LE) — those have different training
        # dynamics and aren't the canonical "current protagonist".
        if not m.group("role").startswith("MA"):
            continue
        step = int(m.group("step"))
        if best is None or step > best[0]:
            best = (step, os.path.join(model_dir, fname))

    if best is None:
        return None

    step, path = best
    saved = _load_league_checkpoint(path, device)
    agent_dict = saved["kwargs"]["agent_dict"]
    checkpoint_step = saved["kwargs"].get("checkpoint_step", step)
    print(
        f"[league_continue] Loaded left checkpoint: step={checkpoint_step} "
        f"from {os.path.basename(path)}"
    )
    return {"agent_dict": agent_dict, "checkpoint_step": checkpoint_step}


def _run_league_continue_exploiter(
    constructor_args: argparse.Namespace,
    agent_dict: dict,
    league_matchup_states: List[str],
    total_timesteps: int,
    rollout_opponent_num: int,
    model_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    log_dir: str = "logs/br_league",
    frozen_side: str = "left",
    loaded_matchup_key: Optional[str] = None,
) -> None:
    """
    Run a one-side-only continue league against a frozen opposing side.

    The function is symmetric in *frozen_side*:

    - ``frozen_side="left"`` (loaded task was a left MA): build the full
      right-side roster (MA/ME/LE per matchup) and train them. The left MA
      acts as a static opponent. Right-side agent weights are loaded from
      sibling ``MA*_right_*.pt`` snapshots in *model_dir*; matchups without a
      sibling fall back to ``agent_dict`` (the loaded left MA's weights).

    - ``frozen_side="right"`` (loaded task was a right MA for one matchup):
      build the full left-side roster (one MA + ME + LE) and train them.
      Right models stay frozen. The right roster is loaded from sibling
      ``MA*_right_*.pt`` snapshots, *except* the entry for *loaded_matchup_key*
      which is overridden with ``agent_dict`` (the freshest snapshot for that
      matchup is the loaded task itself). The left MA is loaded from the
      sibling ``MA*_left_*.pt`` snapshot via ``_load_left_side_checkpoint``;
      if no sibling exists we fall back to ``agent_dict`` (which is
      right-side weights — wrong shape, will likely crash, so we surface a
      clear error instead).

    Args:
        agent_dict: Weights for the loaded checkpoint (left MA when
            frozen_side="left", right MA for *loaded_matchup_key* when
            frozen_side="right").
        loaded_matchup_key: Canonical matchup key (e.g. ``ryu_vs_guile``) the
            loaded task corresponds to. Required when frozen_side="right" so
            we know which right-roster slot to override.
    """
    from train_ma import _extract_chars_from_state_name, _sanitize_matchup_token
    from common.league import PayoffManager, League, Learner

    if frozen_side not in ("left", "right"):
        raise ValueError(
            f"_run_league_continue_exploiter: frozen_side={frozen_side!r}, "
            "expected 'left' or 'right'."
        )

    if save_dir is None:
        save_dir = model_dir if model_dir is not None else "trained_models/br_league"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    constructor_args = deepcopy(constructor_args)
    constructor_args.log_dir = log_dir
    constructor_args.save_dir = save_dir

    # --- Resolve right-side roster weights ---
    right_checkpoints: Dict[str, dict] = {}
    if model_dir is not None:
        right_checkpoints = _load_right_side_checkpoints(model_dir, league_matchup_states)

    # When the loaded task IS a right-side checkpoint, treat it as the
    # freshest snapshot for its matchup and override the file-scan result.
    if frozen_side == "right":
        if loaded_matchup_key is None:
            raise ValueError(
                "_run_league_continue_exploiter: frozen_side='right' requires "
                "loaded_matchup_key (so we know which right-roster slot to "
                "override with the loaded agent_dict)."
            )
        right_checkpoints[loaded_matchup_key] = {
            "agent_dict": agent_dict,
            # Step is intentionally 0 here because the saved checkpoint_step
            # lives one frame up in saved["kwargs"]; the caller (which has
            # already loaded `ftm`) has the authoritative step on
            # `ftm._checkpoint_step` and applies it via set_steps elsewhere.
            # For roster initialization we just need the weights.
            "checkpoint_step": 0,
        }

    # --- Build right roster (always; same shape regardless of frozen_side) ---
    right_models: Dict[str, LeaguePPO] = {}
    state_names: Dict[str, str] = {}
    for state_name in league_matchup_states:
        left_char, right_char = _extract_chars_from_state_name(state_name)
        canonical_key = f"{_sanitize_matchup_token(left_char)}_vs_{_sanitize_matchup_token(right_char)}"
        right_model = league_constructor(
            constructor_args,
            "right",
            log_name=None,
            single_env=True,
            opponent=right_char,
            state_name=state_name,
            matchup_key=canonical_key,
        )
        if canonical_key in right_checkpoints:
            ckpt = right_checkpoints[canonical_key]
            right_model.set_parameters(ckpt["agent_dict"])
            right_model.set_steps(ckpt["checkpoint_step"])
            print(
                f"[league_continue] Resuming right {canonical_key} "
                f"from step {ckpt['checkpoint_step']}"
            )
        else:
            # No sibling right snapshot for this matchup. For frozen_side="left"
            # we fall back to the loaded left MA's weights (legacy behavior;
            # not great but the network shapes match because both sides share
            # the same policy architecture). For frozen_side="right" this
            # branch should not be hit for the loaded matchup (we override it
            # above), but other matchups missing a sibling still degrade to
            # this fallback.
            right_model.set_parameters(agent_dict)
            print(
                f"[league_continue] No right checkpoint for {canonical_key}, "
                f"using loaded agent_dict as fallback"
            )
        right_models[canonical_key] = right_model
        state_names[canonical_key] = state_name

    # --- Resolve left-side weights ---
    if frozen_side == "left":
        # Loaded task is the left MA — use its weights directly.
        left_weights = agent_dict
        left_step = 0  # caller restores authoritative step on ftm separately
    else:
        # Loaded task is right; we need a separate left MA snapshot.
        left_ckpt = _load_left_side_checkpoint(model_dir) if model_dir else None
        if left_ckpt is None:
            raise FileNotFoundError(
                "_run_league_continue_exploiter(frozen_side='right') needs a "
                "sibling MA*_left_*.pt checkpoint in model_dir to initialize "
                f"the left protagonist, but none was found in {model_dir!r}."
            )
        left_weights = left_ckpt["agent_dict"]
        left_step = left_ckpt["checkpoint_step"]

    # --- Build left model (single, matchup_key='left_vs_all') ---
    first_state = league_matchup_states[0]
    left_model = league_constructor(
        constructor_args,
        "left",
        log_name=None,
        single_env=True,
        state_name=first_state,
        matchup_key="left_vs_all",
    )
    left_model.set_parameters(left_weights)
    if left_step:
        left_model.set_steps(left_step)

    initial_agents = {
        "left": left_model,
        "right": right_models,
    }

    # The TRAINING side is the opposite of the frozen side. Filter learners
    # accordingly so we only spawn workers for the side we want to update.
    training_side = "right" if frozen_side == "left" else "left"

    mp_ctx = mp.get_context("spawn")

    with PayoffManager() as manager:
        shared_payoff = manager.Payoff(save_dir)
        league = League(
            args=constructor_args,
            initial_agents=initial_agents,
            constructor=league_constructor,
            payoff=shared_payoff,
            main_agents=1,
            main_exploiters=1,
            league_exploiters=2,
            state_names=state_names,
        )

        processes = []
        for idx in range(league.size()):
            player = league.get_player(idx)
            if player.side != training_side:
                continue
            learner = Learner(player)
            p = mp_ctx.Process(
                target=_league_worker,
                args=(idx, learner, total_timesteps, rollout_opponent_num),
            )
            processes.append(p)

        print(
            f"[league_continue_exploiter] frozen_side={frozen_side}, "
            f"training_side={training_side}: launching {len(processes)} "
            f"worker(s) across {len(league_matchup_states)} matchup(s)"
        )
        for p in processes:
            p.start()
        for p in processes:
            p.join()


def train_best_response(
    game_args: dict,
    model_to_exploit,
    task_file_path: str,
    eval_prot: bool,
    use_mirror: bool,
    eval_only: bool,
    proj_name: str,
    analysis_upload_proj_name: str,
    n_envs: int,
    is_spar_like: bool = False,
    br_index: int = 0,
    from_scratch: bool = False,
    exploiter_save_freq: int = 100000,
    br_tracker_patience: int = 20,
    br_tracker_tolerance: float = 1e-4,
    br_tracker_window_size: int = 50,
    use_br_reward_stagnation: bool = True,
    use_br_entropy_stagnation: bool = True,
    br_use_slope_early_stop: bool = False,
    br_slope_window: int = 20,
    br_slope_tolerance: float = 5e-3,
    br_min_slope_checks: int = 10,
    use_stagnation_early_stop: bool = False,
    use_stagnation_velocity_signal: bool = False,
    use_stagnation_entropy_signal: bool = True,
    stagnation_patience: int = 2000000,
    stagnation_tolerance: float = 1e-4,
    stagnation_rel_tolerance: float = 0.05,
    stagnation_ema_beta: float = 0.99,
    stagnation_eps: float = 1e-8,
    stagnation_eval_games: int = 0,
    entropy_stagnation_weight: float = 100.0,
    stagnation_lr_factor: float = 1.0,
    stagnation_lr_patience: int = 0,
    stagnation_use_slope_early_stop: bool = False,
    stagnation_slope_window: int = 20,
    stagnation_slope_tolerance: float = 5e-3,
    stagnation_min_slope_checks: int = 10,
    device: str = "cuda",
    dedicated_state_subset: Optional[List[str]] = None,
    matchup_label: Optional[str] = None,
    replicate_idx: Optional[int] = None,
    dedicated_job_id: Optional[int] = None,
    manual_stop_file: Optional[str] = None,
    manual_stop_key: Optional[str] = None,
    launch_local_br_eval: bool = False,
    use_wandb: bool = True,
    output_subdir: str = "",
    entropy_stop_ratio: float = 0.15,
    entropy_window_size: int = 50,
    entropy_warmup_checks: int = 100,
    entropy_ratio_only: bool = False,
    local_plot_dir: str = None,
    enable_local_kl_plot: bool = True,
    *,
    # Keyword-only required: total .learn() timesteps for this BR job.
    # Plumbed from the launcher via --br_training_steps; absent value
    # raises TypeError at call time.
    br_training_steps: int,
) -> None:
    """
    The core logic for a single best-response training run.

    Args:
        output_subdir: per-training-process subfolder name forwarded to
            local_br_eval.py via --output_subdir. See _derive_spar_run_subdir
            and the league-branch computation in __main__ for how this is
            populated by the launcher.
        entropy_stop_ratio / entropy_window_size / entropy_warmup_checks:
            forwarded to the BR Exploiter's RatingStagnationTracker.
            Together they define when the entropy-window early stop fires:
            after `entropy_warmup_checks` checks have elapsed, if the
            mean of the last `entropy_window_size` smoothed-entropy values
            is <= `entropy_stop_ratio` * initial-entropy, training stops.
        entropy_ratio_only: when True, ONLY the entropy-window ratio check
            can trigger early stopping (EMA plateau and rating stagnation
            are ignored). Manual stop files still work.
        TODO: Complete this.
    """
    checkpoint_path = task_file_path
    done_model_checkpoint_path = os.path.join(DONE_DIR, os.path.basename(checkpoint_path))
    checkpoint_basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    ftm = model_to_exploit
    # --- This is where your specific BR logic goes ---
    # 1. Load the frozen opponent
    # fixed_opponent = PPO.load(checkpoint_path)
    if use_wandb:
        wandb.init(project=proj_name,
                entity='jw4406',
                group="br_workers",
                config={"eval_rew": 0,
                        "exploiter_rew": 0,
                        "epochs": 0,
                        "br_wr": 0,
                        "main_training_epoch": 0,
                        "torch_device": device,
                        })
    # 2. Create your environment, passing the frozen opponent to it
    #    so the BR agent can play against it.
    # env = YourStreetFighterEnv(opponent_policy=fixed_opponent)
    if is_spar_like == True:
        game_args = argparse.Namespace(**game_args)
        # In dedicated mode, train BR in a subset environment (usually one
        # matchup replicated `n_envs` times). This isolates BR rollouts while
        # leaving the exploited CDS model topology untouched.
        effective_state_list = ftm.state_list if dedicated_state_subset is None else dedicated_state_subset
        env = env_generator(game_args, STATE=effective_state_list, n_envs=n_envs)
        # if eval_prot is True: # we're training an optimal adversary
        #     dstb_action_space = Box(low=ftm.dstb_action_space.low, high=ftm.dstb_action_space.high, shape=ftm.dstb_action_space.shape)
        #     env.action_space = dstb_action_space
        # else:
        #     assert eval_prot is False 
        #     # we're training an optimal ego against the current adversary
        #     ego_action_space = Box(low=ftm.action_space.low, high=ftm.action_space.high, shape=ftm.action_space.shape)
        #     env.action_space = ego_action_space
    else:
        game_args = argparse.Namespace(**game_args)
        effective_state_list = ftm.state_list if dedicated_state_subset is None else dedicated_state_subset
        env = env_generator(game_args, STATE=effective_state_list, n_envs=n_envs)

    # 3. Create a new agent to be the best response
    dedicated_state = dedicated_state_subset[0] if dedicated_state_subset else None
    ego_is_left = _is_ego_left_for_state(ftm, dedicated_state) if dedicated_state else True
    br_agent = Exploiter(
        'CnnPolicy' if is_image_space(env.observation_space) else 'MlpPolicy',
        env,
        device=device,
        exploited=ftm,
        n_steps=1024,
        batch_size=512,
        n_epochs=4,
        exploiting='ego' if eval_prot is True else 'adv',
        ego_is_left=ego_is_left,
        br_tracker_patience=br_tracker_patience,
        br_tracker_tolerance=br_tracker_tolerance,
        br_tracker_window_size=br_tracker_window_size,
        use_br_reward_stagnation=use_br_reward_stagnation,
        use_br_entropy_stagnation=use_br_entropy_stagnation,
        br_use_slope_early_stop=br_use_slope_early_stop,
        br_slope_window=br_slope_window,
        br_slope_tolerance=br_slope_tolerance,
        br_min_slope_checks=br_min_slope_checks,
        entropy_stop_ratio=entropy_stop_ratio,
        entropy_window_size=entropy_window_size,
        entropy_warmup_checks=entropy_warmup_checks,
        entropy_ratio_only=entropy_ratio_only,
        use_wandb=use_wandb,
        verbose=1,
        local_plot_dir=local_plot_dir,
        enable_local_kl_plot=enable_local_kl_plot,
    )
    br_agent.is_spar_like = is_spar_like # TODO: This is a stupid hack to get the BR agent to know if it is a SPAR model or not. Remove this once we have a better way to do this.
    # 4. Train the BR agent
    #
    # Name each BR checkpoint with rich metadata so downstream analysis can
    # directly identify:
    # - which side was exploited (ego/adv),
    # - which matchup slice this dedicated run belongs to,
    # - which replicate index it is within that matchup.
    #
    # This makes your "which character matchup performs worst" analysis much
    # easier because filenames become self-describing.
    exploit_side = "ego" if eval_prot is True else "adv"
    dedicated_suffix_parts: List[str] = []
    if matchup_label is not None:
        dedicated_suffix_parts.append(f"matchup_{_sanitize_for_filename(matchup_label)}")
    if replicate_idx is not None:
        dedicated_suffix_parts.append(f"rep_{replicate_idx}")
    if dedicated_job_id is not None:
        dedicated_suffix_parts.append(f"job_{dedicated_job_id}")
    dedicated_suffix = "_".join(dedicated_suffix_parts)
    if dedicated_suffix != "":
        dedicated_suffix = f"_{dedicated_suffix}"
    br_model_name = (
        f"br{br_index}_to_{os.path.splitext(os.path.basename(checkpoint_path))[0]}"
        f"_exploiting_{exploit_side}{dedicated_suffix}.zip"
    )
    exploiter_callback = ExploiterCheckpointCallback(save_freq=exploiter_save_freq // n_envs, save_path=BR_MODEL_DIR, name_prefix=br_model_name)
    stop_key = _sanitize_for_filename(str(manual_stop_key if manual_stop_key is not None else f"br{br_index}_{exploit_side}"))
    default_stop_file = os.path.join(TASK_DIR, "stop", f"STOP_{stop_key}")
    stop_file_path = str(manual_stop_file) if manual_stop_file is not None else default_stop_file
    os.makedirs(os.path.dirname(stop_file_path), exist_ok=True)
    manual_stop_callback = ManualStopFileCallback(stop_file=stop_file_path, stop_key=stop_key)
    train_callback = CallbackList([exploiter_callback, manual_stop_callback])
    print(
        f"Manual per-job stop configured: key={stop_key}, stop_file={stop_file_path}",
        flush=True,
    )
     
    if eval_only == False:
        print("eval_only was passed as False. Training the BR agent.")
        if from_scratch == True:
            if hasattr(br_agent, "br_convergence_tracker") and br_agent.br_convergence_tracker is not None:
                matchup_tag = f"_{_sanitize_for_filename(matchup_label)}" if matchup_label is not None else ""
                br_agent.br_convergence_tracker.local_plot_prefix = f"dedicated_exploiter_{stop_key}{matchup_tag}_{checkpoint_basename}"
            br_agent.learn(total_timesteps=br_training_steps, callback=train_callback)
        else:
            # if eval prot is True we are training an optimal adversary so we need to update the adversary
            # if eval prot is False we are training an optimal ego against the current adversary so we need to update the ego
            ftm.env = env # this is new (17:16)
            ftm.envs_per_matchup = ftm.envs_per_matchup

            # Re-instantiate Q_RolloutBuffers when the new env has different
            # n_envs than the checkpoint's loading env (mirrors LeaguePPO logic
            # at _continue_train_loaded_league_model).
            actual_n_envs = ftm.env.num_envs

            # ---- Per-matchup continue topology override ----------------
            # When a single-state subset is provided (per-matchup continue
            # exploiter), reshape the model's view of its own topology to
            # "1 adversary × n_envs slots, all on this matchup". CDS's
            # collect_rollouts/train loops route slot i -> adv i (via
            # `i*envs_per_matchup`), and the policy's per-matchup head
            # dicts are keyed by f"{matchup}_{adv_index}". Since num_adv
            # collapses to 1, the loop produces lookup key f"{matchup}_0",
            # which doesn't exist in the loaded policy (the trained head
            # is at f"{matchup}_<X>"). We add an alias entry on each
            # per-matchup ModuleDict so f"{matchup}_0" points to the same
            # nn.Module instance as f"{matchup}_<X>" — gradients update
            # the original head's params during training. The save-time
            # state_dict will contain duplicate entries under both keys
            # (acceptable for analysis snapshots; not for round-tripping
            # back into a multi-matchup training pipeline).
            if (
                is_spar_like
                and dedicated_state_subset is not None
                and len(dedicated_state_subset) >= 1
                and not from_scratch
            ):
                _dedicated_state = dedicated_state_subset[0]
                _fixed_matchup_idx = _resolve_matchup_index_for_state(ftm, _dedicated_state)
                _matchup_label = ftm.matchups[_fixed_matchup_idx * ftm.envs_per_matchup]
                _new_key = f"{_matchup_label}_0"
                _old_key = f"{_matchup_label}_{_fixed_matchup_idx}"
                _aliased = []
                for _attr_name in ("value_net", "dstb_action_net"):
                    _head_dict = getattr(ftm.policy, _attr_name, None)
                    if _head_dict is None:
                        continue
                    if _old_key in _head_dict and _new_key not in _head_dict:
                        _head_dict[_new_key] = _head_dict[_old_key]
                        _aliased.append(_attr_name)
                # Slice the elo trackers down to the single-matchup view.
                if hasattr(ftm, "elo_adversary_ratings") and len(ftm.elo_adversary_ratings) > _fixed_matchup_idx:
                    ftm.elo_adversary_ratings = ftm.elo_adversary_ratings[
                        _fixed_matchup_idx:_fixed_matchup_idx + 1
                    ].copy()
                if hasattr(ftm, "elo_games_played") and len(ftm.elo_games_played) > _fixed_matchup_idx:
                    ftm.elo_games_played = ftm.elo_games_played[
                        _fixed_matchup_idx:_fixed_matchup_idx + 1
                    ].copy()
                # Override topology on the model AND the policy. The buffer
                # reinit immediately below uses ftm.num_adversaries to size
                # adversary_buffers, so this must happen first.
                ftm.matchups = [_matchup_label] * actual_n_envs
                ftm.envs_per_matchup = actual_n_envs
                ftm.num_adversaries = 1
                if hasattr(ftm.policy, "matchups"):
                    ftm.policy.matchups = ftm.matchups
                if hasattr(ftm.policy, "envs_per_matchup"):
                    ftm.policy.envs_per_matchup = ftm.envs_per_matchup
                if hasattr(ftm.policy, "num_adversaries"):
                    ftm.policy.num_adversaries = ftm.num_adversaries
                if hasattr(ftm.policy, "num_env_per_adv"):
                    ftm.policy.num_env_per_adv = actual_n_envs
                print(
                    "[continue] Per-matchup topology override applied: "
                    f"matchup_label={_matchup_label}, "
                    f"fixed_matchup_idx={_fixed_matchup_idx}, "
                    f"-> num_adversaries=1, envs_per_matchup={actual_n_envs}, "
                    f"head alias {_old_key} -> {_new_key} on {_aliased}.",
                    flush=True,
                )

            if ftm.n_envs != actual_n_envs:
                from stable_baselines3.common.buffers import DictRolloutBuffer, Q_RolloutBuffer
                from gymnasium import spaces
                ftm.n_envs = actual_n_envs
                buffer_cls = DictRolloutBuffer if isinstance(ftm.observation_space, spaces.Dict) else Q_RolloutBuffer
                ftm.rollout_buffer = buffer_cls(
                    ftm.n_steps,
                    ftm.observation_space,
                    ftm.action_space,
                    device=ftm.device,
                    gamma=ftm.gamma,
                    gae_lambda=ftm.gae_lambda,
                    n_envs=actual_n_envs,
                )
                if hasattr(ftm, "adversary_buffers") and ftm.adversary_buffers is not None:
                    adversary_buffers = []
                    # Use ftm.num_adversaries (post-override) so the per-
                    # matchup continue case ends up with exactly one adv
                    # buffer; full continue keeps the loaded length since
                    # num_adversaries was unchanged.
                    for _ in range(int(ftm.num_adversaries)):
                        adversary_buffers.append(
                            buffer_cls(
                                ftm.n_steps,
                                ftm.observation_space,
                                ftm.action_space,
                                device=ftm.device,
                                gamma=ftm.gamma,
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
            ftm.br_tracker_patience = br_tracker_patience
            ftm.br_tracker_tolerance = br_tracker_tolerance
            ftm.br_tracker_window_size = br_tracker_window_size
            ftm.use_stagnation_entropy_signal = bool(use_stagnation_entropy_signal)
            ftm.use_stagnation_velocity_signal = bool(use_stagnation_velocity_signal)
            ftm.use_stagnation_early_stop = bool(use_stagnation_early_stop)
            ftm.stagnation_patience_cfg = int(stagnation_patience)
            ftm.stagnation_tolerance_cfg = float(stagnation_tolerance)
            ftm.stagnation_rel_tolerance_cfg = float(stagnation_rel_tolerance)
            ftm.stagnation_ema_beta_cfg = float(stagnation_ema_beta)
            ftm.stagnation_eps_cfg = float(stagnation_eps)
            ftm.stagnation_eval_games_cfg = None if int(stagnation_eval_games) <= 0 else int(stagnation_eval_games)
            ftm.entropy_stagnation_weight_cfg = float(entropy_stagnation_weight)
            ftm.stagnation_lr_factor_cfg = float(stagnation_lr_factor)
            ftm.stagnation_lr_patience_cfg = int(stagnation_lr_patience)
            ftm.stagnation_use_slope_early_stop_cfg = bool(stagnation_use_slope_early_stop)
            ftm.stagnation_slope_window_cfg = int(stagnation_slope_window)
            ftm.stagnation_slope_tolerance_cfg = float(stagnation_slope_tolerance)
            ftm.stagnation_min_slope_checks_cfg = int(stagnation_min_slope_checks)
            ftm.entropy_ratio_only_cfg = bool(entropy_ratio_only)
            if hasattr(ftm, "stagnation_tracker") and ftm.stagnation_tracker is not None:
                tracker = ftm.stagnation_tracker
                tracker.patience = int(stagnation_patience)
                tracker.tolerance = float(stagnation_tolerance)
                tracker.rel_tolerance = float(stagnation_rel_tolerance)
                tracker.ema_beta = float(stagnation_ema_beta)
                tracker.eps = float(stagnation_eps)
                tracker.eval_games = max(1, int(stagnation_eval_games))
                tracker.entropy_weight = float(entropy_stagnation_weight)
                tracker.lr_patience = int(stagnation_lr_patience)
                tracker.use_velocity_signal = bool(use_stagnation_velocity_signal)
                tracker.use_entropy_signal = bool(use_stagnation_entropy_signal)
                tracker.use_slope_early_stop = bool(stagnation_use_slope_early_stop)
                tracker.slope_window = max(2, int(stagnation_slope_window))
                tracker.slope_tolerance = float(stagnation_slope_tolerance)
                tracker.min_slope_checks = max(1, int(stagnation_min_slope_checks))
                tracker.entropy_ratio_only = bool(entropy_ratio_only)
                if local_plot_dir is not None:
                    tracker.local_plot_dir = local_plot_dir
            ftm.use_wandb = use_wandb
            ftm.br_manual_stop_key = stop_key
            ftm._checkpoint_basename = checkpoint_basename
            if is_spar_like:

                ftm.learn(total_timesteps=br_training_steps, callback=train_callback, update_ego=not eval_prot, update_adversary=eval_prot)
            elif isinstance(ftm, LeaguePPO):
                rollout_opponent_num = getattr(
                    ftm.constructor_args,
                    "rollout_opponent_num",
                    getattr(game_args, "rollout_opponent_num", 5),
                )
                league_states = getattr(ftm, "_worker_full_state_list", None) or getattr(ftm, "state_list", [])
                # Frozen side and matchup key were stashed by load_league_model
                # from the saved checkpoint kwargs (pre-flip). Pass them
                # through so the helper trains the opposite side and, when
                # frozen_side='right', overrides the correct right-roster
                # slot with the loaded weights.
                frozen_side = getattr(ftm, "_worker_loaded_side", "left")
                loaded_matchup_key = getattr(ftm, "_worker_loaded_matchup_key", None)
                _run_league_continue_exploiter(
                    constructor_args=ftm.constructor_args if ftm.constructor_args is not None else game_args,
                    agent_dict=ftm.get_parameters(),
                    league_matchup_states=league_states,
                    total_timesteps=br_training_steps,
                    rollout_opponent_num=rollout_opponent_num,
                    model_dir=os.path.dirname(checkpoint_path),
                    frozen_side=frozen_side,
                    loaded_matchup_key=loaded_matchup_key,
                )
            else:
                ftm.learn(total_timesteps=br_training_steps, callback=train_callback)
        #br_agent.learn(total_timesteps=br_training_steps, callback=exploiter_callback)

        local_plot_and_eval_file = os.path.join(current_dir, "local_br_eval.py")
        
        br_interval_num = exploiter_callback.n_calls * env.num_envs // exploiter_callback.save_freq
        #br_model_path = os.path.join(BR_MODEL_DIR, f"{br_model_name}_{br_interval_num}000_steps.zip")
        br_model_path = exploiter_callback.model_path
        if launch_local_br_eval:
            subprocess.run(["python", local_plot_and_eval_file,
            "--eval_prot", str(eval_prot),
            "--main_checkpoint_model_path", checkpoint_path,
            "--done_model_checkpoint_path", done_model_checkpoint_path,
            "--br_checkpoint_model_path", br_model_path,
            # Evaluate exactly the state slice used by this BR run.
            "--full_state_list", str(ftm.state_list),
            "--state_list", str(effective_state_list),
            "--dedicated_exploiter", str(from_scratch),
            "--br_index", str(br_index),
            "--game_args", json.dumps(vars(game_args)),
            "--device", device,
            # Tell local_br_eval whether to use load_league_model +
            # policy/policy_other (league) or CleanDerivativeFreeSPAR.load
            # (CDS). Derived from the loaded model class so it stays in
            # sync with whatever path was taken in run_br_for_task_in_subprocess.
            "--is_league", str(isinstance(ftm, LeaguePPO)),
            # Per-training-process subfolder. Empty string == legacy
            # unsegregated layout. Computed by the launcher from the source
            # dir (league) or the .task filename prefix (SPAR).
            "--output_subdir", output_subdir,
            # Training style label embedded in the .txt filenames so the
            # downstream aggregator can include it in plot filenames.
            # "league" for LeaguePPO; otherwise the CDS arch ("ippo" or
            # "spar") inferred at load time.
            "--training_style", (
                "league" if isinstance(ftm, LeaguePPO)
                else getattr(ftm, "_worker_cds_arch", "spar")
            ),
            ])
        #agg_file = os.path.join(current_dir, "aggregate_to_wandb.py")
        #subprocess.Popen(["python", agg_file, "--read_from_proj_name", proj_name, "--upload_to_proj_name", analysis_upload_proj_name])


def run_br_for_task_in_subprocess(
    game_args: dict,
    task_file_path: str,
    eval_prot: bool,
    use_mirror: bool,
    eval_only: bool,
    proj_name: str,
    analysis_upload_proj_name: str,
    n_envs: int,
    is_spar_like: bool,
    br_index: int,
    from_scratch: bool = False,
    exploiter_save_freq: int = 100000,
    br_tracker_patience: int = 10,
    br_tracker_tolerance: float = 1e-4,
    br_tracker_window_size: int = 50,
    use_br_reward_stagnation: bool = True,
    use_br_entropy_stagnation: bool = True,
    br_use_slope_early_stop: bool = False,
    br_slope_window: int = 20,
    br_slope_tolerance: float = 5e-3,
    br_min_slope_checks: int = 10,
    use_stagnation_early_stop: bool = False,
    use_stagnation_velocity_signal: bool = False,
    use_stagnation_entropy_signal: bool = True,
    stagnation_patience: int = 2000000,
    stagnation_tolerance: float = 1e-4,
    stagnation_rel_tolerance: float = 0.05,
    stagnation_ema_beta: float = 0.99,
    stagnation_eps: float = 1e-8,
    stagnation_eval_games: int = 0,
    entropy_stagnation_weight: float = 100.0,
    stagnation_lr_factor: float = 1.0,
    stagnation_lr_patience: int = 0,
    stagnation_use_slope_early_stop: bool = False,
    stagnation_slope_window: int = 20,
    stagnation_slope_tolerance: float = 5e-3,
    stagnation_min_slope_checks: int = 10,
    device: str = "cuda",
    state_subset: Optional[List[str]] = None,
    matchup_label: Optional[str] = None,
    replicate_idx: Optional[int] = None,
    dedicated_job_id: Optional[int] = None,
    manual_stop_file: Optional[str] = None,
    manual_stop_key: Optional[str] = None,
    launch_local_br_eval: bool = False,
    use_wandb: bool = True,
    is_league: bool = False,
    league_matchup_states: Optional[List[str]] = None,
    output_subdir: str = "",
    entropy_stop_ratio: float = 0.15,
    entropy_window_size: int = 50,
    entropy_warmup_checks: int = 100,
    entropy_ratio_only: bool = False,
    local_plot_dir: str = None,
    enable_local_kl_plot: bool = True,
    *,
    # Keyword-only required: total .learn() timesteps for this BR job.
    # Plumbed from the launcher via --br_training_steps; absent value
    # raises TypeError at call time.
    br_training_steps: int,
) -> None:
    """
    Worker function for running a single BR training instance in a separate process.
    Each subprocess loads its own copy of the model to avoid pickling issues.

    *output_subdir* is forwarded to train_best_response and from there to
    local_br_eval.py so its outputs land in a per-training-process folder.

    *entropy_*_* knobs* are forwarded to train_best_response and used to
    parameterize the BR Exploiter's RatingStagnationTracker.
    """
    _ensure_cpu_only_env(device)
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    if is_league:
        loaded_model = load_league_model(
            game_args,
            task_file_path,
            league_matchup_states=league_matchup_states or [],
            n_envs=n_envs,
            device=device,
            use_wandb=use_wandb,
        )
        dedicated_state_list_for_env = None
        if state_subset is not None:
            if len(state_subset) == 0:
                raise ValueError("state_subset was provided but empty.")
            dedicated_state = state_subset[0]
            dedicated_state_list_for_env = [dedicated_state for _ in range(n_envs)]
            if from_scratch:
                # Dedicated league exploiter: wrap LeaguePPO policy so the
                # frozen side exposes a CDS-compatible call signature for
                # Exploiter.collect_rollouts.
                #
                # Side mapping (consistent with the league branch in __main__):
                #   eval_prot=True  -> exploit ego, BR plays adv -> frozen=left
                #   eval_prot=False -> exploit adv, BR plays ego -> frozen=right
                # The league branch derives eval_prot from the loaded task's
                # saved side, so this assignment is just propagating that
                # decision through to adapter selection.
                frozen_side = "left" if eval_prot else "right"
                loaded_model.policy = _LeaguePolicyAdapter(
                    loaded_model, side=frozen_side
                )
                print(
                    "Dedicated league adapter configured: "
                    f"state={dedicated_state}, frozen_side={frozen_side}, "
                    f"replicated_env_count={n_envs}"
                )
        loaded_model.use_wandb = use_wandb
    elif is_spar_like:
        # NOTE: Do not shrink CDS model topology based on state_subset.
        # We always load full topology and handle dedicated constraints via
        # env slicing + fixed-head policy adapter below.
        loaded_model = load_spar_model(
            game_args,
            task_file_path,
            n_envs=n_envs,
            device=device,
            use_wandb=use_wandb,
        )
        dedicated_state_list_for_env = None
        if state_subset is not None:
            if len(state_subset) == 0:
                raise ValueError("state_subset was provided but empty.")
            dedicated_state = state_subset[0]
            dedicated_state_list_for_env = [dedicated_state for _ in range(n_envs)]
            if from_scratch:
                # Dedicated CDS exploiter: wrap policy so all forwards on the
                # exploited model route through one fixed matchup head. The
                # adapter is consumed by Exploiter.collect_rollouts at
                # *inference* time (BR is the trained side, exploited model
                # is frozen).
                #
                # Continue mode is intentionally NOT wrapped: ftm.learn(...)
                # runs CDS's own training loop and calls self.policy(...)
                # with kwargs the adapter rejects, which would crash mid-
                # update. With a single-state subset the env is already
                # restricted to one matchup, so CDS routes data through the
                # correct head naturally without the adapter.
                fixed_matchup_idx = _resolve_matchup_index_for_state(loaded_model, dedicated_state)
                loaded_model.policy = _FixedMatchupPolicyAdapter(
                    loaded_model.policy, fixed_matchup_idx=fixed_matchup_idx
                )
                print(
                    "Dedicated CDS adapter configured: "
                    f"state={dedicated_state}, fixed_matchup_idx={fixed_matchup_idx}, "
                    f"replicated_env_count={n_envs}"
                )
            else:
                print(
                    "Continue CDS exploiter on single-matchup subset: "
                    f"state={dedicated_state}, replicated_env_count={n_envs} "
                    "(skipping FixedMatchupPolicyAdapter — CDS will route "
                    "data through the matching head via the env's state list)."
                )
        if exploiter_save_freq * len(loaded_model.matchups) > br_training_steps:
            print("-------------------------------------------\n\n ")
            print("ERROR!")
            print("ERROR! Exploiter save frequency is greater than BR training steps. This will result in no exploiter checkpoints being saved AND AN ERROR RIGHT BEFORE LOCAL BR EVAL")
            print("ERROR!")
            print("-------------------------------------------\n\n ")
            quit() # QUIT THE PROGRAM
        loaded_model.c_learning_rate = 1e-4
        loaded_model.d_learning_rate = 2e-4
        loaded_model.v_learning_rate = 5e-4
        loaded_model.policy.ctrl_optimizer.param_groups[0]['lr'] = 1e-4
        loaded_model.policy.dstb_optimizer.param_groups[0]['lr'] = 1e-4
        loaded_model.policy.value_optimizer.param_groups[0]['lr'] = 2e-4
        loaded_model.use_lr_annealing = False
        loaded_model.use_wandb = use_wandb
    else:
        raise NotImplementedError("Non-SPAR multiprocessing BR training is not implemented.")

    run_mode = "dedicated" if from_scratch else "continue"
    exploit_side = "ego" if eval_prot else "adv"
    loaded_arch = getattr(loaded_model, "_worker_cds_arch", "unknown")
    print(
        f"WORKER [{os.getpid()}]: BR launch mode={run_mode}, side={exploit_side}, "
        f"cds_class={loaded_model.__class__.__name__}, cds_arch={loaded_arch}"
    )

    train_best_response(
        game_args,
        loaded_model,
        task_file_path,
        eval_prot=eval_prot,
        use_mirror=use_mirror,
        eval_only=eval_only,
        proj_name=proj_name,
        analysis_upload_proj_name=analysis_upload_proj_name,
        n_envs=n_envs,
        is_spar_like=is_spar_like,
        br_index=br_index,
        from_scratch = from_scratch,
        exploiter_save_freq=exploiter_save_freq,
        br_tracker_patience=br_tracker_patience,
        br_tracker_tolerance=br_tracker_tolerance,
        br_tracker_window_size=br_tracker_window_size,
        use_br_reward_stagnation=use_br_reward_stagnation,
        use_br_entropy_stagnation=use_br_entropy_stagnation,
        br_use_slope_early_stop=br_use_slope_early_stop,
        br_slope_window=br_slope_window,
        br_slope_tolerance=br_slope_tolerance,
        br_min_slope_checks=br_min_slope_checks,
        use_stagnation_early_stop=use_stagnation_early_stop,
        use_stagnation_velocity_signal=use_stagnation_velocity_signal,
        use_stagnation_entropy_signal=use_stagnation_entropy_signal,
        stagnation_patience=stagnation_patience,
        stagnation_tolerance=stagnation_tolerance,
        stagnation_rel_tolerance=stagnation_rel_tolerance,
        stagnation_ema_beta=stagnation_ema_beta,
        stagnation_eps=stagnation_eps,
        stagnation_eval_games=stagnation_eval_games,
        entropy_stagnation_weight=entropy_stagnation_weight,
        stagnation_lr_factor=stagnation_lr_factor,
        stagnation_lr_patience=stagnation_lr_patience,
        stagnation_use_slope_early_stop=stagnation_use_slope_early_stop,
        stagnation_slope_window=stagnation_slope_window,
        stagnation_slope_tolerance=stagnation_slope_tolerance,
        stagnation_min_slope_checks=stagnation_min_slope_checks,
        device=device,
        dedicated_state_subset=dedicated_state_list_for_env,
        matchup_label=matchup_label,
        replicate_idx=replicate_idx,
        dedicated_job_id=dedicated_job_id,
        manual_stop_file=manual_stop_file,
        manual_stop_key=manual_stop_key,
        launch_local_br_eval=launch_local_br_eval,
        use_wandb=use_wandb,
        output_subdir=output_subdir,
        entropy_stop_ratio=entropy_stop_ratio,
        entropy_window_size=entropy_window_size,
        entropy_warmup_checks=entropy_warmup_checks,
        entropy_ratio_only=entropy_ratio_only,
        local_plot_dir=local_plot_dir,
        enable_local_kl_plot=enable_local_kl_plot,
        # Keyword-only required on train_best_response
        br_training_steps=br_training_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_prot", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--eval_adv", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--eval_only", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--proj_name", type=str, required=True)
    parser.add_argument("--analysis_upload_proj_name", type=str, required=True)
    parser.add_argument("--load_br", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--which_env", choices=['my_pendulum', 'my_walker2d_v4', 'my_mountain_car_continuous', 'my_half_cheetah', 'my_hopper', 'my_ant'], required=True)
    parser.add_argument("--is_league", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--league_dir", type=str, required=False)
    parser.add_argument("--use_mirror", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--task_dir", type=str, required=False)
    parser.add_argument("--num_brs", type=int, default=6, help="Number of independent BR agents to train per main checkpoint.")
    parser.add_argument("--n_envs", type=int, default=2, help="Number of environments to run in parallel.")
    parser.add_argument("--DEBUG", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--num_full_exploiters", type=int, default=4, help="Number of full exploiters to train.")
    parser.add_argument("--max_concurrent_jobs", type=int, default=0, help="Max BR subprocesses alive at once. 0 = auto-compute from --num_cores. -1 = unlimited (launch all).")
    parser.add_argument("--num_cores", type=int, default=0, help="CPU cores available to this worker. 0 = auto-detect via os.cpu_count(). Used to compute max_concurrent_jobs when it is 0.")
    parser.add_argument("--num_continue_exploiters", type=int, default=4, help="Number of continue exploiters to train.")
    parser.add_argument("--dedicated_exploiter", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--continue_exploiters", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--exploiter_save_freq", type=int, required=True, default=100000, help="Frequency of exploiter checkpoint saves.")
    parser.add_argument("--br_training_steps", type=int, required=True,
                        help="Total .learn() timesteps for each BR job. "
                             "Required; orchestrator argparse should plumb "
                             "this through from the launcher.")
    parser.add_argument("--br_tracker_patience", type=int, default=10, help="Patience (in checks) for BR convergence early stopping.")
    parser.add_argument("--br_tracker_tolerance", type=float, default=1e-4, help="Tolerance for BR convergence stagnation checks.")
    parser.add_argument("--br_tracker_window_size", type=int, default=50, help="Window size used to smooth BR convergence metric.")
    parser.add_argument("--use_br_reward_stagnation", choices=['True', 'False'], default='True', help="Use reward stability in BR early-stopping tracker.")
    parser.add_argument("--use_br_entropy_stagnation", choices=['True', 'False'], default='True', help="Use entropy stability in BR early-stopping tracker.")
    parser.add_argument("--br_use_slope_early_stop", choices=['True', 'False'], default='False', help="Use slope plateau detection for BR early-stopping tracker.")
    parser.add_argument("--br_slope_window", type=int, default=20, help="Number of BR checks to fit slope for plateau detection.")
    parser.add_argument("--br_slope_tolerance", type=float, default=5e-3, help="Normalized BR slope threshold for plateau-based stopping.")
    parser.add_argument("--br_min_slope_checks", type=int, default=10, help="Minimum BR checks before slope-based stopping can trigger.")
    # Entropy-window early-stop knobs (passed to RatingStagnationTracker
    # inside Exploiter). Stop fires when the running mean of EMA-smoothed
    # rollout entropy drops to <= entropy_stop_ratio * (initial entropy).
    parser.add_argument("--entropy_stop_ratio", type=float, default=0.15, help="Fraction of initial rollout entropy at which the BR exploiter early-stops (default 0.15 = 15%%).")
    parser.add_argument("--entropy_window_size", type=int, default=50, help="Number of stagnation checks to average entropy over for the stop-ratio test.")
    parser.add_argument("--entropy_warmup_checks", type=int, default=100, help="Stagnation checks that must elapse before the entropy-window stop can trigger.")
    parser.add_argument("--entropy_ratio_only", choices=['True', 'False'], default='False', help="When True, ONLY the entropy-window ratio check triggers early stopping (EMA plateau and rating stagnation are ignored). Manual stop files still work.")
    parser.add_argument("--use_stagnation_early_stop", choices=['True', 'False'], default='False', help="Use stagnation tracker for CDS early stopping (continue exploiters).")
    parser.add_argument("--use_stagnation_velocity_signal", choices=['True', 'False'], default='False', help="Use rating-movement velocity in CDS stagnation tracker (continue exploiters).")
    parser.add_argument("--use_stagnation_entropy_signal", choices=['True', 'False'], default='True', help="Use entropy signal in CDS stagnation tracker (continue exploiters).")
    parser.add_argument("--stagnation_patience", type=int, default=2000000, help="Patience checks for CDS stagnation tracker (continue exploiters).")
    parser.add_argument("--stagnation_tolerance", type=float, default=1e-4, help="Absolute tolerance for CDS stagnation tracker.")
    parser.add_argument("--stagnation_rel_tolerance", type=float, default=0.05, help="Relative tolerance for CDS stagnation tracker.")
    parser.add_argument("--stagnation_ema_beta", type=float, default=0.99, help="EMA beta for CDS stagnation tracker.")
    parser.add_argument("--stagnation_eps", type=float, default=1e-8, help="Numerical epsilon floor for CDS stagnation tracker.")
    parser.add_argument("--stagnation_eval_games", type=int, default=0, help="Games between CDS stagnation checks; <=0 uses tracker default.")
    parser.add_argument("--entropy_stagnation_weight", type=float, default=100.0, help="Entropy weight in CDS stagnation metric.")
    parser.add_argument("--stagnation_lr_factor", type=float, default=1.0, help="LR drop factor for CDS stagnation-triggered annealing (1.0 disables).")
    parser.add_argument("--stagnation_lr_patience", type=int, default=0, help="Checks before CDS stagnation-triggered LR drop (0 disables).")
    parser.add_argument("--stagnation_use_slope_early_stop", choices=['True', 'False'], default='False', help="Use slope plateau detection for CDS stagnation early stopping.")
    parser.add_argument("--stagnation_slope_window", type=int, default=20, help="Number of CDS checks to fit slope for plateau detection.")
    parser.add_argument("--stagnation_slope_tolerance", type=float, default=5e-3, help="Normalized CDS slope threshold for plateau-based stopping.")
    parser.add_argument("--stagnation_min_slope_checks", type=int, default=10, help="Minimum CDS checks before slope-based stopping can trigger.")

    parser.add_argument('--reset', choices=['round', 'match', 'game'],help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument("--side", type=str, help="Side", default="left", required=True, choices=["left", "right", "both"])

    parser.add_argument('--render', choices=['True', 'False'], help='Whether to render the game screen', default='False')
    parser.add_argument('--enable_combo', choices=['True', 'False'], help='Enable special move action space for environment', default='True')
    parser.add_argument('--null_combo', choices=['True', 'False'], help='Null action space for special move', default='False')
    parser.add_argument('--transform_action', choices=['True', 'False'], help='Transform action space to MultiDiscrete', default='False')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    parser.add_argument('--launch_local_br_eval', choices=['True', 'False'], help='Launch local br eval', default='False')
    parser.add_argument('--use_wandb', choices=['True', 'False'], help='Enable Weights & Biases logging', default='False')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Torch device for policies and training (e.g. cpu, cuda, cuda:0)',
    )
    parser.add_argument(
        '--league_matchup_states',
        type=str,
        nargs='+',
        default=None,
        help='Retro state strings for league BR matchups. '
             'If omitted, inferred from sibling MA*_right*.pt files in the task directory.',
    )
    # whether or not we want to do full br training or continuing the CDS as exploiter
    args = parser.parse_args()

    # Print all runtime CLI settings in a readable way for debugging/repro.
    def _print_args_human_readable(parsed_args):
        args_dict = vars(parsed_args)
        print("\n====== NEW_BR_WORKER CLI Arguments ======")
        max_key_len = max(len(key) for key in args_dict)
        for key in sorted(args_dict):
            value = args_dict[key]
            value_str = pformat(value, compact=True)
            print(f"{key:<{max_key_len}} : {value_str}")
        print("=========================================\n")

    _print_args_human_readable(args)

    _ensure_cpu_only_env(args.device)
    args.render = True if args.render == 'True' else False
    args.enable_combo = True if args.enable_combo == 'True' else False
    args.null_combo = True if args.null_combo == 'True' else False
    args.transform_action = True if args.transform_action == 'True' else False
    game_args = {
        "reset": args.reset,
        "side": args.side,
        "render": args.render,
        "enable_combo": args.enable_combo,
        "null_combo": args.null_combo,
        "transform_action": args.transform_action,
        "seed": args.seed,
    }
    args.DEBUG = args.DEBUG == 'True'
    args.dedicated_exploiter = args.dedicated_exploiter == 'True'
    args.continue_exploiters = args.continue_exploiters == 'True'
    args.launch_local_br_eval = args.launch_local_br_eval == 'True'
    args.use_wandb = args.use_wandb == 'True'
    args.use_br_reward_stagnation = args.use_br_reward_stagnation == 'True'
    args.use_br_entropy_stagnation = args.use_br_entropy_stagnation == 'True'
    args.br_use_slope_early_stop = args.br_use_slope_early_stop == 'True'
    args.use_stagnation_early_stop = args.use_stagnation_early_stop == 'True'
    args.use_stagnation_velocity_signal = args.use_stagnation_velocity_signal == 'True'
    args.use_stagnation_entropy_signal = args.use_stagnation_entropy_signal == 'True'
    args.stagnation_use_slope_early_stop = args.stagnation_use_slope_early_stop == 'True'
    args.is_league = args.is_league == 'True'
    if args.is_league:
        if args.league_dir is None or args.league_dir == "":
            print("ERROR: League directory is required when is_league is True.")
            exit(1)
    # Linux defaults to "fork", which cannot safely inherit an initialized CUDA
    # runtime. Use explicit "spawn" for BR worker subprocesses.
    mp_ctx = mp.get_context("spawn")


    args.eval_prot = args.eval_prot == 'True'
    args.eval_adv = args.eval_adv == 'True'
    if args.eval_only == 'True':
        print("WARNING!")
        print("This is an EVAL ONLY run. No exploiter training will be performed.")
        print("WARNING!")
    if not args.eval_prot or not args.eval_adv:
        print("WARNING!")
        training_type = "adversary" if args.eval_prot else "ego " # if eval_prot is True we are training an optimal adversary so we need to update the adversary
        print(f"WARNING! Only {training_type} training will be performed.")
    # --- Auto-compute max_concurrent_jobs from num_cores ---
    if args.num_cores <= 0:
        args.num_cores = os.cpu_count() or 1
    if args.max_concurrent_jobs == 0:
        args.max_concurrent_jobs = max(1, args.num_cores // (args.n_envs + 1))
    elif args.max_concurrent_jobs < 0:
        args.max_concurrent_jobs = 0  # 0 = unlimited internally

    print("Dedicated exploiter: ", args.dedicated_exploiter)
    print("Number of full exploiters: ", args.num_full_exploiters)
    print("Continue exploiters: ", args.continue_exploiters)
    print("Number of continue exploiters: ", args.num_continue_exploiters)
    print("Number of environments: ", args.n_envs)
    print("Number of cores (this worker): ", args.num_cores)
    print("Max concurrent jobs: ", args.max_concurrent_jobs if args.max_concurrent_jobs > 0 else "unlimited")
    print("Launch local br eval: ", args.launch_local_br_eval)
    if args.exploiter_save_freq * args.n_envs > args.br_training_steps:
        print("WARNING! Exploiter save frequency is greater than BR training steps. This will result in no exploiter checkpoints being saved AND AN ERROR RIGHT BEFORE LOCAL BR EVAL")
    args.eval_only = args.eval_only == 'True'
    if not args.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    else:
        wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    todo_dir = os.path.join(TASK_DIR, "todo")

    if args.task_dir is not None and args.task_dir != "":
        print(f"WARNING: Using custom task directory: {args.task_dir}")
        todo_dir = os.path.join(args.task_dir, "todo")
    this_file_parents = os.path.dirname(os.path.abspath(__file__))
    this_file_task_dir = this_file_parents + "/trained_models/tasks/"
    processing_dir = os.path.join(this_file_task_dir, "processing")

    error_dir = os.path.join(this_file_task_dir, "error")
    done_dir = os.path.join(this_file_task_dir, "done")
    stop_file = os.path.join(this_file_task_dir, "STOP")
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # if os.path.isfile(curr_dir + "/myfile.txt"):
    #     import json
    #     test = json.load(open(curr_dir + "/myfile.txt"))
    #     print(test)
    # else:
    #     print("myfile.txt does not exist")

    print(f"WORKER [{os.getpid()}]: Starting. Watching {todo_dir} for tasks. device={args.device}")
    if (not args.is_league) and args.load_br == 'False':
        while not os.path.exists(stop_file):
            tasks = [f for f in os.listdir(todo_dir) if f.endswith(".task")]

            if not tasks:
                time.sleep(POLL_INTERVAL)
                continue

            # Grab a random task to reduce the chance of multiple workers grabbing the same one
            task_filename = random.choice(tasks)
            todo_path = os.path.join(todo_dir, task_filename)
            processing_path = os.path.join(processing_dir, task_filename)
            error_path = os.path.join(error_dir, task_filename)

            try:
                # Atomically move the task file to claim it
                os.rename(todo_path, processing_path)

                # Per-training-process subfolder for local_br_eval outputs.
                # SPAR runs share TASK_DIR/todo by default, so we derive the
                # run identifier from the .task filename prefix (stripped of
                # the trailing `_<step>_steps` suffix). All checkpoints from
                # the same ippo training run share that prefix → same folder;
                # different runs → different folders.
                output_subdir = _derive_spar_run_subdir(task_filename)

                # Now that we've claimed it, process it
                # if args.num_brs == 1:
                #     loaded_model = load_spar_model(processing_path)
                #     train_best_response(
                #         loaded_model,
                #         processing_path,
                #         eval_prot=args.eval_prot,
                #         use_mirror=args.use_mirror,
                #         eval_only=args.eval_only,
                #         proj_name=args.proj_name,
                #         analysis_upload_proj_name=args.analysis_upload_proj_name,
                #         is_spar=True,
                #         br_index=2,
                #         from_scratch=True,
                #     )
                # else:
                processes = []

                def _launch_job(
                    eval_prot_flag: bool,
                    br_idx: int,
                    from_scratch: bool,
                    state_subset: Optional[List[str]] = None,
                    matchup_label: Optional[str] = None,
                    replicate_idx: Optional[int] = None,
                    dedicated_job_id: Optional[int] = None,
                    launch_local_br_eval: bool = False,
                ) -> None:
                    """
                    Launch one BR subprocess (or run inline in DEBUG mode).

                    Why this helper is nested here:
                    - It captures current task-specific context (e.g. processing_path).
                    - It removes duplicated argument packing across the many loops.
                    - It ensures every launch path (continue + dedicated) uses the
                      same subprocess signature and consistent debug prints.
                    """
                    target = run_br_for_task_in_subprocess
                    task_stem = os.path.splitext(os.path.basename(processing_path))[0]
                    run_mode = "dedicated" if from_scratch else "continue"
                    side_label = "ego" if eval_prot_flag else "adv"
                    if dedicated_job_id is not None:
                        key_base = (
                            f"{task_stem}_{run_mode}_job{dedicated_job_id}_{side_label}_br{br_idx}"
                        )
                    else:
                        matchup_part = (
                            _sanitize_for_filename(str(matchup_label))
                            if matchup_label is not None
                            else "all_matchups"
                        )
                        rep_part = (
                            f"rep{replicate_idx}" if replicate_idx is not None else "repNA"
                        )
                        key_base = (
                            f"{task_stem}_{run_mode}_{side_label}_{matchup_part}_{rep_part}_br{br_idx}"
                        )
                    stop_key = _sanitize_for_filename(key_base)
                    stop_file_dir = os.path.join(TASK_DIR, "stop")
                    os.makedirs(stop_file_dir, exist_ok=True)
                    stop_file_path = os.path.join(stop_file_dir, f"STOP_{stop_key}")
                    training_args = (
                        game_args,
                        processing_path,
                        eval_prot_flag,
                        args.use_mirror,
                        args.eval_only,
                        args.proj_name,
                        args.analysis_upload_proj_name,
                        args.n_envs,
                        True,  # is_spar_like
                        br_idx,
                        from_scratch,
                        args.exploiter_save_freq,
                        args.br_tracker_patience,
                        args.br_tracker_tolerance,
                        args.br_tracker_window_size,
                        args.use_br_reward_stagnation,
                        args.use_br_entropy_stagnation,
                        args.br_use_slope_early_stop,
                        args.br_slope_window,
                        args.br_slope_tolerance,
                        args.br_min_slope_checks,
                        args.use_stagnation_early_stop,
                        args.use_stagnation_velocity_signal,
                        args.use_stagnation_entropy_signal,
                        args.stagnation_patience,
                        args.stagnation_tolerance,
                        args.stagnation_rel_tolerance,
                        args.stagnation_ema_beta,
                        args.stagnation_eps,
                        args.stagnation_eval_games,
                        args.entropy_stagnation_weight,
                        args.stagnation_lr_factor,
                        args.stagnation_lr_patience,
                        args.stagnation_use_slope_early_stop,
                        args.stagnation_slope_window,
                        args.stagnation_slope_tolerance,
                        args.stagnation_min_slope_checks,
                        args.device,
                        state_subset,
                        matchup_label,
                        replicate_idx,
                        dedicated_job_id,
                        stop_file_path,
                        stop_key,
                        launch_local_br_eval,
                        args.use_wandb,
                    )
                    # SPAR tuple stops short of run_br_for_task_in_subprocess's
                    # is_league/league_matchup_states defaults, so we pass the
                    # newer output_subdir as a kwarg rather than filling in
                    # unrelated positional defaults.
                    training_kwargs = {
                        "output_subdir": output_subdir,
                        "entropy_stop_ratio": args.entropy_stop_ratio,
                        "entropy_window_size": args.entropy_window_size,
                        "entropy_warmup_checks": args.entropy_warmup_checks,
                        "entropy_ratio_only": args.entropy_ratio_only == 'True',
                        # Keyword-only required on run_br_for_task_in_subprocess
                        "br_training_steps": args.br_training_steps,
                    }
                    if args.DEBUG:
                        print(
                            "DEBUG: Launching BR job "
                            f"idx={br_idx}, eval_prot_flag={eval_prot_flag}, "
                            f"from_scratch={from_scratch}, matchup={matchup_label}, "
                            f"replicate={replicate_idx}, dedicated_job_id={dedicated_job_id}, "
                            f"launch_local_br_eval={launch_local_br_eval}, "
                            f"stop_key={stop_key}, stop_file={stop_file_path}, "
                            f"output_subdir={output_subdir}"
                        )
                        target(*training_args, **training_kwargs)
                    else:
                        p = mp_ctx.Process(
                            target=target,
                            args=training_args,
                            kwargs=training_kwargs,
                        )
                        p.start()
                        processes.append(p)

                # Continue-exploiter mode (existing behavior): run a fixed number
                # of jobs against the full checkpoint state list.
                if args.continue_exploiters:
                    from_scratch = False
                    if args.eval_prot:
                        for br_idx in range(args.num_continue_exploiters):
                            _launch_job(eval_prot_flag=True, br_idx=br_idx, from_scratch=from_scratch, launch_local_br_eval=args.launch_local_br_eval)
                    if args.eval_adv:
                        # IMPORTANT: eval_adv jobs must set eval_prot_flag=False so
                        # train_best_response configures Exploiter(exploiting='adv').
                        for br_idx in range(args.num_continue_exploiters):
                            _launch_job(eval_prot_flag=False, br_idx=br_idx, from_scratch=from_scratch, launch_local_br_eval=args.launch_local_br_eval)

                # Dedicated mode (new behavior): schedule one job per unique
                # matchup state per side, with `num_full_exploiters` replicates
                # for each matchup+side pair.
                if args.dedicated_exploiter:
                    unique_states = _extract_unique_states_from_task(processing_path, device=args.device)
                    dedicated_specs = _build_dedicated_job_specs(
                        unique_states=unique_states,
                        replicates_per_matchup=args.num_full_exploiters,
                        run_eval_prot=args.eval_prot,
                        run_eval_adv=args.eval_adv,
                        launch_local_br_eval=args.launch_local_br_eval,
                    )
                    max_conc = args.max_concurrent_jobs
                    print(
                        "Dedicated exploiter scheduling:\n"
                        f"  unique_matchups={len(unique_states)}\n"
                        f"  replicates_per_matchup={args.num_full_exploiters}\n"
                        f"  total_jobs={len(dedicated_specs)}\n"
                        f"  max_concurrent_jobs={max_conc if max_conc > 0 else 'unlimited'}"
                    )
                    for spec in dedicated_specs:
                        if max_conc > 0 and not args.DEBUG:
                            processes = _wait_for_slot(processes, max_conc)
                        _launch_job(
                            eval_prot_flag=spec["eval_prot"],
                            br_idx=spec["job_index"],
                            from_scratch=True,
                            state_subset=spec["state_subset"],
                            matchup_label=spec["matchup_label"],
                            replicate_idx=spec["replicate_idx"],
                            dedicated_job_id=spec["job_index"],
                            launch_local_br_eval=args.launch_local_br_eval,
                        )
                        print("Launched dedicated job: ", spec["job_index"])

                # Wait for all BR processes to finish before marking task as done.
                if not args.DEBUG:
                    for p in processes:
                        p.join()

                # Move it to 'done' when finished
                done_path = os.path.join(done_dir, task_filename)
                print("Finished processing task: ", task_filename)
                #os.rename(processing_path, done_path)

            except FileNotFoundError:
                # Another worker grabbed this file first. No problem.
                continue
            except Exception as e:
                print(f"WORKER [{os.getpid()}]: A critical error occurred. Error: {e}")
                # Move the failed task back to todo or to an error folder
                try:
                    os.rename(processing_path, error_path)
                except:
                    pass
    elif args.is_league and args.load_br == 'False':
        print(f"WORKER [{os.getpid()}]: League mode.")
        print("WORKER [%d]: Processing League files. todo dir reset to %s" % (os.getpid(), args.league_dir))
        todo_dir = args.league_dir
        while not os.path.exists(stop_file):
            tasks = [f for f in os.listdir(todo_dir) if f.endswith(".task")]
            league_aux_files = [f for f in os.listdir(todo_dir) if f.endswith(".pt")]

            if not tasks:
                time.sleep(POLL_INTERVAL)
                continue

            task_filename = random.choice(tasks)
            todo_path = os.path.join(todo_dir, task_filename)
            os.makedirs(processing_dir + "/%s_folder" % task_filename, exist_ok=True)
            processing_path = os.path.join(processing_dir + "/%s_folder" % task_filename, task_filename)
            error_path = os.path.join(error_dir, task_filename)

            try:
                # Atomically claim the .task (rename = exclusive ownership).
                os.rename(todo_path, processing_path)
                # Aux .pt siblings (left MA, right MAs, ME, LE snapshots) are
                # COPIED, not renamed, so other workers picking up other
                # .task files in the same league_dir can still find them.
                # Copies live in the per-task processing folder for the rest
                # of this worker's pipeline (state inference, continue-mode
                # checkpoint loading).
                for file in league_aux_files:
                    src = os.path.join(todo_dir, file)
                    dst = os.path.join(processing_dir + "/%s_folder" % task_filename, file)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)

                # ---- Detect which side this task represents ----------------
                # The CLI flags --eval_prot / --eval_adv are intentionally
                # ignored from here on: the loaded checkpoint dictates which
                # side is frozen and therefore which side BR trains. Any CLI
                # mismatch is silently overridden (governing script needs to
                # process both sides for the plots).
                _peek = _load_league_checkpoint(processing_path, device="cpu")
                loaded_task_side = _peek["kwargs"].get("side")
                loaded_matchup_key = _peek["kwargs"].get("matchup_key")

                # Per-training-process subfolder for local_br_eval outputs.
                # The league filename ("MA0_left_m_00_left_vs_all_<step>_<date>_<time>.task")
                # has no per-RUN component — every league spawn produces the
                # same player-name prefix, and the timestamp varies per
                # checkpoint, not per run. So filename-based segregation
                # would either collide (prefix) or fragment per checkpoint
                # (timestamp).
                #
                # The authoritative per-run identifier is the training
                # process's --save-dir, saved on every Player as `args` (see
                # common/league._create_checkpoint). Two parallel league
                # trainings necessarily use different save_dirs, so this
                # differentiates them even when they share a --league_dir
                # for BR queueing.
                _saved_args = _peek["kwargs"].get("args")
                _saved_save_dir = None
                if _saved_args is not None:
                    if isinstance(_saved_args, dict):
                        _saved_save_dir = _saved_args.get("save_dir")
                    else:
                        _saved_save_dir = getattr(_saved_args, "save_dir", None)
                if _saved_save_dir:
                    output_subdir = _sanitize_for_filename(
                        os.path.basename(str(_saved_save_dir).rstrip("/"))
                    )
                else:
                    # Fallback: league_dir basename (works when each run uses
                    # its own --league_dir).
                    output_subdir = _sanitize_for_filename(
                        os.path.basename(args.league_dir.rstrip("/"))
                    )
                if not output_subdir:
                    output_subdir = "unknown_run"
                del _peek
                if loaded_task_side not in ("left", "right"):
                    raise ValueError(
                        f"League task {task_filename!r} has invalid "
                        f"side={loaded_task_side!r}; expected 'left' or 'right'."
                    )
                # Side mapping (mirrors _LeaguePolicyAdapter convention):
                #   loaded left  -> exploit ego, BR plays adv  -> eval_prot=True
                #   loaded right -> exploit adv, BR plays ego  -> eval_prot=False
                eval_prot_for_this_task = (loaded_task_side == "left")
                print(
                    f"WORKER [{os.getpid()}]: League task side={loaded_task_side}, "
                    f"matchup_key={loaded_matchup_key}, "
                    f"output_subdir={output_subdir!r} "
                    f"(saved save_dir={_saved_save_dir!r}), "
                    f"BR will train {'right (adv)' if eval_prot_for_this_task else 'left (ego)'}, "
                    f"eval_prot_for_this_task={eval_prot_for_this_task}"
                )

                if args.league_matchup_states:
                    league_matchup_states = list(args.league_matchup_states)
                else:
                    league_matchup_states = _infer_league_matchup_states_from_dir(processing_path)
                    print(
                        f"WORKER [{os.getpid()}]: Inferred {len(league_matchup_states)} "
                        f"matchup states from sibling right-models:"
                    )

                problematic_names = ["Chunli", "Ehonda", "Mbison"]
                canonical_names = ["ChunLi", "EHonda", "MBison"]
                for i in range(len(league_matchup_states)):
                    for j in range(len(problematic_names)):
                        if problematic_names[j] in league_matchup_states[i]:
                            league_matchup_states[i] = league_matchup_states[i].replace(problematic_names[j], canonical_names[j])

                for s in league_matchup_states:
                    print(f"  - {s}")

                processes = []

                def _launch_league_job(
                    eval_prot_flag: bool,
                    br_idx: int,
                    from_scratch: bool,
                    state_subset: Optional[List[str]] = None,
                    matchup_label: Optional[str] = None,
                    replicate_idx: Optional[int] = None,
                    dedicated_job_id: Optional[int] = None,
                    launch_local_br_eval: bool = False,
                ) -> None:
                    target = run_br_for_task_in_subprocess
                    task_stem = os.path.splitext(os.path.basename(processing_path))[0]
                    run_mode = "dedicated" if from_scratch else "continue"
                    side_label = "ego" if eval_prot_flag else "adv"
                    if dedicated_job_id is not None:
                        key_base = (
                            f"{task_stem}_{run_mode}_job{dedicated_job_id}_{side_label}_br{br_idx}"
                        )
                    else:
                        matchup_part = (
                            _sanitize_for_filename(str(matchup_label))
                            if matchup_label is not None
                            else "all_matchups"
                        )
                        rep_part = (
                            f"rep{replicate_idx}" if replicate_idx is not None else "repNA"
                        )
                        key_base = (
                            f"{task_stem}_{run_mode}_{side_label}_{matchup_part}_{rep_part}_br{br_idx}"
                        )
                    stop_key = _sanitize_for_filename(key_base)
                    stop_file_dir = os.path.join(TASK_DIR, "stop")
                    os.makedirs(stop_file_dir, exist_ok=True)
                    stop_file_path = os.path.join(stop_file_dir, f"STOP_{stop_key}")
                    training_args = (
                        game_args,
                        processing_path,
                        eval_prot_flag,
                        args.use_mirror,
                        args.eval_only,
                        args.proj_name,
                        args.analysis_upload_proj_name,
                        args.n_envs,
                        False,  # is_spar_like = False for league
                        br_idx,
                        from_scratch,
                        args.exploiter_save_freq,
                        args.br_tracker_patience,
                        args.br_tracker_tolerance,
                        args.br_tracker_window_size,
                        args.use_br_reward_stagnation,
                        args.use_br_entropy_stagnation,
                        args.br_use_slope_early_stop,
                        args.br_slope_window,
                        args.br_slope_tolerance,
                        args.br_min_slope_checks,
                        args.use_stagnation_early_stop,
                        args.use_stagnation_velocity_signal,
                        args.use_stagnation_entropy_signal,
                        args.stagnation_patience,
                        args.stagnation_tolerance,
                        args.stagnation_rel_tolerance,
                        args.stagnation_ema_beta,
                        args.stagnation_eps,
                        args.stagnation_eval_games,
                        args.entropy_stagnation_weight,
                        args.stagnation_lr_factor,
                        args.stagnation_lr_patience,
                        args.stagnation_use_slope_early_stop,
                        args.stagnation_slope_window,
                        args.stagnation_slope_tolerance,
                        args.stagnation_min_slope_checks,
                        args.device,
                        state_subset,
                        matchup_label,
                        replicate_idx,
                        dedicated_job_id,
                        stop_file_path,
                        stop_key,
                        launch_local_br_eval,
                        args.use_wandb,
                        True,  # is_league
                        league_matchup_states,
                    )
                    # Pass output_subdir as a kwarg to mirror the SPAR launch
                    # path; keeps the positional tuple stable so future param
                    # additions don't have to backfill defaults.
                    training_kwargs = {
                        "output_subdir": output_subdir,
                        "entropy_stop_ratio": args.entropy_stop_ratio,
                        "entropy_window_size": args.entropy_window_size,
                        "entropy_warmup_checks": args.entropy_warmup_checks,
                        "entropy_ratio_only": args.entropy_ratio_only == 'True',
                        # Keyword-only required on run_br_for_task_in_subprocess
                        "br_training_steps": args.br_training_steps,
                    }
                    if args.DEBUG:
                        print(
                            "DEBUG: Launching league BR job "
                            f"idx={br_idx}, eval_prot_flag={eval_prot_flag}, "
                            f"from_scratch={from_scratch}, matchup={matchup_label}, "
                            f"replicate={replicate_idx}, dedicated_job_id={dedicated_job_id}, "
                            f"launch_local_br_eval={launch_local_br_eval}, "
                            f"stop_key={stop_key}, stop_file={stop_file_path}, "
                            f"output_subdir={output_subdir}"
                        )
                        target(*training_args, **training_kwargs)
                    else:
                        p = mp_ctx.Process(
                            target=target,
                            args=training_args,
                            kwargs=training_kwargs,
                        )
                        p.start()
                        processes.append(p)

                # Continue-mode and dedicated-mode scheduling both use
                # eval_prot_for_this_task (file-derived) instead of the CLI
                # --eval_prot / --eval_adv flags. See the comment above where
                # eval_prot_for_this_task is computed.
                if args.continue_exploiters:
                    for br_idx in range(args.num_continue_exploiters):
                        _launch_league_job(
                            eval_prot_flag=eval_prot_for_this_task,
                            br_idx=br_idx,
                            from_scratch=False,
                            launch_local_br_eval=args.launch_local_br_eval,
                        )

                if args.dedicated_exploiter:
                    unique_states = list(league_matchup_states)
                    # Pass exactly one side as True so _build_dedicated_job_specs
                    # produces the (state, side, replicate) cross-product for
                    # only the side BR should train.
                    dedicated_specs = _build_dedicated_job_specs(
                        unique_states=unique_states,
                        replicates_per_matchup=args.num_full_exploiters,
                        run_eval_prot=eval_prot_for_this_task,
                        run_eval_adv=(not eval_prot_for_this_task),
                        launch_local_br_eval=args.launch_local_br_eval,
                    )
                    max_conc = args.max_concurrent_jobs
                    print(
                        "League dedicated exploiter scheduling:\n"
                        f"  unique_matchups={len(unique_states)}\n"
                        f"  replicates_per_matchup={args.num_full_exploiters}\n"
                        f"  total_jobs={len(dedicated_specs)}\n"
                        f"  max_concurrent_jobs={max_conc if max_conc > 0 else 'unlimited'}"
                    )
                    for spec in dedicated_specs:
                        if max_conc > 0 and not args.DEBUG:
                            processes = _wait_for_slot(processes, max_conc)
                        _launch_league_job(
                            eval_prot_flag=spec["eval_prot"],
                            br_idx=spec["job_index"],
                            from_scratch=True,
                            state_subset=spec["state_subset"],
                            matchup_label=spec["matchup_label"],
                            replicate_idx=spec["replicate_idx"],
                            dedicated_job_id=spec["job_index"],
                            launch_local_br_eval=args.launch_local_br_eval,
                        )
                        print("Launched league dedicated job: ", spec["job_index"])

                if not args.DEBUG:
                    for p in processes:
                        p.join()

                done_path = os.path.join(done_dir, task_filename)
                print("Finished processing league task: ", task_filename)

            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"WORKER [{os.getpid()}]: A critical error occurred. Error: {e}")
                try:
                    os.rename(processing_path, error_path)
                except:
                    pass
    else:
        while not os.path.exists(stop_file):
            tasks = [f for f in os.listdir(BR_MODEL_DIR) if f.endswith(".task")]

            if not tasks:
                time.sleep(POLL_INTERVAL)
                continue
            ep_info_buffers = []
            rew_arr = np.zeros(len(tasks))
            # Grab a random task to reduce the chance of multiple workers grabbing the same one
            for i in range(len(tasks)):
                loaded_model = Exploiter.load(os.path.join(BR_MODEL_DIR, tasks[i]), env=exploiter_env_generator(STATE=STATE))
                ep_info_buffers.append(loaded_model.ep_info_buffer)
                for j in range(len(loaded_model.ep_info_buffer)):
                    rew_arr[i] = rew_arr[i] + loaded_model.ep_info_buffer[j]['r']
                rew_arr[i] = rew_arr[i] / len(loaded_model.ep_info_buffer)
            
            print("hello")
    print(f"WORKER [{os.getpid()}]: Stop file detected. Shutting down.")
