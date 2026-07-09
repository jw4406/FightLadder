"""
Shared helpers for the BR SLURM orchestrators.

Both `br_slurm_orchestrator.py` (dedicated, from-scratch BR) and
`br_continue_slurm_orchestrator.py` (per-matchup continue BR) need the
same machinery for:
  - watching a TODO_DIR and atomically claiming .task files,
  - copying league aux .pt siblings into the per-task processing folder,
  - detecting model type (CDS spar/ippo vs league),
  - deriving the per-training-process output_subdir,
  - building per-job sbatch scripts and submitting them,
  - polling squeue and sweeping completed tasks to done/.

Keeping all of this in one module lets the two orchestrators differ only
in how they build job specs and which single-matchup runner script the
sbatch invokes — anything else (paths, defaults, sweeper cadence) stays
in lockstep without copy-paste drift.

The orchestrators import these helpers; the per-job runners
(br_single_matchup.py / br_single_continue.py) do not need them.
"""
import argparse
import functools
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

# Reuse model/state helpers already exported by new_br_worker.
from new_br_worker import (
    TASK_DIR,
    _derive_spar_run_subdir,
    _extract_unique_states_from_task,
    _infer_cds_architecture,
    _infer_league_matchup_states_from_dir,
    _load_league_checkpoint,
    _sanitize_for_filename,
)
from stable_baselines3.common.save_util import load_from_zip_file


SQUEUE_FORMAT = "%A"  # job id only — `squeue -h -j ... -o %A`
POLL_INTERVAL = 5     # seconds between watchdog iterations

# Map from SBATCH directive names to argparse dest names.
_SBATCH_TO_ARG = {
    "time": "slurm_time",
    "mem": "slurm_mem",
    "mem-per-cpu": "slurm_mem",
    "gres": "slurm_gres",
    "cpus-per-task": "slurm_cpus_per_task",
    "account": "slurm_account",
}


def load_template_config(path: str) -> Dict[str, str]:
    """Parse a .slurm template and return ``{arg_name: value}`` pairs.

    Extracts two kinds of lines:
      - ``#SBATCH --key=value``  (mapped via ``_SBATCH_TO_ARG``)
      - ``KEY="value"`` or ``KEY=value``  (lowercased key becomes the arg name)

    The caller merges the returned dict with argparse defaults so that
    explicit CLI flags still win.
    """
    config: Dict[str, str] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()

            # #SBATCH directives
            m = re.match(r"^#SBATCH\s+--(\S+?)=(.+)$", line)
            if m:
                key = _SBATCH_TO_ARG.get(m.group(1))
                if key:
                    config[key] = m.group(2).strip()
                continue

            # Bash variable assignments: KEY="val" or KEY=val
            m = re.match(r'^([A-Z_][A-Z_0-9]*)="?(.*?)"?\s*$', line)
            if m and not line.startswith("#"):
                val = m.group(2)
                if "{{" not in val:
                    config[m.group(1).lower()] = val

    return config


def apply_template_config(
    args: argparse.Namespace,
    config: Dict[str, str],
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Fill unset argparse attrs from *config*, respecting types."""
    type_map: Dict[str, type] = {}
    defaults: Dict[str, Any] = {}
    for action in parser._actions:
        if action.dest:
            if action.type:
                type_map[action.dest] = action.type
            defaults[action.dest] = action.default

    for key, val in config.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) != defaults.get(key):
            continue
        cast = type_map.get(key)
        try:
            setattr(args, key, cast(val) if cast else val)
        except (ValueError, TypeError):
            setattr(args, key, val)
    return args


# ----------------------------- bool argparse -----------------------------
def bool_choice(s: str) -> bool:
    return str(s).lower() in ("true", "1", "yes")


# ----------------------------- model-type detection -----------------------------
def detect_model_type(model_path: str, device: str) -> str:
    """
    Return one of {"spar", "ippo", "league"} for *model_path*.

    SB3 zip is tried first (covers both CDS arches; _infer_cds_architecture
    decides between them). Falls back to torch.load + cls_name presence for
    league. Raises on anything unrecognized so the orchestrator surfaces a
    clear error rather than silently dispatching the wrong loader.
    """
    try:
        data, _, _ = load_from_zip_file(model_path, device=device)
    except Exception:
        data = None
    if data is not None:
        return _infer_cds_architecture(data, model_path)

    try:
        loaded = _load_league_checkpoint(model_path, device=device)
    except Exception as exc:
        raise ValueError(
            f"Could not detect model type for {model_path!r}: not a SB3 zip "
            f"and not a torch-saved league checkpoint ({exc})."
        )
    if isinstance(loaded, dict) and "cls_name" in loaded:
        return "league"
    raise ValueError(
        f"Could not detect model type for {model_path!r}: file loaded but "
        "lacks the league 'cls_name' marker."
    )


# ----------------------------- output_subdir derivation -----------------------------
def derive_output_subdir_for_league(model_path: str, fallback_dir: str) -> str:
    """
    Mirror new_br_worker.py's league-branch logic: read save_dir from the
    saved kwargs `args` (Player.args set during league training); use its
    basename as the per-training-process subdir. Falls back to *fallback_dir*
    basename when save_dir isn't recoverable.
    """
    saved = _load_league_checkpoint(model_path, device="cpu")
    saved_args = saved.get("kwargs", {}).get("args")
    saved_save_dir = None
    if saved_args is not None:
        if isinstance(saved_args, dict):
            saved_save_dir = saved_args.get("save_dir")
        else:
            saved_save_dir = getattr(saved_args, "save_dir", None)
    if saved_save_dir:
        return _sanitize_for_filename(
            os.path.basename(str(saved_save_dir).rstrip("/"))
        ) or "unknown_run"
    return (
        _sanitize_for_filename(os.path.basename(fallback_dir.rstrip("/")))
        or "unknown_run"
    )


def derive_output_subdir(model_path: str, model_type: str, league_dir: str) -> str:
    """Dispatch the SPAR vs league subdir-derivation rule."""
    if model_type == "league":
        return derive_output_subdir_for_league(model_path, fallback_dir=league_dir)
    return _derive_spar_run_subdir(os.path.basename(model_path)) or "unknown_run"


# ----------------------------- league side / matchup_key peek -----------------------------
def peek_league_side_and_matchup(model_path: str) -> Tuple[Optional[str], Optional[str]]:
    saved = _load_league_checkpoint(model_path, device="cpu")
    return saved.get("kwargs", {}).get("side"), saved.get("kwargs", {}).get("matchup_key")


# ----------------------------- sbatch script generation -----------------------------
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output={out_log}
#SBATCH --error={err_log}
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jw4406@princeton.edu
{extra_sbatch_lines}
set -euo pipefail
cd {repo_dir}
{data_dir_exports}{env_setup}
{python_cmd}
"""


def build_python_cmd(
    *,
    python_bin: str,
    runner_script: str,
    task_file: str,
    local_plot_dir: str,
    state: str,
    eval_prot: bool,
    replicate_idx: int,
    br_index: int,
    dedicated_job_id: int,
    matchup_label: Optional[str],
    output_subdir: str,
    training_style: str,
    is_league: bool,
    league_matchup_states: Optional[List[str]],
    shared_config_json: str,
) -> str:
    """
    Build the bash snippet that the sbatch script uses to invoke the
    per-job runner. Returns a multi-line ``CMD=( ... )`` array followed by
    ``"${CMD[@]}"`` so the rendered .sbatch is human-inspectable: each
    --flag and its value land on their own indented line, with
    ``--league_matchup_states`` keeping all its nargs="+" values together
    on one line.

    All values are shlex.quoted so the JSON blob in
    ``--shared_config_json`` and any state strings with embedded chars
    survive intact through the shell.

    *runner_script* is the per-job script the orchestrator wants to invoke
    (br_single_matchup.py for dedicated, br_single_continue.py for
    continue). Both runners share the same CLI surface for these args.
    """
    # Each entry is (flag, [values]). Values are kept as a list so flags
    # with nargs="+" (currently just --league_matchup_states) render with
    # all their values on one line — preserving the "one logical flag per
    # line" UX while still being a valid bash array element.
    flag_groups: List[Tuple[str, List[str]]] = [
        ("--task_file", [task_file]),
        ("--local_plot_dir", [local_plot_dir]),
        ("--state", [state]),
        ("--eval_prot", ["True" if eval_prot else "False"]),
        ("--replicate_idx", [str(replicate_idx)]),
        ("--br_index", [str(br_index)]),
        ("--dedicated_job_id", [str(dedicated_job_id)]),
        ("--output_subdir", [output_subdir]),
        ("--training_style", [training_style]),
        ("--is_league", ["True" if is_league else "False"]),
    ]
    if matchup_label is not None:
        flag_groups.append(("--matchup_label", [matchup_label]))
    if is_league and league_matchup_states:
        flag_groups.append(("--league_matchup_states", list(league_matchup_states)))
    flag_groups.append(("--shared_config_json", [shared_config_json]))

    lines = ["CMD=("]
    # Header line: python interpreter + runner script. Both quoted so a
    # path containing spaces wouldn't break the array.
    lines.append(f"    {shlex.quote(python_bin)} {shlex.quote(runner_script)}")
    for flag, values in flag_groups:
        quoted_vals = " ".join(shlex.quote(v) for v in values)
        lines.append(f"    {flag} {quoted_vals}")
    lines.append(")")
    lines.append('"${CMD[@]}"')
    return "\n".join(lines)


def write_sbatch_script(
    *,
    sbatch_path: str,
    job_name: str,
    time_limit: str,
    mem: str,
    gres: str,
    cpus_per_task: int,
    out_log: str,
    err_log: str,
    repo_dir: str,
    env_setup: str,
    python_cmd: str,
    extra_sbatch_lines: str,
    workdir: str = "",
    main_training_dir: str = "",
) -> None:
    if workdir and main_training_dir:
        data_dir_exports = (
            f"export WORKDIR={shlex.quote(workdir)}\n"
            f"export MAIN_TRAINING_DIR={shlex.quote(main_training_dir)}\n"
        )
    else:
        data_dir_exports = ""
    contents = SBATCH_TEMPLATE.format(
        job_name=job_name,
        time_limit=time_limit,
        mem=mem,
        gres=gres,
        cpus_per_task=cpus_per_task,
        out_log=out_log,
        err_log=err_log,
        extra_sbatch_lines=extra_sbatch_lines,
        repo_dir=repo_dir,
        data_dir_exports=data_dir_exports,
        env_setup=env_setup,
        python_cmd=python_cmd,
    )
    with open(sbatch_path, "w") as f:
        f.write(contents)
    os.chmod(sbatch_path, 0o755)


def render_template_sbatch(
    *,
    template_text: str,
    sbatch_path: str,
    job_name: str,
    out_log: str,
    err_log: str,
    python_cmd: str,
    extra_sbatch_lines: str = "",
    extra_placeholders: Optional[Dict[str, str]] = None,
) -> None:
    """Render a .slurm template by substituting ``{{PLACEHOLDERS}}``.

    *extra_placeholders* lets callers inject additional ``{{KEY}}`` -> value
    substitutions on top of the always-substituted JOB_NAME/OUT_LOG/ERR_LOG/
    PYTHON_CMD. Used by the workstation ws_launch_files/*.sh templates to
    receive WS_WORKDIR / MAIN_TRAINING_DIR / WS_REPO_DIR from the
    orchestrator. SLURM .slurm templates that don't contain those markers
    are unaffected.
    """
    subs = {
        "JOB_NAME": job_name,
        "OUT_LOG": out_log,
        "ERR_LOG": err_log,
        "PYTHON_CMD": python_cmd,
    }
    if extra_placeholders:
        subs.update(extra_placeholders)
    rendered = template_text
    for key, val in subs.items():
        rendered = rendered.replace("{{" + key + "}}", val)

    if extra_sbatch_lines:
        rendered = rendered.replace(
            "#SBATCH --mail-user=",
            extra_sbatch_lines + "#SBATCH --mail-user=",
        )

    with open(sbatch_path, "w") as f:
        f.write(rendered)
    os.chmod(sbatch_path, 0o755)


@functools.lru_cache(maxsize=1)
def have_sbatch() -> bool:
    """
    True if `sbatch` is on PATH (i.e. we're on a SLURM cluster). Cached so
    we don't re-shell on every spec submission.
    """
    return shutil.which("sbatch") is not None


def submit_sbatch(
    sbatch_path: str,
    dry_run: bool,
    *,
    local_out_log: Optional[str] = None,
    local_err_log: Optional[str] = None,
) -> Optional[str]:
    """
    Submit *sbatch_path*. Behavior depends on environment:

      - dry_run=True              -> print what would happen, return None.
      - sbatch on PATH            -> `sbatch <path>`, return parsed job id
                                     ("Submitted batch job <id>").
      - sbatch NOT on PATH        -> local fallback: spawn `bash <path>` as
                                     a background subprocess. SBATCH
                                     directives are #-comments to bash so
                                     they're harmless. stdout/stderr are
                                     redirected to *local_out_log* and
                                     *local_err_log* (the same paths the
                                     #SBATCH --output/--error directives
                                     would have used). Returns
                                     ``f"local-{pid}"`` so the sweeper can
                                     distinguish local PIDs from slurm
                                     job ids.

    Returns None on parse failure (sbatch path) or when `dry_run=True`.
    Raises subprocess.CalledProcessError if sbatch itself fails.
    """
    if dry_run:
        print(f"[dry_run] would sbatch: {sbatch_path}")
        return None

    if have_sbatch():
        out = subprocess.check_output(["sbatch", sbatch_path], text=True).strip()
        m = re.search(r"Submitted batch job (\d+)", out)
        if m is None:
            print(f"[warn] sbatch returned unexpected output: {out!r}")
            return None
        return m.group(1)

    # Local fallback: not on a SLURM cluster.
    out_f = open(local_out_log, "w") if local_out_log else None
    err_f = open(local_err_log, "w") if local_err_log else None
    try:
        proc = subprocess.Popen(
            ["bash", sbatch_path],
            stdout=out_f,
            stderr=err_f,
        )
    finally:
        # Close parent-side handles; the child has already dup'd them.
        if out_f is not None:
            out_f.close()
        if err_f is not None:
            err_f.close()
    print(
        f"[local] sbatch not on PATH; ran {sbatch_path} as bash "
        f"(pid={proc.pid}, out={local_out_log}, err={local_err_log})"
    )
    return f"local-{proc.pid}"


# ----------------------------- squeue polling -----------------------------
def _local_pid_alive(pid: int) -> bool:
    """
    "Is this PID still doing work?" test for the local-bash fallback.

    Plain ``os.kill(pid, 0)`` is not enough — it returns success for
    *zombie* children (processes that have exited but haven't been
    reaped by the parent yet), which would make the sweeper never
    advance a finished local job to done/. We handle three cases:

      1) PID is a zombie/exited child of THIS orchestrator
         -> ``os.waitpid(pid, WNOHANG)`` returns (pid, status), reaping
            it. Treat as not alive.
      2) PID is a running child of THIS orchestrator
         -> ``waitpid`` returns (0, 0). Treat as alive.
      3) PID was reparented (orchestrator restarted between dispatch
         and sweep, so we're no longer the parent)
         -> ``waitpid`` raises ``ChildProcessError``. Fall back to
            ``/proc/<pid>/status``: if it's missing or in state ``Z``,
            treat as not alive; otherwise alive. Final fallback is
            ``os.kill(pid, 0)``, which still has the zombie issue but
            covers non-Linux hosts.

    Note: PIDs can theoretically be reused after they're reaped. For
    dev-shell use this is fine. For long-lived shared machines, store a
    process-start-time fingerprint at submit time and validate here.
    """
    # Case 1+2: try to non-blockingly reap. If it works, the child is
    # done; if WNOHANG returns 0, the child is still running.
    try:
        result_pid, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        result_pid = None  # Not our child (or already reaped) — fall through.
    except OSError:
        result_pid = None
    else:
        if result_pid == pid:
            return False
        if result_pid == 0:
            return True

    # Case 3: orchestrator restart or non-child PID. Inspect /proc.
    proc_status = f"/proc/{pid}/status"
    try:
        with open(proc_status, "r") as f:
            for line in f:
                if line.startswith("State:"):
                    state_char = line.split()[1] if len(line.split()) >= 2 else ""
                    return state_char != "Z"
        return True  # State: line not found but file exists -> still there
    except FileNotFoundError:
        return False
    except OSError:
        pass

    # Last resort (non-Linux). Has the zombie blind-spot but better than
    # raising in the sweeper hot path.
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def still_running_job_ids(job_ids: List[str]) -> List[str]:
    """
    Return the subset of *job_ids* still active (queued/running for
    SLURM, alive PID for local bash fallback).

    Job ids prefixed ``"local-"`` are treated as local PIDs and probed
    via :func:`_local_pid_alive`. Everything else is queried via
    ``squeue``; if `squeue` is unreachable (not on PATH, or returns
    non-zero because none of the ids are found) we treat the slurm
    bucket as "all done" — same semantics as before.
    """
    if not job_ids:
        return []

    # Partition the id list. "local-<pid>" -> local PID; otherwise SLURM.
    local_pids: List[int] = []
    slurm_ids: List[str] = []
    for jid in job_ids:
        s = str(jid)
        if s.startswith("local-"):
            try:
                local_pids.append(int(s[len("local-"):]))
            except ValueError:
                # Malformed id; treat as done so we don't block the sweeper.
                continue
        else:
            slurm_ids.append(s)

    still: List[str] = []

    # Local PIDs.
    for pid in local_pids:
        if _local_pid_alive(pid):
            still.append(f"local-{pid}")

    # SLURM ids.
    if slurm_ids:
        try:
            out = subprocess.check_output(
                ["squeue", "-h", "-j", ",".join(slurm_ids), "-o", SQUEUE_FORMAT],
                text=True,
            ).strip()
            if out:
                still.extend(line.strip() for line in out.splitlines() if line.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            # squeue absent (non-cluster) or returned non-zero (none of the
            # ids are queued). In either case treat as "all done".
            pass

    return still


# ----------------------------- task pickup + processing -----------------------------
_STEP_RE = re.compile(r"_(\d+)_steps\.task$")


def _extract_step(filename: str) -> Optional[int]:
    """Return the step count embedded in a task filename, or None."""
    m = _STEP_RE.search(filename)
    return int(m.group(1)) if m else None


def claim_task(
    todo_dir: str,
    processing_dir: str,
    step_stride: int = 0,
) -> Optional[Tuple[str, str, str]]:
    """
    Grab a random `.task` file from *todo_dir*, atomically rename it into
    *processing_dir*/<task_filename>_folder/<task_filename>. Returns
    (task_filename, processing_path, processing_folder) on success, None
    when the dir is empty or another worker beat us to it.

    If *step_stride* > 0, only tasks whose step count is divisible by
    *step_stride* are eligible.  Tasks that don't match the stride are
    left in todo_dir for a future run with a different stride (or stride 0).

    The "_folder" wrapper exists so we can also stash per-task state
    (e.g. _jobs.json) alongside the .task without polluting processing_dir.
    """
    tasks = [f for f in os.listdir(todo_dir) if f.endswith(".task")]
    if step_stride > 0:
        tasks = [f for f in tasks
                 if (s := _extract_step(f)) is not None and s % step_stride == 0]
    if not tasks:
        return None
    task_filename = random.choice(tasks)
    todo_path = os.path.join(todo_dir, task_filename)
    folder = os.path.join(processing_dir, f"{task_filename}_folder")
    os.makedirs(folder, exist_ok=True)
    processing_path = os.path.join(folder, task_filename)
    # Defensive: if the destination already exists, the task is already in
    # flight (sweeper hasn't moved it to done yet). Re-claiming would either
    # double-submit (fresh inodes) or no-op (same inode — POSIX rename(2)
    # is a no-op when src/dst are hard links of the same file, which leaves
    # todo_path in place and causes the orchestrator to spin re-claiming
    # forever). Refuse the claim; the orchestrator will pick another task
    # or sleep until this one is swept.
    if os.path.exists(processing_path):
        return None
    try:
        os.rename(todo_path, processing_path)
    except FileNotFoundError:
        return None
    return task_filename, processing_path, folder


def copy_aux_files(src_dir: str, dst_dir: str) -> None:
    """
    Copy `.pt` siblings (left MA, right MAs, ME, LE snapshots) into the
    per-task processing folder. Same convention as new_br_worker.py's
    league branch — copy not move so other watchers picking up other
    .task files in the same source dir still see the .pt files.
    """
    for name in os.listdir(src_dir):
        if not name.endswith(".pt"):
            continue
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(dst):
            continue
        shutil.copy2(src, dst)


# ----------------------------- shared config builder -----------------------------
def build_shared_config(args: argparse.Namespace, manual_stop_file: Optional[str]) -> Dict[str, Any]:
    """
    Pack everything that's identical across this task's per-spec jobs into
    one dict that gets JSON-encoded and passed via --shared_config_json.

    The runner scripts (br_single_matchup.py / br_single_continue.py)
    each read this dict and forward its contents to
    run_br_for_task_in_subprocess as kwargs.
    """
    game_args = {
        "reset": args.reset,
        "side": args.side,
        "render": args.render,
        "enable_combo": args.enable_combo,
        "null_combo": args.null_combo,
        "transform_action": args.transform_action,
        "seed": args.seed,
    }
    return {
        "game_args": game_args,
        "use_mirror": args.use_mirror,
        "eval_only": args.eval_only,
        "proj_name": args.proj_name,
        "analysis_upload_proj_name": args.analysis_upload_proj_name,
        "n_envs": args.n_envs,
        "exploiter_save_freq": args.exploiter_save_freq,
        "br_training_steps": args.br_training_steps,
        "br_tracker_patience": args.br_tracker_patience,
        "br_tracker_tolerance": args.br_tracker_tolerance,
        "br_tracker_window_size": args.br_tracker_window_size,
        "use_br_reward_stagnation": args.use_br_reward_stagnation,
        "use_br_entropy_stagnation": args.use_br_entropy_stagnation,
        "br_use_slope_early_stop": args.br_use_slope_early_stop,
        "br_slope_window": args.br_slope_window,
        "br_slope_tolerance": args.br_slope_tolerance,
        "br_min_slope_checks": args.br_min_slope_checks,
        "use_stagnation_early_stop": args.use_stagnation_early_stop,
        "use_stagnation_velocity_signal": args.use_stagnation_velocity_signal,
        "use_stagnation_entropy_signal": args.use_stagnation_entropy_signal,
        "stagnation_patience": args.stagnation_patience,
        "stagnation_tolerance": args.stagnation_tolerance,
        "stagnation_rel_tolerance": args.stagnation_rel_tolerance,
        "stagnation_ema_beta": args.stagnation_ema_beta,
        "stagnation_eps": args.stagnation_eps,
        "stagnation_eval_games": args.stagnation_eval_games,
        "entropy_stagnation_weight": args.entropy_stagnation_weight,
        "stagnation_lr_factor": args.stagnation_lr_factor,
        "stagnation_lr_patience": args.stagnation_lr_patience,
        "stagnation_use_slope_early_stop": args.stagnation_use_slope_early_stop,
        "stagnation_slope_window": args.stagnation_slope_window,
        "stagnation_slope_tolerance": args.stagnation_slope_tolerance,
        "stagnation_min_slope_checks": args.stagnation_min_slope_checks,
        "entropy_stop_ratio": args.entropy_stop_ratio,
        "entropy_window_size": args.entropy_window_size,
        "entropy_warmup_checks": args.entropy_warmup_checks,
        "entropy_ratio_only": args.entropy_ratio_only == "True",
        "device": args.device,
        "manual_stop_file": manual_stop_file,
        "manual_stop_key": None,
        "launch_local_br_eval": args.launch_local_br_eval,
        "periodic_eval_freq": args.periodic_eval_freq,
        "enable_local_kl_plot": args.enable_local_kl_plot,
        "use_wandb": args.use_wandb,
    }


# ----------------------------- async sweeper -----------------------------
def registry_path(processing_folder: str) -> str:
    return os.path.join(processing_folder, "_jobs.json")


def write_registry(processing_folder: str, payload: Dict[str, Any]) -> None:
    with open(registry_path(processing_folder), "w") as f:
        json.dump(payload, f, indent=2)


def read_registry(processing_folder: str) -> Optional[Dict[str, Any]]:
    path = registry_path(processing_folder)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _resolve_unique_dst(done_dir: str, entry: str) -> str:
    """
    Pick a destination path under *done_dir* that doesn't exist yet.

    The orchestrator can legitimately process two tasks with the *same*
    filename across separate runs (main training regenerates a task
    with the same step number, or the user re-drops the same .task).
    `os.rename` of a directory onto a non-empty existing directory fails
    with ENOTEMPTY, so we append a timestamp suffix on collision and
    keep both runs' artifacts side-by-side.
    """
    base = os.path.join(done_dir, entry)
    if not os.path.exists(base):
        return base
    ts = time.strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(done_dir, f"{entry}_{ts}")
    n = 0
    while os.path.exists(candidate):
        n += 1
        candidate = os.path.join(done_dir, f"{entry}_{ts}_{n}")
    return candidate


def sweep_completed_tasks(processing_dir: str, done_dir: str) -> int:
    """
    Walk processing_dir for per-task folders, check each registry's job ids
    via squeue, and when a task has zero still-running ids move its task
    folder to done_dir. Returns the count of tasks moved this sweep.
    """
    if not os.path.isdir(processing_dir):
        return 0
    moved = 0
    for entry in sorted(os.listdir(processing_dir)):
        folder = os.path.join(processing_dir, entry)
        if not os.path.isdir(folder) or not entry.endswith("_folder"):
            continue
        registry = read_registry(folder)
        if registry is None:
            continue
        job_ids = list(registry.get("job_ids", []))
        if not job_ids:
            # Empty registry (e.g. dry_run) — leave it; the user can move
            # the folder by hand or rerun.
            continue
        running = still_running_job_ids(job_ids)
        if running:
            continue
        dst_folder = _resolve_unique_dst(done_dir, entry)
        try:
            os.rename(folder, dst_folder)
            moved += 1
            note = (
                "" if dst_folder == os.path.join(done_dir, entry)
                else f" (collision: appended timestamp suffix to {os.path.basename(dst_folder)})"
            )
            print(f"[orch] task complete (all {len(job_ids)} jobs cleared "
                  f"from squeue) -> moved {entry} to {done_dir}{note}")
        except OSError as exc:
            print(f"[orch] could not move {folder} -> {dst_folder}: {exc}")
    return moved


# ----------------------------- shared CLI builder -----------------------------
def add_shared_arguments(parser: argparse.ArgumentParser, *, default_processing_subdir: str,
                         default_done_subdir: str, default_stop_file: str) -> None:
    """
    Register the CLI surface that's identical across both orchestrators.

    The orchestrator-specific bits (single-job runner script path, default
    NUM_*_EXPLOITERS) live in each orchestrator's own parser. Defaults for
    the processing/done dirs are passed in so the two orchestrators can
    coexist on the same TODO_DIR without claiming each other's tasks.
    """
    # Watchdog / paths
    parser.add_argument("--todo_dir", type=str, required=True,
                        help="Directory to watch for incoming .task files.")
    parser.add_argument("--processing_dir", type=str,
                        default=os.path.join(TASK_DIR, default_processing_subdir),
                        help="Per-task working dir (atomic claim target).")
    parser.add_argument("--done_dir", type=str,
                        default=os.path.join(TASK_DIR, default_done_subdir),
                        help="Where finished tasks are moved by the sweeper.")
    parser.add_argument("--slurm_log_dir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "slurm_logs",
                        ),
                        help="Per-task subdir is created here for sbatch "
                             "scripts and stdout/stderr logs.")
    parser.add_argument("--stop_file", type=str,
                        default=os.path.join(TASK_DIR, default_stop_file),
                        help="Touch this file to stop the orchestrator "
                             "watchdog (in-flight SLURM jobs are unaffected).")
    parser.add_argument("--step_stride", type=int, default=0,
                        help="Only process tasks whose step count is "
                             "divisible by this value. 0 = process all.")
    parser.add_argument("--max_local_concurrent", type=int, default=1,
                        help="Cap on simultaneous local-bash jobs (workstation "
                             "mode). Only consulted when sbatch is NOT on PATH; "
                             "ignored on SLURM clusters. Default 1.")

    # Job-side parity with new_br_worker.
    parser.add_argument("--eval_prot", choices=["True", "False"], default="True")
    parser.add_argument("--eval_adv", choices=["True", "False"], default="True")
    parser.add_argument("--eval_only", choices=["True", "False"], default="False")
    parser.add_argument("--proj_name", type=str, default="br_training")
    parser.add_argument("--analysis_upload_proj_name", type=str, default="br_analysis")
    parser.add_argument("--use_mirror", choices=["True", "False"], default="False")
    parser.add_argument("--n_envs", type=int, default=2)
    parser.add_argument("--exploiter_save_freq", type=int, default=100000)
    parser.add_argument("--br_training_steps", type=int, required=True,
                        help="Total timesteps each BR job trains for. "
                             "Required at the launcher level; raises "
                             "argparse error if not passed.")

    # BR convergence tracker
    parser.add_argument("--br_tracker_patience", type=int, default=300)
    parser.add_argument("--br_tracker_tolerance", type=float, default=1e-4)
    parser.add_argument("--br_tracker_window_size", type=int, default=50)
    parser.add_argument("--use_br_reward_stagnation", choices=["True", "False"], default="False")
    parser.add_argument("--use_br_entropy_stagnation", choices=["True", "False"], default="True")
    parser.add_argument("--br_use_slope_early_stop", choices=["True", "False"], default="False")
    parser.add_argument("--br_slope_window", type=int, default=20)
    parser.add_argument("--br_slope_tolerance", type=float, default=5e-3)
    parser.add_argument("--br_min_slope_checks", type=int, default=12)

    # Continue-mode CDS stagnation knobs
    parser.add_argument("--use_stagnation_early_stop", choices=["True", "False"], default="True")
    parser.add_argument("--use_stagnation_velocity_signal", choices=["True", "False"], default="False")
    parser.add_argument("--use_stagnation_entropy_signal", choices=["True", "False"], default="True")
    parser.add_argument("--stagnation_patience", type=int, default=2000)
    parser.add_argument("--stagnation_tolerance", type=float, default=1e-4)
    parser.add_argument("--stagnation_rel_tolerance", type=float, default=0.05)
    parser.add_argument("--stagnation_ema_beta", type=float, default=0.99)
    parser.add_argument("--stagnation_eps", type=float, default=1e-8)
    parser.add_argument("--stagnation_eval_games", type=int, default=20)
    parser.add_argument("--entropy_stagnation_weight", type=float, default=1.0)
    parser.add_argument("--stagnation_lr_factor", type=float, default=0.999)
    parser.add_argument("--stagnation_lr_patience", type=int, default=1000)
    parser.add_argument("--stagnation_use_slope_early_stop", choices=["True", "False"], default="False")
    parser.add_argument("--stagnation_slope_window", type=int, default=20)
    parser.add_argument("--stagnation_slope_tolerance", type=float, default=5e-3)
    parser.add_argument("--stagnation_min_slope_checks", type=int, default=12)

    # Entropy-window early-stop
    parser.add_argument("--entropy_stop_ratio", type=float, default=0.15)
    parser.add_argument("--entropy_window_size", type=int, default=50)
    parser.add_argument("--entropy_warmup_checks", type=int, default=100)
    parser.add_argument("--entropy_ratio_only", choices=["True", "False"], default="False")

    # Game args
    parser.add_argument("--reset", choices=["round", "match", "game"], default="round")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--render", choices=["True", "False"], default="False")
    parser.add_argument("--enable_combo", choices=["True", "False"], default="True")
    parser.add_argument("--null_combo", choices=["True", "False"], default="False")
    parser.add_argument("--transform_action", choices=["True", "False"], default="False")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--launch_local_br_eval", choices=["True", "False"], default="True")
    parser.add_argument(
        "--periodic_eval_freq",
        type=int,
        default=5_000_000,
        help="Env-steps between mid-training local_br_eval snapshots "
             "fired by PeriodicLocalBREvalCallback. The suffixed .txt "
             "files (brstep<N>_<timestamp>) survive a crashed run so "
             "you don't lose all eval data. Only takes effect when "
             "--launch_local_br_eval is True.",
    )
    parser.add_argument("--use_wandb", choices=["True", "False"], default="False")
    parser.add_argument("--enable_local_kl_plot", choices=["True", "False"], default="True",
                        help="Whether the BR Exploiter tracker writes "
                             "per-job KL-divergence CSV+PNG plots into "
                             "--local_plot_dir. Default True (preserves "
                             "existing behavior).")

    # SLURM
    #parser.add_argument("--slurm_partition", type=str, default="gpu")
    parser.add_argument("--slurm_time", type=str, default="12:00:00")
    parser.add_argument("--slurm_mem", type=str, default="16G")
    parser.add_argument("--slurm_gres", type=str, default="gpu:1")
    parser.add_argument("--slurm_cpus_per_task", type=int, default=4)
    parser.add_argument("--slurm_account", type=str, default="")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--env_setup", type=str, default="")

    # Data directory (forwarded to sbatch so workers save to $WORKDIR/$MAIN_TRAINING_DIR)
    parser.add_argument("--workdir", type=str,
                        default=os.environ.get("WORKDIR", ""))
    parser.add_argument("--main_training_dir", type=str,
                        default=os.environ.get("MAIN_TRAINING_DIR", ""))

    # Behavior
    parser.add_argument("--dry_run", choices=["True", "False"], default="False")


def normalize_bool_args(args: argparse.Namespace) -> argparse.Namespace:
    """Convert all string-bool CLI args to real bools in-place."""
    bool_fields = [
        "eval_prot", "eval_adv", "eval_only", "use_mirror",
        "use_br_reward_stagnation", "use_br_entropy_stagnation",
        "br_use_slope_early_stop",
        "use_stagnation_early_stop", "use_stagnation_velocity_signal",
        "use_stagnation_entropy_signal", "stagnation_use_slope_early_stop",
        "render", "enable_combo", "null_combo", "transform_action",
        "launch_local_br_eval", "use_wandb", "dry_run",
        "enable_local_kl_plot",
    ]
    for k in bool_fields:
        setattr(args, k, bool_choice(getattr(args, k)))
    return args
