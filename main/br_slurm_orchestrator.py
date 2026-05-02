"""
SLURM orchestrator for dedicated BR exploiter jobs.

Watches a TASK_DIR for `.task` files (mirrors new_br_worker.py's polling
pattern). When a task arrives:
  1. Atomically claims it (rename to processing/).
  2. Detects the model type (CDS spar/ippo or league).
  3. Computes per-training-process output_subdir + training_style.
  4. Builds the dedicated job specs via `_build_dedicated_job_specs` —
     one spec per (state, side, replicate).
  5. For each spec:
       - Writes a per-job .sbatch file under
         <slurm_log_dir>/<task_stem>/<spec>.sbatch.
       - Submits via `sbatch <path>` and records the returned job id.
  6. Records `[<job_ids>]` in `processing/<task>_folder/_jobs.json`.

ASYNC mode: after submission the orchestrator does NOT block. On every
subsequent loop iteration it sweeps `processing/`, checks each registered
task's job ids via `squeue -h -j <ids> -o %A`, and when none of the ids
remain in the queue moves the .task and aux files to `done/`.

Continue-mode is intentionally NOT supported here — only dedicated. Run
`new_br_worker.py` separately for continue exploiters if needed.
"""
import argparse
import json
import os
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

socket.setdefaulttimeout(None)

# Reuse helpers from new_br_worker / br_preflight rather than duplicating.
from new_br_worker import (
    DONE_DIR,
    TASK_DIR,
    _derive_spar_run_subdir,
    _extract_unique_states_from_task,
    _infer_cds_architecture,
    _infer_league_matchup_states_from_dir,
    _load_league_checkpoint,
    _sanitize_for_filename,
)
from br_preflight import build_dedicated_job_specs
from utils import state2matchup
from stable_baselines3.common.save_util import load_from_zip_file


POLL_INTERVAL = 5  # seconds between watchdog iterations
SQUEUE_FORMAT = "%A"  # job id only — `squeue -h -j ... -o %A`


# ----------------------------- model-type detection -----------------------------
def _detect_model_type(model_path: str, device: str) -> str:
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
def _derive_output_subdir_for_league(model_path: str, fallback_dir: str) -> str:
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


def _derive_output_subdir(model_path: str, model_type: str, league_dir: str) -> str:
    """Dispatch the SPAR vs league subdir-derivation rule."""
    if model_type == "league":
        return _derive_output_subdir_for_league(model_path, fallback_dir=league_dir)
    # CDS / SPAR: use the .task filename prefix.
    return _derive_spar_run_subdir(os.path.basename(model_path)) or "unknown_run"


# ----------------------------- league side / matchup_key peek -----------------------------
def _peek_league_side_and_matchup(model_path: str) -> Tuple[Optional[str], Optional[str]]:
    saved = _load_league_checkpoint(model_path, device="cpu")
    return saved.get("kwargs", {}).get("side"), saved.get("kwargs", {}).get("matchup_key")


# ----------------------------- sbatch script generation -----------------------------
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output={out_log}
#SBATCH --error={err_log}
{extra_sbatch_lines}
set -euo pipefail
cd {repo_dir}
{env_setup}
{python_cmd}
"""


def _build_python_cmd(
    *,
    python_bin: str,
    single_matchup_script: str,
    task_file: str,
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
    Build the python invocation line for the sbatch script. shlex.quote
    everything that might contain spaces/special chars so the JSON blob
    survives intact through the shell.
    """
    parts = [
        python_bin,
        single_matchup_script,
        "--task_file", task_file,
        "--state", state,
        "--eval_prot", "True" if eval_prot else "False",
        "--replicate_idx", str(replicate_idx),
        "--br_index", str(br_index),
        "--dedicated_job_id", str(dedicated_job_id),
        "--output_subdir", output_subdir,
        "--training_style", training_style,
        "--is_league", "True" if is_league else "False",
    ]
    if matchup_label is not None:
        parts.extend(["--matchup_label", matchup_label])
    if is_league and league_matchup_states:
        parts.append("--league_matchup_states")
        parts.extend(league_matchup_states)
    parts.extend(["--shared_config_json", shared_config_json])
    return " ".join(shlex.quote(p) for p in parts)


def _write_sbatch_script(
    *,
    sbatch_path: str,
    job_name: str,
    partition: str,
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
) -> None:
    contents = SBATCH_TEMPLATE.format(
        job_name=job_name,
        partition=partition,
        time_limit=time_limit,
        mem=mem,
        gres=gres,
        cpus_per_task=cpus_per_task,
        out_log=out_log,
        err_log=err_log,
        extra_sbatch_lines=extra_sbatch_lines,
        repo_dir=repo_dir,
        env_setup=env_setup,
        python_cmd=python_cmd,
    )
    with open(sbatch_path, "w") as f:
        f.write(contents)
    os.chmod(sbatch_path, 0o755)


def _submit_sbatch(sbatch_path: str, dry_run: bool) -> Optional[str]:
    """
    Submit *sbatch_path* via `sbatch`. Returns the job id string on success
    (parsed from "Submitted batch job <id>") or None if dry_run / parse
    failure. Raises subprocess.CalledProcessError on sbatch failure.
    """
    if dry_run:
        print(f"[dry_run] would sbatch: {sbatch_path}")
        return None
    out = subprocess.check_output(["sbatch", sbatch_path], text=True).strip()
    # Standard sbatch output: "Submitted batch job 1234567"
    m = re.search(r"Submitted batch job (\d+)", out)
    if m is None:
        print(f"[warn] sbatch returned unexpected output: {out!r}")
        return None
    return m.group(1)


# ----------------------------- squeue polling -----------------------------
def _still_running_job_ids(job_ids: List[str]) -> List[str]:
    """
    Return the subset of *job_ids* still in `squeue` (running or pending).
    A job that finished (success or failure) drops from squeue, so absence
    is treated as "done"; success/failure must be inferred from the job's
    output log if needed.
    """
    if not job_ids:
        return []
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", SQUEUE_FORMAT],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        # squeue returns non-zero when none of the jobs are found. Treat
        # that as "all done".
        return []
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


# ----------------------------- task pickup + processing -----------------------------
def _claim_task(todo_dir: str, processing_dir: str) -> Optional[Tuple[str, str, str]]:
    """
    Grab a random `.task` file from *todo_dir*, atomically rename it into
    *processing_dir*/<task_filename>_folder/<task_filename>. Returns
    (task_filename, processing_path, processing_folder) on success, None
    when the dir is empty or another worker beat us to it.

    The "_folder" wrapper exists so we can also stash per-task state
    (e.g. _jobs.json) alongside the .task without polluting processing_dir.
    """
    tasks = [f for f in os.listdir(todo_dir) if f.endswith(".task")]
    if not tasks:
        return None
    task_filename = random.choice(tasks)
    todo_path = os.path.join(todo_dir, task_filename)
    folder = os.path.join(processing_dir, f"{task_filename}_folder")
    os.makedirs(folder, exist_ok=True)
    processing_path = os.path.join(folder, task_filename)
    try:
        os.rename(todo_path, processing_path)
    except FileNotFoundError:
        return None
    return task_filename, processing_path, folder


def _copy_aux_files(src_dir: str, dst_dir: str) -> None:
    """
    Copy `.pt` siblings (left MA, right MAs, ME, LE snapshots) into the
    per-task processing folder. Same convention as new_br_worker.py's
    league branch — copy not move so other workers / orchestrators picking
    up other .task files in the same source dir still see the .pt files.
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


def _build_shared_config(args: argparse.Namespace, manual_stop_file: Optional[str]) -> Dict[str, Any]:
    """
    Pack everything that's identical across this task's per-spec jobs into
    one dict that gets JSON-encoded and passed via --shared_config_json.
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
        "device": args.device,
        "manual_stop_file": manual_stop_file,
        "manual_stop_key": None,
        "launch_local_br_eval": args.launch_local_br_eval,
        "use_wandb": args.use_wandb,
    }


def _process_task(
    args: argparse.Namespace,
    task_filename: str,
    processing_path: str,
    processing_folder: str,
) -> List[str]:
    """
    Detect type, build dedicated specs, write+submit one sbatch per spec,
    return the list of submitted job ids (empty on dry_run).
    """
    print(f"[orch] Processing task: {task_filename}")
    print(f"[orch] Processing path: {processing_path}")

    model_type = _detect_model_type(processing_path, device="cpu")
    is_league = (model_type == "league")
    print(f"[orch] Detected model_type={model_type} (is_league={is_league})")

    # Output subdir mirrors new_br_worker's convention so all jobs from this
    # task write into the SAME per-training-process folder. For league we
    # need the source directory the .task came from to fall back when
    # save_dir isn't in saved kwargs.
    output_subdir = _derive_output_subdir(
        processing_path, model_type, args.todo_dir
    )
    training_style = "league" if is_league else model_type
    print(f"[orch] output_subdir={output_subdir!r} training_style={training_style!r}")

    # League also needs aux .pt siblings copied so the per-job training can
    # find the matchup roster and sibling MA snapshots.
    if is_league:
        _copy_aux_files(args.todo_dir, processing_folder)

    # State list source: SPAR/IPPO checkpoints carry it inline; league
    # infers from sibling MA*_right_*.pt files (just copied above).
    if is_league:
        league_states = _infer_league_matchup_states_from_dir(processing_path)
        unique_states = list(league_states)
        # Side flip: loaded LEFT task → BR trains right (eval_prot=True);
        # loaded RIGHT task → BR trains left (eval_prot=False). Same
        # convention as new_br_worker.py's league branch.
        loaded_side, _matchup_key = _peek_league_side_and_matchup(processing_path)
        if loaded_side not in ("left", "right"):
            raise ValueError(
                f"League task {task_filename!r} has invalid side="
                f"{loaded_side!r}; expected 'left' or 'right'."
            )
        eval_prot_for_this_task = (loaded_side == "left")
        run_eval_prot = eval_prot_for_this_task
        run_eval_adv = not eval_prot_for_this_task
        print(
            f"[orch] League loaded_side={loaded_side} -> "
            f"eval_prot={run_eval_prot} eval_adv={run_eval_adv}"
        )
    else:
        league_states = None
        unique_states = _extract_unique_states_from_task(processing_path, device="cpu")
        run_eval_prot = args.eval_prot
        run_eval_adv = args.eval_adv

    specs = build_dedicated_job_specs(
        unique_states=unique_states,
        replicates_per_matchup=args.num_full_exploiters,
        run_eval_prot=run_eval_prot,
        run_eval_adv=run_eval_adv,
        launch_local_br_eval=args.launch_local_br_eval,
        state_to_matchup=state2matchup,
    )
    print(f"[orch] Built {len(specs)} dedicated job specs "
          f"({len(unique_states)} matchups x replicates="
          f"{args.num_full_exploiters} x sides=[ego={run_eval_prot}, adv={run_eval_adv}])")

    if not specs:
        print("[orch] No specs to submit (eval_prot/eval_adv both False?). "
              "Returning task to done immediately.")
        return []

    # Per-task slurm log dir — keeps output/error logs and per-spec sbatch
    # scripts colocated for easy debugging.
    task_stem = os.path.splitext(task_filename)[0]
    slurm_log_dir = os.path.join(
        os.path.abspath(args.slurm_log_dir), task_stem
    )
    os.makedirs(slurm_log_dir, exist_ok=True)

    # Per-task manual_stop file lives next to the processing folder so the
    # per-job processes can be cooperatively stopped.
    stop_file_dir = os.path.join(TASK_DIR, "stop")
    os.makedirs(stop_file_dir, exist_ok=True)
    manual_stop_file = os.path.join(
        stop_file_dir, f"STOP_{_sanitize_for_filename(task_stem)}"
    )

    shared_config = _build_shared_config(args, manual_stop_file=manual_stop_file)
    shared_config_json = json.dumps(shared_config)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    single_matchup_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "br_single_matchup.py")
    )

    extra_sbatch_lines = ""
    if args.slurm_account:
        extra_sbatch_lines = f"#SBATCH --account={args.slurm_account}\n"

    submitted: List[str] = []
    for spec in specs:
        side_label = "ego" if spec["eval_prot"] else "adv"
        matchup_safe = spec.get("matchup_label") or "unknown"
        job_name = (
            f"br_{output_subdir}_{matchup_safe}_{side_label}_rep{spec['replicate_idx']}"
        )
        out_log = os.path.join(slurm_log_dir, f"{job_name}.out")
        err_log = os.path.join(slurm_log_dir, f"{job_name}.err")
        sbatch_path = os.path.join(slurm_log_dir, f"{job_name}.sbatch")

        python_cmd = _build_python_cmd(
            python_bin=args.python_bin,
            single_matchup_script=single_matchup_script,
            task_file=processing_path,
            state=spec["state_subset"][0],
            eval_prot=spec["eval_prot"],
            replicate_idx=spec["replicate_idx"],
            br_index=spec["job_index"],
            dedicated_job_id=spec["job_index"],
            matchup_label=spec.get("matchup_label"),
            output_subdir=output_subdir,
            training_style=training_style,
            is_league=is_league,
            league_matchup_states=league_states,
            shared_config_json=shared_config_json,
        )

        _write_sbatch_script(
            sbatch_path=sbatch_path,
            job_name=job_name,
            partition=args.slurm_partition,
            time_limit=args.slurm_time,
            mem=args.slurm_mem,
            gres=args.slurm_gres,
            cpus_per_task=args.slurm_cpus_per_task,
            out_log=out_log,
            err_log=err_log,
            repo_dir=repo_dir,
            env_setup=args.env_setup,
            python_cmd=python_cmd,
            extra_sbatch_lines=extra_sbatch_lines,
        )

        try:
            job_id = _submit_sbatch(sbatch_path, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"[orch] sbatch FAILED for {sbatch_path}: {exc}; "
                  "continuing with remaining specs.")
            continue
        if job_id is not None:
            submitted.append(job_id)
            print(f"[orch] Submitted job_id={job_id} script={sbatch_path}")

    return submitted


# ----------------------------- async sweeper -----------------------------
def _registry_path(processing_folder: str) -> str:
    return os.path.join(processing_folder, "_jobs.json")


def _write_registry(processing_folder: str, payload: Dict[str, Any]) -> None:
    with open(_registry_path(processing_folder), "w") as f:
        json.dump(payload, f, indent=2)


def _read_registry(processing_folder: str) -> Optional[Dict[str, Any]]:
    path = _registry_path(processing_folder)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _sweep_completed_tasks(processing_dir: str, done_dir: str) -> int:
    """
    Walk processing_dir for per-task folders, check each registry's job ids
    via squeue, and when a task has zero still-running ids move its .task
    file (and the folder contents) to done_dir. Returns the count of tasks
    moved this sweep.
    """
    if not os.path.isdir(processing_dir):
        return 0
    moved = 0
    for entry in sorted(os.listdir(processing_dir)):
        folder = os.path.join(processing_dir, entry)
        if not os.path.isdir(folder) or not entry.endswith("_folder"):
            continue
        registry = _read_registry(folder)
        if registry is None:
            continue
        job_ids = list(registry.get("job_ids", []))
        if not job_ids:
            # Empty registry (e.g. dry_run) — leave it; the user can move
            # the folder by hand or rerun.
            continue
        still_running = _still_running_job_ids(job_ids)
        if still_running:
            continue
        # All jobs done — move the folder to done/. The .task file inside
        # the folder is what we ultimately care about; the per-job sbatch
        # scripts and registry travel with it for archival.
        dst_folder = os.path.join(done_dir, entry)
        try:
            os.rename(folder, dst_folder)
            moved += 1
            print(f"[orch] task complete (all {len(job_ids)} jobs cleared "
                  f"from squeue) -> moved {entry} to {done_dir}")
        except OSError as exc:
            print(f"[orch] could not move {folder} -> {dst_folder}: {exc}")
    return moved


# ----------------------------- CLI -----------------------------
def _bool_choice(s: str) -> bool:
    return str(s).lower() in ("true", "1", "yes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLURM orchestrator for dedicated BR exploiter jobs.",
    )
    # Watchdog / paths
    parser.add_argument("--todo_dir", type=str, required=True,
                        help="Directory to watch for incoming .task files. "
                             "For league, this is also the dir whose .pt "
                             "siblings get copied per task for matchup state "
                             "inference.")
    parser.add_argument("--processing_dir", type=str,
                        default=os.path.join(TASK_DIR, "slurm_processing"),
                        help="Per-task working dir (atomic claim target).")
    parser.add_argument("--done_dir", type=str,
                        default=os.path.join(TASK_DIR, "slurm_done"),
                        help="Where finished tasks are moved by the sweeper.")
    parser.add_argument("--slurm_log_dir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "slurm_logs",
                        ),
                        help="Per-task subdir is created here for sbatch "
                             "scripts and stdout/stderr logs.")
    parser.add_argument("--stop_file", type=str,
                        default=os.path.join(TASK_DIR, "STOP_SLURM"),
                        help="Touch this file to stop the orchestrator "
                             "watchdog (in-flight SLURM jobs are unaffected).")

    # Job-side parity with new_br_worker (drives shared_config_json).
    parser.add_argument("--eval_prot", choices=["True", "False"], default="True",
                        help="(SPAR/IPPO only — for league, side is "
                             "auto-derived from the loaded checkpoint.) "
                             "Schedule ego-exploiter specs.")
    parser.add_argument("--eval_adv", choices=["True", "False"], default="True",
                        help="(SPAR/IPPO only — see --eval_prot.) "
                             "Schedule adv-exploiter specs.")
    parser.add_argument("--eval_only", choices=["True", "False"], default="False")
    parser.add_argument("--proj_name", type=str, default="br_training")
    parser.add_argument("--analysis_upload_proj_name", type=str, default="br_analysis")
    parser.add_argument("--use_mirror", choices=["True", "False"], default="False")
    parser.add_argument("--num_full_exploiters", type=int, default=3,
                        help="Replicates per (matchup, side).")
    parser.add_argument("--n_envs", type=int, default=2,
                        help="Per-job env count (vectorized).")
    parser.add_argument("--exploiter_save_freq", type=int, default=100000)

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

    # Continue-mode CDS stagnation knobs (unused in dedicated mode but
    # threaded for parity / future use).
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
    parser.add_argument("--use_wandb", choices=["True", "False"], default="False")

    # SLURM
    parser.add_argument("--slurm_partition", type=str, default="gpu")
    parser.add_argument("--slurm_time", type=str, default="12:00:00")
    parser.add_argument("--slurm_mem", type=str, default="16G")
    parser.add_argument("--slurm_gres", type=str, default="gpu:1")
    parser.add_argument("--slurm_cpus_per_task", type=int, default=4)
    parser.add_argument("--slurm_account", type=str, default="",
                        help="Optional --account string for clusters that "
                             "require it.")
    parser.add_argument("--python_bin", type=str, default="python",
                        help="Python interpreter the per-job sbatch script "
                             "invokes (set to a venv python if needed).")
    parser.add_argument("--env_setup", type=str, default="",
                        help="Optional shell snippet inserted into each "
                             "sbatch script before the python call (e.g. "
                             "'module load cuda; source venv/bin/activate').")

    # Behavior
    parser.add_argument("--dry_run", choices=["True", "False"], default="False",
                        help="Write sbatch scripts but do not submit; "
                             "print what would be submitted.")
    return parser


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Convert all string-bool CLI args to real bools in-place."""
    bool_fields = [
        "eval_prot", "eval_adv", "eval_only", "use_mirror",
        "use_br_reward_stagnation", "use_br_entropy_stagnation",
        "br_use_slope_early_stop",
        "use_stagnation_early_stop", "use_stagnation_velocity_signal",
        "use_stagnation_entropy_signal", "stagnation_use_slope_early_stop",
        "render", "enable_combo", "null_combo", "transform_action",
        "launch_local_br_eval", "use_wandb", "dry_run",
    ]
    for k in bool_fields:
        setattr(args, k, _bool_choice(getattr(args, k)))
    return args


def main() -> None:
    args = _normalize_args(build_parser().parse_args())

    os.makedirs(args.todo_dir, exist_ok=True)
    os.makedirs(args.processing_dir, exist_ok=True)
    os.makedirs(args.done_dir, exist_ok=True)
    os.makedirs(args.slurm_log_dir, exist_ok=True)

    print(f"[orch] Watching todo_dir={args.todo_dir}")
    print(f"[orch] processing_dir={args.processing_dir}")
    print(f"[orch] done_dir={args.done_dir}")
    print(f"[orch] slurm_log_dir={args.slurm_log_dir}")
    print(f"[orch] stop_file={args.stop_file}")
    print(f"[orch] dry_run={args.dry_run}")
    print(f"[orch] SLURM: partition={args.slurm_partition} "
          f"time={args.slurm_time} mem={args.slurm_mem} "
          f"gres={args.slurm_gres} cpus={args.slurm_cpus_per_task}")

    while not os.path.exists(args.stop_file):
        # 1) Async sweep — move completed tasks to done/.
        try:
            _sweep_completed_tasks(args.processing_dir, args.done_dir)
        except Exception as exc:
            print(f"[orch] sweeper error (non-fatal): {exc}")

        # 2) Pickup — claim one new task and dispatch its sbatch jobs.
        claim = _claim_task(args.todo_dir, args.processing_dir)
        if claim is None:
            time.sleep(POLL_INTERVAL)
            continue
        task_filename, processing_path, processing_folder = claim
        try:
            job_ids = _process_task(args, task_filename, processing_path, processing_folder)
            _write_registry(processing_folder, {
                "task_filename": task_filename,
                "submitted_at": time.time(),
                "job_ids": job_ids,
                "dry_run": args.dry_run,
            })
        except Exception as exc:
            # Don't lose the .task — if dispatch failed, leave the
            # processing folder in place with a registry of what we got
            # to so the user can investigate and resubmit by hand.
            err_path = os.path.join(processing_folder, "_dispatch_error.txt")
            with open(err_path, "w") as f:
                f.write(f"{type(exc).__name__}: {exc}\n")
            print(f"[orch] dispatch error for {task_filename}: {exc}; "
                  f"see {err_path}")

    print(f"[orch] Stop file detected at {args.stop_file}; exiting.")


if __name__ == "__main__":
    main()
