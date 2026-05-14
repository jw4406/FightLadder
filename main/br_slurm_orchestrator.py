"""
SLURM orchestrator for **dedicated** (from-scratch) BR exploiter jobs.

Watches a TASK_DIR for `.task` files (mirrors new_br_worker.py's polling
pattern). When a task arrives:
  1. Atomically claims it (rename to processing/).
  2. Detects the model type (CDS spar/ippo or league).
  3. Computes per-training-process output_subdir + training_style.
  4. Builds dedicated job specs via `_build_dedicated_job_specs` —
     one spec per (state, side, replicate).
  5. For each spec, writes a per-job .sbatch file under
     <slurm_log_dir>/<task_stem>/<spec>.sbatch and submits via `sbatch`.
  6. Records `[<job_ids>]` in `processing/<task>_folder/_jobs.json`.

ASYNC: after submission the orchestrator does NOT block. On every
subsequent loop iteration it sweeps `processing/`, checks each registered
task's job ids via `squeue`, and when none of them remain, moves the
task folder to `done/`.

Continue-mode is handled by the parallel `br_continue_slurm_orchestrator.py`.

All shared machinery (sbatch generation, sweeper, model-type detection,
output_subdir, shared-config builder, CLI scaffolding) lives in
`br_slurm_common.py` so this script and the continue-mode sibling stay
in lockstep automatically.
"""
import argparse
import os
import socket
import subprocess
import time
from typing import List

socket.setdefaulttimeout(None)

from new_br_worker import (
    TASK_DIR,
    _extract_unique_states_from_task,
    _infer_league_matchup_states_from_dir,
    _sanitize_for_filename,
)
from br_preflight import build_dedicated_job_specs
from utils import state2matchup
from br_slurm_common import (
    POLL_INTERVAL,
    add_shared_arguments,
    build_python_cmd,
    build_shared_config,
    claim_task,
    copy_aux_files,
    derive_output_subdir,
    detect_model_type,
    have_sbatch,
    normalize_bool_args,
    peek_league_side_and_matchup,
    submit_sbatch,
    sweep_completed_tasks,
    write_registry,
    write_sbatch_script,
)


# ----------------------------- per-task processing -----------------------------
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
    print(f"[orch-dedicated] Processing task: {task_filename}")
    print(f"[orch-dedicated] Processing path: {processing_path}")

    model_type = detect_model_type(processing_path, device="cpu")
    is_league = (model_type == "league")
    print(f"[orch-dedicated] Detected model_type={model_type} (is_league={is_league})")

    output_subdir = derive_output_subdir(processing_path, model_type, args.todo_dir)
    training_style = "league" if is_league else model_type
    print(f"[orch-dedicated] output_subdir={output_subdir!r} "
          f"training_style={training_style!r}")

    if is_league:
        copy_aux_files(args.todo_dir, processing_folder)
        league_states = _infer_league_matchup_states_from_dir(processing_path)
        unique_states = list(league_states)
        # Side flip: loaded LEFT task → BR trains right (eval_prot=True);
        # loaded RIGHT task → BR trains left (eval_prot=False). Same
        # convention as new_br_worker.py's league branch.
        loaded_side, _matchup_key = peek_league_side_and_matchup(processing_path)
        if loaded_side not in ("left", "right"):
            raise ValueError(
                f"League task {task_filename!r} has invalid side="
                f"{loaded_side!r}; expected 'left' or 'right'."
            )
        eval_prot_for_this_task = (loaded_side == "left")
        run_eval_prot = eval_prot_for_this_task
        run_eval_adv = not eval_prot_for_this_task
        print(
            f"[orch-dedicated] League loaded_side={loaded_side} -> "
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
    print(f"[orch-dedicated] Built {len(specs)} dedicated job specs "
          f"({len(unique_states)} matchups x replicates="
          f"{args.num_full_exploiters} x sides=[ego={run_eval_prot}, adv={run_eval_adv}])")

    if not specs:
        print("[orch-dedicated] No specs to submit (eval_prot/eval_adv both "
              "False?). Returning task to done immediately.")
        return []

    task_stem = os.path.splitext(task_filename)[0]
    slurm_log_dir = os.path.join(os.path.abspath(args.slurm_log_dir), task_stem)
    os.makedirs(slurm_log_dir, exist_ok=True)

    stop_file_dir = os.path.join(TASK_DIR, "stop")
    os.makedirs(stop_file_dir, exist_ok=True)
    manual_stop_file = os.path.join(
        stop_file_dir, f"STOP_{_sanitize_for_filename(task_stem)}"
    )

    shared_config_json = __import__("json").dumps(
        build_shared_config(args, manual_stop_file=manual_stop_file)
    )

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner_script = os.path.abspath(
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
            f"br_{output_subdir}_{matchup_safe}_{side_label}_rep{spec['replicate_idx']}_{task_stem}"
        )
        out_log = os.path.join(slurm_log_dir, f"{job_name}.out")
        err_log = os.path.join(slurm_log_dir, f"{job_name}.err")
        sbatch_path = os.path.join(slurm_log_dir, f"{job_name}.sbatch")

        python_cmd = build_python_cmd(
            python_bin=args.python_bin,
            runner_script=runner_script,
            task_file=processing_path,
            local_plot_dir=args.local_plot_dir,
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

        write_sbatch_script(
            sbatch_path=sbatch_path,
            job_name=job_name,
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
            workdir=args.workdir,
            main_training_dir=args.main_training_dir,
        )

        try:
            job_id = submit_sbatch(
                sbatch_path,
                dry_run=args.dry_run,
                local_out_log=out_log,
                local_err_log=err_log,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[orch-dedicated] sbatch FAILED for {sbatch_path}: {exc}; "
                  "continuing with remaining specs.")
            continue
        if job_id is not None:
            submitted.append(job_id)
            print(f"[orch-dedicated] Submitted job_id={job_id} "
                  f"script={sbatch_path}")

    return submitted


# ----------------------------- CLI -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLURM orchestrator for dedicated (from-scratch) BR "
                    "exploiter jobs. One sbatch per (matchup, side, replicate).",
    )
    add_shared_arguments(
        parser,
        default_processing_subdir="slurm_processing",
        default_done_subdir="slurm_done",
        default_stop_file="STOP_SLURM",
    )
    # Dedicated-only knob.
    parser.add_argument("--num_full_exploiters", type=int, default=3,
                        help="Replicates per (matchup, side).")
    parser.add_argument("--local_plot_dir", type=str)
    return parser


def main() -> None:
    args = normalize_bool_args(build_parser().parse_args())

    os.makedirs(args.todo_dir, exist_ok=True)
    os.makedirs(args.processing_dir, exist_ok=True)
    os.makedirs(args.done_dir, exist_ok=True)
    os.makedirs(args.slurm_log_dir, exist_ok=True)

    print(f"[orch-dedicated] Watching todo_dir={args.todo_dir}")
    print(f"[orch-dedicated] processing_dir={args.processing_dir}")
    print(f"[orch-dedicated] done_dir={args.done_dir}")
    print(f"[orch-dedicated] slurm_log_dir={args.slurm_log_dir}")
    print(f"[orch-dedicated] stop_file={args.stop_file}")
    print(f"[orch-dedicated] dry_run={args.dry_run}")
    if have_sbatch():
        print("[orch-dedicated] sbatch detected; jobs will be submitted to SLURM.")
    else:
        print("[orch-dedicated] sbatch NOT on PATH; falling back to local "
              "`bash <sbatch>` execution. SBATCH directives are ignored as "
              "comments. Per-job stdout/stderr go to the same log paths.")
    print(f"[orch-dedicated] SLURM: "
          f"time={args.slurm_time} mem={args.slurm_mem} "
          f"gres={args.slurm_gres} cpus={args.slurm_cpus_per_task}")

    while not os.path.exists(args.stop_file):
        try:
            sweep_completed_tasks(args.processing_dir, args.done_dir)
        except Exception as exc:
            print(f"[orch-dedicated] sweeper error (non-fatal): {exc}")

        claim = claim_task(args.todo_dir, args.processing_dir)
        if claim is None:
            time.sleep(POLL_INTERVAL)
            continue
        task_filename, processing_path, processing_folder = claim
        try:
            job_ids = _process_task(args, task_filename, processing_path, processing_folder)
            write_registry(processing_folder, {
                "task_filename": task_filename,
                "submitted_at": time.time(),
                "job_ids": job_ids,
                "dry_run": args.dry_run,
                "mode": "dedicated",
            })
        except Exception as exc:
            err_path = os.path.join(processing_folder, "_dispatch_error.txt")
            with open(err_path, "w") as f:
                f.write(f"{type(exc).__name__}: {exc}\n")
            print(f"[orch-dedicated] dispatch error for {task_filename}: "
                  f"{exc}; see {err_path}")

    print(f"[orch-dedicated] Stop file detected at {args.stop_file}; exiting.")


if __name__ == "__main__":
    main()
