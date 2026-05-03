"""
SLURM orchestrator for **per-matchup continue** BR exploiter jobs.

Mental model (per the user):
  "Generate the full factorial 1v1 matchup set. For each matchup, continue
  the relevant side's training. For 2 egos × 3 advs you get 6 matchups; in
  each one, the BR copy of that side continues training while the other
  side is loaded from the same checkpoint."

Per-matchup continue exploitation has the same scheduling shape as
dedicated mode (one spec per matchup × side × replicate), so we reuse
`build_dedicated_job_specs`. The only difference is downstream: the spawn
sbatch runs `br_single_continue.py` instead of `br_single_matchup.py`,
which forwards `from_scratch=False` to run_br_for_task_in_subprocess and
the backend skips the FixedMatchupPolicyAdapter wrap so CDS's own learn
loop can update the live model in-place.

LEAGUE: per-matchup continue for league requires loading the right MA
weights for the specific matchup into the LeaguePPO's policy_other (and
similar for the ego side), which `_run_league_continue_exploiter`
currently does not support. The orchestrator detects league tasks and
prints a clear message + skips them so they don't get processed wrongly.
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
    derive_output_subdir,
    detect_model_type,
    have_sbatch,
    normalize_bool_args,
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
    print(f"[orch-continue] Processing task: {task_filename}")
    print(f"[orch-continue] Processing path: {processing_path}")

    model_type = detect_model_type(processing_path, device="cpu")
    is_league = (model_type == "league")
    print(f"[orch-continue] Detected model_type={model_type} (is_league={is_league})")

    if is_league:
        # League per-matchup continue is not yet supported. Surface a clear
        # message and bail out. The .task stays in the processing folder
        # with a stub registry so the user can move it back to todo for
        # the dedicated orchestrator (or hand-process it).
        msg = (
            "[orch-continue] LEAGUE per-matchup continue is not yet "
            "supported. Skipping this task without dispatching jobs. "
            "Move the .task back to todo/ to retry, or run "
            "br_slurm_orchestrator.py (dedicated) instead."
        )
        print(msg)
        with open(os.path.join(processing_folder, "_skipped.txt"), "w") as f:
            f.write(msg + "\n")
        return []

    output_subdir = derive_output_subdir(processing_path, model_type, args.todo_dir)
    training_style = model_type
    print(f"[orch-continue] output_subdir={output_subdir!r} "
          f"training_style={training_style!r}")

    unique_states = _extract_unique_states_from_task(processing_path, device="cpu")
    run_eval_prot = args.eval_prot
    run_eval_adv = args.eval_adv

    # Reuse the dedicated spec builder — per-matchup × side × replicate
    # cross-product is identical for continue mode. The downstream runner
    # script (br_single_continue.py) is what forces from_scratch=False.
    specs = build_dedicated_job_specs(
        unique_states=unique_states,
        replicates_per_matchup=args.num_continue_exploiters,
        run_eval_prot=run_eval_prot,
        run_eval_adv=run_eval_adv,
        launch_local_br_eval=args.launch_local_br_eval,
        state_to_matchup=state2matchup,
    )
    print(f"[orch-continue] Built {len(specs)} continue job specs "
          f"({len(unique_states)} matchups x replicates="
          f"{args.num_continue_exploiters} x sides=[ego={run_eval_prot}, "
          f"adv={run_eval_adv}])")

    if not specs:
        print("[orch-continue] No specs to submit. Returning task to done "
              "immediately.")
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
        os.path.join(os.path.dirname(__file__), "br_single_continue.py")
    )

    extra_sbatch_lines = ""
    if args.slurm_account:
        extra_sbatch_lines = f"#SBATCH --account={args.slurm_account}\n"

    submitted: List[str] = []
    for spec in specs:
        side_label = "ego" if spec["eval_prot"] else "adv"
        matchup_safe = spec.get("matchup_label") or "unknown"
        # "cont_" prefix in the job name so dedicated and continue
        # squeue rows are easy to disambiguate when both orchestrators
        # are running.
        job_name = (
            f"cont_{output_subdir}_{matchup_safe}_{side_label}_"
            f"rep{spec['replicate_idx']}"
        )
        out_log = os.path.join(slurm_log_dir, f"{job_name}.out")
        err_log = os.path.join(slurm_log_dir, f"{job_name}.err")
        sbatch_path = os.path.join(slurm_log_dir, f"{job_name}.sbatch")

        python_cmd = build_python_cmd(
            python_bin=args.python_bin,
            runner_script=runner_script,
            task_file=processing_path,
            state=spec["state_subset"][0],
            eval_prot=spec["eval_prot"],
            replicate_idx=spec["replicate_idx"],
            br_index=spec["job_index"],
            dedicated_job_id=spec["job_index"],
            matchup_label=spec.get("matchup_label"),
            output_subdir=output_subdir,
            training_style=training_style,
            is_league=False,  # league is bailed out above
            league_matchup_states=None,
            shared_config_json=shared_config_json,
        )

        write_sbatch_script(
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
            job_id = submit_sbatch(
                sbatch_path,
                dry_run=args.dry_run,
                local_out_log=out_log,
                local_err_log=err_log,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[orch-continue] sbatch FAILED for {sbatch_path}: {exc}; "
                  "continuing with remaining specs.")
            continue
        if job_id is not None:
            submitted.append(job_id)
            print(f"[orch-continue] Submitted job_id={job_id} "
                  f"script={sbatch_path}")

    return submitted


# ----------------------------- CLI -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLURM orchestrator for per-matchup continue BR "
                    "exploiter jobs. CDS-only; league tasks are skipped "
                    "with a clear message.",
    )
    # Use distinct processing/done dirs so this orchestrator and the
    # dedicated one don't try to claim each other's tasks. The user runs
    # one or the other by convention; if both are running, the first to
    # rename wins per task — same TODO_DIR is fine.
    add_shared_arguments(
        parser,
        default_processing_subdir="slurm_processing_continue",
        default_done_subdir="slurm_done_continue",
        default_stop_file="STOP_SLURM_CONTINUE",
    )
    parser.add_argument("--num_continue_exploiters", type=int, default=1,
                        help="Replicates per (matchup, side) for continue "
                             "mode.")
    return parser


def main() -> None:
    args = normalize_bool_args(build_parser().parse_args())

    os.makedirs(args.todo_dir, exist_ok=True)
    os.makedirs(args.processing_dir, exist_ok=True)
    os.makedirs(args.done_dir, exist_ok=True)
    os.makedirs(args.slurm_log_dir, exist_ok=True)

    print(f"[orch-continue] Watching todo_dir={args.todo_dir}")
    print(f"[orch-continue] processing_dir={args.processing_dir}")
    print(f"[orch-continue] done_dir={args.done_dir}")
    print(f"[orch-continue] slurm_log_dir={args.slurm_log_dir}")
    print(f"[orch-continue] stop_file={args.stop_file}")
    print(f"[orch-continue] dry_run={args.dry_run}")
    print(f"[orch-continue] num_continue_exploiters={args.num_continue_exploiters}")
    if have_sbatch():
        print("[orch-continue] sbatch detected; jobs will be submitted to SLURM.")
    else:
        print("[orch-continue] sbatch NOT on PATH; falling back to local "
              "`bash <sbatch>` execution. SBATCH directives are ignored as "
              "comments. Per-job stdout/stderr go to the same log paths.")
    print(f"[orch-continue] SLURM: partition={args.slurm_partition} "
          f"time={args.slurm_time} mem={args.slurm_mem} "
          f"gres={args.slurm_gres} cpus={args.slurm_cpus_per_task}")

    while not os.path.exists(args.stop_file):
        try:
            sweep_completed_tasks(args.processing_dir, args.done_dir)
        except Exception as exc:
            print(f"[orch-continue] sweeper error (non-fatal): {exc}")

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
                "mode": "continue",
            })
        except Exception as exc:
            err_path = os.path.join(processing_folder, "_dispatch_error.txt")
            with open(err_path, "w") as f:
                f.write(f"{type(exc).__name__}: {exc}\n")
            print(f"[orch-continue] dispatch error for {task_filename}: "
                  f"{exc}; see {err_path}")

    print(f"[orch-continue] Stop file detected at {args.stop_file}; exiting.")


if __name__ == "__main__":
    main()
