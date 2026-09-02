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
import re
import shlex
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
    apply_template_config,
    build_python_cmd,
    build_shared_config,
    claim_task,
    copy_aux_files,
    derive_output_subdir,
    detect_model_type,
    have_sbatch,
    load_template_config,
    normalize_bool_args,
    peek_league_side_and_matchup,
    print_dt_warning,
    read_registry,
    render_template_sbatch,
    submit_sbatch,
    sweep_completed_tasks,
    write_registry,
    write_sbatch_script,
)
from br_ws_concurrency import count_active_local_jobs


# ----------------------------- per-task processing -----------------------------
def _scale_resources(template_text: str, cpu_k: int, mem_k: int) -> str:
    """Scale a packed sbatch's HOST resources: multiply the template's
    #SBATCH --cpus-per-task by cpu_k and --mem by mem_k. The --gres (GPU) line
    is deliberately left as-is: the K processes SHARE one GPU (that's the point
    of packing). cpu_k is the absolute --resource_scale (default 1, so cpus stay
    at the template base); mem_k is the co-located exploiter count, so per-process
    memory still scales with the pack."""
    text = re.sub(r"(?m)^(#SBATCH --cpus-per-task=)(\d+)",
                  lambda m: f"{m.group(1)}{int(m.group(2)) * cpu_k}", template_text)
    text = re.sub(r"(?m)^(#SBATCH --mem=)(\d+)([A-Za-z]+)",
                  lambda m: f"{m.group(1)}{int(m.group(2)) * mem_k}{m.group(3)}", text)
    return text


def _scale_mem(mem_str: str, k: int) -> str:
    m = re.match(r"^\s*(\d+)\s*([A-Za-z]+)\s*$", str(mem_str))
    return f"{int(m.group(1)) * k}{m.group(2)}" if m else mem_str


def _build_packed_block(python_cmds: List[str], gpu_mem_fraction: float,
                        log_dir: str, group_tag: str) -> str:
    """Assemble the bash that co-locates K exploiters on one GPU, each capped
    via BR_GPU_MEM_FRACTION, then retries any FAILED process SOLO once with the
    cap removed (the others have exited, so it owns the whole card). Each
    build_python_cmd snippet defines its own CMD=() array, so we wrap each in a
    subshell to isolate that variable and background it. `set +e` so we manage
    child exit codes explicitly; the job always exits 0 (failures are handled
    and logged here, so the orchestrator's normal squeue-clear sweep applies)."""
    lines = [
        "set +e",
        f'export BR_GPU_MEM_FRACTION="{gpu_mem_fraction}"',
        "_pids=()",
    ]
    for i, cmd in enumerate(python_cmds):
        log = shlex.quote(os.path.join(log_dir, f"{group_tag}_proc{i}.out"))
        lines += [f"# ---- packed process {i} ----", "(", cmd,
                  f") > {log} 2>&1 &", "_pids+=($!)"]
    lines += [
        "_ecs=()",
        'for _p in "${_pids[@]}"; do',
        '  if wait "$_p"; then _ecs+=(0); else _ecs+=($?); fi',
        "done",
        "# ---- downgrade-to-solo retry: any failed process reruns alone, full GPU ----",
        "unset BR_GPU_MEM_FRACTION",
    ]
    for i, cmd in enumerate(python_cmds):
        log = shlex.quote(os.path.join(log_dir, f"{group_tag}_proc{i}.retry.out"))
        lines += [
            f'if [ "${{_ecs[{i}]}}" -ne 0 ]; then',
            f'  echo "[pack] process {i} failed (ec=${{_ecs[{i}]}}); retrying SOLO (full GPU)"',
            "  (", cmd,
            f'  ) > {log} 2>&1 || echo "[pack] process {i} SOLO retry also FAILED (giving up)"',
            "fi",
        ]
    lines.append("exit 0")
    return "\n".join(lines)


def _submit_packed_groups(*, specs, k, gpu_mem_fraction, args, slurm_log_dir,
                          task_stem, output_subdir, training_style, is_league,
                          league_states, shared_config_json, runner_script,
                          repo_dir, template_text, extra_sbatch_lines,
                          processing_path) -> List[str]:
    """Chunk specs into groups of K and submit one sbatch per group; each group
    co-locates K exploiters on one GPU (capped, solo-retry on failure). Returns
    submitted job ids. Mirrors the per-spec path's render/submit exactly, but
    with a packed CMD block and K-scaled host resources."""
    submitted: List[str] = []
    groups = [specs[i:i + k] for i in range(0, len(specs), k)]
    print(f"[orch-dedicated] PACKING: {len(specs)} specs -> {len(groups)} jobs "
          f"({k}/GPU, gpu_mem_fraction={gpu_mem_fraction})")
    for g, group in enumerate(groups):
        group_tag = f"pack{k}_grp{g}"
        cmds = [
            build_python_cmd(
                python_bin=args.python_bin, runner_script=runner_script,
                task_file=processing_path, local_plot_dir=args.local_plot_dir,
                state=spec["state_subset"][0], eval_prot=spec["eval_prot"],
                replicate_idx=spec["replicate_idx"], br_index=spec["job_index"],
                dedicated_job_id=spec["job_index"],
                matchup_label=spec.get("matchup_label"),
                output_subdir=output_subdir, training_style=training_style,
                is_league=is_league, league_matchup_states=league_states,
                shared_config_json=shared_config_json,
            )
            for spec in group
        ]
        block = _build_packed_block(cmds, gpu_mem_fraction, slurm_log_dir, group_tag)
        actual_k = len(group)
        job_name = f"br_{output_subdir}_{group_tag}_{task_stem}"
        out_log = os.path.join(slurm_log_dir, f"{job_name}.out")
        err_log = os.path.join(slurm_log_dir, f"{job_name}.err")
        sbatch_path = os.path.join(slurm_log_dir, f"{job_name}.sbatch")
        if template_text:
            render_template_sbatch(
                template_text=_scale_resources(template_text, int(args.resource_scale), actual_k),
                sbatch_path=sbatch_path, job_name=job_name, out_log=out_log,
                err_log=err_log, python_cmd=block,
                extra_sbatch_lines=extra_sbatch_lines,
                extra_placeholders={
                    "SBATCH_TIME": args.slurm_time,
                    "WS_WORKDIR": args.workdir,
                    "MAIN_TRAINING_DIR": args.main_training_dir,
                    "WS_REPO_DIR": repo_dir,
                },
            )
        else:
            write_sbatch_script(
                sbatch_path=sbatch_path, job_name=job_name,
                time_limit=args.slurm_time, mem=_scale_mem(args.slurm_mem, actual_k),
                gres=args.slurm_gres, cpus_per_task=args.slurm_cpus_per_task * int(args.resource_scale),
                out_log=out_log, err_log=err_log, repo_dir=repo_dir,
                env_setup=args.env_setup, python_cmd=block,
                extra_sbatch_lines=extra_sbatch_lines, workdir=args.workdir,
                main_training_dir=args.main_training_dir,
            )
        try:
            job_id = submit_sbatch(sbatch_path, dry_run=args.dry_run,
                                   local_out_log=out_log, local_err_log=err_log)
        except subprocess.CalledProcessError as exc:
            print(f"[orch-dedicated] sbatch FAILED for {sbatch_path}: {exc}")
            continue
        if job_id:
            submitted.append(job_id)
    return submitted


def _process_task(
    args: argparse.Namespace,
    task_filename: str,
    processing_path: str,
    processing_folder: str,
    template_text: str = "",
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

    _shared_config = build_shared_config(
        args, manual_stop_file=manual_stop_file,
        is_league=is_league, model_type=model_type, model_path=processing_path)
    print_dt_warning(_shared_config.get("dt_provenance"), tag="orch-dedicated")
    shared_config_json = __import__("json").dumps(_shared_config)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "br_single_matchup.py")
    )

    extra_sbatch_lines = ""
    if args.slurm_account:
        extra_sbatch_lines = f"#SBATCH --account={args.slurm_account}\n"

    # GPU packing (opt-in): >1 co-locates K exploiters per sbatch. Default 1
    # falls through to the unchanged one-sbatch-per-spec path below.
    if args.exploiters_per_job and int(args.exploiters_per_job) > 1:
        return _submit_packed_groups(
            specs=specs, k=int(args.exploiters_per_job),
            gpu_mem_fraction=args.gpu_mem_fraction, args=args,
            slurm_log_dir=slurm_log_dir, task_stem=task_stem,
            output_subdir=output_subdir, training_style=training_style,
            is_league=is_league, league_states=league_states,
            shared_config_json=shared_config_json, runner_script=runner_script,
            repo_dir=repo_dir, template_text=template_text,
            extra_sbatch_lines=extra_sbatch_lines, processing_path=processing_path,
        )

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

        if template_text:
            render_template_sbatch(
                template_text=template_text,
                sbatch_path=sbatch_path,
                job_name=job_name,
                out_log=out_log,
                err_log=err_log,
                python_cmd=python_cmd,
                extra_sbatch_lines=extra_sbatch_lines,
                extra_placeholders={
                    "SBATCH_TIME": args.slurm_time,
                    "WS_WORKDIR": args.workdir,
                    "MAIN_TRAINING_DIR": args.main_training_dir,
                    "WS_REPO_DIR": os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    ),
                },
            )
        else:
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


# ------------------- Cross-checkpoint packing (opt-in) -------------------
# NOTE: intentionally duplicates _process_task's "prepare" logic rather than
# refactoring it, to keep the default (per-task) path byte-identical. DRY later.

def _append_job_to_registry(processing_folder: str, job_id: str) -> None:
    """Append a (possibly shared) job id to a task folder's registry. A packed
    cross-checkpoint job carries specs from several tasks, so its id is appended
    to every contributing folder; the sweep then moves each folder to done/ once
    all of ITS ids clear squeue."""
    reg = read_registry(processing_folder) or {}
    ids = list(reg.get("job_ids", []))
    ids.append(job_id)
    reg["job_ids"] = ids
    reg.setdefault("mode", "dedicated_xpack")
    write_registry(processing_folder, reg)


def _prepare_task_for_pack(args, task_filename, processing_path, processing_folder):
    """Copy of _process_task's detect+build-specs+shared-config prep, returning
    (specs, task_ctx) WITHOUT submitting. Kept separate so _process_task stays
    byte-identical (see NOTE above)."""
    model_type = detect_model_type(processing_path, device="cpu")
    is_league = (model_type == "league")
    output_subdir = derive_output_subdir(processing_path, model_type, args.todo_dir)
    training_style = "league" if is_league else model_type
    if is_league:
        copy_aux_files(args.todo_dir, processing_folder)
        league_states = _infer_league_matchup_states_from_dir(processing_path)
        unique_states = list(league_states)
        loaded_side, _mk = peek_league_side_and_matchup(processing_path)
        if loaded_side not in ("left", "right"):
            raise ValueError(f"League task {task_filename!r} invalid side={loaded_side!r}")
        run_eval_prot = (loaded_side == "left")
        run_eval_adv = not run_eval_prot
    else:
        league_states = None
        unique_states = _extract_unique_states_from_task(processing_path, device="cpu")
        run_eval_prot = args.eval_prot
        run_eval_adv = args.eval_adv
    specs = build_dedicated_job_specs(
        unique_states=unique_states,
        replicates_per_matchup=args.num_full_exploiters,
        run_eval_prot=run_eval_prot, run_eval_adv=run_eval_adv,
        launch_local_br_eval=args.launch_local_br_eval,
        state_to_matchup=state2matchup,
    )
    task_stem = os.path.splitext(task_filename)[0]
    stop_file_dir = os.path.join(TASK_DIR, "stop")
    os.makedirs(stop_file_dir, exist_ok=True)
    manual_stop_file = os.path.join(stop_file_dir, f"STOP_{_sanitize_for_filename(task_stem)}")
    _shared_config = build_shared_config(
        args, manual_stop_file=manual_stop_file,
        is_league=is_league, model_type=model_type, model_path=processing_path)
    print_dt_warning(_shared_config.get("dt_provenance"), tag="orch-dedicated")
    shared_config_json = __import__("json").dumps(_shared_config)
    task_ctx = {
        "task_file": processing_path,
        "processing_folder": processing_folder,
        "output_subdir": output_subdir,
        "training_style": training_style,
        "is_league": is_league,
        "league_states": league_states,
        "shared_config_json": shared_config_json,
    }
    return specs, task_ctx


def _submit_cross_pack(group, args, template_text, group_idx, slurm_log_base,
                       repo_dir, runner_script, extra_sbatch_lines) -> bool:
    """Submit ONE packed sbatch for `group` = [(spec, task_ctx), ...] spanning
    possibly-different checkpoints, and append the job id to each contributing
    task's registry. Returns True on submit (or dry-run render), False on failure."""
    k = len(group)
    log_dir = os.path.join(slurm_log_base, "xpack")
    os.makedirs(log_dir, exist_ok=True)
    job_name = f"br_xpack{args.exploiters_per_job}_{group_idx:05d}"
    cmds = [
        build_python_cmd(
            python_bin=args.python_bin, runner_script=runner_script,
            task_file=ctx["task_file"], local_plot_dir=args.local_plot_dir,
            state=spec["state_subset"][0], eval_prot=spec["eval_prot"],
            replicate_idx=spec["replicate_idx"], br_index=spec["job_index"],
            dedicated_job_id=spec["job_index"], matchup_label=spec.get("matchup_label"),
            output_subdir=ctx["output_subdir"], training_style=ctx["training_style"],
            is_league=ctx["is_league"], league_matchup_states=ctx["league_states"],
            shared_config_json=ctx["shared_config_json"],
        )
        for spec, ctx in group
    ]
    block = _build_packed_block(cmds, args.gpu_mem_fraction, log_dir, job_name)
    out_log = os.path.join(log_dir, f"{job_name}.out")
    err_log = os.path.join(log_dir, f"{job_name}.err")
    sbatch_path = os.path.join(log_dir, f"{job_name}.sbatch")
    if template_text:
        render_template_sbatch(
            template_text=_scale_resources(template_text, int(args.resource_scale), k),
            sbatch_path=sbatch_path, job_name=job_name, out_log=out_log,
            err_log=err_log, python_cmd=block, extra_sbatch_lines=extra_sbatch_lines,
            extra_placeholders={"SBATCH_TIME": args.slurm_time, "WS_WORKDIR": args.workdir,
                                "MAIN_TRAINING_DIR": args.main_training_dir, "WS_REPO_DIR": repo_dir},
        )
    else:
        write_sbatch_script(
            sbatch_path=sbatch_path, job_name=job_name, time_limit=args.slurm_time,
            mem=_scale_mem(args.slurm_mem, k), gres=args.slurm_gres,
            cpus_per_task=args.slurm_cpus_per_task * int(args.resource_scale), out_log=out_log, err_log=err_log,
            repo_dir=repo_dir, env_setup=args.env_setup, python_cmd=block,
            extra_sbatch_lines=extra_sbatch_lines, workdir=args.workdir,
            main_training_dir=args.main_training_dir,
        )
    try:
        job_id = submit_sbatch(sbatch_path, dry_run=args.dry_run,
                               local_out_log=out_log, local_err_log=err_log)
    except subprocess.CalledProcessError as exc:
        print(f"[orch-xpack] sbatch FAILED for {sbatch_path}: {exc}; "
              "specs dropped (their tasks may hang until re-queued)")
        return False
    if job_id:
        for folder in {ctx["processing_folder"] for _s, ctx in group}:
            _append_job_to_registry(folder, job_id)
    print(f"[orch-xpack] submitted {job_name} (k={k}, job_id={job_id or 'dry-run'})")
    return True


def _run_crosspack_loop(args, template_text: str) -> None:
    """Streaming watchdog that packs exploiters ACROSS checkpoints: buffer each
    claimed task's specs, flush --exploiters_per_job at a time (or a partial pack
    once the oldest buffered spec exceeds --pack_flush_timeout). The sweep is
    unchanged; a packed job's id is appended to every contributing task."""
    k = int(args.exploiters_per_job)
    flush_timeout = float(args.pack_flush_timeout)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "br_single_matchup.py"))
    extra_sbatch_lines = f"#SBATCH --account={args.slurm_account}\n" if args.slurm_account else ""
    slurm_log_base = os.path.abspath(args.slurm_log_dir)
    buffer = []          # entries: (spec, task_ctx, added_at)
    group_counter = 0
    print(f"[orch-xpack] cross-checkpoint packing: K={k} flush_timeout={flush_timeout}s "
          f"gpu_mem_fraction={args.gpu_mem_fraction}")

    def _flush(entries):
        nonlocal group_counter
        _submit_cross_pack([(s, c) for (s, c, _t) in entries], args, template_text,
                           group_counter, slurm_log_base, repo_dir, runner_script,
                           extra_sbatch_lines)
        group_counter += 1

    while not os.path.exists(args.stop_file):
        try:
            sweep_completed_tasks(args.processing_dir, args.done_dir)
        except Exception as exc:
            print(f"[orch-xpack] sweeper error (non-fatal): {exc}")
        if not have_sbatch() and count_active_local_jobs(args.processing_dir) >= args.max_local_concurrent:
            time.sleep(POLL_INTERVAL)
            continue

        did_work = False
        claim = claim_task(args.todo_dir, args.processing_dir, step_stride=args.step_stride)
        if claim is not None:
            task_filename, processing_path, processing_folder = claim
            try:
                specs, ctx = _prepare_task_for_pack(args, task_filename, processing_path, processing_folder)
                # Empty registry -> sweep leaves the folder alone until a packed
                # job appends this task's job ids.
                write_registry(processing_folder, {
                    "task_filename": task_filename, "submitted_at": time.time(),
                    "job_ids": [], "dry_run": args.dry_run, "mode": "dedicated_xpack",
                })
                now = time.time()
                buffer.extend((s, ctx, now) for s in specs)
                did_work = True
                print(f"[orch-xpack] buffered {len(specs)} specs from {task_filename} "
                      f"(buffer={len(buffer)})")
            except Exception as exc:
                err_path = os.path.join(processing_folder, "_dispatch_error.txt")
                with open(err_path, "w") as f:
                    f.write(f"{type(exc).__name__}: {exc}\n")
                print(f"[orch-xpack] prepare error for {task_filename}: {exc}")

        while len(buffer) >= k:                          # full packs
            grp = buffer[:k]; del buffer[:k]
            _flush(grp); did_work = True
        if buffer and (time.time() - buffer[0][2]) >= flush_timeout:   # partial pack
            print(f"[orch-xpack] flush timeout -> partial pack of {len(buffer)}")
            grp = buffer[:]; buffer.clear()
            _flush(grp); did_work = True

        if not did_work:
            time.sleep(POLL_INTERVAL)

    if buffer:                                           # drain on stop
        print(f"[orch-xpack] STOP detected; draining {len(buffer)} buffered specs")
        while buffer:
            grp = buffer[:k]; del buffer[:k]
            _flush(grp)
    print(f"[orch-xpack] Stop file {args.stop_file} detected; exiting.")


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
    parser.add_argument("--br_dedicated_sh_template", type=str, default="",
                        help="Path to a .slurm template with default config "
                             "values. CLI flags override template values.")
    # Dedicated-only knob.
    parser.add_argument("--num_full_exploiters", type=int, default=3,
                        help="Replicates per (matchup, side).")
    parser.add_argument("--local_plot_dir", type=str)
    # GPU packing (opt-in). Default 1 == today's one-job-one-GPU behavior.
    parser.add_argument("--exploiters_per_job", type=int, default=1,
                        help="Run this many exploiters per sbatch (co-located on "
                             "one GPU). Default 1 = one job per GPU (unchanged). "
                             ">1 packs K/GPU, each capped via --gpu_mem_fraction; a "
                             "failed packed process is retried SOLO once (full GPU).")
    parser.add_argument("--gpu_mem_fraction", type=float, default=0.45,
                        help="Per-process GPU memory cap for packed jobs "
                             "(set_per_process_memory_fraction). Used only when "
                             "--exploiters_per_job > 1.")
    # Cross-checkpoint packing (opt-in). Default False = within-checkpoint only.
    parser.add_argument("--pack_across_checkpoints", choices=["True", "False"], default="False",
                        help="Pack exploiters from DIFFERENT checkpoints onto one GPU: "
                             "buffer specs across tasks and flush --exploiters_per_job at a "
                             "time. Default False = one job per checkpoint (within-checkpoint "
                             "packing only).")
    parser.add_argument("--pack_flush_timeout", type=float, default=300.0,
                        help="Max seconds the oldest buffered spec waits before a partial "
                             "pack is submitted (cross-checkpoint mode only).")
    parser.add_argument("--resource_scale", type=int, default=1,
                        help="Absolute multiplier for a PACKED job's #SBATCH --cpus-per-task "
                             "(cpu only; --mem still scales by the co-located exploiter count). "
                             "Default 1 = template base cpus. Set independently of "
                             "--exploiters_per_job, e.g. --resource_scale 6.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    template_text = ""
    if args.br_dedicated_sh_template:
        template_path = os.path.abspath(args.br_dedicated_sh_template)
        config = load_template_config(template_path)
        apply_template_config(args, config, parser)
        with open(template_path) as fh:
            template_text = fh.read()
        print(f"[orch-dedicated] template={template_path}")

    args = normalize_bool_args(args)

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

    # Cross-checkpoint packing mode (opt-in). Otherwise fall through to the
    # unchanged one-task-at-a-time loop below.
    _xpack = getattr(args, "pack_across_checkpoints", "False")
    if (_xpack is True or str(_xpack) == "True") and int(args.exploiters_per_job) > 1:
        _run_crosspack_loop(args, template_text)
        return

    while not os.path.exists(args.stop_file):
        try:
            sweep_completed_tasks(args.processing_dir, args.done_dir)
        except Exception as exc:
            print(f"[orch-dedicated] sweeper error (non-fatal): {exc}")

        # Workstation-mode concurrency gate. No-op on SLURM clusters because
        # have_sbatch() short-circuits the conjunction.
        if not have_sbatch() and count_active_local_jobs(args.processing_dir) >= args.max_local_concurrent:
            time.sleep(POLL_INTERVAL)
            continue

        claim = claim_task(args.todo_dir, args.processing_dir, step_stride=args.step_stride)
        if claim is None:
            time.sleep(POLL_INTERVAL)
            continue
        task_filename, processing_path, processing_folder = claim
        try:
            job_ids = _process_task(
                args, task_filename, processing_path, processing_folder,
                template_text=template_text,
            )
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
