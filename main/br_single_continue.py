"""
Standalone single-(matchup, side, replicate) **continue** BR runner.
Counterpart to `br_single_matchup.py` for the per-matchup continue
exploitation flow (from_scratch=False).

Conceptually: per the user's mental model, exploiting ego is "generate
the full factorial set of 1v1 matchups; for each matchup, continue the
adversary against the (frozen-this-iteration) ego". The orchestrator
schedules one sbatch per matchup × side × replicate; each sbatch runs
this script, which loads its own copy of the CDS / league checkpoint
and continues training the relevant side via run_br_for_task_in_subprocess.

CLI surface intentionally identical to br_single_matchup.py so the same
sbatch generator (br_slurm_common.build_python_cmd) can drive both. The
only behavioral difference is `from_scratch=False` below.
"""
import argparse
import json
import os
import sys


def _peek_torch_device_argv(argv):
    """
    Mirror new_br_worker's pre-import device peek so ``--device cpu`` hides
    GPUs from PyTorch BEFORE torch is imported. Reads the device key out of
    the shared_config_json blob without doing a full json.loads of objects
    we don't need yet.
    """
    for i, a in enumerate(argv):
        if a == "--shared_config_json" and i + 1 < len(argv):
            try:
                cfg = json.loads(argv[i + 1])
                return cfg.get("device")
            except Exception:
                return None
    return None


_peeked_dev = _peek_torch_device_argv(sys.argv[1:])
if _peeked_dev is not None and str(_peeked_dev).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from new_br_worker import run_br_for_task_in_subprocess


def _bool(s: str) -> bool:
    return str(s).lower() in ("true", "1", "yes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single per-matchup continue BR exploiter as one "
                    "process. Invoked by br_continue_slurm_orchestrator.py "
                    "via sbatch."
    )
    parser.add_argument("--task_file", type=str, required=True,
                        help="Path to the .task checkpoint file.")
    parser.add_argument("--state", type=str, required=True,
                        help="Single retro state string for this matchup.")
    parser.add_argument("--eval_prot", type=str, required=True,
                        choices=["True", "False"],
                        help="True = exploit ego (continue adv side); "
                             "False = exploit adv (continue ego side).")
    parser.add_argument("--replicate_idx", type=int, required=True)
    parser.add_argument("--br_index", type=int, required=True)
    parser.add_argument("--dedicated_job_id", type=int, required=True,
                        help="Same as br_index; kept for symmetry with "
                             "run_br_for_task_in_subprocess's signature.")
    parser.add_argument("--matchup_label", type=str, default=None)
    parser.add_argument("--output_subdir", type=str, default="")
    parser.add_argument("--training_style", type=str, default="")
    parser.add_argument("--is_league", type=str, required=True,
                        choices=["True", "False"])
    parser.add_argument("--league_matchup_states", type=str, nargs="+",
                        default=None)
    parser.add_argument("--shared_config_json", type=str, required=True)
    args = parser.parse_args()

    eval_prot = _bool(args.eval_prot)
    is_league = _bool(args.is_league)
    cfg = json.loads(args.shared_config_json)

    state_subset = [args.state]

    print(
        f"[br_single_continue] task={os.path.basename(args.task_file)} "
        f"state={args.state} eval_prot={eval_prot} "
        f"replicate={args.replicate_idx} br_index={args.br_index} "
        f"is_league={is_league} output_subdir={args.output_subdir!r} "
        f"training_style={args.training_style!r}",
        flush=True,
    )

    run_br_for_task_in_subprocess(
        game_args=cfg["game_args"],
        task_file_path=args.task_file,
        eval_prot=eval_prot,
        use_mirror=cfg.get("use_mirror", False),
        eval_only=cfg.get("eval_only", False),
        proj_name=cfg.get("proj_name", "br_training"),
        analysis_upload_proj_name=cfg.get("analysis_upload_proj_name", "br_analysis"),
        n_envs=cfg.get("n_envs", 2),
        is_spar_like=(not is_league),
        br_index=args.br_index,
        # CONTINUE mode: from_scratch=False — load the existing CDS / league
        # model and continue training the relevant side in-place. The
        # backend gate in run_br_for_task_in_subprocess (CDS branch) skips
        # FixedMatchupPolicyAdapter for this case so CDS's own learn loop
        # routes data through the matching matchup head naturally.
        from_scratch=False,
        exploiter_save_freq=cfg.get("exploiter_save_freq", 100000),
        # BR convergence tracker (used for the Exploiter-trained-from-
        # scratch path; harmless here in continue mode but threaded for
        # parity with the dedicated runner).
        br_tracker_patience=cfg.get("br_tracker_patience", 10),
        br_tracker_tolerance=cfg.get("br_tracker_tolerance", 1e-4),
        br_tracker_window_size=cfg.get("br_tracker_window_size", 50),
        use_br_reward_stagnation=cfg.get("use_br_reward_stagnation", True),
        use_br_entropy_stagnation=cfg.get("use_br_entropy_stagnation", True),
        br_use_slope_early_stop=cfg.get("br_use_slope_early_stop", False),
        br_slope_window=cfg.get("br_slope_window", 20),
        br_slope_tolerance=cfg.get("br_slope_tolerance", 5e-3),
        br_min_slope_checks=cfg.get("br_min_slope_checks", 10),
        # CDS continue-mode stagnation knobs (active in this mode).
        use_stagnation_early_stop=cfg.get("use_stagnation_early_stop", False),
        use_stagnation_velocity_signal=cfg.get("use_stagnation_velocity_signal", False),
        use_stagnation_entropy_signal=cfg.get("use_stagnation_entropy_signal", True),
        stagnation_patience=cfg.get("stagnation_patience", 2000000),
        stagnation_tolerance=cfg.get("stagnation_tolerance", 1e-4),
        stagnation_rel_tolerance=cfg.get("stagnation_rel_tolerance", 0.05),
        stagnation_ema_beta=cfg.get("stagnation_ema_beta", 0.99),
        stagnation_eps=cfg.get("stagnation_eps", 1e-8),
        stagnation_eval_games=cfg.get("stagnation_eval_games", 0),
        entropy_stagnation_weight=cfg.get("entropy_stagnation_weight", 100.0),
        stagnation_lr_factor=cfg.get("stagnation_lr_factor", 1.0),
        stagnation_lr_patience=cfg.get("stagnation_lr_patience", 0),
        stagnation_use_slope_early_stop=cfg.get("stagnation_use_slope_early_stop", False),
        stagnation_slope_window=cfg.get("stagnation_slope_window", 20),
        stagnation_slope_tolerance=cfg.get("stagnation_slope_tolerance", 5e-3),
        stagnation_min_slope_checks=cfg.get("stagnation_min_slope_checks", 10),
        device=cfg.get("device", "cuda"),
        state_subset=state_subset,
        matchup_label=args.matchup_label,
        replicate_idx=args.replicate_idx,
        dedicated_job_id=args.dedicated_job_id,
        manual_stop_file=cfg.get("manual_stop_file"),
        manual_stop_key=cfg.get("manual_stop_key"),
        launch_local_br_eval=cfg.get("launch_local_br_eval", True),
        use_wandb=cfg.get("use_wandb", False),
        is_league=is_league,
        league_matchup_states=args.league_matchup_states,
        output_subdir=args.output_subdir,
        entropy_stop_ratio=cfg.get("entropy_stop_ratio", 0.15),
        entropy_window_size=cfg.get("entropy_window_size", 50),
        entropy_warmup_checks=cfg.get("entropy_warmup_checks", 100),
    )

    print(
        f"[br_single_continue] DONE state={args.state} eval_prot={eval_prot} "
        f"replicate={args.replicate_idx}",
        flush=True,
    )


if __name__ == "__main__":
    main()
