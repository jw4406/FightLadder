"""
Standalone single-(matchup, side, replicate) BR runner used by the SLURM
orchestrator (`br_slurm_orchestrator.py`). One sbatch == one invocation of
this script == one BR exploiter trained against the loaded checkpoint for
one matchup state and one side.

The orchestrator builds the dedicated specs and submits N independent
SLURM jobs that each run this script. Per-job-unique fields come in via
explicit CLI args; everything else (device, n_envs, all tracker / entropy
/ stagnation knobs, project name, etc.) is passed as a single
`--shared_config_json` blob so we don't reproduce 40+ argparse declarations
in two places.

This script is intentionally thin: parse args, deserialize config, call
``run_br_for_task_in_subprocess`` from new_br_worker. All the actual
training logic lives there and is shared with the local-multiprocess BR
worker path.
"""
import argparse
import json
import os
import sys


def _peek_torch_device_argv(argv):
    """
    Mirror new_br_worker's pre-import device peek so ``--device cpu`` hides
    GPUs from PyTorch BEFORE torch is imported (CUDA_VISIBLE_DEVICES has to
    be set early). The shared_config_json carries `device`; we extract just
    that key here without doing a full json.loads of the whole blob into
    Python objects we don't need yet.
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
        description="Run a single dedicated BR matchup as one process. "
                    "Invoked by br_slurm_orchestrator.py via sbatch."
    )
    # Per-job-unique fields.
    parser.add_argument("--task_file", type=str, required=True,
                        help="Path to the .task checkpoint file (in the "
                             "orchestrator's processing dir).")
    parser.add_argument("--local_plot_dir", type=str, required=True,
                        help="Path to the local entropy plotting dir -- should be $WORKDIR/$JOBID/FightLadder/logs/local_entropy_plots")
    parser.add_argument("--state", type=str, required=True,
                        help="Single retro state string for this dedicated "
                             "matchup (e.g. two_player/Ryu_left/...state).")
    parser.add_argument("--eval_prot", type=str, required=True,
                        choices=["True", "False"],
                        help="True = exploit ego (BR plays adv); "
                             "False = exploit adv (BR plays ego).")
    parser.add_argument("--replicate_idx", type=int, required=True,
                        help="Replicate index within this (state, side) bucket.")
    parser.add_argument("--br_index", type=int, required=True,
                        help="Global job index from _build_dedicated_job_specs.")
    parser.add_argument("--dedicated_job_id", type=int, required=True,
                        help="Same as br_index in dedicated mode; kept "
                             "separate to mirror run_br_for_task_in_subprocess.")
    parser.add_argument("--matchup_label", type=str, default=None,
                        help="Sanitized matchup label (e.g. ryu_vs_guile) "
                             "used in checkpoint / log file names.")
    parser.add_argument("--output_subdir", type=str, default="",
                        help="Per-training-process subfolder for "
                             "br_rewards/ and selfplay_rewards/ outputs.")
    parser.add_argument("--training_style", type=str, default="",
                        help="Style label (spar|ippo|league|...) embedded "
                             "in the local_br_eval reward filenames.")
    parser.add_argument("--is_league", type=str, required=True,
                        choices=["True", "False"],
                        help="True if the loaded checkpoint is a LeaguePPO; "
                             "False for CDS SPAR/IPPO.")
    parser.add_argument("--league_matchup_states", type=str, nargs="+",
                        default=None,
                        help="Full league roster of matchup states (used "
                             "when --is_league True).")
    # All shared args in one blob to avoid 40+ argparse duplications.
    parser.add_argument("--shared_config_json", type=str, required=True,
                        help="JSON blob of shared config: game_args, n_envs, "
                             "device, br_tracker_*, entropy_*, stagnation_*, "
                             "etc. Built once per task by the orchestrator.")
    args = parser.parse_args()

    eval_prot = _bool(args.eval_prot)
    is_league = _bool(args.is_league)
    cfg = json.loads(args.shared_config_json)

    # state_subset is a list because run_br_for_task_in_subprocess will
    # repeat it n_envs times to build the env's STATE list. For dedicated
    # mode this is a single state per job — one matchup, isolated.
    state_subset = [args.state]

    print(
        f"[br_single_matchup] task={os.path.basename(args.task_file)} "
        f"state={args.state} eval_prot={eval_prot} "
        f"replicate={args.replicate_idx} br_index={args.br_index} "
        f"is_league={is_league} output_subdir={args.output_subdir!r} "
        f"training_style={args.training_style!r}",
        flush=True,
    )

    run_br_for_task_in_subprocess(
        # required positionals
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
        # dedicated mode: from_scratch=True (BR PPO trained from scratch).
        from_scratch=True,
        exploiter_save_freq=cfg.get("exploiter_save_freq", 100000),
        # Required from the launcher; KeyError here means the orchestrator
        # was invoked without --br_training_steps (argparse should catch
        # that first; this is the second gate).
        br_training_steps=cfg["br_training_steps"],
        # Optional from the launcher; defaults True to preserve existing
        # behavior on launchers that don't set ENABLE_LOCAL_KL_PLOT.
        enable_local_kl_plot=cfg.get("enable_local_kl_plot", True),
        # BR convergence tracker
        br_tracker_patience=cfg.get("br_tracker_patience", 10),
        br_tracker_tolerance=cfg.get("br_tracker_tolerance", 1e-4),
        br_tracker_window_size=cfg.get("br_tracker_window_size", 50),
        use_br_reward_stagnation=cfg.get("use_br_reward_stagnation", True),
        use_br_entropy_stagnation=cfg.get("use_br_entropy_stagnation", True),
        br_use_slope_early_stop=cfg.get("br_use_slope_early_stop", False),
        br_slope_window=cfg.get("br_slope_window", 20),
        br_slope_tolerance=cfg.get("br_slope_tolerance", 5e-3),
        br_min_slope_checks=cfg.get("br_min_slope_checks", 10),
        # CDS continue-mode stagnation knobs (unused in dedicated mode but
        # passed for parity with the local worker so changing modes later
        # doesn't change behavior).
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
        # device + dispatch + bookkeeping
        device=cfg.get("device", "cuda"),
        state_subset=state_subset,
        matchup_label=args.matchup_label,
        replicate_idx=args.replicate_idx,
        dedicated_job_id=args.dedicated_job_id,
        manual_stop_file=cfg.get("manual_stop_file"),
        manual_stop_key=cfg.get("manual_stop_key"),
        launch_local_br_eval=cfg.get("launch_local_br_eval", True),
        periodic_eval_freq=cfg.get("periodic_eval_freq", 5_000_000),
        use_wandb=cfg.get("use_wandb", False),
        is_league=is_league,
        league_matchup_states=args.league_matchup_states,
        output_subdir=args.output_subdir,
        # entropy-window early-stop knobs (forwarded to Exploiter's tracker)
        entropy_stop_ratio=cfg.get("entropy_stop_ratio", 0.15),
        entropy_window_size=cfg.get("entropy_window_size", 50),
        entropy_warmup_checks=cfg.get("entropy_warmup_checks", 100),
        entropy_ratio_only=cfg.get("entropy_ratio_only", False),
        local_plot_dir=args.local_plot_dir
    )

    print(
        f"[br_single_matchup] DONE state={args.state} eval_prot={eval_prot} "
        f"replicate={args.replicate_idx}",
        flush=True,
    )


if __name__ == "__main__":
    main()
