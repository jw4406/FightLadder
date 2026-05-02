#!/bin/bash
#SBATCH --job-name=br_spar_Gu_SaRy_GuileVsRyu_ego_rep1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/jw4406/codebase/FightLadder/slurm_logs/spar_Gu_SaRy_200000_steps/br_spar_Gu_SaRy_GuileVsRyu_ego_rep1.out
#SBATCH --error=/home/jw4406/codebase/FightLadder/slurm_logs/spar_Gu_SaRy_200000_steps/br_spar_Gu_SaRy_GuileVsRyu_ego_rep1.err

set -euo pipefail
cd /home/jw4406/codebase/FightLadder

python /home/jw4406/codebase/FightLadder/main/br_single_matchup.py --task_file /home/jw4406/codebase/FightLadder/main/trained_models/tasks/slurm_processing/spar_Gu_SaRy_200000_steps.task_folder/spar_Gu_SaRy_200000_steps.task --state two_player/Guile_left/Champion.Level1.GuileVsRyu.2Player.state --eval_prot True --replicate_idx 1 --br_index 11 --dedicated_job_id 11 --output_subdir spar_Gu_SaRy --training_style spar --is_league False --matchup_label GuileVsRyu --shared_config_json '{"game_args": {"reset": "round", "side": "both", "render": false, "enable_combo": true, "null_combo": false, "transform_action": false, "seed": 0}, "use_mirror": false, "eval_only": false, "proj_name": "br_training_slurm", "analysis_upload_proj_name": "br_analysis", "n_envs": 2, "exploiter_save_freq": 100000, "br_tracker_patience": 300, "br_tracker_tolerance": 0.0001, "br_tracker_window_size": 50, "use_br_reward_stagnation": false, "use_br_entropy_stagnation": true, "br_use_slope_early_stop": false, "br_slope_window": 20, "br_slope_tolerance": 0.005, "br_min_slope_checks": 12, "use_stagnation_early_stop": true, "use_stagnation_velocity_signal": false, "use_stagnation_entropy_signal": true, "stagnation_patience": 2000, "stagnation_tolerance": 0.0001, "stagnation_rel_tolerance": 0.05, "stagnation_ema_beta": 0.99, "stagnation_eps": 1e-08, "stagnation_eval_games": 20, "entropy_stagnation_weight": 1.0, "stagnation_lr_factor": 0.999, "stagnation_lr_patience": 1000, "stagnation_use_slope_early_stop": false, "stagnation_slope_window": 20, "stagnation_slope_tolerance": 0.005, "stagnation_min_slope_checks": 12, "entropy_stop_ratio": 0.05, "entropy_window_size": 50, "entropy_warmup_checks": 100, "device": "cuda", "manual_stop_file": "/home/jw4406/codebase/FightLadder/main/trained_models/tasks/stop/STOP_spar_Gu_SaRy_200000_steps", "manual_stop_key": null, "launch_local_br_eval": true, "use_wandb": false}'
