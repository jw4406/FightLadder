#!/bin/bash
# DISAMBIGUATION Test B: dedicated BR at reward_scale=1.0 + decision_timing=JOINT
# (dwell 4, actionable 512,514,520) to test whether the canonical Aug-26 sweep
# used joint timing (would reproduce 19.2M br0=138.11). Separate queue + output
# from Test A (off). Serialized, one BR at a time.
set -euo pipefail
source /home/jw4406/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

WORKDIR=/home/jw4406
MAIN_TRAINING_DIR=codebase
REPO_DIR=/home/jw4406/codebase/FightLadder
BR_ROOT="$REPO_DIR/main/br_test_joint"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -u "$REPO_DIR/main/br_slurm_orchestrator.py" \
    --br_dedicated_sh_template "$REPO_DIR/ws_launch_files/br_dedicated_template.sh" \
    --main_training_dir "$MAIN_TRAINING_DIR" \
    --workdir "$WORKDIR" \
    --todo_dir "$BR_ROOT/todo" \
    --processing_dir "$BR_ROOT/processing" \
    --done_dir "$BR_ROOT/done" \
    --stop_file "$BR_ROOT/STOP" \
    --local_plot_dir "$REPO_DIR/logs/local_entropy_plots" \
    --slurm_log_dir /home/jw4406 \
    --max_local_concurrent 1 \
    --br_training_steps 2000000 \
    --reward_scale 1.0 \
    --decision_timing joint \
    --actionable_statuses 512,514,520 \
    --dwell_frames 4 \
    --step_stride 0
