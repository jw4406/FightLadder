#!/bin/bash
# Dedicated-BR orchestrator for the cont2 sweep. Consumes the shared br_todo_cont2
# queue (fed by run_br_cont2_watcher.sh at 19.2M spacing), runs br0+br1 dedicated
# exploiters per checkpoint, arch auto-detected (spar vs ippo). Serialized
# (max_local_concurrent=1) so exactly ONE BR job shares the GPU with the 2 live
# trainings. 2M .learn() steps/seat to match the earlier matched sweep.
set -euo pipefail
source /home/jw4406/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

WORKDIR=/home/jw4406
MAIN_TRAINING_DIR=codebase
REPO_DIR=/home/jw4406/codebase/FightLadder
BR_ROOT="$REPO_DIR/main/br_todo_cont2"
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
    --step_stride 0
