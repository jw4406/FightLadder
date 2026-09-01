#!/bin/bash
set -u
PY=/home/jw4406/anaconda3/envs/fightladder/bin/python
WORKDIR=/home/jw4406/codebase/FightLadder
cd "$WORKDIR"
RUNDIR="$WORKDIR/main/spar_img_scaled_3M"
export FIGHTLADDER_TASK_DIR="$RUNDIR/trained_models/tasks"
mkdir -p "$FIGHTLADDER_TASK_DIR/todo" "$FIGHTLADDER_TASK_DIR/todo_continue"
SAVE_DIR="$FIGHTLADDER_TASK_DIR/todo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=(
  "$PY" -u "$WORKDIR/main/ippo.py"
  --player Ryu --opponents Sagat
  --c_lr 3e-5 --d_lr 1e-4 --v_lr 4e-4
  --ego_style learning --adv_style learning
  --save_dir "$SAVE_DIR"
  --num_env_to_load 1 --env_batch_size 24 --num_perturbs 10
  --use_mirror False --ego_side left --side both --envs_per_matchup 24
  --num_env_steps 512
  --vtrace_enabled True --vtrace_seq_len 64 --vtrace_c_bar 1.0
  --vtrace_rho_bar 5.0 --vtrace_replay_capacity 15000
  --checkpoint_interval 25000 --training_batch_size 1024
  --total_timesteps 3000000
  --transform_action True --model_arch_type spar
  --use_lr_annealing False --lr_anneal_coeff .995
  --render False --async_update False
  --obs_type image
  --reward_scale 0.001
  --gamma 0.94 --gae_lambda 0.95
)
echo "=== SHORT SPAR TEST: obs=image, reward_scale=0.001 (SCALED control), 3M steps ==="
echo "  task_dir : $FIGHTLADDER_TASK_DIR"
exec "${CMD[@]}"
