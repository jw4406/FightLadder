#!/bin/bash
# Matched continuation of the IPPO ent05 run, to compare against the spar ent05_cont.
# Continue ippo_ent05 from its 80M final checkpoint (79.68M) at ent_coef 0.05, +80M,
# via the SAME warm-start mechanism as the spar side (set_parameters -> optimizer
# reset -> num_timesteps restarts at 0, so TOTAL_TIMESTEPS is steps to ADD). All LRs
# uniform 3e-5. 1.92M checkpoint grid to line up with ent05_cont. NEW RUN_SUFFIX.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
CKPT=/home/jw4406/codebase/FightLadder/main/minimax_phase0_vtoff_image_ippo_rs1.0_VegaBlanka_dtj_ent05/trained_models/tasks/todo/ippo_Ve_Bl_79680000_steps.task

export MODEL_ARCH=ippo
export PLAYER=Vega OPPONENTS=Blanka
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export C_LR=3e-5 D_LR=3e-5 V_LR=3e-5 GAMMA=0.94 GAE_LAMBDA=0.95 USE_LR_ANNEALING=False
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export SEED=0 CHECKPOINT_INTERVAL=80000 TOTAL_TIMESTEPS=80000000
export MODEL_FILE="$CKPT" REINIT_EGO=False
export RUN_SUFFIX=VegaBlanka_dtj_ent05_cont
export FOREGROUND=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
