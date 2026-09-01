#!/bin/bash
# +160M continuation of ippo ent05 from its 160M-cumulative continuation final.
# CONTRAST arm for the mixture experiment: ippo is entropy-COLLAPSED, so per
# [[iterate-averaging-mixture-plan]] its mixture should NOT reduce exploitability
# (every iterate is the same dead mode). Same config, LRs uniform 3e-5, NO annealing.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
CKPT=/home/jw4406/codebase/FightLadder/main/minimax_phase0_vtoff_image_ippo_rs1.0_VegaBlanka_dtj_ent05_cont/trained_models/tasks/todo/ippo_Ve_Bl_78720000_steps.task

export MODEL_ARCH=ippo
export PLAYER=Vega OPPONENTS=Blanka
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export C_LR=6e-6 D_LR=6e-6 V_LR=6e-6 GAMMA=0.94 GAE_LAMBDA=0.95 USE_LR_ANNEALING=False  # all LRs /5 vs cont (uniform)
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export SEED=0 CHECKPOINT_INTERVAL=80000 TOTAL_TIMESTEPS=160000000
export MODEL_FILE="$CKPT" REINIT_EGO=False
export RUN_SUFFIX=VegaBlanka_dtj_ent05_cont2
export FOREGROUND=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
