#!/bin/bash
# +160M continuation of spar ent05 from its 160M-cumulative continuation final, to
# EXTEND the oscillating orbit for the strategy-space mixture experiment
# ([[iterate-averaging-mixture-plan]]). Same config -- NO annealing / weight-averaging:
# we WANT it to keep oscillating so the iterates stay diverse to mix over.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
CKPT=/home/jw4406/codebase/FightLadder/main/minimax_phase0_vtoff_image_rs1.0_VegaBlanka_dtj_ent05_vtoff_cont/trained_models/tasks/todo/spar_Ve_Bl_78720000_steps.task

export PLAYER=Vega OPPONENTS=Blanka
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export C_LR=6e-6 D_LR=2e-5 V_LR=8e-5 GAMMA=0.94 GAE_LAMBDA=0.95 USE_LR_ANNEALING=False  # all LRs /5 vs cont
export MINIMAX_HEAD=factored MINIMAX_RANK=4 MINIMAX_W_INIT=0.01
export MINIMAX_EMBED=/home/jw4406/codebase/FightLadder/main/diag/basis_19680000_r4.npz MINIMAX_FREEZE_EMBED=True
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export SEED=0 CHECKPOINT_INTERVAL=80000 TOTAL_TIMESTEPS=160000000
export MODEL_FILE="$CKPT" REINIT_EGO=False
export RUN_SUFFIX=VegaBlanka_dtj_ent05_vtoff_cont2
export FOREGROUND=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
