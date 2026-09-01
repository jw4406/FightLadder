#!/bin/bash
# RESUME of the ent03 continuation after an external `pkill -f python` killed it
# at ~39.24M cont (2026-08-26 14:53). Resumes from ent03's latest saved checkpoint
# (38.40M cont = 117.12M cumulative) for the remaining 41.6M steps to reach the
# originally-planned 80M continuation total. ent_coef stays 0.03 (both seats).
# NEW RUN_SUFFIX (_cont2) so the 1.92M-38.4M checkpoints in the _cont dir are not
# overwritten (set_parameters restarts num_timesteps at 0). For the cumulative
# curve: _cont offset +78.72M, _cont2 offset +78.72M+38.4M = +117.12M.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
CKPT=/home/jw4406/codebase/FightLadder/main/minimax_phase0_vtoff_image_rs1.0_VegaBlanka_dtj_ent03_vtoff_cont/trained_models/tasks/todo/spar_Ve_Bl_38400000_steps.task

export PLAYER=Vega OPPONENTS=Blanka
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95 USE_LR_ANNEALING=False
export MINIMAX_HEAD=factored MINIMAX_RANK=4 MINIMAX_W_INIT=0.01
export MINIMAX_EMBED=/home/jw4406/codebase/FightLadder/main/diag/basis_19680000_r4.npz MINIMAX_FREEZE_EMBED=True
export ENT_COEF=0.03 DSTB_ENT_COEF=0.03
export SEED=0 CHECKPOINT_INTERVAL=80000 TOTAL_TIMESTEPS=41600000
export MODEL_FILE="$CKPT" REINIT_EGO=False
export RUN_SUFFIX=VegaBlanka_dtj_ent03_vtoff_cont2
export FOREGROUND=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
