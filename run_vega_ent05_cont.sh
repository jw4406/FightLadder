#!/bin/bash
# CONTROL for the entropy-cliff sweep: continue the ent05 arm at its OWN entropy
# (ent_coef 0.05, UNCHANGED) from the 78.72M final checkpoint for +80M steps.
# This is the ent05-continued-at-0.05 control that was missing when disentangling
# "lower entropy" from "more training": if this stays combo-ROBUST while the ent03
# (0.03) continuation became comboable, the entropy drop is cleanly the cause.
# Warm-start via set_parameters (num_timesteps restarts at 0 => TOTAL_TIMESTEPS is
# steps to ADD). Everything byte-identical to ent05; NEW RUN_SUFFIX -> new task dir.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
CKPT=/home/jw4406/codebase/FightLadder/main/minimax_phase0_vtoff_image_rs1.0_VegaBlanka_dtj_ent05_vtoff_80M/trained_models/tasks/todo/spar_Ve_Bl_78720000_steps.task

export PLAYER=Vega OPPONENTS=Blanka
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95 USE_LR_ANNEALING=False
export MINIMAX_HEAD=factored MINIMAX_RANK=4 MINIMAX_W_INIT=0.01
export MINIMAX_EMBED=/home/jw4406/codebase/FightLadder/main/diag/basis_19680000_r4.npz MINIMAX_FREEZE_EMBED=True
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export SEED=0 CHECKPOINT_INTERVAL=80000 TOTAL_TIMESTEPS=80000000
export MODEL_FILE="$CKPT" REINIT_EGO=False
export RUN_SUFFIX=VegaBlanka_dtj_ent05_vtoff_cont
export FOREGROUND=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
