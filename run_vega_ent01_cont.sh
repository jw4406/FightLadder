#!/bin/bash
# Continue the ent05 arm (Vega/Blanka, vtoff, image obs, decision-timing joint)
# from its 78.72M FINAL checkpoint, with the entropy coefficient dropped
# 0.05 -> 0.01 on BOTH seats (ego ent_coef AND adversary dstb_ent_coef).
# A PARALLEL fork of ent05 (same MODEL_FILE as the ent03 continuation), one step
# closer to the ent00 collapse -- fills the entropy sweep 0.05 / 0.03 / 0.01 / 0.0.
# +80,000,000 additional steps (num_timesteps restarts at 0 under the
# set_parameters warm-start, so TOTAL_TIMESTEPS == steps to ADD).
#
# Every other knob is byte-identical to the original ent05 run (see
# run_vega_ent03_cont.sh); only ENT_COEF / DSTB_ENT_COEF differ. NEW RUN_SUFFIX
# => new task dir, so ent05/ent03 are neither overwritten nor collision-guarded.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
CKPT=/home/jw4406/codebase/FightLadder/main/minimax_phase0_vtoff_image_rs1.0_VegaBlanka_dtj_ent05_vtoff_80M/trained_models/tasks/todo/spar_Ve_Bl_78720000_steps.task

export PLAYER=Vega OPPONENTS=Blanka
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95 USE_LR_ANNEALING=False
export MINIMAX_HEAD=factored MINIMAX_RANK=4 MINIMAX_W_INIT=0.01
export MINIMAX_EMBED=/home/jw4406/codebase/FightLadder/main/diag/basis_19680000_r4.npz MINIMAX_FREEZE_EMBED=True
export ENT_COEF=0.01 DSTB_ENT_COEF=0.01
export SEED=0 CHECKPOINT_INTERVAL=80000 TOTAL_TIMESTEPS=80000000
export MODEL_FILE="$CKPT" REINIT_EGO=False
export RUN_SUFFIX=VegaBlanka_dtj_ent01_vtoff_cont
export FOREGROUND=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
