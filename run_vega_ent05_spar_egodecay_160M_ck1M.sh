#!/bin/bash
# FRESH spar ent05 training: Vega vs {Guile, Blanka} POOL, 160M steps, joint DT,
# multiplicative LR decay (ExponentialLR gamma=0.9998 per update => ~7% of initial
# LR by 160M / ~13k updates). From SCRATCH (no MODEL_FILE). Based on
# run_vega_ent05_cont.sh minus the warm-start. MINIMAX_Q=False: the phase0 driver's
# 22-action / 484-cell minimax head is INCOMPATIBLE with the current 65-action space
# (it CUDA-device-side-asserts), and it isn't needed for plain training -- so we skip
# it. Keeps minimax_net out of the checkpoints too.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh

export MINIMAX_Q=False
export EGO_LR_ANNEAL_COEFF=0.9996
export ENVS_PER_MATCHUP=12
export PLAYER=Vega OPPONENTS="Guile Blanka"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=True LR_ANNEAL_COEFF=0.9998
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export SEED=0 CHECKPOINT_INTERVAL=41667 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=VegaGuileBlanka_dtj_ent05_egodecay9996_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
