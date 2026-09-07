#!/bin/bash
# MIRROR-CONFOUND TEST: ChunLi (ego/left) vs Balrog (opponent/right) -- a NON-mirror with the
# SAME zero-sum engagement shaping as run_balrog_engage / run_vega_guile_engage. Distinct
# characters per seat means the seat identity is unambiguous, so if learning is cleaner here
# than in the Balrog MIRROR (identical chars both sides), the mirror was confounding the policy
# about which side it controls.
#   reset_close_range=64  counterhit_kappa=1.0  pressure_beta=0.5  trade_kappa=0  aggresive_coeff=1.0
#   PRESSURE_RANGE=100    middle (ChunLi medium-range pokes; Vega~90, Balrog~120)
#   ATTACK_STATUSES=516,522  ASSUMPTION: derived for Vega+Balrog (common). ChunLi not yet derived
#                            (no ChunLi policy existed) -- if counterhit/pressure seem inert on the
#                            ego, re-derive with attack_status_probe.py on a ChunLi checkpoint.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
export MINIMAX_Q=False
export PLAYER=ChunLi OPPONENTS="Balrog"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export ENVS_PER_MATCHUP=6
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=True LR_ANNEAL_COEFF=0.9998
export EGO_LR_ANNEAL_COEFF=0.9996
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export CHARGE_PRESERVING_SKIP=True
# --- zero-sum engagement shaping (matched to balrog_engage; aggresive_coeff stays 1.0) ---
export AGGRESIVE_COEFF=1.0
export RESET_CLOSE_RANGE=64
export COUNTERHIT_KAPPA=1.0
export PRESSURE_BETA=0.5
export PRESSURE_RANGE=93
export ATTACK_STATUSES=516,522,524
export SEED=0 CHECKPOINT_INTERVAL=166667 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=ChunLiVsBalrog_dtj_ent05_egodecay9996_cps_e6_engage_rd_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
