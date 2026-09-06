#!/bin/bash
# COMPARISON to run_balrog_engage_e6_160M.sh: SAME zero-sum engagement shaping, but on the
# Vega-vs-Guile matchup (asymmetric footsie-char vs charge-zoner) instead of the Balrog mirror.
# Only the matchup and PRESSURE_RANGE differ, so the comparison isolates the matchup, not the
# shaping. Tests whether the engagement shaping produces rich contested play across matchup types.
#   RESET_CLOSE_RANGE=64  start each round within 64px (exactly zero-sum, equilibrium unchanged)
#   COUNTERHIT_KAPPA=1.0  counter-hits count 2x damage (zero-sum, raises the interaction/gamma)
#   PRESSURE_BETA=0.5     in-range-and-attacking bonus, ~0.5 HP/step (zero-sum, MOVES equilibrium)
#   ATTACK_STATUSES=516,522   derived non-actionable attack states (common Vega+Balrog)
#   PRESSURE_RANGE=90     Vega hit-spacing median ~90px (Balrog was ~120)
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
export MINIMAX_Q=False
export PLAYER=Vega OPPONENTS="Guile"
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
export PRESSURE_RANGE=90
export ATTACK_STATUSES=516,522
export SEED=0 CHECKPOINT_INTERVAL=166667 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=VegaGuile_dtj_ent05_egodecay9996_cps_e6_engage_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
