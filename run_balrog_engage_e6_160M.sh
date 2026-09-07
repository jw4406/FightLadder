#!/bin/bash
# DIAGNOSTIC: Balrog mirror + ZERO-SUM engagement shaping, to break the passive-standoff
# attractor (contact fell 11.5%@1M -> 8%@3M in the plain Balrog mirror). All three levers
# keep the game zero-sum (aggresive_coeff stays 1.0), so exploitability stays well-defined:
#   RESET_CLOSE_RANGE=64  start each round within 64px. EXACTLY zero-sum (touches no reward,
#                         equilibrium unchanged); measured contact 3.4%->18.8% from close starts.
#                         Caveat: agents can still retreat, so contact-survival is the open Q.
#   COUNTERHIT_KAPPA=1.0  reward counter-hits (damage while the RECIPIENT is attacking). Zero-sum
#                         at ac=1; raises the INTERACTION (gamma) specifically = rich joint play.
#   PRESSURE_BETA=0.5     antisymmetric in-range-and-attacking bonus, additive in HP units (a hit is
#                         ~5-30 HP, so 0.5/step is a real dense nudge; 0.1 would be negligible).
#                         Zero-sum but NOT potential-based -> it MOVES the equilibrium, so a strict
#                         before/after exploitability read is only clean for the other two levers.
#   TRADE_KAPPA=0.0       intentionally OFF (only fires when BOTH already attacking -> can't pull a
#                         disengaged standoff together; it's a later "reward the exchange" layer).
# ATTACK_STATUSES=516,522 derived (attack_status_probe.py): the non-actionable, damage-associated
#                         status codes common to Vega+Balrog (drops the actionable 512/514/520).
# PRESSURE_RANGE=120      Balrog hit-spacing median ~121px (Vega would be ~90). "In threatening range."
# NOTE magnitudes (ch=1.0, pb=0.1) are STARTING points -- tune. For a clean single-lever run, zero
# the others. To retarget to Vega: PLAYER=Vega OPPONENTS="Guile Blanka", PRESSURE_RANGE=90.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
export MINIMAX_Q=False
export PLAYER=Balrog OPPONENTS="Balrog"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export ENVS_PER_MATCHUP=6
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=True LR_ANNEAL_COEFF=0.9998
export EGO_LR_ANNEAL_COEFF=0.9996
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export CHARGE_PRESERVING_SKIP=True
# --- zero-sum engagement shaping (aggresive_coeff stays 1.0) ---
export AGGRESIVE_COEFF=1.0
export RESET_CLOSE_RANGE=64
export COUNTERHIT_KAPPA=1.0
export PRESSURE_BETA=0.5
export PRESSURE_RANGE=120
export ATTACK_STATUSES=516,522
export SEED=0 CHECKPOINT_INTERVAL=166667 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=BalrogMirror_dtj_ent05_egodecay9996_cps_e6_engage_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
