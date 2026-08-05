#!/bin/bash
# Critic diagnostic suite over a checkpoint or a whole series.
#
# Tiers:
#   0  static      weights only, ~1 s, no env. Catches an untrained/random head.
#   1  prediction  held-out EV, calibration, baselines. NEEDS >=100 EPISODES.
#   2  representation  where in [CNN -> trunk -> head] the signal dies.
#   3  behavioral  LBR vs shuffled-critic vs greedy. Reads existing LBR sidecars.
#
# STEPS is the setting that matters most. EV(V,G) regresses against episode
# RETURNS, so effective sample size is the number of COMPLETED EPISODES, not
# timesteps. At ep_len ~350 and 12 envs, 5000 steps gives ~170 episodes; 800
# steps gives ~18 and every number is noise. The suite refuses to render a
# tier-1 verdict below 100 episodes rather than printing something misleading.
#
# Keep the arg set in sync with .vscode/launch.json.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIAG_PATH="${SCRIPT_DIR}/main/critic_diagnostics.py"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

# Exactly one of CKPT / SERIES. SERIES accepts a glob and also emits series.png.
CKPT=""
SERIES="${SCRIPT_DIR}/main/trained_models/tasks/todo/spar_Ry_Sa_*_steps.task"

TIERS="0,1,2,3"
STEPS="5000"             # ~170 episodes at 12 envs, ep_len ~350
N_ENVS="12"
SEED="0"
EVAL_PROT="True"
LBR_MATCHUPS="all"
DEVICE="cuda"
OUT="critic_diagnostics"

# Optional: training log, to recover the IN-BATCH EV and report the
# in-batch/held-out GAP. In-batch EV cannot be reconstructed from a checkpoint --
# it is measured on replay samples that are not saved -- so the log is the only
# faithful source and the gap is simply omitted without it.
LOG="${SCRIPT_DIR}/logs/ippo_worker_1.log"

NO_MLP="False"           # True -> skip nonlinear tier-2 probes (much faster)
TIER3_RUN="False"        # True -> run an LBR screen when no sidecar exists
TIER3_EPISODES="20"

# Cap BLAS threads so a concurrent LBR sweep is not starved. The ridge solves are
# 513x513; extra threads buy nothing and cost the neighbours a lot.
BLAS_THREADS="4"

CMD=( python -u "${DIAG_PATH}" )
if [ -n "${CKPT}" ]; then
    CMD+=( --ckpt "${CKPT}" )
else
    CMD+=( --series "${SERIES}" )
fi
CMD+=(
    --tiers "${TIERS}"
    --steps "${STEPS}"
    --n_envs "${N_ENVS}"
    --seed "${SEED}"
    --eval_prot "${EVAL_PROT}"
    --lbr_matchups "${LBR_MATCHUPS}"
    --device "${DEVICE}"
    --out "${OUT}"
    --tier3_episodes "${TIER3_EPISODES}"
    --blas_threads "${BLAS_THREADS}"
)
[ -f "${LOG}" ] && CMD+=( --log "${LOG}" )
[ "${NO_MLP}" = "True" ] && CMD+=( --no_mlp )
[ "${TIER3_RUN}" = "True" ] && CMD+=( --tier3_run )

echo "=== critic diagnostics ==="
echo "  target : ${CKPT:-${SERIES}}"
echo "  tiers  : ${TIERS}   steps: ${STEPS}   envs: ${N_ENVS}"
echo "  out    : main/${OUT}/"
exec "${CMD[@]}"
