#!/bin/bash
# Probe the LATEST checkpoints that pass BOTH gate conditions. Beyond 12.48M this
# run drifts to 410-470 steps/episode, and at that episode length the action
# contributes nothing to the return by construction (measured: +0.9636 state-only
# vs +0.9632 with a full per-cell slope), so a probe there measures the timer,
# not the hypothesis.
set -u
cd /home/jw4406/codebase/FightLadder
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fightladder
python -c "import torch" 2>/dev/null || { echo "FATAL: no torch"; exit 3; }
CK=main/minimax_phase0_vton/trained_models/tasks/todo
for STEP in 11040000 12000000 12480000; do
  for T in reward return; do
    echo "=== ${STEP} target=${T} $(date '+%H:%M:%S') ==="
    timeout 2400 python -u main/minimax_probe_ceiling.py \
      --ckpt "${CK}/spar_Ry_Sa_${STEP}_steps.task" \
      --steps 6000 --n_envs 12 --device cuda --target "$T" \
      --out "minimax_probe_${T}_vton${STEP}.json" \
      > "logs/probe_${T}_${STEP}.log" 2>&1
    rc=$?
    # FAIL LOUD. A crashed probe must never be read as a null result -- that
    # already happened once, when two ModuleNotFoundError crashes rendered as a
    # confident "ACTION ADDS NOTHING".
    if [ $rc -ne 0 ]; then
      echo "  FAILED rc=${rc} -- NOT a null result"
      tail -3 "logs/probe_${T}_${STEP}.log" | sed 's/^/    /'
    else
      sed -n '/ACTION GAIN/,+1p' "logs/probe_${T}_${STEP}.log" | sed 's/^/  /'
    fi
  done
done
echo "PROBE_SERIES_DONE"
