#!/bin/bash
set -u
cd /home/jw4406/codebase/FightLadder
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fightladder
python -c "import torch" 2>/dev/null || { echo "FATAL: no torch"; exit 3; }
CK=main/minimax_phase0_vton/trained_models/tasks/todo
for T in reward return; do
  echo "=== 12480000 target=${T} UNIFORM actions $(date '+%H:%M:%S') ==="
  timeout 2400 python -u main/minimax_probe_ceiling.py \
    --ckpt "${CK}/spar_Ry_Sa_12480000_steps.task" \
    --steps 6000 --n_envs 12 --device cuda --target "$T" --uniform_actions \
    --out "minimax_probe_${T}_unif12480000.json" \
    > "logs/probe_unif_${T}_12480000.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED rc=${rc} -- NOT a null result"; tail -3 "logs/probe_unif_${T}_12480000.log" | sed 's/^/    /'
  else
    sed -n '/PROBE CEILING/,$p' "logs/probe_unif_${T}_12480000.log" | head -12 | sed 's/^/  /'
  fi
done
echo "UNIFORM_PROBE_DONE"
