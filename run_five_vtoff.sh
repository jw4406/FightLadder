#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
# Launch all five image+unscaled runs as VTOFF, packing as many onto the GPU as
# fit (memory-gated: launch next only when footprint+margin is free).
set -u
cd /home/jw4406/codebase/FightLadder
BASIS=/home/jw4406/codebase/FightLadder/main/diag/basis_19680000_r4.npz
TOTAL=24564; MARGIN=2500
used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
freem(){ echo $((TOTAL - $(used))); }
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
COM="OBS_TYPE=image REWARD_SCALE=1.0 MINIMAX_HEAD=factored MINIMAX_EMBED=$BASIS MINIMAX_FREEZE_EMBED=True ACTIONABLE_STATUSES=512,514,520 TOTAL_TIMESTEPS=3000000 CHECKPOINT_INTERVAL=25000"

# name|command  (standalone runs redirect to their own train.log; phase0 script self-logs)
declare -a NAME=(test3 unsc3M dtjoint nodt vega)
declare -a CMD=(
 "cd main && bash run_spar_img_unscaled_test_vtoff.sh > spar_img_unscaled_test_vtoff/train.log 2>&1"
 "cd main && bash run_spar_img_unscaled_3M_vtoff.sh  > spar_img_unscaled_3M_vtoff/train.log 2>&1"
 "env PLAYER=Ryu OPPONENTS=Sagat ENT_COEF=0.01 DSTB_ENT_COEF=0.01 DECISION_TIMING=joint DWELL_FRAMES=4 $COM RUN_SUFFIX=dtjoint_ent_basis_vtoff bash $PZ vtoff"
 "env PLAYER=Ryu OPPONENTS=Sagat ENT_COEF=0.01 DSTB_ENT_COEF=0.01 DECISION_TIMING=off               $COM RUN_SUFFIX=nodt_ent_basis_vtoff    bash $PZ vtoff"
 "env PLAYER=Vega OPPONENTS=Blanka ENT_COEF=0.05 DSTB_ENT_COEF=0.05 DECISION_TIMING=joint DWELL_FRAMES=4 $COM RUN_SUFFIX=VegaBlanka_dtj_ent05_vtoff bash $PZ vtoff"
)
NEED=9500  # conservative until run1 measured
for i in "${!CMD[@]}"; do
  while [ "$(freem)" -lt "$NEED" ]; do
    echo "$(date +%H:%M:%S) [${NAME[$i]}] waiting for room: free=$(freem) need=$NEED"; sleep 60
  done
  echo "### $(date +%H:%M:%S) launch ${NAME[$i]} ($((i+1))/5)  free=$(freem)"
  ( eval "${CMD[$i]}" ) >/dev/null 2>&1 &
  sleep 200   # boot + first update -> steady memory
  if [ "$i" -eq 0 ]; then F=$(( $(used) - 120 )); NEED=$(( F + MARGIN )); echo "### run1 footprint ~${F} MiB -> NEED=${NEED}"; fi
done
echo "### ALL FIVE LAUNCHED @ $(date +%H:%M:%S)  total_used=$(used)"
