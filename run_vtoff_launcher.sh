#!/bin/bash
# Start all five vtoff runs, each as its OWN systemd --user transient unit
# (independent Main PID = the training python), memory-gated. This launcher is
# itself a unit; when it exits the run units keep going (they're separate).
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
export XDG_RUNTIME_DIR=/run/user/1004
MAIN=/home/jw4406/codebase/FightLadder/main
WD=/home/jw4406/codebase/FightLadder
BASIS=$MAIN/diag/basis_19680000_r4.npz
PZ=$WD/run_minimax_phase0.sh
TOTAL=24564; MARGIN=2500; NEED=9500
used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
freem(){ echo $((TOTAL - $(used))); }
CM="OBS_TYPE=image REWARD_SCALE=1.0 MINIMAX_HEAD=factored MINIMAX_EMBED=$BASIS MINIMAX_FREEZE_EMBED=True ACTIONABLE_STATUSES=512,514,520 TOTAL_TIMESTEPS=3000000 CHECKPOINT_INTERVAL=25000 FOREGROUND=1"

NAME=(test unsc3M dtjoint nodt vega)
PAYLOAD=(
 "exec bash $MAIN/run_spar_img_unscaled_test_vtoff.sh > $MAIN/spar_img_unscaled_test_vtoff/train.log 2>&1"
 "exec bash $MAIN/run_spar_img_unscaled_3M_vtoff.sh > $MAIN/spar_img_unscaled_3M_vtoff/train.log 2>&1"
 "export PLAYER=Ryu OPPONENTS=Sagat ENT_COEF=0.01 DSTB_ENT_COEF=0.01 DECISION_TIMING=joint DWELL_FRAMES=4 RUN_SUFFIX=dtjoint_ent_basis_vtoff $CM; exec bash $PZ vtoff"
 "export PLAYER=Ryu OPPONENTS=Sagat ENT_COEF=0.01 DSTB_ENT_COEF=0.01 DECISION_TIMING=off RUN_SUFFIX=nodt_ent_basis_vtoff $CM; exec bash $PZ vtoff"
 "export PLAYER=Vega OPPONENTS=Blanka ENT_COEF=0.05 DSTB_ENT_COEF=0.05 DECISION_TIMING=joint DWELL_FRAMES=4 RUN_SUFFIX=VegaBlanka_dtj_ent05_vtoff $CM; exec bash $PZ vtoff"
)
for i in "${!PAYLOAD[@]}"; do
  while [ "$(freem)" -lt "$NEED" ]; do echo "$(date +%H:%M:%S) [${NAME[$i]}] wait room free=$(freem) need=$NEED"; sleep 60; done
  echo "### $(date +%H:%M:%S) start unit vtoff_${NAME[$i]}  free=$(freem)"
  systemd-run --user --unit="vtoff_${NAME[$i]}" -p WorkingDirectory="$WD" \
      bash -c "${PAYLOAD[$i]}"
  sleep 200
  [ "$i" -eq 0 ] && { F=$(( $(used) - 120 )); NEED=$(( F + MARGIN )); echo "### footprint ~${F} NEED=${NEED}"; }
done
echo "### ALL FIVE STARTED as independent units @ $(date +%H:%M:%S)"
