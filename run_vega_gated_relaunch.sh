#!/bin/bash
# Wait until the GPU has real headroom (unsc3M finishing provides it), then start
# vtoff_vega as its own unit with expandable_segments to avoid the buffer-prep OOM.
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
export XDG_RUNTIME_DIR=/run/user/1004
WD=/home/jw4406/codebase/FightLadder; PZ=$WD/run_minimax_phase0.sh; BASIS=$WD/main/diag/basis_19680000_r4.npz
TOTAL=24564; NEED=9000
freem(){ echo $((TOTAL - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits))); }
while [ "$(freem)" -lt "$NEED" ]; do
  echo "$(date +%H:%M:%S) waiting: unsc3M=$(systemctl --user is-active vtoff_unsc3M 2>/dev/null) free=$(freem) need=$NEED"
  sleep 30
done
echo "$(date +%H:%M:%S) headroom ok (free=$(freem)) -> launching vega with expandable_segments"
systemd-run --user --unit=vtoff_vega -p WorkingDirectory="$WD" bash -c \
"export PLAYER=Vega OPPONENTS=Blanka ENT_COEF=0.05 DSTB_ENT_COEF=0.05 DECISION_TIMING=joint DWELL_FRAMES=4 RUN_SUFFIX=VegaBlanka_dtj_ent05_vtoff OBS_TYPE=image REWARD_SCALE=1.0 MINIMAX_HEAD=factored MINIMAX_EMBED=$BASIS MINIMAX_FREEZE_EMBED=True ACTIONABLE_STATUSES=512,514,520 TOTAL_TIMESTEPS=3000000 CHECKPOINT_INTERVAL=25000 FOREGROUND=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; exec bash $PZ vtoff"
echo "$(date +%H:%M:%S) vega launch issued"
