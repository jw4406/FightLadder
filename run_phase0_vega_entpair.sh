#!/bin/bash
# Vega-vs-Blanka phase0, SEQUENTIAL (one GPU): UNCONTROLLED entropy (ent=0) then
# ent_coef 0.02. Tests whether the decision-timing collapse is Ryu-SPECIFIC (a
# collapse INTO fireball-zoning, the "easy damage" attractor) -- melee chars have
# no dominant strategy, so ent=0 may hold entropy on its own.
set -u
cd /home/jw4406/codebase/FightLadder
BASIS=/home/jw4406/codebase/FightLadder/main/diag/basis_19680000_r4.npz
run_and_wait() {
    local suffix="$1" ent="$2"
    echo "########## launching ${suffix} (ent_coef=${ent}) ##########  $(date '+%H:%M:%S')"
    env PLAYER=Vega OPPONENTS=Blanka OBS_TYPE=image REWARD_SCALE=1.0 \
        ENT_COEF="$ent" DSTB_ENT_COEF="$ent" MINIMAX_HEAD=factored MINIMAX_EMBED="$BASIS" \
        MINIMAX_FREEZE_EMBED=True DECISION_TIMING=joint DWELL_FRAMES=4 \
        ACTIONABLE_STATUSES=512,514,520 TOTAL_TIMESTEPS=3000000 CHECKPOINT_INTERVAL=25000 \
        RUN_SUFFIX="$suffix" bash run_minimax_phase0.sh vton
    sleep 40
    while pgrep -f "minimax_phase0_vton_image_rs1.0_${suffix}/trained_models" >/dev/null 2>&1; do
        sleep 60
    done
    echo "########## ${suffix} FINISHED ##########  $(date '+%H:%M:%S')"
}
run_and_wait VegaBlanka_dtj_ent00 0.0
run_and_wait VegaBlanka_dtj_ent02 0.02
echo "VEGA_ENTPAIR_DONE"
