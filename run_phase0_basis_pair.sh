#!/bin/bash
# Phase0 factored-head pair, SEQUENTIAL (one GPU). Both use the computed rank-4
# gamma basis (63.7% capture) so the head HAS capacity for the interaction; the
# ONLY difference is decision-timing. If gamma_share grows in TREATMENT but not
# CONTROL, decision-timing is what feeds the head real action-signal.
set -u
cd /home/jw4406/codebase/FightLadder
BASIS=/home/jw4406/codebase/FightLadder/main/diag/basis_19680000_r4.npz

run_and_wait() {
    local suffix="$1"; shift
    echo "########## launching ${suffix} ##########  $(date '+%H:%M:%S')"
    env OBS_TYPE=image REWARD_SCALE=1.0 ENT_COEF=0.01 DSTB_ENT_COEF=0.01 \
        MINIMAX_HEAD=factored MINIMAX_EMBED="$BASIS" MINIMAX_FREEZE_EMBED=True \
        TOTAL_TIMESTEPS=3000000 CHECKPOINT_INTERVAL=25000 \
        RUN_SUFFIX="$suffix" "$@" \
        bash run_minimax_phase0.sh vton
    sleep 40   # let the script spawn its backgrounded python
    while pgrep -f "minimax_phase0_vton_image_rs1.0_${suffix}/trained_models" >/dev/null 2>&1; do
        sleep 60
    done
    echo "########## ${suffix} FINISHED ##########  $(date '+%H:%M:%S')"
}

run_and_wait dtjoint_ent_basis DECISION_TIMING=joint DWELL_FRAMES=4
run_and_wait nodt_ent_basis    DECISION_TIMING=off
echo "PHASE0_PAIR_DONE"
