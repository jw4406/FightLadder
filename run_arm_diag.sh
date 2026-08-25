#!/usr/bin/env bash
# Full diagnostic suite for one arm. Usage: run_arm_diag.sh <arm_suffix> <steps> <reward_scale> [label]
# Reports the CONFOUND-ROBUST metric head_EV / EV_max (fraction of the achievable
# ceiling the head captures) so a one-sided regime's inflated raw EV is normalised
# out. Correlation metrics (corrW, paircorr, gamma_share) are scale-invariant;
# value_gap is scale-matched via --reward_scale.
set -uo pipefail
SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"; cd "${SD}/main"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fightladder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM="$1"; CK="$2"; RS="${3:-0.001}"; LBL="${4:-$ARM}"
DIR="minimax_phase0_vtoff_rammasked_${ARM}/trained_models/tasks/todo"
[ -d "minimax_phase0_vtoff_rammasked_${ARM}" ] || DIR="minimax_phase0_vtoff_popart_rammasked_${ARM}/trained_models/tasks/todo"
CKPT="${DIR}/spar_Ry_Sa_${CK}_steps.task"
RES="${SD}/RESULTS_LIVE.md"; H=headroom
say() { echo "$*" | tee -a "$RES"; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
while [ "$(gpu_free)" -lt 6 ]; do sleep 60; done

say ""; say "# ===== ARM DIAGNOSTICS: ${LBL}  (reward_scale=${RS})  $(date '+%m-%d %H:%M') ====="
[ -f "$CKPT" ] || { say "  MISSING checkpoint $CKPT"; exit 1; }
say '```'

say "-- REGIME (who wins; gates the value read):"
python3 outcome_balance.py --ckpt "$CKPT" --ram_mask ram_mask.npy --episodes 250 \
  --out $H/diag_${ARM}_outcome.json 2>/dev/null | grep -E "ego win|hp diff|=>" | tee -a "$RES"

say "-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):"
# NB: default 4-gamma string 0.75,0.9,0.94,0.99 -- value_ceiling's self-check
# hard-indexes G[:,:,2], so a single --gammas 0.94 crashes with IndexError.
python3 value_ceiling.py --ckpt "$CKPT" --ram_mask ram_mask.npy --n_states 40 --k 16 \
  --horizon 80 --n_envs 16 --out $H/diag_${ARM}_ceiling.json 2>>"$H/diag_${ARM}_ceiling.err" \
  | grep -E "\[check\]|EV_MAX|^  0\.94" | tee -a "$RES"
[ -f "$H/diag_${ARM}_ceiling.json" ] || say "  !! value_ceiling FAILED (see $H/diag_${ARM}_ceiling.err)"

say "-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:"
python3 value_gap.py --ckpt "$CKPT" --ram_mask ram_mask.npy --episodes 150 --max_steps 3500 \
  --n_envs 16 --gammas 0.94 --reward_scale "$RS" --out $H/diag_${ARM}_vg.json 2>>"$H/diag_${ARM}_vg.err" >/dev/null
[ -f "$H/diag_${ARM}_vg.json" ] || say "  !! value_gap FAILED (see $H/diag_${ARM}_vg.err)"
python3 - "$H/diag_${ARM}_vg.json" "$H/diag_${ARM}_ceiling.json" <<'PY' 2>/dev/null | tee -a "$RES"
import json,sys
vg=json.load(open(sys.argv[1]))['arms'][0]
try: cap=[r for r in json.load(open(sys.argv[2]))['rows'] if abs(r['gamma']-0.94)<1e-9][0]['ev_max']
except Exception: cap=float('nan')
he=vg['head_ev']
print(f"  head_EV={he:+.4f}  EV_max={cap:+.4f}  head_EV/EV_max={he/cap if cap==cap and cap>0 else float('nan'):.2f}  (n={vg['n_test']})")
PY

say "-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):"
python3 paircorr_onpolicy.py --ckpt "$CKPT" --ram_mask ram_mask.npy --n_states 250 --n_envs 16 \
  --out $H/diag_${ARM}_paircorr.json 2>/dev/null | grep -E "ON-POLICY|active" | tee -a "$RES"

say "-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)..."
# npz is step-numbered (diag_<arm>_<step>_raw.npz) so head_quality's _(\d+)_raw
# regex resolves the checkpoint. --reward_scale $RS matches the head's scale so
# evW is meaningful on UNSCALED arms (else evW = -1e6 garbage; corrW survives).
NPZ="$H/diag_${ARM}_${CK}_raw.npz"
# --bootstrap --horizon 0 --n_paths 1 pins the LEGACY r + gamma*V(s') leaf:
# head_quality.py scores the head against M, so this pipeline needs the
# critic-bootstrap payoff (not the new pure-MC default). Same M, same speed.
FIGHTLADDER_RAM_STRIDE=8 python3 bootstrap_delta.py --ckpt "$CKPT" --ram_mask ram_mask.npy \
  --n_states 300 --stride 40 --n_envs 6 --reward_scale "$RS" --save_obs \
  --bootstrap --horizon 0 --n_paths 1 \
  --out "$H/diag_${ARM}_${CK}.json" 2>>"$H/diag_${ARM}_enum.err" | tail -1 | tee -a "$RES"
[ -f "$NPZ" ] || say "  !! bootstrap_delta FAILED (see $H/diag_${ARM}_enum.err)"
python3 head_quality.py --run_dir "$DIR" --npz_glob "$NPZ" \
  --ram_mask ram_mask.npy 2>>"$H/diag_${ARM}_hq.err" | grep -iE "corrW|evW|CONST|HEADROOM|,999,|,999," | tee -a "$RES"
say "-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):"
python3 q_decompose.py --npz "$NPZ" 2>/dev/null | grep -iE "mu|alpha|beta|gamma|share" | tee -a "$RES"

say "-- FACTORED-HEAD training metrics (final log dump):"
L="${SD}/logs/minimax_phase0_vtoff_rammasked_${ARM}.log"; [ -f "$L" ] || L="${SD}/logs/minimax_phase0_vtoff_popart_rammasked_${ARM}.log"
for m in minimax_fx_gamma_share minimax_fx_w_norm minimax_fx_anti_share minimax_ev_ego minimax_ev_adv minimax_q_branch_std minimax_target_corr minimax_coverage; do
  v=$(grep "$m " "$L" 2>/dev/null | tail -1 | grep -oE '[-0-9.e+]+ *\|$' | tr -dc '0-9.e+-')
  [ -n "$v" ] && echo "  ${m} = ${v}" | tee -a "$RES"
done
say '```'
say "ARM_DIAG_DONE ${LBL} $(date '+%H:%M')"
echo "ARM_DIAG_DONE_${ARM}"
