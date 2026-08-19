#!/usr/bin/env bash
# Clean re-run of exp1/2/3 after the seed diagnosis.
# KEY CORRECTION: --seed does NOT change the trained policy (vctl and s1_vctlB are
# bitwise-identical policies). Training is deterministic given config, so there is
# no training-seed noise. The noise floor is EVAL variance: vctl scored at 3
# different --eval_seed values. A treatment difference is real only if it exceeds
# that band. GPU is free (all arms done) so nothing serialises/OOMs.
set -uo pipefail
SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"; cd "${SD}/main"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fightladder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RES="${SD}/RESULTS_LIVE.md"; D=minimax_phase0_vtoff; T=trained_models/tasks/todo
say() { echo "$*" | tee -a "$RES"; }
ck() { local d="${D}_rammasked_$1"; [ -d "$d" ] || d="${D}_popart_rammasked_$1"; ls "$d/$T/"*_$2_steps.task 2>/dev/null | head -1; }
vg() { # label ckpt gamma eval_seed
  python3 value_gap.py --ckpt "$2" --ram_mask ram_mask.npy --episodes 120 --max_steps 3500 \
    --n_envs 16 --gammas "$3" --eval_seed "$4" --out "headroom/clean_$1.json" >/dev/null 2>&1
  python3 -c "import json;d=json.load(open('headroom/clean_$1.json'));r=d['arms'][0];print('  %-24s head_EV=%+.4f  n=%d'%('$1',r['head_ev'],r['n_test']))" 2>/dev/null \
    || echo "  $1  FAILED"
}

say ""; say "# CLEAN RE-RUN  ($(date '+%m-%d %H:%M'))  [--seed is inert; noise floor = eval variance]"
say '```'
say "## EXP-1  value head_EV, gamma 0.94 unless noted"
say "-- NOISE FLOOR: same vctl policy, 3 eval seeds (spread = eval variance):"
vg "floor_vctl_s0" "$(ck vctl 11999808)" 0.94 0 | tee -a "$RES"
vg "floor_vctl_s1" "$(ck vctl 11999808)" 0.94 1 | tee -a "$RES"
vg "floor_vctl_s2" "$(ck vctl 11999808)" 0.94 2 | tee -a "$RES"
say "-- TREATMENTS (eval seed 0, comparable to floor_vctl_s0):"
vg "g090_at0.94"  "$(ck g0.90_vg090 11999808)" 0.94 0 | tee -a "$RES"
vg "g090_at0.90"  "$(ck g0.90_vg090 11999808)" 0.90 0 | tee -a "$RES"
vg "lam080"       "$(ck lam0.80_vlam080 11999808)" 0.94 0 | tee -a "$RES"
vg "popart"       "$(ck vpa 11999808)" 0.94 0 | tee -a "$RES"
say "   NOTE: lam080 trained in a DIFFERENT regime (ego_ent -0.30 vs vctl -0.96);"
say "   g090 at 0.90 has ceiling 0.69 not 0.59 -- not raw-comparable to vctl."

say ""; say "## EXP-2  unscaled reward: did un-binding Adam eps help value EV?"
say "   mechanism already confirmed: value sqrt(v) 3.6e-9 -> 1.06e-5 (eps un-bound)"
vg "rs001_scaled"   "$(ck vclip_rs001 11999808)" 0.94 0 | tee -a "$RES"
vg "rs1_UNSCALED"   "$(ck vclip_rs1.0_rs1 11999808)" 0.94 0 | tee -a "$RES"
say "   EV is scale-free; rs1 above rs001 (beyond the exp1 noise band) => eps hurt V."

say ""; say "## EXP-3  aggresive_coeff=3 combat"
say "-- WHO WINS (round outcomes; ep_rew is positive-sum at a=3, cannot say):"
python3 outcome_balance.py --ckpt "$(ck vctl 11999808)" --ram_mask ram_mask.npy \
  --episodes 300 --out headroom/clean_outcome_a1.json 2>/dev/null | grep -E "rounds|ego win|hp diff|=>" | sed 's/^/  a=1(vctl): /' | tee -a "$RES"
python3 outcome_balance.py --ckpt "$(ck ac3.0_ac3 5999904)" --ram_mask ram_mask.npy \
  --episodes 300 --out headroom/clean_outcome_a3.json 2>/dev/null | grep -E "rounds|ego win|hp diff|=>" | sed 's/^/  a=3(ac3):  /' | tee -a "$RES"
say "-- ON-POLICY gamma paircorr (combat regime learnable structure? baseline ~0.005):"
python3 paircorr_onpolicy.py --ckpt "$(ck ac3.0_ac3 5999904)" --ram_mask ram_mask.npy \
  --n_states 250 --n_envs 16 --out headroom/clean_paircorr.json 2>/dev/null | grep -E "ON-POLICY|active|paircorr" | tee -a "$RES"
say '```'
say "DONE $(date '+%H:%M')"
echo "CLEAN_EVALS_DONE"
