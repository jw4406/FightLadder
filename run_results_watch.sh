#!/usr/bin/env bash
# RESULTS WATCHER. Waits for the EARLIEST unevaluated experiment to finish, runs
# its eval, appends a dated section to RESULTS_LIVE.md, prints it, and EXITS so
# the harness re-invokes the assistant -- who relays it and relaunches this for
# the next experiment. Purpose: results never sit unread on disk.
#
# One stage per run. GPU-gated on FREE MEMORY via nvidia-smi (never pgrep -- that
# self-matched a shell and deadlocked once already today). Markers in
# logs/results_done/ prevent re-evaluating a stage.
set -uo pipefail
SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"; cd "${SD}/main"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fightladder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RES="${SD}/RESULTS_LIVE.md"; MK="${SD}/logs/results_done"; mkdir -p "$MK"
M=minimax_phase0_vtoff; T=trained_models/tasks/todo
say() { echo "$*"; echo "$*" >> "$RES"; }

# stage -> "run_dir_suffix:target_steps" list (space-separated). popart differs.
ready() {  # $1=suffix $2=target  -> 0 if final checkpoint present
  local d="${M}_rammasked_$1"; [ -d "${SD}/main/${d}" ] || d="${M}_popart_rammasked_$1"
  ls "${SD}/main/${d}/${T}/"*_$2_steps.task >/dev/null 2>&1
}
gpu_free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }

# ---- the three stages, checked earliest-completion first --------------------
do_exp1() {
  say ""; say "## EXP-1  gamma/lambda/PopArt vs seed-1 noise floor  ($(date '+%m-%d %H:%M'))"
  say '```'
  { echo "REGIME SCREEN (value comparison valid only where regimes match):"
    for a in vctl s1_vctlB g0.90_vg090 lam0.80_vlam080; do
      f="${SD}/logs/${M}_rammasked_${a}.log"
      printf "  %-18s rew=%s ego_ent=%s adv_ent=%s\n" "$a" \
        "$(grep ep_rew_mean $f|tail -1|grep -oE '[-0-9.]+ *\|$'|tr -dc '0-9.-')" \
        "$(grep ego_entropy_loss $f|tail -1|grep -oE '[-0-9.]+ *\|$'|tr -dc '0-9.-')" \
        "$(grep adv_entropy_loss $f|tail -1|grep -oE '[-0-9.]+ *\|$'|tr -dc '0-9.-')"
    done
    f="${SD}/logs/${M}_popart_rammasked_vpa.log"
    printf "  %-18s rew=%s\n" "vpa(popart)" "$(grep ep_rew_mean $f|tail -1|grep -oE '[-0-9.]+ *\|$'|tr -dc '0-9.-')"
    echo ""; echo "VALUE HEAD_EV vs realised MC returns (100 eps, gamma 0.94, episode splits):"
    for a in vctl s1_vctlB g0.90_vg090 lam0.80_vlam080; do
      d="${M}_rammasked_${a}"; ck=$(ls ${d}/${T}/*_11999808_steps.task 2>/dev/null | head -1)
      gg=0.94; [ "$a" = "g0.90_vg090" ] && gg=0.90
      [ -n "$ck" ] && python -u value_gap.py --ckpt "$ck" --ram_mask ram_mask.npy \
        --episodes 100 --max_steps 3000 --n_envs 16 --gammas $gg \
        --out headroom/res_e1_${a}.json 2>/dev/null | grep -E "^  0\.|HEAD" | sed "s/^/  ${a}: /"
    done
    dpa="${M}_popart_rammasked_vpa"; ck=$(ls ${dpa}/${T}/*_11999808_steps.task 2>/dev/null | head -1)
    [ -n "$ck" ] && python -u value_gap.py --ckpt "$ck" --ram_mask ram_mask.npy \
      --episodes 100 --max_steps 3000 --n_envs 16 --gammas 0.94 \
      --out headroom/res_e1_vpa.json 2>/dev/null | grep -E "^  0\." | sed "s/^/  vpa: /"
  } 2>&1 | tee -a "$RES"
  say "  READ: compare g090/lam080/vpa head_EV to vctl; a difference is real only"
  say "  if it exceeds |vctl - s1_vctlB| (the same-config seed noise floor)."
  say '```'
}

do_exp2() {
  say ""; say "## EXP-2  unscaled reward (eps un-binding)  ($(date '+%m-%d %H:%M'))"
  say '```'
  { echo "MECHANISM CHECK -- did unscaling lift value sqrt(v) above eps=1e-8?"
    python -u eps_bind.py --out headroom/res_e2_eps.json --ckpts \
      "scaled_rs001=${M}_rammasked_vclip_rs001/${T}/spar_Ry_Sa_11999808_steps.task" \
      "UNSCALED_rs1=${M}_rammasked_vclip_rs1.0_rs1/${T}/spar_Ry_Sa_11999808_steps.task" \
      2>/dev/null | grep -E "rs001|rs1|VERDICT|=>"
    echo ""; echo "VALUE HEAD_EV (EV is scale-free; each head vs its own-scale MC returns):"
    for a in vclip_rs001 vclip_rs1.0_rs1; do
      ck=$(ls ${M}_rammasked_${a}/${T}/*_11999808_steps.task 2>/dev/null | head -1)
      [ -n "$ck" ] && python -u value_gap.py --ckpt "$ck" --ram_mask ram_mask.npy \
        --episodes 100 --max_steps 3000 --n_envs 16 --gammas 0.94 \
        --out headroom/res_e2_${a}.json 2>/dev/null | grep -E "^  0\." | sed "s/^/  ${a}: /"
    done
  } 2>&1 | tee -a "$RES"
  say "  READ: rs1 sqrt(v) >> eps confirms the treatment engaged. Then rs1 head_EV"
  say "  ABOVE rs001 => eps-binding HURT value; equal => it was benign."
  say '```'
}

do_exp3() {
  say ""; say "## EXP-3  aggresive_coeff=3 combat  ($(date '+%m-%d %H:%M'))"
  say '```'
  ac=$(ls ${M}_rammasked_ac3.0_ac3/${T}/*_5999904_steps.task 2>/dev/null | head -1)
  v1=$(ls ${M}_rammasked_vctl/${T}/*_11999808_steps.task 2>/dev/null | head -1)
  echo "-- WHO WINS (round outcomes; ep_rew is positive-sum at a=3 and cannot say)" | tee -a "$RES"
  [ -n "$v1" ] && python -u outcome_balance.py --ckpt "$v1" --ram_mask ram_mask.npy \
      --episodes 300 --out headroom/res_e3_outcome_a1.json 2>&1 \
      | grep -E "rounds|ego win|hp diff|=>" | sed 's/^/  a=1(vctl): /' | tee -a "$RES"
  [ -n "$ac" ] && python -u outcome_balance.py --ckpt "$ac" --ram_mask ram_mask.npy \
      --episodes 300 --out headroom/res_e3_outcome_a3.json 2>&1 \
      | grep -E "rounds|ego win|hp diff|=>" | sed 's/^/  a=3(ac3):  /' | tee -a "$RES"
  echo "-- ON-POLICY gamma paircorr (does the combat regime have learnable joint structure)" | tee -a "$RES"
  [ -n "$ac" ] && python -u paircorr_onpolicy.py --ckpt "$ac" --ram_mask ram_mask.npy \
      --n_states 250 --n_envs 16 --out headroom/res_e3_paircorr.json 2>&1 \
      | grep -E "ON-POLICY|baseline|paircorr >>" | tee -a "$RES"
  say "  READ 1: if ego win% is ~50% at a=1 AND a=3, 'ego wins b/c ego-perspective"
  say "  value head' is dead. (a=1 cdctl smoke already showed ADVERSARY +35%.)"
  say "  READ 2: paircorr >> 0.005 => combat regime has learnable joint structure;"
  say "  ~0.005 => idiosyncratic even in combat (active-state count noted -- <30 = thin)."
  say '```'
}

# ---- pick the earliest ready, unevaluated stage; wait if none ---------------
while true; do
  if ready ac3.0_ac3 5999904 && [ ! -f "$MK/exp3" ] && [ "$(gpu_free_gb)" -ge 5 ]; then
    do_exp3; touch "$MK/exp3"; echo "STAGE_DONE exp3"; exit 0; fi
  if ready vclip_rs1.0_rs1 11999808 && ready vclip_rs001 11999808 && [ ! -f "$MK/exp2" ] && [ "$(gpu_free_gb)" -ge 5 ]; then
    do_exp2; touch "$MK/exp2"; echo "STAGE_DONE exp2"; exit 0; fi
  if ready vctl 11999808 && ready s1_vctlB 11999808 && ready vpa 11999808 \
     && ready g0.90_vg090 11999808 && ready lam0.80_vlam080 11999808 \
     && [ ! -f "$MK/exp1" ] && [ "$(gpu_free_gb)" -ge 5 ]; then
    do_exp1; touch "$MK/exp1"; echo "STAGE_DONE exp1"; exit 0; fi
  [ -f "$MK/exp1" ] && [ -f "$MK/exp2" ] && [ -f "$MK/exp3" ] && { echo "ALL_STAGES_DONE"; exit 0; }
  sleep 300
done
