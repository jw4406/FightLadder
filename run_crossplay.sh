#!/bin/bash
# Crossplay the 3 ent05 Vega-vs-{Guile,Blanka} runs (spar / ippo / v2-egodecay) at their
# LATEST checkpoints. Auto-discovers the newest checkpoint per run across the original and
# _cont dirs (no hardcoded steps), then calls main/crossplay.py. Extra args pass through, e.g.:
#   bash run_crossplay.sh --rounds 10 --out logs/crossplay_latest.txt
#   bash run_crossplay.sh --decision_timing off
# For an arbitrary set of checkpoints, call main/crossplay.py directly with --participant flags.
cd "$(dirname "$0")"
source /home/jw4406/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate fightladder 2>/dev/null

B=main/minimax_phase0_vtoff_image

# latest <orig_dir> <cont_dir> <resume_steps> <prefix>  ->  "<true_M> <ckpt_path>"
latest(){
  local orig=$1 cont=$2 resume=$3 pfx=$4 d f st true best="" bestt=-1
  for d in "$cont" "$orig"; do
    [ -d "$d" ] || continue
    for f in "$d"/${pfx}_*_steps.task; do
      [ -e "$f" ] || continue
      st=$(echo "$f" | sed -E 's/.*_([0-9]+)_steps.task/\1/')
      if [ "$d" = "$cont" ]; then true=$(( resume + st )); else true=$st; fi
      if [ "$true" -gt "$bestt" ]; then bestt=$true; best="$f"; fi
    done
  done
  echo "$(( bestt / 1000000 )) $best"
}

read spar_M spar_ck < <(latest \
  ${B}_rs1.0_VegaGuileBlanka_dtj_ent05_constant_cps_160M_ck1M/trained_models/tasks/todo \
  ${B}_rs1.0_VegaGuileBlanka_dtj_ent05_constant_cps_160M_ck1M_cont/trained_models/tasks/todo 0 spar_Ve_GuBl)
read ippo_M ippo_ck < <(latest \
  ${B}_ippo_rs1.0_VegaGuileBlanka_dtj_ent05_constant_cps_160M_ck1M/trained_models/tasks/todo \
  ${B}_ippo_rs1.0_VegaGuileBlanka_dtj_ent05_constant_cps_160M_ck1M_cont/trained_models/tasks/todo 0 ippo_Ve_GuBl)
read v2_M v2_ck < <(latest \
  ${B}_rs1.0_VegaGuileBlanka_dtj_ent05_egodecay9996_cps_160M_ck1M/trained_models/tasks/todo \
  ${B}_rs1.0_VegaGuileBlanka_dtj_ent05_egodecay9996_cps_160M_ck1M_cont/trained_models/tasks/todo 0 spar_Ve_GuBl)

echo "latest checkpoints: spar ${spar_M}M | ippo ${ippo_M}M | v2 ${v2_M}M"
[ -n "$spar_ck" ] && [ -n "$ippo_ck" ] && [ -n "$v2_ck" ] || { echo "ERROR: could not find a checkpoint for one or more runs" >&2; exit 1; }

exec python main/crossplay.py \
  --participant "spar_${spar_M}M:spar:$spar_ck" \
  --participant "ippo_${ippo_M}M:ippo:$ippo_ck" \
  --participant "v2_${v2_M}M:spar:$v2_ck" \
  "$@"
