#!/bin/bash
#SBATCH --job-name=crossplay_matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/n/fs/magics/crossplay_matrix_%j.out
#SBATCH --error=/n/fs/magics/crossplay_matrix_%j.err
# ---------------------------------------------------------------------------
# General n=ROUNDS NxN crossplay via crossplay.py -> duel.py. CPU-only eval.
# Each participant is ONE checkpoint; ego uses its left policy, adv its right
# (spar family: ctrl/dstb heads; league family incl psro: policy/policy_other).
# Two modes (env):
#   SELFPLAY=<algo>  [DEPTHS="20,40,.."]   depth matrix for one algo
#                                          (DEPTHS default = all available)
#   PARTS="spar:200 ippo:200 psro:200 .."  arbitrary cross-algo participant list
# Common env: ROUNDS(500)  EGO_CHAR(Guile)  ADV_CHAR(ChunLi)  DUEL_TIMEOUT(900)
# algos: spar ippo 2tsA 2tsC psro league   (see resolve_ckpt.py)
# ---------------------------------------------------------------------------
set -uo pipefail
JOBID=$SLURM_JOB_ID; W=/n/fs/magics/$JOBID; mkdir -p "$W"
REPO=/auto/u/jw4406/FightLadder
cd "$REPO"
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh 2>/dev/null || true
conda activate fightladder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # keep each duel single-core; parallelism is across cells
RES="$REPO/slurm_launch_files/resolve_ckpt.py"
ROUNDS=${ROUNDS:-500}; EGO_CHAR=${EGO_CHAR:-Guile}; ADV_CHAR=${ADV_CHAR:-ChunLi}; JOBS=${JOBS:-16}

# ---- assemble participant tokens "algo:depth" ----
if [ -n "${SELFPLAY:-}" ]; then
  A=$SELFPLAY
  # accept comma-, underscore-, or space-delimited depths (underscore survives
  # sbatch --export, whose comma is a variable separator).
  DPS=${DEPTHS:-$(python3 "$RES" "$A" --list)}; DPS=${DPS//,/ }; DPS=${DPS//_/ }
  TOKENS=""; for d in $DPS; do TOKENS="$TOKENS $A:$d"; done
  LABELMODE=depth; TAG="selfplay_${A}"
else
  TOKENS=${PARTS:?set SELFPLAY=<algo> or PARTS=\"algo:depth ...\"}
  LABELMODE=algo; TAG="crossplay"
fi

# ---- resolve each to --participant label:model_type:path ----
PARGS=(); N=0
echo "resolving participants (ego_char=$EGO_CHAR adv_char=$ADV_CHAR):"
for tok in $TOKENS; do
  a=${tok%%:*}; d=${tok##*:}
  if ! line=$(python3 "$RES" "$a" "$d" 2>/dev/null); then
    echo "  SKIP $tok  (no checkpoint within TOL)"; continue
  fi
  mt=$(echo "$line" | awk '{print $1}')
  absM=$(echo "$line" | awk '{print $2}')
  path=$(echo "$line" | awk '{print $3}')
  if [ "$LABELMODE" = depth ]; then label="d${absM}"; else label="${a}${absM}"; fi
  echo "  $tok -> ${label} :${mt}:  $path"
  PARGS+=(--participant "${label}:${mt}:${path}")
  N=$((N+1))
done
[ "$N" -ge 2 ] || { echo ">> need >=2 participants, got $N -- aborting."; exit 2; }
echo ">> ${N}x${N} matrix = $((N*N)) duels x ${ROUNDS} rounds (serial, CPU)."

OUT="/n/fs/magics/${TAG}_n${ROUNDS}_${EGO_CHAR}_vs_${ADV_CHAR}_${JOBID}.txt"
echo "=== crossplay ${TAG} n=${ROUNDS} ego=${EGO_CHAR} adv=${ADV_CHAR} -> $OUT ==="
echo ">> running with --jobs $JOBS (parallel duels), per-duel timeout ${DUEL_TIMEOUT:-3600}s"
python -u "$REPO/main/crossplay.py" "${PARGS[@]}" \
  --rounds "$ROUNDS" --ego_char "$EGO_CHAR" --adv_char "$ADV_CHAR" \
  --device cpu --jobs "$JOBS" --timeout "${DUEL_TIMEOUT:-3600}" \
  --out "$OUT"
echo "=== done -> $OUT ==="
