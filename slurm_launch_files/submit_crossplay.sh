#!/bin/bash
# ---------------------------------------------------------------------------
# Orchestrate a row-sharded n=ROUNDS NxN crossplay (CPU-only). Resolves
# participants, submits a job array (1 ego-row per task, ~24GB each) + a
# dependent finalizer that stitches the rows into matrix.{txt,csv}.
# Modes (env):
#   SELFPLAY=<algo> [DEPTHS="20,40,.."|"20_40_.."]   depth self-play matrix
#   PARTS="spar:200 ippo:200 psro:200 .."            arbitrary cross-algo
# Common env: ROUNDS(500) EGO_CHAR(Guile) ADV_CHAR(ChunLi) JOBS(1) DUEL_TIMEOUT(3600)
# algos: spar ippo 2tsA 2tsC psro league
# Run from the login node (lightweight: resolve + sbatch only).
# ---------------------------------------------------------------------------
set -uo pipefail
REPO=/auto/u/jw4406/FightLadder; RES="$REPO/slurm_launch_files/resolve_ckpt.py"
ROUNDS=${ROUNDS:-500}; EGO_CHAR=${EGO_CHAR:-Guile}; ADV_CHAR=${ADV_CHAR:-ChunLi}

if [ -n "${SELFPLAY:-}" ]; then
  A=$SELFPLAY; DPS=${DEPTHS:-$(python3 "$RES" "$A" --list)}; DPS=${DPS//,/ }; DPS=${DPS//_/ }
  TOKENS=""; for d in $DPS; do TOKENS="$TOKENS $A:$d"; done; LABELMODE=depth; TAG="selfplay_${A}"
else
  TOKENS=${PARTS:?set SELFPLAY=<algo> or PARTS=\"algo:depth ...\"}; LABELMODE=algo; TAG="crossplay"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
WS=/n/fs/magics/xplay_${TAG}_n${ROUNDS}_${EGO_CHAR}v${ADV_CHAR}_${STAMP}; mkdir -p "$WS"
: > "$WS/participants.txt"
for tok in $TOKENS; do
  a=${tok%%:*}; d=${tok##*:}
  if ! line=$(python3 "$RES" "$a" "$d" 2>/dev/null); then echo "  SKIP $tok (no ckpt within TOL)"; continue; fi
  mt=$(echo "$line"|awk '{print $1}'); absM=$(echo "$line"|awk '{print $2}'); path=$(echo "$line"|awk '{print $3}')
  [ "$LABELMODE" = depth ] && label="d${absM}" || label="${a}${absM}"
  echo "${label}:${mt}:${path}" >> "$WS/participants.txt"
done
N=$(grep -c . "$WS/participants.txt")
[ "$N" -ge 2 ] || { echo ">> need >=2 participants, got $N -- aborting."; exit 2; }
echo "WS=$WS"; echo "N=$N participants:"; sed 's/^/  /' "$WS/participants.txt"

ARR=$(sbatch --parsable --array=0-$((N-1)) \
  --export=ALL,WS="$WS",ROUNDS="$ROUNDS",EGO_CHAR="$EGO_CHAR",ADV_CHAR="$ADV_CHAR",JOBS="${JOBS:-1}",DUEL_TIMEOUT="${DUEL_TIMEOUT:-3600}" \
  "$REPO/slurm_launch_files/crossplay_row.slurm")
FIN=$(sbatch --parsable --dependency=afterany:$ARR --kill-on-invalid-dep=yes --job-name=xplay_fin \
  --output=/n/fs/magics/xplay_fin_%j.out --time=0:30:00 --mem=4G --cpus-per-task=1 \
  --wrap="source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh 2>/dev/null; conda activate fightladder; python3 $REPO/slurm_launch_files/crossplay_finalize.py $WS $WS/matrix")
echo ">> array=$ARR (0-$((N-1)))  finalizer=$FIN (afterany)"
echo ">> results will land at: $WS/matrix.txt  and  $WS/matrix.csv"
