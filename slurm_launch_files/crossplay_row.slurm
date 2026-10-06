#!/bin/bash
#SBATCH --job-name=xplay_row
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=/n/fs/magics/xplay_row_%A_%a.out
#SBATCH --error=/n/fs/magics/xplay_row_%A_%a.err
# ---------------------------------------------------------------------------
# One ego-row of a row-sharded n=ROUNDS crossplay (submitted by submit_crossplay.sh
# as a job array). Reads $WS/participants.txt, computes row $SLURM_ARRAY_TASK_ID
# (its N duels run SERIAL by default -> only ~1 duel's ~18GB resident), writes
# $WS/row_<i>.txt. Per-duel ~18GB at n=500, so keep JOBS=1 unless mem is raised.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO=/auto/u/jw4406/FightLadder; cd "$REPO"
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh 2>/dev/null || true
conda activate fightladder
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${WS:?set WS=<workspace>}"
ROUNDS=${ROUNDS:-500}; EGO_CHAR=${EGO_CHAR:-Guile}; ADV_CHAR=${ADV_CHAR:-ChunLi}; JOBS=${JOBS:-1}
i=$SLURM_ARRAY_TASK_ID
PARGS=(); while IFS= read -r p; do [ -n "$p" ] && PARGS+=(--participant "$p"); done < "$WS/participants.txt"
echo "row $i: $((${#PARGS[@]}/1)) participants, ROUNDS=$ROUNDS JOBS=$JOBS ego=$EGO_CHAR adv=$ADV_CHAR"
python -u "$REPO/main/crossplay.py" "${PARGS[@]}" \
  --row "$i" --jobs "$JOBS" --rounds "$ROUNDS" --ego_char "$EGO_CHAR" --adv_char "$ADV_CHAR" \
  --device cpu --timeout "${DUEL_TIMEOUT:-3600}" --out "$WS/row_${i}.txt"
echo "row $i done -> $WS/row_${i}.txt"
