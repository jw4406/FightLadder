#!/bin/bash
#SBATCH --job-name=spar_egodecay_VeGuBl_ent05_ck1M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jw4406@princeton.edu
# ---------------------------------------------------------------------------
# NEURONIC backup for the LOCAL run_vega_ent05_spar_egodecay_160M_ck1M.sh.
# Each sbatch job gets its OWN GPU, so there is no 3-on-one-GPU OOM here.
# Reuses the run_minimax_phase0.sh driver (it has every knob these runs need:
# ent_coef, decision_timing, minimax_q, ego_lr_anneal_coeff, ...).
#
# Config MATCHES the local reduced-env runs for a drop-in backup (12 envs/matchup,
# CHECKPOINT_INTERVAL 41667 -> ~1M granularity at 24 envs). Since neuronic gives
# this job a full GPU, you MAY instead use the fuller config:
#     ENVS_PER_MATCHUP=24  CHECKPOINT_INTERVAL=20833   (1M granularity at 48 envs)
# ---------------------------------------------------------------------------
echo "$CUDA_VISIBLE_DEVICES"
JOBID=$SLURM_JOB_ID
WORKDIR=/n/fs/magics/
mkdir -p "$WORKDIR/$JOBID"
cd "$WORKDIR/$JOBID"
echo "copying repo..."
cp -r "$HOME/FightLadder/" ./
cd FightLadder/
module purge
module load anaconda3/2024.02
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh 2>/dev/null || true
conda activate fightladder
echo "env activated; launching driver"

# --- run config (matches run_vega_ent05_spar_egodecay_160M_ck1M.sh) ---
export FOREGROUND=1                       # exec python as the job's main process
export MINIMAX_Q=False                    # skip the 22-action minimax head (spar, 65-action branch)
export PLAYER=Vega OPPONENTS="Guile Blanka"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8 CHARGE_PRESERVING_SKIP=True
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=True LR_ANNEAL_COEFF=0.9998
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export EGO_LR_ANNEAL_COEFF=0.9996
export ENVS_PER_MATCHUP=12
export SEED=0 CHECKPOINT_INTERVAL=41667 TOTAL_TIMESTEPS=160000000
export RUN_SUFFIX=VegaGuileBlanka_dtj_ent05_egodecay9996_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec bash "$WORKDIR/$JOBID/FightLadder/run_minimax_phase0.sh" vtoff
