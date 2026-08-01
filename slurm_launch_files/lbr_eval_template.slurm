#!/bin/bash
#SBATCH --job-name=lbr_eval      # Local Best Response checkpoint evaluation
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # one emulator worker per physical core
#SBATCH --mem-per-cpu=2G         # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00          # LBR is minutes per checkpoint, not hours
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jw4406@princeton.edu
echo $CUDA_VISIBLE_DEVICES
JOBID=$SLURM_JOB_ID
WORKDIR=/n/fs/magics/
mkdir -p $WORKDIR/$JOBID
cd $WORKDIR/$JOBID
echo "starting copy op"
cp -r $HOME/FightLadder/ ./
echo "copy op successful"
cd FightLadder/
module purge
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda3/2024.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/2024.02/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate fightladder
echo "fightladder environment activated"

# ---------------------------------------------------------------------------
# Arguments from launch.json (Python Debugger: local_best_response.py).
# Keep this file, .vscode/launch.json, and run_lbr_eval.sh carrying the
# identical arg set.
#
# Because LBR costs minutes rather than hours, one job sweeps a whole
# checkpoint series -- that is the point of the tool.
# ---------------------------------------------------------------------------

CHECKPOINTS=(
    "main/trained_models/tasks/todo/spar_Ry_Gu_500000_steps.task"
    "main/trained_models/tasks/todo/spar_Ry_Gu_1000000_steps.task"
    "main/trained_models/tasks/todo/spar_Ry_Gu_1500000_steps.task"
    "main/trained_models/tasks/todo/spar_Ry_Gu_2000000_steps.task"
    "main/trained_models/tasks/todo/spar_Ry_Gu_2500000_steps.task"
    "main/trained_models/tasks/todo/spar_Ry_Gu_3000000_steps.task"
)

EVAL_PROT="True"
LBR_EGO_TOPK="4"
LBR_STRIDE="1"
LBR_EPISODES="50"
LBR_N_ENVS="16"
LBR_SEED="0"
LBR_HEAD_IDX="0"
LBR_CONTROLS="True"
LBR_MAX_STEPS="100000"
TRAINING_STYLE="spar"
BR_INDEX="0"
DEVICE="cuda"

# local_best_response.py imports sibling modules, so main/ must be the cwd.
cd main

for CKPT in "${CHECKPOINTS[@]}"; do
    echo "=== LBR: $(basename "${CKPT}") ==="
    python local_best_response.py \
        --main_checkpoint_model_path "../${CKPT}" \
        --eval_prot "${EVAL_PROT}" \
        --lbr_ego_topk "${LBR_EGO_TOPK}" \
        --lbr_stride "${LBR_STRIDE}" \
        --lbr_episodes "${LBR_EPISODES}" \
        --lbr_n_envs "${LBR_N_ENVS}" \
        --lbr_seed "${LBR_SEED}" \
        --lbr_head_idx "${LBR_HEAD_IDX}" \
        --lbr_controls "${LBR_CONTROLS}" \
        --lbr_max_steps "${LBR_MAX_STEPS}" \
        --training_style "${TRAINING_STYLE}" \
        --br_index "${BR_INDEX}" \
        --device "${DEVICE}"
done

echo "Done. Results in main/br_rewards/<subdir>/*_lbr_br*.txt (+ .json sidecars)."
cp -r main/br_rewards $HOME/FightLadder/main/ 2>/dev/null || true
cp -r main/selfplay_rewards $HOME/FightLadder/main/ 2>/dev/null || true
