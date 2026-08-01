#!/bin/bash
# Local Best Response (LBR) evaluation of one or more checkpoints.
#
# Arguments from launch.json (Python Debugger: local_best_response.py).
# Keep this file, .vscode/launch.json, and
# slurm_launch_files/lbr_eval_template.slurm carrying the identical arg set.
#
# LBR is ~100x cheaper than training a best response, so sweeping a whole
# checkpoint series here is the intended use.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LBR_PATH="${SCRIPT_DIR}/main/local_best_response.py"

# One or more checkpoints. Every entry gets its own LBR run.
CHECKPOINTS=(
    "${SCRIPT_DIR}/main/trained_models/tasks/todo/spar_Ry_Gu_1000000_steps.task"
)

# True: LBR replaces the adversary (measures the ego policy's exploitability).
# False: LBR replaces the ego.
EVAL_PROT="True"

# Ego actions to marginalize the one-step branch over, weighted by pi_ego.
# 1 would be clairvoyance in a simultaneous-move game and is NOT a lower bound.
LBR_EGO_TOPK="4"

LBR_STRIDE="1"
LBR_EPISODES="50"
LBR_N_ENVS="16"          # measured knee of the env scaling curve on a 16-core box
LBR_SEED="0"
LBR_HEAD_IDX="0"
LBR_CONTROLS="True"      # also run greedy-damage and critic-shuffled baselines
LBR_MAX_STEPS="100000"
TRAINING_STYLE="spar"
BR_INDEX="0"
DEVICE="cuda"
FILENAME_SUFFIX=""
OUTPUT_SUBDIR=""         # empty -> derived from the checkpoint name

RUN_LIVE="True"          # False -> nohup each run into logs/

LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    echo "=== LBR $((i+1))/${#CHECKPOINTS[@]}: $(basename "${CKPT}") ==="

    CMD=(
        python "${LBR_PATH}"
        --main_checkpoint_model_path "${CKPT}"
        --eval_prot "${EVAL_PROT}"
        --lbr_ego_topk "${LBR_EGO_TOPK}"
        --lbr_stride "${LBR_STRIDE}"
        --lbr_episodes "${LBR_EPISODES}"
        --lbr_n_envs "${LBR_N_ENVS}"
        --lbr_seed "${LBR_SEED}"
        --lbr_head_idx "${LBR_HEAD_IDX}"
        --lbr_controls "${LBR_CONTROLS}"
        --lbr_max_steps "${LBR_MAX_STEPS}"
        --training_style "${TRAINING_STYLE}"
        --br_index "${BR_INDEX}"
        --device "${DEVICE}"
    )
    if [ -n "${FILENAME_SUFFIX}" ]; then
        CMD+=(--filename_suffix "${FILENAME_SUFFIX}")
    fi
    if [ -n "${OUTPUT_SUBDIR}" ]; then
        CMD+=(--output_subdir "${OUTPUT_SUBDIR}")
    fi

    # local_best_response.py imports sibling modules (ippo, local_br_eval,
    # new_br_worker), so main/ must be the working directory.
    if [ "${RUN_LIVE}" = "False" ]; then
        ( cd "${SCRIPT_DIR}/main" && nohup "${CMD[@]}" \
            > "${LOGS_DIR}/lbr_$(basename "${CKPT}" .task).log" 2>&1 & )
        echo "  started detached, PID $!"
    else
        ( cd "${SCRIPT_DIR}/main" && "${CMD[@]}" )
    fi
done

echo "Done. Results in main/br_rewards/<subdir>/*_lbr_br*.txt (+ .json sidecars)."
