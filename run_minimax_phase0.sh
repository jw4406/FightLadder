#!/bin/bash
# PHASE 0 of the minimax-Q experiment: train the joint-action critic, use it for
# NOTHING, and measure whether it has branch discrimination that V lacks.
#
# The head is provably inert here -- stop-grad on the shared latent AND an
# optimizer scoped to the head's own parameters, verified as
# max|shared - initial| == 0.0 across optimizer steps. So this run's POLICY
# trajectory is whatever the base config does; only the new head is on trial.
#
# BASE CONFIG = arm A (c_lr 3e-5), deliberately, because the gate is only
# interpretable in a balanced game. Every critic number measured in a collapsed
# or stalling regime this week turned out to be an artifact: EV 0.957 at 34.56M
# where a RANDOM PROJECTION scored 0.949, and the ceiling itself is
# policy-dependent (0.056 balanced vs 0.216 collapsed). Arm A is the only config
# measured to avoid the collapse (score_rollout 0.438 at 10M).
#
# gamma 0.94 rather than the spar default 0.99: the return-prediction ceiling is
# ~3x higher there (0.202 vs 0.055 at 10.08M), so Q has more signal to fit.
#
# popart stays OFF -- one variable at a time, and popart addresses target scale,
# which is not what this experiment is about.
#
# WATCH THESE TWO EARLY (they are abort signals, not the gate):
#   train/minimax_coverage      fraction of the 484 cells ever gradient-updated.
#                               p_max(ego) ~ 0.94 was measured, so sampled joint
#                               actions concentrate on a few cells while LBR
#                               branches over all 22 adversary actions. If this
#                               is low, "Q does not discriminate branches" is
#                               indistinguishable from "Q never saw them" and
#                               the gate result means NOTHING.
#   train/minimax_ev            explained variance of Q against its target. THE
#                               signal metric. minimax_loss alone is
#                               uninterpretable -- loss ~ target variance is
#                               exactly what a head that learned only the mean
#                               produces, and 2e-04 against target_std 0.017
#                               meant precisely that for the whole first run.
#   train/minimax_target_corr   corr(prediction, target). MUST trend POSITIVE.
#                               Negative means the ego/adversary frames have
#                               diverged again -- the first run trained a head
#                               that fit -G almost perfectly (slope -0.990,
#                               corr -0.929) and NOTHING else in the log showed
#                               it, because MSE against a flipped target
#                               descends exactly as fast as against the right one.
#   train/minimax_visits_p10    visit count at the 10th percentile of cells.
#                               `coverage` reads the FRACTION ever touched and
#                               sat at 1.000 all run while hiding a 1,400x
#                               imbalance.
#   train/minimax_q_branch_std  spread of Q ACROSS the matrix at fixed state --
#                               the quantity the whole bet rests on. If it
#                               decays toward V's std, Q is collapsing to a
#                               state-value function with 484x the parameters
#                               and the direction is dead early.
# minimax_loss is NOT the number to watch -- use minimax_ev. And q_branch_std is
# NOT evidence of signal: a head can be arbitrarily spread across the matrix and
# still uncorrelated with the target, which is exactly what the first run did.
#
# PICKING THE GATE CHECKPOINT -- TWO conditions, not one:
#   score_rollout in [0.3, 0.7]   exploitability measured in a one-sided game is
#                                 an artifact (EV read 0.957 at 34.56M while a
#                                 RANDOM PROJECTION scored 0.949).
#   rollout/ep_len_mean < 400     episodes must RESOLVE. Measured on the first
#                                 Phase 0 run at 472 steps/ep: a ridge on the
#                                 frozen latent scored +0.9636 state-only,
#                                 +0.9635 with the actions added, +0.9632 with a
#                                 full per-cell slope. The action contributes
#                                 NOTHING when the timer decides the outcome, so
#                                 such a checkpoint cannot test the joint-action
#                                 hypothesis no matter how well Q is trained.
# The first run was gated on balance alone. That was insufficient.
#
# THE GATE, once a checkpoint satisfies BOTH:
#   LBR_MODES="minimax,minimaxshuffle,greedy" ./run_lbr_sweep.sh
#   minimax vs greedy         is Q a useful leaf evaluator where V is not?
#                             (V-based lbr loses to greedy 3/3: 0.41/0.17/0.25)
#   minimax vs minimaxshuffle does Q carry BRANCH-level information at all?
#                             (V does not: -0.1419 vs -0.1407 over 42 runs)
# Usage:  ./run_minimax_phase0.sh [vton|vtoff]     (default vton)
#
# The two arms cannot share the card at 24 envs: V-trace's device-side replay
# chunks put the vton arm at ~19.3 GiB of 24.5, leaving ~4.7 free against the
# ~4.7 a vtoff run needs. Measured, not guessed -- an earlier attempt had the
# V-trace run grab 18.7 GiB regardless of what was already resident and starve
# the other process 70 s later. Run them SEQUENTIALLY.
set -u

ARM="${1:-vton}"
case "${ARM}" in
    vton)  VTRACE_ENABLED="True"  ;;
    vtoff) VTRACE_ENABLED="False" ;;
    *) echo "usage: $0 [vton|vtoff]   (env: POPART=True|False)" >&2; exit 1 ;;
esac
# PopArt is a SECOND variable. It addresses target SCALE, not branch structure,
# so it is off by default -- but it matters more for the minimax head than it did
# for V: solve_matrix_game's eta is not scale-free, and at the measured
# G_std ~0.0166 an un-normalized matrix collapses the solver to uniform play.
# Normalizing Q makes eta meaningful instead of hand-tuned to the reward scale.
# OBSERVATION. 'ram' is the whole point of this run: pixels resolve 1 of 21
# action-distinct successors at a median decision point at ANY resolution, frame
# count or channel set, while RAM resolves all 21 -- and the distinction is
# PREDICTIVE (12 distinct futures still separated at 16 steps, vs 3 in pixels).
# The 14 curated info variables also resolve 1, so --obs_type info would not have
# helped. Full RAM is 65,536 bytes => ~67M params across the two feature
# extractors; RAM_MASK cuts that to the bytes that actually move.
OBS_TYPE="${OBS_TYPE:-ram}"
RAM_MASK="${RAM_MASK:-}"
POPART="${POPART:-False}"
case "${POPART}" in True|False) ;; *) echo "POPART must be True|False" >&2; exit 1 ;; esac
TAG="${ARM}"; [ "${POPART}" = "True" ] && TAG="${ARM}_popart"
TAG="${TAG}_${OBS_TYPE}"
[ -n "${RAM_MASK}" ] && TAG="${TAG}masked"
# RUN_SUFFIX isolates a run that shares every other setting with an existing
# arm. RUN_ROOT (and therefore FIGHTLADDER_TASK_DIR) derives from TAG, and two
# trainers pointed at one task dir silently overwrite each other's checkpoints
# -- that has already destroyed one baseline. Any re-run of an existing arm
# MUST set this.
[ -n "${RUN_SUFFIX:-}" ] && TAG="${TAG}_${RUN_SUFFIX}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IPPO_PATH="${SCRIPT_DIR}/main/ippo.py"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CRITICAL -- .task checkpoints go to ippo.py's TASK_DIR, which derives from the
# SCRIPT path and ignores both --save_dir and cwd. Every spar Ryu-vs-Sagat run
# writes the same filenames, so without this override concurrent runs silently
# overwrite each other. That already destroyed one baseline checkpoint.
RUN_ROOT="${SCRIPT_DIR}/main/minimax_phase0_${TAG}"
export FIGHTLADDER_TASK_DIR="${RUN_ROOT}/trained_models/tasks"
mkdir -p "${FIGHTLADDER_TASK_DIR}/todo" "${FIGHTLADDER_TASK_DIR}/todo_continue"

CMD=(
    python -u "${IPPO_PATH}"
    --player Ryu --opponents Sagat
    --num_env_to_load 1 --env_batch_size 24 --envs_per_matchup 24
    --c_lr 3e-5 --d_lr 1e-4 --v_lr 4e-4        # arm A: avoids the collapse
    --num_perturbs 10 --use_mirror False --ego_side left --side both
    --transform_action True --model_arch_type spar
    --save_dir "${FIGHTLADDER_TASK_DIR}/todo"
    --use_lr_annealing False --lr_anneal_coeff .995
    --num_env_steps 512 --training_batch_size 1024
    --checkpoint_interval 20000                 # x24 envs = every 480k steps
    --total_timesteps "${TOTAL_TIMESTEPS:-150000000}"
    --ego_style learning --adv_style learning
    --render False --model_file "" --master_use_stag False --async_update False
    --obs_type "${OBS_TYPE}"
    --ram_mask "${RAM_MASK}"
    --gamma 0.94
    --vtrace_enabled "${VTRACE_ENABLED}" --vtrace_seq_len 64
    --vtrace_c_bar 1.0 --vtrace_rho_bar 5.0 --vtrace_replay_capacity 15000
    --popart "${POPART}"
    --minimax_q True --minimax_stop_grad True
    --minimax_iters 1024 --minimax_eta 0.5
    --use_stagnation_early_stop False --use_stagnation_velocity_signal False
    --use_stagnation_entropy_signal False --stagnation_patience 20000
    --stagnation_tolerance 1e-4 --stagnation_rel_tolerance 0.05
    --stagnation_ema_beta 0.99 --stagnation_eps 1e-8 --stagnation_eval_games 0
    --entropy_stagnation_weight 1.0 --stagnation_lr_factor 0.999
    --stagnation_lr_patience 150
)

LOG="${SCRIPT_DIR}/logs/minimax_phase0_${TAG}.log"
mkdir -p "${SCRIPT_DIR}/logs"
echo "=== minimax-Q PHASE 0 (head trains, feeds nothing) ==="
echo "  arm        : ${ARM}   (vtrace_enabled=${VTRACE_ENABLED})"
echo "  base       : arm A c_lr 3e-5, gamma 0.94, popart=${POPART}"
echo "  obs        : ${OBS_TYPE}${RAM_MASK:+  mask=${RAM_MASK}}"
echo "  task_dir   : ${FIGHTLADDER_TASK_DIR}"
echo "  log        : ${LOG}"
echo "  watch      : train/minimax_coverage, train/minimax_q_branch_std"
nohup "${CMD[@]}" > "${LOG}" 2>&1 &
echo "  PID        : $!"
