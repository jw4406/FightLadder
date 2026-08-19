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
# ############################################################################
# THE MASK FILE IS LOAD-BEARING AND EFFECTIVELY IRREPLACEABLE. It is a list of
# RAM BYTE INDICES, and it DEFINES the observation space. A checkpoint records
# the WIDTH of its observation but not WHICH bytes, so:
#
#   * every checkpoint from a masked run is UNEVALUABLE without the exact .npy
#     it was trained with -- local_best_response.py hard-exits rather than guess
#   * regenerating with build_ram_mask.py on a DIFFERENT run gives a DIFFERENT
#     byte set of the same or similar width, which would load without error and
#     silently feed the wrong bytes to every probe
#
# main/ram_mask.npy is therefore COMMITTED and must be treated as immutable for
# as long as any checkpoint trained against it still matters. Do not regenerate
# in place; write a new file under a new name.
# ############################################################################
RAM_MASK="${RAM_MASK:-}"

# Concatenate this many consecutive RAM frames into one observation. 1 = single
# frame, the historical behaviour. A single masked frame is probably already
# Markov for the game's MECHANICAL state, since the mask keeps the per-character
# state-machine bytes, so this is opt-in. It CHANGES THE OBSERVATION WIDTH: a
# checkpoint trained at one stack cannot be loaded at another, and MODEL_FILE
# warm-starts must use the same value.
RAM_STACK="${RAM_STACK:-1}"
RAM_STRIDE="${RAM_STRIDE:-8}"

# CONTACT-DENSITY REWARD VARIANTS. Both default to 0.0 = the historical reward,
# bitwise. The dense reward is D_e - D_a, identically zero when no damage lands,
# so the payoff is CONSTANT in the joint action on ~94% of states and the ANOVA
# interaction term gamma is exactly zero there. These add joint structure while
# staying ANTISYMMETRIC, hence zero-sum, which minimax-Q requires.
#
# Do NOT reach for `aggresive_coeff` instead: r + r_inv = (a-1)(D_e + D_a), so
# any a != 1 makes the game general-sum and invalidates the minimax operator.
#
# ATTACK_STATUSES and PRESSURE_RANGE must come from
# `contact_density.py --mode analyze`, which derives them from data. Guessing
# them silently changes what the reward means.
COUNTERHIT_KAPPA="${COUNTERHIT_KAPPA:-0.0}"
TRADE_KAPPA="${TRADE_KAPPA:-0.0}"
RESET_CLOSE_RANGE="${RESET_CLOSE_RANGE:-0.0}"
# Emulator frames per decision. Halving it doubles agent steps per second of
# game time, so TOTAL_TIMESTEPS must be doubled to budget-match on frames.
NUM_STEP_FRAMES="${NUM_STEP_FRAMES:-8}"

# COUNTERFACTUAL (COMA) BASELINE. Subtracts the OPPONENT's ANOVA main effect
# from each seat's advantage -- ego drops beta (41.3% of within-state energy),
# adversary drops alpha (48.6%). Unbiased: the baseline does not depend on the
# seat's own action, so a wrong head loses variance reduction but cannot bias
# the gradient. That is what makes it safe where "add V to LBR branch
# selection" was not (+0.139 -> -0.229): there the critic CHOSE, here it only
# CENTRES.
#
# RUN COMA_DIAG=True FIRST. It computes the correction and logs what it WOULD
# have done without applying it, bitwise inert. The published alpha/beta shares
# are of WITHIN-STATE energy; if most advantage variance is across-state or
# temporal the realised reduction is much smaller. Gate on
# train/coma_ego_var_reduction beating train/coma_shuffled_var_reduction by a
# clear margin, and >5%, before spending an arm on COMA_COEF>0.
ADAM_EPS="${ADAM_EPS:--1.0}"
VALUE_CLIP_SEPARATE="${VALUE_CLIP_SEPARATE:-False}"
REWARD_SCALE="${REWARD_SCALE:-0.001}"
AGGRESIVE_COEFF="${AGGRESIVE_COEFF:-1.0}"
VALUE_LOSS_FN="${VALUE_LOSS_FN:-mse}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
COMA_COEF="${COMA_COEF:-0.0}"
COMA_DIAG="${COMA_DIAG:-False}"
PRESSURE_BETA="${PRESSURE_BETA:-0.0}"
PRESSURE_RANGE="${PRESSURE_RANGE:-0.0}"
ATTACK_STATUSES="${ATTACK_STATUSES:-}"
POPART="${POPART:-False}"
case "${POPART}" in True|False) ;; *) echo "POPART must be True|False" >&2; exit 1 ;; esac
# What the joint-action head regresses onto. 'returns' (default) is option A --
# the existing lambda-returns, which are DATA and never reference Q, so it
# cannot diverge. 'minimax' is option B, Littman's operator:
#     target = r + gamma * V_mm(s') * (1 - done)
# with V_mm the equilibrium value of the head's OWN matrix at the successor.
# Self-referential, so watch train/minimax_q_scale and train/minimax_target_scale
# -- drifting TOGETHER is divergence. Note minimax_ev and minimax_target_corr
# are MEANINGLESS under 'minimax' (they score agreement with on-policy returns,
# which this target abandons); the gate is the only valid comparison.
MINIMAX_TARGET="${MINIMAX_TARGET:-returns}"
case "${MINIMAX_TARGET}" in returns|minimax) ;;
    *) echo "MINIMAX_TARGET must be returns|minimax" >&2; exit 1 ;; esac
# Joint-action critic PARAMETERIZATION -- orthogonal to MINIMAX_TARGET above.
# 'matrix' is the 484-cell free head: one Linear(512,484), every cell an
# independent row of 513 params with no structural relation to any other.
# 'factored' is the ANOVA form  Q = V + A_ego + A_adv + e_ego^T W(s) e_adv:
# 61 outputs, ~100% gradient density against the matrix head's 0.207%, and W
# zero-initialised so it starts EXACTLY additive and grows interaction only if
# the data pays for it. Adds the train/minimax_fx_* readouts (w_norm,
# gamma_share, anti_share, noop_emb), which turn the offline payoff ANOVA into
# live metrics. CAVEAT on noop_emb: it reads the EMBEDDINGS, but the byte-
# identical-action gap actually lives in a_ego_out (22 independent 513-param
# rows), so it is not a direct measure of that gap.
MINIMAX_HEAD="${MINIMAX_HEAD:-matrix}"
case "${MINIMAX_HEAD}" in matrix|factored) ;;
    *) echo "MINIMAX_HEAD must be matrix|factored" >&2; exit 1 ;; esac
# Rank of the interaction term, 'factored' only. Measured on 2,400 states:
# gamma has median rank 2 and p90 rank 4, so 4 covers p90.
MINIMAX_RANK="${MINIMAX_RANK:-4}"
# Init scale for the factored head's W(s), as a MULTIPLIER on torch's default
# Linear init. 0.0 = the original exact zeros. d(gamma)/d(e_ego) is PROPORTIONAL
# TO W, so at W==0 the action embeddings get NO gradient until W has grown --
# measured at 14.4M, only 4.93% of the true interaction lay inside the learned
# embedding subspace vs 56.43% reachable at the same rank (3.63% = random).
MINIMAX_W_INIT="${MINIMAX_W_INIT:-0.01}"
# Analytic action-embedding basis from gamma_basis.py (energy-optimal rank-r
# subspace of the emulator's interaction). Empty = learn them, which measured
# 4.93% of true gamma against 3.63% random. Rank in the npz MUST match
# MINIMAX_RANK. Frozen by default.
MINIMAX_EMBED="${MINIMAX_EMBED:-}"
MINIMAX_FREEZE_EMBED="${MINIMAX_FREEZE_EMBED:-True}"
# Entropy saturation is ABSORBING: zero entropy -> zero policy gradient -> the
# policy can never recover. p1_clr1e5_winit's ADVERSARY hit it at 3.77M and the
# run spent 34M further steps as single-agent RL against a frozen bot, with a
# plausible-looking score curve the whole time. ent_coef/dstb_ent_coef are 0.0,
# so nothing else prevents it. False = warn but keep going.
# COUNTERFACTUAL ENUMERATION. 0 = OFF and bitwise inert. >0 enumerates the full
# 22x22 payoff at the current env states every N steps, using em.set_state to
# rewind. PRIVILEGED access, training-time only; branch steps are logged as
# train/enum_env_steps and any comparison must be budget-matched (~14.5%% at
# 2M-step intervals).
# x24 envs, so the default 20000 writes a checkpoint every 480k steps. A short
# diagnostic run needs a much smaller value or it finishes without ever writing
# one -- which silently makes the run unmeasurable by every checkpoint-based
# probe (capture, LBR, bootstrap_delta).
# Resume from an existing checkpoint instead of scratch. set_parameters() loads
# the WHOLE policy including the minimax head, which is the point: a 600k run
# from scratch never reaches an ENGAGED state distribution, and contact rate is
# what determines whether there is any interaction to learn (7-12% from scratch
# vs 90.7% at p1_clr1e5's 11.04M).
# Entropy floor. BOTH default 0.0, which is what every run to date used -- and
# zero entropy is an ABSORBING state (no mass to move => no gradient), which
# killed p1_clr1e5_winit's adversary at 3.77M. Separate per side: the collapse
# was on the ADVERSARY, and a bonus on one side of a zero-sum game moves the
# equilibrium being solved for, so raising both is not the same intervention.
# Re-init the EGO to max entropy on resume. An entropy COEFFICIENT cannot escape
# a saturated policy -- measured, ent_coef 0.0/0.001/0.01 from a frozen-ego
# checkpoint gave bit-identical runs. Only a parameter RESET restores ln(22)=3.09.
REINIT_EGO="${REINIT_EGO:-False}"
ENT_COEF="${ENT_COEF:-0.0}"
DSTB_ENT_COEF="${DSTB_ENT_COEF:-0.0}"
MODEL_FILE="${MODEL_FILE:-}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20000}"
ENUM_EVERY="${ENUM_EVERY:-0}"
ENUM_K="${ENUM_K:-484}"
ENUM_BUFFER="${ENUM_BUFFER:-8}"
ENUM_LOSS_COEF="${ENUM_LOSS_COEF:-1.0}"
# Drop enumerated states with gamma == 0. At 6-12% contact in healthy self-play
# ~93% of states are all-identical cells, and averaging over them is the gating
# problem. Judged on R (no critic), so the head cannot fake contact.
ENUM_CONTACT_ONLY="${ENUM_CONTACT_ONLY:-False}"
# Screen with N cheap branches before paying the full 484. 0 = off. Contact
# state COUNT is the binding constraint (614 states/~40 contact -> corrW +0.050;
# ~700 contact -> +0.605), and screening buys them ~13x cheaper.
ENUM_PROBE="${ENUM_PROBE:-0}"
ENUM_WALK="${ENUM_WALK:-40}"
# Fraction of envs parked on contact. 1.0 gave corrW(R) +0.028, worse than the
# natural 6.5% (+0.050): a head trained only where interaction exists learns to
# see it everywhere. Needs ENUM_CONTACT_ONLY=False to keep the ordinary states.
ENUM_PROBE_FRAC="${ENUM_PROBE_FRAC:-1.0}"
ENTROPY_COLLAPSE_ABORT="${ENTROPY_COLLAPSE_ABORT:-True}"
ENTROPY_COLLAPSE_TOL="${ENTROPY_COLLAPSE_TOL:-1e-6}"
ENTROPY_COLLAPSE_PATIENCE="${ENTROPY_COLLAPSE_PATIENCE:-20}"
# PHASE 1 SWITCH. 0.0 = the head feeds NOTHING (diagnostic; every result so far
# was measured here). >0 blends V_minimax into the GAE bootstrap and the head
# starts moving the policy. Requires GAE_LAMBDA=0 -- see ippo.py --gae_lambda.
MINIMAX_BOOTSTRAP_KAPPA="${MINIMAX_BOOTSTRAP_KAPPA:-0.0}"
MINIMAX_BOOTSTRAP_WARMUP="${MINIMAX_BOOTSTRAP_WARMUP:-0}"
GAMMA="${GAMMA:-0.94}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
SEED="${SEED:-0}"
# Learning rates. Defaults are arm A (c_lr 3e-5), which is the config the gate
# was designed around. c_lr 1e-5 is "arm B" from the earlier calibration work.
#
# NOTE THE TIMESCALE RATIO, it is the thing that actually varies here: d_lr/c_lr
# is 3.3x at the arm-A default and 10x at c_lr 1e-5. The adversary already
# learns faster than the ego BY DESIGN, and a faster learner overwhelming a
# slower one is a candidate mechanism for the one-sided drift seen on every arm
# (final scores 0.079 / 0.083 / 0.958 from one algorithm). Raising the ratio
# should make that MORE pronounced, not less -- worth knowing which way it goes.
C_LR="${C_LR:-3e-5}"
D_LR="${D_LR:-1e-4}"
V_LR="${V_LR:-4e-4}"
# LR annealing OFF by default (bitwise the historical constant-LR arms). Turn ON
# for long runs to damp the game-dynamics oscillation -- constant-LR GDA cycles;
# a decaying step size is the standard way to converge. Tune the coeff to the run
# length (ExponentialLR per update; 0.9994 -> ~9% of initial over 50M/~4070 upd).
USE_LR_ANNEALING="${USE_LR_ANNEALING:-False}"
LR_ANNEAL_COEFF="${LR_ANNEAL_COEFF:-.995}"
TAG="${ARM}"; [ "${POPART}" = "True" ] && TAG="${ARM}_popart"
TAG="${TAG}_${OBS_TYPE}"
[ -n "${RAM_MASK}" ] && TAG="${TAG}masked"
# RUN_SUFFIX isolates a run that shares every other setting with an existing
# arm. RUN_ROOT (and therefore FIGHTLADDER_TASK_DIR) derives from TAG, and two
# trainers pointed at one task dir silently overwrite each other's checkpoints
# -- that has already destroyed one baseline. Any re-run of an existing arm
# MUST set this.
[ "${RAM_STACK}" != "1" ] && TAG="${TAG}_k${RAM_STACK}s${RAM_STRIDE}"
[ "${COUNTERHIT_KAPPA}" != "0.0" ] && TAG="${TAG}_ch${COUNTERHIT_KAPPA}"
[ "${TRADE_KAPPA}" != "0.0" ] && TAG="${TAG}_tr${TRADE_KAPPA}"
[ "${RESET_CLOSE_RANGE}" != "0.0" ] && TAG="${TAG}_cr${RESET_CLOSE_RANGE}"
[ "${NUM_STEP_FRAMES}" != "8" ] && TAG="${TAG}_nsf${NUM_STEP_FRAMES}"
[ "${COMA_COEF}" != "0.0" ] && TAG="${TAG}_coma${COMA_COEF}"
[ "${VALUE_LOSS_FN}" != "mse" ] && TAG="${TAG}_${VALUE_LOSS_FN}${HUBER_DELTA}"
[ "${ADAM_EPS}" != "-1.0" ] && TAG="${TAG}_eps${ADAM_EPS}"
[ "${VALUE_CLIP_SEPARATE}" = "True" ] && TAG="${TAG}_vclip"
[ "${REWARD_SCALE}" != "0.001" ] && TAG="${TAG}_rs${REWARD_SCALE}"
[ "${AGGRESIVE_COEFF}" != "1.0" ] && TAG="${TAG}_ac${AGGRESIVE_COEFF}"
[ "${PRESSURE_BETA}" != "0.0" ] && TAG="${TAG}_pb${PRESSURE_BETA}"
[ "${GAMMA}" != "0.94" ] && TAG="${TAG}_g${GAMMA}"
[ "${GAE_LAMBDA}" != "0.95" ] && TAG="${TAG}_lam${GAE_LAMBDA}"
[ "${SEED}" != "0" ] && TAG="${TAG}_s${SEED}"
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

# ############################################################################
# COLLISION GUARD. The comment above records that this has already destroyed a
# baseline -- twice -- and it nearly happened a third time: RUN_TAG was passed
# instead of RUN_SUFFIX, the misspelling was silently ignored, TAG fell through
# to the default, and a throwaway run was pointed at a directory holding 126
# checkpoints from a finished 60.48M arm. It was killed before its first
# checkpoint interval by luck, not by design.
#
# The failure is silent in BOTH directions: a wrong variable name produces no
# error, and an occupied task dir produces no warning. So check the thing that
# actually matters -- is anything already there.
# Catch the specific typo that caused the near-miss, rather than only its effect.
if [ -n "${RUN_TAG:-}" ]; then
    echo "REFUSING TO START: RUN_TAG is not a variable this script reads -- you want RUN_SUFFIX." >&2
    echo "  RUN_TAG='${RUN_TAG}' would be silently ignored and the run would land in the DEFAULT task dir." >&2
    exit 1
fi
_existing="$(find "${FIGHTLADDER_TASK_DIR}/todo" -maxdepth 1 -name '*.task' 2>/dev/null | wc -l)"
if [ "${_existing}" -gt 0 ] && [ "${ALLOW_REUSE:-False}" != "True" ]; then
    echo "REFUSING TO START: ${FIGHTLADDER_TASK_DIR}/todo already holds ${_existing} .task checkpoints." >&2
    echo "  This run would overwrite them -- checkpoint names collide across runs." >&2
    echo "  Set RUN_SUFFIX=<name> for a NEW run, or ALLOW_REUSE=True to resume this one." >&2
    exit 1
fi
# ############################################################################

CMD=(
    python -u "${IPPO_PATH}"
    --player Ryu --opponents Sagat
    --num_env_to_load 1 --env_batch_size 24 --envs_per_matchup 24
    --c_lr "${C_LR}" --d_lr "${D_LR}" --v_lr "${V_LR}"
    --num_perturbs 10 --use_mirror False --ego_side left --side both
    --transform_action True --model_arch_type spar
    --save_dir "${FIGHTLADDER_TASK_DIR}/todo"
    --use_lr_annealing "${USE_LR_ANNEALING}" --lr_anneal_coeff "${LR_ANNEAL_COEFF}"
    --num_env_steps 512 --training_batch_size 1024
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"
    --total_timesteps "${TOTAL_TIMESTEPS:-150000000}"
    --ego_style learning --adv_style learning
    --render False --model_file "${MODEL_FILE}" --async_update False
    --obs_type "${OBS_TYPE}"
    --ram_mask "${RAM_MASK}"
    --ram_stack "${RAM_STACK}"
    --ram_stride "${RAM_STRIDE}"
    --counterhit_kappa "${COUNTERHIT_KAPPA}"
    --trade_kappa "${TRADE_KAPPA}"
    --reset_close_range "${RESET_CLOSE_RANGE}"
    --num_step_frames "${NUM_STEP_FRAMES}"
    --adam_eps "${ADAM_EPS}"
    --value_clip_separate "${VALUE_CLIP_SEPARATE}"
    --reward_scale "${REWARD_SCALE}"
    --aggresive_coeff "${AGGRESIVE_COEFF}"
    --value_loss_fn "${VALUE_LOSS_FN}"
    --huber_delta "${HUBER_DELTA}"
    --coma_coef "${COMA_COEF}"
    --coma_diag "${COMA_DIAG}"
    --pressure_beta "${PRESSURE_BETA}"
    --pressure_range "${PRESSURE_RANGE}"
    --attack_statuses "${ATTACK_STATUSES}"
    --gamma "${GAMMA}"
    --vtrace_enabled "${VTRACE_ENABLED}" --vtrace_seq_len 64
    --vtrace_c_bar 1.0 --vtrace_rho_bar 5.0 --vtrace_replay_capacity 15000
    --popart "${POPART}"
    --minimax_q True --minimax_stop_grad True
    --minimax_target "${MINIMAX_TARGET}"
    --minimax_head "${MINIMAX_HEAD}" --minimax_rank "${MINIMAX_RANK}"
    --minimax_w_init "${MINIMAX_W_INIT}"
    --minimax_freeze_embed "${MINIMAX_FREEZE_EMBED}"
    ${MINIMAX_EMBED:+--minimax_embed "${MINIMAX_EMBED}"}
    --reinit_ego "${REINIT_EGO}"
    --ent_coef "${ENT_COEF}" --dstb_ent_coef "${DSTB_ENT_COEF}"
    --enum_every "${ENUM_EVERY}" --enum_k "${ENUM_K}"
    --enum_buffer "${ENUM_BUFFER}" --enum_loss_coef "${ENUM_LOSS_COEF}"
    --enum_contact_only "${ENUM_CONTACT_ONLY}"
    --enum_probe "${ENUM_PROBE}" --enum_walk "${ENUM_WALK}"
    --enum_probe_frac "${ENUM_PROBE_FRAC}"
    --entropy_collapse_abort "${ENTROPY_COLLAPSE_ABORT}"
    --entropy_collapse_tol "${ENTROPY_COLLAPSE_TOL}"
    --entropy_collapse_patience "${ENTROPY_COLLAPSE_PATIENCE}"
    --minimax_bootstrap_kappa "${MINIMAX_BOOTSTRAP_KAPPA}"
    --minimax_bootstrap_warmup "${MINIMAX_BOOTSTRAP_WARMUP}"
    --gae_lambda "${GAE_LAMBDA}"
    --seed "${SEED}"
    --minimax_iters 1024 --minimax_eta 0.5
)

LOG="${SCRIPT_DIR}/logs/minimax_phase0_${TAG}.log"
mkdir -p "${SCRIPT_DIR}/logs"
echo "=== minimax-Q PHASE 0 (head trains, feeds nothing) ==="
echo "  arm        : ${ARM}   (vtrace_enabled=${VTRACE_ENABLED})"
echo "  lrs        : c_lr=${C_LR}  d_lr=${D_LR}  v_lr=${V_LR}   (d/c ratio $(awk -v d="${D_LR}" -v c="${C_LR}" 'BEGIN{printf "%.1fx", d/c}'))"
echo "  base       : gamma 0.94, gae_lambda=${GAE_LAMBDA}, popart=${POPART}"
if [ "${MINIMAX_BOOTSTRAP_KAPPA}" != "0.0" ]; then
  echo "  *** PHASE 1 : kappa=${MINIMAX_BOOTSTRAP_KAPPA} warmup=${MINIMAX_BOOTSTRAP_WARMUP}"
  echo "  *** the head now MOVES THE POLICY. watch train/minimax_boot_scale_ratio"
  [ "${GAE_LAMBDA}" = "0" ] || [ "${GAE_LAMBDA}" = "0.0" ] || \
    echo "  *** WARNING: kappa>0 with gae_lambda=${GAE_LAMBDA} is UNSOUND (off-policy"
  [ "${MINIMAX_TARGET}" = "minimax" ] || \
    echo "  *** WARNING: minimax_target=returns with kappa>0 is DOUBLY self-referential"
else
  echo "  phase      : 0 (head feeds nothing; kappa=0 is bitwise inert)"
fi
echo "  obs        : ${OBS_TYPE}${RAM_MASK:+  mask=${RAM_MASK}}"
echo "  mm target  : ${MINIMAX_TARGET}$([ "${MINIMAX_TARGET}" = minimax ] && \
    echo '   (option B: r + gamma*V_mm(s'"'"'); ev/target_corr are MEANINGLESS)')"
echo "  mm head    : ${MINIMAX_HEAD}$([ "${MINIMAX_HEAD}" = factored ] && \
    echo "   rank=${MINIMAX_RANK} w_init=${MINIMAX_W_INIT}  (watch train/minimax_fx_w_norm)")"
echo "  task_dir   : ${FIGHTLADDER_TASK_DIR}"
echo "  log        : ${LOG}"
echo "  frames/step: ${NUM_STEP_FRAMES}   close_range: ${RESET_CLOSE_RANGE}px
  adam eps   : ${ADAM_EPS} (-1 = torch default 1e-8)
  value loss : ${VALUE_LOSS_FN} (huber_delta=${HUBER_DELTA} x return std)
  coma       : coef=${COMA_COEF} diag=${COMA_DIAG}
  reward var : counterhit=${COUNTERHIT_KAPPA} trade=${TRADE_KAPPA} pressure=${PRESSURE_BETA} atk_status=[${ATTACK_STATUSES}]
  ckpt every : ${CHECKPOINT_INTERVAL} PER-ENV steps (x24 envs = $(( CHECKPOINT_INTERVAL * 24 )) total)
  watch      : train/minimax_coverage, train/minimax_q_branch_std"
nohup "${CMD[@]}" > "${LOG}" 2>&1 &
echo "  PID        : $!"
