#!/bin/bash
# Watch Phase 0, pick the first checkpoint meeting BOTH gate conditions, run the
# minimax trio on it.
#
#   score_rollout in [0.3, 0.7]   exploitability in a one-sided game is an
#                                 artifact (EV read 0.957 at 34.56M while a
#                                 RANDOM PROJECTION scored 0.949)
#   ep_len_mean   < 400           episodes must RESOLVE. At 472 steps/ep a ridge
#                                 on the frozen latent scored +0.9636 state-only
#                                 vs +0.9632 with a full per-cell slope -- the
#                                 action contributes NOTHING when the timer
#                                 decides the outcome, so such a checkpoint
#                                 cannot test the hypothesis at any level of Q
#                                 training. The first run was gated on balance
#                                 alone; that was insufficient.
#
# Also REFUSES to gate a checkpoint whose head is not learning: minimax_ev must
# be positive and minimax_target_corr must be positive. target_corr < 0 means the
# ego/adversary frames have diverged again, which is invisible in loss (MSE
# against a flipped target descends exactly as fast) and produced a head that fit
# -G to slope -0.990 for an entire run.
#
# MIN_STEPS exists because the head needs training before the gate means
# anything -- a null result from an untrained Q is uninformative.
set -u
REPO=/home/jw4406/codebase/FightLadder
cd "${REPO}"
# The probes import torch. nohup does not inherit an activated env.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder
python -c "import torch" 2>/dev/null || { echo "[watch] FATAL: no torch"; exit 3; }
LOG="${REPO}/logs/minimax_phase0_vton.log"
CKDIR="${REPO}/main/minimax_phase0_vton/trained_models/tasks/todo"
MIN_STEPS=${MIN_STEPS:-4800000}

while true; do
    if ! pgrep -f "[i]ppo.py" >/dev/null 2>&1; then
        echo "[watch] trainer gone at $(date '+%F %T')"; break
    fi
    PICK=$(python3 - "$LOG" "$CKDIR" "$MIN_STEPS" <<'PY'
import re, sys, os, statistics as st
log, ckdir, min_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = open(log, errors="ignore").read()
g = lambda n: [float(x) for x in re.findall(rf"{n} *\| *([-0-9.e+]+)", t)]
sc, el = g("score_rollout"), g("ep_len_mean")
ev, tc = g("minimax_ev"), g("minimax_target_corr")
if not sc or not el:
    sys.exit(1)
n = min(len(sc), len(el))
# head must be learning at all before any gate result is meaningful
if not ev or st.mean(ev[-20:]) <= 0 or not tc or st.mean(tc[-20:]) <= 0:
    sys.exit(1)
cks = sorted(int(re.search(r"_(\d+)_steps", f).group(1))
             for f in os.listdir(ckdir) if f.endswith(".task"))
ITER = 12288
for stp in cks:
    if stp < min_steps:
        continue
    i = int(stp / ITER)
    if i >= n:
        continue
    w = slice(max(0, i - 20), min(n, i + 20))
    s_, e_ = st.mean(sc[w]), st.mean(el[w])
    if 0.3 <= s_ <= 0.7 and e_ < 400:
        print(f"{stp} {s_:.3f} {e_:.0f} {st.mean(ev[-20:]):.4f} {st.mean(tc[-20:]):.3f}")
        sys.exit(0)
sys.exit(1)
PY
) && break
    sleep 300
done

[ -z "${PICK:-}" ] && { echo "[watch] no qualifying checkpoint; exiting"; exit 1; }
set -- $PICK
STEP=$1
echo "[watch] PICKED ${STEP}: score=$2 ep_len=$3 minimax_ev=$4 target_corr=$5"

# PROBE FIRST. The gate costs ~3.5 h; this costs ~10 min and asks the same
# question with NO dependence on Q's training, sign, or coverage -- it hands the
# joint action to a ridge on the frozen latent explicitly, which is the most
# favourable possible test of whether action-conditional value exists here at
# all. Measured on the vton 480k checkpoint (234 steps/ep, healthy):
#     latent only              +0.0808
#     latent + onehot(actions) +0.0721
#     latent + onehot(joint)   +0.0708
# The action made it WORSE. If that holds at >=4.8M the mechanism is absent and
# the gate would be measuring something that is not there.
# TWO probes. The return probe answers "does the action help predict what the
# critic is actually fit to". The reward probe answers "is the action effect
# even IN the representation" -- at gamma=0.94 the horizon is ~16.7 steps, so a
# single action's contribution is ~1/17th of the return's variance and can be
# real but below the noise floor. greedy proves r(a,o) is decision-relevant, so
# a null on the REWARD target is a representation failure; a null on return with
# a hit on reward is dilution, which more data or a shorter horizon could fix.
echo "[watch] probing ${STEP} (reward target: is the action effect in the latent?)"
python -u main/minimax_probe_ceiling.py \
    --ckpt "${CKDIR}/spar_Ry_Sa_${STEP}_steps.task" \
    --steps 6000 --n_envs 12 --device cuda --target reward \
    --out "minimax_probe_reward_vton${STEP}.json" \
    >> logs/probe_reward_gate.log 2>&1
python3 -c "
import json
d=json.load(open('main/minimax_probe_reward_vton${STEP}.json'))
print(f\"[watch] REWARD-target action_gain {d.get('action_gain',0):+.4f}  {d.get('verdict','?')}\")
" 2>/dev/null || echo "[watch] reward probe failed"

echo "[watch] probing ${STEP} (return target) before spending the gate"
python -u main/minimax_probe_ceiling.py \
    --ckpt "${CKDIR}/spar_Ry_Sa_${STEP}_steps.task" \
    --steps 6000 --n_envs 12 --device cuda \
    --out "minimax_probe_ceiling_vton${STEP}.json" \
    >> logs/probe_ceiling_vton_gate.log 2>&1
GAIN=$(python3 -c "
import json,sys
d=json.load(open('main/minimax_probe_ceiling_vton${STEP}.json'))
print(f\"{d.get('action_gain',0.0):.4f} {d.get('verdict','?')}\")" 2>/dev/null)
echo "[watch] action_gain: ${GAIN}"
# FAIL CLOSED. The first version of this treated a missing file as "no gain"
# and printed a confident kill verdict on two probes that had crashed with
# ModuleNotFoundError. A measurement that did not happen is not a result.
RC=$(python3 -c "
import json,sys,os
f='main/minimax_probe_ceiling_vton${STEP}.json'
if not os.path.exists(f): print('MISSING'); sys.exit(0)
try: d=json.load(open(f))
except Exception as e: print('UNREADABLE'); sys.exit(0)
if 'action_gain' not in d: print('NO_FIELD'); sys.exit(0)
print('HELPS' if d['action_gain'] > 0.02 else 'NULL')" 2>/dev/null)
[ -z "${RC}" ] && RC=MISSING
echo "[watch] return-probe status: ${RC}"
if [ "${RC}" != "HELPS" ] && [ "${RC}" != "NULL" ]; then
    echo "[watch] PROBE DID NOT PRODUCE A RESULT (${RC}). No verdict, no gate."
    echo "        Check logs/probe_ceiling_vton_gate.log."
    echo "WATCH_PROBE_ERROR ${STEP}"
    exit 3
fi
if [ "${RC}" = "NULL" ]; then
    echo "[watch] ACTION ADDS NOTHING at ${STEP} -- NOT running the gate."
    echo "        Handing the joint action to a ridge explicitly buys nothing, so"
    echo "        Q(s,a,o) has no mechanism to beat V(s) and the gate would be"
    echo "        measuring an absent effect. This is the cheap kill."
    echo "WATCH_PROBE_NULL ${STEP}"
    exit 0
fi
echo "[watch] action helps -- running the gate"

SEL="${REPO}/main/minimax_phase0_vton/trained_models/tasks/gate_sel"
rm -rf "$SEL"; mkdir -p "$SEL"
ln -sf "${CKDIR}/spar_Ry_Sa_${STEP}_steps.task" "$SEL/"

CKPT_GLOB="${SEL}/spar_Ry_Sa_*_steps.task" \
SWEEP_TAG="mmq_gate2" \
OUTPUT_SUBDIR="spar_mmq_gate2" \
LBR_MODES="minimax,minimaxshuffle,greedy" \
N_WORKERS=2 \
./run_lbr_sweep.sh >> logs/lbr_sweep_mmq_gate2_launch.log 2>&1

echo "[watch] gate finished $(date '+%F %T')"
echo "WATCH_GATE_DONE ${STEP}"
