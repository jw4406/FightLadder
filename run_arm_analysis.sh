#!/usr/bin/env bash
# The three contact-density arms that completed and were never analysed:
#   nsf16    16 emulator frames per decision (contact 8.5% -> 17.0% offline)
#   close64  rounds start within 64px (contact 18.8% at close range offline)
#   stk2     ram_stack=2 stride=1 -- the ONLY arm with a capacity-controlled
#            positive behind it (+0.0687 vs k=1 AND +0.0674 vs its own shuffle)
# against cdctl, the matched 12M control.
#
# EACH ARM NEEDS ITS OWN ENV CONFIG, AND GETTING IT WRONG IS SILENT.
#   nsf16   enumerated at num_step_frames=16. At the default 8 the policy would
#           be evaluated on a DIFFERENT GAME -- and the observation width is
#           identical, so nothing raises.
#   stk2    needs FIGHTLADDER_RAM_STRIDE=1. infer_obs_kwargs recovers the STACK
#           from the obs width but stride is NOT recoverable: k=2/stride=1 and
#           k=2/stride=8 have identical shapes and completely different content.
# This is the same class as the six CLI knobs that silently never reached the
# training env and turned a 12M-step treatment arm into a replicate.
#
# ORDER: regime screen first. HEADROOM read across arms in different regimes is
# the confound that made `engaged` look 12x better than healthy self-play until
# the CONST baseline showed two thirds of it was free.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/main"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fightladder
mkdir -p headroom
LOG="${SCRIPT_DIR}/logs/arm_analysis.log"
say() { echo "[arm $(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }
N_STATES="${N_STATES:-300}"

say "=== 1. REGIME SCREEN across all four arms ==="
python - 2>&1 <<'PY' | tee -a "${LOG}"
import re, glob, os
K = ("total_timesteps","ep_rew_mean","ep_len_mean","ego_entropy_loss","adv_entropy_loss")
arms = ["cdctl","nsf16_nsf16","cr64_close64","k2s1_stk2","huber1.0_huber"]
print(f"  {'arm':>16} {'steps':>11} {'ep_rew':>9} {'ep_len':>7} {'ego_ent':>9} {'adv_ent':>9}")
for a in arms:
    p = f"../logs/minimax_phase0_vtoff_rammasked_{a}.log"
    if not os.path.exists(p): print(f"  {a:>16}  (no log)"); continue
    cur, rows = {}, []
    for ln in open(p, errors="ignore"):
        m = re.match(r"\|\s+(?:\w+/)?([a-z_]+)\s+\|\s+([-\d.e+]+)\s+\|", ln)
        if m and m.group(1) in K:
            cur[m.group(1)] = float(m.group(2))
            if m.group(1) == "total_timesteps": rows.append(dict(cur))
    r = rows[-1] if rows else {}
    print(f"  {a:>16} {r.get('total_timesteps',0):11.0f} {r.get('ep_rew_mean',float('nan')):+9.4f} "
          f"{r.get('ep_len_mean',float('nan')):7.0f} {r.get('ego_entropy_loss',float('nan')):9.3f} "
          f"{r.get('adv_entropy_loss',float('nan')):9.3f}")
print("\n  ep_rew far from 0 => one-sided; entropy at 0 is ABSORBING and terminal.")
print("  Arms whose regimes differ cannot be compared on HEADROOM.")
PY

# arm -> extra env config. Wrong config here is silent, hence the explicit table.
declare -A NSF=( [cdctl]=8 [nsf16_nsf16]=16 [cr64_close64]=8 [k2s1_stk2]=8 )
declare -A STRIDE=( [cdctl]=8 [nsf16_nsf16]=8 [cr64_close64]=8 [k2s1_stk2]=1 )

# Wait for any other GPU job. head_quality asked for 15.5 GiB and OOM'd against
# the concurrently-running huber diagnostic; the two are "parallel" only up to
# the point where both want the card.
while pgrep -f "value_gap.py|critic_ceiling.py" >/dev/null; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "=== 2. ENUMERATE + HEADROOM per arm (final checkpoint) ==="
for arm in cdctl nsf16_nsf16 cr64_close64 k2s1_stk2; do
    dir="minimax_phase0_vtoff_rammasked_${arm}/trained_models/tasks/todo"
    ck=$(ls -1 ${dir}/*_steps.task 2>/dev/null | grep -oE '[0-9]+_steps\.task' \
         | grep -oE '^[0-9]+' | sort -n | tail -1)
    [ -z "${ck}" ] && { say "${arm}: no checkpoint, skipping"; continue; }
    say "--- ${arm} @ ${ck}  (num_step_frames=${NSF[$arm]}, ram_stride=${STRIDE[$arm]})"
    FIGHTLADDER_RAM_STRIDE="${STRIDE[$arm]}" python -u bootstrap_delta.py \
        --ckpt "${dir}/spar_Ry_Sa_${ck}_steps.task" --ram_mask ram_mask.npy \
        --n_states "${N_STATES}" --stride 40 --n_envs 6 --save_obs \
        --num_step_frames "${NSF[$arm]}" \
        --out "headroom/arm_${arm}_${ck}.json" 2>&1 | tail -3 | tee -a "${LOG}"
    FIGHTLADDER_RAM_STRIDE="${STRIDE[$arm]}" python -u head_quality.py \
        --run_dir "${dir}" --npz_glob "headroom/arm_${arm}_${ck}_raw.npz" \
        --ram_mask ram_mask.npy 2>&1 | tail -12 | tee -a "${LOG}"
done

say "=== 3. CONTACT + gamma structure per arm (did the intervention DO anything?) ==="
python - 2>&1 <<'PY' | tee -a "${LOG}"
import numpy as np, glob, os
print(f"  {'arm':>16} {'states':>7} {'contact':>8} {'gamma_share':>12} {'|gamma|':>9} {'paircorr':>9}")
for f in sorted(glob.glob("headroom/arm_*_raw.npz")):
    arm = os.path.basename(f)[4:-8].rsplit("_", 1)[0]
    d = np.load(f)
    if "R" not in d.files: print(f"  {arm:>16}  (no R)"); continue
    M = d["R"].astype(np.float64)
    mu = M.mean(axis=(1,2), keepdims=True)
    G = M - mu - (M.mean(axis=2, keepdims=True)-mu) - (M.mean(axis=1, keepdims=True)-mu)
    gn = (G**2).sum(axis=(1,2)); wn = ((M-mu)**2).sum(axis=(1,2))
    act = gn > 1e-18
    share = gn[act].sum()/max(wn[act].sum(), 1e-30) if act.any() else float("nan")
    pair = float("nan")
    if act.sum() > 1:
        Z = G[act].reshape(int(act.sum()), -1)
        Z = Z/np.linalg.norm(Z, axis=1, keepdims=True)
        C = Z@Z.T; iu = np.triu_indices(len(C), k=1); pair = float(C[iu].mean())
    print(f"  {arm:>16} {len(M):7d} {act.mean():8.1%} {share:12.2%} "
          f"{np.sqrt(gn.mean()):9.5f} {pair:+9.4f}")
print("\n  paircorr is the one that matters: gamma ENERGY without gamma STRUCTURE")
print("  across states is unlearnable, and a random double-centred matrix raises")
print("  gamma_share from 10.98% to 35.13% with zero information.")
PY
say "DONE"
