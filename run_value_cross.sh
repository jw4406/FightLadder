#!/usr/bin/env bash
# 2x2 (+neutral) cross-evaluation: is any Huber-vs-MSE difference the LOSS or
# the REGIME? value_gap scores each head on its OWN rollout, which conflates the
# two. Here one rollout gives one set of states and one set of MC targets, and
# both heads are scored against them.
#
# Compare heads WITHIN a column. Comparing across columns is meaningless --
# off-distribution scoring penalises every head, which is the whole reason for
# running more than one column.
#
# Huber wins every column  -> it is the loss.
# Huber wins only its own  -> it is the regime.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/main"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fightladder
LOG="${SCRIPT_DIR}/logs/value_cross.log"
say() { echo "[xval $(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

# Only cdctl and huber are cross-comparable: both are ram_stack=1 (obs 2105) at
# num_step_frames=8, i.e. the SAME observation space and the SAME game. stk2 is
# 4210 wide and nsf16 is a different frame skip -- scoring either here would be
# comparing heads on inputs they never saw.
H="minimax_phase0_vtoff_rammasked_huber1.0_huber/trained_models/tasks/todo/spar_Ry_Sa_11999808_steps.task"
C="minimax_phase0_vtoff_rammasked_cdctl/trained_models/tasks/todo/spar_Ry_Sa_9600000_steps.task"

while pgrep -f "value_gap.py|bootstrap_delta.py|head_quality.py" >/dev/null; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== column A: states from the MSE control ==="
python -u value_cross.py --dist_ckpt "${C}" --score_ckpts "${C}" "${H}" \
    --ram_mask ram_mask.npy --out headroom/xval_dist_ctrl.json 2>&1 | tee -a "${LOG}"
say "=== column B: states from the Huber arm ==="
python -u value_cross.py --dist_ckpt "${H}" --score_ckpts "${C}" "${H}" \
    --ram_mask ram_mask.npy --out headroom/xval_dist_huber.json 2>&1 | tee -a "${LOG}"
say "=== column C: NEUTRAL -- uniform random actions, neither arm's home turf ==="
python -u value_cross.py --dist_ckpt "${C}" --score_ckpts "${C}" "${H}" --dist_random \
    --ram_mask ram_mask.npy --out headroom/xval_dist_random.json 2>&1 | tee -a "${LOG}"

say "=== VERDICT ==="
python - 2>&1 <<'PY' | tee -a "${LOG}"
import json, glob, os
wins = {}
for f in sorted(glob.glob("headroom/xval_dist_*.json")):
    d = json.load(open(f)); col = d["distribution"]
    for g in sorted({r["gamma"] for r in d["rows"]}):
        rows = [r for r in d["rows"] if r["gamma"] == g]
        if len(rows) < 2: continue
        best = max(rows, key=lambda r: r["ev"])
        hub = [r for r in rows if "huber" in r["head"]]
        ctl = [r for r in rows if "huber" not in r["head"]]
        if not hub or not ctl: continue
        d_ev = hub[0]["ev"] - ctl[0]["ev"]
        wins.setdefault(g, []).append((col, d_ev))
        print(f"  gamma={g:.2f}  {col[:34]:>34}  huber-ctrl EV = {d_ev:+.4f}")
print()
for g, ws in wins.items():
    pos = sum(1 for _, d in ws if d > 0.02)
    print(f"  gamma={g:.2f}: huber beats control by >0.02 in {pos}/{len(ws)} columns")
    if pos == len(ws) and len(ws) >= 2:
        print("    => consistent across distributions: THE LOSS.")
    elif pos == 0:
        print("    => no column favours huber: no detectable improvement.")
    else:
        print("    => inconsistent across distributions: REGIME, not the loss.")
PY
say "DONE"
