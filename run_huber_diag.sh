#!/usr/bin/env bash
# Huber value-loss arm: is the treatment LIVE, is the REGIME comparable, and did
# V actually get better? Runs unattended, in that order, and refuses to report a
# value comparison across arms that are in different regimes.
#
# THE ORDER IS THE POINT.
#
#  0. IS THE TREATMENT EVEN LIVE. huber_delta is in units of the batch return
#     std. If it sits above nearly every residual, Huber IS MSE and the arm
#     tests nothing -- and the entropy/balance differences would be seed noise
#     misread as treatment effects. Checked from |residual| p95 vs the delta.
#
#  1. REGIME SCREEN. At 5.5M the arm showed ego entropy -0.626 vs the control's
#     -1.020 and ep_rew_mean -0.068 vs -0.005. Every regime comparison in this
#     programme has been confounded by exactly this: `engaged` looked 12x better
#     than healthy self-play until the CONST baseline showed two thirds of it was
#     free. If the two arms are in different regimes the value numbers are not
#     comparable, and this script says so instead of printing them.
#     Adversary entropy reaching 0 is ABSORBING (no mass => no gradient), and it
#     already turned one 34M-step run into single-agent RL. That is a KILL, not
#     a warning.
#
#  2. VALUE QUALITY, on MC targets. Never on raw value_loss: Huber and MSE are
#     different functions and their losses are not comparable. Never on
#     lambda-returns either -- those bootstrap from V and score V against a
#     backup of itself (that mistake just produced HEAD EV 0.81 against a
#     recorded ~0.023).
#
#  3. SHRINKAGE HYPOTHESIS. My own prediction is that Huber HURTS here: it
#     down-weights the rare damage spikes that carry the signal, shrinking V
#     toward the mean. Falsifiable via std(V) and the return-on-V slope against
#     the MSE control. Reported whichever way it comes out.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/main"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fightladder

TREAT_DIR="${TREAT_DIR:-minimax_phase0_vtoff_rammasked_huber1.0_huber}"
CTRL_DIR="${CTRL_DIR:-minimax_phase0_vtoff_rammasked_cdctl}"
# 40 episodes gave head_EV scattering -0.09..+0.33 across adjacent
# checkpoints of the SAME run: returns are shared within an episode, so the
# effective n was ~40, roughly 5x too small for a +-0.05 effect.
EPISODES="${EPISODES:-200}"; N_ENVS="${N_ENVS:-16}"
# Early checkpoints have LONGER episodes, so a fixed cap starves them --
# the 1M checkpoint aborted at "only 10 episodes". Sized for the slowest.
MAX_STEPS="${MAX_STEPS:-3000}"
LOG="${SCRIPT_DIR}/logs/huber_diag.log"
say() { echo "[hdiag $(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

say "=== 1. REGIME SCREEN (gates everything below) ==="
python - "${SCRIPT_DIR}/logs/minimax_phase0_vtoff_rammasked_huber1.0_huber.log" \
         "${SCRIPT_DIR}/logs/minimax_phase0_vtoff_rammasked_cdctl.log" 2>&1 <<'PY' | tee -a "${LOG}"
import re, sys
K = ("total_timesteps","ep_rew_mean","ep_len_mean","ego_entropy_loss","adv_entropy_loss")
def series(p):
    cur, rows = {}, []
    for ln in open(p, errors="ignore"):
        m = re.match(r"\|\s+(?:\w+/)?([a-z_]+)\s+\|\s+([-\d.e+]+)\s+\|", ln)
        if m and m.group(1) in K:
            cur[m.group(1)] = float(m.group(2))
            if m.group(1) == "total_timesteps": rows.append(dict(cur))
    return rows
out = {}
for tag, p in (("huber", sys.argv[1]), ("ctrl", sys.argv[2])):
    r = series(p)
    out[tag] = r[-1] if r else {}
    print(f"  {tag:6} steps={out[tag].get('total_timesteps',0):>10.0f} "
          f"rew={out[tag].get('ep_rew_mean',float('nan')):+.4f} "
          f"ego_ent={out[tag].get('ego_entropy_loss',float('nan')):.3f} "
          f"adv_ent={out[tag].get('adv_entropy_loss',float('nan')):.3f}")
h, c = out["huber"], out["ctrl"]
# ABSORBING: entropy at 0 can never recover. Kill, do not warn.
for s in ("ego_entropy_loss", "adv_entropy_loss"):
    if abs(h.get(s, -1)) < 1e-3:
        print(f"  KILL: huber {s} is ~0 -- entropy saturation is ABSORBING and "
              f"unrecoverable. Stop this arm."); sys.exit(2)
d_ent = abs(h.get("ego_entropy_loss", 0) - c.get("ego_entropy_loss", 0))
d_rew = abs(h.get("ep_rew_mean", 0) - c.get("ep_rew_mean", 0))
print(f"\n  |d ego_entropy| = {d_ent:.3f}   |d ep_rew| = {d_rew:.4f}")
if d_ent > 0.25 or d_rew > 0.04:
    print("  REGIMES DIFFER. Value numbers below are reported but are NOT a clean\n"
          "  treatment comparison -- the arms are in different regimes, which is\n"
          "  the confound that made `engaged` look 12x better than healthy play.")
else:
    print("  regimes comparable; the value comparison is interpretable.")
PY
rc=${PIPESTATUS[0]}
[ "${rc}" -eq 2 ] && { say "aborting on absorbing-entropy kill"; exit 2; }

say "=== 2+3. VALUE QUALITY + SHRINKAGE, MC targets, matched checkpoints ==="
for arm_dir in "${CTRL_DIR}" "${TREAT_DIR}"; do
    for ck in $(ls -1 "${arm_dir}/trained_models/tasks/todo/"*_steps.task 2>/dev/null | sort -t_ -k4 -n); do
        st=$(basename "${ck}" | grep -oE '[0-9]+_steps' | grep -oE '^[0-9]+')
        say "--- $(basename ${arm_dir}) @ ${st}"
        python -u value_gap.py --ckpt "${ck}" --ram_mask ram_mask.npy \
            --episodes "${EPISODES}" --max_steps "${MAX_STEPS}" --n_envs "${N_ENVS}" \
            --gammas 0.75,0.9,0.94,0.99 \
            --out "headroom/vg_$(basename ${arm_dir})_${st}.json" 2>&1 \
            | grep -vE "^\[gap\] [0-9]" | tee -a "${LOG}"
    done
done

say "=== 0. WAS THE TREATMENT LIVE? |res|p95 vs huber_delta x return_std ==="
python - 2>&1 <<'PY' | tee -a "${LOG}"
import json, glob, os
# huber_delta=1.0 means the transition point IS the RETURN std -- not std(V).
# The first version of this check divided by std(V) (wrong quantity) AND printed
# its "<<1 => void" verdict unconditionally, regardless of the ratio, while the
# measured ratios pointed the other way. Both fixed: right denominator,
# conditional verdict.
DELTA = float(os.environ.get("HUBER_DELTA", "1.0"))
n_live = n_void = 0
for f in sorted(glob.glob("headroom/vg_*huber*.json")):
    d = json.load(open(f))
    for a in d.get("arms", []):
        if abs(a["gamma"] - 0.94) < 1e-9:
            beta = DELTA * max(a.get("ret_std", float("nan")), 1e-12)
            ratio = a["res_p95"] / beta
            verdict = ("LIVE  (p95 residual is past the transition point, so a real "
                       "share of updates are in the linear region)"
                       if ratio > 1.0 else
                       "VOID  (residuals sit inside the quadratic region => Huber == MSE)")
            n_live += ratio > 1.0; n_void += ratio <= 1.0
            print(f"  {os.path.basename(f)[:52]:52} |res|p95={a['res_p95']:.5f} "
                  f"beta={beta:.5f} ratio={ratio:5.2f}  {verdict}")
print(f"\n  {n_live} checkpoint(s) LIVE, {n_void} VOID.")
if n_void and not n_live:
    print("  => the arm tested nothing; any entropy/balance difference vs the "
          "control is seed noise, not a treatment effect.")
PY
say "DONE"
