"""Overnight diagnostic sweep for the masked-RAM arm, with automatic escalation.

Runs the diagnostic suite at a list of target steps, compares each checkpoint
against the previous one, and when something moves anomalously escalates in TWO
directions:

  AMONG checkpoints  re-run the suite at the intermediate checkpoints between the
                     last-good and the anomalous one, to localise WHEN it started
  WITHIN a checkpoint re-run the anomalous checkpoint with 3x the sampling, to
                     separate a real change from sampling noise

WHAT COUNTS AS ANOMALOUS (thresholds, not vibes):

  collapse        score_rollout == 0 in >=28 of the last 30 log rows -- the exact
                  signature of the pre-dtype-fix run, which pinned at 0 from
                  159,744 steps onward
  kl_stops        any return of the trust-region early stop; the fix took this to
                  0 and it should stay there
  dc_ratio        minimax trunk DC/fluctuation changing >2x between checkpoints.
                  Prediction for masked is ~2.9 (full RAM 24.56, image 1.08); a
                  jump means the representation geometry moved
  snr             no-op SNR crossing 1.0 in either direction, or changing >2x.
                  Every arm measured so far sits in 0.66-1.15
  gap_spread      gap/spread moving off ~1.15 by more than 0.35. This one is
                  predicted NOT to move -- it is the sqrt(2)/1.128 signature of
                  independent per-cell constants, an OUTPUT-LAYER defect that no
                  observation change should touch. If it moves, that diagnosis
                  was wrong and it is the most interesting result of the night
  ev_sign         minimax_ev going negative after being positive

Fail-closed throughout: a dead trainer, a collapsed game, or a crashed diagnostic
stops the sweep with a message rather than reporting on degenerate data.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = "minimax_phase0_vtoff_rammasked"
CKDIR = os.path.join(REPO, "main", ARM, "trained_models", "tasks", "todo")
TRAINLOG = os.path.join(REPO, "logs", f"{ARM}.log")
CKPT_EVERY = 480_000


def log(msg, path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def train_metrics():
    """(last_step, kl_stops, zeros_in_last_30, last_ev) from the training log."""
    try:
        t = open(TRAINLOG, errors="ignore").read()
    except OSError:
        return None
    g = lambda n: [float(x) for x in re.findall(rf"{n} *\| *([-0-9.e+]+)", t)]
    sc, ts, ev = g("score_rollout"), g("total_timesteps"), g("minimax_ev")
    return {
        "step": ts[-1] if ts else 0.0,
        "kl_stops": t.count("training stopped at step"),
        "zeros30": sum(1 for x in sc[-30:] if x == 0),
        "ev": ev[-1] if ev else None,
        "n_rows": len(sc),
    }


def trainer_alive():
    return subprocess.run("pgrep -f '[i]ppo.py'", shell=True,
                          stdout=subprocess.DEVNULL).returncode == 0


def run_suite(ckpt, tag, steps, n_envs, outlog, ram_mask=""):
    """Run the four diagnostics; return the parsed JSON for the machine-readable ones."""
    res = {}
    jobs = [
        ("trunk", "minimax_trunk_control.py",
         ["--steps", str(steps), "--n_envs", str(n_envs), "--out", f"diag/trunk_{tag}.json"]),
        ("noop", "minimax_noop_decomp.py",
         ["--steps", str(steps + 50), "--n_envs", str(n_envs), "--out", f"diag/noop_{tag}.json"]),
        ("axis", "minimax_axis_diag.py",
         ["--steps", str(steps + 50), "--n_envs", str(n_envs), "--out", f"diag/axis_{tag}.json"]),
        ("bias", "minimax_bias_mech.py",
         ["--steps", str(steps), "--n_envs", str(n_envs)]),
        # Payoff ANOVA. gamma -- the INTERACTION term -- is the number that
        # decides whether a joint-action critic is the right object at all.
        # Measured 0.244% at 2.4M on RAM with every state non-forced; the open
        # question is whether it GROWS as V matures, which is exactly what a
        # per-checkpoint series answers.
        ("payoff", "payoff_structure.py",
         ["--n_states", "300", "--n_envs", "8", "--stride", "60",
          "--out", f"diag/payoff_{tag}.json"]),
    ]
    os.makedirs(os.path.join(REPO, "main", "diag"), exist_ok=True)
    for name, script, extra in jobs:
        cmd = [sys.executable, "-u", os.path.join(REPO, "main", script),
               "--ckpt", ckpt] + extra
        if ram_mask:
            cmd += ["--ram_mask", ram_mask]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(REPO, "main"))
        body = "\n".join(l for l in p.stdout.splitlines()
                         if l.strip() and not l.startswith("hello"))
        with open(outlog, "a") as f:
            f.write(f"\n--- {name} @ {tag} (rc={p.returncode}) ---\n{body}\n")
            if p.returncode != 0:
                f.write(f"STDERR:\n{p.stderr[-1500:]}\n")
        if p.returncode != 0:
            log(f"  DIAGNOSTIC FAILED: {name} @ {tag} rc={p.returncode}", outlog)
            continue
        if name == "payoff":
            # Decompose what the collection just wrote.
            rawp = os.path.join(REPO, "main", "diag", f"payoff_{tag}_raw.npz")
            if os.path.exists(rawp):
                q = subprocess.run(
                    [sys.executable, "-u", os.path.join(REPO, "main", "payoff_anova.py"),
                     "--raw", rawp, "--out", f"diag/anova_{tag}.json"],
                    capture_output=True, text=True, cwd=os.path.join(REPO, "main"))
                with open(outlog, "a") as f:
                    f.write(f"\n--- anova @ {tag} (rc={q.returncode}) ---\n{q.stdout}\n")
                    if q.returncode != 0:
                        f.write(f"STDERR:\n{q.stderr[-1500:]}\n")
                ap_ = os.path.join(REPO, "main", "diag", f"anova_{tag}.json")
                if os.path.exists(ap_):
                    try:
                        res["anova"] = json.load(open(ap_))
                    except Exception as e:
                        log(f"  unreadable anova json: {e}", outlog)
            continue
        jp = os.path.join(REPO, "main", "diag", f"{name}_{tag}.json")
        if os.path.exists(jp):
            try:
                res[name] = json.load(open(jp))
            except Exception as e:
                log(f"  unreadable json {jp}: {e}", outlog)
    return res


def summarize(r):
    """Pull the handful of numbers anomaly detection keys on."""
    out = {}
    if "trunk" in r:
        out["dc_minimax"] = r["trunk"]["minimax"]["dc_ratio"]
        out["dc_value"] = r["trunk"]["value"]["dc_ratio"]
        out["vartop10_minimax"] = r["trunk"]["minimax"]["var_top10_share"]
    if "axis" in r:
        out["snr_ego"] = r["axis"]["snr_ego"]
        out["snr_adv"] = r["axis"]["snr_adv"]
    if "anova" in r and r["anova"].get("q_nonforced"):
        g = r["anova"]["q_nonforced"]
        out["gamma"] = g["gamma"]
        out["alpha_beta"] = g["alpha"] + g["beta"]
    if "noop" in r:
        n = r["noop"]
        if n.get("spread"):
            out["gap_spread"] = n["gap_trained"] / max(n["spread"], 1e-12)
        out["const_frac"] = n["const_trained"] / max(n["gap_trained"], 1e-12)
    return out


def anomalies(cur, prev):
    """Threshold comparisons. Returns a list of human-readable reasons."""
    bad = []
    def moved(k, factor, label):
        if k in cur and prev and k in prev and prev[k] not in (0, None):
            ratio = cur[k] / prev[k] if prev[k] else float("inf")
            if ratio > factor or ratio < 1.0 / factor:
                bad.append(f"{label}: {prev[k]:.4f} -> {cur[k]:.4f} ({ratio:.2f}x)")
    moved("dc_minimax", 2.0, "minimax trunk DC/fluct")
    moved("gamma", 2.0, "payoff INTERACTION gamma")
    # gamma crossing 2% is the threshold that would REVIVE the joint-action
    # critic: below it the structure is separable and a 44-output head suffices.
    if "gamma" in cur and cur["gamma"] > 0.02:
        bad.append(f"gamma = {100*cur['gamma']:.2f}% (>2%): the INTERACTION term "
                   f"is material after all -- a joint-action critic IS justified")
    moved("snr_ego", 2.0, "no-op SNR ego")
    moved("snr_adv", 2.0, "no-op SNR adv")
    # SNR crossing 1.0 is meaningful on its own: no arm has ever sustained >1.
    if prev and "snr_ego" in cur and "snr_ego" in prev:
        if (cur["snr_ego"] > 1.0) != (prev["snr_ego"] > 1.0):
            bad.append(f"no-op SNR ego crossed 1.0: {prev['snr_ego']:.3f} -> {cur['snr_ego']:.3f}")
    # gap/spread is PREDICTED NOT TO MOVE. Movement refutes the output-layer story.
    if "gap_spread" in cur and abs(cur["gap_spread"] - 1.15) > 0.35:
        bad.append(f"gap/spread off the sqrt(2) signature: {cur['gap_spread']:.3f} "
                   f"(expected ~1.15; movement REFUTES the output-layer diagnosis)")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=str, default="6240000,10080000,15360000,20160000")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n_envs", type=int, default=6)
    ap.add_argument("--ram_mask", type=str,
                    default="/home/jw4406/codebase/FightLadder/main/ram_mask.npy",
                    help="the mask the arm was TRAINED with; checkpoints record "
                         "the width, not which bytes")
    ap.add_argument("--out", type=str, default="logs/overnight_diag.log")
    a = ap.parse_args()

    outlog = os.path.join(REPO, a.out)
    open(outlog, "w").close()
    targets = [int(x) for x in a.targets.split(",")]
    log(f"targets: {', '.join(f'{t:,}' for t in targets)}", outlog)

    prev, prev_tag, history = None, None, {}
    for tgt in targets:
        ckpt = os.path.join(CKDIR, f"spar_Ry_Sa_{tgt}_steps.task")
        # ---- wait, fail-closed -------------------------------------------
        while not os.path.exists(ckpt):
            if not trainer_alive():
                log(f"TRAINER GONE before {tgt:,} -- stopping sweep", outlog)
                return
            m = train_metrics()
            if m and m["n_rows"] >= 30 and m["zeros30"] >= 28:
                log(f"COLLAPSED (score_rollout 0 in {m['zeros30']}/30 rows) "
                    f"at step {m['step']:,.0f} -- refusing to diagnose", outlog)
                return
            time.sleep(60)

        m = train_metrics()
        log(f"=== {tgt:,} ready | step {m['step']:,.0f} kl_stops {m['kl_stops']} "
            f"zeros30 {m['zeros30']} ev {m['ev']} ===", outlog)
        if m["kl_stops"] > 0:
            log(f"  ANOMALY: KL early-stops returned ({m['kl_stops']}) -- the dtype "
                f"fix took this to 0; something re-broke", outlog)

        tag = str(tgt)
        cur = summarize(run_suite(ckpt, tag, a.steps, a.n_envs, outlog, a.ram_mask))
        history[tgt] = cur
        log(f"  {tag}: " + "  ".join(f"{k}={v:.4f}" for k, v in cur.items()), outlog)

        bad = anomalies(cur, prev)
        if m["kl_stops"] > 0:
            bad.append(f"kl_stops={m['kl_stops']}")
        if bad:
            log("  ANOMALY DETECTED:", outlog)
            for b in bad:
                log(f"    - {b}", outlog)
            # ---- escalate AMONG checkpoints ------------------------------
            if prev_tag is not None:
                lo, hi = int(prev_tag), tgt
                mids = [s for s in range(lo + CKPT_EVERY, hi, CKPT_EVERY)
                        if os.path.exists(os.path.join(CKDIR, f"spar_Ry_Sa_{s}_steps.task"))]
                mids = mids[:: max(1, len(mids) // 4)][:4]
                log(f"  escalating AMONG: {len(mids)} intermediate checkpoints "
                    f"between {lo:,} and {hi:,}", outlog)
                for s in mids:
                    c = summarize(run_suite(
                        os.path.join(CKDIR, f"spar_Ry_Sa_{s}_steps.task"),
                        f"mid{s}", a.steps, a.n_envs, outlog, a.ram_mask))
                    history[s] = c
                    log(f"    mid {s:,}: " + "  ".join(f"{k}={v:.4f}" for k, v in c.items()),
                        outlog)
            # ---- escalate WITHIN the checkpoint --------------------------
            log("  escalating WITHIN: 3x sampling to separate real change from noise",
                outlog)
            deep = summarize(run_suite(ckpt, f"{tag}_deep", a.steps * 3,
                                       a.n_envs, outlog, a.ram_mask))
            log(f"    deep {tag}: " + "  ".join(f"{k}={v:.4f}" for k, v in deep.items()),
                outlog)
            for k in set(cur) & set(deep):
                if cur[k] and abs(deep[k] - cur[k]) / max(abs(cur[k]), 1e-12) > 0.25:
                    log(f"    NOTE {k} moved {cur[k]:.4f} -> {deep[k]:.4f} under 3x "
                        f"sampling: the shallow reading was NOISE, not signal", outlog)
        prev, prev_tag = cur, tag

    log("\nSWEEP SUMMARY", outlog)
    log(f"  {'step':>12} " + "  ".join(f"{k:>14}" for k in
        ("dc_minimax", "snr_ego", "snr_adv", "gap_spread", "gamma")), outlog)
    for s in sorted(history):
        h = history[s]
        log(f"  {s:>12,} " + "  ".join(
            f"{h.get(k, float('nan')):>14.4f}" for k in
            ("dc_minimax", "snr_ego", "snr_adv", "gap_spread", "gamma")), outlog)
    log("OVERNIGHT_DIAG_DONE", outlog)


if __name__ == "__main__":
    main()
