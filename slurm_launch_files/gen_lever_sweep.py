"""Generate one-variable SLURM variants of the arm A config.

Mirrors main_training_orchestrator.py's approach: regex-substitute `^VAR="..."$`
lines in a base script and assert every substitution matched, so a renamed
variable fails loudly instead of silently producing a job with the wrong config.

The base is the arm C1 script, which is arm A + gamma. Each variant below flips
back whatever C1 changed and applies its own single change, so EVERY generated
job differs from ARM A by exactly one knob.

Arm A (c_lr 3e-5) is the reference because it is the config that avoids the
self-play collapse. Its measured position, for reading the results against:
    rating_gap  -81      score_rollout 0.438
    NashConv    0.44 pre-6M -> 0.52 post-6M   (WIDENED)
    critic EV   0.03-0.11 flat, |W| 52 -> 183 still climbing

Judge every variant on eps_greedy / NashConv, NOT rating_gap -- arm A's
rating_gap improved over exactly the window where its NashConv widened.

Usage:  python gen_lever_sweep.py [--dry_run]
"""
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "main_training_spar_gamma094_lam095_p-Ry_o-Sa.slurm")

# name -> (job tag, {VAR: value}, rationale printed into the script header)
VARIANTS = {
    # --- V-trace trace truncation -------------------------------------------
    # c_bar truncates the TRACE ratio: it sets variance and the effective credit
    # horizon, and provably does NOT move the fixed point (unlike rho_bar). So it
    # is the safe knob. Measured c_sat_frac ~0.505 on BOTH the baseline and arm A
    # -- half the traces clip at c_bar=1.0, and the trace coefficient at lag k is
    # prod(c_i) with every c_i <= 1, so the effective horizon is far shorter than
    # seq_len=64 or gamma imply. Completely untested to date.
    # Do NOT remove the bar: vtrace_ratio_max reached 119 on the baseline and 529
    # on arm A, so unclipped products would explode.
    "cbar2": ("cbar2", {"GAMMA": "", "GAE_LAMBDA": "", "VTRACE_C_BAR": "2.0"},
              "c_bar 1.0 -> 2.0: halve the fraction of clipped traces"),
    "cbar5": ("cbar5", {"GAMMA": "", "GAE_LAMBDA": "", "VTRACE_C_BAR": "5.0"},
              "c_bar 1.0 -> 5.0: match rho_bar; near-unclipped credit"),

    # --- timescale separation ------------------------------------------------
    # Arm A REDUCED separation (d_lr/c_lr 10x -> 3.3x, measured drift ratio
    # ~11 -> ~5) and avoided the collapse, which contradicts the Stackelberg
    # "more separation is better" argument. This takes it to 1x. If 1x is WORSE
    # than 3.3x, separation is a window with a floor; if similar or better, the
    # binding constraint was an absolute floor on the ego's learning rate and
    # separation is not the axis at all.
    "sep1x": ("sep1x", {"GAMMA": "", "GAE_LAMBDA": "", "C_LR": "1e-4"},
              "c_lr 3e-5 -> 1e-4: separation 3.3x -> 1x"),

    # --- rollout shape at constant data volume -------------------------------
    # 24x512 = 12,288 transitions/rollout, ~2.5 episodes/env.
    # 16x768 = 12,288 transitions/rollout, ~3.8 episodes/env.
    # Same data volume, more COMPLETE episodes per env. 16 is also the measured
    # knee of the env-throughput curve (4577 vs 4064 agent-steps/s at 24).
    "roll16": ("roll16", {"GAMMA": "", "GAE_LAMBDA": "",
                          "ENVS_PER_MATCHUP": "16", "NUM_ENV_STEPS": "768",
                          "ENV_BATCH_SIZE": "16"},
               "24x512 -> 16x768: same transitions, more complete episodes/env"),

    # --- replay reuse --------------------------------------------------------
    # ~88,500 updates against a 15,000 buffer, mean_age ~7,400. The
    # in-batch/held-out EV gap reached 0.45 on the baseline and swung to 0.42 on
    # arm A -- that reuse is the memorization mechanism. Was hardcoded until now.
    "replay5k": ("replay5k", {"GAMMA": "", "GAE_LAMBDA": "",
                              "VTRACE_REPLAY_CAPACITY": "5000"},
                 "replay 15000 -> 5000: cut reuse ~3x"),
}


def render(text, tag, overrides, rationale):
    for var, val in overrides.items():
        # Allow a trailing comment: several base vars carry one, e.g.
        #   C_LR="3e-5"                 # arm A's value; 1e-5 collapses by ~6M
        # A bare `..."$` anchor silently fails to match those, which the
        # assertion below then catches -- keep the comment rather than drop it.
        pat = rf'(?m)^{var}="[^"]*"(?P<tail>\s+#.*)?$'
        new, n = re.subn(pat, lambda m: f'{var}="{val}"' + (m.group("tail") or ""),
                         text)
        if n == 0:
            raise RuntimeError(f"[{tag}] substitution failed for {var} -- "
                               f"variable renamed or absent in the base script?")
        text = new
    text = text.replace("spar_g094_l095", f"spar_{tag}")
    text = text.replace("=== ARM C1: gamma ${GAMMA} / lambda ${GAE_LAMBDA} ===",
                        f"=== LEVER {tag}: {rationale} ===")
    # Replace C1's rationale block with this variant's.
    text = re.sub(r"(?ms)^# ARM C1 --.*?^set -u$",
                  f"# LEVER SWEEP: {tag}\n#\n"
                  f"#   {rationale}\n#\n"
                  f"# ONE variable against ARM A (c_lr 3e-5, gamma 0.99, lambda 0.95).\n"
                  f"# Generated by gen_lever_sweep.py -- edit that, not this file.\n"
                  f"#\n"
                  f"# Judge on eps_greedy / NashConv from local_best_response.py, NOT\n"
                  f"# rating_gap: arm A's rating_gap improved -154 -> -81 over exactly\n"
                  f"# the window where its NashConv WIDENED 0.44 -> 0.52. Read at >= 8M.\n"
                  f"set -u", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    base = open(BASE).read()
    # An empty GAMMA/GAE_LAMBDA must produce a job with NO such flag, or the
    # variant would carry C1's gamma change and stop being one-variable.
    if 'GAMMA=""' not in render(base, "probe", {"GAMMA": ""}, "x"):
        raise SystemExit("GAMMA override did not take")

    for name, (tag, overrides, why) in VARIANTS.items():
        out = os.path.join(HERE, f"main_training_spar_lever_{tag}_p-Ry_o-Sa.slurm")
        text = render(base, tag, overrides, why)
        print(f"{'would write' if a.dry_run else 'wrote':12s} {os.path.basename(out)}")
        print(f"             {why}")
        if not a.dry_run:
            with open(out, "w") as f:
                f.write(text)
            os.chmod(out, 0o755)


if __name__ == "__main__":
    main()
