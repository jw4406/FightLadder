"""The contact-density reward variants must stay ZERO-SUM and inert by default.

TWO PROPERTIES, AND BOTH FAIL SILENTLY IF BROKEN.

1. DEFAULT INERTNESS. kappa = beta = 0 must reproduce the historical reward
   bitwise. Otherwise every existing baseline is invalidated by flags nobody set.

2. ZERO-SUM. minimax-Q solves a zero-sum matrix game; if r + r_inverse != 0 the
   operator is solving the wrong object and every downstream number is
   meaningless. Nothing in the training loop checks this, and the existing
   `aggresive_coeff` knob ALREADY breaks it: r + r_inv = (a-1)(D_e + D_a). That
   knob is right there, it is named "aggressive", and turning it up is the most
   natural thing anyone would try. This test pins the property so a future edit
   cannot quietly lose it.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_best_response import make_lbr_env

EXPECTED_CHECKS = 9
NC = 0


def chk(name, cond):
    global NC
    NC += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


ST = "Champion.Level1.RyuVsRyu.2Player"
MASK = np.load("ram_mask.npy")
# statuses seen in play; the analysis derives the real set, any subset works here
ATK = (514, 516, 520, 522, 524, 526)
SEQ = [np.array([a % 22, (a * 7 + 3) % 22]) for a in range(400)]


def roll(**kw):
    """Same seed, same scripted actions => identical emulator trajectory."""
    env = make_lbr_env(ST, obs_type="ram", ram_mask=MASK, seed=0, **kw)()
    env.reset()
    out = []
    for a in SEQ:
        _, rl, rr, d, _ = env.step(a)
        out.append((rl, rr))
        if d:
            env.reset()
    env.close()
    return np.array(out)


base = roll()
chk("baseline produced nonzero rewards (the test can detect a change at all)",
    np.abs(base).sum() > 0)
chk("baseline is zero-sum", np.abs(base[:, 0] + base[:, 1]).max() < 1e-12)

inert = roll(counterhit_kappa=0.0, pressure_beta=0.0)
chk("kappa=0, beta=0 is BITWISE identical to the historical reward",
    np.array_equal(inert, base))

ch = roll(counterhit_kappa=4.0, attack_statuses=ATK)
chk("counterhit kappa=4 is zero-sum", np.abs(ch[:, 0] + ch[:, 1]).max() < 1e-12)
chk("counterhit kappa=4 actually CHANGED the reward (not a no-op)",
    not np.array_equal(ch, base))

pr = roll(pressure_beta=0.01, pressure_range=60.0, attack_statuses=ATK)
chk("pressure beta=0.01 is zero-sum", np.abs(pr[:, 0] + pr[:, 1]).max() < 1e-12)
chk("pressure beta=0.01 actually CHANGED the reward",
    not np.array_equal(pr, base))

tr = roll(trade_kappa=4.0, attack_statuses=ATK)
chk("trade kappa=4 is zero-sum", np.abs(tr[:, 0] + tr[:, 1]).max() < 1e-12)
chk("trade kappa=4 actually CHANGED the reward", not np.array_equal(tr, base))

if NC != EXPECTED_CHECKS:
    raise SystemExit(f"FAILED: ran {NC} checks, expected {EXPECTED_CHECKS} -- a "
                     f"check that does not run is indistinguishable from one "
                     f"that passes")
print(f"ALL {NC} PASS")
