"""Frame skip is adjustable -- and special moves must STILL FIRE at the new skip.

THE FAILURE THIS EXISTS TO CATCH. Motion inputs (Hadouken, Shoryuken) are
4-input sequences held for num_step_frames/4 frames each. At the historical
skip of 8 that is 2 frames per input. At 4 it is 1. If one frame per input is
too short for the game engine to register the motion, the six combo actions
become DEAD -- selectable, but they do nothing. Nothing would raise: the agent
would simply have six actions that never work, the action space would silently
shrink from 22 to 16, and every payoff matrix would be quietly degenerate in
six rows and six columns.

That failure is invisible to any test that only checks shapes and asserts, so
this one drives each combo action from a live emulator and checks that the
character's status actually LEAVES neutral.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_best_response import make_lbr_env
from common.const import build_sf_combos, DIRECTIONS_BUTTONS, ATTACKS_BUTTONS

EXPECTED_CHECKS = 6
NC = 0
NEUTRAL = 512          # standing/idle; anything else means an animation started


def chk(name, cond):
    global NC
    NC += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


ST = "Champion.Level1.RyuVsRyu.2Player"
MASK = np.load("ram_mask.npy")
COMBO0 = len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS)     # first combo index


def combo_fire_rate(nsf):
    """Fraction of the 6 motion inputs that visibly leave neutral."""
    env = make_lbr_env(ST, obs_type="ram", ram_mask=MASK, seed=0,
                       num_step_frames=nsf)()
    env.reset()
    for _ in range(8):
        env.step(np.array([0, 0]))
    fired = 0
    for c in range(len(build_sf_combos(nsf))):
        # settle to neutral, then issue the motion several times -- a motion can
        # legitimately fail once if the character is mid-animation.
        for _ in range(6):
            env.step(np.array([0, 0]))
        seen = False
        for _ in range(4):
            info = env.step(np.array([COMBO0 + c, 0]))[-1]
            if info["agent_status"] != NEUTRAL:
                seen = True
        fired += int(seen)
    env.close()
    return fired / len(build_sf_combos(nsf))


for nsf in (8, 4):
    combos = build_sf_combos(nsf)
    chk(f"nsf={nsf}: combos are exactly {nsf} frames",
        all(len(c) == nsf for c in combos))
    rate = combo_fire_rate(nsf)
    print(f"        nsf={nsf}: {rate:.0%} of motion inputs left neutral")
    chk(f"nsf={nsf}: motion inputs still fire (>=2/3 leave neutral)", rate >= 2 / 3)
    del combos

ok = False
try:
    build_sf_combos(6)
except ValueError:
    ok = True
chk("nsf=6 is REJECTED (would truncate a motion input unevenly)", ok)

env = make_lbr_env(ST, obs_type="ram", ram_mask=MASK, seed=0, num_step_frames=4)()
chk("action space is still 22 at nsf=4 (no combo silently dropped)",
    int(env.lbr_config()["n_actions"]) == 22)
env.close()

if NC != EXPECTED_CHECKS:
    raise SystemExit(f"FAILED: ran {NC} checks, expected {EXPECTED_CHECKS} -- a "
                     f"check that does not run is indistinguishable from one "
                     f"that passes")
print(f"ALL {NC} PASS")
