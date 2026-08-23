"""Every CLI knob must actually REACH the training env. Tests ippo.make_env.

WHY THIS EXISTS. test_reward_variants.py passed 9/9 while the training path was
completely broken, because it builds envs with make_lbr_env() from
local_best_response.py -- which forwards **sf_kwargs -- whereas TRAINING uses
ippo.make_env(), which constructed SFWrapper without them. Six knobs
(--num_step_frames, --counterhit_kappa, --trade_kappa, --reset_close_range,
--pressure_beta, --pressure_range) were parsed, threaded through make_env's
signature, and silently dropped at the SFWrapper call.

The cost: a 12M-step close-range arm trained to weights BIT-IDENTICAL to its
control -- all 73 tensors, max abs difference 0.000e+00. Nothing raised. The
launcher tag said cr64, the log looked healthy, the run took nine GPU-hours,
and it was a replicate of the baseline.

A test that exercises a DIFFERENT construction path than the one under test is
indistinguishable from no test at all. This one asserts on the object the
training loop actually builds, and walks the wrapper chain to find SFWrapper
rather than trusting attribute forwarding -- gym.Wrapper.__getattr__ forwards
unknown names DOWN the chain, so a plain getattr on the outer wrapper would
succeed even if SFWrapper never received the value.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ippo
from common.retro_wrappers import SFWrapper

EXPECTED_CHECKS = 13
NC = 0


def chk(name, cond):
    global NC
    NC += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


MASK = np.load("ram_mask.npy")
ST = "Champion.Level1.RyuVsRyu.2Player"


def build(**kw):
    fn = ippo.make_env("StreetFighterIISpecialChampionEdition-Genesis", ST, "both",
                       "round", False, transform_action=True, obs_type="ram",
                       ram_mask=MASK, **kw)
    env = fn()
    sf = env
    while not isinstance(sf, SFWrapper):        # NOT getattr: __getattr__ forwards
        sf = sf.env
    return env, sf


# baseline: the defaults must be the historical values
env, sf = build()
chk("default num_step_frames is 8", sf.num_step_frames == 8)
chk("default counterhit_kappa is 0", sf.counterhit_kappa == 0.0)
chk("default reset_close_range is 0", sf.reset_close_range == 0.0)
env.close(); del env, sf

env, sf = build(num_step_frames=16)
chk("num_step_frames reaches SFWrapper", sf.num_step_frames == 16)
chk("combos were rebuilt to match", all(len(c) == 16 for c in sf.sf_combos))
env.close(); del env, sf

env, sf = build(reset_close_range=64.0)
chk("reset_close_range reaches SFWrapper", sf.reset_close_range == 64.0)
env.close(); del env, sf

env, sf = build(counterhit_kappa=2.0, attack_statuses=(524,))
chk("counterhit_kappa reaches SFWrapper", sf.counterhit_kappa == 2.0)
chk("attack_statuses reaches SFWrapper", sf.attack_statuses == frozenset({524}))
env.close(); del env, sf

env, sf = build(trade_kappa=3.0, attack_statuses=(524,))
chk("trade_kappa reaches SFWrapper", sf.trade_kappa == 3.0)
env.close(); del env, sf

env, sf = build(decision_timing="joint", actionable_statuses=(512, 514, 520),
                dwell_frames=4, max_skip_frames=120)
chk("decision_timing reaches SFWrapper", sf.decision_timing == "joint")
chk("actionable_statuses reaches SFWrapper",
    sf.actionable_statuses == frozenset({512, 514, 520}))
chk("dwell_frames reaches SFWrapper", sf.dwell_frames == 4)
chk("max_skip_frames reaches SFWrapper", sf.max_skip_frames == 120)
env.close(); del env, sf

if NC != EXPECTED_CHECKS:
    raise SystemExit(f"FAILED: ran {NC} checks, expected {EXPECTED_CHECKS} -- a "
                     f"check that does not run is indistinguishable from one "
                     f"that passes")
print(f"ALL {NC} PASS")
