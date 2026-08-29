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

EXPECTED_CHECKS = 22
NC = 0


def chk(name, cond):
    global NC
    NC += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


MASK = np.load("ram_mask.npy")
ST = "Champion.Level1.RyuVsRyu.2Player"
ST_RG = "Champion.Level1.RyuVsGuile"          # distinct chars: left=Ryu (6 macros), right=Guile (3)


def build(state=ST, **kw):
    fn = ippo.make_env("StreetFighterIISpecialChampionEdition-Genesis", state, "both",
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

# --- per-ego / per-seat special-move macros reach SFWrapper via ippo.make_env ---
# distinct-char state -> P1 decoded with LEFT char's table, P2 with RIGHT char's (the 2-player fix).
env, sf = build(state=ST_RG)                              # ego_is_left defaults True -> ego = left = Ryu
chk("per-seat P1 table = left char Ryu (6 macros)", len(sf.sf_combos_p1) == 6)
chk("per-seat P2 table = right char Guile (3 macros)", len(sf.sf_combos_p2) == 3)
chk("ego table = left char when ego_is_left (Ryu, 6)", len(sf.sf_combos) == 6)
chk("ego action space sized by ego char (63+6=69)", int(sf.action_space.nvec[0]) == 69)
# functional: a Ryu special actually FIRES through the ippo path (deterministic one-shot motion).
raw = env
while not hasattr(raw, "get_ram"):
    raw = raw.env
env.reset()
for _ in range(60):
    env.step(np.array([0, 0]))                           # settle both seats neutral
flag = 0
env.step(np.array([63, 0]))                              # P1 = Ryu combo 63 (hadouken_r), P2 neutral
flag = max(flag, int(raw.get_ram()[32773]))              # 0xFF8005 = P1 special flag
for _ in range(20):
    env.step(np.array([0, 0]))
    flag = max(flag, int(raw.get_ram()[32773]))
chk("Ryu P1 special FIRES via ippo per-ego path (flag != 0)", flag != 0)
env.close(); del env, sf

# ego on the RIGHT -> ego table follows the right char; action space subsets to it.
env, sf = build(state=ST_RG, ego_is_left=False)          # ego = right = Guile
chk("ego table follows ego_is_left=False (Guile, 3)", len(sf.sf_combos) == 3)
chk("ego action space subsets for Guile (63+3=66)", int(sf.action_space.nvec[0]) == 66)
chk("per-seat tables unchanged by ego side (P1=6, P2=3)",
    len(sf.sf_combos_p1) == 6 and len(sf.sf_combos_p2) == 3)
env.close(); del env, sf

# same-char state -> both seats get the same (shoto) table; legacy-compatible size.
env, sf = build()                                        # ST = RyuVsRyu
chk("same-char state -> both tables shoto (6/6)",
    len(sf.sf_combos_p1) == 6 and len(sf.sf_combos_p2) == 6)
env.close(); del env, sf

if NC != EXPECTED_CHECKS:
    raise SystemExit(f"FAILED: ran {NC} checks, expected {EXPECTED_CHECKS} -- a "
                     f"check that does not run is indistinguishable from one "
                     f"that passes")
print(f"ALL {NC} PASS")
