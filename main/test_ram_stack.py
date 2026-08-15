"""RAM observation stacking: inertness at stack=1, and snapshot-safety.

TWO THINGS CAN GO WRONG SILENTLY.

1. stack=1 must be BITWISE identical to the pre-feature behaviour, or every
   existing checkpoint is invalidated by a flag nobody set.

2. RamObsWrapper sits OUTSIDE SFWrapper, so env_method("lbr_snapshot") forwards
   straight past its history deque. If the deque is not saved and restored, an
   enumeration branch inherits the PREVIOUS branch's frames -- the observation
   differs from the true one in stack-1 of its stack-many slots while the
   emulator state is correct. Nothing raises. The head is trained on a
   corrupted input and the failure looks like "the head just doesn't learn".
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_best_response import make_lbr_env, infer_obs_kwargs

EXPECTED_CHECKS = 12
NC = 0
def chk(name, cond):
    global NC; NC += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond: raise SystemExit(f"FAILED: {name}")

mask = np.load("ram_mask.npy"); n = mask.size
ST = "Champion.Level1.RyuVsRyu.2Player"
ACTS = [np.array([3, 7]), np.array([11, 2]), np.array([5, 5]), np.array([9, 1])]

# stable-retro allows ONE emulator per process, so the two configurations are
# rolled sequentially and compared through their recorded observations. Same
# state, same seed, same scripted actions => the emulators are identical, so any
# difference in the newest slot is the wrapper's doing.
e1 = make_lbr_env(ST, obs_type="ram", ram_mask=mask, ram_stack=1, seed=0)()
chk("stack=1 width == mask size", e1.observation_space.shape == (n,))
o1 = e1.reset()
chk("stack=1 reset obs width", o1.shape == (n,))
hist1 = [o1]
for a in ACTS:
    hist1.append(e1.step(a)[0])
e1.close()
del e1

e4 = make_lbr_env(ST, obs_type="ram", ram_mask=mask, ram_stack=4, seed=0)()
chk("stack=4 width == 4x mask size", e4.observation_space.shape == (4 * n,))
o4 = e4.reset()
chk("reset fills the stack with copies of frame 0",
    all(np.array_equal(o4[i*n:(i+1)*n], o4[:n]) for i in range(4)))
chk("stack=4 newest slot == the stack=1 obs", np.array_equal(o4[-n:], o1))
hist4 = [o4]
for a in ACTS:
    hist4.append(e4.step(a)[0])
chk("stack=4 newest slot tracks stack=1 at every step",
    all(np.array_equal(b[-n:], a) for a, b in zip(hist1, hist4)))
chk("stack=4 older slots are the PREVIOUS frames (history is real)",
    np.array_equal(hist4[4][-2*n:-n], hist1[3]) and
    np.array_equal(hist4[4][:n], hist1[1]))

# --- the corruption case ---------------------------------------------------
# Snapshot at the root, walk away, restore. If the deque were not restored, the
# post-restore obs would carry the frames from the walk.
e4.lbr_snapshot("root")
root = hist4[-1].copy()
for a in ACTS: e4.step(a)
e4.lbr_restore("root")
back = e4.step(ACTS[0])[0]
e4.lbr_restore("root")
back2 = e4.step(ACTS[0])[0]
chk("restore rewinds the frame history, not just the emulator",
    np.array_equal(back, back2) and np.array_equal(back[:-n], root[n:]))
e4.lbr_drop()
chk("obs width inferred back from the checkpoint width",
    infer_obs_kwargs({"observation_space": type("S", (), {"shape": (4*n,)})()},
                     mask)["ram_stack"] == 4)
e4.close()
del e4

# --- stride ----------------------------------------------------------------
# An agent step advances num_step_frames=8 emulator frames. At the default
# stride 8 the stack samples once per agent step; at stride 1 it samples
# consecutive emulator frames, which is the only way to see events shorter than
# a step (a special move's active frames last ~2-4).
e12 = make_lbr_env(ST, obs_type="ram", ram_mask=mask, ram_stack=12, ram_stride=1,
                   seed=0)()
chk("stride does not change the obs width", e12.observation_space.shape == (12*n,))
o = e12.reset()
chk("reset still fills every slot",
    all(np.array_equal(o[i*n:(i+1)*n], o[:n]) for i in range(12)))
o = e12.step(ACTS[0])[0]
sl = [o[i*n:(i+1)*n] for i in range(12)]
# after ONE step from reset, the newest 8 slots are the 8 emulator frames that
# step generated, so they cannot all be equal unless the game froze.
chk("stride=1 resolves WITHIN an agent step (sub-step frames differ)",
    len({sl[i].tobytes() for i in range(4, 12)}) > 1)
e12.close()

if NC != EXPECTED_CHECKS:
    raise SystemExit(f"FAILED: ran {NC} checks, expected {EXPECTED_CHECKS} -- "
                     f"a check that does not run is indistinguishable from one that passes")
print(f"ALL {NC} PASS")
