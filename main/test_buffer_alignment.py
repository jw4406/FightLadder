"""Does Q_RolloutBuffer keep (obs, ego_action, adv_action, log_prob) aligned?

WHY. In the masked-RAM run the ADVERSARY's first-minibatch KL drifts
0.00000 -> 0.137 and then aborts every update, collapsing self-play at a
reproducible 159,744 steps. At the first minibatch nothing has been updated, so
`log_prob` recomputed from the buffer MUST equal the stored `old_log_prob`.
Measured facts:

  * at COLLECTION, evaluate_adv_actions() reproduces the stored adversary
    log-prob EXACTLY (max |diff| = 0.000000 over 352 samples through 209k steps),
    so the policy forward path and the write are both correct;
  * at TRAINING, the same recomputation on buffer data is off by ~0.15 KL;
  * the EGO path stays clean throughout.

That isolates the corruption to the buffer, or to how training indexes it.
`prepare_data_for_training` reshapes every tensor with

    th_tensor.transpose(0, 1).contiguous().view(shape[0]*shape[1], *shape[2:])

while `env_indices` goes through `swap_and_flatten`. If those two orderings
disagree, or if any field is missed, rows get paired with another timestep's
data -- a defect that produces a STABLE wrong answer rather than a crash, and
that stays invisible while the policy is near-uniform (all log-probs ~ log(1/22))
and only grows as it sharpens. Exactly the observed curve.

METHOD. Fill the buffer with values that ENCODE their own (step, env) origin, so
any permutation is detectable:

    obs[t, e]        = t * 1000 + e         (broadcast)
    ego_action[t, e] = t * 1000 + e + 0.1
    adv_action[t, e] = t * 1000 + e + 0.2
    log_prob[t, e]   = t * 1000 + e + 0.3

After flattening, every row must carry ONE consistent (t, e). This needs no GPU,
no policy and no env.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import Q_RolloutBuffer

N_STEPS, N_ENVS, OBS_DIM = 7, 5, 3


def build():
    obs_space = spaces.Box(0, 255, (OBS_DIM,), dtype=np.uint8)
    act_space = spaces.MultiDiscrete([22])
    buf = Q_RolloutBuffer(N_STEPS, obs_space, act_space, device="cpu", n_envs=N_ENVS)
    buf.reset()
    for t in range(N_STEPS):
        code = np.array([t * 1000 + e for e in range(N_ENVS)], dtype=np.float32)
        buf.add(
            obs=np.repeat(code[:, None], OBS_DIM, axis=1).astype(np.uint8),
            action=(code + 0.1)[:, None],
            adv_action=(code + 0.2)[:, None],
            reward=code.copy(),
            next_obs=np.repeat(code[:, None], OBS_DIM, axis=1).astype(np.uint8),
            dones=np.zeros(N_ENVS, np.float32),
            episode_start=np.zeros(N_ENVS, np.float32),
            value=th.as_tensor(code.copy()),
            log_prob=th.as_tensor(code + 0.3),
            q_value=th.as_tensor(code.copy()),
        )
    return buf


def main():
    buf = build()
    buf.prepare_data_for_training()

    ego = buf.ego_actions.reshape(-1).cpu().numpy()
    adv = buf.adv_actions.reshape(-1).cpu().numpy()
    lp = buf.log_probs.reshape(-1).cpu().numpy()
    acts = buf.actions.reshape(-1).cpu().numpy()
    n = ego.size

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    # Each field encodes its own origin; strip the tag to recover (t, e).
    base_ego = np.round(ego - 0.1).astype(np.int64)
    base_adv = np.round(adv - 0.2).astype(np.int64)
    base_lp = np.round(lp - 0.3).astype(np.int64)

    check("ego_action and adv_action come from the same (step, env)",
          np.array_equal(base_ego, base_adv),
          f"mismatched rows: {int((base_ego != base_adv).sum())}/{n}")
    check("log_prob comes from the same (step, env) as adv_action",
          np.array_equal(base_adv, base_lp),
          f"mismatched rows: {int((base_adv != base_lp).sum())}/{n}")
    check("`actions` alias matches `ego_actions`",
          np.allclose(acts, ego))

    # observations are uint8, so the code only survives for small values; compare
    # ordering rather than exact magnitude.
    obs0 = buf.observations.reshape(n, OBS_DIM)[:, 0].cpu().numpy().astype(np.int64)
    exp = (base_ego % 256)
    check("observations align with actions (mod 256)",
          np.array_equal(obs0, exp),
          f"mismatched rows: {int((obs0 != exp).sum())}/{n}")

    # env_indices uses swap_and_flatten, the others use transpose+view. If those
    # two disagree the samples are labelled with the wrong env.
    ei = np.asarray(buf.env_indices).reshape(-1)
    if ei.size == n and np.any(ei != 0):
        check("env_indices ordering matches the tensor ordering",
              np.array_equal(ei.astype(np.int64), base_ego % 1000))
    else:
        print("  SKIP  env_indices is all zeros in this synthetic buffer")

    print(f"\n  first 10 rows (expect step-major within each env):")
    print(f"    {'row':>4} {'step':>5} {'env':>4}")
    for k in range(min(10, n)):
        print(f"    {k:>4} {base_ego[k]//1000:>5} {base_ego[k]%1000:>4}")

    print("\n  ALL PASS -- the buffer preserves alignment; look elsewhere"
          if ok else "\n  MISALIGNMENT CONFIRMED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
