"""Record the ON-POLICY state visitation of a checkpoint, in GAME-STATE units.

WHY THE DESCRIPTOR IS THE RAM, NOT THE LATENT. The question is whether V-trace
changes WHICH STATES the policy visits. Encoding vton states with vton's encoder
and vtoff states with vtoff's would confound "the representation changed" with
"the visitation changed" -- and those are exactly the two things that must be
told apart. The retro `info` dict is forwarded intact through SFWrapper.step
(retro_wrappers.py:450) and carries the actual emulator variables:

    agent_x agent_y enemy_x enemy_y   positions
    agent_hp enemy_hp                 health
    agent_status enemy_status         crouch / jump / attack / stun
    round_countdown                   clock

No learned component, identical meaning across arms, directly interpretable.

WHAT IS SAMPLED. Both seats act from their OWN policies -- the visitation of a
self-play system IS the joint policy's occupancy, so anything else measures a
different distribution. This is the opposite choice from minimax_probe_ceiling
--uniform_actions, which deliberately breaks on-policy sampling to get action
coverage; here on-policy is the entire point.

EVERY STEP IS A SAMPLE. Unlike the return probes, no episode needs to FINISH --
visitation is a distribution over states, not over outcomes. Completed episodes
are still counted, but only to report ep_len alongside: if two arms differ in
episode length they differ in the mix of early/late states, and that alone moves
every marginal. It is a confound to report, not to hide.

Pairs with compare_visitation.py, which needs >=2 SEEDS per arm to calibrate how
large a distance is from rollout noise alone.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Numeric emulator variables worth recording. Anything absent from the live info
# dict is dropped with a warning rather than crashing -- data.json may vary by
# integration.
INFO_KEYS = ("agent_x", "agent_y", "enemy_x", "enemy_y",
             "agent_hp", "enemy_hp", "agent_status", "enemy_status",
             "round_countdown", "agent_victories", "enemy_victories", "score")

MIN_SAMPLES = 20000
MIN_EPISODES = 50


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=3000, help="vec-steps")
    ap.add_argument("--n_envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0,
                    help="RUN AT LEAST TWO PER ARM. compare_visitation.py uses "
                         "the between-seed distance as the null; without it any "
                         "nonzero cross-arm distance looks like a finding.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", required=True, help="path to the output .npz")
    a = ap.parse_args(argv)

    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    from local_best_response import (build_lbr_venv, load_checkpoint, preflight,
                                     PolicyOps, resolve_matchups)

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, state = resolve_matchups(data, "all")[0]

    venv = build_lbr_venv(state, a.n_envs)
    try:
        model, _ = load_checkpoint(a.ckpt, venv, a.device)
        preflight(venv, model)
        ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)
        rng = np.random.RandomState(a.seed)
        n = venv.num_envs

        keys = None
        COLS, AE, AA, D, EPSTEP = [], [], [], [], []
        ep_step = np.zeros(n, np.int64)
        obs = venv.reset()
        for t in range(a.steps):
            ae = ops.sample_ego(obs, rng)
            aa = ops.sample_adv(obs, rng)
            obs, r_l, r_r, d, infos = venv.step(ops.joint(ae, aa))
            d = np.asarray(d, bool)

            if keys is None:
                keys = [k for k in INFO_KEYS if k in infos[0]]
                missing = [k for k in INFO_KEYS if k not in infos[0]]
                if missing:
                    print(f"   WARNING: info lacks {missing}", flush=True)
                if not keys:
                    raise SystemExit("info dict carries none of INFO_KEYS -- the "
                                     "wrapper stack is not forwarding emulator "
                                     "variables, so there is nothing to compare")
            COLS.append(np.array([[float(inf[k]) for k in keys] for inf in infos],
                                 np.float32))
            AE.append(ae.copy()); AA.append(aa.copy())
            D.append(d); EPSTEP.append(ep_step.copy())
            ep_step += 1
            ep_step[d] = 0
            if (t + 1) % 500 == 0:
                print(f"   {(t+1)*n:,} samples, {int(np.asarray(D).sum())} episodes",
                      flush=True)
    finally:
        venv.close()

    X = np.concatenate(COLS)                      # (N, len(keys))
    ae = np.concatenate(AE); aa = np.concatenate(AA)
    dn = np.concatenate(D); es = np.concatenate(EPSTEP)
    n_ep = int(dn.sum())
    if X.shape[0] < MIN_SAMPLES:
        raise SystemExit(f"only {X.shape[0]:,} samples (<{MIN_SAMPLES:,}): raise --steps")
    if n_ep < MIN_EPISODES:
        raise SystemExit(f"only {n_ep} completed episodes (<{MIN_EPISODES}): "
                         f"ep_len would be unreliable, raise --steps")

    ep_len = float(dn.size / max(1, n_ep))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, X=X, keys=np.array(keys), a_ego=ae, a_adv=aa,
                        done=dn, ep_step=es, ep_len=ep_len,
                        ckpt=os.path.basename(a.ckpt), seed=a.seed)
    print(f"\n   {X.shape[0]:,} samples, {n_ep} episodes, ep_len {ep_len:.0f}")
    print(f"   wrote {a.out}")


if __name__ == "__main__":
    main()
