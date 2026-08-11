"""Which RAM bytes are worth feeding the network?

Full Genesis RAM is 65,536 bytes, but most of it never changes. A full-width
observation is ~33.5M params in the first Linear PER EXTRACTOR, and the actor
and critic do not share one (share_features_extractor=False), so ~67M total --
nearly all of it reading constants. A mask cuts that to a few million.

TWO MODES, and they answer different questions:

  vary    (default) bytes that change AT ALL during normal play. Inclusive and
          assumption-free: anything dynamic is kept, including timers, RNG and
          frame counters. Safest default -- it cannot discard something the
          value function needed.

  action  bytes that differ ACROSS the 22 ego actions at the SAME state, found
          by snapshot/branch/restore. This is the targeted mask: it keeps
          exactly the bytes that carry the action-conditional information the
          pixels were measured to destroy. Much smaller, but it CAN discard
          bytes that matter for state value while not varying with the action --
          so it is opt-in, not the default.

Both write a .npy of int64 indices for --ram_mask. No policy or checkpoint is
needed: actions are random, which is the right sampling distribution for "what
can move at all".
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SNAP = "mask_root"
DEFAULT_STATE = "two_player/Ryu_left/Champion.Level1.RyuVsSagat.2Player.state"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("vary", "action", "both"), default="vary")
    ap.add_argument("--state", type=str, default=DEFAULT_STATE)
    ap.add_argument("--steps", type=int, default=3000,
                    help="vec-steps of random play for `vary`")
    ap.add_argument("--n_expansions", type=int, default=25,
                    help="branch points for `action`")
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--out", type=str, default="ram_mask.npy")
    a = ap.parse_args(argv)

    import numpy as np
    from local_best_response import build_lbr_venv, REPO_ROOT

    venv = build_lbr_venv(a.state, a.n_envs)
    try:
        n = venv.num_envs
        na = 22
        rng = np.random.RandomState(0)
        venv.reset()

        def ram_now():
            return np.stack([np.asarray(r) for r in venv.env_method("lbr_ram")])

        seen_min = None
        seen_max = None
        act_diff = None

        if a.mode in ("vary", "both"):
            for t in range(a.steps):
                venv.step(np.stack([rng.randint(0, na, size=n),
                                    rng.randint(0, na, size=n)], axis=-1))
                r = ram_now()
                seen_min = r.min(0) if seen_min is None else np.minimum(seen_min, r.min(0))
                seen_max = r.max(0) if seen_max is None else np.maximum(seen_max, r.max(0))
                if (t + 1) % 500 == 0:
                    live = int((seen_max > seen_min).sum())
                    print(f"   vary {t+1}/{a.steps}  live bytes {live:,}", flush=True)

        if a.mode in ("action", "both"):
            act_diff = np.zeros(65536, bool)
            for e in range(a.n_expansions):
                for _ in range(a.stride):
                    venv.step(np.stack([rng.randint(0, na, size=n),
                                        rng.randint(0, na, size=n)], axis=-1))
                venv.env_method("lbr_snapshot", SNAP)
                per_action = []
                for i in range(na):
                    venv.env_method("lbr_restore", SNAP)
                    venv.step(np.stack([np.full(n, i), np.zeros(n, dtype=int)], axis=-1))
                    per_action.append(ram_now())
                venv.env_method("lbr_restore", SNAP)
                venv.env_method("lbr_drop", SNAP)
                st = np.stack(per_action)                      # (na, n, 65536)
                # A byte counts if it differs across ACTIONS within an env.
                act_diff |= (st.max(0) != st.min(0)).any(0)
                print(f"   action {e+1}/{a.n_expansions}  "
                      f"action-varying bytes {int(act_diff.sum()):,}", flush=True)
    finally:
        venv.close()

    n_full = 65536
    print("\n" + "=" * 70)
    if seen_min is not None:
        vary = seen_max > seen_min
        print(f"  vary   : {int(vary.sum()):,} / {n_full:,} bytes "
              f"({vary.mean():.2%})")
    if act_diff is not None:
        print(f"  action : {int(act_diff.sum()):,} / {n_full:,} bytes "
              f"({act_diff.mean():.2%})")
    if a.mode == "vary":
        mask = np.flatnonzero(vary)
    elif a.mode == "action":
        mask = np.flatnonzero(act_diff)
    else:
        mask = np.flatnonzero(vary | act_diff)
        print(f"  union  : {mask.size:,} bytes")
    if mask.size == 0:
        raise SystemExit("empty mask -- refusing to write. Raise --steps.")
    # REPO_ROOT is already .../FightLadder/main; joining "main" again gave
    # .../main/main and lost the whole scan at the final line.
    out = a.out if os.path.isabs(a.out) else os.path.join(REPO_ROOT, a.out)
    np.save(out, mask.astype(np.int64))
    first = 2 * mask.size * 512
    print(f"\n  wrote {out}  ({mask.size:,} indices)")
    print(f"  first-layer params for two extractors: {first:,} "
          f"(full RAM would be {2*n_full*512:,})")
    return mask


if __name__ == "__main__":
    main()
