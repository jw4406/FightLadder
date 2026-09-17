#!/usr/bin/env python
"""Re-run the promising-but-pruned configs recorded in the adaptive controller's
revisit manifest, FROM SCRATCH with their original seed (deterministic via the
per-rep seed 1000+rep), to fill in the wr curves that halving cut off.

Run with the SAME ADAPT_* env vars as adaptive_exploiter.py (so SEATS/tasks/states
resolve identically). Reuses the controller's Cfg / reeval / kill machinery."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adaptive_exploiter as A


def main():
    manifest = A.REVISIT_MANIFEST
    if not os.path.exists(manifest):
        A.log(f"no revisit manifest at {manifest}; nothing to do"); return
    recs = [json.loads(l) for l in open(manifest) if l.strip()]
    if not recs:
        A.log("revisit manifest empty"); return
    A.log(f"REVISIT: re-running {len(recs)} promising-but-pruned config(s) from scratch")
    cfgs = []
    for rec in recs:
        c = A.Cfg(rec["seat"], rec["lr"], rec["seed"], rec["net"])
        cfgs.append((c, rec)); c.launch(); time.sleep(20)

    while any(c.alive for c, _ in cfgs):
        time.sleep(A.POLL)
        for c, rec in cfgs:
            if not c.alive:
                continue
            z, st = c.newest_zip()
            if z and st > c.last_step:
                wr, n = A.reeval(c, z)
                if wr is not None:
                    c.traj.append((st, wr, n)); c.last_step = st; A.prune_old_zips(c)
                    lb = A.wilson_lb(wr, n)
                    A.log(f"[revisit orig={rec['label']}] {c.label} step={st/1e6:.1f}M wr={wr:.2f} lb={lb:.2f} n={n}")
                    c.kill_ticks = c.kill_ticks + 1 if (wr >= A.KILL_WR and lb > 0.5) else 0
                    if c.kill_ticks >= A.SUSTAIN:
                        A.log(f"*** revisit {c.label} (orig {rec['label']}) CONFIRMED KILL wr={wr:.2f} @ {st/1e6:.1f}M ***")
                        c.kill("revisit confirmed kill")
            if c.proc.poll() is not None and c.alive:
                c.alive = False
                A.log(f"[revisit] {c.label} proc exited @ {c.last_step/1e6:.1f}M "
                      f"(best_wr={max((w for _,w,_ in c.traj), default=0):.2f})")

    A.log("=== REVISIT DONE ===")
    for c, rec in cfgs:
        A.log(f"  orig={rec['label']} lr={c.lr} seed={c.seed} net={c.net}: "
              f"best_wr={max((w for _,w,_ in c.traj), default=0):.2f} "
              f"traj={[(round(s/1e6,1), round(w,2)) for s,w,_ in c.traj]}")


if __name__ == "__main__":
    main()
