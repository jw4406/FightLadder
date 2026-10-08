#!/usr/bin/env python3
"""Resolve a frozen main checkpoint for a (algo, depth) pair, using the same
closest()+TOL snap the adaptive/crossplay stagers use. One file per participant;
for the league family it carries BOTH seats (policy=ego-left, policy_other=adv-right),
for the spar family it carries both via ctrl/dstb heads.

Algos (duel.py model_type in parens):
  spar (spar) | ippo (ippo) | 2tsA (2timescale, advLR3e4) | 2tsC (2timescale, advLR3e4+crit6e4)
  psro (psro, left main member m_00) | league (league, left main member m_00)

Usage:
  resolve_ckpt.py <algo> <depthM>        -> prints "<model_type> <absM> <realpath>" (exit 2 if none within TOL)
  resolve_ckpt.py <algo> --list          -> prints the available 20M-grid depths for <algo>
"""
import sys, os, glob, re

M20 = list(range(20_000_000, 400_000_001, 20_000_000))

def segs_minimax(run, suffix):
    return f"/n/fs/magics/{run}/FightLadder/main/minimax_phase0_vtoff_image_{suffix}/trained_models/tasks"

ALGOS = {
 "spar": dict(mt="spar", tol=2_500_000, rx=r"_(\d+)_steps", pat="spar_Gu_VeCh_*_steps.task", segs=[
     (segs_minimax(3852570,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M")+"/todo_continue", 0),
     (segs_minimax(3950952,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_cont86")+"/todo", 86_000_688),
     (segs_minimax(4034178,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_cont192")+"/todo", 192_001_536)]),
 "ippo": dict(mt="ippo", tol=2_500_000, rx=r"_(\d+)_steps", pat="ippo_Gu_VeCh_*_steps.task", segs=[
     (segs_minimax(3849619,"ippo_rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M")+"/todo_continue", 0),
     (segs_minimax(3950953,"ippo_rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_cont108")+"/todo", 108_000_864),
     (segs_minimax(4034179,"ippo_rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_cont224")+"/todo", 224_001_792)]),
 "2tsA": dict(mt="2timescale", tol=2_500_000, rx=r"_(\d+)_steps", pat="spar_Gu_VeCh_*_steps.task", segs=[
     (segs_minimax(4005437,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_advLR3e4")+"/todo", 0),
     (segs_minimax(4049346,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_advLR3e4_cont69")+"/todo", 69_000_552),
     (segs_minimax(4059181,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_advLR3e4_cont162")+"/todo", 162_001_296)]),
 "2tsC": dict(mt="2timescale", tol=2_500_000, rx=r"_(\d+)_steps", pat="spar_Gu_VeCh_*_steps.task", segs=[
     (segs_minimax(4004731,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_advLR3e4_crit6e4")+"/todo", 0),
     (segs_minimax(4049347,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_advLR3e4_crit6e4_cont81")+"/todo", 81_000_648),
     (segs_minimax(4059182,"rs1.0_GuileVegaChunLi_dtj_ent05_160M_ck1M_advLR3e4_crit6e4_cont179")+"/todo", 179_001_432)]),
 "2ts": dict(mt="2timescale", tol=2_500_000, rx=r"_(\d+)M", pat="2timescale_ego_*M.task", unit=1_000_000, segs=[
     ("/n/fs/magics/2ts_ego_blocks", 0)]),
 "psro": dict(mt="psro", tol=2_500_000, rx=r"step_(\d+)_0", pat="PSRO0_left_m_00_left_vs_all_historical_step_*_0.pt", segs=[
     ("/n/fs/magics/3806994/FightLadder/main/trained_models/tasks/todo", 0)]),
 "league": dict(mt="league", tol=2_500_000, rx=r"step_(\d+)_0", pat="MA0_left_m_00_left_vs_all_historical_step_*_0.task", segs=[
     ("/n/fs/magics/3861916/FightLadder/main/trained_models/tasks/todo/left_exploit_snapshots", 0)]),
}

def candidates(spec):
    c = []
    unit = spec.get("unit", 1)   # filenames encoding depth-in-M use unit=1_000_000
    for d, off in spec["segs"]:
        for f in glob.glob(os.path.join(d, spec["pat"])):
            m = re.search(spec["rx"], os.path.basename(f))
            if m:
                c.append((int(m.group(1)) * unit + off, os.path.realpath(f)))
    return c

def resolve(algo, M):
    spec = ALGOS[algo]; T = M * 1_000_000
    c = candidates(spec)
    if not c: return None
    best = min(c, key=lambda x: abs(x[0] - T))
    if abs(best[0] - T) > spec["tol"]: return None
    return spec["mt"], best[0], best[1]

def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ALGOS:
        sys.exit(f"usage: resolve_ckpt.py <{'|'.join(ALGOS)}> <depthM | --list>")
    algo = sys.argv[1]
    if sys.argv[2] == "--list":
        got = [M // 1_000_000 for M in M20 if resolve(algo, M // 1_000_000)]
        print(" ".join(str(m) for m in got)); return
    M = int(sys.argv[2])
    r = resolve(algo, M)
    if r is None:
        sys.exit(2)
    print(f"{r[0]} {r[1]//1_000_000} {r[2]}")

if __name__ == "__main__":
    main()
