# n=500 Crossplay / Self-play Matrix — Spec Sheet (for porting to della)

Reproduce the n=500 N×N win-rate matrices (and run arbitrary cross-algorithm
crossplay) on another cluster. Written against the neuronic setup launched
2026-10-06; the **della porting checklist** at the bottom lists every path /
assumption that must change.

---

## 1. What it computes

A matrix cell `M[i][j]` = **ego-side win-rate** of participant *i*'s policy (as
the **left/ego** player) vs participant *j*'s policy (as the **right/adv**
player), over `--num_rounds` SF2 rounds, scored by health-differential.

- Rows = ego, cols = adversary. **Diagonal = self-play.**
- `ROW avg` = ego strength (higher = stronger); `COL avg` = adversary weakness
  (higher = weaker adv). Low cell = adversary dominates.
- Characters: **ego=Guile (left)**, **adv ∈ {ChunLi, Vega} (right)** — the
  original spar study ran one full matrix per adversary character.

Correctness is anchored by one driver, `duel.py` (ego locked to left, adv to
right). `crossplay.py` just shells out to it per cell.

---

## 2. Tooling files (all in this repo)

| File | Role |
|---|---|
| `main/duel.py` | canonical 1v1 dueler; loads ego(left)+adv(right), runs rounds, prints `ego_win_rate=` |
| `main/crossplay.py` | builds the N×N matrix by calling `duel.py` per cell; `--jobs K` (parallel cells), `--row I` (compute one ego-row, for sharding), `--out` (write matrix) |
| `slurm_launch_files/resolve_ckpt.py` | `(algo, depthM) → checkpoint path` via nearest-step snap. **Paths are neuronic-specific — must be repointed on della.** |
| `slurm_launch_files/submit_crossplay.sh` | orchestrator: resolve participants → submit a per-row job **array** + a dependent **finalizer** |
| `slurm_launch_files/crossplay_row.slurm` | one array task = one ego-row (duels serial → ~1 duel's RAM) |
| `slurm_launch_files/crossplay_finalize.py` | stitch `row_*.txt` → `matrix.{txt,csv}` |
| `slurm_launch_files/crossplay_matrix.slurm` | single-node variant (no sharding; needs big RAM — see §5) |
| `slurm_launch_files/crossplay_to_csv.py` | convert any `crossplay.py` matrix `.txt` → CSV |

---

## 3. Model-type / loader contract (the key fact for porting)

`duel.py` has TWO loader families (chosen by `--ego_model_type` / `--adv_model_type`):

- **SPAR family** `{spar, ippo, 2timescale}` — one self-contained `.task`; ego =
  **ctrl** head, adv = **dstb** head (per-matchup `head_idx`). One file supplies
  both seats.
- **LEAGUE family** `{league, psro}` — torch-save `{cls_name, kwargs:{agent_dict}}`
  (cls_name `'Historical'`); ego = `policy` (left), adv = `policy_other` (right).
  One file supplies both seats. **`psro` loads through the league path unchanged**
  — the loader validates the `agent_dict` shape and ignores `cls_name`. (This is
  why `MODEL_TYPES = ["league","psro","spar","ippo","2timescale"]` and
  `LEAGUE_FAMILY = {"league","psro"}` were added to `duel.py`.)

So **every checkpoint is two-sided**: used as ego it drives its left policy, as
adv its right policy. A `--participant label:model_type:path` therefore needs
only ONE path.

---

## 4. Checkpoint sources (what to copy to della)

`resolve_ckpt.py` registry — source run dirs, step→abs offset, filename glob,
and on-grid depths (neuronic paths):

| algo | model_type | glob | source dirs (+offset) | depths (M) |
|---|---|---|---|---|
| spar | spar | `spar_Gu_VeCh_*_steps.task` | 3852570/…/todo_continue(+0), 3950952/…_cont86/todo(+86000688), 4034178/…_cont192/todo(+192001536) | 20–300 |
| ippo | ippo | `ippo_Gu_VeCh_*_steps.task` | 3849619/…/todo_continue(+0), 3950953/…_cont108/todo(+108000864), 4034179/…_cont224/todo(+224001792) | 20–340 |
| 2tsA | 2timescale | `spar_Gu_VeCh_*_steps.task` | 4005437/…_advLR3e4/todo(+0), 4049346/…_advLR3e4_cont69/todo(+69000552) | 20–160 |
| 2tsC | 2timescale | `spar_Gu_VeCh_*_steps.task` | 4004731/…_advLR3e4_crit6e4/todo(+0), 4049347/…_cont81/todo(+81000648) | 20–180 |
| psro | psro | `PSRO0_left_m_00_left_vs_all_historical_step_*_0.pt` | 3806994/…/trained_models/tasks/todo | 20–320 |
| league | league | `MA0_left_m_00_left_vs_all_historical_step_*_0.task` | 3861916/…/tasks/todo/left_exploit_snapshots | 20–200 |

Snap tolerance: 2.5M (all). psro/league `m_00` left-main member carries both
seats. **For Vega-adv matrices**, psro/league `m_00` may have no Vega matchup →
those cells return `nan` (harmless).

**The spar/ippo/psro/league 20M-grid checkpoints (ego+adv) are already bundled**
in `/n/fs/magics/offline_ckpts_2026-10-05.zip` (85 files, 15.4 GiB) — that zip IS
the della checkpoint payload for those four. **2tsA/2tsC are NOT in the bundle**;
copy their `*_steps.task` dirs separately if you want the 10× arms on della.

---

## 5. Run config (defines "n=500")

`crossplay.py` invokes `duel.py` per cell with exactly these flags:

```
--num_rounds 500            # n=500
--ego_char Guile --adv_char {ChunLi|Vega}
--ego_side left --device cpu
--seed 0 --deterministic False --transform_action True
--obs_type image --decision_timing joint --dwell_frames 4
--actionable_statuses 512,514,520
# (no --max_skip_frames: duel.py defaults to 90, matching training; passing it errors)
--timeout 3600              # per-duel wall cap (crossplay.py kills + returns nan past this)
```

These must match the training env config. `transform_action=True` and
`decision_timing=joint, dwell=4` are the eval-time values passed on the CLI.

---

## 6. Resource profile & job structure (measured on neuronic)

- **Per duel: ~18 GB RSS at n=500** (~10 GB model+env baseline, +~15 MB/round over
  500 rounds), **~23 min wall on CPU**, effectively single-threaded
  (`OMP_NUM_THREADS=1`). Inference is batch-1 latency-bound → **CPU, not GPU**
  (GPU gives ~no speedup at batch 1 and wastes the allocation).
- **Do NOT pack many duels per node**: K concurrent duels need K×18 GB
  (K=16 ⇒ ~288 GB → OOM). The single-node `crossplay_matrix.slurm` only works at
  low K on a big-RAM node (e.g. mem≈180G, `--jobs 8`).
- **Preferred = row-sharded array** (`submit_crossplay.sh`): one array task per
  ego-row, its N duels run **serial** (`--jobs 1`) in ~28 GB / 4 CPU. N rows run
  in parallel as small, easily-scheduled jobs. Matrix wall-clock ≈ one row =
  **N × 23 min** (e.g. 17×17 ippo ≈ 6.5 h).

---

## 7. How to run (neuronic; adapt per §8)

Self-play depth matrix (array + finalizer, results at `<WS>/matrix.{txt,csv}`):
```bash
cd slurm_launch_files
SELFPLAY=spar ROUNDS=500 EGO_CHAR=Guile ADV_CHAR=ChunLi bash submit_crossplay.sh
# optional: DEPTHS="100,140,180,220,260,300" to subset; JOBS>1 only with more RAM
```

Arbitrary cross-algo (matched or mixed depths):
```bash
PARTS="spar:160 ippo:160 2tsA:160 2tsC:160 psro:160 league:160" \
  ROUNDS=500 EGO_CHAR=Guile ADV_CHAR=ChunLi bash submit_crossplay.sh
```

**Most portable (no resolver): call `crossplay.py` directly with explicit paths** —
this needs only `crossplay.py` + `duel.py` + the env + the checkpoint files:
```bash
python main/crossplay.py \
  --participant spar160:spar:/della/path/spar_160M.task \
  --participant psro160:psro:/della/path/psro_ego_160M.pt \
  --participant league160:league:/della/path/league_ego_160M.task \
  --rounds 500 --ego_char Guile --adv_char ChunLi \
  --device cpu --jobs 1 --timeout 3600 --out /della/path/cross160.txt
```

---

## 8. della porting checklist

1. **Repo + env.** Copy the repo. Recreate the `fightladder` conda env (torch,
   stable-baselines3, **stable-retro + the Street Fighter II ROM/integration
   data** — this is the main hurdle; duels won't run without the retro env).
2. **conda bootstrap.** Replace `source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh`
   with della's (`module load anaconda3/…` then `conda activate fightladder`) in
   `crossplay_row.slurm`, `crossplay_matrix.slurm`, and the `--wrap` in
   `submit_crossplay.sh`.
3. **Paths.** `REPO=/auto/u/jw4406/FightLadder` → della repo path (in the three
   slurm/orchestrator files). Output root `/n/fs/magics` → della scratch
   (e.g. `/scratch/gpfs/$USER/...`) everywhere it appears.
4. **Checkpoints.** Unzip `offline_ckpts_2026-10-05.zip` on della; either
   (a) repoint `resolve_ckpt.py`'s `ALGOS` dirs to the unzipped locations, or
   (b) skip the resolver and pass explicit `--participant` paths (§7, most
   portable — the bundle's filenames already encode algo/seat/depth).
5. **SLURM.** Set della `--partition`/`--account`/QOS; drop neuronic-only bits
   (`--exclude=neu317` lives only in GPU stagers, not these CPU tools). Keep
   `--mem=28G --cpus-per-task=4` per row task; tune `--time` (24h is ample).
6. **Validate** before trusting numbers — reproduce the known spar values
   (ego=Guile, adv=ChunLi): `d180×d180=0.352`, `d180×d300=0.212`,
   `d300×d180=0.546`, `d300×d300=0.404`. A 2×2 spar{180,300} run should match.

---

## 9. Gotchas

- All-`nan` matrix ⇒ per-duel timeout too low (raise `DUEL_TIMEOUT`) **or**
  OOM (reduce concurrency / raise mem) **or** `duel.py` arg error. `run_duel`
  swallows failures as `nan`, so check the row `.out`/`.err` if cells are nan.
- `sbatch --export` splits on commas → pass `DEPTHS` with underscores
  (`100_140_180`) or `export` the var then `--export=ALL`.
- Vega matrices for psro/league may be partially `nan` (no Vega matchup in the
  `m_00`/`m_01` members). Expected, not a bug.
