# SPAR evaluation, LR-ratio sweep & BR packing

Concise reference for three tools built on top of the SPAR training + best-response (BR) stack.
All are **opt-in**: defaults reproduce prior behavior.

---

## 1. `main/visualize_spar_policy_behavior.py`

Two auto-selected modes.

### Visualize mode (default)
Records a gameplay video of **one** model across **all** its matchups (tiled into one MP4).
```bash
python main/visualize_spar_policy_behavior.py --model_source <file|dir> \
    --episodes 50 --output_video main/videos/behavior.mp4
```
Key args: `--model_source` (checkpoint file, or a dir to sample one from), `--episodes`,
`--state` (single-matchup override; default = all states in the checkpoint's `state_list`),
`--fps`, `--device`.

### Duel mode
Triggered when `--ego_model_file`/`--adv_model_file` are given. Loads **two** models (possibly
different families: `league|spar|ippo|2timescale`), routes each to the `(ego_char, adv_char)`
head, plays N rounds, **counts ego wins**, and records the video.
```bash
python main/visualize_spar_policy_behavior.py \
    --ego_model_type spar  --ego_model_file /abs/ego.zip  --ego_char Guile \
    --adv_model_type league --adv_model_file /abs/adv.task --adv_char Ryu \
    --num_rounds 10 --deterministic True --output_video main/videos/guile_vs_ryu.mp4
```
Prints per-round `ego_won`/HP and a final `ego_win_rate`. Default video path:
`main/videos/duel_<egoType>_<egoChar>_vs_<advType>_<advChar>.mp4`.

**Notes:** run in the `fightladder` conda env (needs `retro`); run so `main/` is importable.
**Caveat:** neither mode supports `use_mirror=True` (FiLM/side-conditioned) checkpoints — no
`side_flag` is threaded, so those will fault. Standard checkpoints are fine.

---

## 2. LR-ratio sweep chain

Sweeps the two-timescale learning-rate **ratios**, trains + best-responds each config, and ranks
configs by exploitability. One command fans out three tiers of Slurm jobs.

```
launch_lr_ratio_sweep.sh          # set dials here
  └─ lr_ratio_sweep.py            # driver: build grid, render, sbatch (login-safe; only submits)
       per config <tag=mdM_mvV>:
         ├─ main_training_*.slurm         (from cds_style_template.slurm)  -> GPU training job
         └─ br_orchestrator_<tag>.slurm   (from br_orchestrator_job_template.slurm) -> CPU watchdog job
                └─ br_slurm_orchestrator.py sbatches exploiter GPU jobs as checkpoints stream in
```

**Config grid:** `c_lr` fixed; `d_lr = c_lr·m_d`, `v_lr = d_lr·m_v`, enforcing `ego < adv < critic`
and skipping `v_lr > MAX_V_LR`. `tag = md<m_d>_mv<m_v>`.

**Phases** (`PHASE` in the launcher):
| phase | does |
|---|---|
| `train` | render + `sbatch` per-config training jobs |
| `br` | render + `sbatch` per-config CPU orchestrator jobs (watch that config's tasks) |
| `both` | both, with the BR job gated on training via `sbatch --dependency=after:<train_id>` |
| `--discover` (br only) | ignore the grid; launch a BR watchdog per **existing** `lr_sweep/*` tree (skips ones already being BR'd) |

**Isolation:** each config lives under a deterministic tree `$WORKDIR/lr_sweep/<tag>/FightLadder/`
(training's `JOBID` is rewritten to `lr_sweep/<tag>`), so tasks, checkpoints, `br_rewards/` and
`br_models/` all co-locate; `MAIN_TRAINING_DIR=lr_sweep/<tag>` wires training↔BR.

**BR = streaming curve:** training drops a `.task` per checkpoint into `…/tasks/todo/`; the watchdog
(a CPU-only compute-node job — never a login `nohup`) fires exploiter jobs as they arrive, giving an
exploitability curve over training. `STEP_STRIDE` thins which checkpoints; `PERIODIC_EVAL_FREQ` adds
mid-training snapshots.

**Cluster:** `CLUSTER=neuronic|della` switches all env-setup blocks (training, orchestrator,
exploiter templates). DELLA skips the module/conda-init block (conda pre-available).

**Plotting:**
- per-config curves: `python main/aggregate_local_eval_data.py --br_rewards_dir <tree>/main/br_rewards`
- cross-config compare: `python main/plot_lr_sweep.py --workdir $WORKDIR` → ranked CSV (by `final_gap`,
  swappable via `RANK_FUNCS`) + top-K overlay (`--top_k 0` = all).

**Launcher dials** (`launch_lr_ratio_sweep.sh`): `DRY_RUN` (default True — renders + prints, submits
nothing), `PHASE`, `DISCOVER`, `CLUSTER`, `WORKDIR`, `PLAYER`/`OPPONENTS`,
`C_LR`/`D_MULTS`/`V_MULTS`/`MAX_V_LR`, `TRAIN_TIME`/`BR_JOB_TIME`, `STEP_STRIDE`/`PERIODIC_EVAL_FREQ`,
and the packing knobs (§4).

```bash
bash slurm_launch_files/launch_lr_ratio_sweep.sh   # DRY_RUN=True first; inspect generated_lr_sweep/
```

---

## 3. Standalone BR launchers — `launch_br_orchestrators_{NEURONIC,DELLA}.sh`

Run the **dedicated** BR watchdog over an existing training run's tasks, independent of the sweep.
Set `WORKDIR` + `MAIN_TRAINING_DIR` to point `TASK_BASE` at that run's tree, then launch.

```bash
bash slurm_launch_files/launch_br_orchestrators_NEURONIC.sh
```
Key config vars: `LAUNCH_DEDICATED`, `WORKDIR`, `MAIN_TRAINING_DIR`, `STEP_STRIDE`,
`BR_TRAINING_STEPS`, `SLURM_TIME`, `PERIODIC_EVAL_FREQ`, plus the packing knobs (§4).

**Notes:** the watchdog runs via `nohup … &`, so launch it somewhere long-running processes are
permitted (not a login node where prohibited). Continue-mode (`CONTINUE_CMD`) is vestigial/unused and
has no packing support — leave it off.

---

## 4. GPU packing (shared by §2 and §3)

Co-locate multiple BR exploiters on one GPU (small RL jobs badly under-use an L40S/A100). Opt-in;
`EXPLOITERS_PER_JOB=1` = one exploiter per GPU, no cap (prior behavior).

| knob | meaning |
|---|---|
| `EXPLOITERS_PER_JOB` (N) | exploiters co-located per GPU sbatch |
| `GPU_MEM_FRACTION` (f) | per-process VRAM cap via `set_per_process_memory_fraction`; keep **`N·f ≲ 0.85`** (e.g. N=8 → f≈0.10) |
| `PACK_ACROSS_CHECKPOINTS` | `True` packs exploiters from **different** checkpoints onto one GPU — needed to fill the card when a config yields `< N` specs/checkpoint (e.g. 1 replicate = 2 specs) |
| `PACK_FLUSH_TIMEOUT` | max seconds the oldest buffered spec waits before a partial pack (cross-checkpoint only) |

Behavior when packing:
- Host `--cpus-per-task` and `--mem` **auto-scale by N** (`6·N`, `8G·N` from the exploiter template);
  the GPU is shared (`--gres` not scaled). So N is bounded by node CPU/mem, not VRAM.
- Each packed process is memory-capped; a process that OOMs/crashes is **retried SOLO once** (cap
  removed → full GPU) then gives up.
- No MPS — processes time-slice the GPU (fine for rollout-heavy RL). Push N until GPU util ~85–95%.

Tuning: raise N and lower f together; watch `nvidia-smi` (util + VRAM), per-proc step rate, and that
jobs finish within `SLURM_TIME`.
