# Running FightLadder on della-fisac

This repo was developed on other machines (a workstation + the `neuronic` cluster)
against **gym-retro 0.8 / gym 0.21 / cu11 PyTorch / Python 3.8**. None of that runs
on **della-fisac**, whose GPUs are **NVIDIA RTX PRO 6000 "Blackwell" (sm_120)** and
need **CUDA 12.8+ / PyTorch 2.7+**, and whose OS (RHEL 9) can no longer build the
abandoned gym-retro. This document describes the della-specific environment and the
code adaptations that were made so the code runs here.

## TL;DR

```bash
# one-time: build the env + populate the Street Fighter II game integration
bash setup_della.sh

# every session
source /usr/licensed/anaconda3/2024.2/etc/profile.d/conda.sh
conda activate fightladder_della

# train (SPAR/IPPO path — the launchers most run_*.sh use)
bash run_vega_ent05_spar_160M_ck1M.sh
# or the League/PSRO/FSP path
cd main && python train_ma.py --reset round --enable-combo --null-combo --transform-action ...
```

Both `main/ippo.py` (via `run_minimax_phase0.sh`) and `main/train_ma.py` have been
smoke-tested end-to-end on a Blackwell GPU in this env (real PPO updates, ~53 fps).

## The environment: `fightladder_della` (Python 3.9)

Built by `setup_della.sh`. Key pins (full list in `requirements_della.txt`):

| piece | version | why |
|-------|---------|-----|
| python | 3.9 | matches the stable-retro cp39 wheel |
| torch / vision / audio | 2.8.0+cu128 | Blackwell sm_120 needs CUDA 12.8 |
| **stable-retro** | 1.0.0 | maintained gym-retro fork (gym-retro 0.8 won't build) |
| gym | 0.21.0 | required by the vendored SB3 (classic gym API) |
| gymnasium | 1.1.1 | stable-retro's own base env class |
| numpy | 1.24.4 | codebase + old SB3 predate numpy 2.x |
| stable_baselines3 | 1.7.0 | **vendored, editable** from `./stable_baselines3` |
| sb3-contrib | 1.7.0 | PyPI |

Three install gotchas (handled by `setup_della.sh`):
1. **torch** must come from `--index-url https://download.pytorch.org/whl/cu128`.
2. **gym 0.21.0** has invalid PyPI metadata (`opencv-python (>=3.)`) that pip>=24.1
   refuses; it is installed under a temporary `pip==23.3.2` downgrade.
3. **vendored SB3** is installed `pip install -e ./stable_baselines3 --no-deps`
   (a normal install would try to re-resolve gym 0.21 and fail again).

> Note: `import stable_baselines3` only resolves to the vendored copy when the
> repo root is **not** on `sys.path` ahead of it (the `./stable_baselines3/` setup
> dir shadows it as a namespace package). Real scripts are unaffected — running
> `python main/ippo.py` puts `main/` on `sys.path[0]`, not the repo root. Just
> don't `cd` to the repo root and `python -c "import stable_baselines3"`.

## The Street Fighter II game integration

The code calls `retro.make("StreetFighterIISpecialChampionEdition-Genesis")`
(**no `-v0` suffix**) and reads states from `<retro_pkg>/data/stable/<that name>/`.
stable-retro only ships an incomplete `...-Genesis-v0` folder (no ROM), so
`setup_della.sh` builds the non-suffixed folder from this repo's `data/`:
ROM (`rom.md`) + the repo's **custom** `data.json`/`scenario.json` (the RAM
variables the wrappers read) + all `.state` files + the `stars/`, `curriculum/`,
and `two_player/` subdirs.

`two_player/` (144 states) and `Champion.RyuVsRyu.2Player.align.state` originally
existed **only** inside the old py3.8 `fightladder` env; they have been copied into
`data/sf/` so the repo is now the self-contained source of truth.

This integration lives inside the conda env's site-packages, so **it must be
re-run whenever the env is rebuilt or relocated** (just re-run `setup_della.sh`,
which is safe over an existing integration).

## Code adaptations made

### 1. gym-retro → stable-retro API shims (the real code blocker)
stable-retro returns a **gymnasium** env; the whole wrapper/vec/SB3 stack is
**gym 0.21**. Fixed at the emulator boundary without touching the downstream:

- `main/common/retro_wrappers.py` — added `_RetroGymCompat(gym.Wrapper)` and wrapped
  the raw retro env with it inside `SFWrapper.__init__`. It collapses the gymnasium
  5-tuple `step` → gym 4-tuple, unwraps `(obs, info)` reset → `obs`, and provides
  `.seed()` (gymnasium removed it). This one shim also makes the `env.seed(...)`
  delegation chain used by `train_ma.py` / `ippo.py` resolve.
- `main/common/const.py` — a process-wide monkeypatch (installed on import) that
  (a) defaults `render_mode=None` in `retro.make` so `reset()/step()` don't try to
  open an X window and crash on headless SLURM/GPU nodes, and (b) restores the
  classic `render(mode="rgb_array")` signature (stable-retro's `render()` takes no
  `mode` arg) so the video-logging code paths work.

### 2. Dead import removed
`from anyio import value` (an IDE auto-import that never resolved — `anyio` has no
`value`) was removed from `algorithms.py`, the `common/justin/*` SPAR files, and the
vendored SB3's `clean_new_policies.py`. It was a hard `ImportError` on every entry
point.

### 3. Machine-specific paths / conda in launch scripts (~120 files)
Applied across `*.sh`, `*.slurm`, and the relevant `*.py`:

- `/home/jw4406/codebase/FightLadder` → `/home/jw4406/FightLadder` (the `codebase/`
  segment does not exist on this machine).
- `conda activate fightladder` → `conda activate fightladder_della`
  (`fightladder` is the old py3.8/gym-retro/cu11 env — it cannot drive Blackwell).
  `run_minimax_phase0.sh` and the vendored-SB3 auto-submit script honor a
  `CONDA_ENV` override.
- Broken conda-init hooks (`~/anaconda3/...`, `/usr/local/anaconda3/2024.02/...`,
  `module load anaconda3/2024.02`) → `/usr/licensed/anaconda3/2024.2/...`.
- `main/*.sh` hardcoded `PY=/home/jw4406/anaconda3/envs/fightladder/bin/python`
  → `.../fightladder_della/bin/python`.
- Neuronic filesystem `/n/fs/magics/` → della scratch `/scratch/gpfs/FISAC/jw4406/`
  (in della-targeted scripts; `*_NEURONIC.*` files were left alone).

Review everything with `git diff`. `run_minimax_phase0.sh` derives the repo root
from `BASH_SOURCE`, so it is otherwise machine-agnostic.

## Verification

```bash
source /usr/licensed/anaconda3/2024.2/etc/profile.d/conda.sh && conda activate fightladder_della
cd main
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# -> 2.8.0+cu128 NVIDIA RTX PRO 6000 Blackwell Server Edition (12, 0)

# tiny end-to-end SPAR/IPPO training run (a few PPO updates, then exits):
cd /home/jw4406/FightLadder
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 ALLOW_REUSE=True FOREGROUND=1 \
  PLAYER=Ryu OPPONENTS=Guile ENVS_PER_MATCHUP=2 ENV_BATCH_SIZE=4 NUM_PERTURBS=1 \
  NUM_ENV_STEPS=64 TRAINING_BATCH_SIZE=64 TOTAL_TIMESTEPS=3000 MINIMAX_Q=False \
  DECISION_TIMING=off RUN_SUFFIX=smoketest \
  bash run_minimax_phase0.sh vtoff
```

## Notes / caveats

- **GPUs**: della-fisac has **2× Blackwell (96 GB each)**. `run_minimax_phase0.sh`'s
  old "runs can't share a 24 GB card, run sequentially" comments are obsolete here;
  you can pin arms to `CUDA_VISIBLE_DEVICES=0` / `=1` and run them concurrently.
- **wandb**: `main/ippo.py` contains a committed W&B key; set `WANDB_MODE=disabled`
  (or your own `WANDB_API_KEY`) as appropriate. Consider rotating that key.
- **SLURM**: the `*.slurm` files have no `#SBATCH --partition`; set the della GPU
  partition when submitting. `main_training_vtrace_armA_*.slurm` has an
  `--constraint=gpu80|gpu40` (A100 feature) that will never match Blackwell — remove
  or repoint it before submitting that one.
- The old `fightladder` (py3.8) and `fl_clone` (py3.9) envs are left untouched.
