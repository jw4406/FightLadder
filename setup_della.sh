#!/bin/bash
# =============================================================================
# FightLadder environment setup for della-fisac (Princeton HPC).
#
# Builds the conda env `fightladder_della` and populates the stable-retro
# Street Fighter II game integration, so both entry points (main/ippo.py and
# main/train_ma.py) run on this machine's NVIDIA RTX PRO 6000 "Blackwell"
# GPUs (compute capability sm_120).
#
# Why a fresh stack (vs the original environment.yml):
#   * gym-retro 0.8 is abandoned and does not build on RHEL 9 / py3.9+.
#     -> replaced by stable-retro 1.0.0 (a maintained fork, prebuilt cp39 wheel).
#   * The cu11 torch in environment.yml cannot drive Blackwell (needs CUDA 12.8+).
#     -> torch 2.8.0+cu128.
#   * gym 0.21 (required by the vendored stable_baselines3) has invalid PyPI
#     metadata that pip>=24.1 rejects -> installed via a temporary pip downgrade.
#
# Idempotent-ish: safe to re-run (conda create will error if the env exists;
# delete it first with `conda env remove -n fightladder_della` to rebuild).
#
# Usage:   bash setup_della.sh
#          ENV_NAME=myenv bash setup_della.sh      # custom env name
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-fightladder_della}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${CONDA_SH:-/usr/licensed/anaconda3/2024.2/etc/profile.d/conda.sh}"

echo "== FightLadder della setup =="
echo "   env      : ${ENV_NAME}"
echo "   repo     : ${REPO_DIR}"
echo "   conda.sh : ${CONDA_SH}"

# shellcheck disable=SC1090
source "${CONDA_SH}"

# ---------------------------------------------------------------------------
# 1) Create the env. python 3.9 matches the stable-retro cp39 manylinux wheel.
# ---------------------------------------------------------------------------
conda create -y -n "${ENV_NAME}" python=3.9 pip
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 2) PyTorch with CUDA 12.8 (Blackwell / sm_120 support).
# ---------------------------------------------------------------------------
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# 3) Core deps (pinned to a known-good set). NOTE: sb3-contrib pulls PyPI
#    stable-baselines3 here; it is replaced by the vendored copy in step 5.
#    numpy is force-pinned to 1.24.4 (torch pulls 2.x; the codebase needs 1.24).
# ---------------------------------------------------------------------------
pip install \
    stable-retro==1.0.0 gymnasium==1.1.1 numpy==1.24.4 opencv-python==4.7.0.72 \
    sb3-contrib==1.7.0 cloudpickle==2.2.1 pandas==2.0.1 scipy==1.10.1 \
    scikit-learn==1.3.2 matplotlib==3.7.1 tensorboard==2.12.1 wandb==0.26.1 \
    optuna==3.5.0 tqdm==4.66.1 pyglet==1.5.27 av==15.1.0 ecos==2.0.14 \
    natsort==8.4.0 psutil==5.9.8 gpustat==1.1.1

# ---------------------------------------------------------------------------
# 4) gym 0.21.0 — invalid metadata is rejected by pip>=24.1, so temporarily
#    downgrade pip and use setuptools<66 to build it, then restore modern pip.
# ---------------------------------------------------------------------------
pip install "setuptools==65.5.0" "wheel==0.38.4"
pip install "pip==23.3.2"
pip install "gym==0.21.0" --no-build-isolation
python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 5) Vendored (modified) stable_baselines3 1.7.0 from the repo, editable.
#    --no-deps keeps gym 0.21 (a normal install would try to re-resolve it).
# ---------------------------------------------------------------------------
pip uninstall -y stable-baselines3 || true
pip install -e "${REPO_DIR}/stable_baselines3" --no-deps

# ---------------------------------------------------------------------------
# 6) Populate the stable-retro SF2 game integration.
#    The code calls retro.make('StreetFighterIISpecialChampionEdition-Genesis')
#    (NO -v0 suffix) and reads states from <retro_pkg>/data/stable/<that name>.
#    stable-retro only ships an incomplete '...-Genesis-v0' folder, so build the
#    non-suffixed folder from the repo's data/ (ROM + custom RAM json + states).
# ---------------------------------------------------------------------------
DEST="$(python -c "import retro,os;print(os.path.join(retro.data.path(),'stable','StreetFighterIISpecialChampionEdition-Genesis'))")"
SRC="${REPO_DIR}/data"
GAMESRC="${SRC}/stable/StreetFighterIISpecialChampionEdition-Genesis"
echo "   integ -> ${DEST}"
mkdir -p "${DEST}"
# ROM + integration definition (repo's CUSTOM data.json/scenario.json/metadata.json)
cp -f "${GAMESRC}"/rom.md "${GAMESRC}"/rom.sha "${GAMESRC}"/data.json \
      "${GAMESRC}"/scenario.json "${GAMESRC}"/metadata.json "${DEST}/"
# flat states (union of both repo dirs)
cp -f "${GAMESRC}"/*.state "${DEST}/" 2>/dev/null || true
cp -f "${SRC}"/sf/*.state  "${DEST}/" 2>/dev/null || true
# subdirs: stars/, curriculum/, two_player/ (states referenced as 'stars/NAME' etc.)
cp -rf "${SRC}"/sf/stars      "${DEST}/stars"
cp -rf "${SRC}"/sf/curriculum "${DEST}/curriculum"
[ -d "${SRC}/sf/two_player" ] && cp -rf "${SRC}"/sf/two_player "${DEST}/two_player"

# ROM integrity check against rom.sha
if command -v sha1sum >/dev/null 2>&1; then
    want="$(cat "${DEST}/rom.sha")"
    got="$(sha1sum "${DEST}/rom.md" | cut -d' ' -f1)"
    [ "${want}" = "${got}" ] && echo "   ROM sha1 OK (${got})" || echo "   WARNING: ROM sha1 mismatch (want ${want}, got ${got})"
fi

# ---------------------------------------------------------------------------
# 7) Smoke check.
# ---------------------------------------------------------------------------
( cd "${REPO_DIR}/main" && python -c "
import warnings; warnings.simplefilter('ignore')
import torch, gym, retro
from common.const import sf_game
import stable_baselines3 as s
assert torch.cuda.is_available(), 'CUDA not available'
e = retro.make(game=sf_game, state='Champion.Level1.RyuVsGuile', players=2,
               use_restricted_actions=retro.Actions.FILTERED, obs_type=retro.Observations.IMAGE)
o = e.reset(); e.step(e.action_space.sample()); e.close()
print('SMOKE OK  | torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0),
      '| gym', gym.__version__, '| retro', retro.__version__)
print('SB3 (vendored):', s.__file__)
" )

echo ""
echo "== Done. =="
echo "   conda activate ${ENV_NAME}"
echo "   Run training from the repo, e.g.:  bash run_vega_ent05_spar_160M_ck1M.sh"
echo "   (See SETUP_DELLA.md for details.)"
