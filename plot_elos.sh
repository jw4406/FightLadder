#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python "${SCRIPT_DIR}/main/plot_elos.py" \
    --elo_data_dir "${SCRIPT_DIR}/main/elo_data" \
    --output_dir "${SCRIPT_DIR}/main/elo_plots" \
    "$@"
