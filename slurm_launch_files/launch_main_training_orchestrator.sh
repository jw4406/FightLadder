#!/bin/bash

# Resolve paths relative to this script so it works from any cwd and any checkout
# location (both the orchestrator and the template live alongside this script).
SCRIPT_DIR="/home/jw4406/FightLadder/slurm_launch_files"


CMD=(python "${SCRIPT_DIR}/main_training_orchestrator.py"
	--main_training_sh_template "${SCRIPT_DIR}/cds_style_template.slurm"
	--main_training_model_arch_types 2timescale
	--c_lr 1e-6
	--d_lr 2e-6
	--v_lr 2e-6
	--ego_value_head_lr 1e-6
	--player Guile Vega Blanka ChunLi Zangief EHonda
	--opponent-list Ryu Sagat Ken Dhalsim Balrog MBison
	--main_training_steps 200000000
	--time 096:00:00
	--workdir /scratch/gpfs/FISAC/jw4406/
	--checkpoint_interval 50000
	#--dry-run
)

"${CMD[@]}"
