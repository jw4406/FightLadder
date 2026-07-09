#!/bin/bash

# Resolve paths relative to this script so it works from any cwd and any checkout
# location (both the orchestrator and the template live alongside this script).
SCRIPT_DIR="/u/jw4406/FightLadder/slurm_launch_files"

CMD=(python "${SCRIPT_DIR}/main_training_orchestrator.py"
	--main_training_sh_template "${SCRIPT_DIR}/cds_style_template.slurm"
	--main_training_model_arch_types 2timescale
	--c_lr 1e-5
	--d_lr 2e-5
	--v_lr 2e-5
	--ego_value_head_lr 1e-5
	--player Guile
	--opponent-list Ryu Sagat Vega Blanka ChunLi Zangief Ken Dhalsim EHonda Guile Balrog MBison
	--main_training_steps 150000000
	--time 001:00:00
	--workdir /scratch/gpfs/FISAC/jw4406/
	--checkpoint_interval 50000
	#--dry-run
)

"${CMD[@]}"
