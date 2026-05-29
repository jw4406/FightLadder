#!/bin/bash

CMD=(python main_training_orchestrator.py 
	--main_training_sh_template /home/jw4406/FightLadder/slurm_launch_files/cds_style_template.slurm
	--main_training_model_arch_types 2timescale
	--c_lr 1e-5 
	--d_lr 2e-5 
	--v_lr 2e-5 
	--player Guile
	--opponent-list Ryu Sagat Vega Blanka Chunli Zangief Ken Dhalsim EHonda Guile Balrog MBison
	--main_training_steps 150000000
	#--dry-run
)

"${CMD[@]}"
