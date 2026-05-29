#!/bin/bash

CMD=(python league_training_orchestrator.py   
	--league_training_sh_template /home/jw4406/FightLadder/slurm_launch_files/league_template.slurm   
	--player Ryu Ken Guile   
	--opponent-list ChunLi Zangief Sagat EHonda 
	--total_steps 1000000
	#--dry-run
)
"${CMD[@]}"
