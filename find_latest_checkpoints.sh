#!/bin/bash

# Script to find the latest checkpoint (largest TRAINING_STEPS) for each unique BASE_STEPS value
# Usage: ./find_latest_checkpoints.sh [directory]
WORKDIR=/n/fs/magics
TRAINING_DIR=2466337
DIR="${1:-$WORKDIR/$TRAINING_DIR/FightLadder/main/trained_models/br_models}"

if [ ! -d "$DIR" ]; then
    echo "Error: Directory $DIR does not exist" >&2
    exit 1
fi

# Get all unique BASE_STEPS values
base_steps_list=$(ls -1 "$DIR" | sed -E 's/br_to_ppo_Guile_([0-9]+)_steps\.zip_.*/\1/' | sort -u -n)

# For each base model, find the file with the largest training steps
for base_steps in $base_steps_list; do
    # Find all files for this base model and extract training steps
    latest_file=""
    max_training_steps=0
    
    for file in "$DIR"/br_to_ppo_Guile_${base_steps}_steps.zip_*_steps.zip; do
        # Check if file exists (glob might not match)
        [ -f "$file" ] || continue
        
        # Extract training steps from filename
        training_steps=$(echo "$file" | sed -E 's/.*_([0-9]+)_steps\.zip$/\1/')
        
        # Compare and update if this is the largest
        if [ "$training_steps" -gt "$max_training_steps" ]; then
            max_training_steps=$training_steps
            latest_file=$(basename "$file")
        fi
    done
    
    # Echo the latest file for this base model
    if [ -n "$latest_file" ]; then
        echo "$latest_file"
    fi
done

