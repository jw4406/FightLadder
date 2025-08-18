#!/bin/bash

# Check if already running in screen, if not, start screen to make sure the Python scripts don't die if the SSH dies.
if [ -z "$STY" ]; then
    exec screen -S pipeline "$0" "$@"
fi


# --- Configuration ---
NUM_WORKERS=12 # Set this to the number of parallel BR trainings you want to run
USE_MIRROR=true
OUTFILE="somefile.out"
ENV_BATCH_SIZE=32

echo "--- Pipeline Orchestrator ---"

# Clean up from previous runs
echo "Cleaning up old tasks and STOP file..."
rm -f ./tasks/todo/*.task
rm -f ./tasks/processing/*.task
rm -f ./tasks/STOP

# 1. Start the main trainer in the background
echo "Starting the main trainer process in the background..."
python ippo.py --player Guile --use_mirror ${USE_MIRROR} --env_batch_size ${ENV_BATCH_SIZE} > ${OUTFILE} &
MAIN_TRAINER_PID=$!
echo "Main trainer started with PID: $MAIN_TRAINER_PID"

# 2. Start the worker processes in the background
echo "Starting $NUM_WORKERS best-response worker processes..."
for i in $(seq 1 $NUM_WORKERS)
do
  if $USE_MIRROR; then
    python br_worker.py --eval_prot --use_mirror &
    python br_worker.py --use_mirror &
  else
    python br_worker.py &
  fi
done

# 3. Wait for the main trainer to finish
echo "Orchestrator is now waiting for the main trainer (PID: $MAIN_TRAINER_PID) to complete."
echo "Workers will process tasks in the meantime."
wait $MAIN_TRAINER_PID

# 4. Signal the workers to stop
echo "Main trainer finished. Creating STOP file to signal workers to shut down gracefully."
touch ./tasks/STOP

echo "Pipeline finished. You may need to wait a few moments for active workers to complete their current tasks."
