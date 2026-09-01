#!/bin/bash
# Watcher: hardlink cont2 checkpoints into the shared dedicated-BR queue at the
# SAME spacing as the earlier ent05 sweep (every 19.2M = 10 checkpoints of 1.92M).
# Hardlink (not move/copy): the BR orchestrator's claim_task MOVES the queued file
# to processing/, so we feed it a hardlink and leave the run-dir original intact.
# Also enqueues each run's FINAL checkpoint (off-stride) once its service exits.
set -u
REPO=/home/jw4406/codebase/FightLadder
QUEUE="$REPO/main/br_todo_cont2/todo"
STOP="$REPO/main/br_todo_cont2/STOP"
STRIDE=19200000          # 19.2M step spacing
POLL=600                 # seconds

# arm: "<run_todo_dir>|<systemd_unit>"
ARMS=(
  "$REPO/main/minimax_phase0_vtoff_image_rs1.0_VegaBlanka_dtj_ent05_vtoff_cont2/trained_models/tasks/todo|spar_ent05_cont2.service"
  "$REPO/main/minimax_phase0_vtoff_image_ippo_rs1.0_VegaBlanka_dtj_ent05_cont2/trained_models/tasks/todo|ippo_ent05_cont2.service"
)

link_if_new() {  # $1 = source .task path
  local src="$1" base; base=$(basename "$src")
  local dst="$QUEUE/$base"
  # skip if already queued, in-flight, or done
  [ -e "$dst" ] && return 0
  [ -e "$REPO/main/br_todo_cont2/processing/$base" ] && return 0
  [ -e "$REPO/main/br_todo_cont2/done/$base" ] && return 0
  ln "$src" "$dst" 2>/dev/null && echo "$(date '+%F %T') queued $base"
}

echo "$(date '+%F %T') watcher up; stride=${STRIDE} queue=$QUEUE"
while :; do
  [ -e "$STOP" ] && { echo "$(date '+%F %T') STOP seen, exiting"; break; }
  for arm in "${ARMS[@]}"; do
    todo="${arm%%|*}"; unit="${arm##*|}"
    [ -d "$todo" ] || continue
    # stride-matching checkpoints
    for f in "$todo"/*_steps.task; do
      [ -e "$f" ] || continue
      step=$(basename "$f" | sed -E 's/.*_([0-9]+)_steps\.task$/\1/')
      [[ "$step" =~ ^[0-9]+$ ]] || continue
      if (( step % STRIDE == 0 )); then link_if_new "$f"; fi
    done
    # once the run has finished, also enqueue its FINAL (highest-step) checkpoint
    if ! systemctl --user is-active --quiet "$unit"; then
      final=$(ls -1 "$todo"/*_steps.task 2>/dev/null \
        | sed -E 's/.*_([0-9]+)_steps\.task$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
      [ -n "$final" ] && link_if_new "$final"
    fi
  done
  sleep "$POLL"
done
