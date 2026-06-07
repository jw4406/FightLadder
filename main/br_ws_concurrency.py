"""
Workstation-mode concurrency helper.

When the BR orchestrators run on a machine without SLURM, ``submit_sbatch``
in ``br_slurm_common`` falls back to ``Popen(["bash", <script>])`` for every
ready task. On a multi-GPU cluster SLURM throttles this naturally; on a
single-GPU workstation it would launch every queued task at once and OOM
the GPU.

This module provides a single helper, ``count_active_local_jobs``, used by
the orchestrator watchdog loops to gate "claim next todo" on a configurable
concurrency cap (``--max_local_concurrent``). On a real SLURM cluster the
gate's first conjunct (``not have_sbatch()``) short-circuits, so this code
is dead-code on the cluster path.

Kept in its own module (rather than added to ``br_slurm_common``) so that
the cluster code path's import surface is unchanged.
"""
from __future__ import annotations

import os
from typing import Iterable

from br_slurm_common import _local_pid_alive, read_registry


def _iter_local_pids(processing_dir: str) -> Iterable[int]:
    """Yield PIDs (ints) from every ``local-<pid>`` job id in every registry
    file under *processing_dir*. Silently skips missing/malformed entries."""
    if not os.path.isdir(processing_dir):
        return
    for entry in os.listdir(processing_dir):
        folder = os.path.join(processing_dir, entry)
        if not os.path.isdir(folder):
            continue
        registry = read_registry(folder)
        if not registry:
            continue
        for jid in registry.get("job_ids", []) or []:
            if not isinstance(jid, str) or not jid.startswith("local-"):
                continue
            try:
                yield int(jid.split("-", 1)[1])
            except (ValueError, IndexError):
                continue


def count_active_local_jobs(processing_dir: str) -> int:
    """
    Return the number of local-bash jobs currently still alive across all
    per-task registries under *processing_dir*.

    Only counts ids of the form ``local-<pid>`` (the marker
    ``submit_sbatch`` writes in its local fallback). Real SLURM job ids
    (numeric strings) are ignored, so calling this on a cluster machine
    just returns 0.
    """
    return sum(1 for pid in _iter_local_pids(processing_dir) if _local_pid_alive(pid))
