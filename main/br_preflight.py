"""
Fast, dependency-light helpers for BR worker preflight checks.

These helpers intentionally avoid importing heavy ML/game modules so they can be
tested quickly in unit tests.
"""

from typing import Any, Callable, Dict, List, Optional


def dedupe_preserve_order(values: List[str]) -> List[str]:
    """Remove duplicates while preserving first-seen order."""
    return list(dict.fromkeys(values).keys())


def sanitize_for_filename(value: Optional[str]) -> str:
    """
    Convert arbitrary labels into filesystem-safe filename fragments.

    Keeps alphanumeric characters, `_`, and `-`; replaces everything else with
    `_`.
    """
    if value is None:
        return "unknown"
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def infer_cds_architecture(data: Dict[str, Any], task_file_path: str) -> str:
    """
    Infer checkpoint architecture family for BR loading.

    Returns:
        "ippo" if checkpoint appears to use IPPO CDS policy/value heads;
        "spar" otherwise.
    """
    explicit_arch = data.get("model_arch_type")
    if isinstance(explicit_arch, str):
        normalized = explicit_arch.strip().lower()
        if normalized in ("ippo", "spar"):
            return normalized

    policy_class = data.get("policy_class")
    policy_class_name = getattr(policy_class, "__name__", str(policy_class))
    if "CleanIPPOActorActorCriticPolicy" in policy_class_name or "IPPO" in policy_class_name:
        return "ippo"

    basename = task_file_path.lower().split("/")[-1]
    if basename.startswith("ippo_") or "_ippo_" in basename:
        return "ippo"
    return "spar"


def extract_unique_states_from_checkpoint_data(
    data: Dict[str, Any], task_file_path: str = "<checkpoint>"
) -> List[str]:
    """Return unique state strings from checkpoint/task metadata."""
    if "state_list" not in data:
        raise KeyError(f"Task/checkpoint {task_file_path} does not contain 'state_list'.")
    state_list = data["state_list"]
    if not isinstance(state_list, list) or len(state_list) == 0:
        raise ValueError(f"Task/checkpoint {task_file_path} has empty or invalid 'state_list'.")
    return dedupe_preserve_order(state_list)


def extract_unique_states_from_checkpoint_loader(
    task_file_path: str,
    checkpoint_loader: Callable[[str], Any],
) -> List[str]:
    """
    Read checkpoint metadata via `checkpoint_loader` and return unique states.

    `checkpoint_loader` is expected to return a 3-tuple `(data, params, vars)`,
    matching `stable_baselines3.common.save_util.load_from_zip_file`.
    """
    data, _, _ = checkpoint_loader(task_file_path)
    return extract_unique_states_from_checkpoint_data(data=data, task_file_path=task_file_path)


def build_dedicated_job_specs(
    unique_states: List[str],
    replicates_per_matchup: int,
    run_eval_prot: bool,
    run_eval_adv: bool,
    launch_local_br_eval: bool,
    state_to_matchup: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build dedicated BR job specs as one entry per (state, side, replicate).
    """
    if replicates_per_matchup < 1:
        raise ValueError("replicates_per_matchup must be >= 1 for dedicated jobs.")
    if state_to_matchup is None:
        state_to_matchup = lambda s: s

    job_specs: List[Dict[str, Any]] = []
    job_index = 0
    for state in unique_states:
        try:
            matchup_label_raw = state_to_matchup(state)
        except Exception:
            matchup_label_raw = state
        matchup_label = sanitize_for_filename(matchup_label_raw)

        if run_eval_prot:
            for rep in range(replicates_per_matchup):
                job_specs.append(
                    {
                        "job_index": job_index,
                        "eval_prot": True,
                        "state_subset": [state],
                        "matchup_label": matchup_label,
                        "replicate_idx": rep,
                        "launch_local_br_eval": launch_local_br_eval,
                    }
                )
                job_index += 1

        if run_eval_adv:
            for rep in range(replicates_per_matchup):
                job_specs.append(
                    {
                        "job_index": job_index,
                        "eval_prot": False,
                        "state_subset": [state],
                        "matchup_label": matchup_label,
                        "replicate_idx": rep,
                        "launch_local_br_eval": launch_local_br_eval,
                    }
                )
                job_index += 1
    return job_specs
