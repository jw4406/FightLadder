import torch
from typing import List, Any
import warnings
import pickle
import numpy as np
import re

CHARACTERS = ["ryu", "guile", "bison"] #TODO: Add all characters

def agent_win(info: dict) -> bool:
    """
    This function returns True if the agent won and False otherwise.

    Args:
        info (dict):
            Information dictionary, returned from env.step.

    Returns:
        True if the agent wins, False otherwise.
    """
    return info['enemy_hp'] < info['agent_hp']

def select_device(device_id: int=0, use_cpu: bool=False) -> torch.device:
    """
    This function returns "cuda" if it is available and "cpu" otherwise
    """
    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")

def move_policy(policy: torch.nn.Module, device: torch.device) -> None:
    """
    This function moves a policy to a selected device.
    """
    def _move_if_exists(policy: torch.nn.Module, device: torch.device, component_name: str) -> None:
        """
        This function moves policy.component to device if it exists.
        """
        if not hasattr(policy, component_name):
            return
        component = getattr(policy, component_name)
        if not isinstance(component, list): #Deal only with lists
            component = [component]
        for item in component:
            item.to(device)
    
    policy.to(device)
    for param in policy.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad = param.grad.to(device)
    for buffer in policy.buffers():
        buffer.data = buffer.data.to(device)
    for child in policy.children():
        move_policy(child, device)
    _move_if_exists(policy, device, "mlp_extractor")
    _move_if_exists(policy, device, "features_extractor")
    _move_if_exists(policy, device, "action_net")
    _move_if_exists(policy, device, "value_net")
    if hasattr(policy, "move_all_optimizers"):
        policy.move_all_optimizers(device)

def get_n_workers() -> tuple:
    """
    This function returns the number of workers available.

    Returns:
        n_gpus (int):
            Number of available GPUs.
        n_workers (int):
            Number of available workers (n_gpus if CUDA is available, 1 otherwise).

    """
    n_gpus = torch.cuda.device_count()
    n_workers = max(1, n_gpus)

    return n_gpus, n_workers

def state2matchup(state: str) -> str:
    """This function returns the matchup from the state."""
    state = state.split(".")
    return state[-3]

def select_matchup_env(matchups: List[str], i: int, envs_per_matchup: int) -> str:
    """This function generates a key string of the format f'<matchup>_{i}'."""
    curr_matchup = matchups[i*envs_per_matchup]
    return f"{curr_matchup}_{i}"

def find_character_name(s: str) -> str:
    """
    This function looks for a character name in a string.
    For matchup-style names like "..._ryu_vs_bison..." it returns the right-side
    character ("bison") to match right-policy environment selection.
    Returns an empty string if no character name is found.
    """
    s_lower = s.lower()

    # Prefer explicit matchup token parsing for new global naming.
    vs_match = re.search(r"_([a-z0-9]+)_vs_([a-z0-9]+)", s_lower)
    if vs_match:
        right_char = vs_match.group(2)
        if right_char in CHARACTERS:
            return right_char

    matches = [character for character in CHARACTERS if character in s_lower]
    if not matches:
        warnings.warn(f"Could not find a character name in {s}.")
        return ""

    # Legacy fallback for old names: keep deterministic and quiet.
    res = matches[0]
    if len(matches) > 1:
        # Multiple matches are expected for matchup-rich names; do not warn.
        if "_vs_" not in s_lower and "vs" not in s_lower:
            warnings.warn(f"Found multiple character names in {s}.")
        res = matches[-1]
    return res

def unpickle_policy(policy: Any) -> torch.nn.Module:
    """This is a helper function that unpickles a policy."""
    if isinstance(policy, bytes):
        policy = pickle.loads(policy)
    return policy

def mirror_flip_attributes(attribute1, attribute2):
    ego_halfway = attribute1.shape[0] // 2
    adv_halfway = attribute2.shape[0] // 2
    second_halfway = ego_halfway // 2
    all_envs_ego_left_ = attribute1[:ego_halfway]
    all_envs_ego_right_ = attribute1[ego_halfway:]

    left_ego_ = all_envs_ego_left_[:second_halfway]
    right_ego_ = all_envs_ego_right_[second_halfway:]

    adv_left_ = attribute2[:adv_halfway]
    adv_right_ = attribute2[adv_halfway:]

    # adv is in env order: adv_left_ = top-half envs (adv on RIGHT), adv_right_ = bottom-half envs (adv on LEFT)
    left_ = np.concatenate([left_ego_, adv_right_], axis=0) if isinstance(left_ego_, np.ndarray) else torch.cat([left_ego_, adv_right_], dim=0)
    right_ = np.concatenate([adv_left_, right_ego_], axis=0) if isinstance(right_ego_, np.ndarray) else torch.cat([adv_left_, right_ego_], dim=0)
    return left_, right_

def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move all optimizer state tensors to the specified device."""
    if not optimizer:
        return
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def merge_models(primary, secondary):
    # assume primary holds left model and secondary holds right model
    # also merge V and Q networks

    primary.policy.value_net = secondary.policy.value_net
    primary.policy.vf_features_extractor = secondary.policy.vf_features_extractor
    primary.policy.dstb_action_net = secondary.policy.dstb_action_net
    primary.policy.dstb_action_dist = secondary.policy.dstb_action_dist
    primary.policy.q_value_net = secondary.policy.q_value_net
    primary.policy.pi_dstb_features_extractor = secondary.policy.pi_dstb_features_extractor
    primary.policy.mlp_extractor.adv_action_extractor = secondary.policy.mlp_extractor.adv_action_extractor
    primary.policy.mlp_extractor.dstb_net = secondary.policy.mlp_extractor.dstb_net
    primary.policy.dstb_optimizer = secondary.policy.dstb_optimizer
    primary.policy.q_value_optimizer = secondary.policy.q_value_optimizer
    primary.policy.value_optimizer = secondary.policy.value_optimizer
    return primary