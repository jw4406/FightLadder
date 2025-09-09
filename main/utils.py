import torch
from typing import List
import warnings

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

def select_device(device_id: int=0) -> torch.device:
    """
    This function returns "cuda" if it is available and "cpu" otherwise
    """
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
    WARNING: If there are more than 1 character in the string, returns an arbitrary one.
    Returns an empty string if no character name is found.
    """
    res = ""
    for character in CHARACTERS:
        if character in s:
            if res:
                warnings.warn(f"Found multiple character names in {s}.")
            res = character
    if not res:
        warnings.warn(f"Could not find a character name in {s}.")
    return res