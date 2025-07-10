import torch

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

def select_device() -> str:
    """
    This function returns "cuda" if it is available and "cpu" otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"