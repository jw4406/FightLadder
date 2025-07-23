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