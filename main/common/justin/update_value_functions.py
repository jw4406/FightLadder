from typing import List
import math
import torch
import torch as th
import numpy as np
import time
from torch.nn import functional as F
TIMING = False
def shard_indices(n_items: int, n_gpus: int) -> List[List[int]]:
    """
    Splits a range of indices [0, n_items) into n_gpus nearly equal-sized chunks.

    This is needed to distribute adversary buffer updates across multiple GPUs.

    Args:
        n_items (int):
            Total number of items to divide (e.g., adversary indices).
        n_gpus (int):
            Number of available GPUs to divide the work among.

    Returns:
        List[List[int]]: A list of `n_gpus` sublists, each containing integer indices.

    Raises:
        ValueError: If n_items < 0 or n_gpus <= 0.
    """
    if n_items < 0 or n_gpus <= 0:
        raise ValueError("n_items must be >= 0 and n_gpus must be > 0.")
    size = math.ceil(n_items / n_gpus)
    return [list(range(i * size, min((i + 1) * size, n_items))) for i in range(n_gpus)]

def _update_single_value_function(batch_size: int, max_grad_norm: float, policy, buffer, adversary_index: int, num_envs: int, device: torch.device, tag: str="", envs_per_matchup: int=None):
    """
    This function has to be placed outside of the object to enable parallel calls.
    TODO: Complete the docstring.
    TODO: Complete static types
    """
    device='cpu'
    def _prep_rollout_data_actions(batch_size: int, buffer) -> tuple:
        """
        This is a helper function that gets all the rollout data and actions once instead of batch by batch.
        """
        all_rollout_data = list(buffer.get(batch_size))
        all_actions = []
        #all_dstb_actions = []
        all_observations = []
        all_returns = []
        all_env_indices = []

        for rollout_data in all_rollout_data:
            all_actions.append(torch.Tensor(rollout_data.actions))
            #all_dstb_actions.append(torch.Tensor(rollout_data.dstb_actions))
            all_observations.append(rollout_data.observations)
            all_returns.append(torch.Tensor(rollout_data.returns))
            all_env_indices.extend(rollout_data.env_indices)
        
        actions_batch = torch.cat(all_actions).to(device)
        #dstb_actions_batch = torch.cat(all_dstb_actions).to(device)
        observations_batch = torch.cat(all_observations).to(device)
        returns_batch = torch.cat(all_returns).to(device)

        return actions_batch, observations_batch, returns_batch, np.array(all_env_indices)
    
    total_start_time = time.time()


    #Process all rollout data and actions at once instead of batch by batch.
    actions_batch, observations_batch, returns_batch, all_env_indices = _prep_rollout_data_actions(batch_size, buffer)
    policy.num_global_env = num_envs
    policy.num_adv = 1
    for i in range(len(returns_batch) // batch_size):
        values = policy.evaluate_states(
        observations_batch[i * batch_size:(i + 1) * batch_size],
        #actions_batch[i * batch_size:(i + 1) * batch_size],
        #dstb_actions_batch[i * batch_size:(i + 1) * batch_size],
        #shuffle_keys=all_env_indices[i * batch_size:(i + 1) * batch_size],
        #network_keys=[adversary_index], envs_per_matchup=envs_per_matchup
        buf_num=[adversary_index],
        env_indices=all_env_indices[i * batch_size:(i + 1) * batch_size]
        )
        #policy.train(True)
        #torch.backends.cudnn.enabled = False
        values = values.flatten()
        # offset = 12 # vf extractor and shared trunk are 12
        # num_per_head = 10 # lstm = 6, 2 linear layers = 2 + 2, total 10
        value_loss = F.mse_loss(values, returns_batch[i * batch_size:(i + 1) * batch_size])
        # indices = list(range(0, offset)) + list(range(offset + adversary_index * num_per_head, offset + (adversary_index + 1) * num_per_head))
        # value_grads = th.autograd.grad(value_loss, [policy.value_optimizer.param_groups[0]['params'][j] for j in indices])
        #value_grads = th.cat([grad.view(-1) for grad in value_grads])
        policy.value_optimizer.zero_grad()
        # for i in range(len(value_grads)):
        #     policy.value_optimizer.param_groups[0]['params'][indices[i]].grad = value_grads[i]
        value_loss.backward()
        #policy.value_optimizer.zero_grad()
        #for i in range(len(policy.value_optimizer.param_groups[0]['params'])):
        #    policy.value_optimizer.param_groups[0]['params'][i].grad = value_grads[i]
        #value_loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy.value_optimizer.step()

    total_end_time = time.time()
    if TIMING:
        print(f"      [Timing] Total _update_single_value_function ({tag}): {total_end_time - total_start_time:.4f}s")
