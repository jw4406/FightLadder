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
    #device='cpu'
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

        return actions_batch, observations_batch, returns_batch, np.array([env_ind.cpu() for env_ind in all_env_indices])
    
    # buffer.device = device

    #Process all rollout data and actions at once instead of batch by batch.
    actions_batch, observations_batch, returns_batch, all_env_indices = _prep_rollout_data_actions(batch_size, buffer)
    policy.num_global_env = num_envs
    policy.num_adv = 1
    for i in range(len(returns_batch) // batch_size):
        values = policy.evaluate_states(
            observations_batch[i * batch_size:(i + 1) * batch_size],
            buf_num=[adversary_index],
            env_indices=all_env_indices[i * batch_size:(i + 1) * batch_size]
            )
        values = -values.flatten()
        value_loss = F.mse_loss(values, returns_batch[i * batch_size:(i + 1) * batch_size])
        policy.value_optimizer.zero_grad()
        value_loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy.value_optimizer.step()

def _update_single_q_function(batch_size: int, max_grad_norm: float, policy, buffer, adversary_index: int, num_envs: int, device: torch.device, tag: str="", envs_per_matchup: int=None):
    """
    This function has to be placed outside of the object to enable parallel calls.
    TODO: Complete the docstring.
    TODO: Complete static types
    """
    #device='cpu'
    def _prep_rollout_data_actions(batch_size: int, buffer) -> tuple:
        """
        This is a helper function that gets all the rollout data and actions once instead of batch by batch.
        """
        all_rollout_data = list(buffer.get(batch_size))
        # need dones
        all_ego_actions = []
        all_adv_actions = []
        #all_dstb_actions = []
        all_observations = []
        all_next_observations = []
        all_returns = []
        all_rewards = []
        all_env_indices = []
        all_dones = []

        for rollout_data in all_rollout_data:
            all_ego_actions.append(torch.Tensor(rollout_data.actions))
            all_adv_actions.append(torch.Tensor(rollout_data.adv_actions))
            all_observations.append(rollout_data.observations)
            all_next_observations.append(rollout_data.next_observations)
            all_returns.append(torch.Tensor(rollout_data.returns))
            all_rewards.append(torch.Tensor(rollout_data.rewards))
            all_env_indices.extend(rollout_data.env_indices)
            all_dones.append(torch.Tensor(rollout_data.dones))
        
        ego_actions_batch = torch.cat(all_ego_actions).to(device)
        adv_actions_batch = torch.cat(all_adv_actions).to(device)
        observations_batch = torch.cat(all_observations).to(device)
        next_observations_batch = torch.cat(all_next_observations).to(device)
        returns_batch = torch.cat(all_returns).to(device)
        rewards_batch = torch.cat(all_rewards).to(device)
        dones_batch = torch.cat(all_dones).to(device)
        return ego_actions_batch, adv_actions_batch, observations_batch, next_observations_batch, returns_batch, np.array([env_ind.cpu() for env_ind in all_env_indices]), rewards_batch, dones_batch
    
    # buffer.device = device

    #Process all rollout data and actions at once instead of batch by batch.
    ego_actions_batch, adv_actions_batch, observations_batch, next_observations_batch, returns_batch, all_env_indices, rewards_batch, dones_batch = _prep_rollout_data_actions(batch_size, buffer)
    policy.num_global_env = num_envs
    policy.num_adv = 1
    for i in range(len(returns_batch) // batch_size):
        # do i need to rewrite the value prediciton to q here?
        curr_q_values = policy.q_value_forward(
            observations_batch[i * batch_size:(i + 1) * batch_size],
            ego_actions_batch[i * batch_size:(i + 1) * batch_size],
            adv_actions_batch[i * batch_size:(i + 1) * batch_size],)
            # buf_num=[adversary_index],
            # env_indices=all_env_indices[i * batch_size:(i + 1) * batch_size]
            # )
        with th.no_grad():
            (next_ego_actions, next_ego_log_prob), (next_adv_actions, next_adv_log_prob) = policy.predict(next_observations_batch[i * batch_size:(i + 1) * batch_size])
            next_q_values = policy.q_value_forward(
                next_observations_batch[i * batch_size:(i + 1) * batch_size],
                next_ego_actions,
                next_adv_actions,
                )
        actual_q_values = rewards_batch[i * batch_size:(i + 1) * batch_size] + policy.gamma * (1-dones_batch[i * batch_size:(i + 1) * batch_size]) *  next_q_values.flatten()
        values = -values.flatten()
        value_loss = F.mse_loss(values, returns_batch[i * batch_size:(i + 1) * batch_size])
        policy.value_optimizer.zero_grad()
        value_loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
