import torch
from typing import List, Any, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.autograd as autograd

def average_tensor_tuples(list_of_tuples: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    Average across a list of tuples of tensors while preserving the shape.
    
    Args:
        list_of_tuples: List of tuples, where each tuple contains tensors of the same shape.
                        For example: [(t1_1, t1_2, ..., t1_n), (t2_1, t2_2, ..., t2_n), ...]
    
    Returns:
        A tuple of averaged tensors: (avg(t1_1, t2_1, ...), avg(t1_2, t2_2, ...), ...)
    
    Example:
        >>> list_of_tuples = [(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
        ...                   (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0]))]
        >>> result = average_tensor_tuples(list_of_tuples)
        >>> # result = (tensor([3.0, 4.0]), tensor([5.0, 6.0]))
    """
    if not list_of_tuples:
        raise ValueError("list_of_tuples cannot be empty")
    
    # Get the number of tensors per tuple (assuming all tuples have the same length)
    num_tensors = len(list_of_tuples[0])
    
    # Verify all tuples have the same length
    if not all(len(tup) == num_tensors for tup in list_of_tuples):
        raise ValueError("All tuples must have the same length")
    
    # Group corresponding tensors together and average them
    averaged_tensors = []
    for i in range(num_tensors):
        # Collect the i-th tensor from each tuple
        tensors_to_average = [tup[i] for tup in list_of_tuples]
        # Stack and average
        stacked = torch.stack(tensors_to_average, dim=0)
        averaged = torch.mean(stacked, dim=0)
        averaged_tensors.append(averaged)
    
    return tuple(averaged_tensors)

from stable_baselines3.common.buffers import AdvRolloutBuffer
from stable_baselines3.common.policies import BasePolicy
from utils import move_policy, select_device, get_n_workers, state2matchup, select_matchup_env, unpickle_policy
DEBUG = False
PARALLEL_CALC_F = True
def _get_buffers_and_keys(ori_buf: AdvRolloutBuffer, perturbed_buf: AdvRolloutBuffer, ego: bool, index: int, num_adversaries: int) -> tuple:
    #TODO: Add docstring
    if ego:
        network_keys = [k for k in range(num_adversaries)]
        curr_buf = ori_buf
        curr_perturbed_buf = perturbed_buf
    else:
        network_keys = [index]
        curr_buf = ori_buf[index]
        curr_perturbed_buf = perturbed_buf[index]
    #if not ego:
    print(f"[DEBUG @ get_buffers]: For adv {index}, using buffer with advantages mean: {curr_buf.advantages.mean().item():.4f}")
    #    print(f"[DEBUG @ get_buffers]: For adv {index}, using buffer with advantages mean: {curr_buf.advantages.mean().item():.4f}")
    return network_keys, curr_buf, curr_perturbed_buf

def _calculate_policy_loss(rollout_data: AdvRolloutBuffer, policy: BasePolicy, ego: bool, clip_range: float, use_sde: bool, device: torch.device, batch_size: int, envs_per_matchup: int, network_keys = None, perturbed=False):
    #TODO: Complete docstring
    actions = torch.Tensor(rollout_data.actions).to(device)
    if ego is False:
        actions = torch.Tensor(rollout_data.adv_actions).to(device)
    #dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(device)

    if use_sde:
        policy.reset_noise(batch_size)

    #with torch.no_grad():
    if not DEBUG:
        with torch.no_grad():
            if ego:
                old_log_prob = rollout_data.old_log_prob
                log_prob, entropy = policy.evaluate_ego_actions(rollout_data.observations, actions)
            else:
                old_log_prob = rollout_data.old_log_prob
                log_prob, entropy = policy.evaluate_adv_actions(rollout_data.observations, actions, buf_num=network_keys)
    else:
        if ego:
            old_log_prob = rollout_data.old_log_prob
            log_prob, entropy = policy.evaluate_ego_actions(rollout_data.observations, actions)
        else:
            old_log_prob = rollout_data.old_log_prob
            log_prob, entropy = policy.evaluate_adv_actions(rollout_data.observations, actions, buf_num=network_keys)
    
    advantages = rollout_data.advantages# if ego else -rollout_data.advantages
    normalize_advantage = True
    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
    if normalize_advantage and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #print(f"[DEBUG @ policy_loss]: Adv advantages mean in minibatch: {advantages.mean().item():.4f}")

    ratio = torch.exp(log_prob - old_log_prob.clone().detach().to(device))
    if not perturbed:
        #assert torch.allclose(log_prob, old_log_prob), "leader_grads, Log probabilities do not match between collection and training."
        pass
    
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
    return policy_loss, log_prob, entropy

def _calculate_q_policy_loss(rollout_data: AdvRolloutBuffer, policy: BasePolicy, ego: bool, clip_range: float, use_sde: bool, device: torch.device, batch_size: int, envs_per_matchup: int, network_keys = None, perturbed=False):
    #TODO: Complete docstring
    actions = torch.Tensor(rollout_data.actions).to(device)
    adv_actions = torch.Tensor(rollout_data.adv_actions).to(device)
    #dstb_actions = torch.Tensor(rollout_data.dstb_actions).to(device)

    if use_sde:
        policy.reset_noise(batch_size)

    #with torch.no_grad():
    if not DEBUG:
        with torch.no_grad():
            if ego:
                old_log_prob = rollout_data.old_log_prob
                log_prob, entropy = policy.evaluate_ego_actions(rollout_data.observations, actions)
            else:
                old_log_prob = rollout_data.old_log_prob
                log_prob, entropy = policy.evaluate_adv_actions(rollout_data.observations, adv_actions, buf_num=network_keys)
    else:
        if ego:
            old_log_prob = rollout_data.old_log_prob
            log_prob, entropy = policy.evaluate_ego_actions(rollout_data.observations, actions)
        else:
            old_log_prob = rollout_data.old_log_prob
            log_prob, entropy = policy.evaluate_adv_actions(rollout_data.observations, adv_actions, buf_num=network_keys)
    
    advantages = rollout_data.q_values# if ego else -rollout_data.advantages
    #print(f"[DEBUG @ policy_loss]: Adv advantages mean in minibatch: {advantages.mean().item():.4f}")

    ratio = torch.exp(log_prob - old_log_prob.clone().detach().to(device))
    if not perturbed:
        #assert torch.allclose(log_prob, old_log_prob), "leader_grads, Log probabilities do not match between collection and training."
        pass
    
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
    return policy_loss, log_prob, entropy
def _compute_grads(d: int, delta: float, ego_v: torch.Tensor, adv_v: torch.Tensor, policy_loss: torch.Tensor, perturbed_policy_loss: torch.Tensor, ego: bool, adv_num=None) -> torch.Tensor:
    if ego is False:
        assert adv_num is not None
    if ego:
        multiplier = ego_v
    else:
        multiplier = adv_v
    F_grad = d / delta * (perturbed_policy_loss - policy_loss) * multiplier
    return F_grad    

def per_batch_calc_F_grad_single(perturbed_buf: AdvRolloutBuffer, perturbed_policy: BasePolicy, ori_buf: AdvRolloutBuffer, ori_policy: BasePolicy, ego: bool, i: int, perturbed_buf_num: int, num_adversaries: int, batch_size: int, clip_range: float, use_sde: bool, device: torch.device, envs_per_matchup: int, d: int, delta: float, ego_v: torch.Tensor, adv_v: torch.Tensor, pg_losses: List[float], entropy_losses: List[float], approx_kl_divs_epoch):
    network_keys, curr_buf, curr_perturbed_buf = _get_buffers_and_keys(ori_buf, perturbed_buf, ego, i, num_adversaries)
    ori_rollout_data = list(curr_buf.get(batch_size))[perturbed_buf_num]

    # this might introduce a bug for ego

    perturbed_rollout_data = list(curr_perturbed_buf.get(batch_size))[perturbed_buf_num]
    policy_loss, log_prob, entropy = _calculate_policy_loss(
        ori_rollout_data, ori_policy, ego, clip_range, use_sde, device, batch_size, envs_per_matchup, network_keys=network_keys, perturbed=False
    )
    pg_losses.append(policy_loss.item())
    entropy_losses.append(entropy.mean().item())

    perturbed_policy_loss, _, _ = _calculate_policy_loss(
        perturbed_rollout_data, perturbed_policy, ego, clip_range, use_sde, device, batch_size, envs_per_matchup, network_keys=network_keys, perturbed=True
    )

    if DEBUG:
        F_grad = autograd.grad(perturbed_policy_loss, perturbed_policy.ctrl_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True) if ego else \
            autograd.grad(perturbed_policy_loss, perturbed_policy.dstb_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True, allow_unused=True)
        F_grad = torch.hstack([t.flatten() for t in F_grad])
    else:
        F_grad = _compute_grads(d, delta, ego_v, adv_v, policy_loss, perturbed_policy_loss, ego, i)# if ego else 0

    reshaped_grad = []
    count = 0
    size_lists = [list(x.shape) for x in ori_policy.ctrl_optimizer.param_groups[0]['params']] if ego else [list(x.shape) for x in ori_policy.dstb_optimizer.param_groups[0]['params']]
    for k in range(len(size_lists)):
        numel = np.prod(size_lists[k])
        reshaped_grad.append(torch.reshape(F_grad[count: count + numel], size_lists[k]))
        count += numel
    
    #F_grads.append(reshaped_grad)
        

    # #assert improvement > 0, "CRITICAL BUG: Policy is making good actions LESS likely!"
    with torch.no_grad():
        old_log_prob_tensor = ori_rollout_data.old_log_prob
        #run forward pass to get the log_prob
        if ego:
            log_prob, entropy = ori_policy.evaluate_ego_actions(ori_rollout_data.observations, ori_rollout_data.actions)
            #log_prob, entropy = ori_policy.evaluate_actions(
            #ori_rollout_data.observations.clone().detach().to(device), ori_rollout_data.actions.clone().detach().to(device), ori_rollout_data.dstb_actions.clone().detach().to(device),
            #shuffle_keys=ori_rollout_data.env_indices, network_keys=network_keys, envs_per_matchup=envs_per_matchup
        #)
        else:
            log_prob, entropy = ori_policy.evaluate_adv_actions(ori_rollout_data.observations, ori_rollout_data.adv_actions, buf_num=[i])
            #_, _, _, log_prob, entropy = ori_policy.evaluate_actions(
                #ori_rollout_data.observations.clone().detach().to(device), ori_rollout_data.actions.clone().detach().to(device), ori_rollout_data.dstb_actions.clone().detach().to(device),
                #shuffle_keys=ori_rollout_data.env_indices, network_keys=network_keys, envs_per_matchup=envs_per_matchup
            #)
        #) 
        #run forward pass to get the log_prob
        with torch.no_grad():
            _, log_prob = ori_policy.ego_forward(ori_rollout_data.observations, deterministic=False)
        log_ratio = log_prob - old_log_prob_tensor
        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
        approx_kl_divs_epoch.append(approx_kl_div)


    return reshaped_grad,policy_loss, perturbed_policy_loss, log_prob, entropy, approx_kl_div, pg_losses, entropy_losses, approx_kl_divs_epoch

def calc_F_grad_single(
        ori_policy: BasePolicy,
        perturbed_policies: List[BasePolicy],
        ori_buf: AdvRolloutBuffer,
        perturbed_bufs: List[AdvRolloutBuffer],
        ego: bool,
        i: int,
        perturbed_buf_num: int,
        num_adversaries: int,
        batch_size: int,
        clip_range: float,
        use_sde: bool,
        device: torch.device,
        envs_per_matchup: int,
        d: int,
        delta: float,
        ego_v: torch.Tensor,
        adv_v: torch.Tensor,
        target_kl: Any,
        first_epoch: bool,
        ):
    #Return values
    pg_losses = []
    entropy_losses = []
    approx_kl_divs_epoch = []
    F_grads = []
    break_signal = False
    if type(perturbed_bufs) == tuple or type(perturbed_bufs) == list:
        #perturbed_bufs = perturbed_bufs[0]
        pass
    perturbed_policies = [unpickle_policy(perturbed_policies[i]) for i in range(len(perturbed_policies))]
    #for ori_rollout_data, perturbed_rollout_data in zip(curr_buf.get(batch_size), curr_perturbed_buf.get(batch_size)):  
    futures = []   
    with ThreadPoolExecutor(max_workers=len(perturbed_bufs)) as executor:
        for perturbed_buf, perturbed_policy in zip(perturbed_bufs, perturbed_policies):
            if PARALLEL_CALC_F:
                    future = executor.submit(per_batch_calc_F_grad_single, perturbed_buf, perturbed_policy, ori_buf, ori_policy, ego, i, perturbed_buf_num, num_adversaries, batch_size, clip_range, use_sde, device, envs_per_matchup, d, delta, ego_v, adv_v, pg_losses, entropy_losses, approx_kl_divs_epoch)
                    futures.append(future)
            else:
                F_grads_test, policy_loss_test, perturbed_policy_loss_test, log_prob_test, entropy_test, approx_kl_div_test, pg_losses_test, entropy_losses_test, approx_kl_divs_epoch_test = per_batch_calc_F_grad_single(perturbed_buf, perturbed_policy, ori_buf, ori_policy, ego, i, perturbed_buf_num, num_adversaries, batch_size, clip_range, use_sde, device, envs_per_matchup, d, delta, ego_v, adv_v, pg_losses, entropy_losses, approx_kl_divs_epoch)
                F_grads.append(F_grads_test)
        if PARALLEL_CALC_F:
            for future in futures:
                F_grads_test, policy_loss_test, perturbed_policy_loss_test, log_prob_test, entropy_test, approx_kl_div_test, pg_losses_test, entropy_losses_test, approx_kl_divs_epoch_test = future.result()
                F_grads.append(F_grads_test)

    if target_kl is not None and np.mean(approx_kl_divs_epoch) > 1.5 * target_kl:
        break_signal = True
    F_grads_averaged = average_tensor_tuples(F_grads)

    return F_grads_averaged, pg_losses, entropy_losses, [], break_signal
