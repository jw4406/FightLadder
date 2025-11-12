import torch
from typing import List, Any
import numpy as np

import torch
import torch.autograd as autograd

from stable_baselines3.common.buffers import AdvRolloutBuffer
from stable_baselines3.common.policies import BasePolicy
from utils import move_policy, select_device, get_n_workers, state2matchup, select_matchup_env, unpickle_policy
DEBUG =False 
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
    #print(f"[DEBUG @ policy_loss]: Adv advantages mean in minibatch: {advantages.mean().item():.4f}")

    ratio = torch.exp(log_prob - old_log_prob.clone().detach().to(device))
    if not perturbed:
        #assert torch.allclose(log_prob, old_log_prob), "leader_grads, Log probabilities do not match between collection and training."
        pass
    
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
    return policy_loss, log_prob, entropy

def calculate_q_policy_loss(rollout_data: AdvRolloutBuffer, policy: BasePolicy, ego: bool, clip_range: float, use_sde: bool, device: torch.device, batch_size: int, envs_per_matchup: int, network_keys = None, perturbed=False):
    #TODO: Complete docstring
    actions = torch.Tensor(rollout_data.actions).to(device)
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

def calc_F_grad_single(
        ori_policy: BasePolicy,
        perturbed_policy: BasePolicy,
        ori_buf: AdvRolloutBuffer,
        perturbed_buf: AdvRolloutBuffer,
        ego: bool,
        i: int,
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
    break_signal = False
    
    perturbed_policy = unpickle_policy(perturbed_policy)
    network_keys, curr_buf, curr_perturbed_buf = _get_buffers_and_keys(ori_buf, perturbed_buf, ego, i, num_adversaries)
    for ori_rollout_data, perturbed_rollout_data in zip(curr_buf.get(batch_size), curr_perturbed_buf.get(batch_size)):                    
        policy_loss, log_prob, entropy = _calculate_policy_loss(
            ori_rollout_data, ori_policy, ego, clip_range, use_sde, device, batch_size, envs_per_matchup, network_keys=network_keys, perturbed=False
        )
        pg_losses.append(policy_loss.item())
        entropy_losses.append(entropy.mean().item())

        perturbed_policy_loss, _, _ = _calculate_policy_loss(
            perturbed_rollout_data, perturbed_policy, ego, clip_range, use_sde, device, batch_size, envs_per_matchup, network_keys=network_keys, perturbed=True
        )

        if DEBUG:
            F_grad = autograd.grad(policy_loss, ori_policy.ctrl_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True) if ego else \
                autograd.grad(policy_loss, ori_policy.dstb_optimizer.param_groups[0]['params'], create_graph=True, retain_graph=True, allow_unused=True)
            F_grad = torch.hstack([t.flatten() for t in F_grad])
        else:
            F_grad = _compute_grads(d, delta, ego_v, adv_v, policy_loss, perturbed_policy_loss, ego, i)# if ego else 0

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
                log_prob, entropy = ori_policy.evaluate_adv_actions(ori_rollout_data.observations, ori_rollout_data.actions, buf_num=[i])
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

    
    if target_kl is not None and np.mean(approx_kl_divs_epoch) > 1.5 * target_kl:
        break_signal = True

    return F_grad, pg_losses, entropy_losses, [], break_signal