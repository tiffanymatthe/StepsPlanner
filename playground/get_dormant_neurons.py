import warnings ; warnings.warn = lambda *args,**kwargs: None

import torch
import numpy as np
import argparse
import csv
import os, io
from collections import deque

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from bottleneck import nanmean

import mocca_envs
from common.envs_utils import (
    make_env,
    make_vec_envs,
)

from algorithms.storage import RolloutStorage
from common.controller import SoftsignActor, Policy

class CPU_Unpickler(pickle.Unpickler):
    def __init__(self, *args, device='cpu', **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
        else:
            return super().find_class(module, name)

def get_mirror_function(indices, device="cpu"):

    negation_obs_indices = torch.from_numpy(indices[0]).to(device)
    right_obs_indices = torch.from_numpy(indices[1]).to(device)
    left_obs_indices = torch.from_numpy(indices[2]).to(device)
    negation_action_indices = torch.from_numpy(indices[3]).to(device)
    right_action_indices = torch.from_numpy(indices[4]).to(device)
    left_action_indices = torch.from_numpy(indices[5]).to(device)

    orl = torch.cat((right_obs_indices, left_obs_indices))
    olr = torch.cat((left_obs_indices, right_obs_indices))
    arl = torch.cat((right_action_indices, left_action_indices))
    alr = torch.cat((left_action_indices, right_action_indices))

    def mirror_function(trajectory_samples):
        (
            observations_batch,
            actions_batch,
        ) = trajectory_samples

        # Only observation and action needs to be mirrored
        observations_mirror = observations_batch.clone()
        observations_mirror[:, negation_obs_indices] *= -1
        observations_mirror[:, orl] = observations_mirror[:, olr]

        actions_mirror = actions_batch.clone()
        actions_mirror[:, negation_action_indices] *= -1
        actions_mirror[:, arl] = actions_mirror[:, alr]

        # Others need to be repeated
        observations_batch = torch.cat([observations_batch, observations_mirror])
        actions_batch = torch.cat([actions_batch, actions_mirror])

        return (
            observations_batch,
            actions_batch,
        )

    return mirror_function

def load_net(net_path, device, dummy_env):
    # net_path has .pt extension
    controller = SoftsignActor(dummy_env)
    actor_critic = Policy(controller)
    if net_path is not None:
        actor_critic.load_state_dict(torch.load(net_path, map_location=torch.device(device)))
        if not hasattr(actor_critic, 'feature_keys'):
            actor_critic.setup_feature_logging()
            actor_critic.actor.setup_feature_logging()

        import os
        base_path = os.path.splitext(net_path)[0]
        gnts_path = f"{base_path}_gnts.pkl"
        if os.path.exists(gnts_path):
            print(f"Loading saved gnts {gnts_path}")
            with open(gnts_path, "rb") as f:
                gnt_dict = CPU_Unpickler(f, device=device).load()
                critic_ages = gnt_dict["critic_gnt"].ages
                actor_ages = gnt_dict["actor_gnt"].ages
    else:
        critic_ages, actor_ages = None, None

    return actor_critic, critic_ages, actor_ages

@torch.no_grad()
def compute_dormant_units_proportion(net: Policy, critic_ages, actor_ages, device, envs, rollouts, num_steps, num_processes, mirror_function, dormant_unit_threshold: float = 0.01):
    """
    Computes the proportion of dormant units.
    """

    obs = envs.reset()
    rollouts.observations[0].copy_(torch.from_numpy(obs))

    episode_rewards = deque(maxlen=int(num_processes * num_steps / 1000))
    curriculum_metrics = [deque(maxlen=int(num_processes * num_steps / 1000)) for _ in range(4)]
    avg_heading_errs = [deque(maxlen=int(num_processes * num_steps / 1000)) for _ in range(4)]
    avg_dist_errs = [deque(maxlen=int(num_processes * num_steps / 1000)) for _ in range(4)]
    avg_timing_mets = [deque(maxlen=int(num_processes * num_steps / 1000)) for _ in range(4)]

    with torch.no_grad():
        for step in range(num_steps):
            value, action, action_log_prob = net.act(
                    rollouts.observations[step]
                )
            cpu_actions = action.cpu().numpy()

            obs, rewards, dones, infos = envs.step(cpu_actions)

            masks = torch.FloatTensor(~dones).unsqueeze(1)
            bad_masks = torch.ones((num_processes, 1))
            for p_index, info in enumerate(infos):
                # This information is added by common.envs_utils.TimeLimitMask
                if "bad_transition" in info:
                    bad_masks[p_index] = 0.0
                # This information is added by common.envs_utils.Monitor
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                if "curriculum_metric" in info:
                    curriculum_metrics[info["mask_combo_id"]].append(info["curriculum_metric"])
                if "avg_heading_err" in info:
                    avg_heading_errs[info["mask_combo_id"]].append(info["avg_heading_err"])
                if "avg_timing_met" in info:
                    avg_timing_mets[info["mask_combo_id"]].append(info["avg_timing_met"])
                if "avg_dist_err" in info:
                    avg_dist_errs[info["mask_combo_id"]].append(info["avg_dist_err"])


            rollouts.insert(
                torch.from_numpy(obs),
                action,
                action_log_prob,
                value,
                torch.from_numpy(rewards).float().unsqueeze(1),
                masks,
                bad_masks,
            )

        for i in range(2):
            avg_heading_err_nanmean = nanmean(avg_heading_errs[i])
            avg_timing_met_nanmean = nanmean(avg_timing_mets[i])
            avg_curriculum_nanmean = nanmean(curriculum_metrics[i])
            avg_dist_err_nanmean = nanmean(avg_dist_errs[i])
            print(f"{i}: {avg_curriculum_nanmean}, {avg_timing_met_nanmean}")

        next_value = net.get_value(rollouts.observations[-1]).detach()

        rollouts.compute_returns(next_value, True, 0.99, 0.95)

        obs_dim = rollouts.observations.size(-1)
        act_dim = rollouts.actions.size(-1)
        observations_shaped = rollouts.observations.view(-1, obs_dim)
        actions_shaped = rollouts.actions.view(-1, act_dim)
        observations_shaped, actions_shaped = mirror_function((observations_shaped, actions_shaped))
        max_index = min(actions_shaped.shape[0], observations_shaped.shape[0])
        observations_shaped, actions_shaped = observations_shaped[0:max_index], actions_shaped[0:max_index]
        (
            values,
            action_log_probs,
            dist_entropy,
        ) = net.evaluate_actions(
            observations_shaped, actions_shaped, to_log_features=True
        )
    
        rollouts.after_update()

    def get_dead_neurons(features_per_layer, ages):
        dead_neurons = torch.zeros(len(features_per_layer), dtype=torch.float32)
        total_number = 0
        for layer_idx in range(len(features_per_layer)):
            # not sure why they don't have abs() in Dohare code, probably because ReLU anyways?
            # != 0 to ignore just reinitialized neurons??? doesn't matter here
            # https://github.com/shibhansh/loss-of-plasticity/blob/63c35f3c758bbb713dd42c72d43dc192fde0d109/lop/incremental_cifar/post_run_analysis.py#L106
            eligible_feature_indices = torch.where(ages[layer_idx] >= 0)[0] # 10000)[0]
            total_number += eligible_feature_indices.shape[0]
            actual_features = features_per_layer[layer_idx][:,eligible_feature_indices]
            score = (actual_features).abs().mean(dim=0)
            normalized_score = score / (score.mean() + 1e-9)

            dead_neurons[layer_idx] = (normalized_score < dormant_unit_threshold).sum()
        number_of_features = total_number
        # print(f"{number_of_features} for {len(features_per_layer)} layers")
        return dead_neurons.sum().item() / number_of_features
    
    def compute_average_weight_magnitude(layers):
        total_magnitude = 0
        total_weights = 0
        
        for layer in layers:
            # Check if the layer has parameters
            if isinstance(layer, torch.nn.Module):
                for param in layer.parameters():
                    if param.requires_grad:  # Only consider trainable parameters
                        total_magnitude += param.abs().sum().item()
                        total_weights += param.numel()
        
        return total_magnitude / total_weights if total_weights > 0 else 0
    
    return get_dead_neurons(net.get_activations(), critic_ages), get_dead_neurons(net.actor.get_activations(), actor_ages), compute_average_weight_magnitude(net.layers_to_check), compute_average_weight_magnitude(net.actor.layers_to_check)

def main(net, curriculum, behavior_curriculum):

    device = "cpu"
    env_name = "Walker3DStepperEnv-v0"
    seed = 16
    num_processes = 8
    num_steps = 5000

    env_kwargs = {
        "start_curriculum": curriculum,
        "start_behavior_curriculum": behavior_curriculum,
        "curriculum": curriculum,
        "behavior_curriculum": behavior_curriculum,
        "from_net": True,
        "determine": True,
    }

    dummy_env = make_env(env_name, **env_kwargs)
    policy, critic_ages, actor_ages = load_net(
        # "../StepsPlannerTwo/runs/dream/dec_8/from_scratch_plasticity_avg_10_cont/models/Walker3DStepperEnv-v0_curr_10_8.pt",
        # "runs/dream/dec_10/plasticity_baseline_cont/models/Walker3DStepperEnv-v0_curr_3_9.pt",
        # "runs/dream/dec_2/from_scratch_plasticity_avg_10/models/Walker3DStepperEnv-v0_curr_2_9.pt",
        net,
        device,
        dummy_env,
    )

    indices = dummy_env.unwrapped.get_mirror_indices()
    mirror_function = get_mirror_function(indices, device=device)
    
    envs = make_vec_envs(
        env_name,
        seed,
        num_processes,
        log_dir="logs",
        **env_kwargs
    )

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])
    action_dim = envs.action_space.shape[0]

    rollouts = RolloutStorage(num_steps, num_processes, obs_shape, action_dim)
    rollouts.to(device)

    dead_critic, dead_actor, avg_w_critic, avg_w_actor = compute_dormant_units_proportion(policy, critic_ages, actor_ages, device, envs, rollouts,num_steps, num_processes, mirror_function, dormant_unit_threshold=0.01)

    print(f"{behavior_curriculum}:{curriculum} for critic={dead_critic:.3f} and actor={dead_actor:.3f}. Avg weights: {avg_w_critic:.3f}, {avg_w_actor:.3f}")
    envs.close()

    return (dead_critic, dead_actor, avg_w_critic, avg_w_actor)

def iterate(net, writer, start_b, end_b, start_c=0, end_c=9):
    for b in range(start_b,end_b+1):
        if b == start_b and b == end_b:
            c_start = start_c
            c_end = end_c
        elif b == start_b:  # First behavior curriculum
            c_start = start_c
            c_end = 9
        elif b == end_b:  # Last behavior curriculum
            c_start = 0
            c_end = end_c
        else:  # Middle behavior curricula
            c_start = 0
            c_end = 9
        for c in range(c_start,c_end+1):
            actual_net = f"{net}/Walker3DStepperEnv-v0_curr_{b}_{c}.pt" if net is not None else None
            dead_critic, dead_actor, avg_w_critic, avg_w_actor = main(actual_net, c, b)
            writer.writerow({
                "behavior_curriculum": b,
                "curriculum": c,
                "dead_actor": dead_actor,
                "dead_critic": dead_critic,
                "avg_w_critic": avg_w_critic,
                "avg_w_actor": avg_w_actor,
                "net": actual_net
            })

if __name__ == "__main__":
    csv_file = "resets_w_weights.csv" # "dormant_proper_reset_0_01.csv"

    with open(csv_file, mode="w", newline="", buffering=1) as file:
        writer = csv.DictWriter(file, fieldnames=["behavior_curriculum", "curriculum", "dead_actor", "dead_critic", "avg_w_critic", "avg_w_actor", "net"])
        if file.tell() == 0:
                writer.writeheader()

        net = "runs/dream/dec_1/from_scratch/models"
        iterate(net,writer,0,0,0,0)

        net="runs/dream/dec_11/plasticity_reset_properly_cont/models"
        iterate(net,writer,0,1,1,3)

        # net="runs/dream/dec_2/from_scratch_plasticity_avg_10/models"
        # iterate(net,writer,0,4,1,5)

        # net="runs/dream/dec_5/from_scratch_plasticity_avg_10_cont/models"
        # iterate(net,writer,4,5,6,8)

        # net="runs/dream/dec_8/from_scratch_plasticity_avg_10_cont/models"
        # iterate(net,writer,6,10,0,8)

        # net = "runs/dream/dec_1/from_scratch/models"
        # iterate(net,writer,0,0,0,0)

        # net = "runs/dream/dec_4/plasticity_baseline/2024_12_04__18_12_16__plasticity_baseline/1/models"
        # iterate(net,writer,0,1,1,2)

        # net = "runs/dream/dec_5/plasticity_baseline_cont/1/models"
        # iterate(net,writer,1,1,3,4)

        # net = "runs/dream/dec_6/plasticity_baseline_cont/models"
        # iterate(net,writer,1,1,5,6)

        # net = "runs/dream/dec_7/plasticity_baseline_cont/models"
        # iterate(net,writer,1,1,7,8)

        # net = "runs/dream/dec_8/plasticity_baseline_cont/models"
        # iterate(net,writer,1,2,9,3)

        # net = "runs/dream/dec_9/plasticity_baseline_cont/models"
        # iterate(net,writer,2,2,4,8)

        # net = "runs/dream/dec_10/plasticity_baseline_cont/models"
        # iterate(net,writer,2,4,9,5)