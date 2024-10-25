from algorithms.dagger import train
import torch

def distill_policies(prev_actor_critic, actor_critic, prev_curriculum, prev_behavior_curriculum, current_curriculum, current_behavior_curriculum, base_env_kwargs, envs, device, num_processes):
    env_kwargs = {
        **base_env_kwargs,
        "start_curriculum": current_curriculum,
        "start_behavior_curriculum": current_behavior_curriculum,
        "determine": True,
    }

    env_kwargs_normal = {
        **base_env_kwargs,
        "start_curriculum": prev_curriculum,
        "start_behavior_curriculum": prev_behavior_curriculum,
        "determine": True,
    }

    torch.set_num_threads(1)

    train(
        actor_critic,
        prev_actor_critic,
        envs,
        [env_kwargs, env_kwargs_normal],
        device=device,
        num_epochs=40,
        num_steps=5000,
        num_processes=num_processes
    )

    envs.set_env_params({"curriculum": current_curriculum, "behavior_curriculum": current_behavior_curriculum, "determine": False})