from algorithms.dagger import train
from common.envs_utils import make_env, make_vec_envs

class Distiller:
    def __init__(self, env_name, base_env_kwargs, seed, device, num_processes):
        env_kwargs = {
            **base_env_kwargs,
            "determine": True,
        }

        self.envs_per_task = [
            # make_env(env_name, seed=seed, **env_kwargs)
            make_vec_envs(
                env_name, seed, num_processes, None, **env_kwargs
            )
            for i in range(2)
        ]
        self.device = device
        self.num_processes = num_processes

    def distill_policies(self, prev_actor_critic, actor_critic, prev_curriculum, prev_behavior_curriculum, current_curriculum, current_behavior_curriculum):
        env_kwargs = {
            "start_curriculum": current_curriculum,
            "start_behavior_curriculum": current_behavior_curriculum,
        }

        env_kwargs_normal = {
            "start_curriculum": prev_curriculum,
            "start_behavior_curriculum": prev_behavior_curriculum,
        }

        self.envs_per_task[0].set_env_params(env_kwargs)
        self.envs_per_task[1].set_env_params(env_kwargs_normal)

        train(
            actor_critic,
            prev_actor_critic,
            self.envs_per_task,
            # [env_kwargs, env_kwargs_normal],
            num_epochs=40,
            num_steps=5000,
            device=self.device,
            num_processes=self.num_processes,
        )