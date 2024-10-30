import torch
import time
import numpy as np
import torch.nn.functional as F
from common.envs_utils import make_env, make_vec_envs

class Distiller:
    def __init__(self, env_name, base_env_kwargs, seed, device, num_processes, num_epochs, envs):
        env_kwargs = {
            **base_env_kwargs,
            "determine": True,
        }

        num_processes = 10 # overwrite

        # self.envs_per_task = [envs, envs]

        self.envs_per_task = [
            # make_env(env_name, seed=seed, **env_kwargs)
            make_vec_envs(
                env_name, seed, num_processes, None, **env_kwargs
            )
            for i in range(2)
        ]

        obs_shape = self.envs_per_task[0].observation_space.shape
        obs_shape = (obs_shape[0], *obs_shape[1:])
        obs_dim = obs_shape[0]
        act_dim = self.envs_per_task[0].action_space.shape[0]

        self.num_steps = 5000
        self.num_epochs = num_epochs
    
        self.buffer_observations_per_task = [torch.zeros(self.num_steps * num_epochs + 1, num_processes, *obs_shape, device=device) for _ in range(2)]
        self.buffer_expert_actions_per_task = [torch.zeros(self.num_steps * num_epochs, num_processes, act_dim, device=device) for _ in range(2)]
        self.buffer_expert_values_per_task = [torch.zeros(self.num_steps * num_epochs, num_processes, act_dim, device=device) for _ in range(2)]

        self.device = device
        self.num_processes = num_processes

    def distill_policies(self, prev_actor_critic, actor_critic, prev_curriculum, prev_behavior_curriculum, current_curriculum, current_behavior_curriculum, small_update):
        if prev_curriculum == 0 and prev_behavior_curriculum == 0:
            print("Not distilling the first curriculum.")
            return
        env_kwargs = {
            "start_curriculum": current_curriculum,
            "start_behavior_curriculum": current_behavior_curriculum,
            "determine": 2 if small_update else 1,
        }

        # expert
        env_kwargs_normal = {
            "start_curriculum": prev_curriculum,
            "start_behavior_curriculum": prev_behavior_curriculum,
            "determine": 1 if small_update else 0,
        }

        self.envs_per_task[0].set_env_params(env_kwargs)
        self.envs_per_task[1].set_env_params(env_kwargs_normal)

        self.train(
            actor_critic,
            prev_actor_critic,
            self.envs_per_task,
            [env_kwargs, env_kwargs_normal],
            num_epochs=self.num_epochs,
            num_steps=self.num_steps,
            device=self.device,
            num_processes=self.num_processes,
        )

        # self.envs_per_task[0].set_env_params({"curriculum": current_curriculum, "behavior_curriculum": current_behavior_curriculum})


    def train(
        self,
        expert_policy,
        student_policy,
        envs_per_task,
        env_per_task_kwargs, # list of kwargs
        num_epochs=20,
        num_steps=5000,
        mini_batch_size=512,
        num_processes=4,
        device="cuda:0",
    ) -> None:
        
        num_tasks = 2

        # envs_per_task[0].set_env_params({"determine": True})
        
        optimizer = torch.optim.Adam(student_policy.parameters(), lr=3e-4)

        obs_shape = envs_per_task[0].observation_space.shape
        obs_shape = (obs_shape[0], *obs_shape[1:])
        obs_dim = obs_shape[0]
        act_dim = envs_per_task[0].action_space.shape[0]

        # assume first task is hopping, second task is everything else
        import copy
        with torch.no_grad():
            expert_policy_for_previous_task = copy.deepcopy(student_policy)
        expert_policies_per_task = [expert_policy, expert_policy_for_previous_task]

        start = time.time()
        for epoch in range(num_epochs):
            observations_shaped_per_task = [None for _ in range(num_tasks)]
            expert_actions_shaped_per_task = [None for _ in range(num_tasks)]
            expert_values_shaped_per_task = [None for _ in range(num_tasks)]
            for task_i in range(num_tasks):
                # envs_per_task[task_i].set_env_params(env_per_task_kwargs[task_i])
                obs = envs_per_task[task_i].reset()
                self.buffer_observations_per_task[task_i][0].copy_(torch.from_numpy(obs))
                with torch.no_grad():
                    for step in range(num_steps):
                        buffer_index = step + epoch * num_steps
                        expert_value, expert_action, _ = expert_policies_per_task[task_i].act(
                            self.buffer_observations_per_task[task_i][buffer_index], deterministic=True
                        )

                        use_expert = np.random.rand() > min(epoch / 10, 1)

                        if not use_expert:
                            # determines if we get observations from the student or teacher, but reference data is from teacher for MSE loss calc
                            _, student_action, _ = student_policy.act(self.buffer_observations_per_task[task_i][buffer_index], deterministic=(np.random.rand() < epoch / num_epochs))

                        if use_expert:
                            cpu_actions = expert_action.cpu().numpy()
                        else:
                            cpu_actions = student_action.cpu().numpy()
                        obs, _, _, _ = envs_per_task[task_i].step(cpu_actions)

                        self.buffer_observations_per_task[task_i][buffer_index + 1].copy_(torch.from_numpy(obs))
                        self.buffer_expert_actions_per_task[task_i][buffer_index].copy_(expert_action)
                        self.buffer_expert_values_per_task[task_i][buffer_index].copy_(expert_value)

                batch_size = num_steps * (epoch + 1) * num_processes
                num_mini_batch = batch_size // mini_batch_size
                shuffled_indices = torch.randperm(
                    num_mini_batch * mini_batch_size, generator=None, device=device
                )
                shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

                observations_shaped_per_task[task_i] = self.buffer_observations_per_task[task_i].view(-1, obs_dim)
                expert_actions_shaped_per_task[task_i] = self.buffer_expert_actions_per_task[task_i].view(-1, act_dim)
                expert_values_shaped_per_task[task_i] = self.buffer_expert_values_per_task[task_i].view(-1, 1)

            ep_action_loss = torch.tensor(0.0, device=device).float()
            ep_value_loss = torch.tensor(0.0, device=device).float()

            for indices in shuffled_indices_batch:
                optimizer.zero_grad()

                for task_i in range(num_tasks):
                    observations_batch = observations_shaped_per_task[task_i][indices]
                    actions_batch = expert_actions_shaped_per_task[task_i][indices]
                    values_batch = expert_values_shaped_per_task[task_i][indices]

                    pred_actions = student_policy.actor(observations_batch)
                    pred_values = student_policy.get_value(observations_batch)

                    action_loss = F.mse_loss(pred_actions, actions_batch)
                    value_loss = F.mse_loss(pred_values, values_batch)

                    (action_loss + value_loss).backward()

                    ep_action_loss.add_(action_loss.detach())
                    ep_value_loss.add_(value_loss.detach())

                optimizer.step()

            L = shuffled_indices_batch.shape[0] * num_tasks
            ep_action_loss.div_(L)
            ep_value_loss.div_(L)

            elapsed_time = time.time() - start

            print(
                (
                    f"Epoch {epoch+1:4d}/{num_epochs:4d} | "
                    f"Elapsed Time {elapsed_time:8.2f} |"
                    f"Action Loss: {ep_action_loss.item():8.4f} | "
                    f"Value Loss: {ep_value_loss.item():8.4f} | "
                )
            )

            if ep_action_loss.item() <= 0.0002:
                print("Quitting early.")
                break