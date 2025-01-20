from common.controller import SoftsignActor, Policy
import torch
from common.envs_utils import make_env, make_vec_envs
import torch.nn.functional as F
import time

def last_five_same_with_tolerance(lst, tolerance=1e-5):
    # Check if the list has fewer than 5 elements
    if len(lst) < 5:
        # If so, check if all elements in the list are approximately the same
        return all(abs(lst[i] - lst[0]) <= tolerance for i in range(1, len(lst)))
    else:
        # Otherwise, check if the last 5 elements are approximately the same
        last_five = lst[-5:]
        return all(abs(last_five[i] - last_five[0]) <= tolerance for i in range(1, 5))

def train(
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

    # envs_per_task = [
    #     make_env(env_name, seed=seed, **env_per_task_kwargs[i])
    #     # make_vec_envs(
    #     #     env_name, seed, num_processes, None, **env_per_task_kwargs[i]
    #     # )
    #     for i in range(num_tasks)
    # ]

    envs_per_task[0].set_env_params({"determine": True})
    
    optimizer = torch.optim.Adam(student_policy.parameters(), lr=3e-4)

    obs_shape = envs_per_task[0].observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])
    obs_dim = obs_shape[0]
    act_dim = envs_per_task[0].action_space.shape[0]
    
    buffer_observations_per_task = [torch.zeros(num_steps + 1, num_processes, *obs_shape, device=device) for _ in range(num_tasks)]
    buffer_expert_actions_per_task = [torch.zeros(num_steps, num_processes, act_dim, device=device) for _ in range(num_tasks)]
    buffer_expert_values_per_task = [torch.zeros(num_steps, num_processes, act_dim, device=device) for _ in range(num_tasks)]

    # assume first task is hopping, second task is everything else
    import copy
    with torch.no_grad():
        expert_policy_for_previous_task = copy.deepcopy(student_policy)
    expert_policies_per_task = [expert_policy, expert_policy_for_previous_task]

    start = time.time()

    value_losses = []

    for epoch in range(num_epochs):
        observations_shaped_per_task = [None for _ in range(num_tasks)]
        expert_actions_shaped_per_task = [None for _ in range(num_tasks)]
        expert_values_shaped_per_task = [None for _ in range(num_tasks)]
        for task_i in range(num_tasks):
            envs_per_task[task_i].set_env_params(env_per_task_kwargs[task_i])
            obs = envs_per_task[task_i].reset()
            buffer_observations_per_task[task_i][0].copy_(torch.from_numpy(obs))
            with torch.no_grad():
                for step in range(num_steps):
                    buffer_index = step
                    expert_value, expert_action, _ = expert_policies_per_task[task_i].act(
                        buffer_observations_per_task[task_i][buffer_index], deterministic=True
                    )
                    if epoch > 0:
                        # determines if we get observations from the student or teacher, but reference data is from teacher for MSE loss calc
                        student_action = student_policy.actor(buffer_observations_per_task[task_i][buffer_index]) #, deterministic=True) # deterministic

                    if epoch == 0:
                        cpu_actions = expert_action.cpu().numpy()
                    else:
                        cpu_actions = student_action.cpu().numpy()
                    obs, _, _, _ = envs_per_task[task_i].step(cpu_actions)

                    buffer_observations_per_task[task_i][buffer_index + 1].copy_(torch.from_numpy(obs))
                    buffer_expert_actions_per_task[task_i][buffer_index].copy_(expert_action)
                    buffer_expert_values_per_task[task_i][buffer_index].copy_(expert_value)

            batch_size = num_steps * num_processes
            num_mini_batch = batch_size // mini_batch_size
            shuffled_indices = torch.randperm(
                num_mini_batch * mini_batch_size, generator=None, device=device
            )
            shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

            observations_shaped_per_task[task_i] = buffer_observations_per_task[task_i].view(-1, obs_dim)
            expert_actions_shaped_per_task[task_i] = buffer_expert_actions_per_task[task_i].view(-1, act_dim)
            expert_values_shaped_per_task[task_i] = buffer_expert_values_per_task[task_i].view(-1, 1)

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

        value_losses.append(ep_value_loss.item())

        if ep_action_loss.item() <= 0.0002 or last_five_same_with_tolerance(value_losses, tolerance=0.00001):
            print("Quitting early.")
            break
    student_file_name = "reset_actor_with_hopping.pt"
    torch.save(student_policy, student_file_name)
    print(f"Saved student policy to {student_file_name}")
    for i in range(num_tasks):
        envs_per_task[i].close()
