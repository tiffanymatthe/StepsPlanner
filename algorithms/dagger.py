from common.controller import SoftsignActor, Policy
import torch
from common.envs_utils import make_env, make_vec_envs
import torch.nn.functional as F
import time

def train(
    expert_policy,
    student_policy,
    env_name,
    env_per_task_kwargs, # list of kwargs
    num_epochs=20,
    num_processes=4, # per task
    num_steps=5000,
    mini_batch_size=512,
    device="cuda:0",
    seed=0,
) -> None:
    
    num_tasks = 2
    
    # dummy_env = make_env(env_name, **env_kwargs)
    # student_actor = student_policy.actor
    # student_actor = torch.load("daggered.pt", map_location=torch.device(device)) #  SoftsignActor(dummy_env).to(device)

    dummy_env = make_env(env_name, **env_per_task_kwargs[0])

    # assume first task is hopping, second task is everything else
    import copy
    expert_policy_for_previous_task = copy.deepcopy(student_policy)
    expert_policies_per_task = [expert_policy, expert_policy_for_previous_task]

    # if student_policy is None:
    controller = SoftsignActor(dummy_env).to(device)
    student_policy = Policy(controller)

    envs_per_task = [
        make_vec_envs(
            env_name, seed, num_processes, None, **env_per_task_kwargs[i]
        )
        for i in range(num_tasks)
    ]

    optimizer = torch.optim.Adam(student_policy.parameters(), lr=3e-4)

    obs_shape = envs_per_task[0].observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])
    obs_dim = obs_shape[0]
    act_dim = envs_per_task[0].action_space.shape[0]
    
    buffer_observations_per_task = [torch.zeros(num_steps * num_epochs + 1, num_processes, *obs_shape, device=device) for _ in range(num_tasks)]
    buffer_expert_actions_per_task = [torch.zeros(num_steps * num_epochs, num_processes, act_dim, device=device) for _ in range(num_tasks)]
    buffer_expert_values_per_task = [torch.zeros(num_steps * num_epochs, num_processes, act_dim, device=device) for _ in range(num_tasks)]

    epoch_threshold = 0

    start = time.time()
    for epoch in range(num_epochs):
        observations_shaped_per_task = [None for _ in range(num_tasks)]
        expert_actions_shaped_per_task = [None for _ in range(num_tasks)]
        expert_values_shaped_per_task = [None for _ in range(num_tasks)]
        for task_i in range(num_tasks):
            obs = envs_per_task[task_i].reset()
            buffer_observations_per_task[task_i][epoch * num_steps].copy_(torch.from_numpy(obs))
            with torch.no_grad():
                for step in range(num_steps):
                    buffer_index = epoch * num_steps + step
                    expert_value, expert_action, _ = expert_policies_per_task[task_i].act(
                        buffer_observations_per_task[task_i][buffer_index], deterministic=True
                    )
                    if epoch > epoch_threshold:
                        # determines if we get observations from the student or teacher, but reference data is from teacher for MSE loss calc
                        student_action = student_policy.actor(buffer_observations_per_task[task_i][buffer_index]) #, deterministic=True) # deterministic

                    if epoch == epoch_threshold:
                        cpu_actions = expert_action.cpu().numpy()
                    else:
                        cpu_actions = student_action.cpu().numpy()
                    obs, _, _, _ = envs_per_task[task_i].step(cpu_actions)

                    buffer_observations_per_task[task_i][buffer_index + 1].copy_(torch.from_numpy(obs))
                    buffer_expert_actions_per_task[task_i][buffer_index].copy_(expert_action)
                    buffer_expert_values_per_task[task_i][buffer_index].copy_(expert_value)

            batch_size = num_steps * (epoch + 1) * num_processes
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

                # can we apply this twice????
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
    student_file_name = "daggered_hopping_2_tasks_BC_random_initialization.pt"
    torch.save(student_policy, student_file_name)
    print(f"Saved student policy to {student_file_name}")
    for i in range(num_tasks):
        envs_per_task[i].close()
