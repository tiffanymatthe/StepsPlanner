from common.controller import SoftsignActor
import torch
from common.envs_utils import make_env, make_vec_envs
import torch.nn.functional as F
import time

def train(
    expert_policy,
    env_name,
    env_kwargs,
    num_epochs=20,
    num_processes=4,
    num_steps=5000,
    mini_batch_size=512,
    device="cuda:0",
    seed=0,
) -> None:
    
    dummy_env = make_env(env_name, **env_kwargs)
    student_actor = SoftsignActor(dummy_env).to(device)

    envs = make_vec_envs(
        env_name, seed, num_processes, None, **env_kwargs
    )

    optimizer = torch.optim.Adam(student_actor.parameters(), lr=3e-4)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])
    obs_dim = obs_shape[0]
    act_dim = envs.action_space.shape[0]
    
    buffer_observations = torch.zeros(num_steps * num_epochs + 1, num_processes, *obs_shape, device=device)
    buffer_expert_actions = torch.zeros(num_steps * num_epochs, num_processes, act_dim, device=device)

    start = time.time()
    for epoch in range(num_epochs):
        obs = envs.reset()
        buffer_observations[epoch * num_steps].copy_(torch.from_numpy(obs))
        with torch.no_grad():
            for step in range(num_steps):
                buffer_index = epoch * num_steps + step
                _, expert_action, _ = expert_policy.act(
                    buffer_observations[buffer_index], deterministic=True
                )
                if epoch > 0:
                    student_action = student_actor(buffer_observations[buffer_index])

                if epoch == 0:
                    cpu_actions = expert_action.cpu().numpy()
                else:
                    cpu_actions = student_action.cpu().numpy()
                obs, _, _, _ = envs.step(cpu_actions)

                buffer_observations[buffer_index + 1].copy_(torch.from_numpy(obs))
                buffer_expert_actions[buffer_index].copy_(expert_action)

        batch_size = num_steps * (epoch + 1) * num_processes
        num_mini_batch = batch_size // mini_batch_size
        shuffled_indices = torch.randperm(
            num_mini_batch * mini_batch_size, generator=None, device=device
        )
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = buffer_observations.view(-1, obs_dim)
        expert_actions_shaped = buffer_expert_actions.view(-1, act_dim)

        ep_action_loss = torch.tensor(0.0, device=device).float()

        for indices in shuffled_indices_batch:
            optimizer.zero_grad()

            observations_batch = observations_shaped[indices]
            actions_batch = expert_actions_shaped[indices]

            pred_actions = student_actor(observations_batch)

            action_loss = F.mse_loss(pred_actions, actions_batch)
            action_loss.backward()

            optimizer.step()

            ep_action_loss.add_(action_loss.detach())

        L = shuffled_indices_batch.shape[0]
        ep_action_loss.div_(L)

        elapsed_time = time.time() - start

        print(
            (
                f"Epoch {epoch+1:4d}/{num_epochs:4d} | "
                f"Elapsed Time {elapsed_time:8.2f} |"
                f"Action Loss: {ep_action_loss.item():8.4f} | "
            )
        )
    student_file_name = "daggered.pt"
    torch.save(student_actor, student_file_name)
    print(f"Saved student actor to {student_file_name}")
    envs.close()
