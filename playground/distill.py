import argparse
import os
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn.functional as F

from common.controller import MixedActor, Policy
from common.envs_utils import (
    make_env,
    make_vec_envs,
)


def main():
    parser = argparse.ArgumentParser(
        description=("Examples:\n" "   python distill.py --env <ENV> --net <NET>\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--mini_batch_size", type=int, default=512)
    parser.add_argument("--num_processes", type=int, default=125)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--plank_class", type=str, default="Plank")
    parser.add_argument("--actor_class", type=str, default="MixedActor")
    args = parser.parse_args()

    env_name = args.env
    args.seed = 1234

    env_name_parts = env_name.split(":")
    save_name = "-".join(env_name_parts) if len(env_name_parts) > 1 else env_name

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_kwargs = {"plank_class": args.plank_class}
    dummy_env = make_env(env_name, **env_kwargs)

    args.save_dir = os.path.join(".", "models")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.set_num_threads(1)

    teacher_actor_critic = torch.load(args.net, map_location=device)

    envs = make_vec_envs(
        env_name,
        args.seed,
        args.num_processes,
        None,
        **env_kwargs,
    )

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])
    obs_dim = obs_shape[0]
    act_dim = envs.action_space.shape[0]

    actor_class = globals().get(args.actor_class)
    print(f"Actor Class: {actor_class}")
    student_actor_critic = Policy(actor_class(dummy_env)).to(device)

    num_steps = args.batch_size // args.num_processes
    num_processes = args.num_processes

    buffer_observations = torch.zeros(num_steps + 1, num_processes, *obs_shape, device=device)
    buffer_actions = torch.zeros(num_steps, num_processes, act_dim, device=device)
    buffer_values = torch.zeros(num_steps, num_processes, 1, device=device)

    optimizer = torch.optim.Adam(student_actor_critic.parameters(), lr=3e-4)

    obs = envs.reset()
    buffer_observations[0].copy_(torch.from_numpy(obs))

    # Depends on teacher network
    envs.set_env_params({"curriculum": 7})

    start = time.time()
    for epoch in range(args.num_epochs):
        with torch.no_grad():
            for step in range(num_steps):
                _, stochastic_action, _ = teacher_actor_critic.act(
                    buffer_observations[step], deterministic=False
                )
                value, action, _ = teacher_actor_critic.act(
                    buffer_observations[step], deterministic=True
                )
                cpu_actions = stochastic_action.cpu().numpy()

                obs, _, _, _ = envs.step(cpu_actions)

                buffer_observations[step + 1].copy_(torch.from_numpy(obs))
                buffer_actions[step].copy_(action)
                buffer_values[step].copy_(value)

        num_mini_batch = args.batch_size // args.mini_batch_size
        shuffled_indices = torch.randperm(num_mini_batch * args.mini_batch_size, generator=None, device=device)
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = buffer_observations.view(-1, obs_dim)
        actions_shaped = buffer_actions.view(-1, act_dim)
        values_shaped = buffer_values.view(-1, 1)

        ep_action_loss = torch.tensor(0.0, device=device).float()
        ep_value_loss = torch.tensor(0.0, device=device).float()

        for indices in shuffled_indices_batch:
            optimizer.zero_grad()

            observations_batch = observations_shaped[indices]
            actions_batch = actions_shaped[indices]
            values_batch = values_shaped[indices]

            pred_actions = student_actor_critic.actor(observations_batch)
            pred_values = student_actor_critic.get_value(observations_batch)

            action_loss = F.mse_loss(pred_actions, actions_batch)
            value_loss = F.mse_loss(pred_values, values_batch)
            (action_loss + value_loss).backward()

            optimizer.step()

            ep_action_loss.add_(action_loss.detach())
            ep_value_loss.add_(value_loss.detach())

        L = shuffled_indices_batch.shape[0]
        ep_action_loss.div_(L)
        ep_value_loss.div_(L)

        elapsed_time = time.time() - start
        torch.save(student_actor_critic, "distilled.pt")

        print(
            (
                f"Epoch {epoch+1:4d}/{args.num_epochs:4d} | "
                f"Elapsed Time {elapsed_time:8.2f} |"
                f"Action Loss: {ep_action_loss.item():8.4f} | "
                f"Value Loss: {ep_value_loss.item():8.2f}"
            )
        )




if __name__ == "__main__":
    main()
