from common.controller import SoftsignActor
import torch
from common.envs_utils import make_env

def train(
    expert_policy,
    env_name,
    env_kwargs,
    num_epochs=10,
    num_steps=500,
    num_processes=4,
    device="gpu",
) -> None:
    
    num_training_loops = 10
    num_rollouts = 500
    
    dataset = [] # log observations and actions

    actor_class = "SoftsignActor"

    dummy_env = make_env(env_name, **env_kwargs)
    student_actor = SoftsignActor(dummy_env)

    envs = ... # TODO: initialize

    act_dim = envs.action_space.shape[0]
    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])
    
    # used for internal rollout
    buffer_observations = torch.zeros(
        num_steps + 1, num_processes, *obs_shape, device=device
    )
    buffer_actions = torch.zeros(num_steps, num_processes, act_dim, device=device)

    for epoch in range(num_epochs):
        
        obs = envs.reset()
        buffer_observations[0].copy_(torch.from_numpy(obs))
        for step in range(num_steps):
            if epoch == 1:
                _, action, _ = expert_policy.act(
                    # TODO: try with and without deterministic
                    buffer_observations[step], deterministic=False
                )
            else:
                _, action, _ = student_actor.act(buffer_observations[step])
            
                cpu_actions = action.cpu().numpy()

            # TODO: create envs
            obs, _, _, _ = envs.step(cpu_actions)

            buffer_observations[step + 1].copy_(torch.from_numpy(obs))
            buffer_actions[step].copy_(action)

        # TODO add buffer_observations and buffer_actions to big array (epochs)

        for data in dataset:
            # TODO query expert to append an expert action to each sample state, append to dataset
            ...
        
        # TODO retrain learned policy with dataset w/ supervised learning, so simple MSE on actions
        # https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/bc.html#BC.train