from common.controller import SoftsignActor

from common.envs_utils import make_env

def train(
    expert_policy,
    env_name,
    env_kwargs,
    num_epochs=10,
    num_steps=500,
) -> None:
    
    num_training_loops = 10
    num_rollouts = 500
    
    dataset = [] # log observations and actions

    actor_class = "SoftsignActor"

    dummy_env = make_env(env_name, **env_kwargs)
    student_actor = SoftsignActor(dummy_env)

    for epoch in range(num_epochs):
        
        observations = []
        for step in range(num_steps):
            observations = []
            if epoch == 1:
                _, action, _ = expert_policy.act(
                    # TODO: try with and without deterministic
                    observations[step], deterministic=False
                )
            else:
                _, action, _ = student_actor.act(observations[step])
            
                cpu_actions = action.cpu().numpy()

            # TODO: create envs
            obs, _, _, _ = envs.step(cpu_actions)

        for data in dataset:
            # query expert to append an expert action to each sample state, append to dataset
            ...
        
        # retrain learned policy with dataset (what algorithm)? supervised learning
        # https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/bc.html#BC.train