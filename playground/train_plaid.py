from algorithms.plaid import Distiller
import torch
from common.envs_utils import make_env
from common.controller import SoftsignActor, Policy
import mocca_envs

def load_net(net_path, device, actor_class, dummy_env):
    # net_path has .pt extension
    controller = actor_class(dummy_env)
    actor_critic = Policy(controller)
    actor_critic.load_state_dict(torch.load(net_path, map_location=torch.device(device)))
    if not hasattr(actor_critic, 'feature_keys'):
        actor_critic.setup_feature_logging()
        actor_critic.actor.setup_feature_logging()

    return actor_critic

if __name__ == "__main__":
    env_name = "Walker3DStepperEnv-v0"
    seed = 16
    plank_class = "VeryLargePlank"
    heading_bonus_weight = 8
    timing_bonus_weight = 1.5
    gauss_width = 12
    foot_angle_weight = 0.1
    determine = 3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    prev_net_path = "runs/dream/nov_8/plaid_plasticity_elaho_new_5/models/Walker3DStepperEnv-v0_curr_distilled_4_9.pt"
    curr_net_path = "runs/dream/nov_8/plaid_plasticity_elaho_new_5/models/Walker3DStepperEnv-v0_curr_5_9.pt"

    env_kwargs = {
        "plank_class": plank_class,
        "heading_bonus_weight": heading_bonus_weight,
        "gauss_width": gauss_width,
        "timing_bonus_weight": timing_bonus_weight,
        "start_curriculum": 0,
        "start_behavior_curriculum": 0,
        "foot_angle_weight": foot_angle_weight,
        "from_net": False,
        "determine": determine,
    }

    dummy_env = make_env(env_name, **env_kwargs)

    distiller = Distiller(
        env_name=env_name,
        base_env_kwargs=env_kwargs,
        seed=seed,
        device=device,
        num_processes=10,
        num_epochs=300,
        envs=None,
        dummy_env=dummy_env,
        log_dir=""
    )

    prev_behavior_actor_critic = load_net(prev_net_path, device, SoftsignActor, dummy_env)
    prev_behavior_actor_critic.to(device)

    actor_critic = load_net(curr_net_path, device, SoftsignActor, dummy_env)
    actor_critic.to(device)

    distiller.distill_policies(prev_behavior_actor_critic, actor_critic, 9, 4, 9, 5, False)

    torch.save(actor_critic.state_dict(), "plaid_distilled_5_9.pt")