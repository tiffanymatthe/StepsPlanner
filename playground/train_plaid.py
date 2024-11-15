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
    determine = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    experts = {
        "runs/dream/oct_14/plasticity_elaho/models/Walker3DStepperEnv-v0_curr_0_9.pt": 0,
        "runs/dream/oct_14/plasticity_elaho/models/Walker3DStepperEnv-v0_curr_1_9.pt": 1,
        "runs/dream/oct_15/plasticity_elaho_cont_lowered_threshold_fixed/models/Walker3DStepperEnv-v0_curr_2_9.pt": 2,
        "runs/dream/oct_15/plasticity_elaho_cont_lowered_threshold_fixed/models/Walker3DStepperEnv-v0_curr_3_9.pt": 3,
        "runs/dream/oct_15/plasticity_elaho_cont_lowered_threshold_fixed/models/Walker3DStepperEnv-v0_curr_4_9.pt": 4,
        "runs/dream/oct_18/plasticity_elaho_cont_lowered_threshold_fixed_cont/models/Walker3DStepperEnv-v0_curr_5_9.pt": 5,
        "runs/dream/oct_18/plasticity_elaho_cont_lowered_threshold_fixed_cont/models/Walker3DStepperEnv-v0_curr_6_9.pt": 6,
        "runs/dream/oct_18/plasticity_elaho_cont_lowered_threshold_fixed_cont/models/Walker3DStepperEnv-v0_curr_7_9.pt": 7,
        "runs/dream/oct_18/plasticity_elaho_cont_lowered_threshold_fixed_cont/models/Walker3DStepperEnv-v0_curr_8_9.pt": 8,
        "runs/dream/oct_18/plasticity_elaho_cont_lowered_threshold_fixed_cont/models/Walker3DStepperEnv-v0_curr_9_8.pt": 9,
        "runs/dream/oct_19/plasticity_elaho_cont_one_step_plant/models/Walker3DStepperEnv-v0_curr_10_8.pt": 10,
    }

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
        num_experts=len(experts),
        num_epochs=300,
        dummy_env=dummy_env,
        log_dir=""
    )

    actor_critics = []
    for policy_path in experts.keys():
        actor_critic = load_net(policy_path, device, SoftsignActor, dummy_env)
        actor_critic.to(device)
        actor_critics.append(actor_critic)

    distilled_policy = distiller.distill_policies(actor_critics, list(experts.values()))

    torch.save(distilled_policy.state_dict(), "plaid_distilled_all.pt")