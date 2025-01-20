'''
python3 -m playground.train_dagger --env Walker3DStepperEnv-v0 --student_net runs/dream/jan_13/only_reset_actor_cont_threshold_change/models/Walker3DStepperEnv-v0_375000000.pt --net runs/dream/sep_4/timing_w_hopping_cont_gpu/models/Walker3DStepperEnv-v0_curr_1_4.pt --num_epochs 40
'''

import argparse
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import numpy as np
import torch
import mocca_envs
from common.controller import SoftsignActor, Policy
from algorithms.dagger import train

from common.envs_utils import (
    make_vec_envs, make_env
)

def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--student_net", type=str, required=False)
    parser.add_argument("--seed", type=int, default=1093)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_processes", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=5000)
    args = parser.parse_args()

    env_kwargs = {
        "plank_class": "VeryLargePlank",
        "heading_bonus_weight": 8,
        "gauss_width": 12,
        "timing_bonus_weight": 1.5,
        "start_curriculum": 9,
        "start_behavior_curriculum": 12,
        "curriculum": 9,
        "behavior_curriculum": 12,
        "determine": True,
    }

    env_kwargs_normal = {
        "plank_class": "VeryLargePlank",
        "heading_bonus_weight": 8,
        "gauss_width": 12,
        "timing_bonus_weight": 1.5,
        "start_curriculum": 9,
        "start_behavior_curriculum": 10,
        "curriculum": 9,
        "behavior_curriculum": 10,
        "determine": True,
    }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(1)

    env_per_task_kwargs = [env_kwargs, env_kwargs_normal]

    envs_per_task = [
        make_vec_envs(
            args.env, args.seed, args.num_processes, None, **env_per_task_kwargs[i]
        )
        for i in range(2)
    ]
 
    dummy_env = make_env(args.env, **env_per_task_kwargs[0])

    # envs_per_task = [
    #     make_env(args.env, seed=args.seed, **env_per_task_kwargs[i])
    #     # make_vec_envs(
    #     #     env_name, seed, num_processes, None, **env_per_task_kwargs[i]
    #     # )
    #     for i in range(2)
    # ]

    try:
        controller = SoftsignActor(dummy_env)
        actor_critic = Policy(controller)
        actor_critic.load_state_dict(torch.load(args.net, map_location=torch.device(device)))
    except:
        actor_critic = torch.load(args.net, map_location=torch.device(device))

    if args.student_net is not None:
        try:
            controller = SoftsignActor(dummy_env)
            actor_critic_student = Policy(controller)
            actor_critic_student.load_state_dict(torch.load(args.student_net, map_location=torch.device(device)))
        except:
            actor_critic_student = torch.load(args.student_net, map_location=torch.device(device))
    else:
        actor_critic_student = None

    train(
        actor_critic,
        actor_critic_student,
        envs_per_task,
        [env_kwargs, env_kwargs_normal],
        device=device,
        seed=args.seed,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        num_processes=args.num_processes
    )

if __name__ == "__main__":
    main()