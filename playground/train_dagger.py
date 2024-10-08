'''
python3 -m playground.train_dagger --env Walker3DStepperEnv-v0 --net runs/dream/sep_30/timing_50_simplifed_cont_c9/models/Walker3DStepperEnv-v0_curr_10_9.pt --student_net runs/dream/sep_4/timing_w_hopping_cont_gpu/models/Walker3DStepperEnv-v0_curr_1_4.pt
'''

import argparse
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import numpy as np
import torch
import mocca_envs
from algorithms.dagger import train

from common.envs_utils import (
    make_vec_envs,
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
        "start_curriculum": 0,
        "start_behavior_curriculum": 11,
        "determine": True,
    }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(1)


    actor_critic = torch.load(args.net, map_location=torch.device(device))
    if args.student_net is not None:
        actor_critic_student = torch.load(args.student_net, map_location=torch.device(device))
    else:
        actor_critic_student = None

    train(
        actor_critic,
        actor_critic_student,
        args.env,
        env_kwargs,
        device=device,
        seed=args.seed,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        num_processes=args.num_processes
    )

if __name__ == "__main__":
    main()