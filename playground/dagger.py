import argparse
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import numpy as np
import torch
import mocca_envs
from algorithms.dagger_train import train

from common.envs_utils import (
    make_vec_envs,
)

def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--plank_class", type=str, default="VeryLargePlank")
    parser.add_argument("--seed", type=int, default=1093)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_processes", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=5000)
    args = parser.parse_args()

    env_kwargs = {
        "plank_class": args.plank_class,
        "heading_bonus_weight": 8,
        "gauss_width": 10,
        "timing_bonus_weight": 2,
        "start_curriculum": 0,
        "cycle_time": 60,
        "start_behavior_curriculum": 0,
        "foot_angle_weight": 0.1,
    }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(1)
    
    actor_critic = torch.load(args.net, map_location=torch.device(device))

    train(
        actor_critic,
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