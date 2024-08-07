import argparse
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import numpy as np
import torch
import mocca_envs

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
    args = parser.parse_args()

    env_kwargs = {
        "plank_class": args.plank_class,
        "heading_bonus_weight": 8,
        "gauss_width": 10,
        "timing_bonus_weight": 2,
        "start_curriculum": 0,
        "cycle_time": 60,
        "start_behavior_curriculum": 0,
        "foot_angle_weight": 0.2,
    }

    envs = make_vec_envs(
        args.env, args.seed, 4, log_dir="runs/dagger", **env_kwargs
    )

    model_path = args.net

    actor_critic = torch.load(model_path, map_location=torch.device('cpu'))

    


if __name__ == "__main__":
    main()