"""
Helper script used for rendering the best learned policy for an existing experiment.

Usage:
```bash
python -m playground.enjoy with experiment_dir=runs/<EXPERIMENT_DIRECTORY>
or
python enjoy.py with env=<ENV> net=<PATH/TO/NET> len=<STEPS>
```
"""

import os
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch

from common.envs_utils import make_env
from common.misc_utils import EpisodeRunner
from common.sacred_utils import ex


@ex.config
def config():
    csv = None
    env = ""
    len = float("inf")
    net = None
    save = False
    experiment_dir = "."
    # loads saved configs
    config_file = os.path.join(experiment_dir, "configs.json")
    if os.path.exists(config_file):
        ex.add_config(config_file)


@ex.automain
def main(_config):
    args = SimpleNamespace(**_config)
    assert args.env != ""

    env = make_env(args.env, render=True)
    env.seed(1093)

    model_path = args.net or os.path.join(args.save_dir, f"{args.env}_best.pt")

    print("Env: {}".format(args.env))
    print("Model: {}".format(os.path.basename(model_path)))

    actor_critic = torch.load(model_path)

    # Set global no_grad
    torch._C.set_grad_enabled(False)

    with EpisodeRunner(env, save=args.save, max_steps=args.len, csv=args.csv) as runner:
        obs = env.reset()
        ep_reward = 0

        while not runner.done:
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            value, action, _ = actor_critic.act(obs, deterministic=True)
            cpu_actions = action.squeeze().cpu().numpy()

            obs, reward, done, _ = env.step(cpu_actions)
            ep_reward += reward

            if done:
                print("--- Episode reward:", ep_reward)
                ep_reward = 0
                obs = env.reset(reset_runner=False)
