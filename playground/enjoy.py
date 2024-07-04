"""
Helper script used for rendering the best learned policy for an existing experiment.

Usage:
```bash
python enjoy.py --env <ENV> --dir
python enjoy.py --env <ENV> --net <PATH/TO/NET> --len <STEPS>
```
"""
import argparse
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch

from bottleneck import nanmean

import mocca_envs
from common.controller import MixedActor
from common.envs_utils import make_env
from common.misc_utils import EpisodeRunner

import numpy as np
DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


def main():
    import numpy as np
    parser = argparse.ArgumentParser(
        description=(
            "Examples:\n"
            "   python enjoy.py --env <ENV> --net <NET>\n"
            "   (Remote) python enjoy.py --env <ENV> --net <NET> --len 1000 --render 0 --save 1\n"
            "   (Faster) python enjoy.py --env <ENV> --net <NET> --len 1000 --save 1 --ffmpeg 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--curriculum", type=int, default=None)
    parser.add_argument("--len", type=int, default=float("inf"))
    parser.add_argument("--plank_class", type=str, default="VeryLargePlank")
    parser.add_argument("--plot", type=int, default=1)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--ffmpeg", type=int, default=0)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    # Save options:
    #   1) render=True ffmpeg=False -> Dump frame by frame using getCameraImage, high quality
    #   2) render=True ffmpeg=True -> Use pybullet.connect option
    #   3) render=False -> Use EGL and getCameraImage
    use_egl = args.save and not args.render
    use_ffmpeg = args.render and args.ffmpeg
    env = make_env(
        args.env,
        render=args.render,
        plank_class=args.plank_class,
        use_egl=use_egl,
        use_ffmpeg=use_ffmpeg,
    )
    env._max_episode_steps = float("inf")
    env.seed(1093)

    model_path = args.net or os.path.join(args.save_dir, f"{args.env}_latest.pt")

    print("Env: {}".format(args.env))
    print("Model: {}".format(os.path.basename(model_path)))

    actor_critic = torch.load(model_path, map_location=torch.device('cpu'))
    actor = actor_critic.actor

    if type(actor) == MixedActor and args.plot:
        from mocca_utils.plots.visplot import Figure, TimeSeriesPlot
        import matplotlib.cm as mpl_color
        import numpy as np

        fig = Figure(size=(400, 400), decorate=True, title="Expert Activations")
        ts_plot = TimeSeriesPlot(
            figure=fig,
            tile_rows=slice(0, 1),
            tile_cols=slice(0, 1),
            ylim=[-0.2, 1.2],
            window_size=500,
            num_lines=actor.num_experts + 2,
            y_axis_options={},
        )

        cmap = mpl_color.get_cmap("tab20")
        colours = cmap(np.linspace(0, 1, actor.num_experts))

    # Set global no_grad
    torch.set_grad_enabled(False)

    runner_options = {
        "save": args.save,
        "use_ffmpeg": use_ffmpeg,
        "max_steps": args.len,
        "csv": args.csv,
    }

    with EpisodeRunner(env, **runner_options) as runner:

        max_curriculum = getattr(env.unwrapped, "max_curriculum", 10)
        curriculum = args.curriculum if args.curriculum is not None else max_curriculum
        env.set_env_params({"curriculum": int(curriculum)})

        obs = env.reset()
        env.camera._cam_yaw = 90
        ep_reward = 0

        controller = actor_critic.actor

        done = False

        while not runner.done:
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            if type(actor) == MixedActor and args.plot:
                action, expert_activations = controller.forward_with_activations(obs)
                for eid, (a, c) in enumerate(zip(expert_activations, colours)):
                    ts_plot.add_point(float(a), eid, {"color": c, "width": 2})
                ts_plot.add_point(0, len(expert_activations))
                ts_plot.add_point(1, len(expert_activations) + 1)
                ts_plot.redraw()
            else:
                action = controller(obs)

            cpu_actions = action.squeeze().cpu().numpy()
            obs, reward, done, _ = env.step(cpu_actions)
            env.camera.lookat(env.robot.body_xyz)
            ep_reward += reward

            if done:
                print(f"--- Episode reward: {ep_reward}")
                obs = env.reset(reset_runner=False)
                ep_reward = 0


if __name__ == "__main__":
    main()
