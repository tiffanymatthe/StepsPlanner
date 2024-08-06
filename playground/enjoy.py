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
    parser.add_argument("--behavior_curriculum", type=int, default=None)
    parser.add_argument("--len", type=int, default=float("inf"))
    parser.add_argument("--plank_class", type=str, default="VeryLargePlank")
    parser.add_argument("--heading_bonus_weight", type=float, default=1.0)
    parser.add_argument("--cycle_time", type=int, default=60)
    parser.add_argument("--plot", type=int, default=1)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--heading", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--determine", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--timing", default=False, action=argparse.BooleanOptionalAction)
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
        heading_bonus_weight=args.heading_bonus_weight,
        determine=args.determine,
        use_egl=use_egl,
        use_ffmpeg=use_ffmpeg,
        cycle_time=args.cycle_time,
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
        max_behavior_curriculum = getattr(env.unwrapped, "max_behavior_curriculum", 10)
        behavior_curriculum = args.behavior_curriculum if args.behavior_curriculum is not None else max_behavior_curriculum
        env.set_env_params({"curriculum": int(curriculum), "behavior_curriculum": int(behavior_curriculum)})

        obs = env.reset()
        env.camera._cam_yaw = 90
        ep_reward = 0

        left_foot_headings = []
        right_foot_headings = []
        left_foot_positions = []
        right_foot_positions = []
        target_indices = []

        expected_start_foot = []
        expected_other_foot = []
        actual_start_foot = []
        actual_other_foot = []

        foot_heading_targets = env.terrain_info[:, 6]
        foot_position_targets = env.terrain_info[:, 0:2]
        swing_targets = env.terrain_info[:, 7]

        controller = actor_critic.actor

        done = False

        fig, axs = plt.subplots(1)
        fig.suptitle('Timing')
        axs.plot(env.start_leg_expected_contact_probabilities)
        axs.plot(env.other_leg_expected_contact_probabilities)
        plt.show()

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

            if args.heading:
                left_foot_headings.append(env.robot.feet_rpy[1, 2])
                left_foot_positions.append(env.robot.feet_xyz[1, 0:2])
                right_foot_headings.append(env.robot.feet_rpy[0, 2])
                right_foot_positions.append(env.robot.feet_xyz[0, 0:2])
                target_indices.append(env.next_step_index)

            cpu_actions = action.squeeze().cpu().numpy()
            obs, reward, done, _ = env.step(cpu_actions)
            env.camera.lookat(env.robot.body_xyz)
            ep_reward += reward

            if args.timing:
                expected_start_foot.append(env.start_expected_contact)
                expected_other_foot.append(env.other_expected_contact)
                actual_start_foot.append(env.robot.feet_contact[env.starting_leg])
                actual_other_foot.append(env.robot.feet_contact[1-env.starting_leg])

            if done:
                if args.heading:
                    # plt.plot(heading_targets, label="Target")
                    # # avg_heading = ema(np.array(robot_headings))
                    # # plt.plot(avg_heading, label="Robot Heading EMA")
                    # plt.plot(robot_headings, label="Actual")
                    # # plt.plot(butt_headings, label="butt heading")
                    # # plt.plot(l_headings, label="left foot heading")
                    # # plt.plot(r_headings, label="right foot heading")
                    # # plt.plot(np.array(l_headings)/2 + np.array(r_headings)/2, label="Avg foot heading")
                    # steps = np.multiply(reached_steps, robot_headings)
                    # steps[steps==0] = np.nan
                    # plt.plot(steps, 'o', mfc='none', label="step reached")
                    # plt.legend()
                    # steps = np.ma.array(steps, mask=np.isnan(steps))
                    # heading_targets = np.ma.array(heading_targets, mask=np.isnan(steps))
                    # mse = np.square(steps - heading_targets).mean()
                    # plt.title(f"MSE at steps: {mse}")
                    # plt.show()
                    # when target indices switches, extract position and index and plot
                    target_change_mask = np.roll(target_indices, 1)
                    target_change_mask[0] = target_indices[0]
                    target_change_mask = target_indices != target_change_mask
                    swing_legs_long = swing_targets[target_indices]
                    left_mask = np.logical_and(swing_legs_long == 1, target_change_mask)
                    right_mask = np.logical_and(swing_legs_long == 0, target_change_mask)
                    plt.quiver(*zip(*np.array(left_foot_positions)[left_mask]), np.cos(np.array(left_foot_headings)[left_mask]), np.sin(np.array(left_foot_headings)[left_mask]), color='red')
                    plt.quiver(*zip(*np.array(right_foot_positions)[right_mask]), np.cos(np.array(right_foot_headings)[right_mask]), np.sin(np.array(right_foot_headings)[right_mask]), color='blue')
                    plt.scatter(*zip(*np.array(left_foot_positions)), color='red', alpha=0.1)
                    plt.scatter(*zip(*np.array(right_foot_positions)), color='blue', alpha=0.1)
                    plt.quiver(*zip(*foot_position_targets), np.cos(foot_heading_targets), np.sin(foot_heading_targets), color="green")
                    plt.axis('square')
                    plt.show()
                    
                    left_foot_headings = []
                    right_foot_headings = []
                    left_foot_positions = []
                    right_foot_positions = []
                    target_indices = []
                if args.timing:
                    fig, axs = plt.subplots(2)
                    fig.suptitle('Timing')
                    axs[0].plot(expected_start_foot)
                    axs[1].plot(expected_other_foot)
                    axs[0].plot(actual_start_foot)
                    axs[1].plot(actual_other_foot)
                    plt.show()
                    expected_start_foot = []
                    expected_other_foot = []
                    actual_start_foot = []
                    actual_other_foot = []
                print(f"--- Episode reward: {ep_reward} and average heading error: {nanmean(env.heading_errors) * RAD2DEG:.2f} deg and timing acc: {nanmean(env.timing_count_errors):.2f}, {nanmean(env.met_times):.2f}")
                obs = env.reset(reset_runner=False)
                if args.heading:
                    foot_heading_targets = env.terrain_info[:, 6]
                    foot_position_targets = env.terrain_info[:, 0:2]
                    swing_targets = env.terrain_info[:, 7]
                ep_reward = 0


if __name__ == "__main__":
    main()
