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
import matplotlib
matplotlib.use('Qt5Agg')
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

    if args.plot:
        fig1, ax1 = plt.subplots(figsize=(12,4))

        # ax1.set_xlim(0, 800)
        # ax1.set_ylim(0,2.2)

        # ax1.set_xlim(0, 60)
        # ax1.set_ylim(-0.2, 1.2)
        
        plt.show(block=False)
        plt.draw()
        background_1 = fig1.canvas.copy_from_bbox(ax1.bbox)
        actual_points_left = ax1.plot([0,1], [0,1], '-', color="slateblue", linewidth=4, animated=True)[0]
        actual_points_right = ax1.plot([0,1], [0,1], '-', color="turquoise", linewidth=4, animated=True)[0]
        actual_x_left = []
        actual_y_left = []
        actual_x_right = []
        actual_y_right = []
        ax1.draw_artist(actual_points_left)
        ax1.draw_artist(actual_points_right)

    # Set global no_grad
    torch.set_grad_enabled(False)
    
    writer = None
    if args.save and args.plot:
        import matplotlib.animation as animation
        writer = animation.FFMpegWriter(fps=1/env.control_step, bitrate=1800)
        filename = os.path.join(parent_dir, "videos", f"{args.behavior_curriculum}_{args.curriculum}_plot.mp4")
        writer.setup(fig1, filename, dpi=100)

    runner_options = {
        "save": args.save,
        "use_ffmpeg": use_ffmpeg,
        "max_steps": args.len,
        "csv": args.csv,
        "ax": ax1 if args.plot else None,
        "dir": os.path.join(parent_dir, "videos"),
        "video_filename": f"{args.behavior_curriculum}_{args.curriculum}_rgb.mp4",
        "plot_writer": writer,
    }

    with EpisodeRunner(env, **runner_options) as runner:

        max_curriculum = getattr(env.unwrapped, "max_curriculum", 10)
        curriculum = args.curriculum if args.curriculum is not None else max_curriculum
        max_behavior_curriculum = getattr(env.unwrapped, "max_behavior_curriculum", 10)
        behavior_curriculum = args.behavior_curriculum if args.behavior_curriculum is not None else max_behavior_curriculum
        env.set_env_params({"curriculum": int(curriculum), "behavior_curriculum": int(behavior_curriculum)})

        obs = env.reset(force=True)
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
        index_switch = []

        foot_heading_targets = env.terrain_info[:, 6]
        foot_position_targets = env.terrain_info[:, 0:2]
        swing_targets = env.terrain_info[:, 7]

        controller = actor_critic.actor

        done = False

        if args.plot:
            if env.mask_info["timing"][2]:
                ax1.clear()
                ax1.set_xlim(0, 800)
                ax1.set_ylim(0,2.2)
                # MUST DO AFTER CLEARING!!!
                actual_points_left = ax1.plot([0,1], [0,1], '-', color="slateblue", linewidth=4, animated=True)[0]
                actual_points_right = ax1.plot([0,1], [0,1], '-', color="turquoise", linewidth=4, animated=True)[0]
            else:
                time_offsets = [0]
                times_left = []
                all_sets_left = []
                times_right = []
                all_sets_right = []
                for current_step in range(1,env.num_steps):
                    time_left = list(np.array(range(int(env.terrain_info[current_step, 8] + env.terrain_info[current_step, 9]) + env.step_delay)) + time_offsets[current_step - 1])
                    sets_left = [1 for _ in range(int(env.terrain_info[current_step, 8]))] + [0 for _ in range(int(env.terrain_info[current_step, 9]))] + [1 for _ in range(env.step_delay)]
                    time_right = list(np.array(range(int(env.terrain_info[current_step, 10] + env.terrain_info[current_step, 11]) + env.step_delay)) + time_offsets[current_step - 1])
                    sets_right = [1 for _ in range(int(env.terrain_info[current_step, 10]))] + [0 for _ in range(int(env.terrain_info[current_step, 11]))] + [1 for _ in range(env.step_delay)]
                    if env.terrain_info[current_step, 7] == 0:
                        time_left, sets_left, time_right, sets_right = time_right, sets_right, time_left, sets_left
                    times_left += time_left
                    all_sets_left += sets_left
                    times_right += time_right
                    all_sets_right += sets_right
                    time_offsets.append(time_left[-1])
                ax1.fill_between(times_left, y1=1, y2=0, where=all_sets_left, color='steelblue', step='post')
                ax1.fill_between(times_right, y1=2.2, y2=1.2, where=all_sets_right, color='paleturquoise', step='post')
            fig1.canvas.draw()
            background_1 = fig1.canvas.copy_from_bbox(ax1.bbox)

        while not runner.done:
            obs = torch.from_numpy(obs).float().unsqueeze(0)
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
                expected_start_foot.append(env.left_expected_contact)
                expected_other_foot.append(env.right_expected_contact)
                actual_start_foot.append(env.left_actual_contact)
                actual_other_foot.append(env.right_actual_contact)
                index_switch.append(env.current_step_time == 0)

            if args.plot and not env.past_last_step:
                fig1.canvas.restore_region(background_1)
                if env.mask_info["timing"][2]:
                    time = env.timestep
                else:
                    time = env.current_step_time + time_offsets[env.current_time_index-1]
                actual_y_left.append(env.left_actual_contact)
                actual_y_right.append(env.right_actual_contact + 1.2)
                actual_x_left.append(time)
                actual_x_right.append(time)

                if time > ax1.get_xlim()[1]:
                    ax1.set_xlim(0, time + 50)
                    fig1.canvas.draw()

                actual_points_left.set_data(actual_x_left, actual_y_left)
                actual_points_right.set_data(actual_x_right, actual_y_right)
                
                ax1.draw_artist(actual_points_left)
                ax1.draw_artist(actual_points_right)

                fig1.canvas.blit(ax1.bbox)

            if done:
                if args.heading:
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
                    fig, axs = plt.subplots(4)
                    fig.suptitle('Timing')
                    axs[0].plot(expected_start_foot, label="Expected Left Foot")
                    axs[1].plot(expected_other_foot)
                    axs[0].plot(actual_start_foot, label="Actual Left Foot")
                    axs[1].plot(actual_other_foot)
                    axs[2].plot(index_switch)
                    left_leg_contacts = []
                    right_leg_contacts = []
                    for i in range(env.num_steps):
                        if env.terrain_info[i, 7] == 1:
                            left_leg_contacts.extend([1] * int(env.terrain_info[i, 8]))
                            left_leg_contacts.extend([0] * int(env.terrain_info[i, 9]))
                            right_leg_contacts.extend([1] * int(env.terrain_info[i, 10]))
                            right_leg_contacts.extend([0] * int(env.terrain_info[i, 11]))
                        else:
                            right_leg_contacts.extend([1] * int(env.terrain_info[i, 8]))
                            right_leg_contacts.extend([0] * int(env.terrain_info[i, 9]))
                            left_leg_contacts.extend([1] * int(env.terrain_info[i, 10]))
                            left_leg_contacts.extend([0] * int(env.terrain_info[i, 11]))
                        axs[3].axvline(x=len(left_leg_contacts))
                    axs[3].plot(left_leg_contacts, label="left")
                    axs[3].plot(right_leg_contacts)
                    axs[3].legend()
                    axs[0].legend()
                    plt.show()
                    expected_start_foot = []
                    expected_other_foot = []
                    actual_start_foot = []
                    actual_other_foot = []
                    index_switch = []
                print(f"--- Episode reward: {ep_reward} and average heading error: {nanmean(env.heading_errors) * RAD2DEG:.2f} deg and timing acc: {nanmean(env.met_times):.2f}")
                obs = env.reset(reset_runner=False)
                if args.heading:
                    foot_heading_targets = env.terrain_info[:, 6]
                    foot_position_targets = env.terrain_info[:, 0:2]
                    swing_targets = env.terrain_info[:, 7]
                if args.plot:
                    if env.mask_info["timing"][2]:
                        ax1.clear()
                        ax1.set_xlim(0, 800)
                        ax1.set_ylim(0,2.2)
                    else:
                        time_offsets = [0]
                        times_left = []
                        all_sets_left = []
                        times_right = []
                        all_sets_right = []
                        for current_step in range(1,env.num_steps):
                            time_left = list(np.array(range(int(env.terrain_info[current_step, 8] + env.terrain_info[current_step, 9]) + env.step_delay)) + time_offsets[current_step - 1])
                            sets_left = [1 for _ in range(int(env.terrain_info[current_step, 8]))] + [0 for _ in range(int(env.terrain_info[current_step, 9]))] + [1 for _ in range(env.step_delay)]
                            time_right = list(np.array(range(int(env.terrain_info[current_step, 10] + env.terrain_info[current_step, 11]) + env.step_delay)) + time_offsets[current_step - 1])
                            sets_right = [1 for _ in range(int(env.terrain_info[current_step, 10]))] + [0 for _ in range(int(env.terrain_info[current_step, 11]))] + [1 for _ in range(env.step_delay)]
                            if env.terrain_info[current_step, 7] == 0:
                                time_left, sets_left, time_right, sets_right = time_right, sets_right, time_left, sets_left
                            times_left += time_left
                            all_sets_left += sets_left
                            times_right += time_right
                            all_sets_right += sets_right
                            time_offsets.append(time_left[-1])
                        ax1.clear()
                        ax1.fill_between(times_left, y1=1, y2=0, where=all_sets_left, color='steelblue', step='post')
                        ax1.fill_between(times_right, y1=2.2, y2=1.2, where=all_sets_right, color='paleturquoise', step='post')
                    # REQUIRED AFTER CLEARING
                    actual_points_left = ax1.plot([0,1], [0,1], '-', color="slateblue", linewidth=4, animated=True)[0]
                    actual_points_right = ax1.plot([0,1], [0,1], '-', color="turquoise", linewidth=4, animated=True)[0]
                    actual_x_left, actual_y_left, actual_x_right, actual_y_right = [], [], [], []
                    fig1.canvas.draw()
                    background_1 = fig1.canvas.copy_from_bbox(ax1.bbox)
                ep_reward = 0

        if args.save and args.plot:
            writer.finish()

if __name__ == "__main__":
    main()
