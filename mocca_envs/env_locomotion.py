from math import sin, cos, atan2, sqrt
import os

from bottleneck import ss, anynan, nanargmax, nanargmin, nanmin, nanmean, nansum
import gym
import numpy as np
from numpy import concatenate
import pybullet
import torch

from mocca_envs.env_base import EnvBase
from mocca_envs.bullet_objects import (
    VSphere,
    VCylinder,
    VArrow,
    VBox,
    Pillar,
    Plank,
    LargePlank,
    VeryLargePlank,
    HeightField,
    MonkeyBar,
)
from mocca_envs.robots import Child3D, Laikago, Mike, Monkey3D, Walker2D, Walker3D

Colors = {
    "dodgerblue": (0.11764705882352941, 0.5647058823529412, 1.0, 1.0),
    "crimson": (0.8627450980392157, 0.0784313725490196, 0.23529411764705882, 1.0),

}

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class Walker3DCustomEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    robot_class = Walker3D
    termination_height = 0.7
    robot_random_start = True
    robot_init_position = None
    robot_init_velocity = None

    def __init__(self, **kwargs):
        super().__init__(self.robot_class, **kwargs)
        self.robot.set_base_pose(pose="running_start")

        # Fix-ordered Curriculum, dummies
        self.curriculum = 1
        self.max_curriculum = 1
        self.advance_threshold = 1

        self.electricity_cost = 4.5 / self.robot.action_space.shape[0]
        self.stall_torque_cost = 0.225 / self.robot.action_space.shape[0]
        self.joints_at_limit_cost = 0.1

        R = self.robot.observation_space.shape[0]
        high = np.inf * np.ones(R + 2, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def create_target(self):
        # Need this to create target in render mode, called by EnvBase
        # Sphere is a visual shape, does not interact physically
        self.target = VSphere(self._p, radius=0.15, pos=None)

    def randomize_target(self):
        self.dist = self.np_random.uniform(3, 5)
        self.angle = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        self.stop_frames = self.np_random.choice([30.0, 60.0])

    def get_observation_components(self):
        sin_ = self.distance_to_target * sin(self.angle_to_target)
        sin_ = sin_ / (1 + abs(sin_))
        cos_ = self.distance_to_target * cos(self.angle_to_target)
        cos_ = cos_ / (1 + abs(cos_))
        return self.robot_state, [sin_, cos_]

    def reset(self):
        self.done = False
        self.randomize_target()

        self.walk_target = np.fromiter(
            [self.dist * cos(self.angle), self.dist * sin(self.angle), 1.0],
            dtype=np.float32,
        )
        self.close_count = 0

        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start,
            pos=self.robot_init_position,
            vel=self.robot_init_velocity,
        )

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)

        self.calc_potential()

        # call after on calc_potential and robot.calc_state()
        state = concatenate(self.get_observation_components())

        return state

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()

        self.robot_state = self.robot.calc_state(self.ground_ids)
        self.calc_env_state(action)

        reward = self.progress + self.target_bonus - self.energy_penalty
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        # call after on calc_potential and robot.calc_state()
        state = concatenate(self.get_observation_components())

        if self.is_rendered or self.use_egl:
            self._handle_keyboard()
            # self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)
            self.target.set_color(
                Colors["dodgerblue"]
                if self.distance_to_target < 0.15
                else Colors["crimson"]
            )

        return state, reward, self.done, {}

    def calc_potential(self):

        walk_target_theta = atan2(
            self.walk_target[1] - self.robot.body_xyz[1],
            self.walk_target[0] - self.robot.body_xyz[0],
        )
        walk_target_delta = self.walk_target - self.robot.body_xyz

        self.angle_to_target = walk_target_theta - self.robot.body_rpy[2]
        self.distance_to_target = sqrt(ss(walk_target_delta[0:2]))

        self.linear_potential = -self.distance_to_target / self.scene.dt

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential
        self.calc_potential()
        linear_progress = self.linear_potential - old_linear_potential

        self.progress = linear_progress

        self.posture_penalty = 0
        if not -0.2 < self.robot.body_rpy[1] < 0.4:
            self.posture_penalty = abs(self.robot.body_rpy[1])

        if not -0.4 < self.robot.body_rpy[0] < 0.4:
            self.posture_penalty += abs(self.robot.body_rpy[0])

        electricity_cost = self.electricity_cost * nansum(
            abs(action * self.robot.joint_speeds)
        )
        stall_torque_cost = self.stall_torque_cost * ss(action)
        self.energy_penalty = electricity_cost + stall_torque_cost

        self.joints_penalty = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )

        # Calculate done
        self.tall_bonus = 2.0 if self.robot_state[0] > self.termination_height else -1.0
        self.done = self.done or self.tall_bonus < 0

    def calc_target_reward(self):
        self.target_bonus = 0
        if self.distance_to_target < 0.15:
            self.close_count += 1
            self.target_bonus = 2

    def calc_env_state(self, action):
        if anynan(self.robot_state):
            print("~INF~", self.robot_state)
            self.done = True

        # Order is important
        # calc_target_reward() potential
        self.calc_base_reward(action)
        self.calc_target_reward()

        if self.close_count >= self.stop_frames:
            self.close_count = 0
            self.randomize_target()
            delta = self.dist * np.fromiter(
                [cos(self.angle), sin(self.angle), 0.0], dtype=np.float32
            )
            self.walk_target += delta
            self.calc_potential()

    def get_mirror_indices(self):

        action_dim = self.robot.action_space.shape[0]
        # _ + 6 accounting for global
        right = self.robot._right_joint_indices + 6
        # _ + action_dim to get velocities, last one is right foot contact
        right = concatenate(
            (
                right,
                right + action_dim,
                [
                    6 + 2 * action_dim + 2 * i
                    for i in range(len(self.robot.foot_names) // 2)
                ],
            )
        )
        # Do the same for left
        left = self.robot._left_joint_indices + 6
        left = concatenate(
            (
                left,
                left + action_dim,
                [
                    6 + 2 * action_dim + 2 * i + 1
                    for i in range(len(self.robot.foot_names) // 2)
                ],
            )
        )

        # Used for creating mirrored observations

        negation_obs_indices = concatenate(
            (
                # vy, roll
                [2, 4],
                # negate part of robot (position)
                6 + self.robot._negation_joint_indices,
                # negate part of robot (velocity)
                6 + self.robot._negation_joint_indices + action_dim,
                # sin(x) component of target location
                [6 + 2 * action_dim + len(self.robot.foot_names)],
            )
        )
        right_obs_indices = right
        left_obs_indices = left

        # Used for creating mirrored actions
        negation_action_indices = self.robot._negation_joint_indices
        right_action_indices = self.robot._right_joint_indices
        left_action_indices = self.robot._left_joint_indices

        return (
            negation_obs_indices,
            right_obs_indices,
            left_obs_indices,
            negation_action_indices,
            right_action_indices,
            left_action_indices,
        )


class Walker2DCustomEnv(Walker3DCustomEnv):
    robot_class = Walker2D
    robot_init_position = [0, 0, 2]

    def reset(self):

        if self.state_id >= 0:
            self._p.restoreState(self.state_id)

        super().reset()

        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        state = concatenate((self.robot_state, [0], [0]))
        return state

    def step(self, action):
        state, reward, self.done, info = super().step(action)

        self.done = False
        if self.is_rendered or self.use_egl:
            self._handle_keyboard()

        return state, reward, self.done, info


class Child3DCustomEnv(Walker3DCustomEnv):

    robot_class = Child3D
    termination_height = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.robot.set_base_pose(pose="crawl")

    def calc_base_reward(self, action):
        super().calc_base_reward(action)


class Walker3DStepperEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4
    max_timestep = 1000

    robot_class = Walker3D
    robot_random_start = True
    robot_init_position = [
            [0, -0.3, 1.32], #backward
            [0, 0.3, 1.32]
        ]
    robot_init_velocity = None

    pre_lift_count = 1000
    ground_stay_count = 1500

    plank_class = VeryLargePlank  # Pillar, Plank, LargePlank
    num_steps = 4
    step_radius = 0.25
    rendered_step_count = 3
    init_step_separation = 0.75

    lookahead = 2
    lookbehind = 1
    walk_target_index = -1
    step_bonus_smoothness = 1
    stop_steps = [3,4,5] # [6, 7, 13, 14]

    def __init__(self, **kwargs):
        # Handle non-robot kwargs
        plank_name = kwargs.pop("plank_class", None)
        self.plank_class = globals().get(plank_name, self.plank_class)

        super().__init__(self.robot_class, remove_ground=True, **kwargs)
        self.robot.set_base_pose(pose="running_start")

        # Fix-ordered Curriculum
        self.curriculum = 0
        self.max_curriculum = 9
        self.advance_threshold = min(12, self.num_steps)  # steps_reached

        # Robot settings
        N = self.max_curriculum + 1
        self.terminal_height_curriculum = np.linspace(0.75, 0.45, N)
        self.applied_gain_curriculum = np.linspace(1.0, 1.2, N)
        self.angle_curriculum = np.linspace(0, np.pi / 2, N)
        self.electricity_cost = 4.5 / self.robot.action_space.shape[0]
        self.stall_torque_cost = 0.225 / self.robot.action_space.shape[0]
        self.joints_at_limit_cost = 0.1

        # Env settings
        self.next_step_index = self.lookbehind

        self.legs_bonus = 0

        # Terrain info
        self.dist_range = np.array([0.65, 1.25])
        self.pitch_range = np.array([-30, +30])  # degrees
        self.yaw_range = np.array([-20, 20])
        self.tilt_range = np.array([-15, 15])
        self.step_param_dim = 6
        # Important to do this once before reset!
        self.swing_leg = 0
        self.walk_forward = True
        self.terrain_info = self.generate_step_placements()

        # Observation and Action spaces
        self.robot_obs_dim = self.robot.observation_space.shape[0]
        K = self.lookahead + self.lookbehind
        high = np.inf * np.ones(
            self.robot_obs_dim + K * self.step_param_dim, dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

        # pre-allocate buffers
        F = len(self.robot.feet)
        self._foot_target_contacts = np.zeros((F, 1), dtype=np.float32)
        self.foot_dist_to_target = np.zeros(F, dtype=np.float32)

    def generate_step_placements(self):
        # Check just in case
        self.curriculum = min(self.curriculum, self.max_curriculum)
        ratio = self.curriculum / self.max_curriculum
        ratio = 0

        # {self.max_curriculum + 1} levels in total
        dist_upper = np.linspace(*self.dist_range, self.max_curriculum + 1)
        dist_range = np.array([self.dist_range[0], dist_upper[self.curriculum]])
        # dist_range = dist_range * 0 + 0.45
        yaw_range = self.yaw_range * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        N = self.num_steps
        dr = self.np_random.uniform(*dist_range, size=N)
        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        # dr[1] = self.init_step_separation
        dphi[1:3] = 0.0
        dtheta[1:3] = np.pi / 2

        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        dphi = np.cumsum(dphi)

        heading_targets = np.copy(dphi) * 0 + 90 * DEG2RAD

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        if not self.walk_forward:
            dy *= -1

        # # Fix overlapping steps
        # dx_max = np.maximum(np.abs(dx[2:]), self.step_radius * 2.5)
        # dx[2:] = np.sign(dx[2:]) * np.minimum(dx_max, self.dist_range[1])

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        x_temp = np.copy(x)
        y_temp = np.copy(y)

        self.swing_legs = np.ones(N, dtype=np.int8)

        sep_dist = 0.16

        # Assuming N is defined somewhere in the class or passed as an argument CHATGPT
        N_half = N // 2
        angles = dphi[:N_half]

        # Calculate shifts
        left_shifts = np.array([np.cos(angles + np.pi / 2), np.sin(angles + np.pi / 2)]) * sep_dist
        right_shifts = np.array([np.cos(angles - np.pi / 2), np.sin(angles - np.pi / 2)]) * sep_dist

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        # Update x and y arrays
        self.swing_legs[:N:2] = 0  # Set swing_legs to 1 at every second index starting from 0

        x[::2] = x_temp[:N_half] + left_shifts[0]
        y[::2] = y_temp[:N_half] + left_shifts[1]

        x[1::2] = x_temp[:N_half] + right_shifts[0]
        y[1::2] = y_temp[:N_half] + right_shifts[1]

        weights = np.linspace(1,10,self.curriculum+1)
        weights /= sum(weights)
        self.path_angle = self.np_random.choice(self.angle_curriculum[0:self.curriculum+1], p=weights)

        indices = np.arange(4, len(x), 2)
        max_horizontal_shift = sep_dist * 4
        max_vertical_shift = max(dist_range)
        extra_vertical_shift = 0.3 * (1 - min(self.path_angle, np.pi / 4) / (np.pi / 4))
        extra_vertical_shifts = extra_vertical_shift * (np.arange(len(indices)) + 1)

        base_hor = max_horizontal_shift * min(self.path_angle, np.pi / 4) / (np.pi / 4)
        horizontal_shifts = base_hor * (np.arange(len(indices)) + 1)
        x[indices] += horizontal_shifts
        x[indices + 1] += horizontal_shifts
        y[indices] += extra_vertical_shifts
        y[indices + 1] += extra_vertical_shifts

        if self.path_angle > np.pi / 4:
            base_ver = max_vertical_shift * (self.path_angle - np.pi / 4) / (np.pi / 4)
            horizontal_shifts = base_ver * (np.arange(len(indices)) + 1)
            y[indices] -= horizontal_shifts
            y[indices + 1] -= horizontal_shifts

        if self.robot.mirrored:
            self.swing_legs = 1 - self.swing_legs
            x *= -1

        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets), axis=1)

    def create_terrain(self):

        self.steps = []
        self.rendered_steps = []
        step_ids = set()
        cover_ids = set()

        options = {
            # self._p.URDF_ENABLE_SLEEPING |
            "flags": self._p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        }

        # One big step for all
        p = self.plank_class(self._p, self.step_radius, options=options)
        for index in range(self.rendered_step_count):
            # p = self.plank_class(self._p, self.step_radius, options=options)
            self.steps.append(p)
            self.rendered_steps.append(VBox(self._p, radius=self.step_radius, length=0.005, pos=None))
            step_ids = step_ids | {(p.id, p.base_id)}
            cover_ids = cover_ids | {(p.id, p.cover_id)}

        # Need set for detecting contact
        self.all_contact_object_ids = set(step_ids) | set(cover_ids)

        if not self.remove_ground:
            self.all_contact_object_ids |= self.ground_ids

    def set_step_state(self, info_index, step_index):
        # sets rendered step state
        pos = self.terrain_info[info_index, 0:3]
        phi, x_tilt, y_tilt = self.terrain_info[info_index, 3:6]
        quaternion = np.array(pybullet.getQuaternionFromEuler([x_tilt, y_tilt, phi]))
        # self.steps[step_index].set_position(pos=pos, quat=quaternion)
        self.rendered_steps[step_index].set_position(pos=pos, quat=quaternion)

    def randomize_terrain(self, replace=True):
        if replace:
            self.terrain_info = self.generate_step_placements()
        for index in range(self.rendered_step_count):
            self.set_step_state(index, index)

    def update_steps(self):
        if self.rendered_step_count == self.num_steps:
            return

        if self.next_step_index >= self.rendered_step_count:
            oldest = self.next_step_index % self.rendered_step_count
            next = min(self.next_step_index, len(self.terrain_info) - 1)
            self.set_step_state(next, oldest)

    def reset(self):
        if self.state_id >= 0:
            self._p.restoreState(self.state_id)

        self.timestep = 0
        self.done = False
        self.target_reached_count = 0
        self.both_feet_hit_ground = False
        self.current_target_count = 0
        self.swing_leg_lifted_count = 0
        self.swing_leg_lifted = False
        self.body_stationary_count = 0

        self.set_stop_on_next_step = False
        self.stop_on_next_step = False

        self.robot.applied_gain = self.applied_gain_curriculum[self.curriculum]
        prev_robot_mirrored = self.robot.mirrored
        prev_forward = self.walk_forward
        self.walk_forward = True # self.np_random.choice([True, False], p=[0.35, 0.65])
        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start,
            pos=self.robot_init_position[self.walk_forward],
            vel=self.robot_init_velocity,
            quat=self._p.getQuaternionFromEuler((0,0,-90 * RAD2DEG)),
            mirror=True
        )
        # self.swing_leg = 1 if self.robot.mirrored else 0 # for backwards
        self.prev_leg = self.swing_leg

        # Randomize platforms
        replace = self.next_step_index >= self.num_steps / 2 or prev_robot_mirrored != self.robot.mirrored or prev_forward != self.walk_forward
        self.next_step_index = self.lookbehind
        self._prev_next_step_index = self.next_step_index - 1
        self.randomize_terrain(replace)
        self.swing_leg = self.swing_legs[self.next_step_index]
        # print(f"swing legs {self.swing_legs}, with first at {self.swing_leg}")
        self.calc_feet_state()

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)

        self.targets = self.delta_to_k_targets()
        assert self.targets.shape[-1] == self.step_param_dim

        # Order is important because walk_target is set up above
        self.calc_potential()

        state = concatenate((self.robot_state, self.targets.flatten()))

        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        return state

    def step(self, action):
        self.timestep += 1

        self.robot.apply_action(action)
        self.scene.global_step()

        # Stop on the 7th and 14th step, but need to specify N-1 as well
        self.set_stop_on_next_step = self.next_step_index in self.stop_steps

        # Don't calculate the contacts for now
        self.robot_state = self.robot.calc_state()
        self.calc_env_state(action)

        reward = self.progress - self.energy_penalty
        reward += self.step_bonus + self.target_bonus - self.speed_penalty
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty
        reward += self.legs_bonus
        reward -= self.heading_penalty * 2

        # if self.progress != 0:
        #     print(f"{self.next_step_index}: {self.progress}, -{self.energy_penalty}, {self.step_bonus}, {self.target_bonus}, {self.tall_bonus}, -{self.posture_penalty}, -{self.joints_penalty}") #, {self.legs_bonus}")

        # targets is calculated by calc_env_state()
        state = concatenate((self.robot_state, self.targets.flatten()))

        if self.is_rendered or self.use_egl:
            self._handle_keyboard(callback=self.handle_keyboard)
            self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)
            self.target.set_color(
                Colors["dodgerblue"]
                if self.distance_to_target < 0.15
                else Colors["crimson"]
            )
            self.rendered_steps[(self.next_step_index-1) % self.rendered_step_count].set_color(Colors["crimson"])
            self.rendered_steps[self.next_step_index % self.rendered_step_count].set_color(Colors["dodgerblue"])
            self.arrow.set_position(
                pos=[
                    self.terrain_info[self.next_step_index, 0],
                    self.terrain_info[self.next_step_index, 1],
                    self.terrain_info[self.next_step_index, 2] + 0.4
                ], heading=self.terrain_info[self.next_step_index, 6]
            )

        info = {}
        if self.done or self.timestep == self.max_timestep - 1:
            info["curriculum_metric"] = self.next_step_index

        return state, reward, self.done, info

    def handle_keyboard(self, keys):
        RELEASED = self._p.KEY_WAS_RELEASED

        # stop at current
        if keys.get(ord("s")) == RELEASED:
            self.set_stop_on_next_step = not self.set_stop_on_next_step

    def create_target(self):
        # Need this to create target in render mode, called by EnvBase
        # Sphere is a visual shape, does not interact physically
        self.target = VSphere(self._p, radius=0.15, pos=None)
        self.arrow = VArrow(self._p)

    def calc_potential(self):

        # walk_target_theta = atan2(
        #     self.walk_target[1] - self.robot.body_xyz[1],
        #     self.walk_target[0] - self.robot.body_xyz[0],
        # )
        # self.angle_to_target = walk_target_theta - self.robot.body_rpy[2]

        walk_target_delta = self.walk_target - self.robot.body_xyz
        self.distance_to_target = sqrt(ss(walk_target_delta[0:2]))
        foot_target_delta = self.terrain_info[self.next_step_index, 0:2] - self.robot.feet_xyz[self.swing_leg, 0:2]
        foot_distance_to_target = sqrt(ss(foot_target_delta[0:2]))
        self.linear_potential = -(self.distance_to_target * 0.5 + foot_distance_to_target) / self.scene.dt

        # walk_target_delta = self.terrain_info[self.next_step_index, 0:2] - self.robot.feet_xyz[self.swing_leg, 0:2]
        # self.distance_to_target = sqrt(ss(walk_target_delta[0:2]))
        # self.linear_potential = -self.distance_to_target / self.scene.dt

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential

        self.calc_potential()

        linear_progress = self.linear_potential - old_linear_potential
        self.progress = linear_progress * 3

        # if self.next_step_index != self._prev_next_step_index:
        #     print(f"{self.next_step_index}: progress {self.progress} with swing leg {self.swing_leg} at {self.robot.feet_xyz} with target {self.terrain_info[self.next_step_index]}")
        #     print(f"Foot distance to target in 3D: {self.foot_dist_to_target[self.swing_leg]}")
        #     print(f"Vertical errors: {self.robot.feet_xyz[self.swing_leg, 2] - self.terrain_info[self.next_step_index, 2]}")
        #     print(f"Next step position: {self.terrain_info[self.next_step_index]}")

        self.posture_penalty = 0
        if not -0.2 < self.robot.body_rpy[1] < 0.4:
            self.posture_penalty = abs(self.robot.body_rpy[1])

        if not -0.4 < self.robot.body_rpy[0] < 0.4:
            self.posture_penalty += abs(self.robot.body_rpy[0])

        speed = sqrt(ss(self.robot.body_vel))
        self.speed_penalty = max(speed - 1.2, 0)

        electricity_cost = self.electricity_cost * nansum(
            abs(action * self.robot.joint_speeds)
        )
        stall_torque_cost = self.stall_torque_cost * ss(action)
        self.energy_penalty = electricity_cost + stall_torque_cost

        self.joints_penalty = self.joints_at_limit_cost * self.robot.joints_at_limit

        terminal_height = self.terminal_height_curriculum[self.curriculum]
        self.tall_bonus = 2 if self.robot_state[0] > terminal_height else -1.0
        abs_height = self.robot.body_xyz[2] - self.terrain_info[self.next_step_index, 2]

        self.legs_bonus = 0
        # if self.swing_leg_lifted and 1 <= self.swing_leg_lifted_count <= 200 and self._foot_target_contacts[self.swing_leg, 0] == 0:
        #     self.legs_bonus += 0.5

        # if abs(self.robot.body_rpy[2]) > 15 * DEG2RAD or abs(self.robot.lower_body_rpy[2]) > 15 * DEG2RAD:
        #     self.legs_bonus -= 1

        swing_foot_tilt = self.robot.feet_rpy[self.swing_leg, 1]

        # if self.target_reached and swing_foot_tilt > 5 * DEG2RAD:
        #     # allow negative tilt since on heels
        #     self.legs_bonus -= abs(swing_foot_tilt) * 2.5

        # if self.swing_leg_has_fallen:
        #     print(f"{self.next_step_index}: swing leg has fallen, terminating")

        if abs(self.progress) < 0.015:
            self.body_stationary_count += 1
        else:
            self.body_stationary_count = 0
        count = 2000
        # if self.body_stationary_count > count:
        #     self.legs_bonus -= 100

        if abs(self.heading_rad_to_target) >= 0.2:
            self.heading_penalty = - np.exp(-0.5 * abs(self.heading_rad_to_target) **2) + 1
        else:
            self.heading_penalty = 0

        self.done = self.done or self.tall_bonus < 0 or abs_height < -3 or self.swing_leg_has_fallen or self.other_leg_has_fallen or self.body_stationary_count > count
        # if self.done:
        #     print(f"Terminated because not tall: {self.tall_bonus} or abs height: {abs_height} or swing leg has fallen {self.swing_leg_has_fallen} or other leg {self.other_leg_has_fallen}")

    def smallest_angle_between(self, angle1, angle2):
        # Normalize the angles to the range [0, 2pi)
        angle1 = angle1 % (2 * np.pi)
        angle2 = angle2 % (2 * np.pi)
        
        # Calculate the absolute difference between the two angles
        diff = abs(angle1 - angle2)
        
        # The smallest angle is the minimum of the difference and (2 * np.pi) - difference
        smallest_angle = min(diff, (2 * np.pi) - diff)

        if smallest_angle >= np.pi:
            smallest_angle -= 2 * np.pi
        
        return smallest_angle

    def calc_feet_state(self):
        # Calculate contact separately for step
        target_cover_index = self.next_step_index % self.rendered_step_count
        next_step = self.steps[target_cover_index]

        curr_cover_index = (self.next_step_index - 1) % self.rendered_step_count
        curr_step = self.steps[curr_cover_index]

        self.foot_dist_to_target = np.sqrt(
            ss(
                self.robot.feet_xyz[:, 0:2]
                - self.terrain_info[self.next_step_index, 0:2],
                axis=1,
            )
        )

        # allow horizontal step to extend + if swing leg is 1 else neg if swing leg is 0 for x direction
        padding_x = self.step_radius if self.swing_leg == 1 else -self.step_radius
        x_dist_to_target = np.abs(self.robot.feet_xyz[:, 0] - (self.terrain_info[self.next_step_index, 0] + padding_x))
        y_dist_to_target = np.abs(self.robot.feet_xyz[:, 1] - self.terrain_info[self.next_step_index, 1])

        robot_id = self.robot.id
        client_id = self._p._client
        target_id_list = [[next_step.id],[curr_step.id]]
        target_cover_id_list = [[next_step.cover_id],[curr_step.cover_id]]
        if self.swing_leg == 1:
            target_id_list.reverse()
            target_cover_id_list.reverse()
        self._foot_target_contacts.fill(0)

        for i, (foot, contact) in enumerate(
            zip(self.robot.feet, self._foot_target_contacts)
        ):
            self.robot.feet_contact[i] = pybullet.getContactStates(
                bodyA=robot_id,
                linkIndexA=foot.bodyPartIndex,
                bodiesB=target_id_list[i],
                linkIndicesB=target_cover_id_list[i],
                results=contact,
                physicsClientId=client_id,
            )

        self.imaginary_step = self.terrain_info[self.next_step_index,2] > 0.01
        self.current_target_count += 1

        self.heading_rad_to_target = self.smallest_angle_between(self.robot.body_rpy[2], self.terrain_info[self.next_step_index, 6])

        if self.next_step_index == 1 or self.swing_leg_lifted:
            # if first step or already lifted, say true
            self.swing_leg_lifted = True
        if self._foot_target_contacts[self.swing_leg, 0] == 0:
            # if in the air, increase count
            self.swing_leg_lifted_count += 1
        else:
            self.swing_leg_lifted_count = 0

        if not self.swing_leg_lifted:
            # if not lifted yet and over count, True
            if self.swing_leg_lifted_count >= 1:
                self.swing_leg_lifted = True

        x_radius = self.step_radius * 2
        y_radius = self.step_radius

        if self.next_step_index > 1:
            x_dist_to_prev_target = np.abs(self.robot.feet_xyz[:, 0] - self.prev_leg_pos[:,0])
            y_dist_to_prev_target = np.abs(self.robot.feet_xyz[:, 1] - self.prev_leg_pos[:,1])
            foot_in_target = x_dist_to_target[self.swing_leg] < x_radius and y_dist_to_target[self.swing_leg] < y_radius
            foot_in_prev_target = x_dist_to_prev_target[self.swing_leg] < x_radius and y_dist_to_prev_target[self.swing_leg] < y_radius
            other_foot_in_prev_target = x_dist_to_prev_target[1-self.swing_leg] < x_radius and y_dist_to_prev_target[1-self.swing_leg] < y_radius
            swing_leg_not_on_steps = not foot_in_target and not foot_in_prev_target

        swing_leg_in_air = self._foot_target_contacts[self.swing_leg, 0] == 0
        other_leg_in_air = self._foot_target_contacts[1-self.swing_leg, 0] == 0

        # if swing leg is not on previous step and not on current step and not in air, should terminate
        self.swing_leg_has_fallen = self.next_step_index > 1 and not swing_leg_in_air and swing_leg_not_on_steps
        self.other_leg_has_fallen = self.next_step_index > 2 and not other_leg_in_air and not other_foot_in_prev_target
        
        self.target_reached = self._foot_target_contacts[self.swing_leg, 0] > 0 and x_dist_to_target[self.swing_leg] < x_radius and y_dist_to_target[self.swing_leg] < y_radius and self.swing_leg_lifted

        if self.target_reached:
            self.target_reached_count += 1

            # Advance after has stopped for awhile
            if self.target_reached_count > 120:
                self.stop_on_next_step = False
                self.set_stop_on_next_step = False

            # Slight delay for target advancement
            # Needed for not over counting step bonus
            delay = 2
            if self.target_reached_count >= delay:
                # print(f"{self.next_step_index}: Reached target after {self.current_target_count}, {self.both_feet_hit_ground}!")
                if not self.stop_on_next_step:
                    self.current_target_count = 0
                    self.prev_leg_pos = self.robot.feet_xyz[:, 0:2]
                    self.prev_leg = self.swing_leg
                    self.next_step_index += 1
                    self.swing_leg = (self.swing_leg + 1) % 2
                    # self.swing_leg = self.swing_legs[self.next_step_index]
                    # if (
                    #     self.next_step_index - 1 in self.stop_steps
                    #     and self.next_step_index - 2 in self.stop_steps
                    # ):
                    #     self.swing_leg = 1 - self.swing_leg
                    self.target_reached_count = 0
                    self.swing_leg_lifted = False
                    self.swing_leg_lifted_count = 0
                    self.both_feet_hit_ground = False
                    self.body_stationary_count = 0
                    self.update_steps()
                self.stop_on_next_step = self.set_stop_on_next_step

            # Prevent out of bound
            if self.next_step_index >= len(self.terrain_info):
                self.next_step_index -= 1

    def calc_step_reward(self):

        self.step_bonus = 0
        if (
            self.target_reached
            and self.target_reached_count == 1
            and self.next_step_index != len(self.terrain_info) - 1  # exclude last step
        ):
            dist = nanmin(self.foot_dist_to_target)
            self.step_bonus = 50 * 2.718 ** (
                -(dist ** self.step_bonus_smoothness) / 0.25
            )

        # For remaining stationary
        self.target_bonus = 0
        last_step = self.next_step_index == len(self.terrain_info) - 1
        if (last_step or self.stop_on_next_step) and self.distance_to_target < 0.15:
            self.target_bonus = 2.0

    def calc_env_state(self, action):
        if anynan(self.robot_state):
            print("~INF~", self.robot_state)
            self.done = True

        cur_step_index = self.next_step_index

        # detects contact and set next step
        self.calc_feet_state()
        self.calc_base_reward(action)
        self.calc_step_reward()
        # use next step to calculate next k steps
        self.targets = self.delta_to_k_targets()

        if cur_step_index != self.next_step_index:
            self.calc_potential()

    def delta_to_k_targets(self):
        """ Return positions (relative to root) of target, and k-1 step after """
        k = self.lookahead
        j = self.lookbehind
        N = self.next_step_index
        if self._prev_next_step_index != self.next_step_index:
            if not self.stop_on_next_step:
                if N - j >= 0:
                    targets = self.terrain_info[N - j : N + k]
                else:
                    targets = concatenate(
                        (
                            [self.terrain_info[0]] * j,
                            self.terrain_info[N : N + k],
                        )
                    )
                if len(targets) < (k + j):
                    # If running out of targets, repeat last target
                    targets = concatenate(
                        (targets, [targets[-1]] * ((k + j) - len(targets)))
                    )
            else:
                targets = concatenate(
                    (
                        self.terrain_info[N - j : N],
                        [self.terrain_info[N]] * k,
                    )
                )
            self._prev_next_step_index = self.next_step_index
            self._targets = targets
        else:
            targets = self._targets

        if (self.path_angle >= 0 and self.swing_leg == 0) or (self.path_angle < 0 and self.swing_leg == 1) or not hasattr(self, 'walk_target'):
            # only change when moving leg in direction
            if self.next_step_index + 2 < self.num_steps:
                self.walk_target = np.copy(self.terrain_info[self.next_step_index + 2, 0:3])
            else:
                self.walk_target = np.copy(self.terrain_info[self.next_step_index, 0:3])
            if self.swing_leg == 1:
                self.walk_target[0] += self.step_radius
            else:
                self.walk_target[0] -= self.step_radius

        delta_pos = targets[:, 0:3] - self.robot.body_xyz
        target_thetas = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])

        angle_to_targets = target_thetas - self.robot.body_rpy[2]
        distance_to_targets = np.sqrt(ss(delta_pos[:, 0:2], axis=1))
        heading_angle_to_targets = targets[:, 6] - self.robot.body_rpy[2]

        deltas = concatenate(
            (
                (np.sin(angle_to_targets) * distance_to_targets)[:, None],  # x
                (np.cos(angle_to_targets) * distance_to_targets)[:, None],  # y
                (delta_pos[:, 2])[:, None],  # z
                (targets[:, 4])[:, None],  # x_tilt
                (targets[:, 5])[:, None],  # y_tilt
                (heading_angle_to_targets)[:, None],
            ),
            axis=1,
        )

        return deltas

    def get_mirror_indices(self):

        action_dim = self.robot.action_space.shape[0]

        right_obs_indices = concatenate(
            (
                # joint angle indices + 6 accounting for global
                6 + self.robot._right_joint_indices,
                # joint velocity indices
                6 + self.robot._right_joint_indices + action_dim,
                # right foot contact
                [
                    6 + 2 * action_dim + 2 * i
                    for i in range(len(self.robot.foot_names) // 2)
                ],
            )
        )

        # Do the same for left, except using +1 for left foot contact
        left_obs_indices = concatenate(
            (
                6 + self.robot._left_joint_indices,
                6 + self.robot._left_joint_indices + action_dim,
                [
                    6 + 2 * action_dim + 2 * i + 1
                    for i in range(len(self.robot.foot_names) // 2)
                ],
            )
        )

        robot_neg_obs_indices = concatenate(
            (
                # vy, roll
                [2, 4],
                # negate part of robot (position)
                6 + self.robot._negation_joint_indices,
                # negate part of robot (velocity)
                6 + self.robot._negation_joint_indices + action_dim,
            )
        )

        steps_neg_obs_indices = np.array(
            [
                (
                    i * self.step_param_dim + 0,  # sin(-x) = -sin(x)
                    i * self.step_param_dim + 3,  # x_tilt
                    i * self.step_param_dim + 5, # heading
                )
                for i in range(self.lookahead + self.lookbehind)
            ],
            dtype=np.int64,
        ).flatten()

        negation_obs_indices = concatenate(
            (robot_neg_obs_indices, steps_neg_obs_indices + self.robot_obs_dim)
        )

        # Used for creating mirrored actions
        negation_action_indices = self.robot._negation_joint_indices
        right_action_indices = self.robot._right_joint_indices
        left_action_indices = self.robot._left_joint_indices

        obs_dim = self.observation_space.shape[0]
        assert len(negation_obs_indices) == 0 or negation_obs_indices.max() < obs_dim
        assert right_obs_indices.max() < obs_dim
        assert left_obs_indices.max() < obs_dim
        assert (
            len(negation_action_indices) == 0
            or negation_action_indices.max() < action_dim
        )
        assert right_action_indices.max() < action_dim
        assert left_action_indices.max() < action_dim

        return (
            negation_obs_indices,
            right_obs_indices,
            left_obs_indices,
            negation_action_indices,
            right_action_indices,
            left_action_indices,
        )


class MikeStepperEnv(Walker3DStepperEnv):
    robot_class = Mike
    robot_init_position = (0.3, 0, 1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.is_rendered:
            self.robot.decorate()


class LaikagoCustomEnv(Walker3DCustomEnv):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 8

    robot_class = Laikago

    termination_height = 0
    robot_random_start = False
    robot_init_position = [0, 0, 0.56]

    def __init__(self, **kwargs):
        kwargs.pop("plank_class", None)
        super().__init__(**kwargs)

        # Fix-ordered Curriculum
        self.curriculum = 0
        self.max_curriculum = 9

        # Need for checking early termination
        links, names = self.robot.parts, self.robot.foot_names
        self.foot_ids = [links[k].bodyPartIndex for k in names]

    def calc_base_reward(self, action):
        super().calc_base_reward(action)

        self.tall_bonus = 0
        contacts = self._p.getContactPoints(bodyA=self.robot.id)
        ground_body_id = self.scene.ground_plane_mjcf[0]

        for c in contacts:
            if c[2] == ground_body_id and c[3] not in self.foot_ids:
                self.tall_bonus = -1
                self.done = True
                break


class LaikagoStepperEnv(Walker3DStepperEnv):
    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    robot_class = Laikago
    robot_random_start = False
    robot_init_position = [0.25, 0, 0.53]
    robot_init_velocity = [0.5, 0, 0.25]

    step_radius = 0.16
    rendered_step_count = 4
    init_step_separation = 0.45

    lookahead = 2
    lookbehind = 2
    walk_target_index = -1
    step_bonus_smoothness = 6

    def __init__(self, **kwargs):
        # Handle non-robot kwargs
        super().__init__(**kwargs)

        N = self.max_curriculum + 1
        self.terminal_height_curriculum = np.linspace(0.20, 0.0, N)
        self.applied_gain_curriculum = np.linspace(1.0, 1.0, N)

        self.dist_range = np.array([0.45, 0.75])
        self.pitch_range = np.array([-20, +20])  # degrees
        self.yaw_range = np.array([-20, 20])
        self.tilt_range = np.array([-10, 10])

        # Need for checking early termination
        links, names = self.robot.parts, self.robot.foot_names
        self.foot_ids = [links[k].bodyPartIndex for k in names]

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential
        self.calc_potential()
        self.progress = self.linear_potential - old_linear_potential

        self.energy_penalty = self.electricity_cost * float(
            np.abs(action * self.robot.joint_speeds).mean()
        )
        self.energy_penalty += self.stall_torque_cost * float(np.square(action).mean())

        self.joints_penalty = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )

        # posture is different from walker3d
        joint_angles = self.robot.joint_angles * RAD2DEG

        hip_x_angles = joint_angles[[0, 3, 6, 9]]
        good_mask = (-25 < hip_x_angles) * (hip_x_angles < 25)
        self.posture_penalty = np.dot(1 * ~good_mask, np.abs(hip_x_angles * DEG2RAD))

        hip_y_angles = joint_angles[[1, 4, 7, 10]]
        good_mask = (-35 < hip_y_angles) * (hip_y_angles < 35)
        self.posture_penalty += np.dot(1 * ~good_mask, np.abs(hip_y_angles * DEG2RAD))

        knee_angles = joint_angles[[2, 5, 8, 11]]
        good_mask = (-75 < knee_angles) * (knee_angles < -15)
        self.posture_penalty += np.dot(1 * ~good_mask, np.abs(knee_angles * DEG2RAD))

        if not -25 < self.robot.body_rpy[1] * RAD2DEG < 25:
            self.posture_penalty += abs(self.robot.body_rpy[1])

        self.progress *= 2
        self.posture_penalty *= 0.2

        contacts = self._p.getContactPoints(bodyA=self.robot.id)
        ids = self.all_contact_object_ids

        self.tall_bonus = 2
        self.speed_penalty = 0

        # Time-based early termination
        self.done = self.timestep > 240 and self.next_step_index <= 4
        foot_ids = self.foot_ids
        for c in contacts:
            if {(c[2], c[4])} & ids and c[3] not in foot_ids:
                self.tall_bonus = -1
                self.done = True
                break


class Walker3DPlannerEnv(EnvBase):
    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4
    max_timestep = 1000

    robot_class = Walker3D
    robot_random_start = False
    robot_init_position = [-15.5, -15.5, 1.32]
    robot_init_orientation = [0.0, 0.0, 0.383, 0.929]
    robot_init_velocity = None
    robot_torso_name = "waist"

    termination_height = 0.5
    paths_to_plan = 1
    # Actually N - 1, since last step of one path is the first step of next path
    steps_to_plan = 5
    num_bridges = 1
    bridge_length = 14

    # base controller
    base_filename = "Walker3DPlannerBase.pt"
    base_lookahead = 2
    base_lookbehind = 1
    base_step_param_dim = 5

    def __init__(self, **kwargs):
        self.curriculum = 9
        self.max_curriculum = 9
        self.advance_threshold = -0.3  # negative linear potential
        self._prev_curriculum = self.curriculum

        super().__init__(self.robot_class, remove_ground=True, **kwargs)
        self.robot_torso_id = self.robot.parts[self.robot_torso_name].bodyPartIndex
        self.query_base_controller = self.load_base_controller(self.base_filename)

        P = self.paths_to_plan
        N = self.steps_to_plan

        self.robot_obs_dim = self.robot.observation_space.shape[0]
        # target direction (xy), (P * N * (xyz + base_step_param_dim))
        high = np.inf * np.ones(self.robot_obs_dim + 2 + P * N * 5)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Probability of choosing one path to take
        high = np.inf * np.ones(P)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Pre-allocate buffer
        self.foot_dist_to_target = np.zeros(len(self.robot.feet))

    def create_terrain(self):
        rendered = self.is_rendered or self.use_egl
        if hasattr(self, "terrain"):
            self._p.removeBody(self.terrain.id)

        filename = "height_field_map_1.npy"
        scale = 16

        # Curriculum cannot be 0, otherwise raycast doesn't work
        self.curriculum = max(min(self.curriculum, self.max_curriculum), 1)
        ratio = self.curriculum / self.max_curriculum

        self.terrain = HeightField(
            self._p,
            xy_scale=scale,
            z_scale=ratio,
            rendered=rendered,
            filename=filename,
            rng=self.np_random,
        )

        if not hasattr(self, "discrete_planks"):
            options = {
                "useMaximalCoordinates": True,
                "flags": self._p.URDF_ENABLE_SLEEPING,
            }
            self.discrete_planks = [
                Pillar(self._p, 0.25, options=options)
                for _ in range(self.num_bridges * self.bridge_length)
            ]
            step_ids = set([(p.id, p.base_id) for p in self.discrete_planks])
            cover_ids = set([(p.id, p.cover_id) for p in self.discrete_planks])
            self.all_contact_object_ids = step_ids | cover_ids | {(self.terrain.id, -1)}

        if rendered and not hasattr(self, "target"):
            self.target = VSphere(self._p, radius=0.15, pos=None)

    def load_base_controller(self, filename):
        dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir, "data", "controllers", filename)
        actor_critic = torch.load(model_path, map_location="cpu")

        def inference(o):
            with torch.no_grad():
                o = torch.from_numpy(o).float().unsqueeze(0)
                value, action, _ = actor_critic.act(o, deterministic=True)
                return value.squeeze().numpy(), action.squeeze().numpy()

        return inference

    def sample_paths(self, x0=None, y0=None, yaw0=None):
        """Generate P paths of N steps in world coordinate.
        Each step has 6 parameters for 3D translation and 3D normal vector.
        """

        dist_range = np.array([0.65, 1.05])
        yaw_range = np.array([-20, 20]) * DEG2RAD

        P = self.paths_to_plan
        N = self.steps_to_plan
        rng = self.np_random

        x0 = self.robot.body_xyz[0] if x0 is None else x0
        y0 = self.robot.body_xyz[1] if y0 is None else y0
        yaw0 = self.robot.body_rpy[2] if yaw0 is None else yaw0

        dr = rng.uniform(*dist_range, size=(P, N))
        dphi = rng.uniform(*yaw_range, size=(P, N))

        # First step needs to be under
        dr[:, 0] = 0
        dphi[:, 0] = 0

        phi = np.cumsum(dphi, axis=-1) + yaw0
        x = np.cumsum(dr * np.cos(phi), axis=-1) + x0
        y = np.cumsum(dr * np.sin(phi), axis=-1) + y0

        z, x_tilt, y_tilt = self.terrain.get_height_and_tilt_at(x, y)

        # Tilts are global, need to convert to its own coordinate axes
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        matrix = np.array(
            [
                [cos_phi, sin_phi],
                [-sin_phi, cos_phi],
            ]
        )
        x_tilt, y_tilt = (matrix * np.stack((x_tilt, y_tilt))).sum(axis=1)

        # Check if a discrete step at xy, if so, use its height
        xy = np.stack((x, y), axis=-1).reshape(P * N, 2)
        delta = xy[:, None] - self.discrete_planks_parameters[None, :, 0:2]
        distance = np.sqrt(ss(delta, axis=-1))

        min_dist_index = nanargmin(distance, axis=-1)
        replace_mask = distance[range(P), min_dist_index] < 0.25
        z_replacement = self.discrete_planks_parameters[min_dist_index, 2]
        z.reshape(P * N)[replace_mask] = z_replacement[replace_mask]
        x_tilt.reshape(P * N)[replace_mask] = 0
        y_tilt.reshape(P * N)[replace_mask] = 0

        if self.total_steps_made + self.steps_to_plan < self.bridge_length:
            a = self.total_steps_made
            b = a + self.steps_to_plan
            return self.discrete_planks_parameters[a:b][None]
        else:
            return np.stack((x, y, z, phi, x_tilt, y_tilt), axis=-1)

    def get_local_coordinates(self, targets):
        # targets should be (P, N, 6)
        if len(targets.shape) < 3:
            targets = targets[None]

        delta_positions = targets[:, :, 0:3] - self.robot.body_xyz
        target_thetas = np.arctan2(delta_positions[:, :, 1], delta_positions[:, :, 0])

        angle_to_targets = target_thetas - self.robot.body_rpy[2]
        distance_to_targets = np.sqrt(ss(delta_positions[:, :, 0:2], axis=-1))

        local_parameters = np.stack(
            (
                np.sin(angle_to_targets) * distance_to_targets,
                np.cos(angle_to_targets) * distance_to_targets,
                delta_positions[:, :, 2],
                targets[:, :, 4],
                targets[:, :, 5],
            ),
            axis=-1,
        )

        return local_parameters.squeeze()

    def get_base_step_parameters(self, path):
        k = self.base_lookahead
        j = self.base_lookbehind
        N = self.next_step_index
        if N - j >= 0:
            base_step_parameters = path[N - j : N + k]
        else:
            base_step_parameters = concatenate(
                (
                    np.repeat(path[[0]], j, axis=0),
                    path[N : N + k],
                )
            )
        if len(base_step_parameters) < (k + j):
            # If running out of targets, repeat last target
            base_step_parameters = concatenate(
                (
                    base_step_parameters,
                    np.repeat(
                        base_step_parameters[[-1]],
                        (k + j) - len(base_step_parameters),
                        axis=0,
                    ),
                )
            )

        return base_step_parameters

    def calc_next_step_index(self, path):
        feet_xy = self.robot.feet_xyz[:, 0:2]
        step_xy = path[self.next_step_index, 0:2]
        self.foot_dist_to_target = np.sqrt(ss(feet_xy - step_xy, axis=-1))
        closest_foot = nanargmin(self.foot_dist_to_target)

        if (
            self.robot.feet_contact[closest_foot]
            and self.foot_dist_to_target[closest_foot] < 0.3
        ):
            self.target_reached_count += 1

            # stop_frame = 120 if self.next_step_index == self.steps_to_plan - 1 else 2
            if self.target_reached_count >= 2:
                self.next_step_index += 1
                self.total_steps_made += 1
                self.target_reached_count = 0

    def get_observation_components(self):
        softsign = lambda x: x / (1 + abs(x))
        dx = 5 * softsign(self.distance_to_target) * cos(self.angle_to_target)
        dy = 5 * softsign(self.distance_to_target) * sin(self.angle_to_target)
        path_parameters = self.get_local_coordinates(self.candidate_paths)
        return (self.robot_state, [dx, dy], path_parameters.flatten())

    def reset(self):
        self.total_steps_made = 0
        self.timestep = 0
        self.done = False

        if self._prev_curriculum != self.curriculum:
            self.create_terrain()
            self._prev_curriculum = self.curriculum

        def make_bridge(planks, default=False):
            L = len(planks)

            if default:
                phi = 45 * DEG2RAD * np.cos(np.linspace(0, 2 * np.pi, L))
                px = np.cumsum(0.8 * np.cos(phi)) + self.robot_init_position[0]
                py = np.cumsum(0.8 * np.sin(phi)) + self.robot_init_position[1]
            else:
                while True:
                    phi_max = 20 * DEG2RAD
                    dphi = self.np_random.uniform(-phi_max, phi_max, L)
                    phi = np.cumsum(dphi) + self.np_random.uniform(0, 2 * np.pi)
                    pr = self.np_random.uniform(0.7, 1.1, L)
                    px = np.cumsum(pr * np.cos(phi)) + self.np_random.uniform(-12, 12)
                    py = np.cumsum(pr * np.sin(phi)) + self.np_random.uniform(-12, 12)
                    # TODO: Hardcode map size
                    if (
                        min(px) > -15
                        and max(px) < 15
                        and min(py) > -15
                        and max(py) < 15
                    ):
                        break

            pz, _, _ = self.terrain.get_height_and_tilt_at(px, py)
            im = np.argmax(pz) / (L - 1)
            nx = np.linspace(0, 1, L)
            s = self.np_random.uniform(0.25, 1)
            a = -4 * (np.max(pz) + s - pz[0] - (pz[-1] - pz[0]) * im)
            b = pz[-1] - pz[0] - a
            pz = np.maximum(a * nx ** 2 + b * nx + pz[0], pz)

            for x, y, z, yaw, plank in zip(px, py, pz, phi, planks):
                q = self._p.getQuaternionFromEuler([0, 0, yaw])
                plank.set_position(pos=(x, y, z), quat=q)

            zero = np.zeros_like(px)
            return np.stack((px, py, pz, phi, zero, zero), axis=1)

        L = self.bridge_length
        self.discrete_planks_parameters = concatenate(
            [
                make_bridge(self.discrete_planks[i * L : (i + 1) * L], i == 0)
                for i in range(self.num_bridges)
            ]
        )

        # self.robot.applied_gain = 1.2
        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start,
            pos=self.robot_init_position,
            quat=self.robot_init_orientation,
        )

        # Sample map smaller than actual terrain to prevent out of bound
        xy = self.np_random.uniform(-12, 12, 2)
        z, _, _ = self.terrain.get_height_and_tilt_at(xy[[0]], xy[[1]])
        self.walk_target = np.array((*xy, z[0]))

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)
            self.target.set_position(self.walk_target)

        self.calc_potential()  # walk_target must be set first
        # must be called before get observation
        self.candidate_paths = self.sample_paths()
        state = concatenate(self.get_observation_components())
        return state

    def step(self, action):
        action_exp = np.exp(action)
        action_prob = action_exp / action_exp.sum()
        selected_action = self.np_random.choice(
            np.arange(self.paths_to_plan), p=action_prob
        )
        selected_path = self.candidate_paths[selected_action]

        self.next_step_index = 0
        self.target_reached_count = 0
        reward = 0

        x_tilt = selected_path[:, 4]
        y_tilt = selected_path[:, 5]
        if not hasattr(self, "raycast_lines"):
            z = [0, 0, 0]
            self.raycast_lines = [self._p.addUserDebugLine(z, z) for _ in selected_path]
            self.normal_lines = [self._p.addUserDebugLine(z, z) for _ in selected_path]
            self.tilt_steps = [
                Pillar(self._p, 0.25, pos=None) for _ in range(self.steps_to_plan)
            ]
            for step in self.tilt_steps:
                for link_id in range(-1, self._p.getNumJoints(step.id)):
                    self._p.setCollisionFilterGroupMask(step.id, link_id, 0, 0)

        for params, id1, id2, step in zip(
            selected_path, self.raycast_lines, self.normal_lines, self.tilt_steps
        ):
            o = [0, 0, 1]
            p = params[0:3]
            q = self._p.getQuaternionFromEuler(params[[4, 5, 3]])
            # q = self._p.getQuaternionFromEuler((*params[[4, 5]], 0))

            m = np.reshape(
                self._p.getMatrixFromQuaternion(
                    self._p.getQuaternionFromEuler((*params[[4, 5]], 0))
                ),
                (3, 3),
            )
            n = m @ [0, 0, 1]

            self._p.addUserDebugLine(
                p + o, p - o, (1, 0, 0), 5, replaceItemUniqueId=id1
            )
            self._p.addUserDebugLine(p, p + n, (0, 0, 1), 5, replaceItemUniqueId=id2)
            step.set_position(pos=p, quat=q)

        while not self.done and self.next_step_index < self.steps_to_plan - 1:

            path_parameters = self.get_local_coordinates(selected_path)
            base_parameters = self.get_base_step_parameters(path_parameters)

            base_obs = concatenate((self.robot_state, base_parameters.flatten()))
            base_value, base_action = self.query_base_controller(base_obs)

            self.timestep += 1
            self.robot.apply_action(base_action)
            self.scene.global_step()

            self.robot_state = self.robot.calc_state(self.all_contact_object_ids)
            self.calc_next_step_index(selected_path)

            self.calc_base_reward()
            # base_value * (1 - discount_factor)
            reward += (self.progress + base_value * 0.01) / 2

            if self.is_rendered or self.use_egl:
                self._handle_keyboard()
                self._handle_mouse()
                self.camera.track(self.robot.body_xyz)

            self.done = (
                self.done
                or self.robot_state[0] < self.termination_height
                or self.robot.body_xyz[2] < -5  # free falling off terrain
                or (self.linear_potential * self.scene.dt) > -0.3  # target within 30cm
                or self.timestep >= self.max_timestep
                or self._p.getContactPoints(
                    bodyA=self.robot.id, linkIndexA=self.robot_torso_id
                )  # torso contact ground
            )

        if not self.done:
            # Need to keep planning
            # must be called before get observation
            x0, y0, yaw0 = selected_path[-1, [0, 1, 3]]
            self.candidate_paths = self.sample_paths(x0, y0, yaw0)

        state = concatenate(self.get_observation_components())
        info = (
            {
                "curriculum_metric": self.linear_potential * self.scene.dt,
            }
            if self.done or self.timestep == self.max_timestep
            else {}
        )

        return state, reward, self.done, info

    def calc_base_reward(self):
        old_linear_potential = self.linear_potential
        self.calc_potential()
        linear_progress = self.linear_potential - old_linear_potential
        self.progress = linear_progress

    def calc_potential(self):
        delta = self.walk_target - self.robot.body_xyz
        theta = np.arctan2(delta[1], delta[0])
        self.angle_to_target = theta - self.robot.body_rpy[2]
        self.distance_to_target = sqrt(ss(delta[0:2]))
        self.linear_potential = -self.distance_to_target / self.scene.dt


class MikePlannerEnv(Walker3DPlannerEnv):
    robot_class = Mike
    robot_init_position = [-15.5, -15.5, 1.05]
    base_filename = "MikePlannerBase.pt"


class Monkey3DStepperEnv(Walker3DStepperEnv):

    robot_class = Monkey3D
    robot_random_start = False
    robot_init_position = [0, 0, 0]
    robot_init_velocity = None

    step_radius = 0.015
    step_length = 5

    stop_steps = []

    def __init__(self, **kwargs):

        self.swing_leg = 0
        super().__init__(**kwargs)
        self.robot.set_base_pose(pose="monkey_start")

        N = self.max_curriculum + 1
        self.applied_gain_curriculum = np.linspace(1.0, 1.0, N)

        # Terrain info
        self.dist_range = np.array([0.3, 0.5])
        self.pitch_range = np.array([-30, +30])
        self.yaw_range = np.array([-20, 20])
        self.tilt_range = np.array([-15, 15])
        self.step_param_dim = 5
        self.terrain_info = self.generate_step_placements()

    def generate_step_placements(self):

        # Check just in case
        self.curriculum = min(self.curriculum, self.max_curriculum)
        ratio = self.curriculum / self.max_curriculum

        # {self.max_curriculum + 1} levels in total
        dist_upper = np.linspace(*self.dist_range, self.max_curriculum + 1)
        dist_range = np.array([self.dist_range[0], dist_upper[self.curriculum]])
        yaw_range = self.yaw_range * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        N = self.num_steps
        dr = self.np_random.uniform(*dist_range, size=N)
        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # Special treatment for first step
        dr[0] = 0
        dphi[0] = 0
        dtheta[0] = np.pi / 2
        x_tilt[0] = 0
        y_tilt[0] = 0

        dphi = np.cumsum(dphi)

        dx = dr * np.sin(dtheta) * np.cos(dphi)
        dy = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        # Fix overlapping steps
        dx_max = np.maximum(np.abs(dx[2:]), self.step_radius * 2.5)
        dx[2:] = np.sign(dx[2:]) * np.minimum(dx_max, self.dist_range[1])

        dx[0] += 0.04
        dz[0] += 0.04

        # Put first step right on hand
        x = np.cumsum(dx) + self.robot.feet_xyz[self.swing_leg, 0]
        y = np.cumsum(dy) + self.robot.feet_xyz[self.swing_leg, 1]
        z = np.cumsum(dz) + self.robot.feet_xyz[self.swing_leg, 2]

        return np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)

    def create_terrain(self):
        self.steps = []
        step_ids = set()
        cover_ids = set()

        for index in range(self.rendered_step_count):
            p = MonkeyBar(self._p, self.step_radius, self.step_length)
            self.steps.append(p)
            step_ids = step_ids | {(p.id, p.base_id)}
            cover_ids = cover_ids | {(p.id, p.cover_id)}

        # Need set for detecting contact
        self.all_contact_object_ids = set(step_ids) | set(cover_ids)

        if not self.remove_ground:
            self.all_contact_object_ids |= self.ground_ids

    def reset(self):
        if self.state_id >= 0:
            self._p.restoreState(self.state_id)

        self.timestep = 0
        self.done = False
        self.target_reached_count = 0

        self.set_stop_on_next_step = False
        self.stop_on_next_step = False

        self.robot.applied_gain = self.applied_gain_curriculum[self.curriculum]
        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start,
            pos=self.robot_init_position,
            vel=self.robot_init_velocity,
        )
        self.swing_leg = 1 if self.robot.mirrored else 0

        # Randomize platforms
        # replace = self.next_step_index >= self.num_steps / 2
        self.next_step_index = self.lookbehind
        self._prev_next_step_index = self.next_step_index - 1
        self.randomize_terrain(replace=True)
        self.calc_feet_state()

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)

        self.targets = self.delta_to_k_targets()
        assert self.targets.shape[-1] == self.step_param_dim

        # Order is important because walk_target is set up above
        self.calc_potential()

        state = concatenate((self.robot_state, self.targets.flatten()))

        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        return state

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential

        self.calc_potential()

        linear_progress = self.linear_potential - old_linear_potential
        self.progress = linear_progress

        self.posture_penalty = 0
        if not -0.2 < self.robot.body_rpy[1] < 0.4:
            self.posture_penalty = abs(self.robot.body_rpy[1])

        if not -0.4 < self.robot.body_rpy[0] < 0.4:
            self.posture_penalty += abs(self.robot.body_rpy[0])

        speed = sqrt(ss(self.robot.body_vel))
        self.speed_penalty = max(speed - 1.6, 0)

        electricity_cost = self.electricity_cost * nansum(
            abs(action * self.robot.joint_speeds)
        )
        stall_torque_cost = self.stall_torque_cost * ss(action)
        self.energy_penalty = electricity_cost + stall_torque_cost

        self.joints_penalty = self.joints_at_limit_cost * self.robot.joints_at_limit

        self.tall_bonus = 1
        abs_height = self.robot.body_xyz[2] - self.terrain_info[self.next_step_index, 2]
        self.done = self.done or self.tall_bonus < 0 or abs_height < -3

    def calc_feet_state(self):
        # Calculate contact separately for step
        target_cover_index = self.next_step_index % self.rendered_step_count
        next_step = self.steps[target_cover_index]
        # target_cover_id = {(next_step.id, next_step.cover_id)}

        self.foot_dist_to_target = np.sqrt(
            ss(
                self.robot.feet_xyz[:, 0:2]
                - self.terrain_info[self.next_step_index, 0:2],
                axis=1,
            )
        )

        robot_id = self.robot.id
        client_id = self._p._client
        target_id_list = [next_step.id]
        target_cover_id_list = [next_step.cover_id]
        self._foot_target_contacts.fill(0)

        for i, (name, contact) in enumerate(
            zip(["right_palm", "left_palm"], self._foot_target_contacts)
        ):
            foot = self.robot.parts[name]
            self.robot.feet_contact[i] = pybullet.getContactStates(
                bodyA=robot_id,
                linkIndexA=foot.bodyPartIndex,
                bodiesB=target_id_list,
                linkIndicesB=target_cover_id_list,
                results=contact,
                physicsClientId=client_id,
            )

        if (
            self.next_step_index - 1 in self.stop_steps
            and self.next_step_index - 2 in self.stop_steps
        ):
            self.swing_leg = nanargmax(self._foot_target_contacts[:, 0])
        self.target_reached = self._foot_target_contacts[self.swing_leg, 0] > 0

        # At least one foot is on the plank
        if self.target_reached:
            self.target_reached_count += 1

            # Advance after has stopped for awhile
            if self.target_reached_count > 120:
                self.stop_on_next_step = False
                self.set_stop_on_next_step = False

            # Slight delay for target advancement
            # Needed for not over counting step bonus
            if self.target_reached_count >= 2:
                if not self.stop_on_next_step:
                    self.swing_leg = (self.swing_leg + 1) % 2
                    self.next_step_index += 1
                    self.target_reached_count = 0
                    self.update_steps()
                self.stop_on_next_step = self.set_stop_on_next_step

            # Prevent out of bound
            if self.next_step_index >= len(self.terrain_info):
                self.next_step_index -= 1
