from math import sin, cos, atan2, sqrt, isclose
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
    VFoot,
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
    "lightgrey": (0.8274509803921568, 0.8274509803921568, 0.8274509803921568, 1.0),
    "yellow": (1.0, 1.0, 0.0, 1.0),
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
                else Colors["lightgrey"]
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
    robot_init_position = [0, 0.3, 1.32]
    robot_init_velocity = None

    plank_class = VeryLargePlank  # Pillar, Plank, LargePlank
    num_steps = 20
    step_radius = 0.25
    foot_sep = 0.16
    rendered_step_count = 4
    init_step_separation = 0.70

    step_delay = 4

    lookahead = 2
    lookbehind = 1
    walk_target_index = -1
    step_bonus_smoothness = 1
    stop_steps = [18, 19] # list(range(4,20))

    def __init__(self, **kwargs):
        # Handle non-robot kwargs
        plank_name = kwargs.pop("plank_class", None)
        self.plank_class = globals().get(plank_name, self.plank_class)

        super().__init__(self.robot_class, remove_ground=False, **kwargs)
        self.robot.set_base_pose(pose="running_start")

        # Fix-ordered Curriculum
        self.curriculum = kwargs.pop("start_curriculum", 0)
        self.max_curriculum = 9
        self.advance_threshold = min(15, self.num_steps)

        # each behavior curriculum has a smaller size-9 curriculum
        self.behavior_curriculum = kwargs.pop("start_behavior_curriculum", 0)
        self.behaviors = ["heading_var", "timing_gaits", "to_standstill", "backward", "random_walks", "random_walks_backward", "turn_in_place", "side_step", "transition_all", "combine_all"] # "transition_all"] # "turn_in_place", "side_step", "random_walks", "combine_all", "transition_all"]
        self.max_behavior_curriculum = len(self.behaviors) - 1

        self.from_net = kwargs.pop("from_net", False)

        self.heading_errors = []
        self.met_times = []
        self.dist_errors = []
        self.heading_bonus_weight = kwargs.pop("heading_bonus_weight", 8)
        self.gauss_width = kwargs.pop("gauss_width", 10)
        self.legs_bonus = 0
        self.heading_bonus = 0
        self.tilt_bonus_weight = 1
        self.timing_bonus = 0
        self.timing_bonus_weight = kwargs.pop("timing_bonus_weight", 2)

        self.current_step_time = 0
        self.current_time_index = 1

        self.mask_info = {
            "xy": [False, 0.5, False],
            "heading": [True, 0.5, False],
            "timing": [True, 0.5, False],
            "leg": [False, 0.5, False],
            "dir": [False, 0.5, True],
            "vel": [False, 0.5, True],
        }

        self.foot_angle_weight = kwargs.pop("foot_angle_weight", 0.1)

        self.past_last_step = False
        self.reached_last_step = False
        self.finished_all = False

        self.determine = kwargs.pop("determine", False)

        self.selected_curriculum = 0

        # Robot settings
        N = self.max_curriculum + 1
        self.terminal_height_curriculum = np.linspace(0.75, 0.45, N)
        self.applied_gain_curriculum = np.linspace(1.2, 1.2, N)
        self.electricity_cost = 4.5 / self.robot.action_space.shape[0]
        self.stall_torque_cost = 0.225 / self.robot.action_space.shape[0]
        self.joints_at_limit_cost = 0.1

        # Env settings
        self.next_step_index = self.lookbehind

        self.elbow_penalty = 0
        self.elbow_weight = 0.4

        self.selected_behavior = self.behaviors[0]

        # Terrain info
        self.angle_curriculum = {
            "to_standstill": np.linspace(np.pi / 12, np.pi / 3, N),
            "random_walks": np.linspace(np.pi / 12, np.pi / 3, N),
            "random_walks_backward": np.linspace(np.pi / 12, np.pi / 3, N),
            "turn_in_place": np.linspace(0, np.pi / 2, N),
            "side_step": None,
            "backward": np.linspace(np.pi / 12, np.pi / 4, N),
            "heading_var": np.linspace(0, np.pi / 3 - np.pi / 8, N),
            "timing_gaits": np.linspace(0, np.pi / 3 - np.pi / 8, N),
        }
        self.dist_range = {
            "to_standstill": np.array([0.65, 0]),
            "random_walks": np.array([0.55, 0.75]),
            "random_walks_backward": np.array([-0.45, -0.65]),
            "turn_in_place": np.array([0.7, 0.1]),
            "side_step": np.array([0.2, 0.7]),
            "backward": np.array([0.0, -0.65]),
            "heading_var": np.array([0.65, 0.65]),
            "timing_gaits": np.array([0.65, 0.65]),
        }

        self.foot_sep_range = {
            "to_standstill": np.array([-0.04,0.16]),
            "random_walks": np.array([-0.04,0.10]),
            "random_walks_backward": np.array([-0.04,0.10]),
            "turn_in_place": np.array([-0.04,0.04]),
            "side_step": np.array([-0.04,0.04]),
            "backward": np.array([-0.04,0.12]),
            "heading_var": np.array([-0.04,0.16]),
            "timing_gaits": np.array([-0.04,0.16]),
        }

        self.dr_curriculum = {k: np.linspace(*dist_range, N) for k, dist_range in self.dist_range.items()}
        self.pitch_range = np.array([0, 0])  # degrees
        self.tilt_range = np.array([0, 0])
        self.yaw_range = {
            "to_standstill": np.array([0.0, 0.0]),
            "random_walks": np.array([-70.0, 70.0]),
            "random_walks_backward": np.array([-70.0, 70.0]),
            "turn_in_place": np.array([0.0, 0.0]),
            "side_step": np.array([0.0, 0.0]),
            "backward": np.array([0.0, 0.0]),
            "heading_var": np.array([0.0, 0.0]),
            "timing_gaits": np.array([0.0, 0.0]),
        }

        self.generated_paths_cache = {
            # "to_standstill":  [[None, None] for _ in range(self.max_curriculum+1)],
            # "turn_in_place": [[None, None] for _ in range(self.max_curriculum+1)],
            # "side_step": [[None, None] for _ in range(self.max_curriculum+1)],
            # "backward": [[None, None] for _ in range(self.max_curriculum+1)],
        }

        self.step_param_dim = 7 + 3 # steps and mask
        # Important to do this once before reset!
        self.swing_leg = 0
        self.terrain_info = self.generate_step_placements()

        # Observation and Action spaces
        self.robot_obs_dim = self.robot.observation_space.shape[0]
        K = self.lookahead + self.lookbehind
        self.extra_step_dim = 4 + 2 + 1 + 3 # timing + direction 2d vec + general velocity + mask
        high = np.inf * np.ones(
            self.robot_obs_dim + K * self.step_param_dim + self.extra_step_dim, dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

        # pre-allocate buffers
        F = len(self.robot.feet)
        self._foot_target_contacts = np.zeros((F, 1), dtype=np.float32)
        self.foot_dist_to_target = np.zeros(F, dtype=np.float32)

    def get_timing(self, N):
        if self.selected_curriculum == 0 and self.behaviors.index(self.selected_behavior) == 0:
            half_cycle_times = np.ones(N) * 30
            timing_0 = half_cycle_times * 0.3
            timing_1 = half_cycle_times * 0.7
        elif self.behaviors.index(self.selected_behavior) == 0:
            half_cycle_times = np.ones(N) * self.np_random.choice([30,40,50])
            timing_0 = half_cycle_times * 0.3
            timing_1 = half_cycle_times * 0.7
        else:
            half_cycle_times = self.np_random.choice([10,20,30,40,50], size=N)
            ground_ratio = self.np_random.choice([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7], size=N)
            half_cycle_times[(ground_ratio >= 0.3) & (half_cycle_times < 30)] = 30
            ground_ratio[(ground_ratio <= 0.1) & (half_cycle_times >= 50)] = 0.2
            timing_0 = half_cycle_times * ground_ratio
            timing_1 = half_cycle_times * (1-ground_ratio)
            half_cycle_times[0:3] = 30
            timing_0[0:3] = half_cycle_times[0:3] * 0.3
            timing_1[0:3] = half_cycle_times[0:3] * 0.7

        timing_0 = timing_0.astype(int)
        timing_1 = timing_1.astype(int)
        timing_2 = timing_0 + timing_1
        timing_3 = np.zeros(N)

        # make first step shorter
        timing_2[0] -= timing_0[0]
        timing_0[0] = 0

        timing_2[1] -= timing_0[1]
        timing_0[1] = 0

        assert (timing_0 + timing_1 == timing_2 + timing_3).all(), f"{timing_0 + timing_1} vs {timing_2+ timing_3}"

        return timing_0, timing_1, timing_2, timing_3

    def generate_timing_gaits_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        ratio = curriculum / self.max_curriculum if self.max_curriculum > 0 else 0

        behavior = "timing_gaits"

        method = "walking"

        yaw_range = self.yaw_range[self.selected_behavior] * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        self.path_angle = self.angle_curriculum[behavior][curriculum]

        N = self.num_steps
        
        self.dr_spacing = self.dr_curriculum[self.selected_behavior][curriculum]
        dr = np.zeros(N) + self.dr_spacing

        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = self.init_step_separation
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dphi[2] = 0.0

        x_tilt[0:2] = 0
        y_tilt[0:2] = 0

        # Update x and y arrays
        if method != "hopping":
            swing_legs = np.ones(N, dtype=np.int8)
            swing_legs[:N:2] = 0 # Set swing_legs to 1 at every second index starting from 0
        else:
            swing_legs = np.ones(N, dtype=np.int8)
            swing_legs[:N:2] = 0 # Set swing_legs to 1 at every second index starting from 0
            # start hopping at index 4
            swing_legs[4:6] = swing_legs[3]
            # walking
            swing_legs[[6,8]] = 1 - swing_legs[3]
            swing_legs[7] = swing_legs[3]
            # hopping
            swing_legs[9:13] = swing_legs[8]
            # walking
            swing_legs[13] = 1 - swing_legs[8]
            # hopping
            swing_legs[14:] = swing_legs[13]

        dphi[self.stop_steps[1::2]] = 0
        dphi = np.cumsum(dphi)

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        dy[self.stop_steps[1::2]] = 0
        dx[self.stop_steps[1::2]] = 0

        heading_targets = np.copy(dphi)

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        foot_sep_range = self.foot_sep_range[behavior]
        foot_seps = self.foot_sep + self.np_random.uniform(*foot_sep_range, size=N)

        # Calculate shifts
        left_shifts = np.array([np.cos(heading_targets + np.pi / 2), np.sin(heading_targets + np.pi / 2)])
        right_shifts = np.array([np.cos(heading_targets - np.pi / 2), np.sin(heading_targets - np.pi / 2)])

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        x += np.where(swing_legs == 1, left_shifts[0], right_shifts[0]) * foot_seps
        y += np.where(swing_legs == 1, left_shifts[1], right_shifts[1]) * foot_seps

        if self.robot.mirrored:
            x *= -1
        else:
            swing_legs = 1 - swing_legs
            heading_targets *= -1

        # switched dy and dx before, so need to rectify
        heading_targets += 90 * DEG2RAD

        dphi *= 0

        if curriculum == 0:
            half_cycle_times = np.ones(N) * self.np_random.choice([30,40,50])
        else:
            if curriculum <= 2:
                cycle_choices = [20,30,40,50]
            else:
                cycle_choices = [10,20,30,40,50,60]
            if self.np_random.rand() < 0.5:
                half_cycle_times = np.ones(N) * self.np_random.choice(cycle_choices)
            else:
                half_cycle_times = self.np_random.choice(cycle_choices, size=N)
        
        half_cycle_times[0:3] = 30 # to start properly

        if method == "walking":
            timing_0 = half_cycle_times * 0.3
            timing_1 = half_cycle_times * 0.7
            if curriculum > 0:
                if curriculum == 1:
                    ratios = [0.2,0.3,0.4]
                elif curriculum == 2:
                    ratios = [0.1,0.2,0.3,0.4]
                elif curriculum == 3:
                    ratios = [0.1,0.2,0.3,0.4,0.5]
                elif curriculum == 4:
                    ratios = [0.0,0.1,0.2,0.3,0.4,0.5,0.6]
                else:
                    ratios = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
                ground_ratio = self.np_random.choice(ratios, size=N)
                half_cycle_times[(ground_ratio >= 0.3) & (half_cycle_times < 30)] = 30
                ground_ratio[(ground_ratio <= 0.1) & (half_cycle_times >= 50)] = 0.2
                timing_0 = half_cycle_times * ground_ratio
                timing_1 = half_cycle_times * (1-ground_ratio)
                half_cycle_times[0:3] = 30
                timing_0[0:3] = half_cycle_times[0:3] * 0.3
                timing_1[0:3] = half_cycle_times[0:3] * 0.7
                # ratio = self.np_random.choice([0.3, 0.4, 0.5])
                # timing_0 = half_cycle_times * ratio
                # timing_1 = half_cycle_times * (1-ratio)
            timing_0 = timing_0.astype(int)
            timing_1 = timing_1.astype(int)
            timing_2 = timing_0 + timing_1
            timing_3 = np.zeros(N)

            # make first step shorter
            timing_2[0] -= timing_0[0]
            timing_0[0] = 0

            timing_2[1] -= timing_0[1]
            timing_0[1] = 0

        elif method == "running":
            timing_0 = np.zeros(N)
            timing_1 = half_cycle_times
            timing_2 = half_cycle_times * 0.8
            timing_3 = half_cycle_times * 0.2
            timing_2 = timing_2.astype(int)
            timing_3 = timing_3.astype(int)
        elif method == "hopping":
            walking_cycle = 30
            hopping_cycle = 40

            timing_0 = np.zeros(N)
            timing_1 = np.zeros(N)
            timing_2 = np.zeros(N)
            timing_3 = np.zeros(N)

            # walking
            timing_0[0:4] = (walking_cycle * 0.4)
            timing_1[0:4] = (walking_cycle * 0.6)
            timing_0 = timing_0.astype(int)
            timing_1 = timing_1.astype(int)
            timing_2[0:4] = timing_0[0:4] + timing_1[0:4]

            # hopping
            timing_0[4:6] = (hopping_cycle * 0.6)
            timing_1[4:6] = (hopping_cycle * 0.4)
            timing_0 = timing_0.astype(int)
            timing_1 = timing_1.astype(int)
            timing_2[4:6] = 0
            timing_2[4] = 4

            # walking
            timing_0[6] = 0
            timing_1[6] = walking_cycle
            timing_0[7:9] = (walking_cycle * 0.4)
            timing_1[7:9] = (walking_cycle * 0.6)
            timing_0 = timing_0.astype(int)
            timing_1 = timing_1.astype(int)
            timing_2[6:9] = timing_0[6:9] + timing_1[6:9]

            # hopping
            timing_0[9:13] = (hopping_cycle * 0.6)
            timing_1[9:13] = (hopping_cycle * 0.4)
            timing_0 = timing_0.astype(int)
            timing_1 = timing_1.astype(int)
            timing_2[9:13] = 0
            timing_2[9] = 4

            # walking
            timing_0[13] = 0
            timing_1[13] = walking_cycle
            timing_0 = timing_0.astype(int)
            timing_1 = timing_1.astype(int)
            timing_2[13] = timing_0[13] + timing_1[13]

            # hopping
            timing_0[14:] = (hopping_cycle * 0.6)
            timing_1[14:] = (hopping_cycle * 0.4)
            timing_0 = timing_0.astype(int)
            timing_1 = timing_1.astype(int)
            timing_2[14:] = 0
            timing_2[14] = 4

            # make first step shorter
            timing_2[0] -= timing_0[0]
            timing_0[0] = 0

            timing_2[1] -= timing_0[1]
            timing_0[1] = 0

            timing_3 = timing_0 + timing_1 - timing_2

        assert (timing_0 + timing_1 == timing_2 + timing_3).all(), f"{timing_0 + timing_1} vs {timing_2+ timing_3}"

        path_angle_possibilities = np.linspace(-self.path_angle, self.path_angle, num=curriculum * 2 + 3, endpoint=True)
        heading_targets[3:] += self.np_random.choice(path_angle_possibilities, size=(N-3))
        
        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets, swing_legs, timing_0, timing_1, timing_2, timing_3, foot_seps), axis=1)

    def generate_to_standstill_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        behavior = 'to_standstill'
        ratio = curriculum / self.max_curriculum

        # {self.max_curriculum + 1} levels in total
        yaw_range = self.yaw_range[behavior] * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        self.path_angle = self.angle_curriculum[behavior][curriculum]
        path_angle_possibilities = np.linspace(-self.path_angle, self.path_angle, num=curriculum * 2 + 3, endpoint=True)

        N = self.num_steps
        
        self.dr_spacing = self.dr_curriculum[behavior][curriculum]
        dr = np.zeros(N) + self.dr_spacing

        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = self.init_step_separation
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dphi[2] = 0.0

        x_tilt[0:2] = 0
        y_tilt[0:2] = 0

        swing_legs = np.ones(N, dtype=np.int8)

        # Update x and y arrays
        swing_legs[:N:2] = 0  # Set swing_legs to 1 at every second index starting from 0

        dphi[self.stop_steps[1::2]] = 0
        dphi = np.cumsum(dphi)

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        dy[self.stop_steps[1::2]] = 0
        dx[self.stop_steps[1::2]] = 0

        heading_targets = np.copy(dphi)

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        foot_sep_range = self.foot_sep_range[behavior]
        foot_seps = self.foot_sep + self.np_random.uniform(*foot_sep_range, size=N)

        # Calculate shifts
        left_shifts = np.array([np.cos(heading_targets + np.pi / 2), np.sin(heading_targets + np.pi / 2)])
        right_shifts = np.array([np.cos(heading_targets - np.pi / 2), np.sin(heading_targets - np.pi / 2)])

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        x += np.where(swing_legs == 1, left_shifts[0], right_shifts[0]) * foot_seps
        y += np.where(swing_legs == 1, left_shifts[1], right_shifts[1]) * foot_seps

        if self.robot.mirrored:
            x *= -1
        else:
            swing_legs = 1 - swing_legs
            heading_targets *= -1

        # switched dy and dx before, so need to rectify
        heading_targets += 90 * DEG2RAD
        heading_targets[3:] += self.np_random.choice(path_angle_possibilities, size=(N-3))

        dphi *= 0

        timing_0, timing_1, timing_2, timing_3 = self.get_timing(N)

        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets, swing_legs, timing_0, timing_1, timing_2, timing_3, foot_seps), axis=1)
    
    def generate_backward_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        behavior = 'backward'
        ratio = curriculum / self.max_curriculum

        # {self.max_curriculum + 1} levels in total
        yaw_range = self.yaw_range[behavior] * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        self.path_angle = self.angle_curriculum[behavior][curriculum]
        path_angle_possibilities = np.linspace(-self.path_angle, self.path_angle, num=curriculum * 2 + 3, endpoint=True)

        N = self.num_steps
        
        self.dr_spacing = self.dr_curriculum[behavior][curriculum]
        dr = np.zeros(N) + self.dr_spacing

        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = self.init_step_separation
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dphi[2] = 0.0

        x_tilt[0:2] = 0
        y_tilt[0:2] = 0

        swing_legs = np.ones(N, dtype=np.int8)

        # Update x and y arrays
        swing_legs[:N:2] = 0  # Set swing_legs to 1 at every second index starting from 0

        dphi[self.stop_steps[1::2]] = 0
        dphi = np.cumsum(dphi)

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        dy[self.stop_steps[1::2]] = 0
        dx[self.stop_steps[1::2]] = 0

        heading_targets = np.copy(dphi)

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        foot_sep_range = self.foot_sep_range[behavior]
        foot_seps = self.foot_sep + self.np_random.uniform(*foot_sep_range, size=N)

        # Calculate shifts
        left_shifts = np.array([np.cos(heading_targets + np.pi / 2), np.sin(heading_targets + np.pi / 2)])
        right_shifts = np.array([np.cos(heading_targets - np.pi / 2), np.sin(heading_targets - np.pi / 2)])

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        x += np.where(swing_legs == 1, left_shifts[0], right_shifts[0]) * foot_seps
        y += np.where(swing_legs == 1, left_shifts[1], right_shifts[1]) * foot_seps

        if self.robot.mirrored:
            x *= -1
        else:
            swing_legs = 1 - swing_legs
            heading_targets *= -1

        # switched dy and dx before, so need to rectify
        heading_targets += 90 * DEG2RAD
        heading_targets[3:] += self.np_random.choice(path_angle_possibilities, size=(N-3))

        dphi *= 0

        timing_0, timing_1, timing_2, timing_3 = self.get_timing(N)

        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets, swing_legs, timing_0, timing_1, timing_2, timing_3, foot_seps), axis=1)
        
    def generate_random_walks_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        ratio = curriculum / self.max_curriculum
        behavior = 'random_walks'

        # {self.max_curriculum + 1} levels in total
        yaw_range = self.yaw_range[behavior] * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        self.path_angle = self.angle_curriculum[behavior][curriculum]
        path_angle_possibilities = np.linspace(-self.path_angle, self.path_angle, num=curriculum * 2 + 3, endpoint=True)

        N = self.num_steps

        self.dr_spacing = self.dr_curriculum[behavior][curriculum]

        dr = np.zeros(N) + self.dr_spacing

        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = self.init_step_separation
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dphi[2] = 0.0

        x_tilt[0:2] = 0
        y_tilt[0:2] = 0

        swing_legs = np.ones(N, dtype=np.int8)

        # Update x and y arrays
        swing_legs[:N:2] = 0  # Set swing_legs to 1 at every second index starting from 0

        dphi[self.stop_steps[1::2]] = 0
        dphi = np.cumsum(dphi)

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        dy[self.stop_steps[1::2]] = 0
        dx[self.stop_steps[1::2]] = 0

        heading_targets = np.copy(dphi)

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        foot_sep_range = self.foot_sep_range[behavior]
        foot_seps = self.foot_sep + self.np_random.uniform(*foot_sep_range, size=N)

        # Calculate shifts
        left_shifts = np.array([np.cos(heading_targets + np.pi / 2), np.sin(heading_targets + np.pi / 2)])
        right_shifts = np.array([np.cos(heading_targets - np.pi / 2), np.sin(heading_targets - np.pi / 2)])

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        x += np.where(swing_legs == 1, left_shifts[0], right_shifts[0]) * foot_seps
        y += np.where(swing_legs == 1, left_shifts[1], right_shifts[1]) * foot_seps

        if self.robot.mirrored:
            x *= -1
        else:
            swing_legs = 1 - swing_legs
            heading_targets *= -1

        # switched dy and dx before, so need to rectify
        heading_targets += 90 * DEG2RAD
        heading_targets[3:] += self.np_random.choice(path_angle_possibilities, size=(N-3))

        dphi *= 0

        timing_0, timing_1, timing_2, timing_3 = self.get_timing(N)

        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets, swing_legs, timing_0, timing_1, timing_2, timing_3, foot_seps), axis=1)
    
    def generate_random_walks_backward_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        ratio = curriculum / self.max_curriculum
        behavior = 'random_walks_backward'

        # {self.max_curriculum + 1} levels in total
        yaw_range = self.yaw_range[behavior] * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        self.path_angle = self.angle_curriculum[behavior][curriculum]
        path_angle_possibilities = np.linspace(-self.path_angle, self.path_angle, num=curriculum * 2 + 3, endpoint=True)

        N = self.num_steps

        self.dr_spacing = self.dr_curriculum[behavior][curriculum]

        dr = np.zeros(N) + self.dr_spacing

        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = self.init_step_separation
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dphi[2] = 0.0

        x_tilt[0:2] = 0
        y_tilt[0:2] = 0

        swing_legs = np.ones(N, dtype=np.int8)

        # Update x and y arrays
        swing_legs[:N:2] = 0  # Set swing_legs to 1 at every second index starting from 0

        dphi[self.stop_steps[1::2]] = 0
        dphi = np.cumsum(dphi)

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        dy[self.stop_steps[1::2]] = 0
        dx[self.stop_steps[1::2]] = 0

        heading_targets = np.copy(dphi)

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        foot_sep_range = self.foot_sep_range[behavior]
        foot_seps = self.foot_sep + self.np_random.uniform(*foot_sep_range, size=N)

        # Calculate shifts
        left_shifts = np.array([np.cos(heading_targets + np.pi / 2), np.sin(heading_targets + np.pi / 2)])
        right_shifts = np.array([np.cos(heading_targets - np.pi / 2), np.sin(heading_targets - np.pi / 2)])

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        x += np.where(swing_legs == 1, left_shifts[0], right_shifts[0]) * foot_seps
        y += np.where(swing_legs == 1, left_shifts[1], right_shifts[1]) * foot_seps

        if self.robot.mirrored:
            x *= -1
        else:
            swing_legs = 1 - swing_legs
            heading_targets *= -1

        # switched dy and dx before, so need to rectify
        heading_targets += 90 * DEG2RAD
        heading_targets[3:] += self.np_random.choice(path_angle_possibilities, size=(N-3))

        dphi *= 0

        timing_0, timing_1, timing_2, timing_3 = self.get_timing(N)

        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets, swing_legs, timing_0, timing_1, timing_2, timing_3, foot_seps), axis=1)
    
    def generate_heading_var_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        ratio = curriculum / self.max_curriculum
        behavior = 'heading_var'

        # {self.max_curriculum + 1} levels in total
        yaw_range = self.yaw_range[behavior] * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        self.path_angle = self.angle_curriculum[behavior][curriculum]

        N = self.num_steps

        self.dr_spacing = self.dr_curriculum[behavior][curriculum]

        dr = np.zeros(N) + self.dr_spacing

        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = self.init_step_separation
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dphi[2] = 0.0

        x_tilt[0:2] = 0
        y_tilt[0:2] = 0

        swing_legs = np.ones(N, dtype=np.int8)

        # Update x and y arrays
        swing_legs[:N:2] = 0  # Set swing_legs to 1 at every second index starting from 0

        dphi[self.stop_steps[1::2]] = 0
        dphi = np.cumsum(dphi)

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        dy[self.stop_steps[1::2]] = 0
        dx[self.stop_steps[1::2]] = 0

        heading_targets = np.copy(dphi)

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        foot_sep_range = self.foot_sep_range[behavior] * ratio
        if self.np_random.rand() < 0.2:
            foot_seps = self.foot_sep + self.np_random.uniform(*foot_sep_range, size=N)
        else:
            foot_seps = self.foot_sep + self.np_random.choice(foot_sep_range, size=N)

        # Calculate shifts
        left_shifts = np.array([np.cos(heading_targets + np.pi / 2), np.sin(heading_targets + np.pi / 2)])
        right_shifts = np.array([np.cos(heading_targets - np.pi / 2), np.sin(heading_targets - np.pi / 2)])

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        x += np.where(swing_legs == 1, left_shifts[0], right_shifts[0]) * foot_seps
        y += np.where(swing_legs == 1, left_shifts[1], right_shifts[1]) * foot_seps

        if self.robot.mirrored:
            x *= -1
        else:
            swing_legs = 1 - swing_legs
            heading_targets *= -1

        # switched dy and dx before, so need to rectify
        heading_targets += 90 * DEG2RAD

        path_angle_possibilities = np.linspace(-self.path_angle, self.path_angle, num=curriculum * 2 + 3, endpoint=True)

        heading_targets[3:] += self.np_random.choice(path_angle_possibilities, size=(N-3))

        dphi *= 0

        timing_0, timing_1, timing_2, timing_3 = self.get_timing(N)

        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets, swing_legs, timing_0, timing_1, timing_2, timing_3, foot_seps), axis=1)

    def get_random_flip_array_every_5(self, N):
        flip_array = np.zeros(N + 1, dtype=np.int8)
        toggle_indices = np.arange(0, N + 1, 5)
        random_choices = self.np_random.choice([True, False], size=len(toggle_indices), p=[0.3,0.7])
        flip_array[toggle_indices[random_choices]] = 1
        toggle_cumsum = np.cumsum(flip_array)
        toggle_cumsum = toggle_cumsum[:-1]
        flip_array = flip_array[:-1]
        flip_array[toggle_cumsum % 2 == 1] = 1
        flip_array[toggle_cumsum % 2 == 0] = 0
        return flip_array

    def generate_turn_in_place_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        ratio = curriculum / self.max_curriculum
        behavior = 'turn_in_place'

        # {self.max_curriculum + 1} levels in total
        yaw_range = self.yaw_range[behavior] * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        self.path_angle = self.angle_curriculum[behavior][curriculum]

        N = self.num_steps

        self.dr_spacing = self.dr_curriculum[behavior][curriculum]
        dr = np.zeros(N) + self.dr_spacing

        dphi = self.np_random.uniform(*yaw_range, size=N) + self.path_angle
        # dphi_flip = self.get_random_flip_array_every_5(N)
        # dphi[dphi_flip.astype(bool)] *= -1
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = self.init_step_separation
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        # dphi[2] = 0.0

        x_tilt[0:2] = 0
        y_tilt[0:2] = 0

        swing_legs = np.ones(N, dtype=np.int8)

        # Update x and y arrays
        swing_legs[:N:2] = 0  # Set swing_legs to 1 at every second index starting from 0

        dphi[self.stop_steps[1::2]] = 0
        dphi = np.cumsum(dphi)

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        dy[self.stop_steps[1::2]] = 0
        dx[self.stop_steps[1::2]] = 0

        heading_targets = np.copy(dphi)

        x = np.roll(np.repeat(dx[:N//2], 2),-1)
        y = np.roll(np.repeat(dy[:N//2], 2),-1)
        z = np.roll(np.repeat(dz[:N//2], 2),-1)
        y[3:] += self.init_step_separation - self.dr_spacing
        heading_targets = np.roll(np.repeat(heading_targets[:N//2], 2),-1)

        foot_sep_range = self.foot_sep_range[behavior]
        foot_seps = self.foot_sep + self.np_random.uniform(*foot_sep_range, size=N)

        # Calculate shifts
        left_shifts = np.array([np.cos(heading_targets + np.pi / 2), np.sin(heading_targets + np.pi / 2)])
        right_shifts = np.array([np.cos(heading_targets - np.pi / 2), np.sin(heading_targets - np.pi / 2)])

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        x += np.where(swing_legs == 1, left_shifts[0], right_shifts[0]) * foot_seps
        y += np.where(swing_legs == 1, left_shifts[1], right_shifts[1]) * foot_seps

        if self.robot.mirrored:
            x *= -1
        else:
            swing_legs = 1 - swing_legs
            heading_targets *= -1

        # switched dy and dx before, so need to rectify
        heading_targets += 90 * DEG2RAD

        dphi *= 0

        x[-1] = x[-3]
        y[-1] = y[-3]
        z[-1] = z[-3]
        heading_targets[-1] = heading_targets[-3]

        x[-2] = x[-4]
        y[-2] = y[-4]
        z[-2] = z[-4]
        heading_targets[-2] = heading_targets[-4]

        timing_0, timing_1, timing_2, timing_3 = self.get_timing(N)

        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets, swing_legs, timing_0, timing_1, timing_2, timing_3, foot_seps), axis=1)

    def generate_side_step_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        ratio = curriculum / self.max_curriculum
        behavior = 'side_step'

        # {self.max_curriculum + 1} levels in total
        yaw_range = self.yaw_range[behavior] * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        self.path_angle = 0

        N = self.num_steps

        self.dr_spacing = self.dr_curriculum[behavior][curriculum]

        dr = np.zeros(N) + self.dr_spacing

        dphi = self.np_random.uniform(*yaw_range, size=N) + self.path_angle
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = self.init_step_separation
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        # dphi[2] = 0.0

        x_tilt[0:2] = 0
        y_tilt[0:2] = 0

        swing_legs = np.ones(N, dtype=np.int8)

        # Update x and y arrays
        swing_legs[:N:2] = 0  # Set swing_legs to 1 at every second index starting from 0

        dphi[self.stop_steps[1::2]] = 0
        dphi = np.cumsum(dphi)

        dy = dr * np.sin(dtheta) * np.cos(dphi)
        dx = dr * np.sin(dtheta) * np.sin(dphi)
        dx[2:] += self.dr_spacing
        dx_flip = self.get_random_flip_array_every_5(N)
        dx[dx_flip.astype(bool)] *= -1
        dz = dr * np.cos(dtheta)

        dy[self.stop_steps[1::2]] = 0
        dx[self.stop_steps[1::2]] = 0

        heading_targets = np.copy(dphi)

        x = np.roll(np.repeat(np.cumsum(dx[:N//2]), 2),-1)
        y = np.roll(np.repeat(dy[:N//2], 2),-1)
        z = np.roll(np.repeat(dz[:N//2], 2),-1)
        y[3:] += self.init_step_separation - self.dr_spacing
        heading_targets = np.roll(np.repeat(heading_targets[:N//2], 2),-1)

        foot_sep_range = self.foot_sep_range[behavior]
        foot_seps = self.foot_sep + self.np_random.uniform(*foot_sep_range, size=N)

        # Calculate shifts
        left_shifts = np.array([np.cos(heading_targets + np.pi / 2), np.sin(heading_targets + np.pi / 2)])
        right_shifts = np.array([np.cos(heading_targets - np.pi / 2), np.sin(heading_targets - np.pi / 2)])

        # Flip the shifts
        left_shifts = np.flip(left_shifts, axis=0)
        right_shifts = np.flip(right_shifts, axis=0)

        x += np.where(swing_legs == 1, left_shifts[0], right_shifts[0]) * foot_seps
        y += np.where(swing_legs == 1, left_shifts[1], right_shifts[1]) * foot_seps

        if self.robot.mirrored:
            x *= -1
        else:
            swing_legs = 1 - swing_legs
            heading_targets *= -1

        # switched dy and dx before, so need to rectify
        heading_targets += 90 * DEG2RAD

        dphi *= 0

        x[-1] = x[-3]
        y[-1] = y[-3]
        z[-1] = z[-3]
        heading_targets[-1] = heading_targets[-3]

        x[-2] = x[-4]
        y[-2] = y[-4]
        z[-2] = z[-4]
        heading_targets[-2] = heading_targets[-4]

        timing_0, timing_1, timing_2, timing_3 = self.get_timing(N)

        return np.stack((x, y, z, dphi, x_tilt, y_tilt, heading_targets, swing_legs, timing_0, timing_1, timing_2, timing_3, foot_seps), axis=1)
    
    def generate_transition_all_step_placements(self, curriculum):
        # Check just in case
        curriculum = min(curriculum, self.max_curriculum)
        
        step_placement_fcns = [
            (self.generate_to_standstill_step_placements, "to_standstill"),
            (self.generate_turn_in_place_step_placements, "turn_in_place"),
            (self.generate_side_step_step_placements, "side_step"),
            (self.generate_random_walks_step_placements, "random_walks"),
            (self.generate_random_walks_step_placements, "random_walks_backward"),
            (self.generate_random_walks_step_placements, "backward"),
        ]

        # randomly pick 3, rotate steps to match last heading of previous and shift
        selected_step_placement_fcns = self.np_random.choice(step_placement_fcns, 5)

        step_placements = None

        transition_indices = [4,8,12,16,self.num_steps]

        for i, selected_step_placement_fcn_tuple in enumerate(selected_step_placement_fcns):
            selected_step_placement_fcn, behavior_str = selected_step_placement_fcn_tuple
            selected_step_curriculum = self.np_random.choice(list(range(0,curriculum+1)))
            if behavior_str in self.generated_paths_cache and self.generated_paths_cache[behavior_str][selected_step_curriculum][int(self.robot.mirrored)] is not None:
                step_placements_part = np.copy(self.generated_paths_cache[behavior_str][selected_step_curriculum][int(self.robot.mirrored)])
            else:
                step_placements_part = selected_step_placement_fcn(selected_step_curriculum)
                if behavior_str in self.generated_paths_cache:
                    self.generated_paths_cache[behavior_str][selected_step_curriculum][int(self.robot.mirrored)] = np.copy(step_placements_part)

            if i == 0:
                step_placements = step_placements_part
            a = transition_indices[i-1]
            b = transition_indices[i]
            heading_shift = -(step_placements_part[a-1, 6] - step_placements[a-1, 6])
            dx = step_placements_part[a:b,0] - step_placements_part[a-1,0]
            dy = step_placements_part[a:b,1] - step_placements_part[a-1,1]
            step_placements_part[a:b,0] = step_placements_part[a-1,0] + dx * np.cos(heading_shift) - dy * np.sin(heading_shift)
            step_placements_part[a:b,1] = step_placements_part[a-1,1] + dx * np.sin(heading_shift) + dy * np.cos(heading_shift)
            step_placements_part[a:b, 6] += heading_shift

            x_shift = step_placements_part[a-1, 0] - step_placements[a-1, 0]
            step_placements_part[a:b, 0] -= x_shift
            y_shift = step_placements_part[a-1, 1] - step_placements[a-1, 1]
            step_placements_part[a:b, 1] -= y_shift
            step_placements[a:b:, :] = step_placements_part[a:b:, :]
        return step_placements

    def generate_step_placements(self):
        self.curriculum = min(self.curriculum, self.max_curriculum)
        self.behavior_curriculum = min(self.behavior_curriculum, self.max_behavior_curriculum)

        factor = 0 if self.determine else 0.35
        train_on_past = self.np_random.rand() < factor and self.behavior_curriculum != 0

        if self.behaviors[self.behavior_curriculum] == "combine_all":
            self.selected_curriculum = self.np_random.choice(list(range(0,self.curriculum+1)))
            self.selected_behavior = self.np_random.choice(self.behaviors[0:self.behavior_curriculum])
        elif self.determine:
            self.selected_curriculum = self.curriculum
            self.selected_behavior = self.behaviors[self.behavior_curriculum]
        else:
            if train_on_past:
                self.selected_curriculum = self.np_random.choice(list(range(0,self.curriculum+1)))
                self.selected_behavior = self.np_random.choice(self.behaviors[0:self.behavior_curriculum])
            else:
                weights = np.linspace(1,10,self.curriculum+1)
                weights /= sum(weights)
                self.selected_curriculum = self.np_random.choice(list(range(0,self.curriculum+1)), p=weights)
                self.selected_behavior = self.behaviors[self.behavior_curriculum]

        if self.selected_behavior in self.generated_paths_cache and self.generated_paths_cache[self.selected_behavior][self.selected_curriculum][int(self.robot.mirrored)] is not None:
            return self.generated_paths_cache[self.selected_behavior][self.selected_curriculum][int(self.robot.mirrored)]

        if self.selected_behavior == "to_standstill":
            if self.np_random.rand() < 0.8:
                path = self.generate_to_standstill_step_placements(self.selected_curriculum)
            else:
                path = self.generate_random_walks_step_placements(min(self.selected_curriculum + 1, self.max_curriculum))
        elif self.selected_behavior == "heading_var":
            path = self.generate_heading_var_step_placements(self.selected_curriculum)
        elif self.selected_behavior == "turn_in_place":
            path = self.generate_turn_in_place_step_placements(self.selected_curriculum)
        elif self.selected_behavior == "side_step":
            path = self.generate_side_step_step_placements(self.selected_curriculum)
        elif self.selected_behavior == "random_walks":
            path = self.generate_random_walks_step_placements(self.selected_curriculum)
        elif self.selected_behavior == "backward":
            path = self.generate_backward_step_placements(self.selected_curriculum)
        elif self.selected_behavior == "random_walks_backward":
            path = self.generate_random_walks_backward_step_placements(self.selected_curriculum)
        elif self.selected_behavior in {"transition_all", "combine_all"}:
            self.selected_behavior = "transition_all"
            path = self.generate_transition_all_step_placements(self.selected_curriculum)
        elif self.selected_behavior == "timing_gaits":
            path = self.generate_timing_gaits_step_placements(self.selected_curriculum)
        else:
            raise NotImplementedError(f"Behavior {self.selected_behavior} is not implemented")
        
        if self.selected_behavior in self.generated_paths_cache:
            self.generated_paths_cache[self.selected_behavior][self.selected_curriculum][int(self.robot.mirrored)] = np.copy(path)

        return path

    def create_terrain(self):

        self.steps = []
        self.rendered_steps = []
        self.rendered_feet = []
        step_ids = set()
        cover_ids = set()

        options = {
            # self._p.URDF_ENABLE_SLEEPING |
            "flags": self._p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        }

        if self.is_rendered or self.use_egl:
            for index in range(self.rendered_step_count):
                self.rendered_steps.append(VCylinder(self._p, radius=self.step_radius, length=0.005, pos=None))
                self.rendered_feet.append(VFoot(self._p, pos=None))

        # Need set for detecting contact
        self.all_contact_object_ids = set(step_ids) | set(cover_ids)

        if not self.remove_ground:
            self.all_contact_object_ids |= self.ground_ids

    def set_step_state(self, info_index, step_index):
        # sets rendered step state
        pos = self.terrain_info[info_index, 0:3]
        phi, x_tilt, y_tilt = self.terrain_info[info_index, 3:6]
        quaternion = np.array(pybullet.getQuaternionFromEuler([x_tilt, y_tilt, phi]))
        heading = self.terrain_info[info_index, 6]
        left = self.terrain_info[info_index, 7]
        self.rendered_steps[step_index].set_position(pos=pos) #, quat=quaternion)
        new_pos = np.copy(pos)
        new_pos[2] += 0.005
        if self.mask_info["heading"][2]:
            self.rendered_feet[step_index].set_position(pos=[-100,-100,0])
        else:
            self.rendered_feet[step_index].set_position(pos=new_pos, heading=heading, left=left)

    def randomize_terrain(self, replace=True):
        if replace:
            self.terrain_info = self.generate_step_placements()
        if self.is_rendered or self.use_egl:
            for index in range(self.rendered_step_count):
                self.set_step_state(index, index)

    def update_steps(self):
        if self.rendered_step_count == self.num_steps or not (self.is_rendered or self.use_egl):
            return

        if self.next_step_index >= self.rendered_step_count:
            oldest = self.next_step_index % self.rendered_step_count
            next = min(self.next_step_index, len(self.terrain_info) - 1)
            self.set_step_state(next, oldest)

    def reset(self, force=False):
        if self.state_id >= 0:
            self._p.restoreState(self.state_id)

        self.timestep = 0
        self.done = False
        self.target_reached_count = 0
        self.both_feet_hit_ground = False
        self.current_target_count = 0
        self.in_air_count = 0
        self.swing_leg_lifted_count = 0
        self.swing_leg_lifted = False
        self.body_stationary_count = 0

        self.current_step_time = 0
        self.current_time_index = 1

        self.heading_errors = []
        self.met_times = []
        self.dist_errors = []
        self.past_last_step = False
        self.finished_all = False

        self.reached_last_step = False

        self.set_stop_on_next_step = False
        self.stop_on_next_step = False

        self.robot.applied_gain = self.applied_gain_curriculum[self.curriculum]
        prev_robot_mirrored = self.robot.mirrored

        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start,
            pos=self.robot_init_position,
            vel=self.robot_init_velocity,
            quat=self._p.getQuaternionFromEuler((0,0,-90 * RAD2DEG)),
            mirror=0 # if self.behavior_curriculum == 0 else 1 # 0 if random, 1 if force True, 2 if force False
        )
        self.prev_leg = self.swing_leg

        if self.mask_info["timing"][0]:
            threshold = self.mask_info["timing"][1] if (self.curriculum < 2 and self.behavior_curriculum == 0) else 0.4
            self.mask_info["timing"][2] = self.np_random.rand() < threshold
        if self.mask_info["heading"][0]:
            self.mask_info["heading"][2] = self.np_random.rand() < self.mask_info["heading"][1]

        # Randomize platforms
        replace = self.next_step_index >= self.num_steps / 2 or prev_robot_mirrored != self.robot.mirrored or force
        self.next_step_index = self.lookbehind
        self._prev_next_step_index = self.next_step_index - 1
        self.randomize_terrain(replace)
        self.swing_leg = int(self.terrain_info[self.next_step_index, 7])
        self.starting_leg = self.swing_leg
        self.prev_leg_pos = self.robot.feet_xyz[:, 0:2]
        self.calc_feet_state()

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)

            for step in self.rendered_steps:
                step.set_color(Colors["lightgrey"])

        self.targets, self.extra_param = self.delta_to_k_targets()
        assert self.targets.shape[-1] == self.step_param_dim
        assert (self.extra_step_dim == 0 and self.extra_param is None) or (self.extra_param.shape[0] == self.extra_step_dim)

        # Order is important because walk_target is set up above
        self.calc_potential()

        if self.extra_step_dim == 0:
            state = concatenate((self.robot_state, self.targets.flatten()))
        else:
            state = concatenate((self.robot_state, self.targets.flatten(), self.extra_param))

        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        return state

    def step(self, action):
        self.timestep += 1
        self.current_step_time += 1

        self.robot.apply_action(action)
        self.scene.global_step()

        # Stop on the 7th and 14th step, but need to specify N-1 as well
        self.set_stop_on_next_step = self.next_step_index in self.stop_steps

        # Don't calculate the contacts for now
        self.robot_state = self.robot.calc_state()
        self.calc_env_state(action)

        reward = self.progress - self.energy_penalty
        if not self.mask_info["xy"][2]:
            reward += self.step_bonus + self.target_bonus # - self.speed_penalty * 0
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty
        reward += self.legs_bonus - self.elbow_penalty * self.elbow_weight
        if not self.mask_info["heading"][2]:
            reward += self.heading_bonus * self.heading_bonus_weight
        if not self.mask_info["timing"][2]:
            reward += self.timing_bonus * self.timing_bonus_weight
        # else:
        #     reward += - self.speed_penalty # need to regulate speed if timing is not in the picture

        # print(f"REWARDS for {self.next_step_index}: progress {self.progress}, energy penalty {self.energy_penalty}, step bonus {self.step_bonus}, target {self.target_bonus}, speed penalty {self.speed_penalty}")
        # print(f"tall {self.tall_bonus}, posture penalty {self.posture_penalty}, joints penalty {self.joints_penalty}, legs {self.legs_bonus}, elbow {self.elbow_penalty * self.elbow_weight}, heading {self.heading_bonus * self.heading_bonus_weight}, timing {self.timing_bonus * self.timing_bonus_weight}")

        # targets is calculated by calc_env_state()
        if self.extra_step_dim == 0:
            state = concatenate((self.robot_state, self.targets.flatten()))
        else:
            state = concatenate((self.robot_state, self.targets.flatten(), self.extra_param))

        if self.is_rendered or self.use_egl:
            self._handle_keyboard(callback=self.handle_keyboard)
            self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)
            self.target.set_color(
                Colors["dodgerblue"]
                if self.distance_to_target < 0.15
                else Colors["lightgrey"]
            )
            self.rendered_steps[(self.next_step_index-1) % self.rendered_step_count].set_color(Colors["lightgrey"])
            self.rendered_steps[self.next_step_index % self.rendered_step_count].set_color(Colors["dodgerblue"])

        info = {}
        if self.done or self.timestep == self.max_timestep - 1:
            behavior_str_index = self.behaviors[self.behavior_curriculum]
            if (
                behavior_str_index == self.selected_behavior or behavior_str_index == "combine_all"
                and (
                    self.curriculum == self.selected_curriculum
                    or behavior_str_index == "combine_all"
                )
            ):
                if self.next_step_index == self.num_steps - 1 and self.reached_last_step:
                    info["curriculum_metric"] = self.next_step_index + 1
                else:
                    info["curriculum_metric"] = self.next_step_index
                info["avg_heading_err"] = nanmean(self.heading_errors)
                info["avg_timing_met"] = nanmean(self.met_times)
                info["mask_combo_id"] = 2 * int(self.mask_info["heading"][2]) + int(self.mask_info["timing"][2])
                info["avg_dist_err"] = nanmean(self.dist_errors)
            else:
                info["curriculum_metric"] = np.nan
                info["avg_heading_err"] = np.nan
                info["avg_timing_met"] = np.nan
                info["avg_dist_err"] = np.nan
                info["mask_combo_id"] = 0

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

    def calc_potential(self):
        walk_target_delta = self.walk_target - self.robot.body_xyz
        body_distance_to_target = sqrt(ss(walk_target_delta[0:2]))

        angle_delta = self.smallest_angle_between(self.robot.feet_rpy[self.swing_leg,2], self.terrain_info[self.next_step_index, 6])

        multiplier = 2 if (self.curriculum > 0 or self.behavior_curriculum > 0 or self.from_net) else 0.1

        if self.mask_info["heading"][2]:
            multiplier = 0

        if self.mask_info["timing"][2] and self.next_step_index <= 2: # and not (self.curriculum > 0 or self.behavior_curriculum > 0):
            # add a foot distance potential if there is no timing signal
            foot_delta = sqrt(ss(self.terrain_info[self.next_step_index, 0:2] - self.robot.feet_xyz[self.swing_leg][0:2])) * 0.3
        else:
            foot_delta = 0

        self.linear_potential = -(body_distance_to_target + angle_delta * multiplier + foot_delta) / self.scene.dt
        self.distance_to_target = body_distance_to_target

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential

        self.calc_potential()

        linear_progress = self.linear_potential - old_linear_potential
        self.progress = linear_progress * 2

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

        self.elbow_penalty = 0

        elbow_angles = self.robot.joint_angles[[16, 20]]
        elbow_good_mask = elbow_angles > 65 * DEG2RAD
        self.elbow_penalty += np.dot(1 * ~elbow_good_mask, np.abs(elbow_angles - 65 * DEG2RAD))

        heights = self.robot.upper_arm_and_head_xyz[:,2]
        min_height_diff = 0.25
        if heights[2] - heights[0] < min_height_diff:
            self.elbow_penalty += abs(heights[2] - heights[0] - min_height_diff)
        if heights[2] - heights[1] < min_height_diff:
            self.elbow_penalty += abs(heights[2] - heights[1] - min_height_diff)

        terminal_height = self.terminal_height_curriculum[self.curriculum]
        self.tall_bonus = 2 if self.robot_state[0] > terminal_height else -1.0
        abs_height = self.robot.body_xyz[2] - self.terrain_info[self.next_step_index, 2]

        self.legs_bonus = 0
        self.heading_bonus = 0

        swing_foot_tilt = self.robot.feet_rpy[self.swing_leg, 1]

        if self.target_reached and swing_foot_tilt < 5 * DEG2RAD and not "backward" in self.selected_behavior:
            self.legs_bonus += self.tilt_bonus_weight

        if abs(self.progress) < 0.02 and (not self.stop_on_next_step or not self.target_reached):
            self.body_stationary_count += 1
        else:
            self.body_stationary_count = 0
        count = 200
        if self.body_stationary_count > count:
            self.legs_bonus -= 100

        if self.mask_info["heading"][2]:
            self.heading_bonus = 0
        else:
            if self.target_reached and not self.past_last_step:
                self.heading_bonus = -( -np.exp(-self.gauss_width * abs(self.heading_rad_to_target) ** 2) + 1)
            else:
                self.heading_bonus = 0

            if not self.mask_info["timing"][2] and self.current_step_time <= self.terrain_info[self.next_step_index, 10] and self.next_step_index > 1 and (self.curriculum > 0 or self.behavior_curriculum > 0 or self.from_net):
                self.heading_bonus += -( -np.exp(-self.gauss_width * abs(self.prev_heading_rad_to_target) ** 2) + 1) * 0.5
        
        if self.mask_info["timing"][2]:
            self.timing_bonus = 0
            self.left_actual_contact = self._foot_target_contacts[1,0]
            self.right_actual_contact = self._foot_target_contacts[0,0]
        else:
            self.calc_timing_reward()

        self.done = self.done or self.tall_bonus < 0 or abs_height < -3 or self.swing_leg_has_fallen or self.other_leg_has_fallen or self.body_stationary_count > count or self.finished_all

    def calc_timing_reward(self):
        self.left_actual_contact = self._foot_target_contacts[1,0]
        self.right_actual_contact = self._foot_target_contacts[0,0]

        self.current_time_index = self.next_step_index

        next_step_time = [
            self.terrain_info[self.current_time_index, 8],
            self.terrain_info[self.current_time_index, 9],
            self.terrain_info[self.current_time_index, 10],
            self.terrain_info[self.current_time_index, 11]
        ]

        if not self.past_last_step:
            # assumes swing leg == 1 (will swap later)
            if self.current_time_index < self.num_steps - 1:
                if self.current_step_time < next_step_time[0]: # first contact
                    self.left_expected_contact = 1
                elif next_step_time[0] <= self.current_step_time < (next_step_time[0] + next_step_time[1]): # first lift
                    self.left_expected_contact = 0
                elif (next_step_time[0] + next_step_time[1]) <= self.current_step_time < (next_step_time[0] + next_step_time[1] + self.step_delay):
                    self.left_expected_contact = 1
                else:
                    self.left_expected_contact = -1 if (self.current_time_index > 2 or not self.target_reached) else 1
            else:
                self.left_expected_contact = 1 if (self.current_step_time <= next_step_time[0] or self.current_step_time >= next_step_time[0] + next_step_time[1]) else 0
            if self.current_time_index < self.num_steps - 1:
                if self.current_step_time < next_step_time[2]: # first contact
                    self.right_expected_contact = 1
                elif next_step_time[2] <= self.current_step_time < (next_step_time[2] + next_step_time[3]): # first lift
                    self.right_expected_contact = 0
                elif (next_step_time[2] + next_step_time[3]) <= self.current_step_time < (next_step_time[2] + next_step_time[3] + self.step_delay):
                    self.right_expected_contact = 0 if next_step_time[3] != 0 else 1
                else:
                    self.right_expected_contact = -1 if (self.current_time_index > 2 or not self.target_reached) else 1
            else:
                self.right_expected_contact = 1 if (self.current_step_time <= next_step_time[2] or self.current_step_time >= next_step_time[2] + next_step_time[3]) else 0
        else:
            self.left_expected_contact = 1
            self.right_expected_contact = 1

        if self.swing_leg == 0:
            # swap happens here if needed
            self.left_expected_contact, self.right_expected_contact = self.right_expected_contact, self.left_expected_contact
            
        expected_contacts = [self.right_expected_contact, self.left_expected_contact]

        met_time = np.sum(expected_contacts == self._foot_target_contacts[:, 0])

        self.timing_bonus = np.sum(2 * (expected_contacts == self._foot_target_contacts[:, 0]) - 1)

        if not self.past_last_step:
            self.met_times.append(met_time)

    def smallest_angle_between(self, angle1, angle2):
        angle1 = angle1 % (2 * np.pi)
        angle2 = angle2 % (2 * np.pi)
        diff = abs(angle1 - angle2)
        smallest_angle = min(diff, (2 * np.pi) - diff)
        if smallest_angle >= np.pi:
            smallest_angle -= 2 * np.pi
        return smallest_angle

    def calc_feet_state(self):
        self.foot_dist_to_target = np.sqrt(
            ss(
                self.robot.feet_xyz[:, 0:2]
                - self.terrain_info[self.next_step_index, 0:2],
                axis=1,
            )
        )

        robot_id = self.robot.id
        client_id = self._p._client
        ground_ids = next(iter(self.ground_ids))
        target_id_list = [ground_ids[0]]
        target_cover_id_list = [ground_ids[1]]
        self._foot_target_contacts.fill(0)

        for i, (foot, contact) in enumerate(
            zip(self.robot.feet, self._foot_target_contacts)
        ):
            self.robot.feet_contact[i] = pybullet.getContactStates(
                bodyA=robot_id,
                linkIndexA=foot.bodyPartIndex,
                bodiesB=target_id_list,
                linkIndicesB=target_cover_id_list,
                results=contact,
                physicsClientId=client_id,
            )

        self.imaginary_step = self.terrain_info[self.next_step_index,2] > 0.01
        self.current_target_count += 1

        self.heading_rad_to_target = self.smallest_angle_between(self.robot.feet_rpy[self.swing_leg,2], self.terrain_info[self.next_step_index, 6])
        if self.next_step_index > 1:
            prev_index = np.where(self.terrain_info[0:self.next_step_index, 7] == 1-self.swing_leg)[0][-1]
            self.prev_heading_rad_to_target = self.smallest_angle_between(self.robot.feet_rpy[1-self.swing_leg,2], self.terrain_info[prev_index, 6])
        else:
            self.prev_heading_rad_to_target = 0

        if self.next_step_index == 1 or self.swing_leg_lifted:
            # if first step or already lifted, say true
            self.swing_leg_lifted = True
        if self._foot_target_contacts[self.swing_leg, 0] == 0:
            # if in the air, increase count
            self.swing_leg_lifted_count += 1
            self.in_air_count += 1
        else:
            self.swing_leg_lifted_count = 0

        if not self.swing_leg_lifted:
            # if not lifted yet and over count, True
            if self.swing_leg_lifted_count >= 1:
                self.swing_leg_lifted = True

        if self.next_step_index > 1:
            dist_to_prev_target = np.sqrt(
                ss(
                    self.robot.feet_xyz[:, 0:2]
                    - self.prev_leg_pos[:, 0:2],
                    axis=1,
                )
            )
            foot_in_target = self.foot_dist_to_target[self.swing_leg] < self.step_radius
            foot_in_prev_target = dist_to_prev_target[self.swing_leg] < self.step_radius
            other_foot_in_prev_target = dist_to_prev_target[1-self.swing_leg] < self.step_radius + 0.1
            swing_leg_not_on_steps = not foot_in_target and not foot_in_prev_target
        # else:
        #     swing_leg_not_on_steps = self.foot_dist_to_target[self.swing_leg] >= self.step_radius

        swing_leg_in_air = self._foot_target_contacts[self.swing_leg, 0] == 0
        other_leg_in_air = self._foot_target_contacts[1-self.swing_leg, 0] == 0

        # if swing leg is not on previous step and not on current step and not in air, should terminate
        self.swing_leg_has_fallen = self.next_step_index > 1 and not swing_leg_in_air and swing_leg_not_on_steps
        # self.swing_leg_has_fallen = not swing_leg_in_air and swing_leg_not_on_steps # self.next_step_index > 1
        self.other_leg_has_fallen = self.next_step_index > 1 and not other_leg_in_air and not other_foot_in_prev_target
        
        self.target_reached = self._foot_target_contacts[self.swing_leg, 0] > 0 and self.foot_dist_to_target[self.swing_leg] < self.step_radius and (self.swing_leg_lifted or self.reached_last_step)

        next_step_time = [
            self.terrain_info[self.next_step_index, 8],
            self.terrain_info[self.next_step_index, 9],
            self.terrain_info[self.next_step_index, 10],
            self.terrain_info[self.next_step_index, 11]
        ]
        if not self.mask_info["timing"][2] and (self.target_reached and self.next_step_index > 2 and self.current_step_time < next_step_time[0] + next_step_time[1]):
            self.target_reached = False

        self.past_last_step = self.past_last_step or (self.reached_last_step and self.target_reached_count >= 2)

        if self.target_reached and not self.past_last_step and not self.mask_info["heading"][2]:
            self.heading_errors.append(abs(self.heading_rad_to_target))

        if self.target_reached:
            self.target_reached_count += 1

            # Advance after has stopped for awhile
            if self.target_reached_count > 120:
                self.stop_on_next_step = False
                self.set_stop_on_next_step = False

            # Slight delay for target advancement
            # Needed for not over counting step bonus
            delay = self.step_delay # 10 if self.next_step_index > 4 else 2
            if self.target_reached_count >= delay:
                if not self.stop_on_next_step:
                    self.prev_foot_yaw = self.robot.feet_rpy[self.swing_leg,2]
                    self.current_step_time = 0
                    self.current_target_count = 0
                    self.in_air_count = 0
                    self.prev_leg_pos[self.swing_leg] = self.terrain_info[self.next_step_index, 0:2]
                    # self.prev_leg_pos = self.robot.feet_xyz[:, 0:2]
                    self.prev_leg = self.swing_leg
                    self.next_step_index += 1
                    self.next_step_start_timestep = self.timestep
                    if self.next_step_index < self.num_steps:
                        self.swing_leg = int(self.terrain_info[self.next_step_index, 7])
                    self.target_reached_count = 0
                    self.swing_leg_lifted = False
                    self.swing_leg_lifted_count = 0
                    self.both_feet_hit_ground = False
                    self.body_stationary_count = 0
                    self.update_steps()
                self.stop_on_next_step = self.set_stop_on_next_step

                self.reached_last_step = self.reached_last_step or self.next_step_index >= len(self.terrain_info) - 1
                self.finished_all = self.finished_all or self.next_step_index >= len(self.terrain_info)

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
            dist = self.foot_dist_to_target[self.swing_leg]
            self.dist_errors.append(dist)
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

        # if cur_step_index != self.next_step_index:
        #     self.calc_potential()

        self.calc_base_reward(action)
        self.calc_step_reward()
        # use next step to calculate next k steps
        self.targets, self.extra_param = self.delta_to_k_targets()

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

        # if self.behaviors.index(self.selected_behavior) == 0 and self.selected_curriculum == 0:
        #     # TODO: bad for mixing everything together
        #     walk_target_full = targets[self.walk_target_index]
        # else:
        walk_target_full = self.terrain_info[self.next_step_index]
        # walk_target_full = targets[self.walk_target_index]
        self.walk_target = np.copy(walk_target_full[0:3])
        heading = walk_target_full[6]
        if int(walk_target_full[7]) == 1:
            self.walk_target[0] += np.cos(heading - np.pi / 2) * walk_target_full[12]
            self.walk_target[1] += np.sin(heading - np.pi / 2) * walk_target_full[12]
        else:
            self.walk_target[0] += np.cos(heading + np.pi / 2) * walk_target_full[12]
            self.walk_target[1] += np.sin(heading + np.pi / 2) * walk_target_full[12]

        delta_pos = targets[:, 0:3] - self.robot.body_xyz
        target_thetas = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])

        angle_to_targets = target_thetas - self.robot.body_rpy[2]
        distance_to_targets = np.sqrt(ss(delta_pos[:, 0:2], axis=1))
        # should angles be per feet? yes so it doesn't change too much
        feet_heading = np.array([self.robot.feet_rpy[int(i), 2] for i in targets[:, 7]])
        heading_angle_to_targets = targets[:, 6] - feet_heading

        # reduce the angle  
        heading_angle_to_targets =  heading_angle_to_targets % (2 * np.pi)
        # force it to be the positive remainder, so that 0 <= heading_angle_to_targets < 2 * np.pi  
        heading_angle_to_targets = (heading_angle_to_targets + 2 * np.pi) % (2 * np.pi)
        # force into the minimum absolute value residue class, so that -180 < heading_angle_to_targets <= 180  
        heading_angle_to_targets[heading_angle_to_targets > np.pi] -= (2 * np.pi)

        swing_legs_at_targets = np.where(targets[:, 7] == 0, -1, 1)

        time_left = np.array([
            targets[1, 8],
            targets[1, 9],
            targets[1, 10],
            targets[1, 11]
        ])
        if self.current_step_time <= time_left[0]:
            time_left[0] -= self.current_step_time
            time_left[2] -= self.current_step_time
        else:
            time_left[0] = 0
            time_left[1] = max(time_left[1] - (self.current_step_time - targets[1, 8]), 0)
            time_left[2] = max(time_left[2] - self.current_step_time, 0)

        xy_mask = np.ones(k + j) if self.mask_info["xy"][2] else np.zeros(k + j)
        heading_mask = np.ones(k + j) if self.mask_info["heading"][2] else np.zeros(k + j)
        swing_leg_mask = np.ones(k + j) if self.mask_info["leg"][2] else np.zeros(k + j)

        if self.mask_info["xy"][2]:
            x = np.zeros(k + j)
            y = np.zeros(k + j)
            z = np.zeros(k + j)
        else:
            x = np.sin(angle_to_targets) * distance_to_targets
            y = np.cos(angle_to_targets) * distance_to_targets
            z = delta_pos[:, 2]

        if self.mask_info["heading"][2]:
            heading_angle_to_targets *= 0

        if self.mask_info["leg"][2]:
            swing_legs_at_targets *= 0

        if self.mask_info["timing"][2]:
            time_left *= 0

        deltas = concatenate(
            (
                (x)[:, None],  # x
                (y)[:, None],  # y
                (z)[:, None],  # z
                (targets[:, 4])[:, None],  # x_tilt
                (targets[:, 5])[:, None],  # y_tilt
                (heading_angle_to_targets)[:, None], # heading
                (swing_legs_at_targets)[:, None],  # swing_legs
                (xy_mask)[:, None],
                (heading_mask)[:, None],
                (swing_leg_mask)[:, None],
            ),
            axis=1,
        )

        dr = np.array([0,0])
        time_and_dr_mask = np.array([self.mask_info["timing"][2],self.mask_info["dir"][2],self.mask_info["vel"][2]]).astype(int)

        return deltas, np.concatenate([time_left, dr, [0], time_and_dr_mask])

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
                    i * self.step_param_dim + 6, # swing legs
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
