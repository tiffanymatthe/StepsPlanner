from __future__ import absolute_import
from __future__ import division

import functools
import inspect
import os
import time
import types

current_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pybullet


class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=None, use_ffmpeg=False, fps=60):
        """Creates a Bullet client and connects to a simulation.

        Args:
          connection_mode:
            `None` connects to an existing simulation or, if fails, creates a
              new headless simulation,
            `pybullet.GUI` creates a new simulation with a GUI,
            `pybullet.DIRECT` creates a headless simulation,
            `pybullet.SHARED_MEMORY` connects to an existing simulation.
        """
        self._shapes = {}

        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY)
            if self._client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT

        options = (
            "--background_color_red=1.0 "
            "--background_color_green=1.0 "
            "--background_color_blue=1.0 "
            "--width=1280 --height=720 "
        )
        if use_ffmpeg:
            from datetime import datetime

            now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            options += f'--mp4="{now_str}.mp4" --mp4fps={fps} '

        self._client = pybullet.connect(connection_mode, options=options)

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            if name not in [
                "invertTransform",
                "multiplyTransforms",
                "getMatrixFromQuaternion",
                "getEulerFromQuaternion",
                "computeViewMatrixFromYawPitchRoll",
                "computeProjectionMatrixFOV",
                "getQuaternionFromEuler",
            ]:  # A temporary hack for now.
                attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute


class Pose_Helper:  # dummy class to comply to original interface
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()


class BodyPart:
    def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        self.initialPosition = self.current_position()
        self.initialOrientation = self.current_orientation()
        self.bp_pose = Pose_Helper(self)

    def state_fields_of_pose_of(
        self, body_id, link_id=-1
    ):  # a method you will most probably need a lot to get pose and orientation
        if link_id == -1:
            (x, y, z), (a, b, c, d) = pybullet.getBasePositionAndOrientation(
                body_id, physicsClientId=self._p._client
            )
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = pybullet.getLinkState(
                body_id, link_id, physicsClientId=self._p._client
            )
        return np.array([x, y, z, a, b, c, d])

    def get_position(self):
        return self.current_position()

    def get_pose(self):
        return self.state_fields_of_pose_of(
            self.bodies[self.bodyIndex], self.bodyPartIndex
        )

    def angular_speed(self):
        if self.bodyPartIndex == -1:
            _, (vr, vp, vy) = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            _, _, _, _, _, _, _, (vr, vp, vy) = self._p.getLinkState(
                self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1
            )
        return np.array([vr, vp, vy])

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), _ = pybullet.getBaseVelocity(
                self.bodies[self.bodyIndex], physicsClientId=self._p._client
            )
        else:
            (_, _, _, _, _, _, (vx, vy, vz), _,) = pybullet.getLinkState(
                self.bodies[self.bodyIndex],
                self.bodyPartIndex,
                computeLinkVelocity=1,
                physicsClientId=self._p._client,
            )
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def get_orientation(self):
        return self.current_orientation()

    def reset_position(self, position):
        pybullet.resetBasePositionAndOrientation(
            self.bodies[self.bodyIndex],
            position,
            self.get_orientation(),
            physicsClientId=self._p._client,
        )

    def reset_orientation(self, orientation):
        pybullet.resetBasePositionAndOrientation(
            self.bodies[self.bodyIndex],
            self.get_position(),
            orientation,
            physicsClientId=self._p._client,
        )

    def reset_velocity(self, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]):
        pybullet.resetBaseVelocity(
            self.bodies[self.bodyIndex],
            linearVelocity,
            angularVelocity,
            physicsClientId=self._p._client,
        )

    def reset_pose(self, position, orientation):
        pybullet.resetBasePositionAndOrientation(
            self.bodies[self.bodyIndex],
            position,
            orientation,
            physicsClientId=self._p._client,
        )

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        return pybullet.getContactPoints(
            bodyA=self.bodies[self.bodyIndex],
            linkIndexA=self.bodyPartIndex,
            physicsClientId=self._p._client,
        )


class Joint:
    def __init__(
        self, bullet_client, joint_name, bodies, bodyIndex, jointIndex, torque_limit=0
    ):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_name
        self.torque_limit = torque_limit

        jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
        self.lowerLimit = jointInfo[8]
        self.upperLimit = jointInfo[9]

        self.power_coeff = 0

    def set_torque_limit(self, torque_limit):
        self.torque_limit = torque_limit

    def set_state(self, x, vx):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

    def current_position(self):
        return self.get_state()

    def current_relative_position(self):
        pos, vel = self.get_state()
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
        return (2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), vel)

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(
            self.bodies[self.bodyIndex], self.jointIndex
        )
        return x, vx

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _, r = self.get_state()
        return r

    def get_velocity(self):
        _, vx = self.get_state()
        return vx

    def set_position(self, position):
        self._p.setJointMotorControl2(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.torque_limit,
        )

    def set_velocity(self, velocity):
        self._p.setJointMotorControl2(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            pybullet.VELOCITY_CONTROL,
            targetVelocity=velocity,
        )

    def set_motor_torque(self, torque):
        self.set_torque(torque)

    def set_torque(self, torque):
        self._p.setJointMotorControl2(
            bodyIndex=self.bodies[self.bodyIndex],
            jointIndex=self.jointIndex,
            controlMode=pybullet.TORQUE_CONTROL,
            force=torque,
        )

    def reset_current_position(self, position, velocity):
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        self._p.resetJointState(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            targetValue=position,
            targetVelocity=velocity,
        )
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=0,
            targetVelocity=0,
            positionGain=0.1,
            velocityGain=0.1,
            force=0,
        )


class Scene:
    "A base class for single- and multiplayer scenes"

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.timestep = timestep
        self.frame_skip = frame_skip

        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(self._p, gravity, timestep, frame_skip)

        self.test_window_still_open = True
        self.human_render_detected = False

        self.multiplayer_robots = {}

    def test_window(self):
        "Call this function every frame, to see what's going on. Not necessary in learning."
        self.human_render_detected = True
        return self.test_window_still_open

    def actor_introduce(self, robot):
        "Usually after scene reset"
        if not self.multiplayer:
            return
        self.multiplayer_robots[robot.player_n] = robot

    def actor_is_active(self, robot):
        """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
        return not self.multiplayer

    def set_physics_parameters(self):
        "This function gets overridden by specific scene, to reset specific objects into their start positions"
        self.cpp_world.set_physics_parameters()

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self.cpp_world.step()


class World:
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.numSolverIterations = 5
        self.set_physics_parameters()

    def set_physics_parameters(self):
        self._p.setGravity(0, 0, -self.gravity)
        self._p.setDefaultContactERP(0.9)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSolverIterations=self.numSolverIterations,
            numSubSteps=self.frame_skip,
        )

    def step(self):
        pybullet.stepSimulation(physicsClientId=self._p._client)


class StadiumScene(Scene):

    stadium_halflen = 105 * 0.25
    stadium_halfwidth = 50 * 0.25

    def initialize(self, remove_ground=False):
        current_dir = os.path.dirname(__file__)

        if not remove_ground:
            filename = os.path.join(
                current_dir, "data", "objects", "misc", "plane_stadium.sdf"
            )
            self.ground_plane_mjcf = self._p.loadSDF(filename)

            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)

    def set_friction(self, lateral_friction):
        for i in self.ground_plane_mjcf:
            self._p.changeDynamics(i, -1, lateralFriction=lateral_friction)


class SinglePlayerStadiumScene(StadiumScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False


class Camera:
    def __init__(self, env, bc, fps=60, dist=2.5, yaw=0, pitch=-5, use_egl=False):

        self._p = bc
        self._cam_dist = dist
        self._cam_yaw = yaw
        self._cam_pitch = pitch
        self._coef = np.array([1.0, 1.0, 0.1])

        self.use_egl = use_egl
        self.tracking = False

        self._fps = fps
        self._target_period = 1 / fps
        self._last_frame_time = time.perf_counter()
        self.env_should_wait = False

        camera = self

        def new_global_step(self):
            time_spent = time.perf_counter() - camera._last_frame_time

            camera.env_should_wait = True
            if camera._target_period < time_spent:
                camera._last_frame_time = time.perf_counter()
                camera.env_should_wait = False

            if not camera.env_should_wait:
                self.cpp_world.step()

        old_apply_action = env.robot.apply_action

        def new_apply_action(self, action):
            if not camera.env_should_wait:
                old_apply_action(action)

        env.scene.global_step = types.MethodType(new_global_step, env.scene)
        env.robot.apply_action = types.MethodType(new_apply_action, env.robot)

    def track(self, pos, smooth_coef=None):

        # self.wait()
        if self.env_should_wait or not self.tracking:
            return

        smooth_coef = self._coef if smooth_coef is None else smooth_coef
        assert (smooth_coef <= 1).all(), "Invalid camera smoothing parameters"

        yaw, pitch, dist, lookat_ = self._p.getDebugVisualizerCamera()[-4:]
        lookat = (1 - smooth_coef) * lookat_ + smooth_coef * pos
        self._cam_target = lookat

        self._p.resetDebugVisualizerCamera(dist, yaw, pitch, lookat)

        # Remember camera for reset
        self._cam_yaw, self._cam_pitch, self._cam_dist = yaw, pitch, dist

    def lookat(self, pos):
        self._cam_target = pos
        self._p.resetDebugVisualizerCamera(
            self._cam_dist, self._cam_yaw, self._cam_pitch, pos
        )

    def dump_rgb_array(self):

        if self.use_egl:
            # use_egl
            width, height = 1920, 1080
            view = self._p.computeViewMatrixFromYawPitchRoll(
                self._cam_target, distance=4, yaw=0, pitch=-20, roll=0, upAxisIndex=2
            )
            proj = self._p.computeProjectionMatrixFOV(
                fov=60, aspect=width / height, nearVal=0.1, farVal=100.0
            )
        else:
            # is_rendered
            width, height, view, proj = self._p.getDebugVisualizerCamera()[0:4]

        (_, _, rgb_array, _, _) = self._p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
            flags=self._p.ER_NO_SEGMENTATION_MASK,
        )

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def wait(self):
        if self.use_egl:
            return

        time_spent = time.perf_counter() - self._last_frame_time

        self.env_should_wait = True
        if self._target_period < time_spent:
            self._last_frame_time = time.perf_counter()
            self.env_should_wait = False
