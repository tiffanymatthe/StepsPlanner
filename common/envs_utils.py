import cloudpickle
import contextlib
import csv
import ctypes
import glob
import json
import multiprocessing
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import gym
from gym import spaces
from gym.core import Wrapper
import numpy as np
import torch


def make_env_fns(env_id, seed, rank, log_dir, **kwargs):
    def _thunk():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)

        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)

        # env = Monitor(env, None, allow_early_resets=True)
        env = LiteMonitor(env)

        return env

    return _thunk


def make_env(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    return env


def make_vec_envs(env_id, seed, num_processes, log_dir, **kwargs):
    assert num_processes > 1

    env_fns = [
        make_env_fns(env_id, seed, i, log_dir, **kwargs) for i in range(num_processes)
    ]

    # Using 'spawn' is more robust than 'fork'
    envs = ShmemVecEnv(env_fns, context="spawn")
    return envs


class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class LiteMonitor(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)

    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        if done:
            eprew = sum(self.rewards)
            info["episode"] = {"r": eprew}
        return (obs, rew, done, info)


class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(
        self,
        env,
        filename,
        allow_early_resets=False,
        reset_keywords=(),
        info_keywords=(),
    ):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": time.time(), "env_id": env.spec and env.spec.id},
                extra_keys=reset_keywords + info_keywords,
            )
        else:
            self.results_writer = None
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = (
            {}
        )  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        self.reset_state()
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError("Expected you to pass kwarg %s into reset" % k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                (
                    "Tried to reset an environment before done. "
                    "If you want to allow early resets, wrap your env "
                    "with Monitor(env, path, allow_early_resets=True)"
                )
            )
        self.rewards = []
        self.needs_reset = False

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {
                "r": round(eprew, 6),
                "l": eplen,
                "t": round(time.time() - self.tstart, 6),
            }
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(epinfo)
            assert isinstance(info, dict)
            if isinstance(info, dict):
                info["episode"] = epinfo

        self.total_steps += 1

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


class ResultsWriter(object):
    def __init__(self, filename, header="", extra_keys=()):
        self.extra_keys = extra_keys
        assert filename is not None
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = "# {} \n".format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(
            self.f, fieldnames=("r", "l", "t") + tuple(extra_keys)
        )
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self


class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(
            self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self):
        self.venv.render()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        shapes, dtypes = {}, {}
        self.keys = []
        obs_space = env.observation_space

        if isinstance(obs_space, spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            subspaces = obs_space.spaces
        else:
            subspaces = {None: obs_space}

        for key, box in subspaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = {
            k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])
            for k in self.keys
        }
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for e in range(self.num_envs):
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[
                e
            ].step(self.actions[e])
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            self.buf_infos.copy(),
        )

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def set_env_params(self, data):
        for e in range(self.num_envs):
            self.envs[e].unwrapped.set_env_params(data)

    def close(self):
        return

    def render(self):
        return [e.render() for e in self.envs]

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        if self.keys == [None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.
    If the child process has MPI environment variables, MPI will think
    that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such
    as when we are starting multiprocessing Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ["OMPI_", "PMI_"]:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


_NP_TO_CT = {
    np.float64: ctypes.c_double,
    np.float32: ctypes.c_float,
    np.int32: ctypes.c_int32,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_char,
    bool: ctypes.c_bool,
}


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, context="spawn"):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = multiprocessing.get_context(context)

        dummy = env_fns[0]()
        observation_space, action_space = dummy.observation_space, dummy.action_space
        dummy.close()
        del dummy

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_shape = observation_space.shape
        self.obs_dtype = observation_space.dtype

        self.obs_bufs = [
            ctx.Array(_NP_TO_CT[self.obs_dtype.type], int(np.prod(self.obs_shape)))
            for _ in env_fns
        ]
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(
                    target=_subproc_worker,
                    args=(
                        child_pipe,
                        parent_pipe,
                        wrapped_fn,
                        obs_buf,
                        self.obs_shape,
                        self.obs_dtype,
                    ),
                )
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            print("Called reset() while waiting for the step to complete")
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(("step", act))

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(obs), np.array(rews), np.array(dones), infos

    def set_env_params(self, params_dict):
        for pipe in self.parent_pipes:
            pipe.send(("set_env_params", params_dict))

    def set_robot_params(self, params_dict):
        for pipe in self.parent_pipes:
            pipe.send(("set_robot_params", params_dict))

    def get_env_param(self, param_name, default):
        for pipe in self.parent_pipes:
            pipe.send(("get_env_param", (param_name, default)))
        return [pipe.recv() for pipe in self.parent_pipes]

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode="human"):
        for pipe in self.parent_pipes:
            pipe.send(("render", None))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self, obs):
        bufs = self.obs_bufs
        o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtype) for b in bufs]
        return np.array(o)


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_buf, obs_shape, obs_dtype):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    dst = obs_buf.get_obj()
    dst_np = np.frombuffer(dst, dtype=obs_dtype).reshape(obs_shape)

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                np.copyto(dst_np, env.reset())
                pipe.send(None)
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                np.copyto(dst_np, obs)
                pipe.send((None, reward, done, info))
            elif cmd == "set_env_params":
                env.set_env_params(data)
            elif cmd == "set_robot_params":
                env.set_robot_params(data)
            elif cmd == "get_env_param":
                param = env.get_env_param(*data)
                pipe.send(param)
            elif cmd == "render":
                pipe.send(env.render(mode="rgb_array"))
            elif cmd == "close":
                pipe.send(None)
                break
            else:
                raise RuntimeError("Got unrecognized cmd %s" % cmd)
    except KeyboardInterrupt:
        print("ShmemVecEnv worker: got KeyboardInterrupt")
    finally:
        env.close()


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)


def get_mirror_function(indices, device="cpu"):

    negation_obs_indices = torch.from_numpy(indices[0]).to(device)
    right_obs_indices = torch.from_numpy(indices[1]).to(device)
    left_obs_indices = torch.from_numpy(indices[2]).to(device)
    negation_action_indices = torch.from_numpy(indices[3]).to(device)
    right_action_indices = torch.from_numpy(indices[4]).to(device)
    left_action_indices = torch.from_numpy(indices[5]).to(device)

    orl = torch.cat((right_obs_indices, left_obs_indices))
    olr = torch.cat((left_obs_indices, right_obs_indices))
    arl = torch.cat((right_action_indices, left_action_indices))
    alr = torch.cat((left_action_indices, right_action_indices))

    def mirror_function(trajectory_samples):
        (
            observations_batch,
            actions_batch,
            value_preds_batch,
            returns_batch,
            masks_batch,
            old_action_log_probs_batch,
            advantages_batch,
        ) = trajectory_samples

        # Only observation and action needs to be mirrored
        observations_mirror = observations_batch.clone()
        observations_mirror[:, negation_obs_indices] *= -1
        observations_mirror[:, orl] = observations_mirror[:, olr]

        actions_mirror = actions_batch.clone()
        actions_mirror[:, negation_action_indices] *= -1
        actions_mirror[:, arl] = actions_mirror[:, alr]

        # Others need to be repeated
        observations_batch = torch.cat([observations_batch, observations_mirror])
        actions_batch = torch.cat([actions_batch, actions_mirror])
        value_preds_batch = value_preds_batch.repeat((2, 1))
        returns_batch = returns_batch.repeat((2, 1))
        masks_batch = masks_batch.repeat((2, 1))
        old_action_log_probs_batch = old_action_log_probs_batch.repeat((2, 1))
        advantages_batch = advantages_batch.repeat((2, 1))

        return (
            observations_batch,
            actions_batch,
            value_preds_batch,
            returns_batch,
            masks_batch,
            old_action_log_probs_batch,
            advantages_batch,
        )

    return mirror_function
