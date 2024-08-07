import argparse
import datetime
import inspect
import math
import os
import time
import types

import numpy as np


class StringEnum(tuple):
    def __getattr__(self, attr):
        if attr in self:
            return attr


class EpisodeRunner(object):
    def __init__(
        self, env, save=False, use_ffmpeg=False, dir=None, max_steps=None, csv=None
    ):
        self.env = env
        self.save = save
        self.use_ffmpeg = use_ffmpeg
        self.csv = csv

        self.max_steps = max_steps or float("inf")
        self.done = False
        self.step = 0

        base_dir = os.path.dirname(os.path.realpath(inspect.stack()[-1][1]))
        self.dump_dir = dir or os.path.join(base_dir, "dump")
        if self.csv is not None:
            self.csv = os.path.join(base_dir, self.csv)

        self.override_reset()  # add progress bar
        self.override_step()  # save frame and csv data

        if self.csv is not None:
            self.csv_data_buffer = {}

        if self.save:
            self.rgb_buffer = []
            self.max_steps = max_steps or env.max_timestep

            now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.filename = os.path.join(self.dump_dir, f"{now_str}.mp4")
            print("\nRecording... Close to terminate recording.")

        self.progress_bar = None
        if self.max_steps != float("inf"):
            try:
                from tqdm import tqdm

                self.progress_bar = tqdm(total=self.max_steps)
            except ImportError:
                pass

    def override_reset(self):
        old_reset_func = self.env.reset
        runner = self

        def new_reset(self, **kwargs):
            if kwargs.get("reset_runner", True):
                runner.step = 0
                if runner.progress_bar:
                    runner.progress_bar.reset()

            kwargs.pop("reset_runner", None)
            return old_reset_func(**kwargs)

        self.env.reset = types.MethodType(new_reset, self.env)

    def override_step(self):

        old_step_func = self.env.step
        runner = self

        def new_step(self, *args):
            ret = old_step_func(*args)

            if not runner.use_ffmpeg:
                runner.store_current_frame()

            runner.save_csv_render_data()
            runner.step += 1

            if runner.progress_bar is not None:
                runner.progress_bar.update(1)

            if runner.step >= runner.max_steps:
                runner.done = True
                if runner.progress_bar is not None:
                    runner.progress_bar.close()

            return ret

        self.env.step = types.MethodType(new_step, self.env)

    def store_current_frame(self):
        if self.save:
            image = self.env.camera.dump_rgb_array()
            self.rgb_buffer.append(image)

    def save_csv_render_data(self):
        if self.csv is not None:
            render_data_dict = self.env.dump_additional_render_data()
            for file, render_data in render_data_dict.items():
                header = render_data["header"]
                data = render_data["data"].clone().cpu().numpy()
                once = render_data.get("once", False)

                if file not in self.csv_data_buffer:
                    list_data = data if once else []
                    self.csv_data_buffer[file] = {"header": header, "data": list_data}

                if not once:
                    self.csv_data_buffer[file]["data"].append(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.save and len(self.rgb_buffer) >= self.max_steps:
            import moviepy.editor as mp

            if not os.path.exists(self.dump_dir):
                print(f"Creating directory {self.dump_dir}")
                os.makedirs(self.dump_dir)

            clip = mp.ImageSequenceClip(self.rgb_buffer, fps=1 / self.env.control_step)
            clip.write_videofile(self.filename)

        if self.csv is not None:
            for file, data_dict in self.csv_data_buffer.items():
                np.savetxt(
                    os.path.join(self.csv, file),
                    np.asarray(data_dict["data"]),
                    delimiter=",",
                    header=data_dict["header"],
                    comments="",
                )

        self.env.close()


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (as_minutes(s), as_minutes(rs))


def linear_decay(epoch, total_num_epochs, initial_value, final_value):
    return initial_value - (initial_value - final_value) * epoch / float(
        total_num_epochs
    )


def exponential_decay(epoch, rate, initial_value, final_value):
    return max(initial_value * (rate ** epoch), final_value)


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def str2bool(v):
    """
    Argument Parse helper function for boolean values
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def interpolate_between_indices(array, start_index, end_index):
    # end index is inclusive
    # Get the values at the start and end indices
    start_value = array[start_index]
    end_value = array[end_index]
    
    # Number of elements to interpolate
    num_elements = end_index - start_index + 1
    
    # Create interpolated values
    interpolated_values = np.linspace(start_value, end_value, num_elements)
    
    # Replace the elements in the array with interpolated values
    array[start_index:end_index + 1] = interpolated_values
        
def get_left_right_indices(middle_index, width):
    offset = (width - 1) // 2
    return (middle_index - offset, middle_index + offset)

def get_contact_gaits(cycle_time, uncertainty_range):
    SS_phase = int(cycle_time * 0.76 / 2)
    DS_phase = int((cycle_time - 2 * SS_phase) / 2)
    assert SS_phase * 2 + DS_phase * 2 == cycle_time, f"SS_phase {SS_phase} and DS_phase {DS_phase} do not make {cycle_time}"
    start_leg_expected_contact_probabilities = np.zeros(cycle_time)
    start_leg_expected_contact_probabilities[SS_phase:] = 1
    other_leg_expected_contact_probabilities = np.ones(cycle_time)
    other_leg_expected_contact_probabilities[SS_phase + DS_phase:SS_phase*2 + DS_phase] = 0

    # smooth out transitions
    start_leg_expected_contact_probabilities[0] = 0.5
    interpolate_between_indices(start_leg_expected_contact_probabilities, 0, uncertainty_range // 2)
    interpolate_between_indices(start_leg_expected_contact_probabilities, *get_left_right_indices(SS_phase, uncertainty_range))
    start_leg_expected_contact_probabilities[-1] = 0.75
    interpolate_between_indices(start_leg_expected_contact_probabilities, cycle_time - uncertainty_range // 2, cycle_time - 1)

    interpolate_between_indices(other_leg_expected_contact_probabilities, *get_left_right_indices(SS_phase+DS_phase, uncertainty_range))
    interpolate_between_indices(other_leg_expected_contact_probabilities, *get_left_right_indices(SS_phase*2+DS_phase, uncertainty_range))

    return start_leg_expected_contact_probabilities, other_leg_expected_contact_probabilities