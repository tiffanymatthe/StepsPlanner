import datetime
import os
import pathlib
import subprocess
import sys


if len(sys.argv) < 2:
    print("No arguments supplied: experiment name required")
    exit(1)

name = sys.argv[1]

del sys.argv[0]
del sys.argv[0]

today = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
log_path = os.path.join("runs", f"{today}__{name}")


pathlib.Path(log_path).mkdir(parents=True, exist_ok=False)

with open(os.path.join(log_path, "slurm.out"), "w") as outfile:
    process = subprocess.Popen(
        [
            "python",
            os.path.join("playground", "train.py"),
            "with",
            f"experiment_dir={log_path}",
            *sys.argv,
        ],
        stdout=outfile,
        stderr=outfile,
    )

with open(os.path.join(log_path, "pid"), "w") as outfile:
    print(f"Process spawned with ID: {process.pid}")
    print(f"Experiment directory: {log_path}")
    print(
        process.pid,
        file=outfile,
        flush=True,
    )
