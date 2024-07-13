git pull
git checkout cc21b96
source env/bin/activate
pip install wandb
wandb login b47e4beb8673053b5f44ae4ccc54e94a44963d05
python -m playground.train with num_processes=100 num_frames=20e7 use_mirror=True use_curriculum=True gauss_width=4 heading_bonus_weight=2 experiment_dir=runs/jul_12/double_step_from_scratch