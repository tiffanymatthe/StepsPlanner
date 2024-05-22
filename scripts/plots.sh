python3 -m playground.plot --load_paths runs/no_policy_reg_no_iter runs/expo_5e5_1e6 runs/const_5e5_1e6 --columns max_rew mean_rew --dotted_lines cycle_count --x_max 5e6

python3 -m playground.plot --load_paths runs/may_19/walker_policy_reg_coef_** runs/no_policy_reg_no_iter runs/expo_5e5 --columns max_rew mean_rew --dotted_lines cycle_count

python3 -m playground.plot --load_paths runs/may_19/walker_value_reg_coef_** runs/no_policy_reg_no_iter runs/expo_5e5 --columns max_rew mean_rew --dotted_lines cycle_count  --x_max 6e6

python3 -m playground.plot --load_paths runs/no_policy_reg_no_iter runs/may_19/walker_no_** runs/may_20/walker_only_rl --columns max_rew mean_rew --dotted_lines cycle_count  --x_max 6e6

python3 -m playground.plot --load_paths runs/no_policy_reg_no_iter runs/may_20/walker_init_10e5_5e5  --columns max_rew mean_rew --dotted_lines cycle_count  --x_max 12e6
python3 -m playground.plot --load_paths runs/no_policy_reg_no_iter runs/may_20/walker_init_5e5_20e5  --columns max_rew mean_rew --dotted_lines cycle_count  --x_max 12e6
python3 -m playground.plot --load_paths runs/no_policy_reg_no_iter runs/may_20/walker_init_longer  --columns max_rew mean_rew --dotted_lines cycle_count  --x_max 12e6


python3 -m playground.plot --load_paths runs/kl_2e6_** runs/kl_4e6 runs/no_policy_reg_no_iter --columns max_rew mean_rew --dotted_lines cycle_count  --x_max 12e6

python3 -m playground.plot --load_paths runs/may_21/walk_longer_start runs/no_policy_reg_no_iter --columns max_rew mean_rew --dotted_lines cycle_count  --x_max 12e6

python3 -m playground.plot --load_paths runs/no_policy_reg_no_iter runs/may_21/walk_no_iter_w_clip runs/may_21/walk_longer_no_value_clip runs/may_21/walk_longer --columns max_rew mean_rew --dotted_lines cycle_count  --x_max 12e6