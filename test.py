
half_cycle_time = 30

terrain_info[]

for next_step_index in range(1,10):

    for current_step_time in range(0,30):
        next_step_time = [
            terrain_info[next_step_index, 8],
            terrain_info[next_step_index, 9],
            terrain_info[next_step_index, 10],
            terrain_info[next_step_index, 11]
        ]

        if next_step_index < num_steps - 1:
            next_next_step_time = [
                terrain_info[next_step_index+1, 8],
                terrain_info[next_step_index+1, 9],
                terrain_info[next_step_index+1, 10],
                terrain_info[next_step_index+1, 11]
            ]

        if not past_last_step:
            # assumes swing leg == 1 (will swap later)
            if next_step_index < num_steps - 1:
                if current_step_time <= next_step_time[0]:
                    left_expected_contact = 1
                else:
                    left_expected_contact = int(next_next_step_time[2] != 0) if terrain_info[next_step_index, 7] != swing_leg else int(next_next_step_time[0] != 0)
            else:
                left_expected_contact = 1 if (current_step_time <= next_step_time[0] or current_step_time >= next_step_time[0] + next_step_time[1]) else 0
            if next_step_index < num_steps - 1:
                if current_step_time <= next_step_time[2]:
                    right_expected_contact = 1
                else:
                    right_expected_contact = int(next_next_step_time[0] != 0) if terrain_info[next_step_index + 1, 7] != swing_leg else int(next_next_step_time[2] != 0)
            else:
                left_expected_contact = 1 if (current_step_time <= next_step_time[2] or current_step_time >= next_step_time[2] + next_step_time[3]) else 0