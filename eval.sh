#!/bin/bash

# for behavior_curriculum in {0..1}; do
#     # Loop through curriculum from 0 to 9
#     for curriculum in {0..9}; do
#         # Run the Python command with the current values of behavior_curriculum and curriculum
#         echo "Running $behavior_curriculum - $curriculum"
#         python3 -m playground.enjoy \
#             --env Walker3DStepperEnv-v0 \
#             --net "runs/dream/oct_14/plasticity_elaho/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0
#     done
# done

# for behavior_curriculum in {2..2}; do
#     # Loop through curriculum from 0 to 9
#     for curriculum in {0..2}; do
#         # Run the Python command with the current values of behavior_curriculum and curriculum
#         echo "Running $behavior_curriculum - $curriculum"
#         python3 -m playground.enjoy \
#             --env Walker3DStepperEnv-v0 \
#             --net "runs/dream/oct_15/plasticity_elaho_cont_lowered_threshold_fixed/models/Walker3DStepperEnv-v0_curr_2_2.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0
#     done
# done

# for behavior_curriculum in {2..5}; do
#     # Loop through curriculum from 0 to 9
#     for curriculum in {0..9}; do
#         # Run the Python command with the current values of behavior_curriculum and curriculum
#         # Skip specific combinations
#         if { [ "$behavior_curriculum" -eq 2 ] && [ "$curriculum" -eq 0 ]; } || \
#            { [ "$behavior_curriculum" -eq 2 ] && [ "$curriculum" -eq 1 ]; } || \
#            { [ "$behavior_curriculum" -eq 2 ] && [ "$curriculum" -eq 2 ]; } || \
#            { [ "$behavior_curriculum" -eq 5 ] && [ "$curriculum" -eq 9 ]; }; then
#             continue
#         fi
#         echo "Running $behavior_curriculum - $curriculum"
#         python3 -m playground.enjoy \
#             --env Walker3DStepperEnv-v0 \
#             --net "runs/dream/oct_15/plasticity_elaho_cont_lowered_threshold_fixed/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0
#     done
# done

# echo "Running 5 - 9"
# python3 -m playground.enjoy \
#     --env Walker3DStepperEnv-v0 \
#     --net "runs/dream/oct_18/plasticity_elaho_cont_lowered_threshold_fixed_cont/models/Walker3DStepperEnv-v0_curr_5_9.pt" \
#     --plank_class VeryLargePlank \
#     --curriculum 9 \
#     --behavior_curriculum 5 \
#     --determine \
#     --render 0 \
#     --plot 0

# for behavior_curriculum in {9..9}; do
#     # Loop through curriculum from 0 to 9
#     for curriculum in {0..9}; do
#         # Run the Python command with the current values of behavior_curriculum and curriculum
#         # Skip specific combinations
#         if { [ "$behavior_curriculum" -eq 9 ] && [ "$curriculum" -eq 9 ]; }; then
#             continue
#         fi
#         echo "Running $behavior_curriculum - $curriculum"
#         python3 -m playground.enjoy \
#             --env Walker3DStepperEnv-v0 \
#             --net "runs/dream/oct_18/plasticity_elaho_cont_lowered_threshold_fixed_cont/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0
#     done
# done

# echo "Running 9 - 9"
# python3 -m playground.enjoy \
#     --env Walker3DStepperEnv-v0 \
#     --net "runs/dream/oct_18/plasticity_elaho_cont_lowered_threshold_fixed_cont/models/Walker3DStepperEnv-v0_600000000.pt" \
#     --plank_class VeryLargePlank \
#     --curriculum 9 \
#     --behavior_curriculum 9 \
#     --determine \
#     --render 0 \
#     --plot 0

# for behavior_curriculum in {10..10}; do
#     # Loop through curriculum from 0 to 9
#     for curriculum in {0..8}; do
#         # Run the Python command with the current values of behavior_curriculum and curriculum
#         # Skip specific combinations
#         echo "Running $behavior_curriculum - $curriculum"
#         python3 -m playground.enjoy \
#             --env Walker3DStepperEnv-v0 \
#             --net "runs/dream/oct_19/plasticity_elaho_cont_one_step_plant/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0
#     done
# done


for behavior_curriculum in {11..11}; do
    # Loop through curriculum from 0 to 9
    for curriculum in {0..8}; do
        # Run the Python command with the current values of behavior_curriculum and curriculum
        echo "Running $behavior_curriculum - $curriculum"
        python3 -m playground.enjoy \
            --env Walker3DStepperEnv-v0 \
            --net "runs/dream/oct_20/plasticity_elaho_cont_heading/models/Walker3DStepperEnv-v0_curr_10_${curriculum}.pt" \
            --plank_class VeryLargePlank \
            --curriculum "$curriculum" \
            --behavior_curriculum "$behavior_curriculum" \
            --determine \
            --render 0 \
            --plot 0
    done
done

echo "Running 11 - 9"
python3 -m playground.enjoy \
    --env Walker3DStepperEnv-v0 \
    --net "runs/dream/oct_20/plasticity_elaho_cont_heading/models/Walker3DStepperEnv-v0_400000000.pt" \
    --plank_class VeryLargePlank \
    --curriculum 9 \
    --behavior_curriculum 11 \
    --determine \
    --render 0 \
    --plot 0