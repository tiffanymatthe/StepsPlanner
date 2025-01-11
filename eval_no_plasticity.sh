set -e

# # Define the ranges and special conditions
# behavior_start=0   # Starting value for behavior_curriculum
# behavior_end=5     # Ending value for behavior_curriculum
# a=0                # Starting value for curriculum in the first behavior_curriculum
# b=8                # Ending value for curriculum in the last behavior_curriculum

# Define the ranges and special conditions
behavior_start=3   # Starting value for behavior_curriculum
behavior_end=3     # Ending value for behavior_curriculum
a=4                # Starting value for curriculum in the first behavior_curriculum
b=4                # Ending value for curriculum in the last behavior_curriculum

for behavior_curriculum in $(seq $behavior_start $behavior_end); do
    if [ $behavior_curriculum -eq $behavior_start ] && [ $behavior_curriculum -eq $behavior_end ]; then
        # If behavior_curriculum is both the start and the end
        curriculum_start=$a
        curriculum_end=$b
    elif [ $behavior_curriculum -eq $behavior_start ]; then
        # For the first behavior_curriculum
        curriculum_start=$a
        curriculum_end=9
    elif [ $behavior_curriculum -eq $behavior_end ]; then
        # For the last behavior_curriculum
        curriculum_start=0
        curriculum_end=$b
    else
        # For all other behavior_curriculum values
        curriculum_start=0
        curriculum_end=9
    fi

    for curriculum in $(seq $curriculum_start $curriculum_end); do
        echo "Running $behavior_curriculum - $curriculum"
        python3 -m playground.enjoy \
            --env Walker3DStepperEnv-v0 \
            --net "runs/dream/jan_3/no_plasticity/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
            --plank_class VeryLargePlank \
            --curriculum "$curriculum" \
            --behavior_curriculum "$behavior_curriculum" \
            --determine \
            --render 0 \
            --plot 0 \
            --save_folder no_plasticity
    done
done

# # Define the ranges and special conditions
# behavior_start=5   # Starting value for behavior_curriculum
# behavior_end=11     # Ending value for behavior_curriculum
# a=9                # Starting value for curriculum in the first behavior_curriculum
# b=1                # Ending value for curriculum in the last behavior_curriculum

# for behavior_curriculum in $(seq $behavior_start $behavior_end); do
#     if [ $behavior_curriculum -eq $behavior_start ] && [ $behavior_curriculum -eq $behavior_end ]; then
#         # If behavior_curriculum is both the start and the end
#         curriculum_start=$a
#         curriculum_end=$b
#     elif [ $behavior_curriculum -eq $behavior_start ]; then
#         # For the first behavior_curriculum
#         curriculum_start=$a
#         curriculum_end=9
#     elif [ $behavior_curriculum -eq $behavior_end ]; then
#         # For the last behavior_curriculum
#         curriculum_start=0
#         curriculum_end=$b
#     else
#         # For all other behavior_curriculum values
#         curriculum_start=0
#         curriculum_end=9
#     fi

#     for curriculum in $(seq $curriculum_start $curriculum_end); do
#         echo "Running $behavior_curriculum - $curriculum"
#         python3 -m playground.enjoy \
#             --env Walker3DStepperEnv-v0 \
#             --net "runs/dream/jan_6/no_plasticity_cont/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0 \
#             --save_folder no_plasticity
#     done
# done

# # Define the ranges and special conditions
# behavior_start=11   # Starting value for behavior_curriculum
# behavior_end=11     # Ending value for behavior_curriculum
# a=2                # Starting value for curriculum in the first behavior_curriculum
# b=8                # Ending value for curriculum in the last behavior_curriculum

# for behavior_curriculum in $(seq $behavior_start $behavior_end); do
#     if [ $behavior_curriculum -eq $behavior_start ] && [ $behavior_curriculum -eq $behavior_end ]; then
#         # If behavior_curriculum is both the start and the end
#         curriculum_start=$a
#         curriculum_end=$b
#     elif [ $behavior_curriculum -eq $behavior_start ]; then
#         # For the first behavior_curriculum
#         curriculum_start=$a
#         curriculum_end=9
#     elif [ $behavior_curriculum -eq $behavior_end ]; then
#         # For the last behavior_curriculum
#         curriculum_start=0
#         curriculum_end=$b
#     else
#         # For all other behavior_curriculum values
#         curriculum_start=0
#         curriculum_end=9
#     fi

#     for curriculum in $(seq $curriculum_start $curriculum_end); do
#         echo "Running $behavior_curriculum - $curriculum"
#         python3 -m playground.enjoy \
#             --env Walker3DStepperEnv-v0 \
#             --net "runs/dream/jan_9/no_plasticity_cont/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0 \
#             --save_folder no_plasticity
#     done
# done

# # Define the ranges and special conditions
# behavior_start=11   # Starting value for behavior_curriculum
# behavior_end=11     # Ending value for behavior_curriculum
# a=9                # Starting value for curriculum in the first behavior_curriculum
# b=9                # Ending value for curriculum in the last behavior_curriculum

# for behavior_curriculum in $(seq $behavior_start $behavior_end); do
#     if [ $behavior_curriculum -eq $behavior_start ] && [ $behavior_curriculum -eq $behavior_end ]; then
#         # If behavior_curriculum is both the start and the end
#         curriculum_start=$a
#         curriculum_end=$b
#     elif [ $behavior_curriculum -eq $behavior_start ]; then
#         # For the first behavior_curriculum
#         curriculum_start=$a
#         curriculum_end=9
#     elif [ $behavior_curriculum -eq $behavior_end ]; then
#         # For the last behavior_curriculum
#         curriculum_start=0
#         curriculum_end=$b
#     else
#         # For all other behavior_curriculum values
#         curriculum_start=0
#         curriculum_end=9
#     fi

#     for curriculum in $(seq $curriculum_start $curriculum_end); do
#         echo "Running $behavior_curriculum - $curriculum"
#         python3 -m playground.enjoy \
#             --env Walker3DStepperEnv-v0 \
#             --net "runs/dream/jan_9/no_plasticity_cont/models/Walker3DStepperEnv-v0_525000000.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0 \
#             --save_folder no_plasticity
#     done
# done