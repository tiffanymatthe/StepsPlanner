# set -e

# # Define the ranges and special conditions
# behavior_start=0   # Starting value for behavior_curriculum
# behavior_end=1     # Ending value for behavior_curriculum
# a=0                # Starting value for curriculum in the first behavior_curriculum
# b=6                # Ending value for curriculum in the last behavior_curriculum

# for behavior_curriculum in $(seq $behavior_start $behavior_end); do
#     if [ $behavior_curriculum -eq $behavior_start ]; then
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
#             --net "runs/dream/nov_15/base_mike/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0
#     done
# done

# # Define the ranges and special conditions
# behavior_start=1   # Starting value for behavior_curriculum
# behavior_end=5     # Ending value for behavior_curriculum
# a=7                # Starting value for curriculum in the first behavior_curriculum
# b=8                # Ending value for curriculum in the last behavior_curriculum

# for behavior_curriculum in $(seq $behavior_start $behavior_end); do
#     if [ $behavior_curriculum -eq $behavior_start ]; then
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
#             --net "runs/dream/nov_16/base_mike_cont/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0
#     done
# done


# # Define the ranges and special conditions
# behavior_start=5   # Starting value for behavior_curriculum
# behavior_end=8     # Ending value for behavior_curriculum
# a=9                # Starting value for curriculum in the first behavior_curriculum
# b=9                # Ending value for curriculum in the last behavior_curriculum

# for behavior_curriculum in $(seq $behavior_start $behavior_end); do
#     if [ $behavior_curriculum -eq $behavior_start ]; then
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
#             --net "runs/dream/nov_19/base_mike_cont_5_9/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
#             --plank_class VeryLargePlank \
#             --curriculum "$curriculum" \
#             --behavior_curriculum "$behavior_curriculum" \
#             --determine \
#             --render 0 \
#             --plot 0
#     done
# done

# Define the ranges and special conditions
behavior_start=9   # Starting value for behavior_curriculum
behavior_end=9     # Ending value for behavior_curriculum
a=0                # Starting value for curriculum in the first behavior_curriculum
b=8                # Ending value for curriculum in the last behavior_curriculum

for behavior_curriculum in $(seq $behavior_start $behavior_end); do
    if [ $behavior_curriculum -eq $behavior_start ]; then
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
            --net "runs/dream/nov_19/base_mike_cont_5_9/models/Walker3DStepperEnv-v0_curr_10_${curriculum}.pt" \
            --plank_class VeryLargePlank \
            --curriculum "$curriculum" \
            --behavior_curriculum "$behavior_curriculum" \
            --determine \
            --render 0 \
            --plot 0
    done
done