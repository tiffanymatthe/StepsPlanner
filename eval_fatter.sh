# set -e

# Define the ranges and special conditions
behavior_start=0   # Starting value for behavior_curriculum
behavior_end=4     # Ending value for behavior_curriculum
a=0                # Starting value for curriculum in the first behavior_curriculum
b=7                # Ending value for curriculum in the last behavior_curriculum

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
            --net "runs/dream/nov_23/fatter_morphology/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
            --plank_class VeryLargePlank \
            --curriculum "$curriculum" \
            --behavior_curriculum "$behavior_curriculum" \
            --determine \
            --render 0 \
            --plot 0
    done
done

# Define the ranges and special conditions
behavior_start=4   # Starting value for behavior_curriculum
behavior_end=8     # Ending value for behavior_curriculum
a=7                # Starting value for curriculum in the first behavior_curriculum
b=9                # Ending value for curriculum in the last behavior_curriculum

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
            --net "runs/dream/nov_26/fatter_morphology_cont_4_7/models/Walker3DStepperEnv-v0_curr_${behavior_curriculum}_${curriculum}.pt" \
            --plank_class VeryLargePlank \
            --curriculum "$curriculum" \
            --behavior_curriculum "$behavior_curriculum" \
            --determine \
            --render 0 \
            --plot 0
    done
done
