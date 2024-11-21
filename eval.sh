#!/bin/bash

for behavior_curriculum in {0..9}; do
    # Loop through curriculum from 0 to 9
    for curriculum in {0..9}; do
        # Run the Python command with the current values of behavior_curriculum and curriculum
        echo "Running $behavior_curriculum - $curriculum"
        python3 -m playground.enjoy \
            --env Walker3DStepperEnv-v0 \
            --net "runs/dream/nov_20/base_mike_cont_heading_mask/models/Walker3DStepperEnv-v0_curr_10_8.pt" \
            --plank_class VeryLargePlank \
            --curriculum "$curriculum" \
            --behavior_curriculum "$behavior_curriculum" \
            --determine \
            --render 0 \
            --plot 0
    done
done
