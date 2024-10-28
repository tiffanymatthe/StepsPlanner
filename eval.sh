#!/bin/bash

# Loop through behavior_curriculum from 0 to 10
for behavior_curriculum in {8..10}; do
    # Loop through curriculum from 0 to 9
    for curriculum in {0..9}; do
        # Run the Python command with the current values of behavior_curriculum and curriculum
        python3 -m playground.enjoy \
            --env Walker3DStepperEnv-v0 \
            --net runs/dream/oct_20/plasticity_elaho_cont_heading/models/Walker3DStepperEnv-v0_400000000.pt \
            --plank_class VeryLargePlank \
            --curriculum "$curriculum" \
            --behavior_curriculum "$behavior_curriculum" \
            --determine \
            --render 0 \
            --plot 0
    done
done
