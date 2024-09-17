
behavior_curriculum=(0 1 2 3 4 5 6 7 8)
curriculum=(9 9 9 6 9 6 9 9 9)

for i in ${!behavior_curriculum[@]}; do
    python3 -m playground.enjoy --env Walker3DStepperEnv-v0 --net runs/dream/sep_16/timing_50_curr_6/models/Walker3DStepperEnv-v0_300000000.pt  --plank_class VeryLargePlank --curriculum ${curriculum[$i]} --behavior_curriculum ${behavior_curriculum[$i]} --determine --save 1 --len 1000
    source stack_videos.sh "${behavior_curriculum[$i]}_${curriculum[$i]}"
done