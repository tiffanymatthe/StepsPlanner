#!/bin/bash

# Input videos
DATE_TIME_STR=$1 # must set as argument to bash script

VIDEO1="videos/${DATE_TIME_STR}_rgb.mp4"
VIDEO2="videos/${DATE_TIME_STR}_plot.mp4"
TRIMMED_VIDEO1="videos/trimmed_video1.mp4"
TRIMMED_VIDEO2="videos/trimmed_video2.mp4"
OUTPUT="videos/${DATE_TIME_STR}_COMBINED.mp4"

# Get dimensions of both videos
DIMENSIONS1=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 $VIDEO1)
DIMENSIONS2=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 $VIDEO2)

# Get the duration of the shorter video
shorter_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO2")
longer_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO1")
# if [[ "$shorter_duration" != "$longer_duration" ]]; then
#     echo "DURATIONS ARE NOT EQUAL $shorter_duration vs $longer_duration"
#     exit 1
# fi

WIDTH1=$(echo $DIMENSIONS1 | cut -d'x' -f1)
HEIGHT1=$(echo $DIMENSIONS1 | cut -d'x' -f2)
WIDTH2=$(echo $DIMENSIONS2 | cut -d'x' -f1)
HEIGHT2=$(echo $DIMENSIONS2 | cut -d'x' -f2)

# Determine the target width (use the width of the wider video)
TARGET_WIDTH=$((WIDTH1 > WIDTH2 ? WIDTH1 : WIDTH2))

if [ $((TARGET_WIDTH%2)) -ne 0 ]; then
    TARGET_WIDTH=$((TARGET_WIDTH - 1))
fi

# Determine which video is less wide and scale
if [ "$WIDTH1" -lt "$TARGET_WIDTH" ]; then
    ffmpeg -y -i "$VIDEO1" -vf "scale=${TARGET_WIDTH}:trunc(ow/a/2)*2" "$TRIMMED_VIDEO1"
    VIDEO1="$TRIMMED_VIDEO1"
fi

if [ "$WIDTH2" -lt "$TARGET_WIDTH" ]; then
    ffmpeg -y -i "$VIDEO2" -vf "scale=${TARGET_WIDTH}:trunc(ow/a/2)*2" "$TRIMMED_VIDEO2"
    VIDEO2="$TRIMMED_VIDEO2"
fi

echo "finished trimming"

# Stack the videos vertically
# https://www.ffmpeg.org/ffmpeg-all.html#toc-vstack-1
ffmpeg -y -i "$VIDEO1" -i "$VIDEO2" -filter_complex "[0:v][1:v]vstack=shortest=1" "$OUTPUT"

echo "Videos stacked and saved as $OUTPUT"
