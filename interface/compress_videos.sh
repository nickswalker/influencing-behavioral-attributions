#!/bin/bash
mkdir -p $1/out
for file in $(ls $1/*.webm); do
  new_file=out/$(basename ${file})
  ffmpeg -i $file -c:v libvpx-vp9 -b:v 1M -pass 1 -an -f null /dev/null && \
ffmpeg -i $file -c:v libvpx-vp9 -b:v 1M -pass 2 -c:a libopus $new_file
done