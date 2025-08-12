#!/bin/bash

# Start streaming in background
muselsl stream --name Muse-D074 --acc --gyro &

# Time for establishing connection with muse
sleep 15

# Start visual viewer
muselsl view -w 5 -s 100 -r 0.2