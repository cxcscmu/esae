#!/usr/bin/bash

# Validate the arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <num_gpu>"
    exit 1
fi

NUM_GPU=$1

# Acquire a compute node with 32 CPUs, 128GB RAM, and specified number of GPUs.
# The node will be acquired for 7 days, and the session will be interactive.
# Please run this script with tmux to avoid losing the session.
srun \
        --partition=debug --time=12:00:00 \
        --mem=96GB --gres=gpu:A6000:$NUM_GPU \
        --pty bash
