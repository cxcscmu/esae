#!/usr/bin/bash

# Acquire a compute node with 32 CPUs, 96GB RAM, and 1 A6000 GPU.
# The node will be acquired for 7 days, and the session will be interactive.
# Please run this script with tmux to avoid losing the session.
srun \
    --partition=long --time=07-00:00:00 \
    --cpus-per-task=32 --mem=96GB --gres=gpu:A6000:1 \
    --pty bash
