#!/usr/bin/bash

# Acquire a compute node with 32 CPUs, 96GB memory, and 1 A6000 GPU.
# Please run this script with tmux to avoid losing the session.
srun \
    --partition=long --time=07-00:00:00 \
    --cpus-per-task=32 --mem=96GB --gres=gpu:A6000:1 \
    --pty bash
