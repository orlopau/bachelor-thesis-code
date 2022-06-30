#!/bin/bash

# command running on interactive bash task
srun --pty --ntasks=1 --cpus-per-task=4 --time=1:00:00 --mem=16384 --gres=gpu:2 bash -l