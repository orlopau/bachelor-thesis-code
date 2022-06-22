#!/bin/bash

salloc --ntasks=4 --ntasks-per-node=4 -N 2 --cpus-per-task=2 --time=0:10:00 --mem=40G --partition=gpu2-interactive --gres=gpu:4 /lustre/ssd/ws/s8979104-horovod/sync/bin/stress_mpi.sh