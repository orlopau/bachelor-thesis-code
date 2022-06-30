#!/bin/bash

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

echo "Starting a mpi run via horovod..."

mpirun -np 1 \
    -bind-to none -map-by slot \
    $VENV/bin/python $WS_PATH/sync/code/stress_analysis/stress.py --data "$WS_PATH/sync/code/data/"