#!/bin/bash

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

source $VENV/bin/activate

horovodrun -np 2 -H localhost:2 python $WS_PATH/sync/code/stress_analysis/stress.py --data "$WS_PATH/sync/code/data/"