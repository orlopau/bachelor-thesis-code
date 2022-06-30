#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core

#SBATCH --nodes=1
#SBATCH --tasks-per-node=7
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --gres="gpu:7"
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH -p alpha
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch_%j.log

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

source $VENV/bin/activate

$NUM_GPUS=7

horovodrun -np 7 -H localhost:7 $VENV/bin/python -u $WS_PATH/sync/code/stress_cnn_horovod.py --data $WS_PATH/data