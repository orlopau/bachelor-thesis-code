#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core, 10312M -> 6 cores per gpu
# gpu2: 24 cores, 4 gpus, ~2.5G per core, 2583M -> 6 cores per gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2583M
#SBATCH --gres="gpu:1"
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --nice=200
#SBATCH -x taurusi2095
#SBATCH -p gpu2
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/sbatch_%j.log
#SBATCH -J stress

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_mpi

source $VENV/bin/activate

$VENV/bin/python -u $WS_PATH/sync/code/stress.py --data $WS_PATH/data --group base --name base
