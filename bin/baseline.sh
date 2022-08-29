#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=10312M
#__SBATCH --mem=0
#SBATCH --gres="gpu:1"
#SBATCH --time=1:00:00
#SBATCH -p alpha
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/sbatch_%j.log

ml restore default

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_mpi

source $VENV/bin/activate

srun --ntasks=1 --cpu-bind=none,v --accel-bind=gn $VENV/bin/python $WS_PATH/sync/src/stress.py --data $WS_PATH/data --group base --project alpha