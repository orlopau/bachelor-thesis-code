#!/bin/bash
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=0:20:00
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch.log

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

source $VENV/bin/activate

python $WS_PATH/sync/bin/ray/ray_slurm.py "$VENV/bin/python $WS_PATH/sync/code/cnn.py --data $WS_PATH/data"