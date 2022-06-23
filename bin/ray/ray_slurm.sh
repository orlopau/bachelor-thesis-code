#!/bin/bash
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

source $VENV/bin/activate

python $WS_PATH/bin/ray/ray_slurm.py