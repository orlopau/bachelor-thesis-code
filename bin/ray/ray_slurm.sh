#!/bin/bash
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --gres="gpu:4"
#SBATCH --time=1:30:00
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch.log

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

source $VENV/bin/activate

python $WS_PATH/sync/bin/ray/ray_slurm.py "$VENV/bin/python -u $WS_PATH/sync/code/raytune.py --data $WS_PATH/data"