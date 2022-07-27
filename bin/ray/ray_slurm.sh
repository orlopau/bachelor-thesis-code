#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --gres="gpu:3"
#SBATCH --time=4:00:00
#SBATCH -p hpdlf
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/sbatch_%j.log

ml restore default

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_mpi

source $VENV/bin/activate

python $WS_PATH/sync/bin/ray/ray_slurm.py "$VENV/bin/python -u $WS_PATH/sync/code/stress_tune.py --data $WS_PATH/data"