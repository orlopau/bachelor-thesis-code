#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --gres="gpu:1"
#SBATCH --time=1:00:00
#SBATCH -p gpu2
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch_gpu2.log

ml restore default

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_mpi

source $VENV/bin/activate

export WANDB_MODE=disabled
export SCOREP_TOTAL_MEMORY=8589934592

srun $VENV/bin/python -u -m scorep --noinstrumenter $WS_PATH/sync/src/stress.py --data $WS_PATH/data --scorep