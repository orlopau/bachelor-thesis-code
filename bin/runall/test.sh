#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core, 10312M -> 6 cores per gpu
# gpu2: 24 cores, 4 gpus, ~2.5G per core, 2583M -> 6 cores per gpu
# hpdlf: 12 cores, 3 gpus, 7916M per core, 3 gpus, 4 cores per GPU

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --gres="gpu:3"
#SBATCH --time=0:01:00
#SBATCH --exclusive=user
#SBATCH -p gpu2
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/test.log

ml restore default

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/loool

echo $SLURM_TASKS_PER_NODE