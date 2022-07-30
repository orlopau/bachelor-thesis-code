#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core, 10312M -> 6 cores per gpu
# gpu2: 24 cores, 4 gpus, ~2.5G per core, 2583M -> 6 cores per gpu
# hpdlf: 12 cores, 3 gpus, 7916M per core, 3 gpus, 4 cores per GPU

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10310M
#SBATCH --gres="gpu:4"
#SBATCH --time=4:00:00
#SBATCH -p alpha
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/ray/sbatch_%j.log

ml restore default

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_mpi

source $VENV/bin/activate

python $WS_PATH/sync/bin/ray/ray_slurm.py "$VENV/bin/python -u $WS_PATH/sync/code/stress_tune.py --data $WS_PATH/data --project alpha"