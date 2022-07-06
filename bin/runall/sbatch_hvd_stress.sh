#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core

#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus}
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres="gpu:{gpus}"
#SBATCH --time=1:00:00
#____SBATCH --exclusive
#SBATCH -p alpha
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/sbatch_%j.log
#SBATCH -J runall
#___SBATCH --nodelist {nodelist}
#SBATCH --hint=nomultithread

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

source $VENV/bin/activate

mpirun -N {gpus} \
    -bind-to none --oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    $VENV/bin/python -u $WS_PATH/sync/code/stress_cnn_horovod.py --data $WS_PATH/data
