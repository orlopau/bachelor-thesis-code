#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#___SBATCH -m plane=4
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=10312M
#SBATCH --gres="gpu:1"
#SBATCH --time=1:00:00
#___SBATCH --exclusive=user
#SBATCH -p alpha
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch.log

ml restore alpha

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_mpi

source $VENV/bin/activate

OMPI_MCA_btl='^ofi'
OMPI_MCA_mtl='^ofi'
export OMPI_MCA_btl='^ofi'
export OMPI_MCA_mtl='^ofi'
export NCCL_DEBUG=INFO

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled
export SCOREP_TOTAL_MEMORY=8589934592

srun --cpu-bind=none,v --accel-bind=gn $VENV/bin/python -u -m scorep --mpp=mpi --noinstrumenter $WS_PATH/sync/src/stress.py --data $WS_PATH/data --dist --scorep