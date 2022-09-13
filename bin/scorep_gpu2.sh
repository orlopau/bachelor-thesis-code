#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH -m plane=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --gres="gpu:4"
#SBATCH --time=1:00:00
#SBATCH -p gpu2
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/gpu2/%j.log

ml restore default

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
export SCOREP_EXPERIMENT_DIRECTORY=/lustre/ssd/ws/s8979104-horovod/scorep/gpu2_mpi_8_1024/$RANDOM

srun --cpu-bind=none,v --accel-bind=gn $VENV/bin/python -u -m scorep --mpp=mpi --noinstrumenter $WS_PATH/sync/src/stress.py --data $WS_PATH/data --dist --scorep