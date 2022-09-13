#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH -m plane=8
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --gres="gpu:8"
#SBATCH --time=1:00:00
#SBATCH -p alpha
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/scorep-alpha_%j.log

ml restore alpha

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_mpi

source $VENV/bin/activate

OMPI_MCA_btl='^ofi'
OMPI_MCA_mtl='^ofi'
export OMPI_MCA_btl='^ofi'
export OMPI_MCA_mtl='^ofi'
export NCCL_DEBUG=INFO

export WANDB_MODE=disabled
export SCOREP_TOTAL_MEMORY=8589934592

export SCOREP_EXPERIMENT_DIRECTORY=/lustre/ssd/ws/s8979104-horovod/scorep/

srun --cpu-bind=none,v --accel-bind=gn $VENV/bin/python -u -m scorep --mpp=mpi \
    --noinstrumenter $WS_PATH/sync/src/stress.py --data $WS_PATH/data --dist --scorep