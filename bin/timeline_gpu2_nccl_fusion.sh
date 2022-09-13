#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH -m plane=4
#SBATCH --mem=0
#SBATCH --gres="gpu:4"
#SBATCH --time=0:30:00
#SBATCH -p gpu2
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/sbatch_%j.log

ml restore default

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_nccl

source $VENV/bin/activate

# export WANDB_MODE=disabled

OMPI_MCA_btl='^ofi'
OMPI_MCA_mtl='^ofi'
export OMPI_MCA_btl='^ofi'
export OMPI_MCA_mtl='^ofi'
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1

export HOROVOD_TIMELINE=/lustre/ssd/ws/s8979104-horovod/timeline_gpu2_nccl_fusion.json
export HOROVOD_FUSION_THRESHOLD=0

srun --ntasks=16 --cpu-bind=none,v --accel-bind=gn $VENV/bin/python $WS_PATH/sync/src/stress.py --data $WS_PATH/data --project gpu2 --group fusion --dist