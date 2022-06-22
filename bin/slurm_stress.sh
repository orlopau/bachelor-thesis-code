#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16384
#SBATCH --gres=gpu:2

srun /lustre/ssd/ws/s8979104-horovod/sync/bin/stress.sh