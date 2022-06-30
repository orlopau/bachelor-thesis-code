salloc --ntasks=2 -N 2 --ntasks-per-node=1 --cpus-per-task=2 --time=0:10:00 --mem=1G --partition=gpu2-interactive --gres=gpu:1 /lustre/ssd/ws/s8979104-horovod/sync/bin/slurm_ray.sh

sacct --user=s8979104

squeue --me

sbatch slurm_multi_node.sh

sstat -j 26661893 -o MaxRSS,ConsumedEnergy