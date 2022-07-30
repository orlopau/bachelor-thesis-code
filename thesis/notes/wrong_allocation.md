slurm does not correctly set SLURM_TASKS_PER_NODE.

When using `salloc -N 2 --ntasks-per-node=4 --cpus-per-task=6 --time=2:00:00 --mem=0 --gres=gpu:4 -p alpha --exclusive=user /usr/bin/zsh -l`, slurm does not set the environment variable SLURM_TASKS_PER_NODE correctly.

[]: # * probieren mit modenv/hiera (weil amd cpus)