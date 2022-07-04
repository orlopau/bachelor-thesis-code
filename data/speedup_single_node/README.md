# Info

All tests are run with "--exclusive" option to allocate the whole node.
Tests are run without NCCL support!

```sh
#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core

#SBATCH --nodes={nodes}
#SBATCH --tasks-per-node={gpus}
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --gres="gpu:{gpus}"
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH -p alpha
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/sbatch_%j.log
#SBATCH -J runall

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

source $VENV/bin/activate

horovodrun -np {gpus} -H localhost:{gpus} $VENV/bin/python -u $WS_PATH/sync/code/stress_cnn_horovod.py --data $WS_PATH/data
```