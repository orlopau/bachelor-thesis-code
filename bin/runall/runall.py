import copy
from pathlib import Path
import subprocess
import time
import argparse
import shutil
import uuid


def _grid_search(lists):
    if len(lists) == 1:
        return [[x] for x in lists[0]]

    grid = []
    after = _grid_search(lists[1:])
    for x in lists[0]:
        for y in after:
            grid.append([x] + y)

    return grid


def grid_search(parameters: dict):
    """
    Performs a grid search on the given parameter dict, returning a list of 
    dicts with each resulting parameters configuration.

    Examples
    --------
    Input: {"p1": [1, 2, 3], "p2": [1]}
    Output: [{"p1": 1, "p2": 1}, {"p1": 2, "p2": 1}, {"p1": 3, "p2": 1}]
    """
    items = list(parameters.items())
    combinations = _grid_search([x[1] for x in items])

    param = []
    for combination in combinations:
        param.append({items[i][0]: combination[i] for i in range(len(items))})

    return param


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", help="gpus as list", default="1")
parser.add_argument("--nodes", help="nodes as list", default="1")
parser.add_argument("--group", help="group for wandb", required=True)
parser.add_argument("--name", help="name for wandb")
parser.add_argument("--nodelist", help="nodelist to use")
parser.add_argument("--script", help="script")
args = parser.parse_args()

grid_args = copy.deepcopy(vars(args))
grid_args.pop("name")
grid_args.pop("group")
grid_args.pop("nodelist")
grid_args.pop("script")

grid_config = {k: v.split(",") for k, v in grid_args.items()}
configs = grid_search(grid_config)

print(f"config: {configs}")

# copy script to another file to prevent changes mid run
ws_path = Path("/lustre/ssd/ws/s8979104-horovod")
script_name = args.script
script_path = ws_path / "sync/code" / script_name
utils_path = ws_path / "sync/code/utils"

target_path = ws_path / "data/tmp_scripts" / f"tmp_{uuid.uuid4().hex[:8]}"
target_path.mkdir(exist_ok=True, parents=True)

shutil.copy(script_path, target_path)
shutil.copytree(utils_path, target_path / "utils", dirs_exist_ok=True)

for config in configs:
    sbatch = f"""\
#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core

#SBATCH --nodes={config["nodes"]}
#SBATCH --ntasks-per-node={config["gpus"]}
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres="gpu:{config["gpus"]}"
#SBATCH --time=2:00:00
#___SBATCH --exclusive
#SBATCH -p alpha
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/sbatch_%j.log
#SBATCH -J runall
{"" if args.nodelist is None else f"#SBATCH --nodelist {args.nodelist}"}

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_torch

source $VENV/bin/activate

mpirun -N {config["gpus"]} \\
    -bind-to none --oversubscribe \\
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \\
    -mca pml ob1 -mca btl ^openib \\
    $VENV/bin/python -u {(target_path / script_name).resolve()} --data $WS_PATH/data --group {args.group} {"" if args.name is None else f"--name {args.name}"}
    """

    print(f"running with {config}")
    print(sbatch)
    subprocess.run(f"sbatch", shell=True, check=True, input=str.encode(sbatch))
    time.sleep(2)