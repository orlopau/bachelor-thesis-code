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


def run(sbatch_generator):
    """
    The sbatch generator is a function taking a config dict of grid parameters and returning a valid sbatch script.
    The config includes the following entries:
     - gpus
     - nodes
     - nodelist
     - timeline
     - script_path
     - group
     - name
     - sequential
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", help="gpus as list", default="1")
    parser.add_argument("--nodes", help="nodes as list", default="1")
    parser.add_argument("--group", help="group for wandb", required=True)
    parser.add_argument("--name", help="name for wandb")
    parser.add_argument("--nodelist", help="nodelist to use")
    parser.add_argument("--script", help="script")
    parser.add_argument("--timeline", help="set for horovod timeline", action="store_true")
    parser.add_argument("--sequential", help="set for sequential", action="store_true")
    args = parser.parse_args()

    grid_args = copy.deepcopy(vars(args))
    grid_args.pop("name")
    grid_args.pop("group")
    grid_args.pop("nodelist")
    grid_args.pop("script")
    grid_args.pop("timeline")
    grid_args.pop("sequential")

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

    static_config = {
        "nodelist": args.nodelist,
        "timeline": args.timeline,
        "script_path": (target_path / script_name).resolve(),
        "group": args.group,
        "name": args.name,
        "sequential": args.sequential,
    }

    for config in configs:
        config.update(static_config)
        sbatch = sbatch_generator(config)
        print(f"running with {config}")
        print(sbatch)
        subprocess.run(f"sbatch", shell=True, check=True, input=str.encode(sbatch))
        time.sleep(2)