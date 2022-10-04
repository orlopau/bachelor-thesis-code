import copy
import math
from pathlib import Path
import subprocess
import time
import argparse
import shutil
import uuid
from dataclasses import dataclass, field


@dataclass
class _PartitonConfig:
    cpus_per_task: int
    gpus: int
    mem_per_cpu: int
    args: dict = field(default_factory=lambda: {"nccl": "", "mpi": ""})
    venvs: dict = field(default_factory=lambda: {"nccl": "venv_hvd_nccl", "mpi": "venv_hvd_mpi"})
    module: str = "default"


partitions = {
    "gpu2":
        _PartitonConfig(
            6,
            4,
            2583,
            args={
                "nccl": "export NCCL_P2P_DISABLE=1",
            }),
    "hpdlf":
        _PartitonConfig(
            4,
            3,
            7916),
    "alpha":
        _PartitonConfig(6,
                        8,
                        10312,
                        venvs={
                            "nccl": "venv_hvd_nccl_alpha",
                            "mpi": "venv_hvd_mpi"
                        },
                        module="alpha"),
}


def gen_sbatch(config):
    p = partitions[config["partition"]]

    return f"""\
#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core, 10312M -> 6 cores per gpu
# gpu2: 24 cores, 4 gpus, ~2.5G per core, 2583M -> 6 cores per gpu
# hpdlf: 12 cores, 3 gpus, 7916M per core, 3 gpus, 4 cores per GPU

#SBATCH --nodes={config["nodes"]}
#SBATCH --ntasks={config["tasks"]}
#SBATCH -m plane={config["gpus"]}
#SBATCH --cpus-per-task={p.cpus_per_task}
#__SBATCH --mem=0
#SBATCH --mem-per-cpu={p.mem_per_cpu}M
#SBATCH --gres="gpu:{config["gpus"]}"
#SBATCH --time=0:40:00
#__SBATCH --exclusive=user
#SBATCH -p {config["partition"]}
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/{config["partition"]}/sbatch_%j.log
#SBATCH -J runall_{config["partition"]}_{config["mode"]}
{"" if config["nodelist"] is None else f"#SBATCH --nodelist {config['nodelist']}"}

ml restore {p.module}

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/{p.venvs[config["mode"]]}

source $VENV/bin/activate

# export SLURM_TASKS_PER_NODE="{config["gpus"]}(x{config["nodes"]})"
#
# mpirun -np {config["tasks"]} \\
#     -bind-to none -map-by slot \\
#     -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_IB_DISABLE=1 {"-x HOROVOD_TIMELINE=$WS_PATH/timeline.data" if config["timeline"] else ""}\\
#     -mca pml ob1 -mca btl ^openib {p.args.get(config["mode"], "")} \\
#     $VENV/bin/python -u {config["script_path"]} --data $WS_PATH/data --dist --group {config["group"]} \\
#     --project {config["partition"]} {"" if config["name"] is None else f"--name {config['name']}"}

OMPI_MCA_btl='^ofi'
OMPI_MCA_mtl='^ofi'
export OMPI_MCA_btl='^ofi'
export OMPI_MCA_mtl='^ofi'
export NCCL_DEBUG=INFO
{p.args.get(config["mode"], "")}

export SLURM_CPU_BIND_TYPE="threads,none"
export SLURM_CPU_BIND="verbose,threads,none"
export SLURM_CPU_BIND_LIST=""

srun --cpu-bind=none,v --accel-bind=gn $VENV/bin/python -u {config["script_path"]} --data $WS_PATH/data --dist --group {config["group"]} \\
      --project {config["partition"]} {"" if config["name"] is None else f"--name {config['name']}"}
"""


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


def sequential(parameters: dict):
    p = dict(parameters)
    # use first element from each parameter entry
    for k, v in p.items():
        p[k] = v[0]

    nodes = int(p["nodes"])
    gpus = int(p["gpus"])
    del p["nodes"]
    del p["gpus"]

    # iterate to max gpus then add node
    for n in range(nodes):
        for g in range(gpus):
            yield {"nodes": n + 1, "gpus": gpus, "tasks": (n * gpus) + g + 1, **p}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", help="partition", required=True)
    parser.add_argument("--gpus", help="gpus as list", default="1")
    parser.add_argument("--tasks", help="tasks to run", default="1")
    parser.add_argument("--nodes", help="nodes as list", default="1")
    parser.add_argument("--mode", help="mode, nccl or mpi", default="nccl")
    parser.add_argument("--group", help="group for wandb", required=True)
    parser.add_argument("--name", help="name for wandb")
    parser.add_argument("--nodelist", help="nodelist to use")
    parser.add_argument("--script", help="script")
    parser.add_argument("--timeline", help="set for horovod timeline", action="store_true")
    parser.add_argument("--sequential", help="set for sequential", action="store_true")
    parser.add_argument("--single", help="set for single", action="store_true")
    args = parser.parse_args()

    grid_args = copy.deepcopy(vars(args))
    grid_args.pop("name")
    grid_args.pop("group")
    grid_args.pop("nodelist")
    grid_args.pop("script")
    grid_args.pop("timeline")
    grid_args.pop("sequential")
    grid_args.pop("single")
    grid_args.pop("tasks")

    grid_config = {k: v.split(",") for k, v in grid_args.items()}

    if args.sequential:
        configs = list(sequential(grid_config))
    elif args.single:
        config = list(sequential(grid_config))[-1]
        config["tasks"] = args.tasks
        configs = [config]
    else:
        configs = grid_search(grid_config)
        for c in configs:
            c["tasks"] = int(c["nodes"]) * int(c["gpus"])

    print(f"config: {configs}")

    # copy script to another file to prevent changes mid run
    ws_path = Path("/lustre/ssd/ws/s8979104-horovod")
    script_name = args.script
    script_path = ws_path / "sync/src" / script_name
    utils_path = ws_path / "sync/src/utils"

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
        sbatch = gen_sbatch(config)
        print(f"running with {config}")
        print(sbatch)
        subprocess.run(f"sbatch", shell=True, check=True, input=str.encode(sbatch))
        time.sleep(2)


if __name__ == "__main__":
    run()