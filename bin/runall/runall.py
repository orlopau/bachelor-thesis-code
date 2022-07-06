import subprocess
import pathlib
import time
import sys
import argparse


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
parser.add_argument("--nodelist", help="nodelist to use")
# parser.add_argument("executable", help="executable to run", nargs=1)
args = parser.parse_args()

# executable = args.executable
# del args.executable

grid_config = {k: v.split(",") for k, v in vars(args).items()}
configs = grid_search(grid_config)

print(f"config:\n{configs}")

with open((pathlib.Path(__file__).parent / "sbatch_hvd_stress.sh").resolve(),
          "r") as template_file:

    template = template_file.read()
    for config in configs:
        actual = template
        for (k, v) in config.items():
            actual = actual.replace("{" + k + "}", str(v))

        actual = actual.replace("{mpi_np}", str(int(config["gpus"]) * int(config["nodes"])))

        print(f"running with {config}")
        subprocess.run(f"sbatch", shell=True, check=True, input=str.encode(actual))
        time.sleep(2)
