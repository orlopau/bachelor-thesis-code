import subprocess
import pathlib

config = {"nodes": [1 for _ in range(9)], "gpus": [i + 1 for i in range(9)]}

print(f"config:\n{config}")

iters = len(list(config.values())[0])
for vals in config.values():
    if not len(vals) == iters:
        raise Exception("config not valid")

with open((pathlib.Path(__file__).parent / "sbatch_hvd_stress.sh").resolve(), "r") as template_file:
    template = template_file.read()

    for i in range(iters):
        actual = template
        for (key, vals) in config.items():
            actual = actual.replace("{" + key + "}", str(vals[i]))
        print(f"running with config {i}")
        subprocess.run(f"sbatch", shell=True, check=True, input=str.encode(actual))
