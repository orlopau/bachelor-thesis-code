from cProfile import label
import math
from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
import traceback


def generate_plots(dir):
    out_dir = dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"generating plots for {dir.resolve()}...")

    def save_fig(fig, name):
        # fig.savefig((out_dir / f"{name}.svg").resolve())
        fig.savefig((out_dir / f"{name}.png").resolve(), facecolor='white', transparent=False)

    results = []
    for p in dir.glob("2022*"):
        with open((p / "meta.json").resolve()) as f:
            meta = json.loads(f.read())
            df = pd.read_csv((p / "data.csv").resolve())
            results += [(meta, df)]

    def keyer(x):
        key = (int(x[0]["slurm"]["SLURM_NNODES"]), x[0]["horovod"]["size"])
        return key

    results = sorted(results, key=keyer)

    metas = [x[0] for x in results]

    # plot speedup
    fig, ax1 = plt.subplots(sharex=True, sharey=True)
    fig.set_size_inches(9, 6)
    fig.suptitle(f"Speedup")
    ax1.set_ylabel("t (s)")
    ax1.set_xlabel("# of GPUs")
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax2 = ax1.twinx()

    nnode_configs = np.unique([x["slurm"]["SLURM_NNODES"] for x in metas])
    first_run_time = metas[0]["run"]["duration"]
    for nnode in nnode_configs:
        runs = list(filter(lambda x: x["slurm"]["SLURM_NNODES"] == nnode, metas))
        xs = [x["horovod"]["size"] for x in runs]
        ys = [x["run"]["duration"] for x in runs]

        ax1.plot(xs, ys, marker='o')
        ax1.grid(True)
        ax2.set_ylabel("speedup")
        ax2.plot(xs, np.divide(ys, first_run_time), label=f"Nodes: {nnode}")

    ax2.legend()
    save_fig(fig, "speedup")

    # plot acc
    fig, ax = plt.subplots(sharex=True)
    fig.set_size_inches(9, 6)
    fig.suptitle("Train accuracy")
    ax.set_ylabel("train accuracy")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    ax.xaxis.get_major_locator().set_params(integer=True)

    for result in results:
        df = result[1]
        # df["acc_test_roll"] = df["acc_test"].rolling(3).mean()
        cut = 0
        ax.plot(df["epoch"][cut:], df["acc_train"][cut:], label=result[0]["horovod"]["size"])

    fig.legend()
    save_fig(fig, "acc_train")

    # plot all train vs test accs
    fig, _ = plt.subplots(math.ceil(len(results) / 2), ncols=2, sharex=True, sharey=True)
    axs = fig.axes
    fig.set_size_inches(10, 1.5 * len(results))

    for result, ax in zip(results, axs):
        ax.set_title(
            f'#gpus: {result[0]["horovod"]["size"]}, #nodes: {result[0]["slurm"]["SLURM_NNODES"]}, lr: {result[0]["run"]["config"]["lr"]:.5f}'
        )
        ax.set_ylabel("mse")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.xaxis.get_major_locator().set_params(integer=True)
        df = result[1]
        cut = 1

        ax.plot(df["epoch"][cut:], df["acc_train"][cut:], label="acc_train")
        ax.plot(df["epoch"][cut:], df["acc_test"][cut:], label="acc_test")
        ax.legend()

    fig.tight_layout()
    save_fig(fig, "acc_test")

    # create meta file
    with open((dir / "summary.txt").resolve(), "w") as file:
        meta = f"""
        tbd
        """
        file.write(meta)


dir = Path(argv[1])
generate_plots(dir)
# for sub in dir.glob('.'):
#     try:
#         generate_plots(sub)
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         traceback.print_exc()
