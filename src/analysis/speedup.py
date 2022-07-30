# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.10.4 ('venv-pytorch')
#     language: python
#     name: python3
# ---

# %%
import wandb
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import numpy as np

plt.rcParams['figure.dpi'] = 100


# %%
def runs_to_df(runs):
    d = []
    mean_metrics = ["time_epoch", "time_train", "time_test"]
    min_metrics = ["acc_test", "acc_train"]
    metrics = ["epoch"]
    epoch_cut = 75

    for run in runs:
        summary = run.summary
        config = run.config
        history = run.history(keys=mean_metrics+min_metrics+metrics, samples=9999)[:epoch_cut]

        entry = {
            "epoch": summary.epoch,
            "nodes": int(config["slurm"]["SLURM_NNODES"]),
            **history[mean_metrics].add_suffix("_med").median().to_dict(),
            **history[min_metrics].add_suffix("_min").min().to_dict()
        }

        if "horovod" in config:
            entry.update({
                "gpus": config["horovod"]["size"] // int(config["slurm"]["SLURM_NNODES"]),
                "size": config["horovod"]["size"],
            })

        d.append(entry)

    df = pd.DataFrame.from_dict(d)
    return df


# %%
api = wandb.Api(timeout=30)

base_run = api.run("orlopau/hvd/17iyshfo")
base_time_epoch = float(runs_to_df([base_run])["time_epoch_med"])
base_time_epoch

# %%
runs_base = runs_to_df(api.runs("orlopau/hvd", {"group": "baseline_nocrash", "config.horovod.nccl_built": 0, "state": "finished"}))
runs_nccl = runs_to_df(api.runs("orlopau/hvd", {"group": "gpu2_nccl_noib", "state": "finished"}))


# %%
def plot_speedup(dfs: dict, ahmdahl=[0.9, 0.99, 1], size=(8,6)):
    """
    this function plots the speedup of runs vs the number of gpus in relation to ahmdahls law.
    """
    speedup = [x for x in range(1,17)]

    fig, axs = plt.subplots(2,2, figsize=size, sharex=True, sharey=True)

    # ahmdahl
    x = np.arange(1,17)


    width = 0.5
    for i, ax in enumerate(axs.flat, start=1):
        for name, df in dfs.items():
            df_n = df.loc[df["nodes"] == i]
            df_n = df_n.sort_values(by=["size"])

            ax.plot(df_n["size"], base_time_epoch/df_n["time_epoch_med"].to_numpy(), label=f"{name}", linewidth=1, marker="x", markersize=4)

        ax.set_ylabel("speedup")
        ax.set_xlabel("number of gpus")
        ax.set_title(f"{i} node{'s' if i > 1 else ''}")
        ax.set_xticks(x)

    for ax in axs.flat:
        ax.grid()
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # ax_sec = ax.secondary_yaxis("right", functions=(lambda epoch_time: speed / epoch_time, lambda speedup: speed / speedup))
        # ax_sec.set_ylabel("time per epoch")
        for part_speedup in ahmdahl:
            ax.plot(x, 1 / (1-part_speedup+part_speedup/x), label=f"{part_speedup*100:.0f}% parallel", linewidth=.7, linestyle="dashed")

    fig.suptitle("speedup vs # gpus")
    fig.tight_layout()


    labels = [l.get_label() for l in axs[0][0].get_lines()]
    fig.legend(labels, loc='lower right', bbox_to_anchor=(1,-0.05), ncol=len(labels))

plot_speedup({"NCCL": runs_nccl, "MPI": runs_base}, size=(8,5.5))
