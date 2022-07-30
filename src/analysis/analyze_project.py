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
#     display_name: 'Python 3.10.5 (''venv-pytorch'': venv)'
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
import sys
api = wandb.Api(timeout=30)
max_samples = 999999

def exclude_map_keys(map, keys):
    return {k: map[k] for k in map.keys() if not any(e in k for e in keys)}

def run_to_dict(run, exclude_keys=["gradients", "parameters"]):
    summary = exclude_map_keys(run.summary, exclude_keys)
    config = run.config
    history = pd.DataFrame.from_dict([exclude_map_keys(h, exclude_keys) for h in run.scan_history()])
    history_metrics = run.history(samples=max_samples, stream="system")

    return {
        "summary": summary,
        "config": config,
        "history": history,
        "history_metrics": history_metrics
    }


def runs_to_df(runs):
    d = []
    mean_metrics = ["time_epoch", "time_train", "time_test", "time_step_avg", "time_step"]
    min_metrics = ["acc_test", "acc_train"]
    metrics = ["epoch"]
    epoch_cut = 75

    for run in runs:
        summary = run.summary
        config = run.config
        history = run.history(keys=mean_metrics+min_metrics+metrics, samples=9999)[:epoch_cut]

        entry = {
            "epoch": summary.epoch,
            "nodes": int(config["slurm"]["SLURM_NNODES"]) if "slurm" in config else 1,
            "gpus": config["horovod"]["size"] // int(config["slurm"]["SLURM_NNODES"]) if "horovod" in config else 1,
            "size": config["horovod"]["size"] if "horovod" in config else 1,
            "batch_size": config["run"]["batch_size"] if "run" in config else config["batch_size"],
            **history[mean_metrics].add_suffix("_med").median().to_dict(),
            **history[min_metrics].add_suffix("_min").min().to_dict()
        }

        d.append(entry)

    df = pd.DataFrame.from_dict(d)
    df = df.sort_values(by="size")
    return df

run = api.runs("orlopau/hpdlf", {"group": "nccl"})[0]
run_to_dict(run)


# %%
def get_named_run(project, name):
    runs = list(filter(lambda x: x.name == "base", api.runs(project)))
    if len(runs) != 1:
        raise Exception("No base run found, must have name 'base'")
        
    return runs[0]

def get_group_runs(project, group, cache_evict=False):
    runs = api.runs(project, {"group": group})
    return runs_to_df(runs)

nccl_gpu2 = get_group_runs("orlopau/gpu2", "nccl")
nccl_gpu2_sequential = get_group_runs("orlopau/gpu2", "nccl_sequential")
nccl_hpdlf = get_group_runs("orlopau/hpdlf", "nccl")
nccl_hpdlf_sequential = get_group_runs("orlopau/hpdlf", "nccl_sequential")
nccl_hpdlf_nop2p = get_group_runs("orlopau/hpdlf", "nccl_no_p2p")

# %%
batch_speedup = {
    "gpu2": get_group_runs("orlopau/gpu2", "batch_speedup"),
    "hpdlf": get_group_runs("orlopau/hpdlf", "batch_speedup"),
    "alpha": get_group_runs("orlopau/alpha", "batch_speedup"),
}

# %%
hpdlf_base = runs_to_df([get_named_run("orlopau/hpdlf", "base")])
hpdlf_base

# %%
f = dict.fromkeys(nccl_gpu2, 'min')
f.update({
    "size": "first",
    "epoch": "first",
    "nodes": "mean",
    "gpus": "mean"
})

nccl_gpu2.groupby(by="size").agg(f)

# %%
from sklearn.linear_model import LinearRegression

def calc_base_time(base_df):
    base = base_df.loc[base_df["size"] == 1]["time_epoch_med"]
    return float(base)

def plot_ahmdahl(ax, x, parallel_parts=[0.8,0.9,1]):
    """
    Plots the Ahmdahl speedup according to ahmdahl's law.
    """
    for parallel_part in parallel_parts:
        ax.plot(x, 1 / (1-parallel_part+parallel_part/x), label=f"{parallel_part*100:.0f}% parallel", linewidth=.7, linestyle="dashed")


def plot_speedup_sequential(ax, run_dfs, base=None):
    """
    Plots the speedup of the given runs compared to the base run.
    """
    
    ax.set_ylabel("speedup")
    ax.set_xlabel("number of gpus")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for name, df in run_dfs.items():
        if base is not None:
            base_time = calc_base_time(base)
        else:
            base_time = calc_base_time(df)
        df = df.sort_values(by="size")

        speedup = base_time/df["time_epoch_med"]
        max_speedup = np.max(speedup)
        x = df["size"]

        # plot linear speedup
        reg = LinearRegression()
        reg.fit(x.values.reshape(-1, 1), speedup)

        l = ax.plot(x, speedup, label=name, linewidth=1.5, marker="o", markersize=4)[0]
        ax.plot(x.values.reshape(-1, 1), reg.predict(x.values.reshape(-1, 1)), label=f"{float(reg.coef_):.2f}*gpus", linewidth=1, linestyle="dashdot", color=l.get_color())

    x = np.floor(ax.get_xlim()[1]).astype(int)
    ax.plot(np.arange(1,x+1), np.arange(1,x+1), linewidth=1, linestyle="dashdot", label="1.00*gpus")
    ax.legend()

def plot_step_time(ax, run_dfs):
    ax.set_ylabel("step time (ms)")
    ax.set_xlabel("number of gpus")
    
    for name, df in run_dfs.items():
        x = df["size"]
        y = df["time_step_avg_med"]

        ax.plot(x, y, label=f"{name}", linewidth=1.5, marker="o", markersize=4)

    ax.legend(loc=4)

def plot_epoch_time(ax, run_dfs):
    ax.set_ylabel("epoch time (s)")
    ax.set_xlabel("number of gpus")
    
    for name, df in run_dfs.items():
        x = df["size"]
        y = df["time_epoch_med"]

        ax.plot(x, y, label=f"{name}", linewidth=1.5, marker="o", markersize=4)

    ax.legend(loc=1)

def plot_batch_epoch_time(run_dfs):
    """
    Plots the time per epoch and speedup vs the batch size.
    """

    fig, axs = plt.subplots(2, figsize=(6,6), sharex=True)
    fig.suptitle("Epoch Time vs. Batchsize of 75 to 5000")

    (ax_time, ax_speed) = axs
    for name, df in run_dfs.items():
        df = df.sort_values(by="batch_size", ascending=True).reset_index(drop=True)
        x = df["batch_size"]
        y = df["time_epoch_med"]

        ax_speed.plot(x, y[0]/y.to_numpy(), label=f"{name}", linewidth=1.5, marker="o", markersize=3)
        ax_time.plot(x, y, label=f"{name}", linewidth=1.5, marker="o", markersize=3)

    ax_time.set_ylabel("time per epoch (s)")
    axs[0].legend()
    ax_speed.set_ylabel("speedup")

    for ax in axs:
        ax.grid()
        ax.set_xlabel("batch size")
        ax.set_xscale("log")

    fig.tight_layout()
    return fig, axs

def plot_batch_gpu(run_dfs):
    """
    Plots the gpu usage vs the batch size.
    """
    
    fig, axs = plt.subplots(2, figsize=(6,6), sharex=True)
    fig.suptitle("GPU Usage vs. Batchsize of 75 to 5000")

    (ax_gpu, ax_time) = axs
    for name, df in run_dfs.items():
        df = df.sort_values(by="batch_size", ascending=True).reset_index(drop=True)
        x = df["batch_size"]
        y = df["gpu_usage"]

        ax_gpu.plot(x, y[0]/y.to_numpy(), label=f"{name}", linewidth=1.5, marker="o", markersize=3)
        ax_time.plot(x, y, label=f"{name}", linewidth=1.5, marker="o", markersize=3)

    ax_time.set_ylabel("time per epoch (s)")
    axs[0].legend()
    ax_gpu.set_ylabel("gpu usage")

    for ax in axs:
        ax.grid()
        ax.set_xlabel("batch size")
        ax.set_xscale("log")

    fig.tight_layout()
    return fig, axs

plot_batch_epoch_time(batch_speedup)
plot_batch_gpu(batch_speedup)
