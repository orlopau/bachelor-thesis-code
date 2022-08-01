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
import runs_store as rs

# %%
perf_runs = {}
for p in ["gpu2","hpdlf", "alpha"]:
    runs = rs.load_runs(f"../data/runs/orlopau/{p}/batch_speed_perf.pickle")
    run = rs.reduce_runs(runs)
    # join values in dataframe based on column "batch_size"
    run = run.groupby(run["batch_size"]).aggregate("mean").reset_index()
    perf_runs[p] = run

perf_runs["alpha"].head()


# %%
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
        y = df["time_train_med"]

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

    (ax_gpu, ax_gpu_watt) = axs
    for name, df in run_dfs.items():
        df = df.sort_values(by="batch_size", ascending=True).reset_index(drop=True)
        x = df["batch_size"]

        ax_gpu.plot(x, df["gpu_usage"], label=f"{name}", linewidth=1.5, marker="o", markersize=3)
        ax_gpu_watt.plot(x, df["gpu_power"], label=f"{name}", linewidth=1.5, marker="o", markersize=3)

    ax_gpu_watt.set_ylabel("gpu power (%)")
    axs[0].legend()
    ax_gpu.set_ylabel("gpu usage (%)")

    for ax in axs:
        ax.grid()
        ax.set_xlabel("batch size")
        ax.set_xscale("log")
        ax.set_ylim(0, 100)

    fig.tight_layout()
    return fig, axs

plot_batch_epoch_time(perf_runs)
plot_batch_gpu(perf_runs)
