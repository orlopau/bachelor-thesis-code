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
import runs_store as rs

plt.rcParams['figure.dpi'] = 100


# %%
def load_group(group, partitions=["alpha", "hpdlf", "gpu2"]):
    result = {}
    for p in partitions:
        runs = rs.load_runs(f"../data/runs/orlopau/{p}/{group}.pickle")
        runs = rs.reduce_runs(runs).sort_values("size").reset_index(drop=True)
        result[p] = runs

    return result

def aggregate_runs(runs: pd.DataFrame, groupby="size"):
    runs = runs.groupby([groupby]).mean().reset_index()
    int_cols = ["size", "epoch", "nodes", "gpus", "batch_size"]
    runs[int_cols] = runs[int_cols].astype(int)
    return runs

aggregate_runs(load_group("nccl_sequential")["gpu2"]).head()

# %%
from sklearn.linear_model import LinearRegression

def calc_base_time(dfs):
    times = [df.loc[df["size"] == 1]["time_epoch_med"].max() for df in dfs]
    return np.max(times)

def plot_ahmdahl(ax, x, parallel_parts=[0.8,0.9,1]):
    """
    Plots the Ahmdahl speedup according to ahmdahl's law.
    """
    for parallel_part in parallel_parts:
        ax.plot(x, 1 / (1-parallel_part+parallel_part/x), label=f"{parallel_part*100:.0f}% parallel", linewidth=.7, linestyle="dashed")


def plot_speedup_sequential(ax, groups, base_group=False):
    """
    Plots the speedup of the given runs compared to the base run.
    """
    
    ax.set_ylabel("speedup")
    ax.set_xlabel("number of gpus")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if base_group: base_time = calc_base_time(groups.values())
    for name, df in groups.items():
        if not base_group: base_time = calc_base_time([df])
        df = aggregate_runs(df)
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
    ax.plot(np.arange(1,x+1), np.arange(1,x+1), linewidth=1, linestyle="dashdot", label="Ideal")
    ax.legend()

def plot_efficiency(ax, groups):
    ax.set_ylabel("efficiency")
    ax.set_xlabel("number of gpus")
    
    for name, df in groups.items():
        df = aggregate_runs(df)
        base_time = calc_base_time([df])
        df = df.sort_values(by="size")

        efficiency = (base_time/df["size"]) / df["time_epoch_med"]
        x = df["size"]

        ax.plot(x, efficiency, label=f"{name}", linewidth=1.5, marker="o", markersize=3)

    max_x = int(ax.get_xlim()[1])
    min_x = int(ax.get_xlim()[0]) + 1
    ax.plot(np.arange(min_x, max_x), np.ones((max_x-min_x)), linewidth=1, linestyle="dashdot", label="Ideal")
    ax.legend(loc=1)

def plot_step_time(ax, run_dfs):
    ax.set_ylabel("step time (ms)")
    ax.set_xlabel("number of gpus")
    
    for name, df in run_dfs.items():
        df = aggregate_runs(df)
        df.sort_values(by="size", inplace=True)
        x = df["size"]
        y = (df["time_step_avg_med"]) * 1000

        ax.plot(x, y, label=f"{name}", linewidth=1.5, marker="o", markersize=4)

    ax.legend(loc=4)

def plot_synchro_time(ax, run_dfs):
    ax.set_ylabel("synchronization time (ms)")
    ax.set_xlabel("number of gpus")
    
    for name, df in run_dfs.items():
        df = aggregate_runs(df)
        df.sort_values(by="size", inplace=True)
        x = df["size"]
        base = df["time_step_avg_med"][0]
        y = (df["time_step_avg_med"] - base) * 1000

        ax.plot(x, y, label=f"{name}", linewidth=1.5, marker="o", markersize=4)

    ax.legend(loc=4)

def plot_epoch_time(ax, groups):
    ax.set_ylabel("epoch time (s)")
    ax.set_xlabel("number of gpus")
    
    for name, df in groups.items():
        df = aggregate_runs(df)
        x = df["size"]
        y = df["time_train_med"]

        ax.plot(x, y, label=f"{name}", linewidth=1.5, marker="o", markersize=3)

    ax.legend(loc=1)

def plot_gpu_usage(groups):
    fig, axs = plt.subplots(2, figsize=(6,6), sharex=True)
    fig.suptitle("GPU Usage vs. # of GPUs")

    (ax_gpu, ax_gpu_watt) = axs
    for name, df in groups.items():
        df = aggregate_runs(df)
        df = df.sort_values(by="size").reset_index()
        x = df["size"]

        ax_gpu.plot(x, df["gpu_usage"], label=f"{name}", linewidth=1.5, marker="o", markersize=3)
        ax_gpu_watt.plot(x, df["gpu_power"], label=f"{name}", linewidth=1.5, marker="o", markersize=3)

    ax_gpu_watt.set_ylabel("gpu power (%)")
    axs[0].legend()
    ax_gpu.set_ylabel("gpu usage (%)")

    for ax in axs:
        ax.grid()
        ax.set_xlabel("number of gpus")
        ax.set_ylim(0, 100)

    fig.tight_layout()

def plot_step_time_epoch(ax, run_dfs):
    ax.set_ylabel("synchronization time per epoch (s)")
    ax.set_xlabel("number of gpus")

    ax_batches = ax.twinx()
    ax_batches.set_ylabel("batches per epoch")
    x = np.arange(17)
    ax_batches.plot(x, (750000/x) / 75, label="batches per epoch", linewidth=1.5, linestyle="dashdot", color="black")
    ax_batches.legend(loc=3)
    
    for name, df in run_dfs.items():
        df = aggregate_runs(df)
        df.sort_values(by="size", inplace=True)
        x = df["size"]
        base = df.iloc[0]["time_step_avg_med"]
        y = (df["time_step_avg_med"] - base) * ((750000/x) / 75)

        ax.plot(x[1:], y[1:], label=f"{name}", linewidth=1.5, marker="o", markersize=4)

    ax.legend()

def plot_communication_share(ax, groups):
    ax.set_ylabel("synchronization time share (%)")
    ax.set_xlabel("number of gpus")
    
    for name, df in groups.items():
        df = aggregate_runs(df)
        df.sort_values(by="size", inplace=True)
        x = df["size"]
        base_time_share = df.iloc[0]["time_step_med"] / df.iloc[0]["time_train_med"]
        y = (df["time_step_med"]) / df["time_train_med"] - base_time_share
        y *= 100
        ax.plot(x, y, label=f"{name}", linewidth=1.5, marker="o", markersize=4)

    ax.legend()

def plot_all(groups):
    fig, (ax, ax1) = plt.subplots(2, figsize=(6,6))
    fig.suptitle("Speedup and Efficiency vs. # of GPUs")
    ax.grid()
    ax1.grid()

    plot_speedup_sequential(ax, groups, base_group=False)
    plot_efficiency(ax1, groups)

    fig.tight_layout()

    fig, ax = plt.subplots(1)
    fig.suptitle("Epoch time vs. # of GPUs")
    ax.grid()

    plot_epoch_time(ax, groups)

    fig.tight_layout()

    plot_gpu_usage(groups)

    fig, ax = plt.subplots(1)
    ax.grid()
    fig.suptitle("Gradient synchronization time vs. # of GPUs")
    plot_synchro_time(ax, groups)

    fig, ax = plt.subplots()
    ax.grid()
    plot_communication_share(ax, groups)
    fig, ax = plt.subplots()
    ax.grid()
    plot_step_time_epoch(ax, groups)
    fig, ax = plt.subplots()
    ax.grid()
    plot_step_time(ax, groups)



nccl_sequential = load_group("nccl_sequential")
plot_all(nccl_sequential)

# mpi_sequential = load_group("mpi_sequential", partitions=["alpha", "hpdlf"])
# plot_all(mpi_sequential)

# %%
batch_runs = load_group("batch_size", partitions=["gpu2"])["gpu2"]
batch_runs = batch_runs.sort_values(by=["batch_size", "size"]).reset_index(drop=True)
df = batch_runs

fig, ax = plt.subplots()
ax.grid()
ax.set_ylabel("mean squared error")
ax.set_xlabel("# of GPUs")
fig.suptitle("Test MSE vs. GPUs")

for bs in df["batch_size"].unique():
    df_bs = df[df["batch_size"] == bs]
    df_bs = df_bs.sort_values(by="size").reset_index(drop=True)
    x = df_bs["size"]
    y = df_bs["acc_test_min"]
    ax.plot(x, y, label=f"{bs}", linewidth=1.5, marker="o", markersize=4)
fig.tight_layout()

# %%
runs = rs.load_runs(f"../data/runs/orlopau/gpu2/batch_size.pickle")
runs = pd.DataFrame.from_dict([rs.run_to_reduced_dict(run) for run in runs])
runs = runs.sort_values(by=["batch_size", "size"]).reset_index(drop=True)
runs.iloc[0]["history"]

# %%
fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(12,3))

for bs, ax in zip(runs["batch_size"].unique(), axs.flatten()):
    ax.set_yscale("log")
    ax.set_ylabel("MSE")
    ax.set_xlabel("time (min)")
    ax.set_title(f"Batch Size {bs}")
    bs_runs = runs[runs["batch_size"] == bs]
    for r in bs_runs.itertuples():
        ax.plot(r.history["time"] / 60, r.history["acc_test"], label=f"{r.size} GPUs")

axs.flatten()[0].legend()
fig.tight_layout()
