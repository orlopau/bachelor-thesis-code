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
import pandas as pd
import numpy as np

# %%
api = wandb.Api(timeout=30)
runs = api.runs("orlopau/hvd", {"group": "baseline"})

# %%
d = []
mean_metrics = ["time_epoch", "time_train", "time_test"]
min_metrics = ["acc_test", "acc_train"]
metrics = ["epoch"]
epoch_cut = 75

for run in runs:
    summary = run.summary
    config = run.config
    history = run.history(keys=mean_metrics+min_metrics+metrics, samples=9999)[:epoch_cut]

    d.append({
        "epoch": history["epoch"].max(),
        "nodes": int(config["slurm"]["SLURM_NNODES"]),
        "gpus": config["horovod"]["size"] // int(config["slurm"]["SLURM_NNODES"]),
        "size": config["horovod"]["size"],
        **history[mean_metrics].add_suffix("_med").median().to_dict(),
        **history[min_metrics].add_suffix("_min").min().to_dict()
    })

df = pd.DataFrame.from_dict(d)
df

# %%
speedup = [x for x in range(1,17)]
speed = float(df.loc[df["size"] == 1]["time_epoch_med"])


fig, ax = plt.subplots()

width = 0.5
for n, color in zip(range(1,5), ["red", "green", "blue", "orange"]):
    df_n = df.loc[df["nodes"] == n]
    df_n = df_n.sort_values(by=["size"])

    ax.plot(df_n["size"], speed / df_n["time_epoch_med"], color=color, label=f"Nodes: {n}", linewidth=1, marker="x", markersize=6)

x = np.arange(1,17)
ax.plot(x, x)

# ahmdahl
part_speedup = 0.9
ax.plot(x, 1 / (1-part_speedup+part_speedup/x), label=f"Ahmdal's law, {part_speedup}")

ax.set_ylabel("speedup")
ax.set_xlabel("# of gpus")
ax.legend()

ax.set_xticks(x)
ax.set_yticks(x)

fig.set_size_inches(8, 5)
fig.tight_layout(pad=5)

plt.tight_layout()
