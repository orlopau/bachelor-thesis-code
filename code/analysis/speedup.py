import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

results = []
dir = Path("../../data/speedup_single_node")
out_dir = dir / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    fig.savefig((out_dir / f"{name}.svg").resolve())
    fig.savefig((out_dir / f"{name}.png").resolve(), facecolor='white', transparent=False)

for p in dir.glob("2022*"):
    with open((p / "meta.json").resolve()) as f:
        meta = json.loads(f.read())
        df = pd.read_csv((p / "data.csv").resolve())
        results += [(meta, df)]

results = sorted(results, key=lambda x: x[0]["horovod"]["size"])

metas = [x[0] for x in results]
xs = [x["horovod"]["size"] for x in metas]
ys = [x["run"]["duration"] for x in metas]

fig, ax1 = plt.subplots(sharex=True)
fig.set_size_inches(9,6)
fig.suptitle(f"Speedup; {metas[0]['horovod']['size']} Node, no NCCL")
ax1.set_ylabel("t (s)")
ax1.set_xlabel("# of GPUs")
ax1.xaxis.get_major_locator().set_params(integer=True)
ax1.plot(xs, ys)
ax1.grid(True)
ax2 = ax1.twinx()
ax2.set_ylabel("speedup")
ax2.plot(xs, np.divide(ys, ys[0]))
save_fig(fig, "speedup")


# plot test acc for all networks
fig, ax = plt.subplots(sharex=True)
fig.set_size_inches(9,6)
fig.suptitle("Train accuracy; 1 Node, no NCCL; same batch size")
ax.set_ylabel("test accuracy")
ax.set_xlabel("epoch")
#ax.set_yscale("log")
ax.xaxis.get_major_locator().set_params(integer=True)

for result in results[0:8]:
    df = result[1]
    cut = 5
    ax.plot(df["epoch"][cut:], df["acc_train"][cut:], label=result[0]["horovod"]["size"])

fig.legend()
save_fig(fig, "acc_train")
