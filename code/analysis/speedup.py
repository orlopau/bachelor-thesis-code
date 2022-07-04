# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.10.5 ('venv_torch')
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

# %%
results = []

for p in Path("../../data/speedup_single_node").glob("2022*"):
    with open((p / "meta.json").resolve()) as f:
        meta = json.loads(f.read())
        df = pd.read_csv((p / "data.csv").resolve())
        results += [(meta, df)]

results[0][0]

# %%
results = sorted(results, key=lambda x: x[0]["run"]["horovod"]["size"])
xs = [x["run"]["horovod"]["size"] for x in results]
ys = [x["run"]["duration"] for x in results]

fig, ax1 = plt.subplots(sharex=True)
fig.set_size_inches(9,6)
fig.suptitle("Speedup; 1 Node, no NCCL")
ax1.set_ylabel("speedup")
ax1.set_xlabel("# GPUs")
ax1.plot(xs, ys)
ax1.grid(True)
ax2 = ax1.twinx()
ax2.set_ylabel("t (ms)")
ax2.plot(xs, np.divide(ys, ys[0]))


# %%
# import all data as csv
