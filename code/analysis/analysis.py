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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams["figure.figsize"] = (10,10)

# %%
df = pd.read_csv("./data/logs/2022-06-30_12-32/log.csv")
df.head()

# %%
df["acc_test_roll"] = df["acc_test"].rolling(3).mean()
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE")
lines = plt.plot(df.iloc[:, 0], df[["acc_train", "acc_test", "acc_test_roll"]])
plt.legend(iter(lines), ("Accuracy training", "Accuracy test", "Accuracy test averaged"))
