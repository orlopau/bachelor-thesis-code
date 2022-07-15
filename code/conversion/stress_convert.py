"""
Converts stress data to normalized format, with adjustable x and y normalization.

Args:

0:
xy - normalize x+y
x  - normalize x
y  - normalize y

1:
output path for npz file

2
input dir for original files
ex: "../data"
"""

import sys
import numpy as np
import pandas as pd

raw_y = np.load(sys.argv[3] + "/parameter.train")
raw_x = np.load(sys.argv[3] + "/prepared_data.train")

# split data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(raw_x,
                                                    raw_y,
                                                    test_size=0.25,
                                                    random_state=42,
                                                    shuffle=True)
del raw_x
del raw_y

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def stat_x(x):
    max = x.max(axis=(0, 2))[:, np.newaxis]
    min = x.min(axis=(0, 2))[:, np.newaxis]
    mean = x.mean(axis=(0, 2))[:, np.newaxis]
    std = x.std(axis=(0, 2))[:, np.newaxis]
    return max, min, mean, std


def stat_y(y):
    max = y.max(axis=0)
    min = y.min(axis=0)
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    return max, min, mean, std


def format_stat(stat):
    return f"max: {stat[0]}\nmin: {stat[1]}\nmean: {stat[2]}\nstd: {stat[3]}"


def describe():
    print("X_train")
    print(format_stat(stat_x(X_train)))
    print("X_test")
    print(format_stat(stat_x(X_test)))
    print("Y_train")
    print(format_stat(stat_y(Y_train)))
    print("Y_test")
    print(format_stat(stat_y(Y_test)))


print("Pre")
describe()

x_max, x_min = X_train.max(axis=(0, 2))[:, np.newaxis], X_train.min(axis=(0, 2))[:, np.newaxis]
y_max, y_min = Y_train.max(axis=0), Y_train.min(axis=0)


def normalize(data, max, min):
    return np.divide(np.subtract(data, min), max - min)

if "y" in sys.argv[1]:
    print("norm y train")
    Y_train = normalize(Y_train, y_max, y_min)
    print("norm y test")
    Y_test = normalize(Y_test, y_max, y_min)

if "x" in sys.argv[1]:
    print("norm x train")
    X_train = normalize(X_train, x_max, x_min)
    print("norm x test")
    X_test = normalize(X_test, x_max, x_min)

print("\n\n\nNormalized")
describe()

meta = {"x_max": x_max, "x_min": x_min, "y_max": y_max, "y_min": y_min}
np.savez(sys.argv[2],
         X_train=X_train,
         X_test=X_test,
         Y_train=Y_train,
         Y_test=Y_test,
         meta=meta)
