import numpy as np
import pandas as pd

raw_y = np.load("../data/parameter.train")
raw_x = np.load("../data/prepared_data.train")

# split data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(raw_x,
                                                    raw_y,
                                                    test_size=0.25,
                                                    random_state=42,
                                                    shuffle=True)

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


X_train, X_test, Y_train, Y_test = normalize(X_train, x_max,
                                             x_min), normalize(X_test, x_max, x_min), normalize(
                                                 Y_train, y_max, y_min), normalize(Y_test, y_max, y_min)

print("\n\n\nNormalized")
describe()

meta = {
    "x_max": x_max,
    "x_min": x_min,
    "y_max": y_max,
    "y_min": y_min
}
np.savez("/lustre/ssd/ws/s8979104-horovod/data/stress_normalized", X_train=X_train, X_test = X_test, 
Y_train = Y_train, Y_test = Y_test, meta = meta)