import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
import argparse
import platform
import utils.cnn as cnn
import time
import matplotlib.pyplot as plt

print(f"starting a run on node with hostname {platform.node()}")

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="path to data dir", default="/home/paul/dev/bachelor-thesis/code/data")
args = parser.parse_args()

def create_datasets():
    print(f"importing data from {args.data}")

    raw_y = np.load(args.data + "/parameter.train")
    raw_x = np.load(args.data + "/prepared_data.train")

    X_train, X_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.25, random_state=42)

    def normalize(data, max, min):
        max = data.max(axis=0)
        min = data.min(axis=0)

        return np.divide(np.subtract(data, min), max - min)

    max = y_train.max(axis=0)
    min = y_train.min(axis=0)

    y_train = normalize(y_train, max, min)
    y_test = normalize(y_test, max, min)

    dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    print(f"created datasets; train={len(dataset_train)}, test={len(dataset_test)}")

    return (dataset_train, dataset_test)

# create datasets, loaders and start training
(dataset_train, dataset_test) = create_datasets()

net_config = cnn.CNNConfig(
    dim_in=dataset_train.tensors[0][1].size(),
    dim_out=dataset_train.tensors[1][1].size(),
    cnn_channels=[50,50],
    cnn_convolution_gen=lambda c: nn.Conv1d(c[0], c[1], kernel_size=5),
    cnn_pool_gen=lambda: nn.MaxPool1d(kernel_size=3),
    linear_layers=[200],
    loss_function=nn.MSELoss(),
    out_activation=None,
    batch_accuracy_func=F.mse_loss
)

BATCH_SIZE = 200
LEARNING_RATE = 0.001
EPOCHS = 50

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True, num_workers=4, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True, num_workers=4)

device = torch.device("cuda")

net = cnn.GenericCNN(net_config, loader_train, loader_test, device)

optimizer = torch.optim.Adam(net.net.parameters(), lr=LEARNING_RATE)
print(net.summary(BATCH_SIZE))

points = []
start = time.time()
for epoch in range(200):
    start_epoch = time.time()
    train_acc = net.train(optimizer)
    test_acc = net.test()
    time_epoch = time.time() - start_epoch
    points.append({"epoch": epoch, "acc_train": train_acc, "acc_test": test_acc, "time_epoch": time_epoch, "time": time.time() - start})
    print(f"Epoch {epoch}, AccTrain={train_acc}, AccTest={test_acc}, Took={time_epoch}")

import pandas as pd
df = pd.DataFrame(points)
df.to_csv(args.data + "/stats.csv")