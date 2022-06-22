# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
import horovod.torch as hvd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
import math
import torchinfo
import matplotlib.pyplot as plt
import argparse

# %%
import platform
print(f"starting a run on node with hostname {platform.node()}")


# %%
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False


# %%
BATCH_SIZE = 100
LEARNING_RATE = 0.001
DEEP_LAYERS = [200, 50, 5]
# output channels of conv layers
CONV_LAYERS = [100, 40]

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="path to data dir", default="../data/")

if (isnotebook()):
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

args

# %%
print(f"importing data from {args.data}")

raw_y = np.load(args.data + "parameter.train")
raw_x = np.load(args.data + "prepared_data.train")

X_train, X_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.25, random_state=42)
del raw_x
del raw_y


# %%
def normalize(data):
    max = data.max(axis=0)
    min = data.min(axis=0)

    return np.divide(np.subtract(data, min), max - min)

y_train = normalize(y_train)
y_test = normalize(y_test)

y_train.mean(axis=0)

# %%
dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

del X_train
del X_test
del y_train
del y_test

print(f"created datasets; train={len(dataset_train)}, test={len(dataset_test)}")

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1000, drop_last=True)


# %%
def conv_out_dim(dim_in, convolutions):
    """
    Calculates the output dimensions for a layer of 1D convolutions, given their kernel and stride.

    convolutions: [(kernel_0, stride_0), ..., (kernel_n, stride_n)]
    """

    dims = [dim_in]
    for i, (kernel, stride) in enumerate(convolutions):
        dims.append(math.floor((dims[i] - kernel) / stride + 1))

    return dims[1:]

conv_out_dim(32, [(5,1), (3,3)])

# %%
dim_in = dataset_train.tensors[0][1].size()
dim_out = dataset_train.tensors[1][1].size()
print(f"in: {dim_in}, out: {dim_out}")

# calc conv and deep layers
conv_layer_channels = [dim_in[0]] + CONV_LAYERS
conv_layers = []
for ch_in, ch_out in zip(conv_layer_channels, conv_layer_channels[1:]):
    conv_layers.append(nn.Conv1d(ch_in, ch_out, kernel_size=5))
    conv_layers.append(nn.ReLU())
    conv_layers.append(nn.MaxPool1d(kernel_size=3))


# calc conv output size
def tuple_reducer(t):
    return t[0] if type(t) is tuple else t


conv_dims = conv_out_dim(dim_in[1], [(tuple_reducer(layer.kernel_size), tuple_reducer(layer.stride))
                                     for layer in filter(lambda l: type(l) is not nn.ReLU, conv_layers)])
flatten_dim = conv_layer_channels[-1] * conv_dims[-1]


dense_layer_neurons = [flatten_dim] + DEEP_LAYERS
dense_layers = []
for n_in, n_out in zip(dense_layer_neurons, dense_layer_neurons[1:]):
    dense_layers.append(nn.Linear(n_in, n_out))
    dense_layers.append(nn.ReLU())

dense_layers.append(nn.Linear(dense_layer_neurons[-1], dim_out[0]))

model = nn.Sequential(*conv_layers, nn.Flatten(), *dense_layers)


# %%
hvd.init()

if (hvd.rank() == 0):
    print(torchinfo.summary(model, input_size=(BATCH_SIZE, *dim_in)))

local_rank = hvd.local_rank()

print(f"Local rank {hvd.local_rank()}, world size: {hvd.size()}, world rank: {hvd.rank()}")


device = torch.device("cuda", hvd.local_rank())
model.to(device)

sampler_distributed_train = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
loader_distributed_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=sampler_distributed_train)

criterion = nn.MSELoss()

lr = LEARNING_RATE * hvd.size()
print(f"actual learning rate is {lr}")

optimizer = hvd.DistributedOptimizer(torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * hvd.size()), named_parameters=model.named_parameters())

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

PRINT_EACH = 500
print(f"printing loss each {PRINT_EACH} batches")

import time
start = time.time()

for epoch in range(8):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (X, y) in enumerate(loader_distributed_train):
        optimizer.zero_grad()

        outputs = model(X.to(device))
        loss = criterion(outputs, y.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i + 1) % PRINT_EACH == 0:
            print(f'[Epoch:{epoch + 1}, {(i + 1) * BATCH_SIZE:5d}] loss: {running_loss / PRINT_EACH:.6f}')
            running_loss = 0.0

print(f"training took {(time.time() - start):.2f}s")


# %%
def test_network():
    criterion = nn.MSELoss(reduction='mean')

    with torch.no_grad():
        loss = 0
        for (X, y) in loader_test:
            out = model(X.to(device))
            loss += float(criterion(out, y.to(device)))
        
        return loss / len(loader_test)

print(f"validated loss: {test_network()}")
