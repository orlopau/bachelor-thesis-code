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

# %% [markdown]
# Generic CNN script with variable hyperparameters for batch size, learning rate, number and width of fully connected layers and channels and number of conv layers.

# %%
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import argparse
import math
from torchinfo import summary
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray


# %%
# Utility functions
def isnotebook():
    """Returns true if the method was called in a notebook."""
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False

def conv_out_dim(dim_in, convolutions, padding):
    """
    Calculates the output dimensions for a layer of 1D convolutions, given their kernel and stride.

    convolutions: [(kernel_0, stride_0), ..., (kernel_n, stride_n)]
    """

    dims = [dim_in]
    for i, (kernel, stride) in enumerate(convolutions):
        dims.append(math.floor((dims[i] - kernel + padding * 2) / stride + 1))

    return dims[1:]


# %%
# Args
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="path to data dir", default="/home/paul/dev/bachelor-thesis/code/data")


if (isnotebook()):
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

# %%
torch.manual_seed(42)

def create_datasets():
    """
    Returns a tuple of train and test datasets.
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5070937, 0.48655552, 0.44092253), 
            (0.26733243, 0.256427, 0.27613324)),
    ])

    ds_train = torchvision.datasets.CIFAR10(args.data, transform=transforms, download=False)
    ds_test = torchvision.datasets.CIFAR10(args.data, train=False, transform=transforms, download=False)
    return (ds_train, ds_test)

def create_loaders(ds_train, ds_test, batch_size=100):
    """Creates the loaders from the given datasets."""
    return (
        torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2),
        torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)
    )

def create_network(
        dim_in, dim_out, out_activation, cnn_channels=[32,32], cnn_activation=nn.ReLU, cnn_convolution=nn.Conv2d,
        hidden_neurons=[64], hidden_activation=nn.ReLU
    ):
    """
    Creates the network.
    """

    # add input channels to cnn layers
    conv_layer_channels = [dim_in[0]] + cnn_channels
    conv_layers = []
    for i, (ch_in, ch_out) in enumerate(zip(conv_layer_channels, conv_layer_channels[1:])):
        conv_layers.append(cnn_convolution(ch_in, ch_out, kernel_size=3, padding=1))
        conv_layers.append(cnn_activation())
        if i != len(conv_layer_channels) - 2:
            conv_layers.append(nn.MaxPool2d(kernel_size=2, padding=1))

    
    # calc conv output size
    def tuple_reducer(t):
        return t[0] if type(t) is tuple else t


    conv_dims = conv_out_dim(dim_in[1], [(tuple_reducer(layer.kernel_size), tuple_reducer(layer.stride))
                                        for layer in filter(lambda l: type(l) is not cnn_activation, conv_layers)], 1)
    flatten_dim = conv_layer_channels[-1] * conv_dims[-1]**2

    print(flatten_dim)

    dense_layer_neurons = [flatten_dim] + hidden_neurons
    dense_layers = []
    for n_in, n_out in zip(dense_layer_neurons, dense_layer_neurons[1:]):
        dense_layers.append(nn.Linear(n_in, n_out))
        dense_layers.append(hidden_activation())

    dense_layers.append(nn.Linear(dense_layer_neurons[-1], dim_out[0]))
    if out_activation is not None:
        dense_layers.append(out_activation)
    
    net = nn.Sequential(*conv_layers, nn.Flatten(), *dense_layers)

    if isnotebook():
        print(net)
        
    return net

def batch_accuracy(Y, y):
    """Calculates the accuracy of the batch prediciton."""
    truth_map = y.argmax(axis=1).eq(Y)
    return int(truth_map.sum()) / len(truth_map)

def test_network(net, loader_test, device):
    with torch.no_grad():
        net.eval()
        accuracy = 0
        for (X, Y) in loader_test:
            y = net(X.to(device))
            accuracy += batch_accuracy(Y.to(device), y)
        net.train()
        return accuracy / len(loader_test)
        
def train(model, optimizer, criterion, loader_train, loader_test, device, epochs=20):
    for epoch in range(epochs):
        accuracy = 0
        for i, (X, Y) in enumerate(loader_train):
            optimizer.zero_grad()

            y = model(X.to(device))
            accuracy += batch_accuracy(Y.to(device), y)

            loss = criterion(y, Y.to(device))
            loss.backward()

            # clip gradients, arbitrary 1
            # nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        
        tune.report(mean_accuracy=test_network(model, loader_test, device), train_acc=accuracy / len(loader_train))


# %%
def train_experiment(config):
    ds_train, ds_test = create_datasets()
    dl_train, dl_test = create_loaders(ds_train, ds_test, batch_size=config["batch_size"])

    #hidden_neurons = [config["l0"] for i in range(config["depth"])]
    net = create_network(ds_train[0][0].shape, [len(ds_train.classes)], nn.Softmax(dim=1),
                         cnn_channels=config["c"], hidden_neurons=config["d"])

    device = torch.device("cuda")
    net.to(device)

    train(net, torch.optim.Adam(net.parameters(), lr=config["lr"]),
          nn.CrossEntropyLoss(), dl_train, dl_test, device, epochs=50)


if not isnotebook():
    ray.init(address="auto")

config = {
    "lr": 0.0008,
    "batch_size": 200,
    "d": [64],
    "c": [32, 64, 64],
    # "depth": tune.grid_search([1, 2]),
    # "log_sys_usage": True
}

scheduler = ASHAScheduler(
    max_t=30,
    grace_period=10,
    reduction_factor=2
)

reporter = None
if isnotebook():
    reporter = tune.JupyterNotebookReporter(overwrite=True)

analysis = tune.run(
    train_experiment, 
    config=config,
    resources_per_trial={"cpu": 2, "gpu": 1},
    local_dir=args.data + "/ray",
    scheduler=scheduler,
    mode="max",
    metric="mean_accuracy",
    num_samples=1,
    fail_fast=True,
    progress_reporter=reporter,
    verbose=3
)

best_experiment = analysis.get_best_trial()

