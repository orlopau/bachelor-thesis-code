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
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

# %%
ds_train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
ds_test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())

# %%
ds_train.classes

# %%
mean = ds_train.data.mean(axis=(0,1,2)) / 255
std = ds_train.data.std(axis=(0,1,2)) / 255

mean, std

# %%
ds_train.data.shape

# %%
next(iter(ds_test))[:][1].shape

# %%
X = torch.Tensor()
Y = torch.Tensor()

b = torch.Tensor(10000, 3, 32, 32)
torch.cat(tuple([x[0].unsqueeze(dim=0) for x in ds_test]), out=b).shape


# %%
def to_mem(ds, size):
    X = torch.Tensor(size, 3, 32, 32)
    Y = torch.Tensor(size, 10)

    torch.cat(tuple([x[0].unsqueeze(dim=0) for x in ds]), out=X).shape
    torch.cat(tuple([torch.Tensor([x[1]]) for x in ds]), out=Y).shape

    return X, Y

X, Y = to_mem(ds_test, 10000)
Y.shape
