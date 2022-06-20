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
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

# %%
mnist = datasets.MNIST("data", download=True, transform=transforms.ToTensor())
len(mnist)

# %%
img = mnist[0]
plt.title("Number " + str(img[1]))
plt.imshow(img[0][0], cmap="gray")
