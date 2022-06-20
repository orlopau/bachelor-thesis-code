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

# %% pycharm={"name": "#%%\n"}
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

# %% pycharm={"name": "#%%\n"}
mnist = datasets.MNIST("data", download=True, transform=transforms.ToTensor())
len(mnist)

# %% pycharm={"name": "#%%\n"}
img = mnist[0]
plt.title("Number " + str(img[1]))
plt.imshow(img[0][0], cmap="gray")

# %% pycharm={"name": "#%%\n"}
# classifiy number 7 with logistic regression
torch.manual_seed(213)

data_train = mnist.data[:-10000]
labels_train = mnist.targets[:-10000]

data_test = mnist.data[-10000:]
labels_test = mnist.targets[-10000:]

weights = torch.randn((1, 28 * 28))


def logreg(x):
    return torch.sigmoid(torch.matmul(weights, x.t()))


def test_accuracy():
    # check accuracy on test dataset
    correct = 0
    incorrect = 0
    for data, label in zip(data_test, labels_test):
        result = logreg(data.flatten().float())
        if (label.item() == 7 and result.item() >= .5):
            correct += 1
            continue
        if (label.item() != 7 and result.item() < .5):
            correct += 1
            continue

        incorrect += 1

    return correct / (correct + incorrect)


for epoch in range(0, 10):
    # create random indexes to sample data
    update = torch.zeros_like(weights)
    counter = 0

    for i in torch.randperm(len(data_train)).tolist():
        data = data_train[i]
        label = labels_train[i]

        y = 1 if label.item() == 7 else 0

        result = logreg(data.flatten().float())
        update += (y - result) * data.flatten().float()

    weights += 0.05 * update
    print(f"accuracy after {epoch}: {test_accuracy()}")

