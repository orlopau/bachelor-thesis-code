import utils.distributed as distributed
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.cnn as cnn
import horovod.torch as hvd
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchinfo import summary
from utils import resnet


def create_datasets(args, mem=False):
    print(f"importing data from {args.data}")

    trans_norm = [
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    ]

    trans_test = transforms.Compose(trans_norm)

    trans_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomCrop(size=(32, 32), padding=4),
        *trans_norm,
    ])

    ds_train = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=trans_train)
    ds_test = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=trans_test)

    def to_mem(ds):
        X = torch.Tensor()
        Y = torch.LongTensor()

        torch.cat(tuple([x[0].unsqueeze(dim=0) for x in ds]), out=X)
        torch.cat(tuple([torch.LongTensor([x[1]]) for x in ds]), out=Y)

        return X, Y

    if mem:
        return (TensorDataset(*to_mem(ds_train)), TensorDataset(*to_mem(ds_test)))
    else:
        return (ds_train, ds_test)


def batch_accuracy_onehot(y, Y):
    """Calculates the accuracy of the batch prediciton, given one hot outputs for classification."""
    truth_map = y.argmax(axis=1).eq(Y)
    return truth_map.sum() / len(truth_map)


class CifarRunner(distributed.Runnable):

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, datasets, meta) -> None:
        super().__init__(net, optimizer, datasets, meta)
        self.loss = nn.CrossEntropyLoss()

    def train(self, loader, optimizer, device):
        accuracy = 0
        for i, (X, Y) in enumerate(loader):
            Y = Y.to(device)

            y = self.net(X.to(device))
            accuracy += batch_accuracy_onehot(y, Y)

            loss = self.loss(y, Y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        return {"acc_train": float(accuracy / len(loader))}

    def test(self, loader, mean_reduce, device):
        with torch.no_grad():
            self.net.eval()

            predictions = torch.Tensor().to(device)
            targets = torch.Tensor().to(device)

            for (X, Y) in loader:
                y = self.net(X.to(device))
                predictions = torch.cat((predictions, y))
                targets = torch.cat((targets, Y.to(device)))

            self.net.train()

            accuracy = mean_reduce(batch_accuracy_onehot(predictions, targets), name="test_reduce")
            test_point = {
                "acc_test": float(accuracy),
            }

            return test_point


config = {
    "batch_size": 256 * 10,
    "lr": 1e-4 * 10,
    "epochs": 50,
    "workers": 8,
    "pre_kernel": 3,
    "pre_stride": 1,
    "pre_padding": 1,
    "model": "res18"
}

models = {"res18": resnet.create_resnet18, "res34": resnet.create_resnet34}

model = models[config["model"]](10, config["pre_kernel"], config["pre_stride"], config["pre_padding"])

optimizer = torch.optim.SGD(model.parameters(), config['lr'], weight_decay=1e-4, momentum=0.9)
runnable = CifarRunner(model, optimizer, create_datasets(distributed.get_args(), mem=False), config)

runner = distributed.DistributedHvdRunner(runnable,
                                          project="resnet_cifar",
                                          batch_size=config["batch_size"],
                                          workers=config["workers"])

if distributed.is_logger():
    summary(model, (1, 3, 32, 32))
runner.start_training(config["epochs"])