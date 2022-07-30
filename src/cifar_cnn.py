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


def create_datasets(args, mem=True):
    print(f"importing data from {args.data}")

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784])
    ])

    ds_train = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=trans)
    ds_test = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=trans)

    def to_mem(ds):
        X = torch.Tensor()
        Y = torch.LongTensor()

        torch.cat(tuple([x[0].unsqueeze(dim=0) for x in ds]), out=X)
        torch.cat(tuple([torch.LongTensor([x[1]]) for x in ds]), out=Y)

        return X, Y

    if mem:
        return (TensorDataset(*to_mem(ds_train)),
                TensorDataset(*to_mem(ds_test)))
    else:
        return (ds_train, ds_test)


config = {
    "batch_size": 128,
    "lr": 1e-4,
    "epochs": 300,
    "cnn_channels": [32, 32, 64, 64],
    "pool_each": 2,
    "linear_layers": [512],
    "workers": 8,
    "dropout": 0.2
}


class DistributedCifarNet(distributed.DistributedNet):

    def create_test_points(self):
        (y, Y) = self.cnn_net.test()
        accuracy = cnn.batch_accuracy_onehot(y, Y)

        test_point = {
            "test_acc": accuracy,
        }

        return test_point


dist_net = DistributedCifarNet(config=config, project="cifar_cnn")
(dataset_train, dataset_test) = create_datasets(dist_net.args)

net_config = cnn.CNNConfig(
    dim_in=(3, 32, 32),
    dim_out=(10,),
    cnn_channels=config["cnn_channels"],
    cnn_convolution_gen=lambda c: nn.Conv2d(c[0], c[1], kernel_size=3, padding=1),
    cnn_pool_gen=lambda: nn.MaxPool2d(kernel_size=2),
    cnn_pool_each=config["pool_each"],
    linear_layers=config["linear_layers"],
    loss_function=nn.CrossEntropyLoss(),
    out_activation=lambda: nn.Softmax(dim=0),
    cnn_activation=nn.ReLU,
    linear_activation=nn.ReLU,
    dropout=config["dropout"])

dist_net.init((dataset_train, dataset_test), net_config)
dist_net.start_training()
