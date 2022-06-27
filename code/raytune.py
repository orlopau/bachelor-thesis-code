# Generic CNN script with variable hyperparameters for batch size, learning rate, number and width of fully connected layers and channels and number of conv layers.

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import argparse
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
import utils.cnn as cnn
import utils.generic as utils_generic
import utils.ml as utils_ml

torch.manual_seed(42)

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="path to data dir",
                    default="/home/paul/dev/bachelor-thesis/code/data")
parser.add_argument(
    "--ray", help="set to local to use new ray instance", default="remote")

if (utils_generic.isnotebook()):
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

class CIFARTrainable(tune.Trainable):
    def setup(self, config):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5070937, 0.48655552, 0.44092253),
                                             (0.26733243, 0.256427, 0.27613324)),
        ])
        ds_train = torchvision.datasets.CIFAR10(
            args.data, transform=transforms, download=False)
        ds_test = torchvision.datasets.CIFAR10(
            args.data, train=False, transform=transforms, download=False)

        print(config)
        dl_train, dl_test = utils_ml.create_loaders(
            ds_train, ds_test, batch_size=config["batch_size"])

        net_config = cnn.CNNConfig(
            ds_train[0][0].shape,
            [len(ds_train.classes)],
            config["c"],
            nn.Conv2d,
            config["d"],
            torch.nn.CrossEntropyLoss(),
        )

        device = torch.device("cuda")
        self.net = cnn.GenericCNN(net_config, dl_train, dl_test, device)
        self.optimizer = torch.optim.Adam(
            self.net.net.parameters(), lr=config["lr"])

    def step(self):
        acc_train = self.net.train(self.optimizer)
        acc_test = self.net.test()
        return {"acc_train": acc_train, "acc_test": acc_test}


if args.ray == "debug":
    ray.init(local_mode=True)
elif args.ray == "remote":
    ray.init(address="auto")

config = {
    "lr": 0.0008,
    "batch_size": 200,
    "d": [64],
    "c": [32, 64, 64],
    # "depth": tune.grid_search([1, 2]),
    "log_sys_usage": True
}

scheduler = ASHAScheduler(
    max_t=30,
    grace_period=10,
    reduction_factor=2
)

reporter = tune.CLIReporter(max_progress_rows=16, print_intermediate_tables=True, metric="acc_test", mode="max", sort_by_metric=True)
if utils_generic.isnotebook():
    reporter = tune.JupyterNotebookReporter(overwrite=True)

analysis = tune.run(
    CIFARTrainable,
    config=config,
    resources_per_trial={"cpu": 2, "gpu": 1},
    local_dir=args.data + "/ray",
    scheduler=scheduler,
    mode="max",
    metric="acc_test",
    num_samples=1,
    fail_fast=True,
    progress_reporter=reporter,
    verbose=2
)

best_experiment = analysis.get_best_trial()
print(best_experiment)
