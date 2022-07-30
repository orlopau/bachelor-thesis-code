import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import argparse
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.horovod import DistributedTrainableCreator
import ray
import utils.cnn as cnn
import utils.generic as utils_generic
import utils.ml as utils_ml
import horovod.torch as hvd
import os

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


max_epochs = 50


def trainable(config):
    hvd.init()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5070937, 0.48655552, 0.44092253),
                                         (0.26733243, 0.256427, 0.27613324)),

    ])

    ds_train = torchvision.datasets.CIFAR10(
        args.data, transform=transforms, download=False)
    ds_test = torchvision.datasets.CIFAR10(
        args.data, train=False, transform=transforms, download=False)

    _, dl_test = utils_ml.create_loaders(
        ds_train, ds_test, batch_size=config["batch_size"], workers=8)

    sampler_distributed_train = torch.utils.data.distributed.DistributedSampler(
        ds_train, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=config["batch_size"], sampler=sampler_distributed_train, num_workers=4, drop_last=True)

    print(f"created loader for hv rank {hvd.local_rank()}")

    net_config = cnn.CNNConfig(
        ds_train[0][0].shape,
        [len(ds_train.classes)],
        config["c"],
        nn.Conv2d,
        config["d"],
        torch.nn.CrossEntropyLoss(),
    )

    device = torch.device(f"cuda:{hvd.local_rank()}")

    net = cnn.GenericCNN(net_config, dl_train, dl_test, device)
    optimizer = torch.optim.Adam(net.net.parameters(), lr=config["lr"])
    optimizer = hvd.DistributedOptimizer(optimizer)
    hvd.broadcast_parameters(net.net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    print(f"Horovod world size: {hvd.size()}")
    print(f"{hvd.local_rank()} gpus: {os.environ['CUDA_VISIBLE_DEVICES']}")

    last_report = time.time()
    for epoch in range(max_epochs):
        acc_train = net.train(optimizer)
        if epoch % 5 == 0:
            acc_test = net.test()
            tune.report(acc_train=acc_train, acc_test=acc_test, took=time.time(
            )-last_report, epoch=epoch)
            last_report = time.time()


if args.ray == "debug":
    ray.init(local_mode=True)
elif args.ray == "remote":
    ray.init(address="auto")

num_workers = 4

config = {
    "lr": 0.002,
    "batch_size": 2000,
    "d": [tune.qrandint(32, 512, 32), tune.qrandint(32, 512, 32)],
    "c": [tune.qrandint(16, 256, 16), tune.qrandint(16, 256, 16), tune.qrandint(16, 256, 16)],
    "log_sys_usage": True
}

# each t is 
scheduler = ASHAScheduler(
    time_attr="epoch",
    max_t=max_epochs,
    grace_period=10,
    reduction_factor=2,
)

reporter = tune.CLIReporter(
    max_progress_rows=16, print_intermediate_tables=True, sort_by_metric=True, infer_limit=20)
if utils_generic.isnotebook():
    reporter = tune.JupyterNotebookReporter(overwrite=True)

horovod_trainable = DistributedTrainableCreator(
    trainable,
    use_gpu=True,
    num_hosts=4,
    num_workers=4,
    num_cpus_per_worker=4,
)

analysis = tune.run(
    horovod_trainable,
    config=config,
    local_dir=args.data + "/ray",
    scheduler=scheduler,
    mode="max",
    metric="acc_test",
    num_samples=50,
    fail_fast=True,
    progress_reporter=reporter,
    verbose=3,

)

best_experiment = analysis.get_best_trial()
print(best_experiment)
