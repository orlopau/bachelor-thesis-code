import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
import argparse
import platform
import utils.cnn as cnn
import time
import horovod.torch as hvd
import os
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--data",
                    help="path to data dir",
                    default="/home/paul/dev/bachelor-thesis/code/data")
parser.add_argument("--small",
                    help="specifiy if a small dataset should be used",
                    action="store_true")
parser.add_argument("--log-dir", help="sub dir under --data for logging results", default="logs")
parser.add_argument("--no-test",
                    help="specify if no test acc should be calculated after each epoch",
                    action="store_true")
parser.add_argument(
    "--single-gpu",
    help="specify if only a single gpu but multiple horovod processes should be used",
    action="store_true")
parser.add_argument("--group", help="group", required=True)
parser.add_argument("--name", help="name")
args = parser.parse_args()


def horovod_meta():
    return {
        "size": hvd.size(),
        "mpi_enabled": hvd.mpi_enabled(),
        "gloo_enabled": hvd.gloo_enabled(),
        "nccl_built": hvd.nccl_built()
    }


def slurm_meta():
    return {x[0]: x[1] for x in os.environ.items() if x[0].startswith("SLURM")}


def create_datasets():
    print(f"importing data from {args.data}")

    p_y = "/parameter_small.train.npy" if args.small else "/parameter.train"
    p_x = "/prepared_data_small.train.npy" if args.small else "/prepared_data.train"

    raw_y = np.load(args.data + p_y)
    raw_x = np.load(args.data + p_x)

    X_train, X_test, y_train, y_test = train_test_split(raw_x,
                                                        raw_y,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        shuffle=True)

    def normalize(data, max, min):
        max = data.max(axis=0)
        min = data.min(axis=0)

        return np.divide(np.subtract(data, min), max - min)

    max = y_train.max(axis=0)
    min = y_train.min(axis=0)

    y_train = normalize(y_train, max, min)
    y_test = normalize(y_test, max, min)

    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float())
    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float())
    print(f"created datasets; train={len(dataset_train)}, test={len(dataset_test)}")

    return (dataset_train, dataset_test)


# create datasets, loaders and start training
hvd.init()
np.random.seed(hvd.rank())
torch.manual_seed(hvd.rank())
random.seed(hvd.rank())

config = {
    "batch_size": 512,
    "lr": 1e-3,
    "epochs": 200,
    "cnn_channels": [64, 128, 128, 64],
    "pool_each": 2,
    "linear_layers": [512, 256],
    "workers": 4
}

if hvd.rank() == 0:
    wandb.init(project="stress_cnn_horovod",
               config={
                   "config": config,
                   "horovod": horovod_meta(),
                   "slurm": slurm_meta()
               },
               group=args.group,
               name=f"N:{int(hvd.size() / hvd.local_size())} G:{hvd.size()}"
               if args.name is None else args.name,
               dir="/tmp")

print(f"starting a run on node with hostname {platform.node()}, rank {hvd.rank()}")

device_string = "cuda:0" if args.single_gpu else f"cuda:{hvd.local_rank()}"
device = torch.device(device_string if torch.cuda.is_available() else "cpu")

print(f"running hvd on rank {hvd.rank()}, device {device}")

(dataset_train, dataset_test) = create_datasets()

net_config = cnn.CNNConfig(
    dim_in=dataset_train.tensors[0][1].size(),
    dim_out=dataset_train.tensors[1][1].size(),
    cnn_channels=config["cnn_channels"],
    cnn_convolution_gen=lambda c: nn.Conv1d(c[0], c[1], kernel_size=5, padding=2),
    cnn_pool_gen=lambda: nn.MaxPool1d(kernel_size=2),
    cnn_pool_each=config["pool_each"],
    linear_layers=config["linear_layers"],
    loss_function=nn.MSELoss(),
    out_activation=None,
    batch_accuracy_func=F.mse_loss,
    cnn_activation=nn.ReLU,
    linear_activation=nn.ReLU)

sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                num_replicas=hvd.size(),
                                                                rank=hvd.rank(),
                                                                shuffle=True,
                                                                drop_last=True)

sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test,
                                                               num_replicas=hvd.size(),
                                                               rank=hvd.rank(),
                                                               shuffle=False)

loader_train = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=config["batch_size"],
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=config["workers"],
                                           sampler=sampler_train)

loader_test = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=config["batch_size"],
                                          pin_memory=True,
                                          drop_last=False,
                                          num_workers=config["workers"],
                                          sampler=sampler_test)

##### NN Code #####
net = cnn.GenericCNN(net_config, loader_train, loader_test, device)
if hvd.rank() == 0:
    print(net.summary(config["batch_size"]))
    wandb.watch(net.net, log_freq=100, log="all")

optimizer = torch.optim.SGD(net.net.parameters(), lr=config["lr"])
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.net.named_parameters(), op=hvd.Sum)

hvd.broadcast_parameters(net.net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

points = []
start = time.time()
for epoch in range(config["epochs"]):
    sampler_train.set_epoch(epoch)

    time_start = time.time()
    (train_acc, time_train_step) = net.train(optimizer)
    time_train = time.time()

    test_acc_dist = net.test()
    time_test = time.time()

    test_accs = hvd.allgather(torch.Tensor([test_acc_dist]), name="gather_test_acc")
    test_acc_dist = float(test_accs.mean())
    time_test_reduce = time.time()

    time_end = time_test_reduce
    if hvd.rank() == 0:
        point = {
            "epoch": epoch,
            "acc_train": train_acc,
            "acc_test": test_acc_dist,
            "time_epoch": time_end - time_start,
            "time_train": time_train - time_start,
            "time_train_step": time_train_step,
            "time_test": time_test - time_train,
            "time_test_allreduce": time_test_reduce - time_test,
            "time": time.time() - start
        }
        wandb.log(point)
        points.append(point)
        print(f"""Epoch {epoch}, AccTrain={train_acc}, AccTest={test_acc_dist}, 
            Took={point['time_epoch']}""")

if hvd.rank() == 0:
    import pandas as pd
    from pathlib import Path
    import json
    import uuid

    took = time.time() - start

    log_dir = Path(
        args.data) / args.log_dir / (time.strftime("%Y-%m-%d_%H-%M") + "_" + uuid.uuid4().hex[:8])
    log_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(points)
    df.to_csv((log_dir / "data.csv").resolve(), index=False)
    with open((log_dir / "meta.json").resolve(), 'w') as outfile:
        json.dump(
            {
                "run": {
                    "duration": took,
                    "time_epoch": time.time(),
                    "train_size": len(dataset_train),
                    "test_size": len(dataset_test),
                    "config": config,
                },
                "horovod": horovod_meta(),
                "slurm": slurm_meta()
            }, outfile)

    wandb.save(str((log_dir / "*").resolve()), policy="end")