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

print(f"starting a run on node with hostname {platform.node()}")

parser = argparse.ArgumentParser()
parser.add_argument("--data",
                    help="path to data dir",
                    default="/home/paul/dev/bachelor-thesis/code/data")
parser.add_argument("--small",
                    help="specifiy if a small dataset should be used",
                    action="store_true")
parser.add_argument("--log-dir", help="sub dir under --data for logging results", default="logs")
args = parser.parse_args()


def create_datasets():
    print(f"importing data from {args.data}")

    p_y = "/parameter_small.train.npy" if args.small else "/parameter.train"
    p_x = "/prepared_data_small.train.npy" if args.small else "/prepared_data.train"

    raw_y = np.load(args.data + p_y)
    raw_x = np.load(args.data + p_x)

    X_train, X_test, y_train, y_test = train_test_split(raw_x,
                                                        raw_y,
                                                        test_size=0.25,
                                                        random_state=42)

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
torch.manual_seed(hvd.rank())
device = torch.device(f"cuda:{hvd.local_rank()}" if torch.cuda.is_available() else "cpu")

print(f"running hvd on rank {hvd.local_rank()}")

(dataset_train, dataset_test) = create_datasets()

config = {
    "batch_size": 512,
    "lr": 2e-4,
    "epochs": 100,
    "cnn_channels": [50, 50],
    "linear_layers": [200]
}

net_config = cnn.CNNConfig(dim_in=dataset_train.tensors[0][1].size(),
                           dim_out=dataset_train.tensors[1][1].size(),
                           cnn_channels=config["cnn_channels"],
                           cnn_convolution_gen=lambda c: nn.Conv1d(c[0], c[1], kernel_size=5),
                           cnn_pool_gen=lambda: nn.MaxPool1d(kernel_size=3),
                           linear_layers=config["linear_layers"],
                           loss_function=nn.MSELoss(),
                           out_activation=None,
                           batch_accuracy_func=F.mse_loss)

sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                num_replicas=hvd.size(),
                                                                rank=hvd.rank(),
                                                                shuffle=True)
loader_train = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=config["batch_size"],
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=4,
                                           sampler=sampler_train)
loader_test = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=config["batch_size"],
                                          pin_memory=True,
                                          drop_last=True,
                                          num_workers=4)

net = cnn.GenericCNN(net_config, loader_train, loader_test, device)
if hvd.local_rank() == 0:
    print(net.summary(config["batch_size"]))

optimizer = torch.optim.Adam(net.net.parameters(), lr=config["lr"])
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.net.named_parameters())

hvd.broadcast_parameters(net.net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

points = []
start = time.time()
for epoch in range(config["epochs"]):
    start_epoch = time.time()
    train_acc = net.train(optimizer)
    if hvd.local_rank() == 0:
        test_acc = net.test()
        time_epoch = time.time() - start_epoch
        points.append({
            "epoch": epoch,
            "acc_train": train_acc,
            "acc_test": test_acc,
            "time_epoch": time_epoch,
            "time": time.time() - start
        })
        print(f"Epoch {epoch}, AccTrain={train_acc}, AccTest={test_acc}, Took={time_epoch}")


def horovod_meta():
    return {
        "size": hvd.size(),
        "mpi_enabled": hvd.mpi_enabled(),
        "gloo_enabled": hvd.gloo_enabled(),
        "nccl_built": hvd.nccl_built()
    }


if hvd.local_rank() == 0:
    import pandas as pd
    from pathlib import Path
    import json
    import uuid

    took = time.time() - start

    log_dir = Path(
        args.data) / args.log_dir / (time.strftime("%Y-%m-%d_%H-%M") + uuid.uuid4().hex[:8])
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
                    "horovod": horovod_meta()
                },
                "config": config
            }, outfile)
