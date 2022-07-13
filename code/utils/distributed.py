from torchinfo import summary
import torch
import horovod.torch as hvd
import wandb
import os
import argparse
import random
import numpy as np
import platform
import time
from abc import ABC, abstractmethod


def horovod_meta():
    return {
        "size": hvd.size(),
        "mpi_enabled": hvd.mpi_enabled(),
        "gloo_enabled": hvd.gloo_enabled(),
        "nccl_built": hvd.nccl_built()
    }


def slurm_meta():
    return {x[0]: x[1] for x in os.environ.items() if x[0].startswith("SLURM")}


_args = None


def get_args():
    if _args is not None:
        return _args

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        help="path to data dir",
                        default="/home/paul/dev/bachelor-thesis/code/data")
    parser.add_argument("--small", help="specifiy if a small dataset should be used", action="store_true")
    parser.add_argument("--log-dir", help="sub dir under --data for logging results", default="logs")
    parser.add_argument("--no-test",
                        help="specify if no test acc should be calculated after each epoch",
                        action="store_true")
    parser.add_argument("--single-gpu",
                        help="specify if only a single gpu but multiple horovod processes should be used",
                        action="store_true")
    parser.add_argument("--group", help="group", required=True)
    parser.add_argument("--name", help="name")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    return args


class DistributedRunnable(ABC):

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, datasets, meta) -> None:
        self.net = net
        self.optimizer = optimizer
        self.datasets = datasets
        self.meta = meta

    @abstractmethod
    def train_epoch(self, loader, optimizer, device):
        """Performs a training epoch. Should return a dict of results."""
        pass

    @abstractmethod
    def test(self, loader, mean_reduce, device):
        """Performs testing of the network. Should return a dict of results."""
        pass


class DistributedRunner():

    def __init__(self, runnable: DistributedRunnable, project="test", batch_size=128, workers=4) -> None:
        self.project = project
        self.points = []
        self.runnable = runnable
        self.args = get_args()

        self.init()

        self.loaders = self.create_loaders(*runnable.datasets, batch_size, workers)

    def init(self):
        hvd.init()
        np.random.seed(hvd.rank())
        torch.manual_seed(hvd.rank())
        random.seed(hvd.rank())

        print(f"initializing a run on node with hostname {platform.node()}, rank {hvd.rank()}")

        device_string = "cuda:0" if self.args.single_gpu else f"cuda:{hvd.local_rank()}"
        self.device = torch.device(device_string if torch.cuda.is_available() else "cpu")
        print(f"running hvd on rank {hvd.rank()}, device {self.device}")

        self.runnable.net.to(self.device)

        if hvd.rank() == 0:
            wandb.init(project=self.project,
                       config={
                           "horovod": horovod_meta(),
                           "slurm": slurm_meta(),
                           "run": self.runnable.meta
                       },
                       group=self.args.group,
                       name=f"N:{int(hvd.size() / hvd.local_size())} G:{hvd.size()}"
                       if self.args.name is None else self.args.name,
                       dir="/lustre/ssd/ws/s8979104-horovod/data/wandb",
                       settings=wandb.Settings(_stats_sample_rate_seconds=0.5, _stats_samples_to_average=2))

    def is_logger(self):
        return hvd.rank() == 0

    def start_training(self, epochs):
        r = self.runnable

        if hvd.rank() == 0:
            wandb.watch(r.net, log_freq=100, log="all")

        optimizer = hvd.DistributedOptimizer(r.optimizer,
                                             named_parameters=r.net.named_parameters(),
                                             op=hvd.Sum)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(r.net.state_dict(), root_rank=0)

        self.start = time.time()

        for epoch in range(epochs):
            self.loaders[0].sampler.set_epoch(epoch)

            time_train_start = time.time()
            data_train = r.train_epoch(self.loaders[0], optimizer, self.device)
            time_train = time.time() - time_train_start

            time_test_start = time.time()
            data_test = r.test(self.loaders[1], hvd.allreduce, self.device)
            time_test = time.time() - time_test_start

            if hvd.rank() == 0:
                point = {
                    "epoch": epoch,
                    "time_train": time_train,
                    "time_test": time_test,
                    "time_epoch": time.time() - time_train_start,
                    "time": time.time() - self.start,
                    **data_train,
                    **data_test,
                }
                wandb.log(point)
                self.points.append(point)

                print(
                    f"""Epoch {epoch}, AccTrain={point['acc_train']:.4f}, AccTest={point['acc_test']:.4f}, Took={point['time_epoch']:.2f}"""
                )

        self.end_training()

    def end_training(self):
        if hvd.rank() == 0:
            import pandas as pd
            from pathlib import Path
            import json
            import uuid

            took = time.time() - self.start

            log_dir = Path(self.args.data) / self.args.log_dir / (time.strftime("%Y-%m-%d_%H-%M") + "_" +
                                                                  uuid.uuid4().hex[:8])
            log_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(self.points)
            df.to_csv((log_dir / "data.csv").resolve(), index=False)
            with open((log_dir / "meta.json").resolve(), 'w') as outfile:
                json.dump({
                    "horovod": horovod_meta(),
                    "slurm": slurm_meta(),
                    "run": self.runnable.meta
                }, outfile)
            wandb.save(str((log_dir / "*").resolve()), policy="end")

    def create_loaders(self, dataset_train, dataset_test, batch_size,
                       workers) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

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
                                                   batch_size=batch_size,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   num_workers=workers,
                                                   sampler=sampler_train)

        loader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  num_workers=workers,
                                                  sampler=sampler_test)

        return (loader_train, loader_test)