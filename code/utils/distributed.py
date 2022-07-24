from torchinfo import summary
import torch
from torch.utils.data import DataLoader
import horovod.torch as hvd
import wandb
import os
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


def is_logger():
    return not hvd.is_initialized() or hvd.rank() == 0


def get_device_hvd(single_gpu=False):
    if not hvd.is_initialized():
        hvd.init()

    device_string = "cuda:0" if single_gpu else f"cuda:{hvd.local_rank()}"
    return torch.device(device_string if torch.cuda.is_available() else "cpu")


def create_loaders(dataset_train,
                   dataset_test,
                   batch_size_train,
                   batch_size_test,
                   workers,
                   pin_memory=True,
                   sampler_train=None,
                   sampler_test=None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    loader_train = DataLoader(dataset_train,
                              batch_size=batch_size_train,
                              pin_memory=pin_memory,
                              drop_last=True,
                              num_workers=workers,
                              sampler=sampler_train)

    loader_test = DataLoader(dataset_test,
                             batch_size=batch_size_test,
                             pin_memory=pin_memory,
                             drop_last=False,
                             num_workers=workers,
                             sampler=sampler_test)

    loader_train.sampler

    print(f"created loaders with batch sizes: {batch_size_train} train; {batch_size_test} test")
    return (loader_train, loader_test)


def create_loaders_hvd(dataset_train,
                       dataset_test,
                       batch_size_train,
                       batch_size_test,
                       workers,
                       pin_memory=True) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                    num_replicas=hvd.size(),
                                                                    rank=hvd.rank(),
                                                                    shuffle=True,
                                                                    drop_last=True)

    sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test,
                                                                   num_replicas=hvd.size(),
                                                                   rank=hvd.rank(),
                                                                   shuffle=False)

    return create_loaders(dataset_train, dataset_test, batch_size_train, batch_size_test, workers,
                          pin_memory, sampler_train, sampler_test)


class Runnable(ABC):

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, loader_train, loader_test,
                 device) -> None:
        self.net = net.to(device)
        self.optimizer = optimizer
        self.device = device
        self.test_hooks = []
        self.loader_train, self.loader_test = loader_train, loader_test

    @abstractmethod
    def train_batch(self, X, Y):
        """Performs training for a mini-batch."""
        pass

    @abstractmethod
    def train(self):
        """Performs a training epoch. Should return a dict of results."""
        return None

    @abstractmethod
    def test(self, mean_reduce=lambda x: x):
        """Performs testing of the network. Should return a dict of results."""
        pass


class Runner():

    def __init__(self) -> None:
        self.epoch_hooks = []
        print(f"initializing runner on node with hostname {platform.node()}")

    def set_runnable(self, runnable: Runnable):
        self.runnable = runnable

    def start_training(self, epochs, mean_reduce=lambda x: x):
        r = self.runnable

        self.start = time.time()

        for epoch in range(epochs):
            if is_logger(): print(f"Epoch {epoch}:")
            
            if r.loader_train.sampler is not None and hasattr(r.loader_train.sampler, 'set_epoch'):
                r.loader_train.sampler.set_epoch(epoch)

            time_train_start = time.time()
            data_train = r.train()
            time_train = time.time() - time_train_start

            time_test_start = time.time()
            data_test = r.test(mean_reduce)
            time_test = time.time() - time_test_start

            point = {
                "epoch": epoch,
                "time_train": time_train,
                "time_test": time_test,
                "time_epoch": time.time() - time_train_start,
                "time": time.time() - self.start,
                **data_train,
                **data_test,
            }

            if is_logger():
                print(point)
                print(f"Took: {point['time_epoch']:2f}s")
                print("============================")

            for hook in self.epoch_hooks:
                hook(point)


class DistributedHvdRunner(Runner):

    def __init__(self,
                 group,
                 project="test",
                 name=None,
                 data_dir="/home/paul/dev/bachelor-thesis/code/data",
                 log_dir="logs") -> None:

        super().__init__(device, data_dir, log_dir)
        self.name, self.group, self.project = name, group, project
        self.data_dir, self.log_dir = data_dir, log_dir
        self.points = []
        self.epoch_hooks = []

        print("init hvd and seeding...")
        hvd.init()
        np.random.seed(hvd.rank())
        torch.manual_seed(hvd.rank())
        random.seed(hvd.rank())

        print(f"initializing runner on node with hostname {platform.node()}, rank {hvd.rank()}")
        self.device = get_device_hvd()
        print(f"running hvd on rank {hvd.rank()}, device {self.device}")

        if hvd.rank() == 0:
            wandb.init(project=self.project,
                       config={
                           "horovod": horovod_meta(),
                           "slurm": slurm_meta(),
                       },
                       group=self.group,
                       name=f"N:{int(hvd.size() / hvd.local_size())} G:{hvd.size()}"
                       if self.name is None else self.name,
                       dir="/lustre/ssd/ws/s8979104-horovod/data/wandb",
                       settings=wandb.Settings(_stats_sample_rate_seconds=0.5, _stats_samples_to_average=2))

    def set_runnable(self, runnable: Runnable):
        self.runnable = runnable
        wandb.config["run"] = self.runnable.meta

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
            if is_logger():
                print(f"Epoch {epoch}:")
            self.loaders[0].sampler.set_epoch(epoch)

            time_train_start = time.time()
            data_train = r.train(self.loaders[0], optimizer)
            time_train = time.time() - time_train_start

            time_test_start = time.time()
            data_test = r.test(self.loaders[1], hvd.allreduce)
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
                for hook in self.epoch_hooks:
                    hook(point)

                print(f"Took: {point['time_epoch']:2f}s")
                print("============================")

        self.end_training()

    def end_training(self):
        if hvd.rank() == 0:
            import pandas as pd
            from pathlib import Path
            import json
            import uuid

            log_dir = Path(self.data_dir) / self.log_dir / (time.strftime("%Y-%m-%d_%H-%M") + "_" +
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