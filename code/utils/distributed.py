import torch
import torch.nn as nn
import torch.nn.functional as F
import horovod.torch as hvd
import wandb
import os
import argparse
import random
import numpy as np
import platform
import utils.cnn as cnn
import time


def horovod_meta():
    return {
        "size": hvd.size(),
        "mpi_enabled": hvd.mpi_enabled(),
        "gloo_enabled": hvd.gloo_enabled(),
        "nccl_built": hvd.nccl_built()
    }


def slurm_meta():
    return {x[0]: x[1] for x in os.environ.items() if x[0].startswith("SLURM")}


class DistributedNet():

    def __init__(self, config, project="test") -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data",
                            help="path to data dir",
                            default="/home/paul/dev/bachelor-thesis/code/data")
        parser.add_argument("--small",
                            help="specifiy if a small dataset should be used",
                            action="store_true")
        parser.add_argument("--log-dir",
                            help="sub dir under --data for logging results",
                            default="logs")
        parser.add_argument("--no-test",
                            help="specify if no test acc should be calculated after each epoch",
                            action="store_true")
        parser.add_argument(
            "--single-gpu",
            help="specify if only a single gpu but multiple horovod processes should be used",
            action="store_true")
        parser.add_argument("--group", help="group", required=True)
        parser.add_argument("--name", help="name")

        self.args = parser.parse_args()
        self.config = config
        self.project = project
        self.points = []

    def init(self, datasets, net_config):
        hvd.init()
        np.random.seed(hvd.rank())
        torch.manual_seed(hvd.rank())
        random.seed(hvd.rank())

        print(f"starting a run on node with hostname {platform.node()}, rank {hvd.rank()}")

        device_string = "cuda:0" if self.args.single_gpu else f"cuda:{hvd.local_rank()}"
        self.device = torch.device(device_string if torch.cuda.is_available() else "cpu")
        print(f"running hvd on rank {hvd.rank()}, device {self.device}")

        (self.ds_train, self.ds_test) = datasets
        self.loaders = self.create_loaders(*datasets)

        cnn_net = cnn.GenericCNN(net_config, *self.loaders, self.device)
        self.cnn_net = cnn_net

        optimizer = torch.optim.Adam(cnn_net.net.parameters(), lr=self.config["lr"])
        self.optimizer = hvd.DistributedOptimizer(optimizer,
                                                  named_parameters=cnn_net.net.named_parameters(),
                                                  op=hvd.Sum)

        hvd.broadcast_parameters(cnn_net.net.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        if hvd.rank() == 0:
            wandb.init(project=self.project,
                       config={
                           "config": self.config,
                           "horovod": horovod_meta(),
                           "slurm": slurm_meta()
                       },
                       group=self.args.group,
                       name=f"N:{int(hvd.size() / hvd.local_size())} G:{hvd.size()}"
                       if self.args.name is None else self.args.name,
                       dir="/tmp")

            wandb.watch(cnn_net.net, log_freq=100, log="all")
            print(cnn_net.summary(self.config["batch_size"]))

    def create_test_points(self):
        return {}

    def start_training(self):
        self.start = time.time()

        for epoch in range(self.config["epochs"]):
            self.loaders[0].sampler.set_epoch(epoch)

            time_start = time.time()
            (train_acc, time_train_step) = self.cnn_net.train(self.optimizer)
            time_train = time.time()

            test_point = self.create_test_points()
            if hvd.rank() == 0:
                point = {
                    "epoch": epoch,
                    "acc_train": train_acc,
                    **test_point,
                    "time_epoch": time.time() - time_start,
                    "time_train": time_train - time_start,
                    "time_train_step": time_train_step,
                    "time": time.time() - self.start,
                }
                wandb.log(point)
                self.points.append(point)

                print(f"""Epoch {epoch}, AccTrain={train_acc}, TestAcc={point['test_acc']} Took={point['time_epoch']}""")

        self.end_training()


    def end_training(self):
        if hvd.rank() == 0:
            import pandas as pd
            from pathlib import Path
            import json
            import uuid

            took = time.time() - self.start

            log_dir = Path(self.args.data) / self.args.log_dir / (time.strftime("%Y-%m-%d_%H-%M") +
                                                                  "_" + uuid.uuid4().hex[:8])
            log_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(self.points)
            df.to_csv((log_dir / "data.csv").resolve(), index=False)
            with open((log_dir / "meta.json").resolve(), 'w') as outfile:
                json.dump(
                    {
                        "run": {
                            "duration": took,
                            "time_epoch": time.time(),
                            "train_size": len(self.ds_train),
                            "test_size": len(self.ds_test),
                            "config": self.config,
                        },
                        "horovod": horovod_meta(),
                        "slurm": slurm_meta()
                    }, outfile)
            wandb.save(str((log_dir / "*").resolve()), policy="end")


    def create_loaders(
            self, dataset_train,
            dataset_test) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
                                                   batch_size=self.config["batch_size"],
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   num_workers=self.config["workers"],
                                                   sampler=sampler_train)

        loader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=self.config["batch_size"],
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  num_workers=self.config["workers"],
                                                  sampler=sampler_test)

        return (loader_train, loader_test)