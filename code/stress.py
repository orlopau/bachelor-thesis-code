#!/usr/bin/env python

from contextlib import ExitStack
import platform
import random
import wandb
import utils.distributed as distributed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import resnet
from utils import cnn
from utils import args

if args.get_args().dist:
    import horovod.torch as hvd

if args.get_args().tprof:
    from torch.profiler import profile, record_function, ProfilerActivity, schedule


def create_datasets(path):
    print(f"importing data from {path}")

    with np.load(path + "/stress_norm_y.npz", allow_pickle=True) as data:
        X_train, X_test, Y_train, Y_test, meta = data["X_train"], data["X_test"], data["Y_train"], data[
            "Y_test"], data["meta"].item()

        dataset_train = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(Y_train).float())
        dataset_test = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(Y_test).float())
        print(f"created datasets; train={len(dataset_train)}, test={len(dataset_test)}")

        return (dataset_train, dataset_test, meta["y_max"], meta["y_min"])


class StressRunnable(distributed.Runnable):

    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loader_train,
                 loader_test,
                 device,
                 y_max=0,
                 y_min=0,
                 scheduler=None) -> None:
        super().__init__(net, optimizer, loader_train, loader_test, device)
        self.loss = nn.MSELoss()
        self.y_max = y_max
        self.y_min = y_min
        self.scheduler = scheduler

    def train_batch(self, X, Y):
        y = self.net(X)

        loss = self.loss(y, Y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.float()

    def train(self):
        with ExitStack() as stack:
            if args.get_args().tprof:
                stack.enter_context(record_function("model_train"))
            if args.get_args().dlprof:
                stack.enter_context(torch.autograd.profiler.emit_nvtx())

            accuracy = 0
            for X, Y in self.loader_train:
                accuracy += self.train_batch(X.to(self.device), Y.to(self.device))
                if args.get_args().tprof:
                    prof.step()

            mean_acc = float(accuracy / len(self.loader_train))

            res = {"acc_train": mean_acc}

            if self.scheduler is not None:
                self.scheduler.step()
                res["sched_lr"] = self.optimizer.param_groups[0]['lr']

            if distributed.is_logger():
                print(f"TRAIN: Acc={mean_acc:.6f}")

            return res

    def test(self, mean_reduce=lambda x: x):
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if args.get_args().tprof:
                stack.enter_context(record_function("model_test"))

            self.net.eval()

            predictions = torch.Tensor().to(self.device)
            targets = torch.Tensor().to(self.device)

            for (X, Y) in self.loader_test:
                y = self.net(X.to(self.device))
                predictions = torch.cat((predictions, y))
                targets = torch.cat((targets, Y.to(self.device)))

            mse = mean_reduce(F.mse_loss(predictions, targets))
            l1loss = mean_reduce(torch.mean(torch.abs(targets - predictions), dim=0))
            l1loss_scaled = l1loss.cpu() * (self.y_max - self.y_min)

            test_point = {
                "acc_test": float(mse),
                "acc_test_g0": float(l1loss_scaled[0]),
                "acc_test_a": float(l1loss_scaled[1]),
            }

            self.net.train()

            if distributed.is_logger():
                print(
                    f"TEST: Acc={test_point['acc_test']:.6f}, AccG0={test_point['acc_test_g0']:.3f}, AccA={test_point['acc_test_a']:.3f}"
                )
            return test_point


def create_model(config):
    if "res" in config["model"]:
        res_params = (2, config["pre_kernel"], config["pre_stride"], config["pre_padding"], "1d")

    if config["model"] == "res18":
        model = resnet.create_resnet18(*res_params)
    elif config["model"] == "res34":
        model = resnet.create_resnet34(*res_params)
    elif config["model"] == "cnn":
        net_config = cnn.CNNConfig(dim_in=(3, 201),
                                   dim_out=(2,),
                                   cnn_channels=config["cnn_channels"],
                                   cnn_convolution_gen=lambda c: nn.Conv1d(c[0], c[1], kernel_size=5),
                                   cnn_pool_gen=lambda: nn.MaxPool1d(kernel_size=3),
                                   cnn_pool_each=config["pool_each"],
                                   linear_layers=config["linear_layers"],
                                   loss_function=nn.MSELoss(),
                                   out_activation=None,
                                   batch_accuracy_func=F.mse_loss,
                                   cnn_activation=nn.ReLU,
                                   linear_activation=nn.ReLU,
                                   dropout=config["dropout"])
        model = cnn.create_cnn(net_config)
    else:
        raise Exception(f"no model found for {config['model']}")

    optimizer = {
        "sgd": torch.optim.SGD(model.parameters(), config["lr"], momentum=0.9),
        "adam": torch.optim.Adam(model.parameters(), config["lr"])
    }[config["optimizer"]]

    scheduler = None
    if config["sched"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["step_size"], config["gamma"])
    elif config["sched"] == "Cyclic":
        # step size should be 2-10 times the batches (steps) per epoch
        # TODO fixed step size!!!!!!!
        batches_per_epoch = 750000 // config["batch_size"]
        step_size = batches_per_epoch * 4
        print(f"step size: {step_size} @ batches per epoch {batches_per_epoch}")
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=config["lr"],
                                                      max_lr=config["lr_max"],
                                                      step_size_up=step_size)

    return (model, optimizer, scheduler)


conf_res = {
    "pre_kernel": 3,
    "pre_stride": 2,
    "pre_padding": 1,
    "batch_size": 128,
    "lr": 6e-5,
}

conf_cnn = {
    "cnn_channels": [15, 38],
    "linear_layers": [197],
    "pool_each": 1,
    "dropout": None,
}

config = {
    "epochs": 80,
    "lr": 4e-3,
    "batch_size": 512,
    "optimizer": "adam",
    "model": "cnn",
    "workers": 2,
    "sched": "StepLR",
    "step_size": 10,
    "gamma": 0.67,
    "pin_memory": False,
    "dataset_mem": "ram"
}

config_baseline = {
    "epochs": 80,
    "lr": 4.7331e-04,
    "batch_size": 75,
    "optimizer": "adam",
    "model": "cnn",
    "workers": 2,
    "sched": None,
    "pin_memory": False,
    "dataset_mem": "ram",
    "hvd_mode": "sum"
}

config = config_baseline

if "res" in config["model"]:
    config = {**config, **conf_res}
else:
    config = {**config, **conf_cnn}

if __name__ == "__main__":
    a = args.get_args()

    if a.lrf:
        from torch_lr_finder import LRFinder
        config["sched_cyclic"] = False
        runnable = create_runnable(config, distributed.get_device_hvd(), a.data)
        runner = distributed.Runner(torch.device("cuda"))
        # runner = create_runner(config, runnable, a.group, a.name, a.data)

        lr_finder = LRFinder(runnable.net, runnable.optimizer, nn.MSELoss(), device="cuda:0")
        lr_finder.range_test(runner.loaders[0], start_lr=1e-9, end_lr=100, smooth_f=0)
        lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state
    elif a.dist:
        print("running horovod version")
        hvd.init()
        np.random.seed(hvd.rank())
        torch.manual_seed(hvd.rank())
        random.seed(hvd.rank())

        print(f"initializing runner on node with hostname {platform.node()}, rank {hvd.rank()}")
        device = distributed.get_device_hvd(a.single_gpu)
        print(f"running hvd on rank {hvd.rank()}, device {device}")

        if hvd.rank() == 0:
            wandb.init(
                project="hvd",
                config={
                    "horovod": distributed.horovod_meta(),
                    "slurm": distributed.slurm_meta(),
                    "run": config
                },
                group=a.group,
                name=f"N:{int(hvd.size() / hvd.local_size())} G:{hvd.size()}" if a.name is None else a.name,
                dir="/lustre/ssd/ws/s8979104-horovod/data/wandb",
                settings=wandb.Settings(_stats_sample_rate_seconds=0.5, _stats_samples_to_average=2))

        (model, optimizer, scheduler) = create_model(config)
        model.to(device)
        (dataset_train, dataset_test, y_max, y_min) = create_datasets(a.data)
        (loader_train, loader_test) = distributed.create_loaders_hvd(dataset_train, dataset_test,
                                                                     config["batch_size"],
                                                                     config["batch_size"], config["workers"],
                                                                     config["pin_memory"])

        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             op=hvd.Sum)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

        runnable = StressRunnable(model, optimizer, loader_train, loader_test, device, y_max, y_min,
                                  scheduler)

        if hvd.rank() == 0:
            wandb.watch(model, log_freq=100, log="all")

        runner = distributed.Runner()
        runner.runnable = runnable

        def hook(p):
            if hvd.rank() == 0: 
                wandb.log(p)
        
        runner.epoch_hooks.append(hook)
        runner.start_training(config["epochs"], hvd.allreduce)
    else:
        with ExitStack() as stack:
            if args.get_args().dlprof:
                import nvidia_dlprof_pytorch_nvtx as nvtx
                nvtx.init()

            (model, optimizer, scheduler) = create_model(config)
            (dataset_train, dataset_test, y_max, y_min) = create_datasets(a.data)
            (loader_train, loader_test) = distributed.create_loaders(dataset_train, dataset_test,
                                                                     config["batch_size"],
                                                                     config["batch_size"], config["workers"],
                                                                     config["pin_memory"])
            runnable = StressRunnable(model, optimizer, loader_train, loader_test, torch.device("cuda"),
                                      y_max, y_min, scheduler)

            runner = distributed.Runner()
            runner.runnable = runnable

            if args.get_args().tprof:
                s = schedule(wait=0, warmup=10, active=10)

                def trace_handler(p):
                    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=30)
                    print(output)

                prof = stack.enter_context(
                    profile(activities=[ProfilerActivity.CPU],
                            record_shapes=True,
                            schedule=s,
                            on_trace_ready=trace_handler))

            runner.start_training(config["epochs"])
