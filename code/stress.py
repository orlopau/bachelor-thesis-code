from contextlib import ExitStack
from sched import scheduler
import sched
import utils.distributed as distributed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import resnet
from utils import cnn
from torchinfo import summary
from utils import args


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


class StressRunnable(distributed.DistributedRunnable):

    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 datasets,
                 meta,
                 device,
                 y_max=0,
                 y_min=0,
                 scheduler=None) -> None:
        super().__init__(net, optimizer, datasets, meta, device)
        self.loss = nn.MSELoss()
        self.y_max = y_max
        self.y_min = y_min
        self.scheduler = scheduler

    def train_batch(self, X, Y, optimizer):
        y = self.net(X)

        loss = self.loss(y, Y)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.float()

    def train_epoch(self, loader, optimizer):
        accuracy = 0
        for X, Y in loader:
            accuracy += self.train_batch(X.to(self.device), Y.to(self.device), optimizer)

        mean_acc = float(accuracy / len(loader))

        if distributed.is_logger():
            print(f"TRAIN: Acc={mean_acc:.6f}")
        return {"acc_train": mean_acc}

    def test(self, loader, mean_reduce):
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())

            self.net.eval()

            predictions = torch.Tensor().to(self.device)
            targets = torch.Tensor().to(self.device)

            for (X, Y) in loader:
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
        return resnet.create_resnet18(*res_params)
    elif config["model"] == "res34":
        return resnet.create_resnet34(*res_params)
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
        return cnn.create_cnn(net_config)
    else:
        raise Exception(f"no model found for {config['model']}")


def create_runnable(config, device, data_dir):
    model = create_model(config)
    (dataset_train, dataset_test, max, min) = create_datasets(data_dir)

    optimizer = {
        "sgd": torch.optim.SGD(model.parameters(), config["lr"], momentum=0.9),
        "adam": torch.optim.Adam(model.parameters(), config["lr"])
    }[config["optimizer"]]

    scheduler = None
    if config["sched_cyclic"]:
        # step size should be 2-10 times the batches (steps) per epoch
        batches_per_epoch = len(dataset_train) // config["batch_size"]
        step_size = batches_per_epoch * 4
        print(f"step size: {step_size} @ batches per epoch {batches_per_epoch}")
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["lr"], max_lr=config["lr_max"], step_size_up=step_size)

    runnable = StressRunnable(model,
                              optimizer, (dataset_train, dataset_test),
                              config,
                              device,
                              max,
                              min,
                              scheduler=scheduler)

    # if distributed.is_logger():
    #     summary(model, (config["batch_size"], 3, 201))

    return runnable


def create_runner(config, runnable, group, name, data_dir):
    runner = distributed.DistributedHvdRunner(group, project="resnet_stress", name=name, data_dir=data_dir)
    runner.set_runnable(runnable, batch_size_train=config["batch_size"], batch_size_test=50000, workers=config["workers"])
    return runner


# conf = {
#     "pre_kernel": 3,
#     "pre_stride": 2,
#     "pre_padding": 1,
#     "cnn_channels": [15, 38],
#     "linear_layers": [197],
#     "pool_each": 1,
#     "dropout": None,
#     "batch_size": 128,
#     "lr": 6e-5,
#     "epochs": 100,
#     "workers": 8,
#     "model": "cnn",
#     "optimizer": "adam"
# }

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
    "batch_size": 75,
    "lr": 4.733e-4,
    "lr_max": 4.7e-3,
}

config = {"epochs": 100, "workers": 8, "model": "cnn", "optimizer": "sgd", "sched_cyclic": True}

if "res" in config["model"]:
    config = {**config, **conf_res}
else:
    config = {**config, **conf_cnn}

if __name__ == "__main__":
    a = args.get_args()

    if a.lrf:
        from torch_lr_finder import LRFinder
        config["sched_cyclic"] = False
        runnable = create_runnable(config, distributed.get_device(), a.data)
        runner = create_runner(config, runnable, a.group, a.name, a.data)

        lr_finder = LRFinder(runnable.net, runnable.optimizer, nn.MSELoss(), device="cuda:0")
        lr_finder.range_test(runner.loaders[0], start_lr=1e-9, end_lr=100, smooth_f=0)
        lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state
    else:
        runnable = create_runnable(config, distributed.get_device(), a.data)
        runner = create_runner(config, runnable, a.group, a.name, a.data)
        runner.start_training(config["epochs"])
