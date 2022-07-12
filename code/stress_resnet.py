from sklearn.model_selection import train_test_split
import utils.distributed as distributed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import resnet
from torchinfo import summary


def create_datasets(args):
    print(f"importing data from {args.data}")

    with np.load(args.data + "/stress_normalized.npz", allow_pickle=True) as data:
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


def create_raw_datasets(args):
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

    return (dataset_train, dataset_test, max, min)


class StressRunnable(distributed.DistributedRunnable):

    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 datasets,
                 meta,
                 y_max=0,
                 y_min=0) -> None:
        super().__init__(net, optimizer, datasets, meta)
        self.loss = nn.MSELoss()
        self.y_max = y_max
        self.y_min = y_min

    def train_epoch(self, loader, optimizer, device):
        accuracy = 0
        for i, (X, Y) in enumerate(loader):
            Y = Y.to(device)

            y = self.net(X.to(device))

            loss = self.loss(y, Y)
            accuracy += loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        return {"acc_train": float(accuracy / len(loader))}

    def test(self, loader, mean_reduce, device):
        with torch.no_grad():
            self.net.eval()

            predictions = torch.Tensor().to(device)
            targets = torch.Tensor().to(device)

            for (X, Y) in loader:
                y = self.net(X.to(device))
                predictions = torch.cat((predictions, y))
                targets = torch.cat((targets, Y.to(device)))

            mse = mean_reduce(F.mse_loss(predictions, targets))
            l1loss = mean_reduce(torch.mean(torch.abs(targets - predictions), dim=0))
            l1loss_scaled = l1loss.cpu() * (self.y_max - self.y_min)

            test_point = {
                "acc_test": float(mse),
                "acc_test_g0": float(l1loss_scaled[0]),
                "acc_test_a": float(l1loss_scaled[1]),
            }

            print(test_point)

            self.net.train()
            return test_point


config = {
    "batch_size": 1024 * 3,
    "lr": 1e-5 * 3,
    "epochs": 200,
    "workers": 8,
    "resnet": 34,
    "pre_kernel": 3,
    "pre_stride": 2,
    "pre_padding": 1,
    "model": "res34"
}

(dataset_train, dataset_test, max, min) = create_datasets(distributed.get_args())

model = resnet.create_resnet34(2,
                               config["pre_kernel"],
                               config["pre_stride"],
                               config["pre_padding"],
                               conv_dim="1d")

optimizer = torch.optim.SGD(model.parameters(), config['lr'], momentum=0.9, weight_decay=1e-4)
runnable = StressRunnable(model, optimizer, (dataset_train, dataset_test), config, max, min)

runner = distributed.DistributedRunner(runnable,
                                       project="resnet_stress",
                                       batch_size=config["batch_size"],
                                       workers=config["workers"])

if runner.is_logger():
    summary(model, (1, 3, 201))
runner.start_training(config["epochs"])