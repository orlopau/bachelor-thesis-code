from sklearn.model_selection import train_test_split
import utils.distributed as distributed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.cnn as cnn
import horovod.torch as hvd

config = {
    "batch_size": 512,
    "lr": 1e-3,
    "epochs": 500,
    "cnn_channels": [40, 50, 40],
    "pool_each": 1,
    "linear_layers": [200, 100],
    "workers": 4
}

def create_datasets(args):
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

    return (dataset_train, dataset_test, max, min)


class DistributedStressNet(distributed.DistributedNet):
    def create_test_points(self):
        (y, Y) = self.cnn_net.test()

        mse = F.mse_loss(y, Y)
        loss = hvd.allreduce(mse)

        l1loss = hvd.allreduce(torch.mean(torch.abs(Y - y), dim=0))
        l1loss_scaled = l1loss.cpu() * (self.max - self.min)

        test_point = {
            "test_acc": float(loss),
            "test_acc_l1": float(l1loss),
            "test_acc_g0": float(l1loss_scaled[0]),
            "test_acc_a": float(l1loss_scaled[1]),
        }

        return test_point


dist_net = DistributedStressNet(config=config, project="stress_cnn")
(dataset_train, dataset_test, max, min) = create_datasets(dist_net.args)
dist_net.max = torch.from_numpy(max)
dist_net.min = torch.from_numpy(min)

net_config = cnn.CNNConfig(
    dim_in=dataset_train.tensors[0][1].size(),
    dim_out=dataset_train.tensors[1][1].size(),
    cnn_channels=config["cnn_channels"],
    cnn_convolution_gen=lambda c: nn.Conv1d(c[0], c[1], kernel_size=5, padding=2),
    cnn_pool_gen=lambda: nn.MaxPool1d(kernel_size=2),
    cnn_pool_each=config["pool_each"],
    linear_layers=config["linear_layers"],
    loss_function=nn.MSELoss(),
    out_activation=lambda: nn.Sigmoid(),
    batch_accuracy_func=F.mse_loss,
    cnn_activation=nn.ReLU,
    linear_activation=nn.ReLU)

dist_net.init((dataset_train, dataset_test), net_config)
dist_net.start_training()
