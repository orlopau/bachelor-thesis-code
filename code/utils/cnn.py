import dataclasses
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary
import utils.generic as utils_generic
import utils.ml as utils_ml
import torch


def batch_accuracy(Y, y):
    """Calculates the accuracy of the batch prediciton, given one hot outputs."""
    truth_map = y.argmax(axis=1).eq(Y)
    return int(truth_map.sum()) / len(truth_map)


@dataclasses.dataclass
class CNNConfig:
    dim_in: tuple
    dim_out: tuple
    cnn_channels: tuple
    cnn_convolution: nn.Module
    linear_layers: tuple
    loss_function: torch.nn.Module
    cnn_activation: torch.nn.Module = nn.ReLU
    linear_activation: torch.nn.Module = nn.ReLU
    out_activation: torch.nn.Module = nn.Softmax(dim=1)


class GenericCNN:
    def __init__(self, net_config: CNNConfig, loader_train: torch.utils.data.DataLoader,
                 loader_test: torch.utils.data.DataLoader, device: torch.device) -> None:
        self.net_config = net_config
        self.net = self._create_network(net_config).to(device)
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.device = device

    def _create_network(self, net_config: CNNConfig):
        """
        Creates the network from a given config.
        """

        # add input channels to cnn layers
        conv_layer_channels = [net_config.dim_in[0]] + net_config.cnn_channels
        conv_layers = []
        for i, (ch_in, ch_out) in enumerate(zip(conv_layer_channels, conv_layer_channels[1:])):
            conv_layers.append(net_config.cnn_convolution(
                ch_in, ch_out, kernel_size=3, padding=1))
            conv_layers.append(net_config.cnn_activation())
            if i != len(conv_layer_channels) - 2:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, padding=1))

        # calc conv output size
        def tuple_reducer(t):
            return t[0] if type(t) is tuple else t

        conv_dims = utils_ml.conv_out_dim(net_config.dim_in[1], [(tuple_reducer(layer.kernel_size), tuple_reducer(layer.stride))
                                                                               for layer in filter(lambda l: type(l) is not net_config.cnn_activation, conv_layers)], 1)
        flatten_dim = conv_layer_channels[-1] * conv_dims[-1]**2

        dense_layer_neurons = [flatten_dim] + net_config.linear_layers
        dense_layers = []
        for n_in, n_out in zip(dense_layer_neurons, dense_layer_neurons[1:]):
            dense_layers.append(nn.Linear(n_in, n_out))
            dense_layers.append(net_config.linear_activation())

        dense_layers.append(
            nn.Linear(dense_layer_neurons[-1], net_config.dim_out[0]))
        if net_config.out_activation is not None:
            dense_layers.append(net_config.out_activation)

        net = nn.Sequential(*conv_layers, nn.Flatten(), *dense_layers)

        return net

    def test(self) -> float:
        """
        Tests the network on the test loader returning the accuracy.
        """
        with torch.no_grad():
            self.net.eval()
            accuracy = 0
            for (X, Y) in self.loader_test:
                y = self.net(X.to(self.device))
                accuracy += batch_accuracy(Y.to(self.device), y)
            self.net.train()
            return accuracy / len(self.loader_test)

    def train(self, optimizer) -> float:
        """
        Trains the network on the train loader for one iteration through it, returning the train accuracy.
        """
        accuracy = 0
        for i, (X, Y) in enumerate(self.loader_train):
            optimizer.zero_grad()

            y = self.net(X.to(self.device))
            accuracy += batch_accuracy(Y.to(self.device), y)

            loss = self.net_config.loss_function(y, Y.to(self.device))
            loss.backward()

            # clip gradients, arbitrary 1
            # nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        return accuracy / len(self.loader_train)
