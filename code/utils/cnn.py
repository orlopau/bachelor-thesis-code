import dataclasses
import string
import time
from typing import Callable
from grpc import Call
import torch.nn as nn
import torch.nn.functional as F
import torch
import utils.ml as utils_ml
from torchinfo import summary


def batch_accuracy_onehot(y, Y):
    """Calculates the accuracy of the batch prediciton, given one hot outputs for classification."""
    truth_map = y.argmax(axis=1).eq(Y)
    return int(truth_map.sum()) / len(truth_map)

@dataclasses.dataclass
class CNNConfig:
    dim_in: tuple
    dim_out: tuple
    cnn_channels: list
    cnn_convolution_gen: Callable[[tuple], nn.Module]
    cnn_pool_gen: Callable[[], nn.Module]
    linear_layers: list
    loss_function: torch.nn.Module
    cnn_activation: torch.nn.Module = nn.ReLU
    linear_activation: torch.nn.Module = nn.ReLU
    out_activation: torch.nn.Module = nn.Softmax(dim=1),
    # function calculating the accuracy given the target and prediction of a whole batch, having the dimension [batch_size, ...]
    batch_accuracy_func: Callable[[torch.Tensor, torch.Tensor], float] = batch_accuracy_onehot


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
            conv_layers.append(net_config.cnn_convolution_gen((ch_in, ch_out)))
            conv_layers.append(net_config.cnn_activation())
            conv_layers.append(net_config.cnn_pool_gen())

        # calc conv output size
        def tuple_reducer(t):
            return t[0] if type(t) is tuple else t

        conv_dims = utils_ml.conv_out_dim(net_config.dim_in[1], [(tuple_reducer(layer.kernel_size), tuple_reducer(layer.stride), tuple_reducer(layer.padding))
                                                                               for layer in filter(lambda l: type(l) is not net_config.cnn_activation, conv_layers)])
        
        conv_out_dim = conv_dims[-1]
        if net_config.cnn_convolution_gen((1,1)) == nn.Conv2d:
            conv_out_dim = conv_out_dim**2

        flatten_dim = conv_layer_channels[-1] * conv_out_dim

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

    def test(self, loader=None) -> float:
        """
        Tests the network on the test loader returning the accuracy.
        """
        if loader == None: loader = self.loader_test

        samples = 0
        with torch.no_grad():
            self.net.eval()
            accuracy = 0
            iters = 0
            for (X, Y) in loader:
                iters += 1
                samples += X.shape[0]
                y = self.net(X.to(self.device))
                accuracy += self.net_config.batch_accuracy_func(y, Y.to(self.device))
            self.net.train()
            return float(accuracy / len(loader))

    def train(self, optimizer):
        """
        Trains the network on the train loader for one iteration, returning the train accuracy.
        """
        accuracy = 0
        for i, (X, Y) in enumerate(self.loader_train):
            optimizer.zero_grad()

            y = self.net(X.to(self.device))
            accuracy += self.net_config.batch_accuracy_func(y, Y.to(self.device))

            loss = self.net_config.loss_function(y, Y.to(self.device))
            loss.backward()

            # clip gradients, arbitrary 1
            # nn.utils.clip_grad_norm_(model.parameters(), 1)
            step_pre_time = time.time()
            optimizer.step()
            step_time = time.time() - step_pre_time

        return (float(accuracy / len(self.loader_train)), step_time)

    def summary(self, batch_size) -> string:
        summary(self.net, (batch_size, *self.net_config.dim_in))
