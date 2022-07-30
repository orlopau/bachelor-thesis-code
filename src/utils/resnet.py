import torch.nn as nn


def activation_func(activation):
    return nn.ModuleDict([['relu', nn.ReLU()], ['relu_leak', nn.LeakyReLU()], ['selu', nn.SELU()],
                          ['none', nn.Identity()]])[activation]


_conv = {"1d": nn.Conv1d, "2d": nn.Conv2d, "3d": nn.Conv3d}
_max_pool = {"1d": nn.MaxPool1d, "2d": nn.MaxPool2d, "3d": nn.MaxPool3d}
_batch_norm = {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}


def _adaptive_avg_pool(dim, pool=1):
    return nn.ModuleDict([["1d", nn.AdaptiveAvgPool1d(pool)], ["2d",
                                                               nn.AdaptiveAvgPool2d((pool, pool))],
                          ["3d", nn.AdaptiveAvgPool3d((pool, pool, pool))]])[dim]


class ResBlock(nn.Module):

    def __init__(self, blocks: nn.Module, shortcut: nn.Module, activation) -> None:
        super().__init__()
        self.blocks, self.shortcut, self.activation = blocks, shortcut, activation

    def forward(self, x):
        res = self.shortcut(x)
        x = self.blocks(x)
        x += res
        return activation_func(self.activation)(x)


class ResBlockConv(ResBlock):

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride_first=1,
                 kernel=3,
                 activation='relu',
                 projection_shortcut=False,
                 conv_dim='2d') -> None:

        blocks = nn.Sequential(
            _conv[conv_dim](ch_in,
                            ch_out,
                            kernel_size=kernel,
                            stride=stride_first,
                            padding=kernel // 2,
                            bias=False), _batch_norm[conv_dim](ch_out),
            activation_func(activation), _conv[conv_dim](ch_out,
                                                         ch_out,
                                                         kernel_size=kernel,
                                                         padding=kernel // 2,
                                                         bias=False), _batch_norm[conv_dim](ch_out))

        shortcut = nn.Sequential(
            _conv[conv_dim](ch_in, ch_out, kernel_size=1, stride=stride_first, bias=False),
            _batch_norm[conv_dim](ch_out)) if projection_shortcut else nn.Identity()

        super().__init__(blocks, shortcut, activation)


def create_superblock(ch_in, ch_out, n_blocks, downsample=True, conv_dim="2d"):
    blocks = []
    for i in range(n_blocks):
        if i == 0 and downsample:
            blocks.append(
                ResBlockConv(ch_in, ch_out, stride_first=2, projection_shortcut=True, conv_dim=conv_dim))
        else:
            blocks.append(ResBlockConv(ch_out, ch_out, conv_dim=conv_dim))
    return nn.Sequential(*blocks)


def create_res_pre(kernel=7, stride=2, padding=3, conv_dim="2d"):
    return nn.Sequential(
        _conv[conv_dim](3, 64, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        _batch_norm[conv_dim](64), nn.ReLU(), _max_pool[conv_dim](kernel_size=3, stride=2, padding=1))


def create_res_post(outputs, conv_dim="2d"):
    return nn.Sequential(_adaptive_avg_pool(conv_dim), nn.Flatten(), nn.Linear(512, outputs))


def create_resnet18(outputs, kernel=7, stride=2, padding=3, conv_dim="2d"):
    return nn.Sequential(create_res_pre(kernel, stride, padding, conv_dim),
                         create_superblock(64, 64, 2, False, conv_dim),
                         create_superblock(64, 128, 2, conv_dim=conv_dim),
                         create_superblock(128, 256, 2, conv_dim=conv_dim),
                         create_superblock(256, 512, 2, conv_dim=conv_dim),
                         create_res_post(outputs, conv_dim))


def create_resnet34(outputs, kernel=7, stride=2, padding=3, conv_dim="2d"):
    return nn.Sequential(create_res_pre(kernel, stride, padding, conv_dim),
                         create_superblock(64, 64, 3, False, conv_dim),
                         create_superblock(64, 128, 4, conv_dim=conv_dim),
                         create_superblock(128, 256, 6, conv_dim=conv_dim),
                         create_superblock(256, 512, 3, conv_dim=conv_dim),
                         create_res_post(outputs, conv_dim))
