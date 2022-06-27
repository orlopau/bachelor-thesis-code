import math
import torch

def conv_out_dim(dim_in, convolutions, padding):
    """
    Calculates the output dimensions for a layer of 1D convolutions, given their kernel and stride.

    Args:
        convolutions: [(kernel_0, stride_0), ..., (kernel_n, stride_n)]
        padding: padding for each convolution

    Returns:
        A list of size n with the output dimensions of n convolutions
    """

    dims = [dim_in]
    for i, (kernel, stride) in enumerate(convolutions):
        dims.append(math.floor((dims[i] - kernel + padding * 2) / stride + 1))

    return dims[1:]

def create_loaders(ds_train, ds_test, batch_size=100, workers=2):
    """Creates the loaders from the given datasets."""
    return (
        torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers),
        torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)
)
