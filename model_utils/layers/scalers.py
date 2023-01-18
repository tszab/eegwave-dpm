
from torch import nn


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels: int, use_conv: bool = True, factor: int = 2, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.upsample = nn.Upsample(scale_factor=factor, mode="nearest")
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, (3, 1), padding=(1, 0))

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels: int, use_conv: bool = True, factor: int = 2, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, (3, 1), stride=(factor, 1), padding=(1, 0))
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=factor, stride=factor)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
