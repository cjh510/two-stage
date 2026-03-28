""" Create Conv3d Factory Method

Hacked together by / Copyright 2020 Ross Wightman
"""

from .mixed_conv3d import MixedConv3d
from .conv3d_same import create_conv3d_pad


def create_conv3d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 3d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv3d, Conv3dSame, or MixedConv3d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        assert 'groups' not in kwargs  # MixedConv groups are defined by kernel list
        m = MixedConv3d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        m = create_conv3d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m
