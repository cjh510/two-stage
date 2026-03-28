

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class UNesTBlock_v1(nn.Module):
    """
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,  # type: ignore
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super(UNesTBlock_v1, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.transp_conv2 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.atten_gate_6 = Region_Atten_block(out_channels, out_channels, out_channels)

        if res_block:
            self.conv_block1 = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            self.conv_block2 = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block1 = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            self.conv_block2 = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, skip, inp1, inp2):
        # number of channels for skip should equals to out_channels
        x_PT = self.transp_conv1(inp1)
        x_MLN = self.transp_conv2(inp2)
        x_gate_PT, x_gate_MLN = self.atten_gate_6(skip, x_PT, x_MLN)
        out_PT = torch.cat((x_gate_PT, x_PT), dim=1)
        out_MLN = torch.cat((x_gate_MLN, x_MLN), dim=1)
        out_PT = self.conv_block1(out_PT)
        out_MLN = self.conv_block2(out_MLN)
        return out_PT, out_MLN


class UNesTBlock(nn.Module):
    """
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,  # type: ignore
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super(UNesTBlock, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UNestUpBlock(nn.Module):
    """
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetResBlock(
                                spatial_dims=3,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=3,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UNesTConvBlock(nn.Module):
    """
    UNesT block with skip connections
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        out = self.layer(inp)
        return out


class Region_Atten_block(nn.Module):

    def __init__(self, channel_x, channel_g, channel_num):
        super().__init__()

        self.Conv_g_1 = nn.Conv3d(channel_g, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_g_1 = nn.BatchNorm3d(channel_num)

        self.Conv_g_2 = nn.Conv3d(channel_g, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_g_2 = nn.BatchNorm3d(channel_num)

        self.Conv_x_1 = nn.Conv3d(channel_x, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_x_1 = nn.BatchNorm3d(channel_num)

        self.Conv_x_2 = nn.Conv3d(channel_x, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_x_2 = nn.BatchNorm3d(channel_num)

        self.Conv_relu = nn.Conv3d(channel_num * 2, 3, kernel_size=1, stride=1, padding='same')
        self.BN_relu = nn.BatchNorm3d(3)

        self.ReLU = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)

        self.AvgPool = nn.AvgPool3d(2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x_in, g_in_1, g_in_2):
        # print(x_in.shape, g_in_1.shape)
        g = self.Conv_g_1(g_in_1)
        g_int_1 = self.BN_g_1(g)

        g = self.Conv_g_2(g_in_2)
        g_int_2 = self.BN_g_2(g)

        x = self.Conv_x_1(x_in)
        x = self.BN_x_1(x)
        x_int_1 = x
        # x_int_1 = self.AvgPool(x)

        x = self.Conv_x_2(x_in)
        x = self.BN_x_2(x)
        x_int_2 = x
        # x_int_2 = self.AvgPool(x)

        x_1 = torch.add(x_int_1, g_int_1)
        x_2 = torch.add(x_int_2, g_int_2)
        x = torch.cat([x_1, x_2], dim=1)
        x_relu = self.ReLU(x)

        x = self.Conv_relu(x_relu)
        x = self.BN_relu(x)
        x = self.Softmax(x)
        # x_mask_1 = self.Upsample(x[:, 0:1, :, :, :])
        # x_mask_2 = self.Upsample(x[:, 1:2, :, :, :])
        x_mask_1 = x[:, 0:1, :, :, :]
        x_mask_2 = x[:, 1:2, :, :, :]

        x_out_1 = torch.mul(x_in, x_mask_1)
        x_out_2 = torch.mul(x_in, x_mask_2)

        return x_out_1, x_out_2
