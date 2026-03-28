from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from Src.unest_block import UNesTConvBlock, UNestUpBlock, CBAM, UNesTBlock_v1
from monai.networks.blocks import Convolution
from Src.nest_transformer_3D import NestTransformer3D
from monai.networks.blocks.dynunet_block import UnetOutBlock


class Dual_UNseT(nn.Module):
    def __init__(self, in_channels, out_channels, cfig):
        super().__init__()
        channel_num = 64
        self.encoder = UNesT(in_channels=in_channels,
                             out_channels=out_channels,
                             img_size=cfig['img_size'],
                             patch_size=cfig['patch_size'],
                             depths=cfig['depth'],
                             num_heads=cfig['num_heads'],
                             embed_dim=cfig['embed_dims'])
        self.decoder = Diverging_decoder(in_channels=channel_num, out_channels=out_channels)
        self.PT_map = map_project()
        self.MLN_map = map_project()

    def forward(self, x):
        enc0, enc1, enc2, enc3, enc4, dec4 = self.encoder(x)
        Seg_PT_pred, Seg_MLN_pred, feature = self.decoder(enc0, enc1, enc2, enc3, enc4, dec4)
        PT_classes, PT_project = self.PT_map(feature[:5])
        MLN_classes, MLN_project = self.MLN_map(feature[5:])
        return Seg_PT_pred, Seg_MLN_pred, PT_classes, MLN_classes, PT_project, MLN_project


class map_project(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbam = CBAM(channel=512)
        self.mlp = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
        self.conv = UNesTConvBlock(
            spatial_dims=3,
            in_channels=512 + 256 + 128 + 64 + 32,
            out_channels=512,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

    def forward(self, feacture):
        feacture2 = feacture[2]
        feacture0 = F.interpolate(feacture[0], size=feacture2.shape[2:], mode='trilinear')
        feacture1 = F.interpolate(feacture[1], size=feacture2.shape[2:], mode='trilinear')
        feacture3 = F.interpolate(feacture[3], size=feacture2.shape[2:], mode='trilinear')
        feacture4 = F.interpolate(feacture[4], size=feacture2.shape[2:], mode='trilinear')

        fusion_feature = torch.cat([feacture0, feacture1, feacture2, feacture3, feacture4], dim=1)
        fusion_feature = self.conv(fusion_feature)
        fusion_feature = self.cbam(fusion_feature)
        fusion_feature = nn.AdaptiveAvgPool3d(8)(fusion_feature)
        B, C, D, H, W = fusion_feature.shape
        total = D * H * W
        project = fusion_feature.view(B, C, total)

        stack_feature = []
        for i in range(total):
            stack_feature.append(project[:, :, i])
        
        stack_feature = torch.stack(stack_feature, dim=1)
        stack_feature = stack_feature.view(B * total, C)
        stack_feature = self.mlp(stack_feature)
        return stack_feature, project


class Diverging_decoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_name: Union[Tuple, str] = "instance",
                 res_block: bool = True,):
        super().__init__()
        self.decoder1 = UNesTBlock_v1(
            spatial_dims=3,
            in_channels=in_channels * 16,
            out_channels=in_channels * 8,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UNesTBlock_v1(
            spatial_dims=3,
            in_channels=in_channels * 8,
            out_channels=in_channels * 4,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UNesTBlock_v1(
            spatial_dims=3,
            in_channels=in_channels * 4,
            out_channels=in_channels * 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UNesTBlock_v1(
            spatial_dims=3,
            in_channels=in_channels * 2,
            out_channels=in_channels,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UNesTBlock_v1(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels // 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out_PT = UnetOutBlock(spatial_dims=3, in_channels=in_channels // 2, out_channels=out_channels)
        self.out_MLN = UnetOutBlock(spatial_dims=3, in_channels=in_channels // 2, out_channels=out_channels)

    def forward(self, x_0, x_1, x_2, x_3, x_4, x_5):
        x_PT_1, x_MLN_1 = self.decoder1(x_4, x_5, x_5)
        x_PT_2, x_MLN_2 = self.decoder2(x_3, x_PT_1, x_MLN_1)
        x_PT_3, x_MLN_3 = self.decoder3(x_2, x_PT_2, x_MLN_2)
        x_PT_4, x_MLN_4 = self.decoder4(x_1, x_PT_3, x_MLN_3)
        x_PT_5, x_MLN_5 = self.decoder5(x_0, x_PT_4, x_MLN_4)

        # Segmentation output
        x = self.out_PT(x_PT_5)
        Seg_PT_pred = x

        x = self.out_MLN(x_MLN_5)
        Seg_MLN_pred = x

        Surv_feature = [x_PT_1, x_PT_2, x_PT_3, x_PT_4, x_PT_5, x_MLN_1, x_MLN_2, x_MLN_3, x_MLN_4, x_MLN_5]
        return Seg_PT_pred, Seg_MLN_pred, Surv_feature


class UNesT(nn.Module):
    """
    UNesT model implementation
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: int = 96,
            feature_size: int = 16,
            patch_size: int = 2,
            depths: Tuple[int, int, int] = [2, 2, 2],
            num_heads: Tuple[int, int, int] = [3, 6, 12],
            embed_dim: Tuple[int, int, int] = [128, 256, 512],
            norm_name: Union[Tuple, str] = "instance",
            res_block: bool = True,
            dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            # >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            # >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.embed_dim = embed_dim

        self.nestViT = NestTransformer3D(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=patch_size,
            num_levels=3,
            embed_dims=embed_dim,
            num_heads=num_heads,
            depths=depths,
            num_classes=1000,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.5,
            norm_layer=None,
            act_layer=None,
            pad_type='',
            weight_init='',
            global_pool='avg',
        )

        self.encoder1 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UNestUpBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )

        self.encoder3 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[1],
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder10 = Convolution(
            spatial_dims=3,
            in_channels=32 * feature_size,
            out_channels=64 * feature_size,
            strides=2,
            adn_ordering="ADN",
            dropout=0.0,
        )

    def forward(self, x_in):
        x, hidden_states_out = self.nestViT(x_in)
        enc0 = self.encoder1(x_in)  # 2, 32, 96, 96, 96 #UNesTConvBlock
        x1 = hidden_states_out[0]  # 2, 128, 24, 24, 24
        enc1 = self.encoder2(x1)  # 2, 64, 48, 48, 48  UNestUpBlock
        x2 = hidden_states_out[1]  # 2, 128, 24, 24, 24
        enc2 = self.encoder3(x2)  # 2, 128, 24, 24, 24 UNesTConvBlock
        x3 = hidden_states_out[2]  # 2, 256, 12, 12, 12
        enc3 = self.encoder4(x3)  # 2, 256, 12, 12, 12 UNesTConvBlock
        x4 = hidden_states_out[3]
        enc4 = x4  # 2, 512, 6, 6, 6
        dec4 = x  # 2, 512, 6, 6,
        dec4 = self.encoder10(dec4)  # 2, 1024, 3, 3, 3  Convolution
        return enc0, enc1, enc2, enc3, enc4, dec4


if __name__ == '__main__':
    import yaml
    yaml_file = 'yaml/mulorganseg_base.yaml'
    with open(yaml_file, 'r') as f:
        cfig = yaml.safe_load(f)
    x = torch.randn([1, 1, 128, 128, 128])
    net = Dual_UNseT(in_channels=1, out_channels=2, cfig=cfig)
    pre = net(x)
    print(pre[0].shape)