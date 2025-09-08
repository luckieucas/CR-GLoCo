
'''
Descripttion: 
version: 
Author: Luckie
Date: 2021-12-29 11:07:34
LastEditors: Luckie
LastEditTime: 2022-01-08 18:58:10
'''
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import \
    get_matching_instancenorm, convert_dim_to_conv_op
from monai.networks.nets import UNet

from networks.attention_unet import Attention_UNet
from networks.DenseNet3D import SinglePathDenseNet
from networks.DenseVox3D import DenseVoxelNet
from networks.nnunet import initialize_network
from networks.PlaninUNet_sr import PlainConvUNetSR
from networks.resnet3D import generate_resnet3d
from networks.unet_3D import unet_3D
from networks.unet_3D_dv_semi import unet_3D_dv_semi
from networks.unet_3D_sr import unet_3D_sr
from unet3d.model import get_model


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2,
                   model_config=None, device=None, condition_noise=False,
                   large_patch_size=(108, 208, 288)):
    if net_type == "unet_3D":
        model_config['out_channels'] = class_num
        net = get_model(model_config)
    elif net_type == 'unet_3D_old':
        net = unet_3D(n_classes=class_num, in_channels=in_chns).to(device)
    elif net_type == 'PlainConvUNet':
        print('nnunet', "*" * 10)
        conv_or_blocks_per_stage = {'n_conv_per_stage': [2, 2, 2, 2, 2],
                                    'n_conv_per_stage_decoder': [2, 2, 2, 2]}
        kwargs = {'conv_bias': True,
                  'norm_op': get_matching_instancenorm(nn.Conv3d),
                  'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                  'dropout_op': None, 'dropout_op_kwargs': None,
                  'nonlin': nn.LeakyReLU,
                  'nonlin_kwargs': {'inplace': True}
                  }
        net = PlainConvUNet(input_channels=in_chns, n_stages=5,
                            features_per_stage=[min(16 * 2 ** i,
                                                    384) for i in range(5)],
                            conv_op=nn.Conv3d,
                            num_classes=class_num,
                            kernel_sizes=[[1, 3, 3], [3, 3, 3], [3, 3, 3],
                                          [3, 3, 3], [3, 3, 3]],
                            strides=[[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2],
                                     [2, 2, 2]],
                            deep_supervision=False,
                            **conv_or_blocks_per_stage,
                            **kwargs
                            )
        net.apply(InitWeights_He(1e-2))
        print(f"net:{net}")
    elif net_type == 'unet_3D_sr':
        net = unet_3D_sr(feature_scale=4, n_classes=class_num, is_deconv=True,
                         in_channels=1, is_batchnorm=True,
                         large_patch_size=large_patch_size).to(device)
    elif net_type == 'PlainConvUNetSR':
        conv_or_blocks_per_stage = {'n_conv_per_stage': [2, 2, 2, 2, 2],
                                    'n_conv_per_stage_decoder': [2, 2, 2, 2]}
        kwargs = {'conv_bias': True,
                  'norm_op': get_matching_instancenorm(nn.Conv3d),
                  'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                  'dropout_op': None, 'dropout_op_kwargs': None,
                  'nonlin': nn.LeakyReLU,
                  'nonlin_kwargs': {'inplace': True}
                  }
        net = PlainConvUNetSR(input_channels=in_chns, n_stages=5,
                              features_per_stage=[min(16 * 2 ** i,
                                                      384) for i in range(5)],
                              conv_op=nn.Conv3d,
                              num_classes=class_num,
                              kernel_sizes=[[1, 3, 3], [3, 3, 3], [3, 3, 3],
                                            [3, 3, 3], [3, 3, 3]],
                              strides=[[1, 1, 1], [1, 2, 2], [2, 2, 2],
                                       [2, 2, 2], [2, 2, 2]],
                              deep_supervision=False,
                              **conv_or_blocks_per_stage,
                              **kwargs
                              )
        net.apply(InitWeights_He(1e-2))
    else:
        net = None
    return net


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, \
                                                        nn.Conv2d) or isinstance(
            module, nn.ConvTranspose2d) or isinstance(module,
                                                       nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, \
                                                    a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
