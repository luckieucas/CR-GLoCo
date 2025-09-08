'''
Descripttion: 
version: 
Author: Luckie
Date: 2021-04-12 15:07:59
LastEditors: Luckie
LastEditTime: 2021-06-09 16:18:34
'''
import argparse

import torch
import yaml

from . import utils
logger = utils.get_logger('ConfigLoader')


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', default='/data/liupeng/semi-supervised_segmentation/3D_U-net/train_config_dice.yaml',type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
    #     if device_str.startswith('cuda') and not torch.cuda.is_available():
    #         logger.warn('CUDA not available, using CPU')
    #         device_str = 'cpu'
    else:
        device_str = 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
