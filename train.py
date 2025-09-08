import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from trainer.semi_trainer_3D import SemiSupervisedTrainer3D
from utils.util import save_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/train_config_3d.yaml',
                    help='training configuration')
parser.add_argument('--c', action='store_true', required=False,
                    help="[OPTIONAL] Continue training from latest checkpoint")

if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    if not config['deterministic']:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    snapshot_path = "../model/{}_{}_{}_{}_{}_{}/{}" \
        .format(
        config['dataset_name'],
        config['DATASET']['labeled_num'],
        config['method'],
        config['exp'],
        config['optimizer_type'],
        config['optimizer2_type'],
        config['backbone']
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # move config to snapshot_path
    shutil.copyfile(args.config, snapshot_path + "/" + 
                    time.strftime("%Y-%m-%d=%H-%M-%S", 
                                  time.localtime()) + 
                    "_train_config.yaml")
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H-%M-%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    if config['train_3D']:
        trainer = SemiSupervisedTrainer3D(config=config,
                                          output_folder=snapshot_path,
                                          logging=logging,
                                          continue_training=args.c
                                          )
    trainer.initialize_network()
    trainer.initialize()
    if args.c:  # continue training
        trainer.load_checkpoint()
    trainer.train()