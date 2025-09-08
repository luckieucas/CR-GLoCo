import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from batchgenerators.utilities.file_and_folder_operations import join
from cv2 import threshold
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.BCVData import DatasetSR
from dataset.dataset import TwoStreamBatchSampler
from networks.net_factory_3d import net_factory_3d
from unet3d.losses import DiceLoss  # test loss
from utils import losses, ramps
from utils.validation import test_all_case


class SemiSupervisedTrainer3D:
    def __init__(self, config, output_folder, logging,
                 continue_training: bool = False) -> None:
        self.config = config
        self.device = torch.device(f"cuda:{config['gpu']}")
        self.output_folder = output_folder
        self.logging = logging
        self.exp = config['exp']
        self.weight_decay = config['weight_decay']
        self.lr_scheduler_eps = config['lr_scheduler_eps']
        self.lr_scheduler_patience = config['lr_scheduler_patience']
        self.seed = config['seed']
        self.initial_lr = config['initial_lr']
        self.initial2_lr = config['initial2_lr']
        self.optimizer_type = config['optimizer_type']
        self.optimizer2_type = config['optimizer2_type']
        self.backbone = config['backbone']
        self.backbone2 = config['backbone2']
        self.max_iterations = config['max_iterations']
        self.began_semi_iter = config['began_semi_iter']
        self.began_eval_iter = config['began_eval_iter']
        self.save_checkpoint_freq = config['save_checkpoint_freq']
        self.val_freq = config['val_freq']

        # config for training from checkpoint
        self.continue_training = continue_training
        self.wandb_id = config['wandb_id']
        self.network_checkpoint = config['model_checkpoint']
        self.network2_checkpoint = config['model2_checkpoint']

        # config for semi-supervised
        self.consistency_rampup = config['consistency_rampup']
        self.consistency = config['consistency']

        # config for dataset
        dataset = config['DATASET']
        self.patch_size = dataset['patch_size']
        self.labeled_num = dataset['labeled_num']

        self.batch_size = dataset['batch_size']
        self.labeled_bs = dataset['labeled_bs']
        self.cutout = dataset['cutout']
        self.rotate_trans = dataset['rotate_trans']
        self.scale_trans = dataset['scale_trans']
        self.random_rotflip = dataset['random_rotflip']
        self.normalization = dataset['normalization']
        self.dataset_name = config['dataset_name']

        dataset_config = dataset[self.dataset_name]
        self.num_classes = dataset_config['num_classes']
        self.class_name_list = dataset_config['class_name_list']
        self.training_data_num = dataset_config['training_data_num']
        self.testing_data_num = dataset_config['testing_data_num']
        self.train_list = dataset_config['train_list']
        self.test_list = dataset_config['test_list']
        self.cut_upper = dataset_config['cut_upper']
        self.cut_lower = dataset_config['cut_lower']
        self.weights = dataset_config['weights']

        # config for method
        self.method_name = 'CR-GLoCo'
        self.method_config = config['METHOD'][self.method_name]

        self.experiment_name = None
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = losses.DiceLoss(self.num_classes)
        self.best_performance = 0.0
        self.best_performance2 = 0.0
        self.current_iter = 0
        self.model = None
        self.model2 = None
        self.scaler = None
        self.scaler2 = None

        self.dataset = None
        self.dataloader = None
        self.wandb_logger = None
        self.tensorboard_writer = None
        self.labeled_idxs = None
        self.unlabeled_idxs = None

        # generate by training process
        self.current_lr = self.initial_lr
        self.current2_lr = self.initial2_lr
        self.loss = None
        self.loss_ce = None
        self.loss_dice = None
        self.consistency_loss = None
        self.consistency_weight = None
        self.grad_scaler1 = GradScaler()
        self.grad_scaler2 = GradScaler()

    def initialize(self):
        self.experiment_name = f"{self.dataset_name}_{self.method_name}_" \
                               f"labeled{self.labeled_num}_" \
                               f"{self.optimizer_type}_{self.optimizer2_type}" \
                               f"_{self.exp}"

        self.wandb_logger = wandb.init(name=self.experiment_name,
                                       project="semi-supervised-segmentation",
                                       config=self.config)

        wandb.tensorboard.patch(root_logdir=self.output_folder + '/log')
        self.tensorboard_writer = SummaryWriter(self.output_folder + '/log')
        self.load_dataset()
        self.initialize_optimizer_and_scheduler()

    def initialize_network(self):
        self.model = net_factory_3d(
            net_type=self.backbone, in_chns=1, class_num=self.num_classes,
            model_config=self.config['model'], device=self.device
        )
        self.model2 = net_factory_3d(
            self.backbone2, in_chns=1, class_num=self.num_classes,
            device=self.device,
            large_patch_size=self.method_config['patch_size_large']
        )
        self._kaiming_normal_init_weight()

    def load_checkpoint(self, fname="latest"):
        checkpoint = torch.load(join(self.output_folder,
                                     "model1_" + fname + ".pth"))
        network_weights = checkpoint['network_weights']
        self.model.load_state_dict(network_weights)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler1 is not None:
            self.grad_scaler1.load_state_dict(
                checkpoint['grad_scaler_state'])
        self.current_iter = checkpoint['current_iter']
        print(
            f"=====> Load checkpoint from "
            f"{join(self.output_folder, 'model1_' + fname + '.pth')}"
            f" for model1 Successfully")

        # load  checkpoint for model2
        if self.model2 is not None:
            checkpoint2 = torch.load(join(self.output_folder,
                                          "model2_" + fname + ".pth"))
            network_weights2 = checkpoint2['network_weights']
            self.model2.load_state_dict(network_weights2)
            self.optimizer2.load_state_dict(checkpoint2['optimizer_state'])
            if self.grad_scaler2 is not None:
                self.grad_scaler2.load_state_dict(
                    checkpoint2['grad_scaler_state'])
            print(
                f"=====> Load checkpoint from "
                f"{join(self.output_folder, 'model2_' + fname + '.pth')}"
                f" for model2 Successfully")

    def load_dataset(self):
        self.dataset = DatasetSR(
            img_list_file=self.train_list,
            patch_size_small=self.patch_size,
            patch_size_large=self.method_config['patch_size_large'],
            num_class=self.num_classes,
            stride=self.method_config['stride'],
            iou_bound=[self.method_config['iou_bound_low'],
                       self.method_config['iou_bound_high']],
            labeled_num=self.labeled_num,
            cutout=self.cutout,
            rotate_trans=self.rotate_trans,
            scale_trans=self.scale_trans,
            random_rotflip=self.random_rotflip,
            upper=self.cut_upper,
            lower=self.cut_lower,
            weights=self.weights
        )

    def get_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True, num_workers=2,
                                     pin_memory=True)

    def initialize_optimizer_and_scheduler(self):
        assert self.model is not None, "self.initialize_network must be called first"
        self.scaler = GradScaler()
        if self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              self.initial_lr,
                                              weight_decay=self.weight_decay,
                                              amsgrad=True)
        elif self.optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.initial_lr,
                                             momentum=0.9,
                                             weight_decay=self.weight_decay)
        else:
            print("unrecognized optimizer, use Adam instead")
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              self.initial_lr,
                                              weight_decay=self.weight_decay,
                                              amsgrad=True)

        self.scaler2 = GradScaler()
        if self.optimizer2_type == 'Adam':
            self.optimizer2 = torch.optim.Adam(
                self.model2.parameters(),
                self.initial2_lr,
                weight_decay=self.weight_decay,
                amsgrad=True)
        elif self.optimizer2_type == 'SGD':
            self.optimizer2 = torch.optim.SGD(self.model2.parameters(),
                                              lr=self.initial2_lr,
                                              momentum=0.9,
                                              weight_decay=self.weight_decay)
        else:
            print("unrecognized optimizer type, use adam instead!")
            self.optimizer = torch.optim.Adam(
                self.model2.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=True
            )
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min', factor=0.2,
            patience=self.lr_scheduler_patience,
            verbose=True, threshold=self.lr_scheduler_eps,
            threshold_mode="abs")

    def train(self):
        self.labeled_idxs = list(range(0, self.labeled_num))
        self.unlabeled_idxs = list(
            range(self.labeled_num, self.training_data_num))
        batch_sampler = TwoStreamBatchSampler(self.labeled_idxs,
                                              self.unlabeled_idxs,
                                              self.batch_size,
                                              self.batch_size - self.labeled_bs)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_sampler=batch_sampler,
                                     num_workers=2, pin_memory=True)
        self.max_epoch = self.max_iterations // len(self.dataloader) + 1
        self._train_CR_GLoCo()

    def evaluation(self, model, do_SR=False, model_name="model"):
        """
        do_SR: whether do super resolution model
        """
        print("began evaluation!")
        model.eval()
        class_id_list = range(1, self.num_classes)
        best_performance = self.best_performance
        con_list = None
        test_num = self.testing_data_num

        avg_metric = test_all_case(
            model, test_list=self.test_list,
            num_classes=self.num_classes,
            patch_size=self.method_config[
                'patch_size_large'] if do_SR else self.patch_size,
            batch_size=2,
            stride_xy=64, stride_z=64,
            overlap=0.2,
            cut_upper=self.cut_upper,
            cut_lower=self.cut_lower,
            do_SR=do_SR,
            test_num=test_num,
            method=self.method_name.lower(),
            con_list=con_list,
            normalization=self.normalization)
        print("avg metric shape:", avg_metric.shape)
        if avg_metric[:, 0].mean() > best_performance:
            best_performance = avg_metric[:, 0].mean()
            save_name = f'iter_{self.current_iter}_dice_{round(best_performance, 4)}'
            self._save_checkpoint(save_name)

        self.tensorboard_writer.add_scalar(
            f'info/{model_name}_val_dice_score',
            avg_metric[:, 0].mean(), self.current_iter)
        self.tensorboard_writer.add_scalar(f'info/{model_name}val_hd95',
                                           avg_metric[:,
                                           1].mean(), self.current_iter)
        self.logging.info(
            'iteration %d : %s_dice_score : %f %s_hd95 : %f' % (
                self.current_iter,
                model_name,
                avg_metric[:, 0].mean(),
                model_name,
                avg_metric[:, 1].mean()))
        # print metric of each class
        for i, class_id in enumerate(class_id_list):
            class_name = self.class_name_list[class_id - 1]
            self.tensorboard_writer.add_scalar(
                f'DSC_each_class/{model_name}_{class_name}',
                avg_metric[i, 0], self.current_iter
            )
            self.tensorboard_writer.add_scalar(
                f'HD_each_class/{model_name}_{class_name}',
                avg_metric[i, 1], self.current_iter
            )

        return avg_metric

    def _train_CR_GLoCo(self):
        """cross supervision use high resolution"""
        print("================> Training CR-GLoCo<===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        self.model = self.model.float().to(self.device)
        self.model2 = self.model2.float().to(self.device)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self.model.train()
                self.model2.train()
                volume_large, label_large = (
                    sampled_batch['image_large'].float().to(self.device),
                    sampled_batch['label_large'].float().to(self.device)
                )
                volume_small, label_small = (
                    sampled_batch['image_small'].float().to(self.device),
                    sampled_batch['label_small'].float().to(self.device)
                )

                ul_large, br_large = sampled_batch['ul1'], sampled_batch['br1']
                ul_small, br_small = sampled_batch['ul2'], sampled_batch['br2']
                ul_large_u = [x[self.labeled_bs:] for x in ul_large]
                br_large_u = [x[self.labeled_bs:] for x in br_large]
                ul_small_u = [x[self.labeled_bs:] for x in ul_small]
                br_small_u = [x[self.labeled_bs:] for x in br_small]
                noise1 = torch.clamp(
                    torch.randn_like(volume_small) * 0.1,
                    -0.2,
                    0.2
                ).to(self.device)
                noise2 = torch.clamp(
                    torch.randn_like(volume_large) * 0.1,
                    -0.2,
                    0.2
                ).to(self.device)
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()
                with autocast():
                    outputs1 = self.model(volume_small + noise1)
                    outputs_soft1 = torch.softmax(outputs1, dim=1)
                    outputs2 = self.model2(volume_large + noise2)
                    outputs_soft2 = torch.softmax(outputs2, dim=1)

                    self.consistency_weight = self._get_current_consistency_weight(
                        self.current_iter // 150
                    )
                    loss1 = 0.5 * (self.ce_loss(outputs1[:self.labeled_bs],
                                                label_small[
                                                :self.labeled_bs].long()) +
                                   self.dice_loss(
                                       outputs_soft1[:self.labeled_bs],
                                       label_small[:self.labeled_bs].
                                           unsqueeze(1)))
                    loss2 = 0.5 * (self.ce_loss(outputs2[:self.labeled_bs],
                                                label_large[
                                                :self.labeled_bs].long()) +
                                   self.dice_loss(
                                       outputs_soft2[:self.labeled_bs],
                                       label_large[:self.labeled_bs].
                                           unsqueeze(1)))
                    max_prob1, pseudo_outputs1 = torch.max(
                        outputs_soft1[self.labeled_bs:].detach(), dim=1
                    )
                    filter1 = (
                            ((max_prob1 > 0.99) & (pseudo_outputs1 == 0)) |
                            ((max_prob1 > 0.95) & (pseudo_outputs1 != 0))
                    )

                    max_prob2, pseudo_outputs2 = torch.max(
                        outputs_soft2[self.labeled_bs:].detach(), dim=1
                    )
                    filter2 = (
                            ((max_prob2 > 0.99) & (pseudo_outputs2 == 0)) |
                            ((max_prob2 > 0.95) & (pseudo_outputs2 != 0))
                    )

                    if self.current_iter < self.began_semi_iter:
                        pseudo_supervision1 = torch.FloatTensor([0]).to(
                            self.device)
                        pseudo_supervision2 = torch.FloatTensor([0]).to(
                            self.device)
                    else:
                        pseudo_outputs2[filter2 == 0] = 255
                        pseudo_supervision1 = self.ce_loss(
                            outputs1[self.labeled_bs:, :, 
                            ul_small_u[0]:br_small_u[0],
                            ul_small_u[1]:br_small_u[1],
                            ul_small_u[2]:br_small_u[2]],
                            pseudo_outputs2[:, ul_large_u[0]:br_large_u[0],
                            ul_large_u[1]:br_large_u[1],
                            ul_large_u[2]:br_large_u[2]]
                        )
                        pseudo_outputs1[filter1 == 0] = 255
                        pseudo_supervision2 = self.ce_loss(
                            outputs2[self.labeled_bs:, :, 
                            ul_large_u[0]:br_large_u[0],
                            ul_large_u[1]:br_large_u[1],
                            ul_large_u[2]:br_large_u[2]],
                            pseudo_outputs1[:, ul_small_u[0]:br_small_u[0],
                            ul_small_u[1]:br_small_u[1],
                            ul_small_u[2]:br_small_u[2]]
                        )
                    model1_loss = loss1 + self.consistency_weight * pseudo_supervision1
                    model2_loss = loss2 + self.consistency_weight * pseudo_supervision2
                    loss = model1_loss + model2_loss

                self.grad_scaler1.scale(model1_loss).backward()
                self.grad_scaler1.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.grad_scaler1.step(self.optimizer)
                self.grad_scaler1.update()
                self.grad_scaler2.scale(model2_loss).backward()
                self.grad_scaler2.unscale_(self.optimizer2)
                torch.nn.utils.clip_grad_norm_(self.model2.parameters(), 12)
                self.grad_scaler2.step(self.optimizer2)
                self.grad_scaler2.update()

                self._adjust_learning_rate()
                self.current_iter += 1

                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar(
                    'consistency_weight/consistency_weight',
                    self.consistency_weight,
                    self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model1_loss',
                                                   model1_loss,
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss',
                                                   model2_loss,
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/pseudo1_loss',
                                                   pseudo_supervision1,
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/pseudo2_loss',
                                                   pseudo_supervision2,
                                                   self.current_iter)

                self.logging.info(
                    'iteration %d : model1 loss : %f model2 loss : %f' % (
                        self.current_iter, model1_loss.item(),
                        model2_loss.item()))

                if (
                        self.current_iter > self.began_eval_iter and
                        self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    self.evaluation(model=self.model)
                    self.model.train()
                    self.model2.train()

                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint("latest")
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()

    def _get_current_consistency_weight(self, epoch):
        return self.consistency * ramps.sigmoid_rampup(epoch, 
                                                       self.consistency_rampup)

    def _worker_init_fn(self, worker_id):
        random.seed(self.seed + worker_id)     

    def _save_checkpoint(self, filename: str) -> None:
        checkpoint1 = {
            'network_weights': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'grad_scaler_state': self.grad_scaler1.state_dict() if self.grad_scaler1 is not None else None,
            'current_iter': self.current_iter + 1,
            'wandb_id': self.wandb_logger.id
        }
        torch.save(checkpoint1,
                   join(self.output_folder, "model1_" + filename + ".pth"))
        if self.model2 is not None:
            checkpoint2 = {
                'network_weights': self.model2.state_dict(),
                'optimizer_state': self.optimizer2.state_dict(),
                'grad_scaler_state': self.grad_scaler2.state_dict() if self.grad_scaler2 is not None else None,
                'current_iter': self.current_iter + 1,
                'wandb_id': self.wandb_logger.id
            }
            torch.save(checkpoint2,
                       join(self.output_folder, "model2_" + filename + ".pth"))
        self.logging.info(
            f'save model to {join(self.output_folder, filename)}')

    def _kaiming_normal_init_weight(self):
        for m in self.model2.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _adjust_learning_rate(self):
        if self.optimizer_type == 'SGD':  # no need to adjust learning rate for adam optimizer
            print("current learning rate: ", self.current_lr)
            self.current_lr = self.initial_lr * (
                    1.0 - self.current_iter / self.max_iterations
            ) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr

        if self.optimizer2_type == 'SGD':
            print("current learning rate model2: ", self.current_lr)
            self.current2_lr = self.initial2_lr * (
                    1.0 - self.current_iter / self.max_iterations
            ) ** 0.9
            for param_group in self.optimizer2.param_groups:
                param_group['lr'] = self.current2_lr

    def _add_information_to_writer(self):
        for param_group in self.optimizer.param_groups:
            self.current_lr = param_group['lr']
        self.tensorboard_writer.add_scalar('info/lr', self.current_lr, 
                                           self.current_iter)
        self.tensorboard_writer.add_scalar('info/total_loss', self.loss, 
                                           self.current_iter)
        self.tensorboard_writer.add_scalar('info/loss_ce', self.loss_ce, 
                                           self.current_iter)
        self.tensorboard_writer.add_scalar('info/loss_dice', self.loss_dice, 
                                           self.current_iter)
        self.logging.info(
            'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
            (self.current_iter, self.loss.item(), self.loss_ce.item(), 
             self.loss_dice.item()))
        self.tensorboard_writer.add_scalar('loss/loss', self.loss, 
                                           self.current_iter)
        if self.consistency_loss:
            self.tensorboard_writer.add_scalar('info/consistency_loss',
                                               self.consistency_loss,
                                               self.current_iter)
        if self.consistency_weight:
            self.tensorboard_writer.add_scalar('info/consistency_weight',
                                               self.consistency_weight,
                                               self.current_iter)


if __name__ == "__main__":
    # test semiTrainer
    pass
