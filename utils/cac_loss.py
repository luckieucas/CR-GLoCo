'''
Descripttion: Context Aware Consistency Loss
version: 
Author: Luckie
Date: 2021-06-21 22:08:51
LastEditors: Luckie
LastEditTime: 2021-11-29 16:01:51
'''
import math, time
import random
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint
import numpy as np
import pickle


class CAC(nn.Module):
    def __init__(self, num_classes, stride=16, selected_num=400, b = 500, 
                 step_save=1, temp=0.1, proj_final_dim=128, pos_thresh_value=0.1, 
                 weight=0.1):

        super(CAC, self).__init__()
        self.num_classes = num_classes
        self.proj_final_dim = proj_final_dim
        self.stride = stride
        self.selected_num = selected_num
        self.b = b
        self.step_save = step_save
        self.temp = temp
        self.step_count = 0
        self.feature_bank = []
        self.pseudo_label_bank = []
        self.pos_thresh_value = pos_thresh_value
        self.weight = weight

    def forward(self,output_ul1=None, output_ul2=None, logits1=None, 
                logits2=None, ul1=None, br1=None, ul2=None, br2=None):
        device = output_ul1.device
        pseudo_logits_1 = F.softmax(logits1, 1).max(1)[0].detach() #[batch_size, h, w]
        pseudo_logits_2 = F.softmax(logits2, 1).max(1)[0].detach()          
        pseudo_label1 = logits1.max(1)[1].detach() #[batch_size, h, w]
        pseudo_label2 = logits2.max(1)[1].detach()

        # # get overlap part
        output_feature_list1 = []
        output_feature_list2 = []
        pseudo_label_list1 = []
        pseudo_label_list2 = []
        pseudo_logits_list1 = []
        pseudo_logits_list2 = []
        # target1_overlap = target1[:, ul1[0]:br1[0], ul1[1]:br1[1], ul1[2]:br1[2]]
        # target2_overlap = target2[:, ul2[0]:br2[0], ul2[1]:br2[1], ul2[2]:br2[2]]
        # assert (target1_overlap!=target2_overlap).sum() == 0,"error"
        for idx in range(logits1.size(0)): # iterate use batch size
            output_ul1_idx = output_ul1[idx]
            output_ul2_idx = output_ul2[idx]
            pseudo_label1_idx = pseudo_label1[idx]
            pseudo_label2_idx = pseudo_label2[idx]
            pseudo_logits_1_idx = pseudo_logits_1[idx]
            pseudo_logits_2_idx = pseudo_logits_2[idx]
            output_feature_list1.append(output_ul1_idx[:, ul1[0]//self.stride:br1[0]//self.stride, ul1[1]//self.stride:br1[1]//self.stride, ul1[2]//self.stride:br1[2]//self.stride].permute(1, 2, 3, 0).contiguous().view(-1, output_ul1.size(1)))
            output_feature_list2.append(output_ul2_idx[:, ul2[0]//self.stride:br2[0]//self.stride, ul2[1]//self.stride:br2[1]//self.stride, ul2[2]//self.stride:br2[2]//self.stride].permute(1, 2, 3, 0).contiguous().view(-1, output_ul2.size(1)))
            pseudo_label_list1.append(pseudo_label1_idx[ul1[0]//self.stride:br1[0]//self.stride, ul1[1]//self.stride:br1[1]//self.stride, ul1[2]//self.stride:br1[2]//self.stride].contiguous().view(-1))
            pseudo_label_list2.append(pseudo_label2_idx[ul2[0]//self.stride:br2[0]//self.stride, ul2[1]//self.stride:br2[1]//self.stride, ul2[2]//self.stride:br2[2]//self.stride].contiguous().view(-1))
            pseudo_logits_list1.append(pseudo_logits_1_idx[ul1[0]//self.stride:br1[0]//self.stride, ul1[1]//self.stride:br1[1]//self.stride, ul1[2]//self.stride:br1[2]//self.stride].contiguous().view(-1))
            pseudo_logits_list2.append(pseudo_logits_2_idx[ul2[0]//self.stride:br2[0]//self.stride, ul2[1]//self.stride:br2[1]//self.stride, ul2[2]//self.stride:br2[2]//self.stride].contiguous().view(-1))
        output_feat1 = torch.cat(output_feature_list1, 0) #[n, c]
        output_feat2 = torch.cat(output_feature_list2, 0) #[n, c]
        # # print("output feat1 shape:", output_feat1.shape)
        # # print("output feat2 shape:", output_feat2.shape)
        pseudo_label1_overlap = torch.cat(pseudo_label_list1, 0) #[n,]
        pseudo_label2_overlap = torch.cat(pseudo_label_list2, 0) #[n,]
        pseudo_logits1_overlap = torch.cat(pseudo_logits_list1, 0) #[n,]
        pseudo_logits2_overlap = torch.cat(pseudo_logits_list2, 0) #[n,] 
        assert output_feat1.size(0) == output_feat2.size(0)
        assert pseudo_label1_overlap.size(0) == pseudo_label2_overlap.size(0)
        assert output_feat1.size(0) == pseudo_label1_overlap.size(0)

        # concat across multi-gpus
        #可以对output ul1 和 pseudo_label1 先进行一次过滤 把概率低的过滤掉 还有预测不正确的
        b, c, d, h, w = output_ul1.size()
        selected_num = self.selected_num
        output_ul1_flatten = output_ul1.permute(0, 2, 3, 4, 1).contiguous().view(b*d*h*w, c)
        output_ul2_flatten = output_ul2.permute(0, 2, 3, 4, 1).contiguous().view(b*d*h*w, c)
        selected_idx1 = np.random.choice(range(b*d*h*w), selected_num, replace=False)
        selected_idx2 = np.random.choice(range(b*d*h*w), selected_num, replace=False)
        output_ul1_flatten_selected = output_ul1_flatten[selected_idx1]
        output_ul2_flatten_selected = output_ul2_flatten[selected_idx2]
        output_ul_flatten_selected = torch.cat([output_ul1_flatten_selected, output_ul2_flatten_selected], 0) #[2*kk, c]
        #output_ul_all = self.concat_all_gather(output_ul_flatten_selected) #[2*N, c]
        output_ul_all = output_ul_flatten_selected
        pseudo_label1_flatten_selected = pseudo_label1.view(-1)[selected_idx1]
        pseudo_label2_flatten_selected = pseudo_label2.view(-1)[selected_idx2]
        pseudo_label_flatten_selected = torch.cat([pseudo_label1_flatten_selected, pseudo_label2_flatten_selected], 0) #[2*kk]
        
        # get selected pred logits
        pseudo_logits1_flatten_selected = pseudo_logits_1.view(-1)[selected_idx1]
        pseudo_logits2_flatten_selected = pseudo_logits_2.view(-1)[selected_idx2]
        pseudo_logits_flatten_selected = torch.cat([pseudo_logits1_flatten_selected, pseudo_logits2_flatten_selected], 0) #[2*kk]
                
        # pseudo_label_all = self.concat_all_gather(pseudo_label_flatten_selected) #[2*N]
        pseudo_label_all = pseudo_label_flatten_selected


        self.feature_bank.append(output_ul_all)
        self.pseudo_label_bank.append(pseudo_label_all)
        if self.step_count > self.step_save:
            self.feature_bank = self.feature_bank[1:]
            self.pseudo_label_bank = self.pseudo_label_bank[1:]
        else:
            self.step_count += 1
        output_ul_all = torch.cat(self.feature_bank, 0).detach()
        pseudo_label_all = torch.cat(self.pseudo_label_bank, 0).detach()
        
      
      
        eps = 1e-8
        pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / self.temp #[n, 1]
        pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp #[n, 1]
        b = self.b
        def run1(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1):
            # print("gpu: {}, i_1: {}".format(gpu, i))
            mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float() #[n, b]
            neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp #[n, b]
            logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1) #[n, ]
            return logits1_neg_idx

        def run1_0(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap):
            # print("gpu: {}, i_1_0: {}".format(gpu, i))
            mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float() #[n, b]
            neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp #[n, b]
            neg1_idx = torch.cat([pos, neg1_idx], 1) #[n, 1+b]
            mask1_idx = torch.cat([torch.ones(mask1_idx.size(0), 1).float().to(device), mask1_idx], 1) #[n, 1+b]
            if len(neg1_idx) == 0:
                 print("size neg1 idx is 0")
            neg_max1 = torch.max(neg1_idx, 1, keepdim=True)[0] #[n, 1]
            logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1) #[n, ]
            return logits1_neg_idx, neg_max1




        N = output_ul_all.size(0)
        # print("n:",N)
        logits1_down = torch.zeros(pos1.size(0)).float().to(device)
        for i in range((N-1)//b + 1):
            # print("i:",i)
            # print("gpu: {}, i: {}".format(gpu, i))
            pseudo_label_idx = pseudo_label_all[i*b:(i+1)*b]
            output_ul_idx = output_ul_all[i*b:(i+1)*b]
            if i == 0:
                #logits1_neg_idx, neg_max1 = run1_0(pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap)
                logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(run1_0, pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap)
            else:
                #logits1_neg_idx = run1(pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1)
                logits1_neg_idx = torch.utils.checkpoint.checkpoint(run1, pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1)
            logits1_down += logits1_neg_idx

        logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)
        
        pos_mask_1 = ((pseudo_logits2_overlap > self.pos_thresh_value) & (pseudo_logits1_overlap < pseudo_logits2_overlap)).float()
        
        loss1 = -torch.log(logits1 + eps)
        loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)

        # compute loss2
        def run2(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2):
            # print("gpu: {}, i_2: {}".format(gpu, i))
            mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float() #[n, b]
            neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp #[n, b]
            logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1) #[n, ]
            return logits2_neg_idx

        def run2_0(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap):
            # print("gpu: {}, i_2_0: {}".format(gpu, i))
            mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float() #[n, b]
            neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp #[n, b]
            neg2_idx = torch.cat([pos, neg2_idx], 1) #[n, 1+b]
            mask2_idx = torch.cat([torch.ones(mask2_idx.size(0), 1).float().to(device), mask2_idx], 1) #[n, 1+b]
            neg_max2 = torch.max(neg2_idx, 1, keepdim=True)[0] #[n, 1]
            logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1) #[n, ]
            return logits2_neg_idx, neg_max2

        N = output_ul_all.size(0)
        logits2_down = torch.zeros(pos2.size(0)).float().to(device)
        for i in range((N-1)//b + 1):
            pseudo_label_idx = pseudo_label_all[i*b:(i+1)*b]
            output_ul_idx = output_ul_all[i*b:(i+1)*b]
            if i == 0:
                logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(run2_0, pos2, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap)
            else:
                logits2_neg_idx = torch.utils.checkpoint.checkpoint(run2, pos2, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2)
            logits2_down += logits2_neg_idx

        logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_down + eps)
        pos_mask_2 = ((pseudo_logits1_overlap > self.pos_thresh_value) & (pseudo_logits2_overlap < pseudo_logits1_overlap)).float()

        #print("pos mask 2:", pos_mask_2.sum())
        loss2 = -torch.log(logits2 + eps)
        loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum() + 1e-12)

        loss_unsup = self.weight * (loss1 + loss2 )


        return loss_unsup

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output