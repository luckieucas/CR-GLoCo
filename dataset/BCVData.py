'''
Descripttion: 
version: 
Author: Luckie
Date: 2022-01-09 16:44:21
LastEditors: Luckie
LastEditTime: 2022-02-17 10:42:48
'''
import os
import random
import torch
from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader
import numpy as np  
import SimpleITK as sitk
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
from batchgenerators.augmentations.utils import pad_nd_image
try:
    from sampler import BatchSampler,ClassRandomSampler
except:
    from dataset.sampler import BatchSampler,ClassRandomSampler

try:
    from data_augmentation import (
        rotation, affine_transformation, random_cutout, random_rotate_flip
    )
except:
    from dataset.data_augmentation import (
        rotation, affine_transformation, random_cutout, random_rotate_flip
    )
task_name_id_dict={"full":0,"spleen":1,"kidney":2,"liver":4,
                   "pancreas":5, 'heart':0}
class BCVDataset(dataset):
    def __init__(self, img_list_file, patch_size, labeled_num, 
                     cutout=False, affine_trans=False, 
                    upper=1000, lower=-1000,
                    num_class=6, train_supervised=False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.cutout = cutout 
        self.affine_trans = affine_trans
        self.upper = upper
        self.lower = lower # for cut the image
        self.num_class = num_class
        self.labeled_num = labeled_num
        self.train_supervised = train_supervised
        print(f"label num:{self.labeled_num}")
        with open(img_list_file, 'r') as f:
            self.img_list = [img.replace('\n','') for img in f.readlines()]
        if self.train_supervised:
            self.img_list = self.img_list[:self.labeled_num]
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        if len(self.img_list[index].strip().split()) > 1:
            img_path, mask_path = self.img_list[index].strip().split()
        else:
            img_path = self.img_list[index]
            mask_path = img_path.replace('img','label')
        if index < self.labeled_num:
            assert os.path.isfile(mask_path),"invalid mask path error"

        #Read image and mask
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        if index < self.labeled_num:
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            mask = mask.astype(np.uint8)
        else:
            mask = np.zeros_like(img)

        # clip image


        # get random patch
        D,W,H = img.shape
        #print("image name:{}, image shape:{}".format(img_path, img.shape))

        if D <= self.patch_size[0] or W <= self.patch_size[1] or H <=self.patch_size[2]:
            pw = max((self.patch_size[0] - D) // 2 + 3, 0)
            ph = max((self.patch_size[1] - W) // 2 + 3, 0)
            pd = max((self.patch_size[2] - H) // 2 + 3, 0)
            img = np.pad(img, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            mask = np.pad(mask, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
        
        np.clip(img, self.lower, self.upper, out=img)

        #normalize image
        img = (img-img.mean())/img.std()
        D,W,H = img.shape
        margin_len_x = self.patch_size[0]//2
        margin_len_y = self.patch_size[1]//2
        margin_len_z = self.patch_size[2]//2
        for _ in range(10):
            center_x = random.randint(margin_len_x, D-margin_len_x)
            center_y = random.randint(margin_len_y, W-margin_len_y)
            center_z = random.randint(margin_len_z, H-margin_len_z)
            img_array = img[center_x-margin_len_x:center_x+margin_len_x,
                        center_y-margin_len_y:center_y+margin_len_y,\
                            center_z-margin_len_z:center_z+margin_len_z]
            mask_array = mask[center_x-margin_len_x:center_x+margin_len_x,\
                                    center_y-margin_len_y:center_y+margin_len_y,\
                                    center_z-margin_len_z:center_z+margin_len_z]
            label_list = list(np.unique(mask_array))
            if len(label_list) < 2 and 0 in label_list:
                continue
            else:
                break
        if 0 in label_list:
            label_list.remove(0)
        if len(label_list)>0:
            condition = np.random.choice(label_list)
        else:
            condition = np.random.choice([i for i in range(1, self.num_class)])
        """Do Random cutout"""
        if self.cutout:
            mask_array = random_cutout(mask_array)
        
        
        img_array = torch.FloatTensor(img_array).unsqueeze(0)
        mask_array = torch.FloatTensor(mask_array).unsqueeze(0)
        condition = torch.Tensor([condition])
        gt_onehot = torch.zeros((self.num_class, mask_array.shape[1], mask_array.shape[2],mask_array.shape[3]))
        gt_onehot.scatter_(0, mask_array.long(), 1)
        #mask_array = gt_onehot

        """Do Affine Transformation"""
        if self.affine_trans:
            angle_x = random.uniform(-0.08,0.08)
            angle_y = random.uniform(-0.08,0.08)
            angle_z = random.uniform(-0.08,0.08)
            scale_x = random.uniform(0.8,1.2)
            scale_y = random.uniform(0.8,1.2)
            scale_z = random.uniform(0.8,1.2)     
            img = affine_transformation(img_array[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=0, border_mode="nearest", 
                                        constant_val=0, is_reverse=False)
            mask = affine_transformation(mask_array[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=0, border_mode="nearest",
                                        constant_val=0, is_reverse=False)
            img_array = img[0,:]
            mask_array = mask[0,:].squeeze()



        sample = {'image': img_array, 'label': mask_array.long(), 
                  'condition':condition.long(),'idx':index}
        #get image patch

        return sample


class BCVDatasetCAC(dataset):
    def __init__(self, img_list_file, labeled_num=None, patch_size=(48, 224, 224),
                cutout=False, rotate_trans=False, scale_trans=False,
                random_rotflip=False,
                num_class=2, edge_prob=0., upper=200, lower=-68, 
                stride=8, iou_bound=[0.25,0.95], con_list=None, 
                addi_con_list=None, weights=None):
        self.patch_size = patch_size
        self.cutout = cutout
        self.rotate_trans = rotate_trans
        self.scale_trans = scale_trans
        self.random_rotflip = random_rotflip
        self.num_class = num_class
        self.edge_prob = edge_prob
        self.upper = upper
        self.lower = lower
        self.stride = stride
        self.iou_bound = iou_bound
        self.labeled_num = labeled_num
        if con_list:
            self.con_list = con_list 
        else:
            self.con_list = range(1, self.num_class)
        self.addi_con_list = addi_con_list
        if weights:
            self.weights = [weights[i-1] for i in self.con_list ]
        else:
            self.weights = [1]*len(self.con_list)
        with open(img_list_file, 'r') as f:
            self.img_list = [img.replace("\n","") for img in f.readlines()]
        print(f"total img:{len(self.img_list)}")
        print(f"lower:{self.lower},upper:{self.upper}")
        print(f"do affine:{self.rotate_trans}, do cutout:{self.cutout}")
        print(f"condition list:{self.con_list}")

    def __getitem__(self, index):
        if len(self.img_list[index].strip().split()) > 1:
            img_path, mask_path = self.img_list[index].strip().split()
        else:
            img_path = self.img_list[index]
            mask_path = img_path.replace("img","label")
        if index < self.labeled_num:
            assert os.path.isfile(mask_path),f"invalid mask path: {mask_path},index:{index}!"

        """get task name"""
        _,img_name = os.path.split(img_path)
        task_name = "full"
        task_id = task_name_id_dict[task_name]
        
        """read image and mask"""
        if ".npy" in img_path:
            img_array = np.load(img_path).squeeze()
            if index < self.labeled_num:
                mask_array = np.load(mask_path).squeeze()
                mask_array[mask_array<0] = 0
            else:
                mask_array = np.zeros_like(img_array)           
        else:
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            if index < self.labeled_num: 
                mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            else:
                mask_array = np.zeros_like(img_array)
        mask_array = mask_array.astype(np.uint8)
        
        """ do random rotate and flip"""
        if self.random_rotflip:
            img_array, mask_array = random_rotate_flip(img_array, mask_array)
            
        """ padding image """
        img_array = pad_nd_image(img_array,self.patch_size)
        mask_array = pad_nd_image(mask_array,self.patch_size)
        # Clip pixel values outside the threshold range
        if ".npy" not in img_name:
            np.clip(img_array, self.lower, self.upper, out=img_array)
            """Normalize the image"""
            img_array = (img_array - img_array.mean())/img_array.std()
        shape = img_array.shape
        
        
        """ get one hot of mask"""
        mask_array = torch.FloatTensor(mask_array).unsqueeze(0)
        #print(f"unique value:{mask_array.unique()}")
        if task_id > 0 and task_id !=2:
            mask_array[mask_array!=0] = task_id
        if task_id == 2:
            mask_array[mask_array==2] = 3
            mask_array[mask_array==1] = 2
        #print(f"img name:{img_name},task_name:{task_name},task id:{task_id},unique value:{mask_array.unique()}")
        gt_onehot = torch.zeros((self.num_class, mask_array.shape[1], mask_array.shape[2],mask_array.shape[3]))
        gt_onehot.scatter_(0, mask_array.long(), 1)
        mask_array = gt_onehot     
        
        """get image patch"""
        ul1 = br1 = ul2 = br2 =  []
        lb_x = 0
        ub_x = shape[0]  - self.patch_size[0]
        lb_y = 0
        ub_y = shape[1]  - self.patch_size[1]
        lb_z = 0
        ub_z = shape[2]  - self.patch_size[2]


        bbox_x_lb1 = np.random.randint(lb_x, ub_x + 1)
        bbox_y_lb1 = np.random.randint(lb_y, ub_y + 1)
        bbox_z_lb1 = np.random.randint(lb_z, ub_z + 1)
        
        bbox_x_ub1 = bbox_x_lb1 + self.patch_size[0]
        bbox_y_ub1 = bbox_y_lb1 + self.patch_size[1]
        bbox_z_ub1 = bbox_z_lb1 + self.patch_size[2]
        max_iters = 50
        k = 0
        while k < max_iters:
            bbox_x_lb2 = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb2 = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb2 = np.random.randint(lb_z, ub_z + 1)
            # crop relative coordinates should be a multiple of 8
            bbox_x_lb2 = (bbox_x_lb2-bbox_x_lb1) // self.stride * self.stride + bbox_x_lb1
            bbox_y_lb2 = (bbox_y_lb2-bbox_y_lb1) // self.stride * self.stride + bbox_y_lb1
            bbox_z_lb2 = (bbox_z_lb2-bbox_z_lb1) // self.stride * self.stride + bbox_z_lb1
            if bbox_x_lb2 < 0: bbox_x_lb2 += self.stride
            if bbox_y_lb2 < 0: bbox_y_lb2 += self.stride
            if bbox_z_lb2 < 0: bbox_z_lb2 += self.stride
            if self.patch_size[0] - abs(bbox_x_lb2-bbox_x_lb1) < 0 or self.patch_size[1] - abs(bbox_y_lb2-bbox_y_lb1) < 0 or self.patch_size[2] - abs(bbox_z_lb2-bbox_z_lb1) < 0:
                k += 1
                continue
            inter = (self.patch_size[0] - abs(bbox_x_lb2-bbox_x_lb1)) * (self.patch_size[1] - abs(bbox_y_lb2-bbox_y_lb1)) * (self.patch_size[2] - abs(bbox_z_lb2-bbox_z_lb1))
            union = 2*self.patch_size[0]*self.patch_size[1]*self.patch_size[2] - inter
            iou = inter / union
            if iou >= self.iou_bound[0] and iou <= self.iou_bound[1]:
                break
            k+=1
        if k == max_iters:
            bbox_x_lb2  = bbox_x_lb1
            bbox_y_lb2  = bbox_y_lb1
            bbox_z_lb2  = bbox_z_lb1
        overlap1_ul = [max(0, bbox_x_lb2-bbox_x_lb1), max(0, bbox_y_lb2-bbox_y_lb1),max(0,bbox_z_lb2-bbox_z_lb1) ]
        overlap1_br = [min(self.patch_size[0], self.patch_size[0]+bbox_x_lb2-bbox_x_lb1, shape[0]//self.stride * self.stride), 
                        min(self.patch_size[1], self.patch_size[1]+bbox_y_lb2-bbox_y_lb1, shape[1]//self.stride * self.stride),
                        min(self.patch_size[2], self.patch_size[2]+bbox_z_lb2-bbox_z_lb1, shape[2]//self.stride * self.stride)]
        
        overlap2_ul = [max(0, bbox_x_lb1-bbox_x_lb2), max(0, bbox_y_lb1 - bbox_y_lb2),max(0,bbox_z_lb1- bbox_z_lb2) ]
        
        overlap2_br = [min(self.patch_size[0], self.patch_size[0]+bbox_x_lb1-bbox_x_lb2, shape[0]//self.stride * self.stride), 
                        min(self.patch_size[1], self.patch_size[1]+bbox_y_lb1-bbox_y_lb2, shape[1]//self.stride * self.stride),
                        min(self.patch_size[2], self.patch_size[2]+bbox_z_lb1-bbox_z_lb2, shape[2]//self.stride * self.stride)]
        try:
            assert (overlap1_br[0]-overlap1_ul[0]) * (overlap1_br[1]-overlap1_ul[1]) * (overlap1_br[2]-overlap1_ul[2]) == (overlap2_br[0]-overlap2_ul[0]) * (overlap2_br[1]-overlap2_ul[1]) * (overlap2_br[2]-overlap2_ul[2])
        except:
            print("x: {}, y: {}, z: {}".format(shape[0], shape[1], shape[2]))
            exit()
        bbox_x_ub2 = bbox_x_lb2 + self.patch_size[0]
        bbox_y_ub2 = bbox_y_lb2 + self.patch_size[1]
        bbox_z_ub2 = bbox_z_lb2 + self.patch_size[2]
        # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        # later
        valid_bbox_x_lb1 = max(0, bbox_x_lb1)
        valid_bbox_x_ub1 = min(shape[0], bbox_x_ub1)
        valid_bbox_y_lb1 = max(0, bbox_y_lb1)
        valid_bbox_y_ub1 = min(shape[1], bbox_y_ub1)
        valid_bbox_z_lb1 = max(0, bbox_z_lb1)
        valid_bbox_z_ub1 = min(shape[2], bbox_z_ub1)

        valid_bbox_x_lb2 = max(0, bbox_x_lb2)
        valid_bbox_x_ub2 = min(shape[0], bbox_x_ub2)
        valid_bbox_y_lb2 = max(0, bbox_y_lb2)
        valid_bbox_y_ub2 = min(shape[1], bbox_y_ub2)
        valid_bbox_z_lb2 = max(0, bbox_z_lb2)
        valid_bbox_z_ub2 = min(shape[2], bbox_z_ub2)
        img_array1 = img_array[np.newaxis,valid_bbox_x_lb1:valid_bbox_x_ub1,
                               valid_bbox_y_lb1:valid_bbox_y_ub1,
                               valid_bbox_z_lb1:valid_bbox_z_ub1, ]
        img_array2 = img_array[np.newaxis,valid_bbox_x_lb2:valid_bbox_x_ub2,
                        valid_bbox_y_lb2:valid_bbox_y_ub2,
                        valid_bbox_z_lb2:valid_bbox_z_ub2, ]
       
        mask_array1 = mask_array[np.newaxis,:,valid_bbox_x_lb1:valid_bbox_x_ub1,
                               valid_bbox_y_lb1:valid_bbox_y_ub1,
                               valid_bbox_z_lb1:valid_bbox_z_ub1, ]
        mask_array2 = mask_array[np.newaxis,:,valid_bbox_x_lb2:valid_bbox_x_ub2,
                        valid_bbox_y_lb2:valid_bbox_y_ub2,
                        valid_bbox_z_lb2:valid_bbox_z_ub2, ]
        
        mask_array1 = np.argmax(mask_array1, axis=1)
        mask_array2 = np.argmax(mask_array2, axis=1)
        if self.cutout and index< self.labeled_num:
            mask_array1 = random_cutout(mask_array1[0], size=(4,4,4))
            mask_array2 = random_cutout(mask_array2[0], size=(4,4,4))
        else:
            mask_array1 = mask_array1[0]
            mask_array2 = mask_array2[0]
            
        """ do affine transformation"""
        if (self.rotate_trans or self.scale_trans) and index < self.labeled_num:
            if self.rotate_trans:
                angle_x = random.uniform(-0.08,0.08)
                angle_y = random.uniform(-0.08,0.08)
                angle_z = random.uniform(-0.08,0.08)
            else:
                angle_x,angle_y,angle_z = 0.0,0.0,0.0
            
            if self.scale_trans:
                scale_x = random.uniform(0.8,1.2)
                scale_y = random.uniform(0.8,1.2)
                scale_z = random.uniform(0.8,1.2) 
            else: 
                scale_x,scale_y,scale_z = 1.0,1.0,1.0 
                
            img1 = affine_transformation(img_array1[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=3, border_mode="nearest", 
                                        constant_val=0, is_reverse=False)
            if index < self.labeled_num:
                mask1 = affine_transformation(mask_array1[np.newaxis,np.newaxis,:], 
                                            radius=(angle_x, angle_y, angle_z), 
                                            translate=(0, 0, 0),
                                            scale=(scale_x, scale_y, scale_z),
                                            bspline_order=0, border_mode="nearest",
                                            constant_val=0, is_reverse=False)
                mask_array1 = mask1[0,0,:]
            if self.rotate_trans:
                angle_x = random.uniform(-0.08,0.08)
                angle_y = random.uniform(-0.08,0.08)
                angle_z = random.uniform(-0.08,0.08)
            else:
                angle_x,angle_y,angle_z = 0.0,0.0,0.0
            
            if self.scale_trans:
                scale_x = random.uniform(0.8,1.2)
                scale_y = random.uniform(0.8,1.2)
                scale_z = random.uniform(0.8,1.2) 
            else: 
                scale_x,scale_y,scale_z = 1.0,1.0,1.0           
            img2 = affine_transformation(img_array2[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=3, border_mode="nearest", 
                                        constant_val=0, is_reverse=False)
            
            if index < self.labeled_num:
                mask2 = affine_transformation(mask_array2[np.newaxis,np.newaxis,:], 
                                            radius=(angle_x, angle_y, angle_z), 
                                            translate=(0, 0, 0),
                                            scale=(scale_x, scale_y, scale_z),
                                            bspline_order=0, border_mode="nearest",
                                            constant_val=0, is_reverse=False)
                mask_array2 = mask2[0,0,:]
            
            img_array1 = img1[0,:]
            img_array2 = img2[0,:]
        
        img_array = np.concatenate((img_array1, img_array2), axis=0)
        mask_array = np.concatenate((mask_array1.unsqueeze(0).unsqueeze(0), 
                                     mask_array2.unsqueeze(0).unsqueeze(0)), axis=0)

        
        ul1=overlap1_ul
        br1=overlap1_br
        ul2=overlap2_ul
        br2=overlap2_br


        # get condition list
        label_list = list(np.unique(mask_array1))
        if 0 in label_list:
            label_list.remove(0)
        inter_label_list = list(set(label_list) & set(self.con_list))
        if len(inter_label_list) == 0:
            inter_label_list = self.con_list
        condition1 = np.random.choice(inter_label_list)

        # get condition2 for all mask array2
        label_list = list(np.unique(mask_array2))
        if 0 in label_list:
            label_list.remove(0)
        inter_label_list = list(set(label_list) & set(self.con_list))
        if len(inter_label_list) == 0:
            inter_label_list = self.con_list
        # use num_classes as conditon label to predict foreground
        inter_label_list = inter_label_list + self.addi_con_list
        condition2 = np.random.choice(inter_label_list)


        img_array = torch.FloatTensor(img_array).unsqueeze(1)
        mask_array = torch.FloatTensor(mask_array).squeeze()
        condition_each_volume = torch.Tensor([condition1, condition2])
        sample = {'image': img_array, 'label': mask_array.long(), 'ul1': ul1, 
                  'br1': br1, 'ul2': ul2, 'br2': br2, 
                  'condition': condition_each_volume.long(),
                  'img_path':img_path,'task_id':task_id}
        return sample

    def __len__(self):
        return len(self.img_list)


class DatasetSR(dataset):
    def __init__(self, img_list_file, labeled_num=None, 
                 patch_size_large=(128, 256, 320),
                 patch_size_small=(96,160,160),
                 cutout=False, rotate_trans=False, scale_trans=False,
                 random_rotflip=False,
                 num_class=2, edge_prob=0., upper=200, lower=-68, 
                 stride=8, iou_bound=[0.25,0.95], con_list=None, 
                 addi_con_list=None, weights=None):
        self.patch_size_large = patch_size_large
        self.patch_size_small = patch_size_small
        self.cutout = cutout
        self.rotate_trans = rotate_trans
        self.scale_trans = scale_trans
        self.random_rotflip = random_rotflip
        self.num_class = num_class
        self.edge_prob = edge_prob
        self.upper = upper
        self.lower = lower
        self.stride = stride
        self.iou_bound = iou_bound
        self.labeled_num = labeled_num
        if con_list:
            self.con_list = con_list 
        else:
            self.con_list = range(1, self.num_class)
        self.addi_con_list = addi_con_list
        if weights:
            self.weights = [weights[i-1] for i in self.con_list ]
        else:
            self.weights = [1]*len(self.con_list)
        with open(img_list_file, 'r') as f:
            self.img_list = [img.replace("\n","") for img in f.readlines()]
        print(f"total img:{len(self.img_list)}")
        print(f"lower:{self.lower},upper:{self.upper}")
        print(f"do affine:{self.rotate_trans}, do cutout:{self.cutout}")
        print(f"condition list:{self.con_list}")

    def __getitem__(self, index):
        if len(self.img_list[index].strip().split()) > 1:
            img_path, mask_path = self.img_list[index].strip().split()
        else:
            img_path = self.img_list[index]
            mask_path = img_path.replace("img","label")
        if index < self.labeled_num:
            assert os.path.isfile(mask_path),f"invalid mask path: {mask_path},index:{index}!"

        """get task name"""
        _,img_name = os.path.split(img_path)
        """read image and mask"""
        if ".npy" in img_path:
            img_array = np.load(img_path).squeeze()
        else:
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        if index < self.labeled_num: 
            if ".npy" in mask_path:
                mask_array = np.load(mask_path).squeeze()
                mask_array[mask_array<0] = 0
            else:
                mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        else:
            mask_array = np.zeros_like(img_array)
        mask_array = mask_array.astype(np.uint8)
        
        """ do random rotate and flip"""
        if self.random_rotflip:
            img_array, mask_array = random_rotate_flip(img_array, mask_array)
            
        """ padding image """
        img_array = pad_nd_image(img_array,self.patch_size_large)
        mask_array = pad_nd_image(mask_array,self.patch_size_large)
        # Clip pixel values outside the threshold range
        if ".npy" not in img_name:
            img_array = np.clip(img_array, self.lower, self.upper)
            img_array = img_array.astype(np.float32)
            """Normalize the image"""
            img_array = (img_array - img_array.mean())/img_array.std()
        shape = img_array.shape
        
        
        """ get one hot of mask"""
        mask_array = torch.FloatTensor(mask_array).unsqueeze(0)
        gt_onehot = torch.zeros((self.num_class, mask_array.shape[1], mask_array.shape[2],mask_array.shape[3]))
        gt_onehot.scatter_(0, mask_array.long(), 1)
        mask_array = gt_onehot     
        
        """get image patch"""
        ul1 = br1 = ul2 = br2 =  []
        lb_x = 0
        ub_x_large = shape[0]  - self.patch_size_large[0]
        ub_x_small = shape[0]  - self.patch_size_small[0]
        lb_y = 0
        ub_y_large = shape[1]  - self.patch_size_large[1]
        ub_y_small = shape[1]  - self.patch_size_small[1]
        lb_z = 0
        ub_z_large = shape[2]  - self.patch_size_large[2]
        ub_z_small = shape[2]  - self.patch_size_small[2]


        bbox_x_lb1 = np.random.randint(lb_x, ub_x_large + 1)
        bbox_y_lb1 = np.random.randint(lb_y, ub_y_large + 1)
        bbox_z_lb1 = np.random.randint(lb_z, ub_z_large + 1)
        
        bbox_x_ub1 = bbox_x_lb1 + self.patch_size_large[0]
        bbox_y_ub1 = bbox_y_lb1 + self.patch_size_large[1]
        bbox_z_ub1 = bbox_z_lb1 + self.patch_size_large[2]
        max_iters = 50
        k = 0
        while k < max_iters:
            bbox_x_lb2 = np.random.randint(lb_x, ub_x_small + 1)
            bbox_y_lb2 = np.random.randint(lb_y, ub_y_small + 1)
            bbox_z_lb2 = np.random.randint(lb_z, ub_z_small+ 1)
            # crop relative coordinates should be a multiple of 8
            bbox_x_lb2 = (bbox_x_lb2-bbox_x_lb1) // self.stride * self.stride + bbox_x_lb1
            bbox_y_lb2 = (bbox_y_lb2-bbox_y_lb1) // self.stride * self.stride + bbox_y_lb1
            bbox_z_lb2 = (bbox_z_lb2-bbox_z_lb1) // self.stride * self.stride + bbox_z_lb1
            if bbox_x_lb2 < 0: bbox_x_lb2 += self.stride
            if bbox_y_lb2 < 0: bbox_y_lb2 += self.stride
            if bbox_z_lb2 < 0: bbox_z_lb2 += self.stride
            gap_x = bbox_x_lb2-bbox_x_lb1
            gap_y = bbox_y_lb2-bbox_y_lb1
            gap_z = bbox_z_lb2-bbox_z_lb1
            if ((gap_x + self.patch_size_small[0] <0) or (gap_x - self.patch_size_large[0] > 0)
                or (gap_y + self.patch_size_small[1] <0) or (gap_y - self.patch_size_large[1] > 0)
                or (gap_z + self.patch_size_small[2] <0) or (gap_z - self.patch_size_large[2] > 0)):
                k += 1
                continue
            
            overlap1_ul = [max(0, bbox_x_lb2-bbox_x_lb1), max(0, bbox_y_lb2-bbox_y_lb1),max(0,bbox_z_lb2-bbox_z_lb1) ]
            overlap1_br = [min(self.patch_size_large[0], self.patch_size_small[0]+bbox_x_lb2-bbox_x_lb1, shape[0]//self.stride * self.stride), 
                        min(self.patch_size_large[1], self.patch_size_small[1]+bbox_y_lb2-bbox_y_lb1, shape[1]//self.stride * self.stride),
                        min(self.patch_size_large[2], self.patch_size_small[2]+bbox_z_lb2-bbox_z_lb1, shape[2]//self.stride * self.stride)]
            
            inter = (overlap1_br[0]-overlap1_ul[0])*(overlap1_br[1]-overlap1_ul[1])*(overlap1_br[2]-overlap1_ul[2])
            #inter = (self.patch_size_large[0] - abs(bbox_x_lb2-bbox_x_lb1)) * (self.patch_size_large[1] - abs(bbox_y_lb2-bbox_y_lb1)) * (self.patch_size_large[2] - abs(bbox_z_lb2-bbox_z_lb1))
            iou = inter / (self.patch_size_small[0]*self.patch_size_small[1]*self.patch_size_small[2])
            if iou >= self.iou_bound[0] and iou <= self.iou_bound[1]:
                break
            k+=1
        if k == max_iters:
            print(f"k:{k}")
            bbox_x_lb2  = bbox_x_lb1
            bbox_y_lb2  = bbox_y_lb1
            bbox_z_lb2  = bbox_z_lb1
        
        overlap1_ul = [max(0, bbox_x_lb2-bbox_x_lb1), max(0, bbox_y_lb2-bbox_y_lb1),max(0,bbox_z_lb2-bbox_z_lb1) ]
        overlap1_br = [min(self.patch_size_large[0], self.patch_size_small[0]+bbox_x_lb2-bbox_x_lb1, shape[0]//self.stride * self.stride), 
                        min(self.patch_size_large[1], self.patch_size_small[1]+bbox_y_lb2-bbox_y_lb1, shape[1]//self.stride * self.stride),
                        min(self.patch_size_large[2], self.patch_size_small[2]+bbox_z_lb2-bbox_z_lb1, shape[2]//self.stride * self.stride)]
        
        overlap2_ul = [max(0, bbox_x_lb1-bbox_x_lb2), max(0, bbox_y_lb1 - bbox_y_lb2),max(0,bbox_z_lb1- bbox_z_lb2) ]
        
        overlap2_br = [min(self.patch_size_small[0], self.patch_size_large[0]+bbox_x_lb1-bbox_x_lb2, shape[0]//self.stride * self.stride), 
                       min(self.patch_size_small[1], self.patch_size_large[1]+bbox_y_lb1-bbox_y_lb2, shape[1]//self.stride * self.stride),
                       min(self.patch_size_small[2], self.patch_size_large[2]+bbox_z_lb1-bbox_z_lb2, shape[2]//self.stride * self.stride)]
        
        if overlap1_br[1] < overlap1_ul[1]:
            print("error")
        try:
            assert (overlap1_br[0]-overlap1_ul[0]) * (overlap1_br[1]-overlap1_ul[1]) * (overlap1_br[2]-overlap1_ul[2]) == (overlap2_br[0]-overlap2_ul[0]) * (overlap2_br[1]-overlap2_ul[1]) * (overlap2_br[2]-overlap2_ul[2])
        except:
            print(f"ul1:{overlap1_ul},br1:{overlap1_br},ul2:{overlap2_ul},br2:{overlap2_br}")
            print(f"bbox_x_lb1:{bbox_x_lb1},bbox_x_lb2:{bbox_x_lb2}")
            print(f"bbox_y_lb1:{bbox_y_lb1},bbox_y_lb2:{bbox_y_lb2}")
            print(f"bbox_z_lb1:{bbox_z_lb1},bbox_z_lb2:{bbox_z_lb2}")
            print("x: {}, y: {}, z: {}".format(shape[0], shape[1], shape[2]))
            exit()

        # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        # later
        bbox_x_ub2 = bbox_x_lb2 + self.patch_size_small[0]
        bbox_y_ub2 = bbox_y_lb2 + self.patch_size_small[1]
        bbox_z_ub2 = bbox_z_lb2 + self.patch_size_small[2]
        valid_bbox_x_lb1 = max(0, bbox_x_lb1)
        valid_bbox_x_ub1 = min(shape[0], bbox_x_ub1)
        valid_bbox_y_lb1 = max(0, bbox_y_lb1)
        valid_bbox_y_ub1 = min(shape[1], bbox_y_ub1)
        valid_bbox_z_lb1 = max(0, bbox_z_lb1)
        valid_bbox_z_ub1 = min(shape[2], bbox_z_ub1)

        valid_bbox_x_lb2 = max(0, bbox_x_lb2)
        valid_bbox_x_ub2 = min(shape[0], bbox_x_ub2)
        valid_bbox_y_lb2 = max(0, bbox_y_lb2)
        valid_bbox_y_ub2 = min(shape[1], bbox_y_ub2)
        valid_bbox_z_lb2 = max(0, bbox_z_lb2)
        valid_bbox_z_ub2 = min(shape[2], bbox_z_ub2)
        img_array1 = img_array[np.newaxis,valid_bbox_x_lb1:valid_bbox_x_ub1,
                               valid_bbox_y_lb1:valid_bbox_y_ub1,
                               valid_bbox_z_lb1:valid_bbox_z_ub1, ]
        img_array2 = img_array[np.newaxis,valid_bbox_x_lb2:valid_bbox_x_ub2,
                        valid_bbox_y_lb2:valid_bbox_y_ub2,
                        valid_bbox_z_lb2:valid_bbox_z_ub2, ]
       
        mask_array1 = mask_array[np.newaxis,:,valid_bbox_x_lb1:valid_bbox_x_ub1,
                               valid_bbox_y_lb1:valid_bbox_y_ub1,
                               valid_bbox_z_lb1:valid_bbox_z_ub1, ]
        mask_array2 = mask_array[np.newaxis,:,valid_bbox_x_lb2:valid_bbox_x_ub2,
                        valid_bbox_y_lb2:valid_bbox_y_ub2,
                        valid_bbox_z_lb2:valid_bbox_z_ub2, ]
        
        mask_array1 = np.argmax(mask_array1, axis=1)
        mask_array2 = np.argmax(mask_array2, axis=1)
        if self.cutout and index< self.labeled_num:
            mask_array1 = random_cutout(mask_array1[0], size=(4,4,4))
            mask_array2 = random_cutout(mask_array2[0], size=(4,4,4))
        else:
            mask_array1 = mask_array1[0]
            mask_array2 = mask_array2[0]
            
        """ do affine transformation"""
        if (self.rotate_trans or self.scale_trans) and index < self.labeled_num:
            if self.rotate_trans:
                angle_x = random.uniform(-0.08,0.08)
                angle_y = random.uniform(-0.08,0.08)
                angle_z = random.uniform(-0.08,0.08)
            else:
                angle_x,angle_y,angle_z = 0.0,0.0,0.0
            
            if self.scale_trans:
                scale_x = random.uniform(0.8,1.2)
                scale_y = random.uniform(0.8,1.2)
                scale_z = random.uniform(0.8,1.2) 
            else: 
                scale_x,scale_y,scale_z = 1.0,1.0,1.0 
                
            img1 = affine_transformation(img_array1[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=3, border_mode="nearest", 
                                        constant_val=0, is_reverse=False)
            if index < self.labeled_num:
                mask1 = affine_transformation(mask_array1[np.newaxis,np.newaxis,:], 
                                            radius=(angle_x, angle_y, angle_z), 
                                            translate=(0, 0, 0),
                                            scale=(scale_x, scale_y, scale_z),
                                            bspline_order=0, border_mode="nearest",
                                            constant_val=0, is_reverse=False)
                mask_array1 = mask1[0,0,:]
            if self.rotate_trans:
                angle_x = random.uniform(-0.08,0.08)
                angle_y = random.uniform(-0.08,0.08)
                angle_z = random.uniform(-0.08,0.08)
            else:
                angle_x,angle_y,angle_z = 0.0,0.0,0.0
            
            if self.scale_trans:
                scale_x = random.uniform(0.8,1.2)
                scale_y = random.uniform(0.8,1.2)
                scale_z = random.uniform(0.8,1.2) 
            else: 
                scale_x,scale_y,scale_z = 1.0,1.0,1.0           
            img2 = affine_transformation(img_array2[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=3, border_mode="nearest", 
                                        constant_val=0, is_reverse=False)
            
            if index < self.labeled_num:
                mask2 = affine_transformation(mask_array2[np.newaxis,np.newaxis,:], 
                                            radius=(angle_x, angle_y, angle_z), 
                                            translate=(0, 0, 0),
                                            scale=(scale_x, scale_y, scale_z),
                                            bspline_order=0, border_mode="nearest",
                                            constant_val=0, is_reverse=False)
                mask_array2 = mask2[0,0,:]
            
            img_array1 = img1[0,:]
            img_array2 = img2[0,:]
        
        img_array1 = resize3d(img_array1[0],tuple(self.patch_size_small))[np.newaxis,:]

        
        ul1=overlap1_ul
        br1=overlap1_br
        ul2=overlap2_ul
        br2=overlap2_br


        sample = {'image_large': img_array1, 'label_large': mask_array1.long(),
                  'image_small': img_array2, 'label_small': mask_array2.long(),
                  'ul1': ul1, 
                  'br1': br1, 'ul2': ul2, 'br2': br2, 
                  'img_path':img_path}
        return sample

    def __len__(self):
        return len(self.img_list)


def resize3d(image, shape, mode='trilinear', data_type='numpy'):
    """
        resize 3d image
    """
    if data_type == 'numpy':
        image = torch.tensor(image)[None,None,:,:,:]
    image = F.interpolate(torch.as_tensor(image), size=shape, mode=mode)
    image = image[0,0,:,:,:].numpy()
    return image


if __name__ == '__main__':
    """
    test dataset class 
    """
    train_file_list = "../../data/BCV/train_cvcl.txt"
    # db_train = BCVDataset(img_list_file=train_file_list,
    #                         patch_size=(96,160,160),
    #                         labeled_num=4,
    #                         num_class=6)
    
    db_train = BCVDatasetCAC(
                img_list_file=train_file_list, 
                patch_size=(96,160,160), 
                num_class=6, 
                stride=8, 
                iou_bound=[0.3,0.95],
                labeled_num=500,
                cutout=True,
                rotate_trans=True,
                con_list=[2,3,5],
                addi_con_list=[8],
                weights=[0.2,0.2,0.2,0.1,0.3]
            )
    # db_train = DatasetSR(
    #             img_list_file=train_file_list, 
    #             num_class=6, 
    #             stride=8, 
    #             iou_bound=[0.2,1.01],
    #             labeled_num=4,
    #             cutout=True,
    #             rotate_trans=True,
    #             con_list=[2,3,5],
    #             addi_con_list=[8],
    #             weights=[0.2,0.2,0.2,0.1,0.3]
    # )
    dataloader = DataLoader(db_train, 
                                batch_sampler = BatchSampler(
                                    ClassRandomSampler(db_train), 
                                    2, True), 
                                num_workers=2, pin_memory=True)
    con_list =[0]*5
    con_vol_list = [0]*5
    for data_batch in tqdm(dataloader):
        task_id = data_batch['task_id']
        img_path = data_batch['img_path']
        print(f'task_id:{task_id},image path:{img_path}')
        continue
        #img_path, = data_batch['img_path'], 
        #image_large,label_large = data_batch['image_large'], data_batch['label_large']
        #image_small,label_small = data_batch['image_small'], data_batch['label_small']
        #condition = data_batch['condition']
        #print(f"con shape:{condition.shape}")
        # print(f"label large shape:{label_large.shape}")
        # print(f"label small shape:{label_small.shape}")
        # print(f"image large shape:{image_large.shape}")
        # print(f"image small shape:{image_small.shape}")
        # ul1,br1,ul2,br2 = data_batch['ul1'], data_batch['br1'], data_batch['ul2'], data_batch['br2']
        # target1_overlap = label_large[ ul1[0]:br1[0], ul1[1]:br1[1], ul1[2]:br1[2]]
        # target2_overlap = label_small[ ul2[0]:br2[0], ul2[1]:br2[1], ul2[2]:br2[2]]
        #assert (target1_overlap!=target2_overlap).sum() == 0,"error"
        # print("image shape:", image.shape)
        # print("conditon:",condition)
        # print("condition volume:",condition_volume)
        # for con in condition:
        #     print("con:",con)
        #     con_list[con.item()-1]+=1
    
    # print("con list:",con_list)
    # print("con volume list:", con_vol_list)
        #print("ul1:{}, br1:{}, ul2:{}, br2:{}".format(ul1, br1, ul2, br2))
        #print(label.dtype)