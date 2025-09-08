import os
import math
import random
from glob import glob

import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import (
    load_pickle, join)
from medpy import metric
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImage, ToTensor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from tqdm import tqdm


def test_single_case(net, image, stride_xy, stride_z, patch_size,
                     num_classes=1,
                     condition=-1, do_SR=False, method='regular',
                     return_scoremap=False):
    w, h, d = image.shape
    print("do SR: ", do_SR)

    # if the size of image is less than patch_size, then padding it
    device = next(net.parameters()).device
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print(f"img shape:{image.shape}, sx:{sx}, sy:{sy}, sz:{sz}")
    if condition > 0:
        score_map = np.zeros((2,) + image.shape).astype(np.float32)
    else:
        score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                                   ys:ys + patch_size[1],
                                   zs:zs + patch_size[2]]

                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                if do_SR:
                    test_patch = F.interpolate(
                        torch.as_tensor(test_patch),
                        size=(96, 160, 160),
                        mode='trilinear'
                    ).to(device)
                else:
                    test_patch = torch.from_numpy(test_patch).to(device)

                with torch.no_grad():
                    if condition > 0:
                        condition = torch.tensor([condition],
                                                 dtype=torch.long,
                                                 device=device)
                        pred1 = net(test_patch, condition)
                    else:
                        pred1 = net(test_patch)
                        if len(pred1) > 0 and isinstance(pred1, (tuple, list)):
                            pred1 = pred1[0]
                    # ensemble
                    pred = torch.softmax(pred1, dim=1)
                pred = pred.cpu().data.numpy()
                pred = pred[0, :, :, :, :]
                score_map[
                :, 
                xs:xs + patch_size[0],
                ys:ys + patch_size[1],
                zs:zs + patch_size[2]
                ] += pred
                cnt[
                    xs:xs + patch_size[0],
                    ys:ys + patch_size[1],
                    zs:zs + patch_size[2]
                ] += 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                              hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad +
                                        w, hl_pad:hl_pad + h,
                              dl_pad:dl_pad + d]

    if return_scoremap:
        return label_map, score_map
    else:
        return label_map


def process_fn(seg_prob_tuple, window_data, importance_map_):
    """seg_prob_tuple, importance_map =
    process_fn(seg_prob_tuple, window_data, importance_map_)
    """
    if len(seg_prob_tuple) > 0 and isinstance(seg_prob_tuple, (tuple, list)):
        seg_prob = torch.softmax(seg_prob_tuple[0], dim=1)
        return tuple(
            seg_prob.unsqueeze(0)) + seg_prob_tuple[1:], importance_map_
    else:
        seg_prob = torch.softmax(seg_prob_tuple, dim=1)
        return seg_prob, importance_map_


def test_single_case_monai(net, image, patch_size, overlap=0.5,
                           batch_size=2,
                           do_SR=False):
    device = next(net.parameters()).device
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    if do_SR:
        image = F.interpolate(image, size=patch_size, mode='trilinear')
    with torch.no_grad():
        prediction = sliding_window_inference(
            image.float(), patch_size, batch_size, net, overlap=overlap,
            mode='gaussian', process_fn=process_fn
        )
        if len(prediction) > 1:
            prediction = prediction[0].cpu().numpy().squeeze()
        else:
            prediction = prediction.cpu().numpy().squeeze()
    label_map = np.argmax(prediction, axis=0)
    return label_map


def calculate_metric(gt, pred, cal_hd95=False, cal_asd=False):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # Add Jaccard metric calculation
        jc = metric.binary.jc(pred, gt)
        if cal_hd95:
            hd95 = metric.binary.hd95(pred, gt)
        else:
            hd95 = 0.0
        if cal_asd:
            asd = metric.binary.asd(pred, gt)
        else:
            asd = 0.0
        # Organize the array in the order of Dice, ASD, HD95, Jaccard
        return np.array([dice, asd, hd95, jc])
    else:
        # Update the default value accordingly
        return np.array([0.0, 150, 150, 0.0])


def test_all_case(net, test_list="full_test.list", num_classes=4,
                  patch_size=(48, 160, 160),
                  batch_size=2,
                  stride_xy=32, stride_z=24, overlap=0.5,
                  do_condition=False, do_SR=False, method="regular",
                  cal_metric=True,
                  save_prediction=False,
                  prediction_save_path='./',
                  test_num=2,
                  cut_upper=200,
                  cut_lower=-68,
                  con_list=None,
                  normalization='Zscore',
                  test_all_cases=False):
    if os.path.isdir(test_list):
        image_list = glob(test_list + "*.nii.gz")
    else:
        with open(test_list, 'r') as f:
            image_list = [img.replace('\n', '') for img in f.readlines()]
    print("Total test images:", len(image_list))
    if con_list:
        total_metric = np.zeros((len(con_list), 4))
    else:
        total_metric = np.zeros((num_classes - 1, 4))
    print("Validation begin")
    if con_list:
        condition_list = con_list  # for condition learning
    else:
        condition_list = [i for i in range(1, num_classes)]
    # shuffle(condition_list)
    random.shuffle(image_list)
    if not do_condition:
        test_num = len(image_list)
    if "Flare" in test_list:
        # for flare data, randomly select 10 cases for validatation
        test_num = 20
        if "fullres" in test_list:
            test_num = 5
        if do_condition:
            test_num = 6
    if test_all_cases:
        test_num = len(image_list)
    for i, image_path in enumerate(tqdm(image_list)):
        if i > test_num - 1:
            break
        if len(image_path.strip().split()) > 1:
            image_path, mask_path = image_path.strip().split()
        else:
            mask_path = image_path.replace('img', 'label')
        if cal_metric:
            assert os.path.isfile(mask_path), "invalid mask path error"
        if ".npy" in image_path:
            image = np.load(image_path).squeeze()
        else:
            image_sitk = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(image_sitk)

        if cal_metric:  # whether calculate metrics
            if ".npy" in mask_path:
                label = np.load(mask_path).squeeze()
                label[label < 0] = 0
            else:
                label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        else:
            label = np.zeros_like(image)
        if ".npy" not in image_path:
            if "heartMR" in image_path or normalization == 'MinMax':
                min_val_1p = np.percentile(image, 1)
                max_val_99p = np.percentile(image, 99)
                # min-max norm on total 3D volume
                print('min max norm')
                image = (image - min_val_1p) / (max_val_99p - min_val_1p)
                image = np.clip(image, 0.0, 1.0)
            else:
                image = np.clip(image, cut_lower, cut_upper)
                image = (image - image.mean()) / image.std()
        if do_condition:
            print(f"===>test image:{image_path}")
            for condition in condition_list:
                prediction = test_single_case(
                    net, image, stride_xy, stride_z, patch_size,
                    num_classes=2, condition=condition, method=method)
                if cal_metric:
                    if condition < num_classes:
                        metric = calculate_metric(label == condition,
                                                  prediction, cal_hd95=True,
                                                  cal_asd=True)
                    else:
                        metric = calculate_metric(label > 0, prediction,
                                                  cal_hd95=True,
                                                  cal_asd=True)
                    print(f"condition:{condition}, metric:{metric}")
                    total_metric[condition - 1, :] += metric
        else:
            if do_SR:
                prediction = test_single_case(
                    net, image, stride_xy, stride_z, patch_size,
                    num_classes=num_classes, condition=-1, do_SR=do_SR,
                    method=method)
            else:
                prediction = test_single_case_monai(net=net, image=image,
                                                    patch_size=patch_size,
                                                    batch_size=batch_size,
                                                    overlap=overlap)
            if cal_metric:
                for i in range(1, num_classes):
                    total_metric[i - 1, :] += calculate_metric(
                        gt=(label == i).astype(np.int32),
                        pred=(prediction == i).astype(np.int32),
                        cal_hd95=True,
                        cal_asd=True)

        # save prediction
        if save_prediction:
            _, image_name = os.path.split(image_path)

            if ".npy" not in image_path:
                spacing = image_sitk.GetSpacing()
                pred_itk = sitk.GetImageFromArray(
                    prediction.astype(np.uint8))
                pred_itk.SetSpacing(spacing)
                sitk.WriteImage(pred_itk, prediction_save_path +
                                image_name.replace(".nii.gz", "_pred.nii.gz"))
            else:
                save_name = join(prediction_save_path,
                                 image_name.replace(".npy", "_pred.nii.gz"))
                image_reader_writer = SimpleITKIO()
                properties = load_pickle(image_path.replace(".npy", ".pkl"))
                image_reader_writer.write_seg(prediction, save_name,
                                              properties)

    print("Validation end")
    if con_list:
        return total_metric[[con - 1 for con in con_list]] / test_num
    else:
        return total_metric / test_num
