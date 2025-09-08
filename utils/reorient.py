import SimpleITK as sitk

# 读取目标图像（作为坐标系参考，比如原始CT/MRI）
reference_image = sitk.ReadImage("/projects/weilab/liupeng/dataset/ssl_seg/Task012_Heart/labelsTr/heart_12_crop.nii.gz")

# 读取待对齐的图像（比如分割结果）
moving_image = sitk.ReadImage("/projects/weilab/liupeng/semi-seg/model/MMWHS_4_CSSR_test_SGD_SGD/unet_3D_old/Prediction_full_monai/heart_12_0000_crop_pred.nii.gz")

# 重采样
# 将参考图像的几何信息赋给目标图像
moving_image.SetOrigin(reference_image.GetOrigin())
moving_image.SetSpacing(reference_image.GetSpacing())
moving_image.SetDirection(reference_image.GetDirection())

# 保存为新的 NIfTI 文件
sitk.WriteImage(moving_image, "segmentation_aligned.nii.gz")
