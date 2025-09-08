# processing the dataset
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def save_array_to_nii(image_arr, direction, origin, spacing, save_name):
    """
        save numpy array to sitk nii file

        Parameters:
            image_arr: image array need to be saved
            direction: sitk direction
            origin:
            spacing:
            save_name: save name for sitk image
    """
    image_itk = sitk.GetImageFromArray(image_arr)
    image_itk.SetOrigin(origin)
    image_itk.SetSpacing(spacing)
    image_itk.SetDirection(direction)
    sitk.WriteImage(image_itk, save_name)


def crop_image_by_mask(img_file):
    """
        use mask to crop the image
    """
    with open(img_file, 'r') as f:
        img_list = [img.replace("\n", "") for img in f.readlines()]
    min_w_list = []
    max_w_list = []
    min_d_list = []
    max_d_list = []
    for img_mask_path in tqdm(img_list):
        # img_path, mask_path = img_mask_path.strip().split()
        if len(img_mask_path.strip().split()) > 1:
            img_path, mask_path = img_mask_path.strip().split()
        else:
            img_path = img_mask_path
            mask_path = img_path.replace('img', 'label')
        img_path = img_path.replace("_crop", "")
        mask_path = mask_path.replace("_crop", "")
        img_itk = sitk.ReadImage(img_path)
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        origin = img_itk.GetOrigin()
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image_np = sitk.GetArrayFromImage(img_itk)
        print(f"=====>processing:{img_path},shape:{image_np.shape}")
        h, w, d = mask_np.shape
        boud_h, boud_w, boud_d = np.where(mask_np >= 1)
        print(boud_h)
        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        min_w_list.append(bbx_w_min)
        max_w_list.append(bbx_w_max)
        min_d_list.append(bbx_d_min)
        max_d_list.append(bbx_d_max)
        print(f"h min: {bbx_h_min},h max: {bbx_h_max}")
        print(f"w min: {bbx_w_min},w max: {bbx_w_max}")
        print(f"d min: {bbx_d_min},d max: {bbx_d_max}")
        bbx_mask = mask_np[
                   max(bbx_h_min, 0):min(bbx_h_max, h),
                   16:464,  # max(bbx_w_min-60,0):min(bbx_w_max+60,w),
                   5:465  # max(bbx_d_min-50,0):min(bbx_d_max+50,d)
                   ]
        bbx_image = image_np[
                    max(bbx_h_min, 0):min(bbx_h_max, h),
                    16:464,  # max(bbx_w_min-60,0):min(bbx_w_max+60,w),
                    5:465  # max(bbx_d_min-50,0):min(bbx_d_max+50,d)
                    ]
        print(
            f"before crop shape:{image_np.shape},shape after crop: {bbx_image.shape}")
        # save_array_to_nii(bbx_mask,direction,origin,spacing,
        #                   mask_path.replace(".nii.gz","_crop_h.nii.gz"))

        # save_array_to_nii(bbx_image,direction,origin,spacing,
        #                   img_path.replace(".nii.gz","_crop_h.nii.gz"))
    print(f"min w:{min(min_w_list)} max w:{max(max_w_list)}")
    print(f"min d:{min(min_d_list)} max d:{max(max_d_list)}")


def main(img_file):
    with open(img_file, 'r') as f:
        img_list = [img.replace("\n", "") for img in f.readlines()]
    for img_mask_path in tqdm(img_list):
        # img_path, mask_path = img_mask_path.strip().split()
        if len(img_mask_path.strip().split()) > 1:
            img_path, mask_path = img_mask_path.strip().split()
        else:
            img_path = img_mask_path
            mask_path = img_path.replace('img', 'label')
        print(f"=====>processing:{img_path}")
        img_itk = sitk.ReadImage(img_path)
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        origin = img_itk.GetOrigin()
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image_np = sitk.GetArrayFromImage(img_itk)
        h, w, d = mask_np.shape
        boud_h, boud_w, boud_d = np.where(mask_np >= 1)
        print(boud_h)
        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        bbx_mask = mask_np[
                   bbx_h_min:bbx_h_max,
                   # max(bbx_h_min-25,0):min(bbx_h_max+25,h),
                   max(bbx_w_min - 60, 0):min(bbx_w_max + 60, w),
                   max(bbx_d_min - 50, 0):min(bbx_d_max + 50, d)
                   ]
        bbx_image = image_np[
                    bbx_h_min:bbx_h_max,
                    # max(bbx_h_min-25,0):min(bbx_h_max+25,h),
                    max(bbx_w_min - 60, 0):min(bbx_w_max + 60, w),
                    max(bbx_d_min - 50, 0):min(bbx_d_max + 50, d)
                    ]
        bbx_mask = np.flip(bbx_mask, axis=1)
        bbx_image = np.flip(bbx_image, axis=1)
        assert bbx_mask.shape == bbx_image.shape, "shape must equal after crop"
        print(
            f"before crop shape:{mask_np.shape}, after crop shape:{bbx_mask.shape}")
        bbx_mask_itk = sitk.GetImageFromArray(bbx_mask)
        bbx_mask_itk.SetOrigin(origin)
        bbx_mask_itk.SetSpacing(spacing)
        bbx_mask_itk.SetDirection(direction)
        sitk.WriteImage(bbx_mask_itk,
                        mask_path.replace(".nii.gz", "_crop.nii.gz"))

        bbx_image_itk = sitk.GetImageFromArray(bbx_image)
        bbx_image_itk.SetOrigin(origin)
        bbx_image_itk.SetSpacing(spacing)
        bbx_image_itk.SetDirection(direction)
        sitk.WriteImage(bbx_image_itk,
                        img_path.replace(".nii.gz", "_crop.nii.gz"))


if __name__ == "__main__":
    print("test")
    img_file = "/data1/liupeng/semi-supervised_segmentation/"
               "SSL4MIS-master/data/MMWHS/MMWHS_train.txt"
    # main(img_file)
    crop_image_by_mask(img_file)
