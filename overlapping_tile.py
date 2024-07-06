"""
统一符号语言：
    分割窗口： B Hp Wp C Np
    输入图像： B C H W

Next:
    统一注释
    每一个函数给出注释，方便复用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import math
from einops import rearrange
from timm.models.layers import to_2tuple

PI = math.pi
PI_2 = math.pi * 0.5
PI2 = math.pi * 2

def same_padding(rows, cols, ksizes, strides, rates):
    """
    Args:
        rows: int
        cols: int
        ksizes: tuple (Kh, Kw)
        strides: tuple (Sh, Sw)
        rates: tuple (1, 1)
    """   
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    return paddings

def partion_overlapping_window(img_token, window_size, overlap_ratio, input_order="CHW", padding_mode='reflect'):
    """
    Args:
        img: (B, C, H, W)
        input_order: "CHW" or "HWC", default input is image tensor
    Return:
        img_patch: (B, C*Hp*Wp, Np)
        paddings: tuple (left, right, top, bottom)
    """
    if input_order == "HWC":
        img = rearrange(img_token, "B H W C -> B C H W")
    else:
        img = img_token
    h,w = img.shape[2:4]
    window_size = to_2tuple(window_size)
    strides = int(window_size[0] * (1-overlap_ratio)), int(window_size[1] * (1-overlap_ratio))
    paddings = same_padding(rows=h, cols=w, ksizes=window_size, strides=strides, rates=(1,1))
    img_pad = F.pad(img, paddings, mode=padding_mode)  # mode='reflect' mode="constant"
    img_patch = F.unfold(img_pad, kernel_size=window_size,  # (B, C*Hp*Wp, Np)
                                padding=0,
                                stride=strides)

    return img_patch, paddings

def reverse_overlapping_window(img_patch, origin_img_shape, paddings, window_size, overlap_ratio, padding_mode="constant"):
    """
    Args:
        img_patch: [B, C*Hp*Wp, Np]
        origin_img_shape: [B, C, H, W]
    Return:
        recovered_img: [B, C, H, W]
    """
    window_size = to_2tuple(window_size)
    strides = int(window_size[0] * (1-overlap_ratio)), int(window_size[1] * (1-overlap_ratio))
    if paddings is None:
        paddings = same_padding(rows=origin_img_shape[2], cols=origin_img_shape[3], ksizes=window_size, strides=strides, rates=(1,1))
    
    mask = torch.ones(origin_img_shape, dtype=torch.float32, device=img_patch.device)
    # print(f'mask:{mask.size()}')

    mask_pad = F.pad(mask, paddings, mode='constant')
    mask_patch = F.unfold(mask_pad, kernel_size=window_size,
                                    padding=0,
                                    stride=strides)
    

    mask_fold = F.fold(mask_patch, output_size=(mask_pad.shape[2],mask_pad.shape[3]), kernel_size=window_size, 
                                    padding=0, 
                                    stride=strides)
                                    
    img_fold = F.fold(img_patch, output_size=(mask_pad.shape[2],mask_pad.shape[3]), kernel_size=window_size, 
                                padding=0, 
                                stride=strides)
    
    padding_left, padding_right, padding_top, padding_bottom = paddings

    # print(f'img_fold: {img_fold.size()}')
    # print(f'mask_fold: {mask_fold.size()}')
    img_unpad = img_fold[:, :, padding_top:img_fold.shape[2]-padding_bottom, padding_left:img_fold.shape[3]-padding_right]
    mask_unpad = mask_fold[:, :, padding_top:mask_fold.shape[2]-padding_bottom, padding_left:mask_fold.shape[3]-padding_right]

    # print(f'img_unpad: {img_unpad.size()}')
    # print(f'mask_unpad: {mask_unpad.size()}')
    return img_unpad / mask_unpad



if __name__ == "__main__":
    import cv2
    import numpy as np
    import torchvision.utils as tvu
    from basicsr.utils import img2tensor,tensor2img

    def save_img_from_pt(img_pt, img_name):
        """
            img_pt: of shape (B, C, H, W)
        """
        img_np = img_pt[0].permute(1, 2, 0).numpy()
        img_np = img_np * 255
        cv2.imwrite(f'{img_name}.jpg', img_np.astype(np.uint8))

    img_np = cv2.imread('ERP_GT_0000.png', cv2.IMREAD_COLOR)
    img_tensor = img2tensor(img_np/255., bgr2rgb=True, float32=True).unsqueeze(0)
    # B = 1
    # img_new = img.astype(np.float32) / 255
    # img_new = np.transpose(img_new, [2, 0, 1])
    # img_tensor = torch.from_numpy(img_new)
    # img_tensor = img_tensor.unsqueeze(0).repeat(B, 1, 1, 1)
    B, C ,H, W = img_tensor.shape 

    # 超参数
    overlap_ratio = 0.2  # 0 表示没有重叠
    window_size = (256, 1024)

    img_window, _paddings = partion_overlapping_window(img_tensor, window_size, overlap_ratio)  # input x (B, C, H, W) ,output (B, C*Hp*Wp, Np)
    print(f"num of patch: {img_window.shape[-1]}")

    # img_window.shape:  (B, C*Hp*Wp, Np)
    visual_img_window =  rearrange(img_window, 'B (C Hp Wp) Np -> (B Np) C Hp Wp', C=C, Hp=window_size[0], Np=img_window.shape[-1])
    tvu.save_image(visual_img_window, "tiled_img.png")

    
    recovered_img = reverse_overlapping_window(img_window, (B, C, H, W), _paddings, window_size, overlap_ratio)
    


