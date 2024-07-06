import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY

# from .ws_utils.common import *


@METRIC_REGISTRY.register()
def my_calculate_ws_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """
    Calculate WS_PSNR (Peak Signal-to-Noise Ratio).
    Args:
        img  (ndarray): Images with range [0, 255], [H,W,C], BGR
        img2 (ndarray): Images with range [0, 255], [H,W,C], BGR
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = (img - img2) ** 2  # HWC

    spherical_weight = make_spherical_weight(mse.shape)

    if spherical_weight.shape == mse.shape:
        ws_mse = np.sum(spherical_weight * mse) / np.sum(spherical_weight)
    else:
        assert spherical_weight.shape == mse.shape, (
            f'weights and loss shapes are different: {spherical_weight.shape}, {mse.shape}.')

    try:
        ws_psnr = 10. * np.log10(255. * 255. / ws_mse)  # psnr 10. * np.log10(255. * 255. / mse)
    except ZeroDivisionError:
        ws_psnr = np.inf

    return ws_psnr


@METRIC_REGISTRY.register()
def my_calculate_ws_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255], [H,W,C], BGR
        img2 (ndarray): Images with range [0, 255], [H,W,C], BGR
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ws_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def _ws_ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ws_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'. single channel
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    spherical_weight = make_spherical_weight(img.shape)[5:-5, 5:-5]
    assert spherical_weight.shape == ssim_map.shape, (
        f'weights and ssim_map shapes are different: {spherical_weight.shape}, {ssim_map.shape}.')
    ws_ssim_map = np.sum(spherical_weight * ssim_map) / np.sum(spherical_weight)
    return ws_ssim_map


def make_spherical_weight(img_shape):
    """
    Args:
        img_shape:  [H,W,C] or [H,W] if Y channel
    """
    h = img_shape[0]
    w_map = np.zeros(img_shape, dtype=np.float64)  # [H,W,C]
    for j in range(h):
        row_weight = np.cos((j - (h / 2) + 0.5) * np.pi / h)
        w_map[j] = row_weight
    return w_map
