import os
import cv2

import math
import random
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import normalize
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import glob
import pickle
import numpy as np
import imageio
import omegaconf

import torch
from torch.utils import data

# from basicsr.data.data_util import scandir
from basicsr.archs.arch_util import to_2tuple
from basicsr.utils import rgb2ycbcr

from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Multi_Scale_SR_Dataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        # self.name = self.opt['name']  # dataset name, 冇用嘅, 但係要有
        self.scale = self.opt.get('scale', None)  # [1.1,...,4.0]
        if self.scale is None:
            self.scale = [s/10 for s in range(int(self.opt['scale_min']*10), int(self.opt['scale_max']*10+1))]

        self.phase = self.opt['phase']  # 'train' 'val' 'test'
        self.repeat = self.opt.get('repeat', 1)  # train: 100, val: 1, test: 1
        self.scale2idx = {scale: idx for idx, scale in enumerate(self.scale)}
        self.sub_image = self.opt.get('sub_image', False)  # 是否为 sub_image
        self.flatten = self.opt.get('flatten', False)  # 是否 flatten
        self.mean = self.opt.get('mean', None)
        self.std = self.opt.get('std', None)
        # self.idx_scale = 0
        # self.first_epoch =False
        self._set_filesystem()
        if self.opt['ext'].find('img') < 0:  # 如果 ars.ext 中没有 'img'，则要创建图像对应的bin文件。  python find方法：检测字符串中是否包含子字符串，若没有找到，则返回 -1
            path_bin_gt = os.path.join(self.apath_gt, 'bin')  # dataset/lau_dataset_resize_clean/odisr/training/bin
            path_bin_lq = os.path.join(self.apath_lq, 'bin')  # dataset/lau_dataset_resize_clean/odisr/training/bin
            os.makedirs(path_bin_gt, exist_ok=True)
            os.makedirs(path_bin_lq, exist_ok=True)

        list_gt, list_lq = self._scan()  # .png图像的文件路径列表
        # if ('first_k' in self.opt) and self.phase!='train':
        #     print(f'use first_k: {self.opt["first_k"]} while in {self.phase} phase')
        #     list_gt = list_gt[:self.opt['first_k']]
        #     list_lq = [l[:self.opt['first_k']] for l in list_lq]

        if self.opt['ext'].find('bin') >= 0:  # store whole imgs in a single bin file
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            self.images_gt = self._check_and_load(
                self.opt['ext'], list_gt, self._name_gt_bin()
            )
            self.images_lq = [
                self._check_and_load(self.opt['ext'], l, self._name_lq_bin(s)) \
                for s, l in zip(self.scale, list_lq)
            ]
        elif self.opt['ext'].find('img') >=0:
            self.images_gt, self.images_lq = list_gt, list_lq
        elif self.opt['ext'].find('sep') >= 0:
            dir_bin_gt = os.path.join(os.path.dirname(self.dir_gt), 'bin', os.path.basename(self.dir_gt))
            # self.dir_gt: dataset/lau_dataset_resize_clean/odisr/training/HR
            # dir_bin_gt: dataset/lau_dataset_resize_clean/odisr/training/bin/HR
            os.makedirs(
                dir_bin_gt,
                # self.dir_gt.replace(self.apath_gt, path_bin_gt),
                exist_ok=True
            )
            dir_bin_lq = os.path.join(os.path.dirname(self.dir_lq), 'bin', os.path.basename(self.dir_lq))
            # self.dir_lq: dataset/lau_dataset_resize_clean/odisr/training/LR_fisheye
            # dir_bin_lq: dataset/lau_dataset_resize_clean/odisr/training/bin/LR_fisheye
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        dir_bin_lq,
                        'X{:.2f}'.format(s)
                    ),
                    exist_ok=True
                )

            self.images_gt, self.images_lq = [], [[] for _ in range(len(self.scale))]
            for h in list_gt:  # 读取每一个 gt.png 图像的文件路径
                b = h.replace(self.apath_gt, path_bin_gt)
                b = b.replace(self.ext[0], '.pt')
                self.images_gt.append(b)
                self._check_and_load(
                    self.opt['ext'], [h], b, verbose=True, load=False
                )

            for i, ll in enumerate(list_lq):
                for l in ll:
                    b = l.replace(self.apath_lq, path_bin_lq)  # replace 不会改变原字符串，而是返回一个新字符串
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lq[i].append(b)
                    # 需要制作 bin 文件时开启下列注释
                    self._check_and_load(
                        self.opt['ext'], [l], b, verbose=True, load=False
                    )

    def _set_filesystem(self):
        self.dir_gt = self.opt['dataroot_gt']   #  dataset/lau_dataset_resize_clean/odisr/training/HR
        self.dir_lq = self.opt['dataroot_lq']   #  dataset/lau_dataset_resize_clean/odisr/training/LR_fisheye
        self.apath_gt = os.path.dirname(self.dir_gt)  # dataset/lau_dataset_resize_clean/odisr/training/
        self.apath_lq = os.path.dirname(self.dir_lq)  # dataset/lau_dataset_resize_clean/odisr/training/
        self.ext = ('.png', '.png')

    # Below functions as used to prepare images
    def _scan(self):
        names_gt = sorted(
            glob.glob(os.path.join(self.dir_gt, '*' + self.ext[0]))  # 返回包括 dir_gt 的图片文件路径。
        )
        if 'first_k' in self.opt:
            print(f'use first_k: {self.opt["first_k"]} while in {self.phase} phase')
            names_gt = names_gt[:self.opt['first_k']]

        names_lq = [[] for _ in range(len(self.scale))]
        print(f'num of gt: {len(names_gt)}')
        for f in names_gt:
            filename, _ = os.path.splitext(os.path.basename(f))  # (000, png)
            for si, s in enumerate(self.scale):
                names_lq[si].append(os.path.join(
                    self.dir_lq, 'X{:.2f}/{}{}'.format(s, filename, self.ext[1])
                    ))
        return names_gt, names_lq
    
    def _name_gt_bin(self):
        return os.path.join(
            os.path.dirname(self.dir_gt),  # dataset/lau_dataset_resize_clean/odisr/training/
            'bin',  # bin
            '{}_bin_HR.pt'.format(self.phase)
        )

    def _name_lq_bin(self, scale):
        return os.path.join(
            os.path.dirname(self.dir_lq),
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.phase, scale)
        )
    
    def _check_and_load(self, ext, list_img, f, verbose=True, load=True):
        if os.path.exists(f) and ext.find('reset') < 0:  # 存在文件，并且不需要reset
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f:
                    ret = pickle.load(_f)
                return ret
            else:
                return None
        else: # 或者不存在bin文件f，或者要求reset
            if verbose:
                if ext.find('reset') >= 0:  # 要求reset
                    print('Making a new binary: {}'.format(f))
                else:  # 不存在bin文件f
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],  # 0000
                'image': imageio.imread(_l)
            } for _l in list_img]
            with open(f, 'wb') as _f:
                pickle.dump(b, _f)
            return b
        
    def __len__(self):
        if self.phase == 'train':
            return len(self.images_gt) * self.repeat
        else:
            return len(self.images_gt)
        
    def _get_index(self, idx):
        if self.phase == 'train':
            return idx % len(self.images_gt)
        else:
            return idx
        
    def _load_file(self, idx, scale):
        idx = self._get_index(idx)  # 对应 gt.png 的序号
        f_hr = self.images_gt[idx]
        f_lr = self.images_lq[self.scale2idx[scale]][idx]

        if self.opt['ext'].find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.opt['ext'] == 'img':
                hr = imageio.imread(f_hr)  # [H, W, C], RGB, uint8, ndarray
                lr = imageio.imread(f_lr)
                # hr = transforms.ToTensor()(Image.open(f_hr).convert('RGB'))
                # lr = transforms.ToTensor()(Image.open(f_lr).convert('RGB'))
            elif self.opt['ext'].find('sep') >= 0:
                with open(f_hr, 'rb') as _f:
                    hr = np.load(_f, allow_pickle=True)[0]['image']
                with open(f_lr, 'rb') as _f:
                    lr = np.load(_f, allow_pickle=True)[0]['image']
                # with open(f_hr, 'rb') as _f:
                #     hr = pickle.load(_f)[0]['image']
                #     hr = np.ascontiguousarray(hr.transpose(2, 0, 1))
                #     hr = torch.from_numpy(hr).float() / 255  # 0~1
                # with open(f_lr, 'rb') as _f:
                #     lr = pickle.load(_f)[0]['image']
                #     lr = np.ascontiguousarray(lr.transpose(2, 0, 1))
                #     lr = torch.from_numpy(lr).float() / 255  # 0~1

        return lr, hr, filename
    
    def __getitem__(self, idx, scale:float=None):
        # debug: idx = 11
        if scale is not None:
            s = scale
        elif self.phase=='train':
            s = random.sample(self.scale, 1)[0]
            # s = random.uniform(self.scale_min*10, self.scale_max*10)//1/10
        elif self.phase=='val':
            s = random.sample(self.scale, 1)[0]
        elif self.phase=='test':   # test 指定唯一的 scale
            s = self.scale[0]

        # (scale_min, scale_max) = (1.1, 4.0), step=0.1
        lr_ndarray, hr_ndarray, filename = self._load_file(idx, s)  # numpy 格式的图片  # uint8  # 0~255
        lr_ndarray, hr_ndarray = set_channel(lr_ndarray, hr_ndarray, n_channels=self.opt['n_colors'])  # 设置图片通道数
        lr_tensor, hr_tensor = np2Tensor(
            lr_ndarray, hr_ndarray, rgb_range=self.opt['rgb_range']
        )  # [C H W] # float32, 0~1
        # lr_tensor, hr_tensor, filename = self._load_file(idx, s)  # tensor 格式的图片  # uint8  # 0~255
        if self.mean is not None or self.std is not None:
            normalize(lr_tensor, self.mean, self.std, inplace=True)
            normalize(hr_tensor, self.mean, self.std, inplace=True)

        if self.opt.get('inp_size', None) is None:  # do not crop
            crop_lr, crop_hr = lr_tensor, hr_tensor
            h_lr, w_lr = lr_tensor.shape[-2:]
            condition_dict = get_condition(h=h_lr, w=w_lr, condition_types=self.opt['condition_types'])
        else: # do random crop
            # Meta-SR styled random crop
            if s==int(s):
                step = 1
            elif (s*2)== int(s*2):  # .5
                step = 2
            elif (s*5) == int(s*5):  # .4 / .2 / .6
                step = 5
            else:
                step = 10

            ih, iw = lr_tensor.shape[-2:]

            # length
            h_ip, w_ip = to_2tuple(self.opt['inp_size'])  # [64, 128] [64, 64]
            h_op, w_op = int(round(s * h_ip)), int(round(s * w_ip))

            # if iw == w_ip:
            #     print()

            # left top point
            if step > 2:  # ArbSR style (start:2)
                y0_lr = random.randrange(2, (ih-h_ip)//step-2) * step
                x0_lr = random.randrange(2, (iw-w_ip)//step-2) * step
            else:  # MetaSR style (start:0)
                y0_lr = random.randrange(0, (ih-h_ip+step)//step) * step  # random.randrange(0, (ih-h_ip+step)//step) * step
                x0_lr = random.randrange(0, (iw-w_ip+step)//step) * step
            
            y0_hr = int(round(y0_lr * s))
            x0_hr = int(round(x0_lr * s))
            
            crop_hr = hr_tensor[:, y0_hr: y0_hr + h_op, x0_hr: x0_hr + w_op]  # crop from (x0, y0)
            crop_lr = lr_tensor[:, y0_lr: y0_lr + h_ip, x0_lr: x0_lr + w_ip]  # crop from (x0, y0)

            if self.sub_image:  # 意味着 [ih, iw] 不是 erp 尺寸
                sub_h, sub_w = os.path.split(filename)[-1].split('_')[3:5]
                sub_h, sub_w = int(int(sub_h) / s), int(int(sub_w) / s)

                gt_erp_size = self.opt.get('gt_size', [1024, 2048])
                
                lr_erp_size = [int(i/s) for i in gt_erp_size]
            else:
                sub_h, sub_w = 0, 0
                lr_erp_size = [ih, iw]
            # make_coord [H, W, YX]
            crop_grid_lr = make_coord(lr_erp_size , flatten=False).flip(-1).permute(2, 0, 1)[:, sub_h+y0_lr: sub_h+y0_lr + h_ip, sub_w+x0_lr: sub_w+x0_lr + w_ip]  # [XY, H, W]
            grid_x_lr, grid_y_lr = crop_grid_lr[0], crop_grid_lr[1]  # shape [h, w]
            
            condition_dict = get_condition(grid_x=grid_x_lr, grid_y=grid_y_lr, condition_types=self.opt['condition_types'])

        if self.phase == 'train':
            hflip = ('hflip' in self.opt.get('data_aug', [])) and (random.random() < 0.5)
            vflip = ('vflip' in self.opt.get('data_aug', [])) and (random.random() < 0.5)
            dflip = ('dflip' in self.opt.get('data_aug', [])) and (random.random() < 0.5)

            def augment(x):
                if hflip:
                    x = x.flip(-2)  # [C, H, W]
                if vflip:
                    x = x.flip(-1)  # Conforms to the panorama
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord(crop_hr.shape[-2:], flatten=False)  # no flattern, [H, W, YX]

        # import torchvision.utils as tvu
        # import torch.nn.functional as F
        # tvu.save_image(hr_tensor, './debug_output/hr_tensor.png')
        # tvu.save_image(lr_tensor, './debug_output/lr_tensor.png')
        # tvu.save_image(crop_hr, './debug_output/crop_hr_x4.png')
        # tvu.save_image(crop_lr, './debug_output/crop_lr_x4.png')
        # up_from_lr = F.grid_sample(crop_lr.unsqueeze(0), hr_coord.unsqueeze(0).flip(-1), mode='bilinear',padding_mode='border', align_corners=False)
        # tvu.save_image(up_from_lr, './debug_output/up_from_lr.png')
        
        # tvu.save_image(crop_hr, './debug_output/out_crop_hr_x4.png')
        # up_from_lr = F.grid_sample(crop_lr.unsqueeze(0), hr_coord.unsqueeze(0).flip(-1), mode='bilinear',padding_mode='border', align_corners=False)
        # tvu.save_image(up_from_lr, './debug_output/out_up_from_lr.png')

        if 'out_size' in self.opt:
            h_ip, w_ip = to_2tuple(self.opt['out_size'])
            h_op, w_op = crop_hr.shape[-2:]
            x0          = random.randint(0, w_op - w_ip)
            y0          = random.randint(0, h_op - h_ip)
            hr_coord    = hr_coord[y0: y0 + h_ip, x0: x0 + w_ip, :]
            crop_hr     = crop_hr[:, y0: y0 + h_ip, x0: x0 + w_ip]

        cell = torch.ones_like(hr_coord)  # [H, W, YX]
        cell[..., 0] *= 2 / crop_hr.shape[-2]  # H
        cell[..., 1] *= 2 / crop_hr.shape[-1]  # W

        if self.sub_image:  # 意味着 hr_tensor 不是 erp 尺寸
            gt_erp_size = self.opt.get('gt_size', [1024, 2048])
            lr_erp_size = [int(i/s) for i in gt_erp_size]
        else:
            gt_erp_size = [hr_tensor.shape[1], hr_tensor.shape[2]]
            lr_erp_size = [lr_tensor.shape[1], lr_tensor.shape[2]]
        
        condition_dict['gt_erp_size'] = torch.tensor(gt_erp_size)
        condition_dict['lr_erp_size'] = torch.tensor(lr_erp_size)
        condition_dict['out_size'] = torch.tensor(crop_hr.shape[-2:])
        condition_dict['inp_size'] = torch.tensor(crop_lr.shape[-2:])

        # flatten
        if self.flatten:
            hr_rgb = crop_hr.permute(1, 2, 0).reshape(-1, crop_hr.shape[0])   # [C, H, W] -> [H, W, C] -> [H*W, C]
            cell = cell.reshape(-1, cell.shape[-1])  # [H, W, YX] -> [H*W, YX]
            hr_coord = hr_coord.reshape(-1, hr_coord.shape[-1])  # [H, W, YX] -> [H*W, YX]

        else:
            hr_rgb = crop_hr
            cell = cell
            hr_coord = hr_coord

        if 'sample_q' in self.opt:   # 从 crop_hr 中随机采样
            if not self.flatten:
                raise RuntimeError('[sample_q] only support when [flatten]')
                
            sample_lst = np.random.choice(
            len(hr_coord), self.opt['sample_q'], replace=False)

            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            cell = cell[sample_lst]
            
        return {'inp': crop_lr, 'gt': hr_rgb, 'gt_path': self.opt['dataroot_gt'], 'coord':hr_coord, 'cell':cell, 'condition':condition_dict, 'filename':filename}
    
    # def get_patch(self, lr, hr, scale):
    #     multi_scale = len(self.scale) > 1
    #     if self.phase == 'train':
    #         lr, hr = common.get_patch(
    #             lr,
    #             hr,
    #             patch_size=self.args.patch_size,
    #             scale=scale,
    #             multi_scale=multi_scale
    #         )
    #         if not self.args.no_augment:
    #             lr, hr = common.augment(lr, hr)
    #     else:
    #         ih, iw = lr.shape[:2]
    #         hr = hr[0:int(ih * scale), 0:int(iw * scale)]

    #     return lr, hr
    # def set_scale(self, idx_scale):
    #     self.idx_scale = idx_scale



### other tools ###
    
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)  # [*shape, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # [*, 2]
    return ret

def get_condition(grid_x=None, grid_y=None ,h=None, w=None, condition_types:list=[]):
    """
    grid_x, gird_y: (-1, +1) shape [h, w]
    condition_types: ['zeros'., 'cos_latitude', 'sin_latitude', 'latitude', 'lon_lat', 'grid', 'coord']
    """
    _condition_dict = dict()
    if (grid_x is None) and (grid_y is None) and (h is not None) and (w is not None):
        grid = make_coord([h,w], flatten=False).flip(-1) # [h, w, XY]
        grid_x = grid[...,0]  # [h, w]
        grid_y = grid[...,1]  # [h, w]
    elif (grid_x is not None) and (grid_y is not None) and (h is None) and (w is None):
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [h, w ,XY]
        grid_x = grid_x  # [h, w]
        grid_y = grid_y
    else:
        raise RuntimeError('Unsupported shape data')
        
    if not isinstance(condition_types, list) and not isinstance(condition_types, omegaconf.listconfig.ListConfig):
        condition_types = [condition_types]

    for _condition_type in condition_types:
        if _condition_type == 'zeros':
            _condition_dict[_condition_type] = torch.zeros_like(grid_y.unsqueeze(0))  # [1, h, w]
        elif _condition_type == 'cos_latitude':
            _condition_dict[_condition_type] = torch.cos(grid_y.unsqueeze(0) * math.pi / 2 )   # [1, h, w]
        elif _condition_type == 'sin_latitude':
            _condition_dict[_condition_type] = torch.sin(grid_y.unsqueeze(0) * math.pi / 2 )   # [1, h, w]
        elif _condition_type == 'latitude':
            _condition_dict[_condition_type] = grid_y.unsqueeze(0) * math.pi / 2  # [1, h, w] y方向上的纬度 -90度~90度
        elif _condition_type == 'lon_lat':
            _condition_dict[_condition_type] = torch.concat([grid_x.unsqueeze(0) * math.pi, grid_y.unsqueeze(0) * math.pi / 2], dim=0)  # [2, h, w]
        elif _condition_type == 'grid':
            _condition_dict[_condition_type] = grid.permute(2, 0, 1)  # [XY(2), h, w]
        elif _condition_type == 'coord':
            _condition_dict[_condition_type] = grid.permute(2, 0, 1)  # [XY(2), h, w]
        else:
            raise RuntimeError('Unsupported condition type')
        
    return _condition_dict

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:  # img shape (H,W)
            img = np.expand_dims(img, axis=2)  # img shape (H, W, 1)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]