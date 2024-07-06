import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import time
import os
from os import makedirs
from os.path import join, exists

from einops import rearrange

class MultiViewer(nn.Module):
    def __init__(self):
        super(MultiViewer, self).__init__()
        cur_path = os.path.dirname(__file__)
        print(cur_path)

        self.pers2equi_grid_dir = join(cur_path, 'multi_viewer_cache', 'pers2equi_grid')
        if not exists(self.pers2equi_grid_dir):
            makedirs(self.pers2equi_grid_dir)
        self.pers2equi_grid = dict()

        self.equi2pers_grid_dir = join(cur_path, 'multi_viewer_cache', 'equi2pers_grid') 
        if not exists(self.equi2pers_grid_dir):
            makedirs(self.equi2pers_grid_dir)
        self.equi2pers_grid = dict()

    def pair(self,t):
        return t if isinstance(t, tuple) else (t, t)

    def uv2xyz(self,uv):
        """
            z     
            ↑
            . —→ y
        x ↙

        """
        # uv = uv.cpu().numpy()
        # xyz = np.zeros((*uv.shape[:-1], 3), dtype = np.float32)
        # xyz[..., 0] = np.multiply(np.cos(uv[..., 1]), np.sin(uv[..., 0]))
        # xyz[..., 1] = np.multiply(np.cos(uv[..., 1]), np.cos(uv[..., 0]))
        # xyz[..., 2] = np.sin(uv[..., 1])

        xyz = torch.zeros((*uv.shape[:-1], 3), dtype = torch.float32, device=uv.device)
        xyz[..., 0] = torch.cos(uv[..., 1]) * torch.sin(uv[..., 0])
        xyz[..., 1] = torch.cos(uv[..., 1]) * torch.cos(uv[..., 0])
        xyz[..., 2] = torch.sin(uv[..., 1])
        return xyz
    
    def equi2pers(self,erp_img, fov, nrows, patch_size, pre_upscale=1, pre_mode='bicubic', proj_mode='bicubic'):
        if pre_upscale > 1:
            # 'bicubic' 'nearest' 'area' 'bilinear'
            if pre_mode != 'nearest':
                erp_img = F.interpolate(erp_img, scale_factor=pre_upscale, mode=pre_mode, align_corners=True)
            else: # nearest mode
                erp_img = F.interpolate(erp_img, scale_factor=pre_upscale, mode=pre_mode)
        device = erp_img.device
        erp_size = erp_img.shape[-2:]
        bs, erp_c, erp_h, erp_w = erp_img.shape
        height, width = self.pair(patch_size)
        fov_h, fov_w = self.pair(fov)
        # layer_name = rf"fov{self.pair(fov)}_nrows{nrows}_patch_size{self.pair(patch_size)}_erp_size{self.pair(erp_size)}"
        layer_name = rf"fov_{fov_h}_{fov_w}-nrows_{nrows}-patch_size_{height}_{width}-erp_size_{erp_size[0]}_{erp_size[1]}"
        grid_file = join(self.equi2pers_grid_dir, layer_name + '.pth') 

        if layer_name in self.equi2pers_grid:  # 首先, 看内存中是否有 投影配置参数
            load_file = self.equi2pers_grid[layer_name]
            load_file = {k: load_file[k].float().to(device) for k in load_file}
            grid = load_file["grid"]#.float().to(device)
            xyz = load_file["xyz"]#.float().to(device)
            uv = load_file["uv"]#.float().to(device)
            center_p = load_file["center_p"]#.float().to(device)
        elif exists(grid_file):  # 其次, 看磁盘中是否有 投影配置参数
            load_file = torch.load(grid_file, map_location=device)
            load_file = {k: load_file[k].float().to(device) for k in load_file}
            self.equi2pers_grid[layer_name] = load_file  # 把磁盘中的数据转移到内存中, 方便下次使用
            grid = load_file["grid"]#.float()
            xyz = load_file["xyz"]#.float()
            uv = load_file["uv"]#.float()
            center_p = load_file["center_p"]#.float()
        else:
            FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32).to(device)

            PI = math.pi
            PI_2 = math.pi * 0.5
            PI2 = math.pi * 2
            yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij')
            screen_points = torch.stack([xx.flatten(), yy.flatten()], -1).to(device)
            
            if nrows==4:    
                num_rows = 4
                num_cols = [3, 6, 6, 3]
                phi_centers = [-67.5, -22.5, 22.5, 67.5]
            if nrows==6:    
                num_rows = 6
                num_cols = [3, 8, 12, 12, 8, 3]
                phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
            if nrows==3:
                num_rows = 3
                num_cols = [3, 4, 3]
                phi_centers = [-60, 0, 60]     
            if nrows==5:
                num_rows = 5
                num_cols = [3, 6, 8, 6, 3]
                phi_centers = [-72.2, -36.1, 0, 36.1, 72.2]
                    
            phi_interval = 180 // num_rows
            all_combos = []
            erp_mask = []
            for i, n_cols in enumerate(num_cols):
                for j in np.arange(n_cols):
                    theta_interval = 360 / n_cols
                    theta_center = j * theta_interval + theta_interval / 2

                    center = [theta_center, phi_centers[i]]
                    all_combos.append(center)
                    up = phi_centers[i] + phi_interval / 2
                    down = phi_centers[i] - phi_interval / 2
                    left = theta_center - theta_interval / 2
                    right = theta_center + theta_interval / 2
                    up = int((up + 90) / 180 * erp_h)
                    down = int((down + 90) / 180 * erp_h)
                    left = int(left / 360 * erp_w)
                    right = int(right / 360 * erp_w)
                    mask = np.zeros((erp_h, erp_w), dtype=int)
                    mask[down:up, left:right] = 1
                    erp_mask.append(mask)
            all_combos = np.vstack(all_combos) 
            shifts = np.arange(all_combos.shape[0]) * width
            shifts = torch.from_numpy(shifts).float().to(device)
            erp_mask = np.stack(erp_mask)
            erp_mask = torch.from_numpy(erp_mask).float().to(device)
            num_patch = all_combos.shape[0]

            center_point = torch.from_numpy(all_combos).float().to(device)  # -180 to 180, -90 to 90
            center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
            center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

            cp = center_point * 2 - 1
            center_p = cp.clone()
            cp[:, 0] = cp[:, 0] * PI
            cp[:, 1] = cp[:, 1] * PI_2
            cp = cp.unsqueeze(1)
            convertedCoord = screen_points * 2 - 1
            convertedCoord[:, 0] = convertedCoord[:, 0] * PI
            convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
            convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32, device=device) * FOV)
            convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

            x = convertedCoord[:, :, 0]
            y = convertedCoord[:, :, 1]

            rou = torch.sqrt(x ** 2 + y ** 2)
            c = torch.atan(rou)
            sin_c = torch.sin(c)
            cos_c = torch.cos(c)
            lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
            lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
            lat_new = lat / PI_2 
            lon_new = lon / PI 
            lon_new[lon_new > 1] -= 2
            lon_new[lon_new<-1] += 2 

            lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
            lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
            grid = torch.stack([lon_new, lat_new], -1)
            grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)

            grid_tmp = torch.stack([lon, lat], -1)
            xyz = self.uv2xyz(grid_tmp)  # 原来为 numpy 函数, 后修改为 torch 函数
            xyz = xyz.reshape(num_patch, height, width, 3).permute(0, 3, 1, 2)
            # xyz = torch.from_numpy(xyz).to(erp_img.device).contiguous()
            
            uv = grid[0, ...].reshape(height, width, num_patch, 2).permute(2, 3, 0, 1)
            uv = uv.contiguous()

            save_file = {"grid":grid, "xyz":xyz, "uv":uv, "center_p":center_p}
            self.equi2pers_grid[layer_name] = save_file  # 既保存在内存中
            torch.save(save_file, grid_file)  # 也保存在磁盘中
        # 'bilinear' 'nearest' 'area' 'bicubic'
        erp_img = erp_img.float()
        grid = grid.float()
        if proj_mode != 'nearest':
            #
            # erp_img.device is cuda
            pers = F.grid_sample(erp_img, grid, mode=proj_mode, padding_mode='border', align_corners=True)
        else:  # nearest mode
            pers = F.grid_sample(erp_img, grid, mode=proj_mode, padding_mode='border')
        pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
        pers = pers.reshape(bs, erp_c, height, width, -1)
    

        return pers, xyz, uv, center_p
    
    def pers2equi(self, pers_img, fov, nrows, erp_size, pre_upscale=1, pre_mode='bicubic'):
        if pre_upscale > 1:
            num_patch = pers_img.shape[-1]
            pers_img = rearrange(pers_img, 'B C H W Np -> (B Np) C H W')
            # 'bilinear' 'nearest' 'area' 'bicubic'
            if pre_mode != 'nearest':
                pers_img_up = F.interpolate(pers_img, scale_factor=pre_upscale, mode=pre_mode, align_corners=True)
            else:  # nearest mode
                pers_img_up = F.interpolate(pers_img, scale_factor=pre_upscale, mode=pre_mode)
            pers_img = rearrange(pers_img_up, '(B Np) C H W -> B C H W Np', Np=num_patch)
            
        bs = pers_img.shape[0]
        channel = pers_img.shape[1]
        device = pers_img.device
        height, width = pers_img.shape[2], pers_img.shape[3]
        fov_h, fov_w = self.pair(fov)
        erp_h, erp_w = self.pair(erp_size)
        n_patch = pers_img.shape[-1]
        # layer_name = rf"fov{self.pair(fov)}_nrows{nrows}_patch_size{self.pair(patch_size)}_erp_size{self.pair(erp_size)}"
        layer_name = rf"fov_{fov_h}_{fov_w}-nrows_{nrows}-patch_size_{height}_{width}-erp_size_{erp_size[0]}_{erp_size[1]}"
        grid_file = join(self.pers2equi_grid_dir, layer_name + '.pth') 
        if layer_name in self.pers2equi_grid:  # 首先, 看内存中是否有 投影配置参数
            load_file = self.pers2equi_grid[layer_name]
            # load_file = {k: load_file[k].float().to(device) for k in load_file}
            #print('load_file')
            x0 = load_file['x0'].to(device)
            y0 = load_file['y0'].to(device)
            x1 = load_file['x1'].to(device)
            y1 = load_file['y1'].to(device)
            w_list = load_file['w_list'].float().to(device)
            mask = load_file['mask'].to(device)
        elif exists(grid_file):  # 其次, 看磁盘中是否有 投影配置参数
            # the online merge really takes time
            # pre-calculate the grid for once and use it during training
            load_file = torch.load(grid_file, map_location=device)
            # load_file = {k: load_file[k].float().to(device) for k in load_file}
            self.pers2equi_grid[layer_name] = load_file  # 把磁盘中的数据转移到内存中, 方便下次使用
            #print('load_file')
            x0 = load_file['x0'].to(device)
            y0 = load_file['y0'].to(device)
            x1 = load_file['x1'].to(device)
            y1 = load_file['y1'].to(device)
            w_list = load_file['w_list'].float().to(device)
            mask = load_file['mask'].to(device)
        else:  # 在内存和磁盘中都不存在, 则计算 投影配置参数
            FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32).to(device)

            PI = math.pi
            PI_2 = math.pi * 0.5
            PI2 = math.pi * 2

            if nrows==4:    
                num_rows = 4
                num_cols = [3, 6, 6, 3]
                phi_centers = [-67.5, -22.5, 22.5, 67.5]
            if nrows==6:    
                num_rows = 6
                num_cols = [3, 8, 12, 12, 8, 3]
                phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
            if nrows==3:
                num_rows = 3
                num_cols = [3, 4, 3]
                phi_centers = [-59.6, 0, 59.6]
            if nrows==5:
                num_rows = 5
                num_cols = [3, 6, 8, 6, 3]
                phi_centers = [-72.2, -36.1, 0, 36.1, 72.2] 
            phi_interval = 180 // num_rows
            all_combos = []

            for i, n_cols in enumerate(num_cols):
                for j in np.arange(n_cols):
                    theta_interval = 360 / n_cols
                    theta_center = j * theta_interval + theta_interval / 2

                    center = [theta_center, phi_centers[i]]
                    all_combos.append(center)
                    
                    
            all_combos = np.vstack(all_combos) 
            n_patch = all_combos.shape[0]
            
            center_point = torch.from_numpy(all_combos).float().to(device)  # -180 to 180, -90 to 90
            center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
            center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

            cp = center_point * 2 - 1
            cp[:, 0] = cp[:, 0] * PI
            cp[:, 1] = cp[:, 1] * PI_2
            cp = cp.unsqueeze(1)
            
            lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, erp_h, device=device), torch.linspace(-PI, PI, erp_w, device=device), indexing='ij')
            lon_grid = lon_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1)
            lat_grid = lat_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1) 
            cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(lon_grid - cp[..., 0])
            new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
            new_y = (torch.cos(cp[..., 1])*torch.sin(lat_grid) - torch.sin(cp[...,1])*torch.cos(lat_grid)*torch.cos(lon_grid-cp[...,0])) / cos_c
            new_x = new_x / FOV[0] / PI   # -1 to 1
            new_y = new_y / FOV[1] / PI_2
            cos_c_mask = cos_c.reshape(n_patch, erp_h, erp_w)
            cos_c_mask = torch.where(cos_c_mask > 0, 1, 0)
            
            w_list = torch.zeros((n_patch, erp_h, erp_w, 4), dtype=torch.float32, device=device)

            new_x_patch = (new_x + 1) * 0.5 * height
            new_y_patch = (new_y + 1) * 0.5 * width 
            new_x_patch = new_x_patch.reshape(n_patch, erp_h, erp_w)
            new_y_patch = new_y_patch.reshape(n_patch, erp_h, erp_w)
            mask = torch.where((new_x_patch < width) & (new_x_patch > 0) & (new_y_patch < height) & (new_y_patch > 0), 1, 0)
            mask *= cos_c_mask

            x0 = torch.floor(new_x_patch).type(torch.int64)
            x1 = x0 + 1
            y0 = torch.floor(new_y_patch).type(torch.int64)
            y1 = y0 + 1

            x0 = torch.clamp(x0, 0, width-1)
            x1 = torch.clamp(x1, 0, width-1)
            y0 = torch.clamp(y0, 0, height-1)
            y1 = torch.clamp(y1, 0, height-1)

            wa = (x1.type(torch.float32)-new_x_patch) * (y1.type(torch.float32)-new_y_patch)
            wb = (x1.type(torch.float32)-new_x_patch) * (new_y_patch-y0.type(torch.float32))
            wc = (new_x_patch-x0.type(torch.float32)) * (y1.type(torch.float32)-new_y_patch)
            wd = (new_x_patch-x0.type(torch.float32)) * (new_y_patch-y0.type(torch.float32))

            wa = wa * mask.expand_as(wa)
            wb = wb * mask.expand_as(wb)
            wc = wc * mask.expand_as(wc)
            wd = wd * mask.expand_as(wd)
    
            w_list[..., 0] = wa
            w_list[..., 1] = wb
            w_list[..., 2] = wc
            w_list[..., 3] = wd
            w_list = torch.where(torch.isnan(w_list), torch.zeros_like(w_list), w_list)

        
            save_file = {'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1, 'w_list': w_list, 'mask':mask}
            self.pers2equi_grid[layer_name] = save_file  # 既保存在内存中
            torch.save(save_file, grid_file)  # 也保存在磁盘中

        w_list = w_list.to(device)
        mask = mask.to(device)
        z = torch.arange(n_patch, device=device)
        z = z.reshape(n_patch, 1, 1)
        #start = time.time()
        Ia = pers_img[:, :, y0, x0, z]
        Ib = pers_img[:, :, y1, x0, z]
        Ic = pers_img[:, :, y0, x1, z]
        Id = pers_img[:, :, y1, x1, z]
        #print(time.time() - start)
        output_a = Ia * mask.expand_as(Ia)
        output_b = Ib * mask.expand_as(Ib)
        output_c = Ic * mask.expand_as(Ic)
        output_d = Id * mask.expand_as(Id)

        output_a = output_a.permute(0, 1, 3, 4, 2)
        output_b = output_b.permute(0, 1, 3, 4, 2)
        output_c = output_c.permute(0, 1, 3, 4, 2)
        output_d = output_d.permute(0, 1, 3, 4, 2)   
        #print(time.time() - start)
        w_list = w_list.permute(1, 2, 0, 3)
        w_list = w_list.flatten(2)
        w_list *= torch.gt(w_list, 1e-5).type(torch.float32)
        w_list = F.normalize(w_list, p=1, dim=-1).reshape(erp_h, erp_w, n_patch, 4)
        w_list = w_list.unsqueeze(0).unsqueeze(0)
        w_list = torch.where(torch.isnan(w_list), torch.zeros_like(w_list), w_list)  # 防御性代码，修复 nan 的情况
        output = output_a * w_list[..., 0] + output_b * w_list[..., 1] + \
            output_c * w_list[..., 2] + output_d * w_list[..., 3]
        img_erp = output.sum(-1) 

        return img_erp

if __name__ == "__main__":
    from basicsr.utils import img2tensor,tensor2img
    import torchvision.utils as tvu
    import time
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    start_time = time.time()
    old_erp_np = cv2.imread('datasets/MyLTE_load/lau_dataset_resize_clean/odisr/testing/HR/0000.png', cv2.IMREAD_COLOR)
    # img_np = cv2.imread('/workspace/SuperResolution/StableSR_lry/datasets/MyLTE_load/lau_dataset_resize_clean/odisr/testing/LR_erp/X4.00/0000.png', cv2.IMREAD_COLOR)
    old_erp_tensor = img2tensor(old_erp_np/255., bgr2rgb=True, float32=True).unsqueeze(0)#.cuda()

    pre_upscale = 4
    nrows = 4
    fov = (90, 90) 
    patch_size = (768, 768)

    # pre_upscale = 4
    # nrows = 4
    # fov = (75, 75)
    # patch_size = (512, 512)

    # pre_upscale = 4
    # nrows = 4 
    # fov = (90, 90) 
    # patch_size = (512, 512)
    # nrows = 5
    # fov = (60, 60)
    # patch_size = (512, 512)
    # nrows = 5
    # fov = (60, 60)  # nrows=5, fov >= 60.  nrows=4, fov >= 75. 
    # patch_size = (512, 512) # 800*800 = 64w > 12w
    print(f"nrows: {nrows}, fov: {fov}, patch_size: {patch_size}, pre_upscale: {pre_upscale}")
    multi_viewer = MultiViewer()
    multi_viewer = multi_viewer
    old_pers, _, _, _ = multi_viewer.equi2pers(old_erp_tensor, fov=fov, nrows=nrows, patch_size=patch_size, pre_upscale=pre_upscale) 
    # pers of shape: [N, C, p_h, p_w, patch_num=18]
    # for idx in range(old_pers.shape[-1]):
    #     tvu.save_image(old_pers[...,idx], f'test_pers_{idx}.png')

    num_patch = old_pers.shape[-1]
        
    new_erp_tensor = multi_viewer.pers2equi(old_pers, fov=fov, nrows=nrows, erp_size=(old_erp_tensor.shape[-2], old_erp_tensor.shape[-1]), pre_upscale=pre_upscale)
    # new_pers, _, _, _ = multi_viewer.equi2pers(new_erp_tensor, fov=fov, nrows=nrows, patch_size=patch_size, pre_upscale=4) 
    tvu.save_image(new_erp_tensor, 'test_interp_erp_0000.png')
    # new_erp_np = tensor2img(new_erp_tensor)
    end_time = time.time()
    print(f"erp-perp-erp finished: [{end_time-start_time:.2f}s], now calculate ws-psnr")
    from odisr.metrics import calculate_psnr
    from odisr.metrics.odi_metric import calculate_psnr_ws

    # print("calculate erp -> pers")
    # old_pers = rearrange(old_pers, 'n c h w p -> (n p) c h w')
    # new_pers = rearrange(new_pers, 'n c h w p -> (n p) c h w')
    # total_psnr = 0
    # for i in range(len(new_pers)):
    #     old_per_np = tensor2img(old_pers[i])
    #     new_per_np = tensor2img(new_pers[i])
    #     psnr = calculate_psnr(old_per_np, new_per_np, crop_border=0, input_order='HWC', test_y_channel=False)
    #     total_psnr += psnr

    # psnr_avg = total_psnr / len(new_pers)
    # print(f"psnr_avg: {psnr_avg:.4f}")
    
    print("calculate pers -> erp")
    new_erp_np = tensor2img(new_erp_tensor)
    ws_psnr = calculate_psnr_ws(old_erp_np, new_erp_np, crop_border=0, input_order='HWC', test_y_channel=False)
    print(f"ws_psnr: {ws_psnr:.4f}")

    """
    [hyper param]: 不同超参数对应的 ws_psnr 和 psnr
    nrows = 4 
    fov = (75, 75) 
    patch_size = (1024, 1024)
    ws_psnr: 30.6881
    psnr: 31.7662

    nrows: 4, 
    fov: (75, 75), 
    patch_size: (800, 800)
    ws_psnr: 28.4507
    psnr: 29.5802

    
    nrows = 5
    fov = (60, 60)
    patch_size = (1024, 1024)
    ws_psnr: 32.5184
    psnr: 33.5331

    nrows = 5
    fov = (60, 60)
    patch_size = (800, 800)
    ws_psnr: 30.2318
    psnr: 31.2823

    nrows = 5
    fov = (60, 60)
    patch_size = (768, 768)
    ws_psnr: 29.8584
    psnr: 30.9168


    [Code Example]: 一个最简单的使用例子

    img_tensor = img_read()
    if cuda:
        img_tensor = img_tensor.cuda()
    nrows = 5
    fov = (60, 60)
    patch_size = (1024, 1024)
    multi_viewer = MultiViewer()
    pers, _, _, _ = multi_viewer.equi2pers(img_tensor, fov=fov, nrows=nrows, patch_size=patch_size)
    erp_tensor = multi_viewer.pers2equi(pers, fov=fov, nrows=nrows, patch_size=patch_size, erp_size=(1024, 2048))

    if nrows==3:
        num_rows = 3
        num_cols = [3, 4, 3]
        phi_centers = [-59.6, 0, 59.6]
    if nrows==4:    
        num_rows = 4
        num_cols = [3, 6, 6, 3]  # sum(num_cols) = 18
        phi_centers = [-67.5, -22.5, 22.5, 67.5]
    if nrows==5:
        num_rows = 5
        num_cols = [3, 6, 8, 6, 3]  # sum(num_cols) = 26
        phi_centers = [-72.2, -36.1, 0, 36.1, 72.2] 
    if nrows==6:    
        num_rows = 6
        num_cols = [3, 8, 12, 12, 8, 3]
        phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
    """

