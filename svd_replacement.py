import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class BlurDown(nn.Module):
    def __init__(self, ratio, kernel_size, sigma, device):
        super().__init__()
        self.ratio = ratio
        self.device = device
        self.blur_layer = get_gaussian_kernel(kernel_size=kernel_size, sigma=sigma).to(self.device)

    def forward(self, x):
        # x:     batch, B, H,   W
        # x_out: batch, B, H/r, W/r
#         x_blur = self.blur_layer(x)
        x_blur = x
        x_out = F.interpolate(x_blur.clone(), scale_factor=1/self.ratio, mode='bicubic')
        return x_out


class PseudoInverse(nn.Module):
    def __init__(self, ratio, channels, device):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        self.device = device
        self.kernel_size = 7
        self.conv = nn.Conv2d(self.channels,
                              self.channels,
                              kernel_size=self.kernel_size,
                              stride=1,
                              padding=self.kernel_size//2,
                              bias=False
                              ).to(self.device)

    def forward(self, x_in):
        # x_in   : batch, C, H/r, W/r
        # x_out  : batch, C, H,   W
        x_up = F.interpolate(x_in, scale_factor=self.ratio, mode='nearest')
        x_out = self.conv(x_up)
        return x_out

    
class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        
#         temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def H_pinv_eta(self, vec, eta):
        """
        Multiplies the input vector by the pseudo inverse of H with factor eta
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars*singulars+eta)
#         print(temp.size(), factors.size(), singulars.size())
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def Lambda(self, vec, a, sigma_0, sigma_t, eta):
        raise NotImplementedError()

    def Lambda_noise(self, vec, a, sigma_0, sigma_t, eta, epsilon):
        raise NotImplementedError()
        

class CS(H_functions):
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // 32
        self.ratio = 32
        H = torch.randn(32**2, 32**2).to(device)
        _, _, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)
        self.singulars_small = torch.ones(int(32 * 32 * ratio), device=device)
        self.cs_size = self.singulars_small.size(0)

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)
        # temp = vec.clone().reshape(vec.shape[0], -1)[:, :self.channels * self.y_dim ** 2 * self.cs_size]
        # temp = temp.reshape(vec.size(0), -1, self.cs_size)
        # patches = torch.zeros(vec.size(0), temp.size(1), self.ratio ** 2, device=vec.device)
        # patches[:, :, :self.cs_size] = temp[:, :, :]

        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.size(0), self.channels * self.y_dim ** 2, self.ratio ** 2, device=vec.device)
        patches[:, :, :self.cs_size] = temp[:, :self.channels * self.y_dim ** 2 * self.cs_size].contiguous().reshape(
            vec.size(0), -1, self.cs_size)
        patches[:, :, self.cs_size:] = temp[:, self.channels * self.y_dim ** 2 * self.cs_size:].contiguous().reshape(
            vec.size(0), self.channels * self.y_dim ** 2, -1)

        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim ** 2 * self.cs_size] = patches[:, :, :, :self.cs_size].contiguous().reshape(
            vec.shape[0], -1)
        recon[:, self.channels * self.y_dim ** 2 * self.cs_size:] = patches[:, :, :, self.cs_size:].contiguous().reshape(
            vec.shape[0], -1)
        return recon

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

    
import torch.nn as nn
class MeanDownsampling_fix(nn.Module):
    def __init__(self, patch):
        super(MeanDownsampling_fix, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 1, patch, patch) / (patch * patch))
        self.weight.requires_grad = False
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.bias.requires_grad = False
        self.patch = patch

    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, self.bias, stride=self.patch, padding=0)
        return x


class MeanUpsample(nn.Module):
    def __init__(self, patch=2):
        super(MeanUpsample, self).__init__()

        self.patch = patch

    def forward(self, x):
        n, c, h, w = x.shape

        out = torch.zeros(n, c, h, self.patch, w, self.patch).to(x.device) + x.view(n, c, h, 1, w, 1)
        out = out.view(n, c, self.patch * h, self.patch * w)

        return out


def color2gray(x):
    x = x[:, 0:1, :, :] * 0.3333 + x[:, 1:2, :, :] * 0.3334 + x[:, 2:, :, :] * 0.3333
    return x


def gray2color(x):
    base = 0.3333 ** 2 + 0.3334 ** 2 + 0.3333 ** 2
    return torch.stack((x * 0.3333 / base, x * 0.3334 / base, x * 0.3333 / base), 1)

class ColorSRInpainting(H_functions):
    def __init__(self, channels=3, ratio=4): #ratio = 2 or 4
        self.channels = channels
        self.ratio = ratio
        self.meandown = MeanDownsampling_fix(patch=self.ratio).cuda()
        self.upsampling = MeanUpsample(patch=self.ratio).cuda()
        import cv2
        mask = torch.from_numpy(cv2.imread("/userhome/yjw/ddrm_plus/exp/datasets/mask/8.png")[:, :, 0])
        mask = (mask == 255.) * 1.
        h, w = mask.size()
        print(mask.size())
        self.mask = mask.reshape(1, 1, h, w).cuda()

    def H(self, vec):
        vec = vec.reshape(vec.size(0), -1)
        b, hwc = vec.size()
        hw = hwc // self.channels
        h = w = int(hw**0.5)
        vec = vec.reshape(vec.size(0), self.channels, h, w)
        out = self.meandown(color2gray(vec * self.mask))
        return out.reshape(out.size(0), -1)

    def H_pinv(self, vec):
        vec = vec.reshape(vec.size(0), -1)
        b, hwc = vec.size()
        hw = hwc // 1
        h = w = int(hw ** 0.5)
        vec = vec.reshape(vec.size(0), 1, h, w)
        out = gray2color(self.upsampling(vec)) * self.mask
        return out.reshape(out.size(0), -1)
    

class PD(H_functions):
    def __init__(self, channels=3, img_size=256, ratio=4): #ratio = 2 or 4
        self.channels = channels
        self.ratio = ratio
        self.img_size = img_size
#         self.meandown = MeanDownsampling_fix(patch=self.ratio).cuda()
        self.avgdown = torch.nn.AvgPool2d(self.ratio, self.ratio).cuda()
#         self.upsampling = MeanUpsample(patch=self.ratio).cuda()

    def H(self, vec):
        vec = vec.reshape(vec.size(0), 3, self.img_size, self.img_size)
        out = self.avgdown(vec)
        return out.reshape(out.size(0), -1)

    def H_pinv(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_size // self.ratio, self.img_size // self.ratio)
        vec = vec.repeat(1, 1, self.ratio, self.ratio)
        vec = vec.reshape(vec.size(0), self.channels, self.ratio, self.img_size // self.ratio, self.ratio, self.img_size // self.ratio)
        out = vec.permute(0, 1, 3, 2, 5, 4).reshape(vec.size(0), self.channels, self.img_size, self.img_size)
        return out.reshape(out.size(0), -1)


class MaxPooling(H_functions):
    def __init__(self, channels=3, ratio=4): #ratio = 2 or 4
        self.channels = channels
        self.ratio = ratio
        self.maxdown = torch.nn.MaxPool2d(self.ratio, self.ratio)
        self.upsampling = MeanUpsample(patch=self.ratio).cuda()


    def H(self, vec):
        vec = vec.reshape(vec.size(0), -1)
        b, hwc = vec.size()
        hw = hwc // self.channels
        h = w = int(hw**0.5)
        vec = vec.reshape(vec.size(0), self.channels, h, w)
        out = self.maxdown(vec)
        return out.reshape(out.size(0), -1)

    def H_pinv(self, vec):
        vec = vec.reshape(vec.size(0), -1)
        b, hwc = vec.size()
        hw = hwc // self.channels
        h = w = int(hw ** 0.5)
        vec = vec.reshape(vec.size(0), self.channels, h, w)
        out = self.upsampling(vec)
        return out.reshape(out.size(0), -1)
    

class OneBit(H_functions):
    def __init__(self):
        self.channels = 3

    def H(self, vec):
        return torch.sign(vec)

    def H_pinv(self, vec):
        return vec
    

def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())
    

class JPEG(H_functions):
    def __init__(self, channels, img_dim, qf, device):
        self.img_dim = img_dim
        self.channels = channels
        self.qf = qf
        self.device = device

    def H(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_dim, self.img_dim)
        vec = (vec + 1.0) / 2.0
        
        img = tensor2uint(vec)
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf])
        img = cv2.imdecode(encimg, 1)
        img = uint2tensor4(img)
        
        img = img.reshape(img.size(0), -1)
        vec = img * 2.0 - 1.0
        return vec.to(self.device)

    def H_pinv(self, vec):
        return vec

# quantification
class Quant(H_functions):
    def __init__(self, channels, img_dim, qf, device):
        self.img_dim = img_dim
        self.channels = channels
        self.qf = qf
        self.device = device

    def H(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_dim, self.img_dim)
        vec = (vec + 1.0) / 2.0
        
        img = tensor2uint(vec)
#         result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf])
#         img = cv2.imdecode(encimg, 1)
        img = np.uint8(img / self.qf) * self.qf
        img = uint2tensor4(img)
        
        img = img.reshape(img.size(0), -1)
        vec = img * 2.0 - 1.0
        return vec.to(self.device)

    def H_pinv(self, vec):
        return vec
    
class SRJPEG(H_functions):
    def __init__(self, channels, img_dim, device, qf=30, ratio=4):
        self.img_dim = img_dim
        self.channels = channels
        self.qf = qf
        self.device = device
        self.ratio = ratio
        self.avgdown = torch.nn.AvgPool2d(self.ratio, self.ratio)
        self.upsampling = MeanUpsample(patch=self.ratio).cuda()

    def H(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_dim, self.img_dim)
        vec = self.avgdown(vec)
        vec = (vec + 1.0) / 2.0
        
        img = tensor2uint(vec)
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf])
        img = cv2.imdecode(encimg, 1)
        img = uint2tensor4(img)
        
        img = img.reshape(img.size(0), -1)
        vec = img * 2.0 - 1.0
        return vec.to(self.device)

    def H_pinv(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_dim // self.ratio, self.img_dim // self.ratio)
        out = self.upsampling(vec)
        return out.reshape(out.size(0), -1)
    

class BlurSR(H_functions):
    def __init__(self, channels, img_dim, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = 4
        self.kernel_size = 5
        self.sigma = 5
        self.blursr_op = BlurDown(self.ratio, self.kernel_size, self.sigma, device=device)
        self.blursr_pinv = PseudoInverse(self.ratio, self.channels, device=device)
        
        state_dict_op = torch.load("/userhome/yjw/ddrm_plus/functions/bd_op.pth")
        self.blursr_op.load_state_dict(state_dict_op)
        
        state_dict_pinv = torch.load("/userhome/yjw/ddrm_plus/functions/bd_pinv.pth")
        self.blursr_pinv.load_state_dict(state_dict_pinv)

    def H(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_dim, self.img_dim)
        
        img = self.blursr_op(vec)
    
        img = img.reshape(img.size(0), -1)
        return img

    def H_pinv(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_dim // self.ratio, self.img_dim // self.ratio)
        
        img = self.blursr_pinv(vec)
        
        img = img.reshape(img.size(0), -1)
        return img
    

class BlurSRJPEG(H_functions):
    def __init__(self, channels, img_dim, device, qf=70, ratio=4):
        self.img_dim = img_dim
        self.channels = channels
        self.qf = qf
        self.device = device
        self.ratio = ratio
#         self.avgdown = torch.nn.AvgPool2d(self.ratio, self.ratio)
#         self.upsampling = MeanUpsample(patch=self.ratio).cuda()
        
        self.kernel_size = 5
        self.sigma = 5
        self.blursr_op = BlurDown(self.ratio, self.kernel_size, self.sigma, device=device)
        self.blursr_pinv = PseudoInverse(self.ratio, self.channels, device=device)
        
        state_dict_op = torch.load("/userhome/yjw/ddgm_latest/functions/bd{}_op.pth".format(self.ratio))
        self.blursr_op.load_state_dict(state_dict_op)
        
        state_dict_pinv = torch.load("/userhome/yjw/ddgm_latest/functions/bd{}_pinv.pth".format(self.ratio))
        self.blursr_pinv.load_state_dict(state_dict_pinv)

    def H(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_dim, self.img_dim)
        vec = self.blursr_op(vec)
        vec = (vec + 1.0) / 2.0
        
        img = tensor2uint(vec)
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf])
        img = cv2.imdecode(encimg, 1)
        img = uint2tensor4(img)
        
        img = img.reshape(img.size(0), -1)
        vec = img * 2.0 - 1.0
        return vec.to(self.device)

    def H_pinv(self, vec):
        vec = vec.reshape(vec.size(0), self.channels, self.img_dim // self.ratio, self.img_dim // self.ratio)
        out = self.blursr_pinv(vec)
        return out.reshape(out.size(0), -1)
    
    
#a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,
                        1)).view(v.shape[0], M.shape[0])

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out

#Inpainting
class Inpainting(H_functions):
    def __init__(self, channels, img_dim, missing_indices, device):
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.ones(channels * img_dim**2 - missing_indices.shape[0]).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = torch.Tensor([i for i in range(channels * img_dim**2) if i not in missing_indices]).to(device).long()

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_0, sigma_t, eta):

        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]

        singulars = self._singulars
        lambda_t = torch.ones(temp.size(1), device=vec.device)
        temp_singulars = torch.zeros(temp.size(1), device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_0)

        lambda_t = lambda_t.reshape(1, -1)
        out = out * lambda_t

        result = torch.zeros_like(temp)
        result[:, self.kept_indices] = out[:, :self.kept_indices.shape[0]]
        result[:, self.missing_indices] = out[:, self.kept_indices.shape[0]:]
        return result.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Lambda_noise(self, vec, a, sigma_0, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out_vec = torch.zeros_like(temp_vec)
        out_vec[:, :self.kept_indices.shape[0]] = temp_vec[:, self.kept_indices]
        out_vec[:, self.kept_indices.shape[0]:] = temp_vec[:, self.missing_indices]

        temp_eps = epsilon.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out_eps = torch.zeros_like(temp_eps)
        out_eps[:, :self.kept_indices.shape[0]] = temp_eps[:, self.kept_indices]
        out_eps[:, self.kept_indices.shape[0]:] = temp_eps[:, self.missing_indices]

        singulars = self._singulars
        d1_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * eta
        d2_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp_singulars = torch.zeros(temp_vec.size(1), device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_0 ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        out_vec = out_vec * d1_t
        out_eps = out_eps * d2_t

        result_vec = torch.zeros_like(temp_vec)
        result_vec[:, self.kept_indices] = out_vec[:, :self.kept_indices.shape[0]]
        result_vec[:, self.missing_indices] = out_vec[:, self.kept_indices.shape[0]:]
        result_vec = result_vec.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

        result_eps = torch.zeros_like(temp_eps)
        result_eps[:, self.kept_indices] = out_eps[:, :self.kept_indices.shape[0]]
        result_eps[:, self.missing_indices] = out_eps[:, self.kept_indices.shape[0]:]
        result_eps = result_eps.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)
        
        return result_vec + result_eps

#Denoising
class Denoising(H_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = torch.ones(channels * img_dim**2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
    
    def Lambda(self, vec, a, sigma_0, sigma_t, eta):
        if sigma_t < a * sigma_0:
            factor = (sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_0).item()
            return vec * factor
        else:
            return vec
    
    def Lambda_noise(self, vec, a, sigma_0, sigma_t, eta, epsilon):
        if sigma_t >= a * sigma_0:
            factor = torch.sqrt(sigma_t ** 2 - a ** 2 * sigma_0 ** 2).item()
            return vec * factor
        else:
            return vec * sigma_t * eta 

#Super Resolution
class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2-1):
            patches[:, :, :, idx+1] = temp[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1].view(vec.shape[0], self.channels, -1)
        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :, :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2-1):
            recon[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1] = patches[:, :, :, idx+1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_0, sigma_t, eta):
        singulars = self.singulars_small
        
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        
        lambda_t = torch.ones(self.ratio ** 2, device=vec.device)
        
        temp = torch.zeros(self.ratio ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.
        
        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
#             eta = 0.
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_0)
            
        lambda_t = lambda_t.reshape(1, 1, 1, -1)
#         print("lambda_t:", lambda_t)
#         print("V:", self.V_small)
#         print(lambda_t.size(), self.V_small.size())
#         print("Sigma_t:", torch.matmul(torch.matmul(self.V_small, torch.diag(lambda_t.reshape(-1))), self.Vt_small))
        patches = patches * lambda_t
        
        
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1))
        
        patches = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = patches.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        return patches

    def Lambda_noise(self, vec, a, sigma_0, sigma_t, eta, epsilon):
        singulars = self.singulars_small
        
        patches_vec = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches_vec = patches_vec.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches_vec = patches_vec.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        patches_eps = epsilon.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches_eps = patches_eps.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches_eps = patches_eps.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        d1_t = torch.ones(self.ratio ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.ratio ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5
        
        temp = torch.zeros(self.ratio ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.
        
#         eta = 0.
        if a != 0 and sigma_0 != 0:
            
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index  * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)
             
#             eta = 0.
            change_index = (sigma_t > a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(change_index * (sigma_t ** 2 - a ** 2 * sigma_0 ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)
            
#             eta = 0.85
            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5
#             d1_t = d1_t * (-change_index + 1.0) + change_index  * sigma_t
#             d2_t = d2_t * (-change_index + 1.0)
        
        d1_t = d1_t.reshape(1, 1, 1, -1)
        d2_t = d2_t.reshape(1, 1, 1, -1)
#         print("d1_t:", d1_t)
#         print("d2_t:", d2_t)
        patches_vec = patches_vec * d1_t
        patches_eps = patches_eps * d2_t
        
        patches_vec = torch.matmul(self.V_small, patches_vec.reshape(-1, self.ratio**2, 1))
        
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches_vec = patches_vec.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        patches_eps = torch.matmul(self.V_small, patches_eps.reshape(-1, self.ratio**2, 1))
        
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches_eps = patches_eps.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
#         return self.V(patches_vec) + patches_eps
        return patches_vec + patches_eps
    

#Colorization
class Colorization(H_functions):
    def __init__(self, img_dim, device):
        self.channels = 3
        self.img_dim = img_dim
        #Do the SVD for the per-pixel matrix
        H = torch.Tensor([[0.3333, 0.3334, 0.3333]]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        #get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1) #shape: B, WH, C'
        #multiply each needle by the small V
        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels) #shape: B, WH, C
        #permute back to vector representation
        recon = needles.permute(0, 2, 1) #shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        #get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1) #shape: B, WH, C
        #multiply each needle by the small V transposed
        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels) #shape: B, WH, C'
        #reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        temp[:, :self.img_dim**2] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_0, sigma_t, eta):
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)

        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels)

        singulars = self.singulars_small
        lambda_t = torch.ones(self.channels, device=vec.device)
        temp = torch.zeros(self.channels, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                        singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_0)

        lambda_t = lambda_t.reshape(1, 1, self.channels)
        needles = needles * lambda_t

        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels)

        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def Lambda_noise(self, vec, a, sigma_0, sigma_t, eta, epsilon):
        singulars = self.singulars_small

        needles_vec = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)
        needles_epsilon = epsilon.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)

        d1_t = torch.ones(self.channels, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.channels, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp = torch.zeros(self.channels, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_0 ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, 1, self.channels)
        d2_t = d2_t.reshape(1, 1, self.channels)

        needles_vec = needles_vec * d1_t
        needles_epsilon = needles_epsilon * d2_t

        needles_vec = torch.matmul(self.V_small, needles_vec.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels)
        recon_vec = needles_vec.permute(0, 2, 1).reshape(vec.shape[0], -1)

        needles_epsilon = torch.matmul(self.V_small, needles_epsilon.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1,self.channels)
        recon_epsilon = needles_epsilon.permute(0, 2, 1).reshape(vec.shape[0], -1)
        
        return recon_vec + recon_epsilon

#Walsh-Hadamard Compressive Sensing
class WalshHadamardCS(H_functions):
    def fwht(self, vec): #the Fast Walsh Hadamard Transform is the same as its inverse
        a = vec.reshape(vec.shape[0], self.channels, self.img_dim**2)
        h = 1
        while h < self.img_dim**2:
            a = a.reshape(vec.shape[0], self.channels, -1, h * 2)
            b = a.clone()
            a[:, :, :, :h] = b[:, :, :, :h] + b[:, :, :, h:2*h]
            a[:, :, :, h:2*h] = b[:, :, :, :h] - b[:, :, :, h:2*h]
            h *= 2
        a = a.reshape(vec.shape[0], self.channels, self.img_dim**2) / self.img_dim
        return a

    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = torch.ones(channels * img_dim**2 // ratio, device=device)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.channels, self.img_dim**2, device=vec.device)
        temp[:, :, self.perm] = vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        out[:, :self.channels * self.img_dim**2 // self.ratio] = vec.clone().reshape(vec.shape[0], -1)
        return out
    
    def Lambda(self, vec, a, sigma_0, sigma_t, eta):
        temp_vec = self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

        singulars = self._singulars
        lambda_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device)
        temp = torch.zeros(self.channels * self.img_dim ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_0)

        lambda_t = lambda_t.reshape(1, -1)
        temp_vec = temp_vec * lambda_t

        temp_out = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out[:, :, self.perm] = temp_vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp_out).reshape(vec.shape[0], -1)
        
    def Lambda_noise(self, vec, a, sigma_0, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim ** 2)[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)
        temp_eps = epsilon.clone().reshape(
            vec.shape[0], self.channels, self.img_dim ** 2)[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

        d1_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5
        
        singulars = self._singulars
        temp = torch.zeros(self.channels * self.img_dim ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_0 ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        
        temp_vec = temp_vec * d1_t
        temp_eps = temp_eps * d2_t

        temp_out_vec = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out_vec[:, :, self.perm] = temp_vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        temp_out_vec = self.fwht(temp_out_vec).reshape(vec.shape[0], -1)

        temp_out_eps = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out_eps[:, :, self.perm] = temp_eps.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        temp_out_eps = self.fwht(temp_out_eps).reshape(vec.shape[0], -1)
        
        return temp_out_vec + temp_out_eps

#Convolution-based super-resolution
class SRConv(H_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                        dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                        dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, stride = 1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        #build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride//2, img_dim + stride//2, stride):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                j_effective = j
                #reflective padding
                if j_effective < 0: j_effective = -j_effective-1
                if j_effective >= img_dim: j_effective = (img_dim - 1) - (j_effective - img_dim)
                #matrix building
                H_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1), self.singulars_small.reshape(1, small_dim)).reshape(small_dim**2)
        #permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)] + \
                                  [self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.small_dim**2, self.channels, device=vec.device)
        temp[:, :self.small_dim**2, :] = vec.clone().reshape(vec.shape[0], self.small_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

#Deblurring
class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO = 3e-2):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        #ZERO = 3e-2
        self.singulars_small_orig = self.singulars_small.clone()
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars_orig = torch.matmul(self.singulars_small_orig.reshape(img_dim, 1), self.singulars_small_orig.reshape(1, img_dim)).reshape(img_dim**2)
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)
        self._singulars_orig = self._singulars_orig[self._perm]

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
    
    def H_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self._singulars.repeat(1, 3).reshape(-1)
#         singulars = self._singulars_orig.repeat(1, 3).reshape(-1)
        
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        
#         temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def Lambda(self, vec, a, sigma_0, sigma_t, eta):
        temp_vec = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp_vec = self.img_by_mat(temp_vec, self.V_small).reshape(vec.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)

        singulars = self._singulars_orig
#         singulars = self._singulars
        lambda_t = torch.ones(self.img_dim ** 2, device=vec.device)
        temp_singulars = torch.zeros(self.img_dim ** 2, device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_0)

        lambda_t = lambda_t.reshape(1, -1, 1)
        temp_vec = temp_vec * lambda_t

        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = temp_vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Lambda_noise(self, vec, a, sigma_0, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(vec.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)
#         temp_vec = temp_vec.permute(0, 2, 1)

        temp_eps = epsilon.clone().reshape(vec.shape[0], self.channels, -1)
        temp_eps = temp_eps[:, :, self._perm].permute(0, 2, 1)
#         temp_eps = temp_eps.permute(0, 2, 1)

        singulars = self._singulars_orig
#         singulars = self._singulars
        d1_t = torch.ones(self.img_dim ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.img_dim ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp_singulars = torch.zeros(self.img_dim ** 2, device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_0 != 0:
            change_index = (sigma_t < a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_0 * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_0 ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1, 1)
        d2_t = d2_t.reshape(1, -1, 1)

        temp_vec = temp_vec * d1_t
        temp_eps = temp_eps * d2_t

        temp_vec_new = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp_vec_new[:, self._perm, :] = temp_vec
#         temp_vec_new = temp_vec
        out_vec = self.mat_by_img(self.V_small, temp_vec_new.permute(0, 2, 1))
        out_vec = self.img_by_mat(out_vec, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)

        temp_eps_new = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp_eps_new[:, self._perm, :] = temp_eps
#         temp_eps_new = temp_eps
        out_eps = self.mat_by_img(self.V_small, temp_eps_new.permute(0, 2, 1))
        out_eps = self.img_by_mat(out_eps, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)

        return out_vec + out_eps

#Anisotropic Deblurring
class Deblurring2D(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix - kernel1
        H_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0]//2, i + kernel1.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small1[i, j] = kernel1[j - i + kernel1.shape[0]//2]
        #build 1D conv matrix - kernel2
        H_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0]//2, i + kernel2.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small2[i, j] = kernel2[j - i + kernel2.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(H_small1, some=False)
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(H_small2, some=False)
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small1.reshape(img_dim, 1), self.singulars_small2.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(out, self.V_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)





if __name__ == "__main__":
    factor = 4
    def bicubic_kernel(x, a=-0.5):
        if abs(x) <= 1:
            return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
        elif 1 < abs(x) and abs(x) < 2:
            return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
        else:
            return 0
    k = np.zeros((factor * 4))
    for i in range(factor * 4):
        x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    kernel = torch.from_numpy(k).float().to('cuda')
    H_funcs = SRConv(kernel / kernel.sum(), \
                        3, 256, 'cuda', stride = factor)
    H_funcs.H(x0_t.reshape(x0_t.size(0), -1))
    print()