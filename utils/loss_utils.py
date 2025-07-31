#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

#SUMO
def E_temp(curr:dict,prev:dict, alpha=10.0,lambdas: dict = None):
    if lambdas is None:
        lambdas = {"opacity": 1.0, "scaling": 0.1, "features": 0.1}
    
    if 'xyz' in curr and 'xyz' in prev:
        pos_diff = curr["xyz"].detach()  - prev["xyz"].detach() 
        w = torch.exp(-alpha * torch.sum(pos_diff ** 2, dim=1, keepdim=True))
    else:
        w=1.0
    loss=0

    if 'opacity' in curr and 'opacity' in prev:
        opacity_diff = curr["opacity"] - prev["opacity"]
        loss += lambdas["opacity"] * torch.mean(torch.square(opacity_diff)) * w
    if 'scaling' in curr and 'scaling' in prev:
        scaling_diff = curr["scaling"] - prev["scaling"]
        loss += lambdas["scaling"] * torch.mean(torch.square(scaling_diff)) * w
    if 'features_dc' in curr and 'features_dc' in prev:
        features_dc_diff = curr["features_dc"] - prev["features_dc"]
        loss += lambdas["features"] * torch.mean(torch.square(features_dc_diff)) * w
    if 'features_rest' in curr and 'features_rest' in prev:
        features_rest_diff = curr["features_rest"]- prev["features_rest"]
        loss += lambdas["features"] * torch.mean(torch.square(features_rest_diff)) * w
    return loss[0].item()

from utils.general_utils import build_rotation
def E_smooth(curr:dict,prev:dict,indices_i,indices_j,alpha=10.0):
    pos_diff = curr["xyz"][indices_i] - prev["xyz"][indices_i]
    rot_delta=torch.matmul(build_rotation(curr["rotation"]),build_rotation(prev["rotation"]).transpose(1,2))
    threshold = 0.002  # 设置阈值
    w_i_t = torch.where(
    torch.sum(pos_diff**2, dim=1) > threshold,
    torch.zeros_like(pos_diff[:, 0]),
    torch.exp(-alpha * torch.sum(pos_diff**2, dim=1))
    )
    # 批量计算位移差
    p_diff_prev = prev["xyz"][indices_j] - prev["xyz"][indices_i]
    p_diff_curr = curr["xyz"][indices_j] - curr["xyz"][indices_i]
    
    # 批量矩阵乘法
    p_diff_transformed = torch.bmm(
        rot_delta[indices_i],
        p_diff_prev.unsqueeze(-1)
    ).squeeze(-1)

    count = len(indices_i)
    # 应用权重并计算总损失
    loss_per_edge = torch.sum((p_diff_transformed - p_diff_curr)**2, dim=1)

    total_loss = torch.sum(w_i_t * loss_per_edge)
    # total_loss = torch.sum( loss_per_edge)
    return total_loss / max(count, 1)

def E_scale(scales, min_scale=0.01):
    """
    带梯度计算的高斯尺度损失函数
    
    Args:
        scales (torch.Tensor): 尺度参数张量，需要requires_grad=True
        min_scale (float): 最小尺度阈值
    
    Returns:
        torch.Tensor: 尺度损失值（保持梯度）
    """
    # 使用torch.maximum来保持梯度信息
    max_scales = torch.maximum(torch.full_like(scales, min_scale), scales)
    loss_scale = torch.sum(max_scales)
    
    return loss_scale/scales.shape[0]