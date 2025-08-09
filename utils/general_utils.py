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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

#SUMO
from decimal import Decimal, ROUND_HALF_UP
def quantize_by_decimal(value: float, decimals: int = 2) -> str:
    """
    使用Decimal进行精确量化，返回字符串作为key
    
    Args:
        value: 要量化的浮点数
        decimals: 保留的小数位数
    
    Returns:
        量化后的字符串表示
    """
    # 创建量化模式
    quantize_pattern = '0.' + '0' * decimals if decimals > 0 else '1'
    
    # 转换为Decimal并量化
    decimal_value = Decimal(str(value))
    quantized = decimal_value.quantize(Decimal(quantize_pattern), rounding=ROUND_HALF_UP)
    
    return str(quantized)

def build_quaternion(R):
    """
    将批量旋转矩阵转换为四元数 (支持任意批量维度)
    
    输入:
        R: 形状为 (..., 3, 3) 的旋转矩阵张量
    
    输出:
        q: 形状为 (..., 4) 的四元数张量 [w, x, y, z]
    """
    # 保存原始形状并展平
    original_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    batch_size = R_flat.size(0)
    
    q = torch.zeros((batch_size, 4), device=R.device, dtype=R.dtype)
    
    # 计算矩阵的迹
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    
    # 情况1: trace > 0 (最稳定的情况)
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        q[mask1, 0] = 0.25 * s  # qw
        q[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s  # qx
        q[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s  # qy
        q[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s  # qz
    
    # 情况2: R[0,0] 是最大的对角元素
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2
        q[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s  # qw
        q[mask2, 1] = 0.25 * s  # qx
        q[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s  # qy
        q[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s  # qz
    
    # 情况3: R[1,1] 是最大的对角元素
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2
        q[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s  # qw
        q[mask3, 1] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s  # qx
        q[mask3, 2] = 0.25 * s  # qy
        q[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s  # qz
    
    # 情况4: R[2,2] 是最大的对角元素
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2
        q[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s  # qw
        q[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s  # qx
        q[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s  # qy
        q[mask4, 3] = 0.25 * s  # qz
    
    # 确保四元数的实部为正（标准化约定）
    negative_w = q[:, 0] < 0
    q[negative_w] = -q[negative_w]
    
    # 恢复原始形状
    q = q.reshape(*original_shape, 4)
    
    return q

def weighted_quaternion_log_space(quaternions, weights):
    """
    在对数空间进行加权 (理论上更正确，但计算更复杂)
    """
    import torch.nn.functional as F
    # 确保符号一致
    quaternions = torch.where(quaternions[..., 0:1] >= 0, quaternions, -quaternions)
    
    # 映射到对数空间 (轴角表示)
    w = quaternions[..., 0:1].clamp(-1, 1)
    xyz = quaternions[..., 1:4]
    
    theta = 2 * torch.acos(torch.abs(w))
    sin_half_theta = torch.sin(theta / 2)
    safe_sin = torch.where(sin_half_theta.abs() < 1e-6, 
                          torch.ones_like(sin_half_theta), sin_half_theta)
    
    # 对数映射
    log_quaternions = (theta / 2) * (xyz / safe_sin)
    
    # 在对数空间加权
    weighted_log = torch.sum(log_quaternions * weights.unsqueeze(-1), dim=1)
    weight_sum = torch.sum(weights, dim=1, keepdim=True)
    avg_log = weighted_log / weight_sum
    
    # 映射回四元数空间
    angle = torch.norm(avg_log, dim=-1, keepdim=True)
    safe_angle = torch.where(angle < 1e-6, torch.ones_like(angle) * 1e-6, angle)
    
    cos_half = torch.cos(safe_angle)
    sin_half = torch.sin(safe_angle)
    axis = avg_log / safe_angle
    
    result = torch.cat([cos_half, sin_half * axis], dim=-1)
    return F.normalize(result, p=2, dim=-1)
