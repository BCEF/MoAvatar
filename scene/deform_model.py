#WDD 根据苏妺的代码改写得到

import sys
import pickle
import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F



#这个函数是用来进行位置编码的
#它的功能是将输入的坐标进行正弦和余弦变
import torch
import math

def positional_encoding(xyz, num_freqs=8, include_input=True, log_sampling=True, alpha=None):
    """
    NeRF 风格的标准位置编码函数（设备安全版）
    :param xyz: (N, D) 或 (B, N, D) 输入点坐标，可是 nn.Parameter
    :param num_freqs: 频率个数 (默认 8)
    :param include_input: 是否包含原始 xyz
    :param log_sampling: 是否使用指数采样频率 (2^k)
    :param alpha: 渐进频率训练 (0~num_freqs)，None 时全部启用
    :return: 编码特征 (N, D_out) 或 (B, N, D_out)
    """
    device = xyz.device  # 确保所有张量在同一设备上

    # 统一输入为 (B, N, D)
    if xyz.ndim == 2:
        xyz = xyz.unsqueeze(0)  # (1, N, D)
    B, N, D = xyz.shape

    # 生成频率带宽 (放在 xyz.device 上)
    if log_sampling:
        freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs, device=device)
    else:
        freq_bands = torch.linspace(2. ** 0., 2. ** (num_freqs - 1), num_freqs, device=device)

    # 扩展为 (B, N, num_freqs, D)
    xyz_expanded = xyz.unsqueeze(-2)                  # (B, N, 1, D)
    freqs = freq_bands.view(1, 1, -1, 1)              # (1, 1, num_freqs, 1)
    xyz_scaled = xyz_expanded * freqs                 # (B, N, num_freqs, D)

    # 正弦和余弦编码
    sin_enc = torch.sin(xyz_scaled)
    cos_enc = torch.cos(xyz_scaled)

    # 渐进频率训练 (progressive frequency training)
    if alpha is not None:
        weights = (1 - torch.cos(math.pi * torch.clamp(alpha - torch.arange(num_freqs, device=device), 0, 1))) * 0.5
        weights = weights.view(1, 1, -1, 1)
        sin_enc = sin_enc * weights
        cos_enc = cos_enc * weights

    # 拼接结果
    encodings = [sin_enc, cos_enc]
    if include_input:
        encodings = [xyz.unsqueeze(-2)] + encodings

    encoded = torch.cat(encodings, dim=-2)            # (B, N, 1+2*num_freqs, D)
    encoded = encoded.reshape(B, N, -1)               # (B, N, D_out)

    # 如果原始输入是 (N, D)，输出也转为 (N, D_out)
    if encoded.shape[0] == 1:
        encoded = encoded.squeeze(0)

    return encoded





#网络的基础类
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # 如果输入是 2D (N, D)，自动加上 batch 维
        if input.dim() == 2:
            input = input.unsqueeze(0)  # (1, N, D)
        batch_size, N_v, input_dim = input.shape

        input_ori = input.reshape(batch_size * N_v, -1)
        h = input_ori
        for l in self.fcs:
            h = l(h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)
        return output
    
#========================================================================================


# 单层 SIREN，带特殊初始化
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        # 避免 leaf variable in-place 报错
        with torch.no_grad():
            if self.is_first:
                # 第一层初始化范围大一些
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        # sin 激活，保留高频特征
        return torch.sin(self.omega_0 * self.linear(x))

# SIREN 主网络，替代原始 MLP
class SIREN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8, omega_0=30.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        # 第一层（特殊初始化）
        layers.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0))
        # 中间隐藏层
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        self.net = nn.Sequential(*layers)

        # 输出层（线性，不加 sin）
        self.final_linear = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():  # 避免 in-place 错误
            self.final_linear.weight.uniform_(
                -math.sqrt(6 / hidden_dim) / omega_0,
                 math.sqrt(6 / hidden_dim) / omega_0
            )

    def forward(self, input):
        # 兼容 (N, D) 和 (B, N, D) 输入
        if input.dim() == 2:
            input = input.unsqueeze(0)  # (1, N, D)
        batch_size, N_v, input_dim = input.shape

        x = input.reshape(batch_size * N_v, -1)
        h = self.net(x)
        output = self.final_linear(h)
        output = output.reshape(batch_size, N_v, -1)
        return output