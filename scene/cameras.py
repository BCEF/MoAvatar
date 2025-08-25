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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 # 添加新的参数
                 kid=0,timecode=0.0,alpha=None, head_mask=None, mouth_mask=None,
                 shape_params=None
                #  shape_param=None,exp_param=None, global_rotation=None, jaw_pose=None, neck_pose=None, eyes_pose=None, transl=None,scale_factor=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        #SUMO
        self.kid=kid
        self.timecode=timecode
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        # if resized_image_rgb.shape[0] == 4:
        #     self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        # else: 
        #     self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        # if train_test_exp and is_test_view:
        #     if is_test_dataset:
        #         self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
        #     else:
        #         self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        #SUMO
        # 处理alpha图像
        self.alpha = None
        if alpha is not None:
            try:
                if isinstance(alpha, torch.Tensor):
                    # 如果已经是tensor，需要调整分辨率
                    alpha_resized = torch.nn.functional.interpolate(
                        alpha.unsqueeze(0), size=resolution, mode='nearest'
                    ).squeeze(0)
                else:
                    # 如果是PIL图像，先转换
                    alpha_resized = PILtoTorch(alpha, resolution)
                
                # 取第一个通道（如果是灰度图可能只有一个通道，如果是RGB取第一个）
                if alpha_resized.shape[0] >= 1:
                    self.alpha = alpha_resized[0:1, ...].clamp(0.0, 1.0).to(self.data_device)
                else:
                    self.alpha = alpha_resized.clamp(0.0, 1.0).to(self.data_device)
                
                alpha_binary = (self.alpha > 0.5).float()
        
                self.alpha = alpha_binary.to(self.data_device)
            except Exception as e:
                print(f"Warning: Failed to process alpha image: {e}")
                self.alpha = None

        # 处理head_mask图像
        self.head_mask = None
        if head_mask is not None:
            try:
                if isinstance(head_mask, torch.Tensor):
                    head_mask_resized = torch.nn.functional.interpolate(
                        head_mask.unsqueeze(0), size=resolution, mode='nearest'
                    ).squeeze(0)
                else:
                    head_mask_resized = PILtoTorch(head_mask, resolution)
                
                if head_mask_resized.shape[0] >= 1:
                    self.head_mask = head_mask_resized[0:1, ...].clamp(0.0, 1.0).to(self.data_device)
                else:
                    self.head_mask = head_mask_resized.clamp(0.0, 1.0).to(self.data_device)
            except Exception as e:
                print(f"Warning: Failed to process head mask: {e}")
                self.head_mask = None

        # 处理mouth_mask图像
        self.mouth_mask = None
        if mouth_mask is not None:
            try:
                if isinstance(mouth_mask, torch.Tensor):
                    mouth_mask_resized = torch.nn.functional.interpolate(
                        mouth_mask.unsqueeze(0), size=resolution, mode='nearest'
                    ).squeeze(0)
                else:
                    mouth_mask_resized = PILtoTorch(mouth_mask, resolution)
                
                if mouth_mask_resized.shape[0] >= 1:
                    self.mouth_mask = mouth_mask_resized[0:1, ...].clamp(0.0, 1.0).to(self.data_device)
                else:
                    self.mouth_mask = mouth_mask_resized.clamp(0.0, 1.0).to(self.data_device)
            except Exception as e:
                print(f"Warning: Failed to process mouth mask: {e}")
                self.mouth_mask = None

        #SUMO
        self.shape_params=shape_params

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    # SUMO
    def has_alpha(self):
        return self.alpha is not None
    
    def has_head_mask(self):
        return self.head_mask is not None
    
    def has_mouth_mask(self):
        return self.mouth_mask is not None
    
    def has_flame_params(self):
        # return any([
        #     self.exp_param is not None,
        #     self.global_rotation is not None,
        #     self.jaw_pose is not None,
        #     self.neck_pose is not None,
        #     self.eyes_pose is not None,
        #     self.transl is not None,
        #     self.scale_factor is not None
        # ])
        return self.shape_params is not None
    
    def get_flame_params(self):
        # """返回所有FLAME参数的字典"""
        
        # return {
        #     'shape':self.shape_param,
        #     'exp': self.exp_param,
        #     'global_rotation': self.global_rotation,
        #     'jaw': self.jaw_pose,
        #     'neck': self.neck_pose,
        #     'eyes': self.eyes_pose,
        #     'transl': self.transl,
        #     'scale_factor': self.scale_factor
        # }
        return self.shape_params
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

