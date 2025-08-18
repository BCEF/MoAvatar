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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

#WDD
from scene.deform_model import positional_encoding,MLP,SIREN
#SUMO
from flame_pytorch import FLAME, parse_args
import trimesh
from deformation_graph import apply_deformation_to_gaussians,DeformationGraph,generate_deformation_graph,compute_deformation_transforms
from deformation_graph import get_deformation_info_fixed_influences,apply_deformation_to_gaussians_fix_influ,apply_deformation_to_gaussians2
from deformation_graph import DeformationTransforms
from sklearn.neighbors import NearestNeighbors #SUMO
from utils.general_utils import build_quaternion,weighted_quaternion_log_space
import time
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

from plyfile import PlyData, PlyElement
def storePly(path, xyz):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    one=np.ones_like(vertices['x'])*255
    colors = np.vstack([one, one, one]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def quatProduct_batch(q1, q2):
    r1 = q1[:,0] # [B]
    r2 = q2[:,0]
    v1 = torch.stack((q1[:,1], q1[:,2], q1[:,3]), dim=-1) #[B,3]
    v2 = torch.stack((q2[:,1], q2[:,2], q2[:,3]), dim=-1)

    r = r1 * r2 - torch.sum(v1*v2, dim=1) # [B]
    v = r1.unsqueeze(1) * v2 + r2.unsqueeze(1) * v1 + torch.linalg.cross(v1, v2) #[B,3]
    q = torch.stack((r, v[:,0], v[:,1], v[:,2]), dim=1)

    return q

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree,optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
         
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()


        #WDD
        #新增
        self._xyz_0= torch.empty(0)
        self._xyz_t= torch.empty(0)

        self._rotation_0 = torch.empty(0)
        self._rotation_t = torch.empty(0)
        
        self._scaling_0=torch.empty(0)
        self._scaling_t=torch.empty(0)

        # 初始化位置编码相关参数
        self.num_freqs = 8
        self.include_input = True
        self.xyz_encoder = lambda xyz: positional_encoding(
            xyz, num_freqs=self.num_freqs, include_input=self.include_input
        )

        #SUMO
        self.base_xyz=None
        self.dg=None
        self.temp_flame_vertices=None
        self.vertex_deformer=None
        # flame_config = parse_args()
        # self.flame_model = FLAME(flame_config).to("cuda")
        # flame_dim=flame_config.expression_params + flame_config.pose_params + flame_config.neck_params+flame_config.eye_params + flame_config.translation_params + flame_config.scale_params+1
        # self.influ=None
        self.influ_nums=20
        code_dim=1+(1+3+4)*self.influ_nums#t,w,q,t
        # 根据编码后的维度初始化 MLP
        dim_encoded = 3 * (1 + 2 * self.num_freqs)  # 51 维
        # self.xyz_mlp = MLP(input_dim=dim_encoded+code_dim, output_dim=3, hidden_dim=256, hidden_layers=8).to(device='cuda')
        # 替换
        self.xyz_mlp = SIREN(input_dim=dim_encoded+code_dim, output_dim=3, hidden_dim=256, hidden_layers=8, omega_0=3.0).to(device='cuda')

        #self.rot_mlp = SIREN(input_dim=dim_encoded, output_dim=4, hidden_dim=256, hidden_layers=8, omega_0=30.0).to(device='cuda')
        self.rot_mlp = MLP(input_dim=dim_encoded+code_dim, output_dim=4, hidden_dim=256, hidden_layers=8).to(device='cuda')

        #SUMO
        self.scale_mlp=MLP(input_dim=dim_encoded+code_dim, output_dim=3, hidden_dim=256, hidden_layers=8).to(device='cuda')

        #WDD
        self.inverse_deform_transforms={}

    #SUMO 初始化变形图相关参数
    def deform_init(self,dg_path):
        self.dg_path=dg_path
        self.dg = DeformationGraph()
        self.dg.load(dg_path)
        if self.base_xyz is None:
            self.base_xyz=self._xyz_0.detach().cpu().clone()
            # storePly("/home/momo/Desktop/data/0724/data/base_xyz.ply",self.base_xyz)
        if self.temp_flame_vertices is None:
            self.temp_flame_vertices = {}
        if self.vertex_deformer is None:
            self.vertex_deformer={}
      
    #获取所有参数
    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            # self._scaling,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            #WDD
            self._xyz_0,
            self._rotation_0, 
            self._scaling_0,  # 新增：保存缩放参数
            self.xyz_mlp.state_dict(),   # 新增：保存 MLP 权重
            self.rot_mlp.state_dict(),   # 新增：保存 MLP 权重
            self.scale_mlp.state_dict(), # 新增：保存 MLP 权重
            # self.canonical_flame_code,
            # self.temp_flame_vertices,
            self.dg_path,
            self.base_xyz,
            self.vertex_deformer,
            self.temp_flame_vertices,
            self.inverse_deform_transforms,
            self._edge_indices
            
        )
    
    #在step3恢复参数，不优化标准空间，加载变形图各种参数
    def restore_step3(self, model_args, training_args):
        (
            self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        # self._scaling,  
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        #WDD
        self._xyz_0, 
        self._rotation_0,
        self._scaling_0,  # 新增：恢复缩放参数
        xyz_mlp_state_dict,
        rot_mlp_state_dict,
        scale_mlp_state_dict,
        dg_path,
        base_xyz,
        self.vertex_deformer,
        self.temp_flame_vertices,
        self.inverse_deform_transforms,
        self._edge_indices
        ) = model_args
        self.training_setup_freeze_x0(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

        #WDD
        self.xyz_mlp.load_state_dict(xyz_mlp_state_dict)
        self.rot_mlp.load_state_dict(rot_mlp_state_dict)
        self.scale_mlp.load_state_dict(scale_mlp_state_dict)

        #SUMO
        self.deform_init(dg_path)
        self.base_xyz=base_xyz.cpu()
    
    #在step2恢复参数，优化标准空间
    def restore_step2(self, model_args, training_args):
        (
            self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        # self._scaling,  
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        #WDD
        self._xyz_0, 
        self._rotation_0,
        self._scaling_0,  # 新增：恢复缩放参数
        xyz_mlp_state_dict,
        rot_mlp_state_dict,
        scale_mlp_state_dict,
        dg_path,
        self.base_xyz,
        self.vertex_deformer,
        self.temp_flame_vertices,
        self.inverse_deform_transforms,
        self._edge_indices
        ) = model_args
        self.training_setup(training_args)

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

        #WDD
        self.xyz_mlp.load_state_dict(xyz_mlp_state_dict)
        self.rot_mlp.load_state_dict(rot_mlp_state_dict)
        self.scale_mlp.load_state_dict(scale_mlp_state_dict)

        #SUMO
        self.deform_init(dg_path)
        self.base_xyz=self.base_xyz.cpu()
    
    #在step3恢复参数，加载step2的训练结果，只加载标准空间
    def restore_from_keyframe(self, model_args, training_args):
        (
            self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        # self._scaling,  
        self._opacity,
        max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        spatial_lr_scale,
        #WDD
        self._xyz_0, 
        self._rotation_0,
        self._scaling_0, 
        xyz_mlp_state_dict,
        rot_mlp_state_dict,
        scale_mlp_state_dict,
        dg_path,
        base_xyz,
        vertex_deformer,
        temp_flame_vertices,
        inverse_deform_transforms,
        _edge_indices
        ) = model_args
        # self.xyz_mlp.load_state_dict(xyz_mlp_state_dict)
        # self.rot_mlp.load_state_dict(rot_mlp_state_dict)
        # self.scale_mlp.load_state_dict(scale_mlp_state_dict)

        self._xyz_t = self._xyz_0.clone().detach().to("cuda")
        self._rotation_t=self._rotation_0.clone().detach().to("cuda")
        self._scaling_t = self._scaling_0.clone().detach().to("cuda") 
        
        self.training_setup_freeze_x0(training_args)
        self.deform_init(dg_path)

    #获取必要的参数，不保存缓存
    def capture_render(self):
        return (
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._xyz_0,
            self._rotation_0, 
            self._scaling_0,  
            self.xyz_mlp.state_dict(),   # 新增：保存 MLP 权重
            self.rot_mlp.state_dict(),   # 新增：保存 MLP 权重
            self.scale_mlp.state_dict(), # 新增：保存 MLP 权重
            self.dg,
            self.base_xyz,
            # self.vertex_deformer,
            # self.temp_flame_vertices  
        )
    
    #推理时调用的恢复参数
    def restore_render(self, model_args):
        (
        self._features_dc, 
        self._features_rest, 
        self._opacity,
        self._xyz_0, 
        self._rotation_0,
        self._scaling_0,  
        xyz_mlp_state_dict,
        rot_mlp_state_dict,
        scale_mlp_state_dict,
        self.dg,
        self.base_xyz,
        
        ) = model_args
        #WDD
        self.xyz_mlp.load_state_dict(xyz_mlp_state_dict)
        self.rot_mlp.load_state_dict(rot_mlp_state_dict)
        self.scale_mlp.load_state_dict(scale_mlp_state_dict)
        self.xyz_mlp.eval()
        self.rot_mlp.eval()
        self.scale_mlp.eval()
        self.vertex_deformer={}
        self.temp_flame_vertices={}

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation_t)
    
    @property
    def get_xyz(self):
        #WDD 
        return self._xyz_t
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation_t)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        

        self.spatial_lr_scale = spatial_lr_scale
        points=pcd.points
        colors=pcd.colors
        points,colors=_interpolation_densify(pcd.points, pcd.colors, 4.0)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))


        #WDD
        #训练的版本
        self._xyz_0 = nn.Parameter(fused_point_cloud.clone().detach().to("cuda"), requires_grad=True)
        self._xyz_t = fused_point_cloud.clone().detach().to("cuda")  # 纯数据副本


        self._rotation_0 = nn.Parameter(rots.requires_grad_(True))
        self._rotation_t = rots.clone().detach().to("cuda")  # 纯数据副本，不需要 grad

        self._scaling_0 = nn.Parameter(scales.requires_grad_(True))
        self._scaling_t = scales.clone().detach().to("cuda")  # 纯数据副本，不需要 grad
 
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
    
    #所有参数都优化
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

 

        # 移除 xyz group
        l = [
            {'params': self.xyz_mlp.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz_mlp"},
            {'params': self.rot_mlp.parameters(), 'lr':  training_args.rotation_lr*0.01, "name": "rot_mlp"},
            {'params': self.scale_mlp.parameters(), 'lr':  training_args.scaling_lr*0.01, "name": "scale_mlp"},
            {'params': [self._xyz_0], 'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'xyz_0'}, 
            {'params': [self._rotation_0], 'lr': training_args.rotation_lr, "name": "rotation_0"},
            {'params': [self._scaling_0], 'lr': training_args.scaling_lr, "name": "scaling_0"},
        
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"}
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"}
            ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
    
    #不优化标准空间
    def training_setup_freeze_x0(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 移除 xyz group
        l = [
            {'params': self.xyz_mlp.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz_mlp"},
            {'params': self.rot_mlp.parameters(), 'lr':  training_args.rotation_lr*0.01, "name": "rot_mlp"},
            {'params': self.scale_mlp.parameters(), 'lr':  training_args.scaling_lr*0.01, "name": "scale_mlp"},

        
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"}
            ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz_0":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling_0.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation_0.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    #WDD
    # 组合四元数，避免归一化误差
    def combine_quaternions(self,q_old, delta_q):
        q_old = q_old / (torch.norm(q_old, dim=-1, keepdim=True) + 1e-8)
        delta_q = delta_q / (torch.norm(delta_q, dim=-1, keepdim=True) + 1e-8)
        # Hamilton product
        w0, x0, y0, z0 = q_old.unbind(-1)
        w1, x1, y1, z1 = delta_q.unbind(-1)
        new_q = torch.stack([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1
        ], dim=-1)
        return new_q / (torch.norm(new_q, dim=-1, keepdim=True) + 1e-8)

    #WDD    
    # 前向传播函数，计算当前帧的动态点坐标
    def forward(self, codedict=None,update=False):
        encoded = self.xyz_encoder(self._xyz_0)    # 可能是 CPU
        batch_size = self._xyz_0.shape[0]

        deformer_path=codedict['deformer_path']
        kid=codedict['kid']
        t=codedict['t']

        if kid not in self.vertex_deformer or self._xyz_0.shape[0]!=self.vertex_deformer[kid].shape[0]:
            print("缓存被更新！！！！！")
            transforms=DeformationTransforms()
            transforms.load(deformer_path)
            result=get_deformation_info_fixed_influences(self.dg,self.base_xyz.cpu().clone().numpy(),transforms,self.influ_nums,'inverse_distance')
            weights=torch.as_tensor(np.array(result['weights']),dtype=torch.float32).to('cuda')
            RT=torch.as_tensor(np.array(result['RT']),dtype=torch.float32).to('cpu')
            quaternions, translations = build_quaternion(RT[..., :3, :3]), RT[..., :3, 3]

            self.vertex_deformer[kid]=torch.cat([weights.view(batch_size, -1), quaternions.to('cuda').view(batch_size, -1), translations.to('cuda').reshape(batch_size, -1)], dim=-1)

            # st=time.time()
            # deform_points=apply_deformation_to_gaussians(self.dg,self.base_xyz.cpu().clone().numpy(),transforms)
            # print(f"计算全局变形 kid {kid} in {time.time()-st:.2f} seconds")
            # st=time.time()
            deform_points=apply_deformation_to_gaussians2(self.dg,self.base_xyz.cpu().clone().numpy(),transforms)
            # print(f"计算全局变形2 kid {kid} in {time.time()-st:.2f} seconds")
            # st=time.time()
            # deform_points,self.influ=apply_deformation_to_gaussians_fix_influ(self.dg,self.base_xyz.cpu().clone().numpy(),transforms,influ=self.influ)
            # print(f"计算局部变形 kid {kid} in {time.time()-st:.2f} seconds")
            self.temp_flame_vertices[kid]=torch.as_tensor(deform_points).to(self._xyz_0.device)
            # test store deformed_ply
            # path=os.path.join("/home/momo/Desktop/data/0724/data","temp"+str(kid).zfill(2)+".ply")
            # storePly(path,deform_points['xyz'])

            inv_trans_path=os.path.join(os.path.dirname(deformer_path),"inv_transforms.json")
            inv_transform=DeformationTransforms()
            inv_transform.load(inv_trans_path)
            self.inverse_deform_transforms[kid]=inv_transform
        
        t= torch.as_tensor(codedict['t']).to(self._xyz_0.device).repeat(batch_size,1) # time parameter
        

        encoded = torch.cat([encoded, t,self.vertex_deformer[kid]], dim=-1)

        delta_xyz = self.xyz_mlp(encoded)  # 这时不会报错
        delta_xyz = delta_xyz.squeeze(0)

        _xyz_t = self.temp_flame_vertices[kid] + delta_xyz

        delta_rot = self.rot_mlp(encoded)  # 计算旋转
        delta_rot = delta_rot.squeeze(0)
        # SUMO
        _rotation_t=quatProduct_batch(self._rotation_0, delta_rot)  # 使用四元数乘法组合旋转
        delta_scale = self.scale_mlp(encoded)  # 计算缩放
        delta_scale = delta_scale.squeeze(0)
        _scaling_t =self._scaling_0 +delta_scale
        del delta_xyz, delta_rot, delta_scale
        torch.cuda.empty_cache()


        if update:
            self._xyz_t=_xyz_t
            self._rotation_t=_rotation_t
            self._scaling_t=_scaling_t
        return _xyz_t,_rotation_t,_scaling_t

    #只使用标准空间的xyz
    def forward_x0(self):
        self._xyz_t=self._xyz_0
        self._rotation_t=self._rotation_0
        self._scaling_t=self._scaling_0

    #推理时调用的前向渲染函数
    def forward_render(self, codedict=None):
        encoded = self.xyz_encoder(self._xyz_0)    # 可能是 CPU
        batch_size = self._xyz_0.shape[0]

        deformer_path=codedict['deformer_path']
        kid=codedict['kid']
        t=codedict['t']

        if kid not in self.vertex_deformer:
            
            transforms=DeformationTransforms()
            transforms.load(deformer_path)
            result=get_deformation_info_fixed_influences(self.dg,self.base_xyz.cpu().clone().numpy(),transforms,self.influ_nums,'inverse_distance')
            weights=torch.as_tensor(np.array(result['weights']),dtype=torch.float32).to('cuda')
            RT=torch.as_tensor(np.array(result['RT']),dtype=torch.float32).to('cpu')
            quaternions, translations = build_quaternion(RT[..., :3, :3]), RT[..., :3, 3]
            self.vertex_deformer[kid]=torch.cat([weights.view(batch_size, -1), quaternions.to('cuda').view(batch_size, -1), translations.to('cuda').reshape(batch_size, -1)], dim=-1)
            
            deform_points=apply_deformation_to_gaussians2(self.dg,self.base_xyz.cpu().clone().numpy(),transforms)
            
            self.temp_flame_vertices[kid]=torch.as_tensor(deform_points).to(self._xyz_0.device)
            
        t= torch.as_tensor(codedict['t']).to(self._xyz_0.device).repeat(batch_size,1) # time parameter
        

        encoded = torch.cat([encoded, t,self.vertex_deformer[kid]], dim=-1)

        delta_xyz = self.xyz_mlp(encoded)  # 这时不会报错
        delta_xyz = delta_xyz.squeeze(0)

        _xyz_t = self.temp_flame_vertices[kid] + delta_xyz

        delta_rot = self.rot_mlp(encoded)  # 计算旋转
        delta_rot = delta_rot.squeeze(0)
        # SUMO
        _rotation_t=quatProduct_batch(self._rotation_0, delta_rot)  # 使用四元数乘法组合旋转
        delta_scale = self.scale_mlp(encoded)  # 计算缩放
        delta_scale = delta_scale.squeeze(0)
        _scaling_t =self._scaling_0 +delta_scale
        del delta_xyz, delta_rot, delta_scale
        torch.cuda.empty_cache()

        self._xyz_t=_xyz_t
        self._rotation_t=_rotation_t
        self._scaling_t=_scaling_t

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        #WDD
        #xyz = self._xyz.detach().cpu().numpy()
        xyz = self.get_xyz.detach().cpu().numpy()
        rotation = self._rotation_t.detach().cpu().numpy()
        scale = self._scaling_t.detach().cpu().numpy()

        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))

            
        #SUMO
        self._xyz_0 = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_0 = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling_0 = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._xyz_t = self._xyz_0.clone().detach().to("cuda")  # 纯数据副本
        self._rotation_t = self._rotation_0.clone().detach().to("cuda")  # 纯数据副本
        self._scaling_t = self._scaling_0.clone().detach().to("cuda")  # 纯数据副本

        self.active_sh_degree = self.max_sh_degree

    #SUMO 完善加载ply时未能初始化的参数
    def fixup_params(self,cam_infos,spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self.tmp_radii = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:

            #WDD
            # 跳过 MLP 的参数，不参与 prune
            if group.get("name", "") == "xyz_mlp":
                continue
            if group.get("name", "") == "rot_mlp":
                continue
            if group.get("name", "") == "scale_mlp":
                continue


            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            #WDD
            # 跳过 MLP 的参数，不参与 prune
            if group.get("name", "") == "xyz_mlp":
                continue
            if group.get("name", "") == "rot_mlp":
                continue
            if group.get("name", "") == "scale_mlp":
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)


        #WDD
        self._xyz_0 = optimizable_tensors["xyz_0"] 
        self._xyz_t = self._xyz_0.detach().clone() #重新初始化
        self._rotation_0 = optimizable_tensors["rotation_0"]
        self._rotation_t = self._rotation_0.detach().clone() #重新初始化
        #SUMO
        self._scaling_0 = optimizable_tensors["scaling_0"] 
        self._scaling_t = self._scaling_0.detach().clone() 

         
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

        #SUMO
        self.base_xyz=self.base_xyz[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            #WDD 
            # 跳过 MLP 参数组（可能有多个 param，不参与 densify）
            if group.get("name", "") == "xyz_mlp":
                continue
            if group.get("name", "") == "rot_mlp":
                continue

            if group.get("name", "") == "scale_mlp":
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz_0, new_features_dc, new_features_rest, new_opacities, new_scaling_0, new_rotation_0, new_tmp_radii):
        d = {
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        #WDD
        "xyz_0":new_xyz_0,
        "rotation_0" : new_rotation_0,
        "scaling_0" : new_scaling_0
        
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        #WDD
        self._xyz_0 = optimizable_tensors["xyz_0"]
        self._xyz_t = self._xyz_0.detach().clone()
        self._rotation_0 = optimizable_tensors["rotation_0"]
        self._rotation_t = self._rotation_0.detach().clone()
        self._scaling_0 = optimizable_tensors["scaling_0"]
        self._scaling_t = self._scaling_0.detach().clone()


        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        
        
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #SUMO
        # self.base_xyz=torch.cat((self.base_xyz,new_xyz_0.clone().detach().cpu()))
        self.base_xyz=self._xyz_0.clone().detach().cpu()

    #WDD
    def densify_and_split(self, grads, grad_threshold, scene_extent,kid,N=2):
        # self.canonical_pos_encoded

        n_init_points = self._xyz_0.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        
        rots = build_rotation(self._rotation_t[selected_pts_mask]).repeat(N,1,1)
        new_xyz_t= torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_xyz_0=apply_deformation_to_gaussians2(self.dg, new_xyz_t.cpu().numpy(), self.inverse_deform_transforms[kid])
        new_xyz_0 = torch.as_tensor(new_xyz_0).to(new_xyz_t.device)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation_0[selected_pts_mask].repeat(N,1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz_0, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        #WDD
        new_xyz_0 = self._xyz_0[selected_pts_mask]
        new_rotation_0 = self._rotation_0[selected_pts_mask]
        #SUMO
        new_scaling_0 = self._scaling_0[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        
        
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz_0, new_features_dc, new_features_rest, new_opacities, new_scaling_0, new_rotation_0, new_tmp_radii)

    #WDD 增加kid
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii,kid):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        # self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent,kid)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    #SUMO 构建高斯图
    def build_knn_graph(self, k=4):
        points = self.get_xyz.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
        _, indices = nbrs.kneighbors(points)

        edge_dict = {}
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # 跳过自身
                edge_dict.setdefault(i, set()).add(j)
                edge_dict.setdefault(j, set()).add(i)
        
        # 预处理边为索引对
        edge_pairs = []
        for i, neighbors in edge_dict.items():
            if neighbors:  # 确保邻居集合不为空
                edge_pairs.extend([(i, j) for j in neighbors])
        
        # 处理没有边的情况
        if not edge_pairs:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        
        # 解压边对并转换为张量
        indices_i, indices_j = zip(*edge_pairs)
        
        # 确保索引为整数
        indices_i = [int(i) for i in indices_i]
        indices_j = [int(j) for j in indices_j]
        
        # 转换为张量
        indices_i = torch.tensor(indices_i, dtype=torch.long)
        indices_j = torch.tensor(indices_j, dtype=torch.long)
        
        indices_i = indices_i.cuda()
        indices_j = indices_j.cuda()

        self._edge_indices = torch.stack((indices_i, indices_j), dim=0)
    
#三个插值函数，pcd点云稀疏时调用
def _interpolation_densify(points, colors, density_factor):
    """基于最近邻插值的稠密化"""
    n_points = points.shape[0]
    n_new_points = int(n_points * (density_factor - 1))
    
    if n_new_points <= 0:
        return points, colors
    
    # 构建KNN
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    new_points = []
    new_colors = []
    
    # 在相邻点之间插值
    for i in range(n_new_points):
        idx = i % n_points
        neighbor_idx = indices[idx, 1]  # 最近邻（排除自己）
        
        # 线性插值
        t = np.random.uniform(0.2, 0.8)  # 插值参数
        new_point = points[idx] * (1 - t) + points[neighbor_idx] * t
        new_color = colors[idx] * (1 - t) + colors[neighbor_idx] * t
        
        new_points.append(new_point)
        new_colors.append(new_color)
    
    if new_points:
        densified_points = np.vstack([points, np.array(new_points)])
        densified_colors = np.vstack([colors, np.array(new_colors)])
    else:
        densified_points, densified_colors = points, colors
    
    return densified_points, densified_colors

def _random_sampling_densify(points, colors, density_factor, noise_scale):
    """基于随机采样的稠密化"""
    n_points = points.shape[0]
    n_new_points = int(n_points * (density_factor - 1))
    
    if n_new_points <= 0:
        return points, colors
    
    # 计算点云的平均距离作为噪声尺度参考
    if points.shape[0] > 1:
        distances = pdist(points)
        avg_distance = np.mean(distances)
        noise_std = avg_distance * noise_scale
    else:
        noise_std = noise_scale
    
    # 随机选择源点并添加噪声
    source_indices = np.random.choice(n_points, n_new_points, replace=True)
    
    new_points = []
    new_colors = []
    
    for idx in source_indices:
        # 添加高斯噪声
        noise = np.random.normal(0, noise_std, 3)
        new_point = points[idx] + noise
        new_color = colors[idx] + np.random.normal(0, 0.01, 3)  # 颜色也添加小幅噪声
        new_color = np.clip(new_color, 0, 1)  # 确保颜色在有效范围内
        
        new_points.append(new_point)
        new_colors.append(new_color)
    
    densified_points = np.vstack([points, np.array(new_points)])
    densified_colors = np.vstack([colors, np.array(new_colors)])
    
    return densified_points, densified_colors

def _adaptive_sampling_densify(points, colors, density_factor, noise_scale):
    """基于局部密度的自适应稠密化"""
    n_points = points.shape[0]
    n_new_points = int(n_points * (density_factor - 1))
    
    if n_new_points <= 0 or n_points < 3:
        return points, colors
    
    # 计算局部密度
    nbrs = NearestNeighbors(n_neighbors=min(10, n_points), algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # 计算每个点的局部密度（平均到最近邻的距离）
    local_densities = np.mean(distances[:, 1:], axis=1)  # 排除距离为0的自己
    
    # 密度越低，被选中的概率越高
    weights = 1.0 / (local_densities + 1e-8)
    weights = weights / np.sum(weights)
    
    # 根据权重选择源点
    source_indices = np.random.choice(n_points, n_new_points, replace=True, p=weights)
    
    new_points = []
    new_colors = []
    
    for idx in source_indices:
        # 在局部邻域内采样
        noise_std = local_densities[idx] * noise_scale
        noise = np.random.normal(0, noise_std, 3)
        new_point = points[idx] + noise
        
        # 颜色插值（从最近邻）
        neighbor_idx = indices[idx, 1]
        t = np.random.uniform(0.3, 0.7)
        new_color = colors[idx] * (1 - t) + colors[neighbor_idx] * t
        
        new_points.append(new_point)
        new_colors.append(new_color)
    
    densified_points = np.vstack([points, np.array(new_points)])
    densified_colors = np.vstack([colors, np.array(new_colors)])
    
    return densified_points, densified_colors


