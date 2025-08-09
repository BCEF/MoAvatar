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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
#SUMO
import torch,os
WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
    
    #SUMO
    # 加载alpha图像
    alpha = None
    if hasattr(cam_info, 'alpha_path') and cam_info.alpha_path and (os.path.exists(cam_info.alpha_path) or os.path.exists(cam_info.alpha_path.replace('.png','.jpg'))):
        try:
            if not os.path.exists(cam_info.alpha_path):
                cam_info.alpha_path=cam_info.alpha_path.replace('.png','.jpg')
            alpha_image = Image.open(cam_info.alpha_path)
            alpha=alpha_image
            # alpha = PILtoTensor(alpha_image)
        except Exception as e:
            print(f"Warning: Failed to load alpha image from {cam_info.alpha_path}: {e}")
            alpha = None
    
    # 加载head_mask图像
    head_mask = None
    if hasattr(cam_info, 'head_mask_path') and cam_info.head_mask_path and os.path.exists(cam_info.head_mask_path):
        try:
            head_mask_image = Image.open(cam_info.head_mask_path)
            head_mask=head_mask_image
            # head_mask = PILtoTensor(head_mask_image)
        except Exception as e:
            print(f"Warning: Failed to load head mask from {cam_info.head_mask_path}: {e}")
            head_mask = None
    
    # 加载mouth_mask图像
    mouth_mask = None
    if hasattr(cam_info, 'mouth_mask_path') and cam_info.mouth_mask_path and os.path.exists(cam_info.mouth_mask_path):
        try:
            mouth_mask_image = Image.open(cam_info.mouth_mask_path)
            mouth_mask=mouth_mask_image
            # mouth_mask = PILtoTensor(mouth_mask_image)
        except Exception as e:
            print(f"Warning: Failed to load mouth mask from {cam_info.mouth_mask_path}: {e}")
            mouth_mask = None
    
    # 提取FLAME参数
    shape_param=None
    exp_param = None
    global_rotation = None
    jaw_pose = None
    neck_pose = None
    eyes_pose = None
    transl = None
    scale_factor=None
    if hasattr(cam_info, 'flame_params') and cam_info.flame_params:
        flame_params = cam_info.flame_params
        try:
            if flame_params.get('shape') is not None:
                shape_param = torch.as_tensor(flame_params['shape'])
            if flame_params.get('exp') is not None:
                exp_param = torch.as_tensor(flame_params['exp'])
            if flame_params.get('global_rotation') is not None:
                global_rotation = torch.as_tensor(flame_params['global_rotation'])
            if flame_params.get('jaw') is not None:
                jaw_pose = torch.as_tensor(flame_params['jaw'])
            if flame_params.get('neck') is not None:
                neck_pose = torch.as_tensor(flame_params['neck'])
            if flame_params.get('eyes') is not None:
                eyes_pose = torch.as_tensor(flame_params['eyes'])
            if flame_params.get('transl') is not None:
                transl = torch.as_tensor(flame_params['transl'])
            if flame_params.get('scale_factor') is not None:
                scale_factor = torch.as_tensor(flame_params['scale_factor']).reshape(shape_param.shape[0],1)
        except Exception as e:
            print(f"Warning: Failed to process FLAME parameters: {e}")

    deformer_path=cam_info.deformer_path

    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  # 添加新的参数
                  kid=cam_info.kid,timecode=cam_info.timecode,
                  # flame参数
                  alpha=alpha, head_mask=head_mask, mouth_mask=mouth_mask,
                  shape_param=shape_param,
                  exp_param=exp_param, global_rotation=global_rotation, jaw_pose=jaw_pose, 
                  neck_pose=neck_pose, eyes_pose=eyes_pose, transl=transl,scale_factor=scale_factor,
                  #deformer参数
                  deformer_path=deformer_path
                  )
def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry