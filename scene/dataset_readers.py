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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.param_model_utils import create_np_flame_geometry,create_np_smplx_geometry

import torch

from dataclasses import dataclass
@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    #WDD 添加新的字段
    alpha_path: str = ""
    head_mask_path: str = ""
    mouth_mask_path: str = ""
    # flame_params: dict = None
    kid:int=0
    timecode: float = 0.0
    params_path:str=""

@dataclass
class SceneInfo:
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

#WDD 保存点云

def save_pcd_ply(vertices,ply_path):
    
    if len(vertices)>0:
        all_vertices = vertices
        
        # Remove duplicate vertices (optional)
        # unique_vertices = np.unique(all_vertices, axis=0)
        unique_vertices = all_vertices  # Or use all vertices directly
        
        num_pts = len(unique_vertices)
        # print(f"Total vertices: {num_pts}")
        
        # Generate colors for vertices (can use default colors or adjust as needed)
        # Using skin tone as default color
        skin_color = np.array([0.8, 0.6, 0.4])  # Skin tone
        colors = np.tile(skin_color, (num_pts, 1))
        
        # Generate normals (set to zero vectors here, can calculate actual normals if needed)
        normals = np.zeros((num_pts, 3))
        
        # Create point cloud
        pcd = BasicPointCloud(points=unique_vertices, colors=colors, normals=normals)
        
        # Save PLY file
        storePly(ply_path, unique_vertices, colors * 255)
        
    else:
        # If no FLAME vertices, fallback to random point cloud
        print("No FLAME vertices generated, falling back to random point cloud...")
        if not os.path.exists(ply_path):
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    return pcd


# #WDD 读取 包含FLAME的数据
# def readFlameSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8,colmap_folder=None,flame_path=None,alpha_folder=None,head_folder=None,mouth_folder=None,kid=0,timecode=0.0):
#     sparse_folder=os.path.join(path, "sparse/0") if colmap_folder is None else colmap_folder

#     #try:
#     cameras_extrinsic_file = os.path.join(sparse_folder, "images.bin")
#     cameras_intrinsic_file = os.path.join(sparse_folder, "cameras.bin")
#     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#     #except:
#     #    cameras_extrinsic_file = os.path.join(sparse_folder, "images.txt")
#     #    cameras_intrinsic_file = os.path.join(sparse_folder, "cameras.txt")
#     #    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#     #    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

#     depth_params_file = os.path.join(sparse_folder, "depth_params.json")
#     ## if depth_params_file isnt there AND depths file is here -> throw error
#     depths_params = None
#     if depths != "":
#         try:
#             with open(depth_params_file, "r") as f:
#                 depths_params = json.load(f)
#             all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
#             if (all_scales > 0).sum():
#                 med_scale = np.median(all_scales[all_scales > 0])
#             else:
#                 med_scale = 0
#             for key in depths_params:
#                 depths_params[key]["med_scale"] = med_scale

#         except FileNotFoundError:
#             print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
#             sys.exit(1)
#         except Exception as e:
#             print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
#             sys.exit(1)

#     if eval:
#         if "360" in path:
#             llffhold = 8
#         if llffhold:
#             print("------------LLFF HOLD-------------")
#             cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
#             cam_names = sorted(cam_names)
#             test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
#         else:
#             with open(os.path.join(sparse_folder, "test.txt"), 'r') as file:
#                 test_cam_names_list = [line.strip() for line in file]
#     else:
#         test_cam_names_list = []

#     reading_dir = "images" if images == None else images
#     cam_infos_unsorted = readColmapCameras(
#         cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
#         images_folder=os.path.join(path, reading_dir), 
#         depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list,
#         )

#     ply_path = os.path.join(path, "input.ply")
#     flame_path=os.path.join(path,"flame.frame") if flame_path==None else flame_path
#     payload=torch.load(flame_path,weights_only=False)
#     flame_params = {}
#     if 'flame' in payload:
#         flame_data = payload['flame']
#         flame_params = {
#             'shape': flame_data.get('shape', None),
#             'exp': flame_data.get('exp', None),
#             'global_rotation': flame_data.get('global_rotation', None),
#             'jaw': flame_data.get('jaw', None),
#             'neck': flame_data.get('neck', None),
#             'eyes': flame_data.get('eyes', None),
#             'transl': flame_data.get('transl', None),
#             'scale_factor': flame_data.get('scale_factor', None)
#         }
#         # pcd=None
#         # if not os.path.exists(ply_path): 
#         # Extract FLAME parameters
#         shape_params = torch.as_tensor(flame_params['shape']) if 'shape' in flame_params else None
#         expression_params = torch.as_tensor(flame_params['exp']) if 'exp' in flame_params else None
        
#         # Process pose parameters
#         global_rotation = torch.as_tensor(flame_params.get('global_rotation', torch.zeros(3))) if 'global_rotation' in flame_params else torch.zeros(3)
#         jaw_pose = torch.as_tensor(flame_params.get('jaw', torch.zeros(3))) if 'jaw' in flame_params else torch.zeros(3)
#         neck_pose = torch.as_tensor(flame_params.get('neck', torch.zeros(3))) if 'neck' in flame_params else torch.zeros(3)
#         eye_pose = torch.as_tensor(flame_params['eyes']) if 'eyes' in flame_params else torch.zeros(6)
#         transl_pose = torch.as_tensor(flame_params.get('transl', torch.zeros(3))) if 'transl' in flame_params else torch.zeros(3)
#         scale_factor = torch.as_tensor(flame_params.get('scale_factor', torch.ones(1))) if 'scale_factor' in flame_params else torch.ones(1)
#         pose_params = torch.cat([global_rotation, jaw_pose], dim=1)
        
        
#         try:
#             config = parse_args()
#             flamelayer = FLAME(config)
#             # Generate FLAME vertices and landmarks
#             # vertices, landmarks = flamelayer(
#             #     shape_params, expression_params, pose_params, neck_pose, eye_pose, transl_pose
#             # )

#             vertices=flamelayer.forward_geo_subdivided(
#                 shape_params, expression_params, pose_params, neck_pose, eye_pose, transl_pose,scale_factor,1
#             )
            
#             # Convert vertices to numpy and add to list
#             if isinstance(vertices, torch.Tensor):
#                 vertices_np = vertices.detach().cpu().numpy()
#                 if vertices_np.ndim == 3:  # If has batch dimension, take first
#                     vertices_np = vertices_np[0]
#                 # vertices_np = vertices_np * scale_factor.cpu().numpy()  # Apply scaling factor
#                 pcd=save_pcd_ply(vertices_np,ply_path)
            
#         except Exception as e:
#             print(f"\nWarning: Failed to generate FLAME vertices : {e}")

#     for camera_info in cam_infos_unsorted:
#         alpha_reading_dir = "alpha" if alpha_folder == None else alpha_folder
#         neckhead_reading_dir="neckhead" if head_folder==None else head_folder
#         mouth_reading_dir="mouth" if mouth_folder==None else mouth_folder
#         camera_info.alpha_path=os.path.join(path,alpha_reading_dir,camera_info.image_name)
#         camera_info.head_mask_path=os.path.join(path,neckhead_reading_dir,camera_info.image_name)
#         camera_info.mouth_mask_path=os.path.join(path,mouth_reading_dir,camera_info.image_name)
#         camera_info.flame_params=flame_params
#         camera_info.kid=kid
#         camera_info.timecode=timecode

#     cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

#     train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
#     test_cam_infos = [c for c in cam_infos if c.is_test]

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path,
#                            is_nerf_synthetic=False)
#     return scene_info

def readFlameSmplxSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8,colmap_folder=None,flame_path=None,alpha_folder=None,head_folder=None,mouth_folder=None,kid=0,timecode=0.0):
    sparse_folder=os.path.join(path, "sparse/0") if colmap_folder is None else colmap_folder

    cameras_extrinsic_file = os.path.join(sparse_folder, "images.bin")
    cameras_intrinsic_file = os.path.join(sparse_folder, "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    depth_params_file = os.path.join(sparse_folder, "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(sparse_folder, "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list,
        )

    ply_path = os.path.join(path, "input.ply")
    if flame_path is None:
        flame_path=os.path.join(path,"flame.frame")
        if not os.path.exists(flame_path):
            flame_path=os.path.join(path,"smplx.pkl")
    
    for camera_info in cam_infos_unsorted:
        alpha_reading_dir = "alpha" if alpha_folder == None else alpha_folder
        neckhead_reading_dir="neckhead" if head_folder==None else head_folder
        mouth_reading_dir="mouth" if mouth_folder==None else mouth_folder
        camera_info.alpha_path=os.path.join(path,alpha_reading_dir,camera_info.image_name)
        camera_info.head_mask_path=os.path.join(path,neckhead_reading_dir,camera_info.image_name)
        camera_info.mouth_mask_path=os.path.join(path,mouth_reading_dir,camera_info.image_name)
        camera_info.params_path=flame_path
        camera_info.kid=kid
        camera_info.timecode=timecode

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if flame_path.endswith(".frame"):
        vertices_np=create_np_flame_geometry(flame_path)
        pcd=save_pcd_ply(vertices_np,ply_path)
    elif flame_path.endswith(".pkl"):
        vertices_np=create_np_smplx_geometry(flame_path)
        pcd=save_pcd_ply(vertices_np,ply_path)
    else:
        pcd=None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "ParamsGeo":readFlameSmplxSceneInfo
}