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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.dataset_readers import SceneInfo
import torch
import gc
# from utils.general_utils import quantize_by_decimal
from utils.param_model_utils import load_flame_codedict,load_smplx_codedict
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        #SUMO
        self.args = args
        self.resolution_scales = resolution_scales
        self.scale=1
        self.loadMultiFrameSceneInfo(args, shuffle)

    #SUMO
    def loadOneFrameSceneInfo(self,args,shuffle=True):

        self.train_cameras = {}
        self.test_cameras = {}
        #WDD
        def has_frame_file(source_path):
            for fname in os.listdir(source_path):
                if fname.lower().endswith('.frame'):
                    return True
            return False
        
        #WDD 增加了FLAME的调用
        if has_frame_file(args.source_path): 
            scene_info=sceneLoadTypeCallbacks["ParamsGeo"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        #SUMO
        self.scene_info = scene_info
        self.loadSceneInfo(args, shuffle)

    #SUMO
    def loadMultiFrameSceneInfo(self,args,shuffle=True,params_type="smplx"):
        self.train_cameras = {}
        self.test_cameras = {}
        self.flame_codes={}

        root_folder=args.source_path
        # subfolders=os.listdir(root_folder)
        subfolders = [f for f in os.listdir(root_folder) if 'sparse' not in f]
        subfolders.sort()

        self.scene_info=None
        for frame_id,subfolder in enumerate(subfolders):
            kid=frame_id
            timecode=float(frame_id)/len(subfolders)
            frame_folder=os.path.join(root_folder,subfolder)
            colmap_folder=os.path.join(frame_folder,'sparse/0')
            if not os.path.exists(colmap_folder):
                colmap_folder = os.path.join(root_folder, 'sparse/0')
            if params_type=="smplx":
                flame_path=os.path.join(frame_folder,"smplx.pkl")
            elif params_type=="flame":
                flame_path=os.path.join(frame_folder,'flame.frame')
            else:
                flame_path=None
            alpha_folder=os.path.join(frame_folder,'alpha')
            head_folder=os.path.join(frame_folder,'neckhead')
            mouth_folder=os.path.join(frame_folder,'mouth')

            scene_info=sceneLoadTypeCallbacks["ParamsGeo"](frame_folder,args.images, args.depths, args.eval, args.train_test_exp,
                                                       colmap_folder=colmap_folder,flame_path=flame_path,
                                                       alpha_folder=alpha_folder,head_folder=head_folder,mouth_folder=mouth_folder,
                                                       kid=kid,timecode=timecode)
            if self.scene_info is None:
                self.scene_info=scene_info
            else:
                self.scene_info.train_cameras.extend(scene_info.train_cameras)
                self.scene_info.test_cameras.extend(scene_info.test_cameras)
            
            self.flame_codes[kid]=self.loadFlameCodes(flame_path,kid,timecode)
        
        self.loadSceneInfo(args, shuffle)

    #SUMO
    def loadSceneInfo(self,args,shuffle):
        scene_info = self.scene_info
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def loadFlameCodes(self,flame_path:str,kid,timecode):
        if flame_path.endswith("frame"):
            codedict=load_flame_codedict(flame_path)
            codedict["kid"]=kid
            codedict["t"]=timecode
        elif flame_path.endswith(".pkl"):
            codedict=load_smplx_codedict(flame_path)
            codedict["kid"]=kid
            codedict["t"]=timecode
        return None

    # def loadTrainCameras(self, cam_infos=None, resolution_scale=1.0):
    #     for resolution_scale in self.resolution_scales:
    #         self.loadTrainCamera(cam_infos=cam_infos,resolution_scale=resolution_scale)

    def loadTrainCameras(self, cam_infos=None, resolution_scale=1.0):
        """
        根据指定的camera info加载训练相机
        :param cam_infos: 相机信息列表，如果为None则加载所有训练相机
        :param resolution_scale: 分辨率缩放比例
        """
        self.scale=resolution_scale
        if cam_infos is None:
            cam_infos = self.scene_info.train_cameras
        
        print(f"Loading {len(cam_infos)} Training Cameras at resolution scale {resolution_scale}")
        
        # 确保resolution_scale在字典中存在
        if resolution_scale not in self.train_cameras:
            self.train_cameras[resolution_scale] = []
        
        # 加载指定的相机
        loaded_cameras = cameraList_from_camInfos(cam_infos, resolution_scale, self.args, self.scene_info.is_nerf_synthetic, False)
        self.train_cameras[resolution_scale].extend(loaded_cameras)

        
        
        return loaded_cameras

    def loadTestCameras(self, cam_infos=None, resolution_scale=1.0):
        """
        根据指定的camera info加载测试相机
        :param cam_infos: 相机信息列表，如果为None则加载所有测试相机
        :param resolution_scale: 分辨率缩放比例
        """
        self.scale=resolution_scale
        if cam_infos is None:
            cam_infos = self.scene_info.test_cameras
        
        # print(f"Loading {len(cam_infos)} Test Cameras at resolution scale {resolution_scale}")
        
        # 确保resolution_scale在字典中存在
        if resolution_scale not in self.test_cameras:
            self.test_cameras[resolution_scale] = []
        
        # 加载指定的相机
        loaded_cameras = cameraList_from_camInfos(cam_infos, resolution_scale, self.args, self.scene_info.is_nerf_synthetic, True)
        self.test_cameras[resolution_scale].extend(loaded_cameras)
        
        return loaded_cameras

    def loadAllCameras(self, resolution_scales=None):
        """
        加载所有相机（训练和测试）
        :param resolution_scales: 分辨率尺度列表，如果为None则使用初始化时的尺度
        """
        if resolution_scales is None:
            resolution_scales = self.resolution_scales
            
        for resolution_scale in resolution_scales:
            self.loadTrainCameras(resolution_scale=resolution_scale)
            self.loadTestCameras(resolution_scale=resolution_scale)

    def clearCameras(self, resolution_scale=None):
        """
        清空已加载的相机
        :param resolution_scale: 指定分辨率尺度，如果为None则清空所有
        """
        if resolution_scale is None:
            for scale in self.resolution_scales:
                del self.train_cameras[scale]
                del self.test_cameras[scale]
                # 强制Python垃圾回收
                gc.collect()
                # 清空PyTorch的GPU缓存
                torch.cuda.empty_cache()
                # 如果需要，可以同步GPU操作
                torch.cuda.synchronize()

                self.train_cameras[scale] = []
                self.test_cameras[scale] = []
        else:
            if resolution_scale in self.train_cameras:
                del self.train_cameras[resolution_scale]
                # 强制Python垃圾回收
                gc.collect()
                # 清空PyTorch的GPU缓存
                torch.cuda.empty_cache()
                # 如果需要，可以同步GPU操作
                torch.cuda.synchronize()
                self.train_cameras[resolution_scale] = []
            if resolution_scale in self.test_cameras:
                del self.test_cameras[resolution_scale]
                # 强制Python垃圾回收
                gc.collect()
                # 清空PyTorch的GPU缓存
                torch.cuda.empty_cache()
                # 如果需要，可以同步GPU操作
                torch.cuda.synchronize()
                self.test_cameras[resolution_scale] = []

    def getAvailableCamInfos(self):
        """
        获取可用的相机信息
        :return: 包含训练和测试相机信息的字典
        """
        return {
            'train_cameras': self.scene_info.train_cameras,
            'test_cameras': self.scene_info.test_cameras
        }

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
