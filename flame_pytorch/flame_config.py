# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import argparse
from pathlib import Path

from yacs.config import CfgNode as CN

cfg = CN()

cfg.flame_model_path = "model/generic_model.pkl"
cfg.static_landmark_embedding_path = "model/flame_static_embedding.pkl"
cfg.dynamic_landmark_embedding_path="model/flame_dynamic_embedding.npy"
cfg.shape_params=300
cfg.expression_params=100
cfg.pose_params=6
cfg.neck_params=3
cfg.eye_params=6
cfg.translation_params=3
cfg.scale_params=1
cfg.vertex_num=5023
cfg.use_face_contour=True
cfg.use_3D_translation=True # Flase for RingNet project
cfg.optimize_eyeballpose=True # Flase for RingNet project
cfg.optimize_neckpose=True # Flase for RingNet project
cfg.num_worker=4
cfg.batch_size=1
cfg.ring_margin=0.5
cfg.ring_loss_weight=1.0


def get_cfg_defaults():
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    cfg = get_cfg_defaults()
    return cfg


def parse_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, cfg_file)
    cfg.cfg_file = cfg_file

    cfg.config_name = Path(cfg_file).stem

    return cfg
