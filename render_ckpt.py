import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2

def training(dataset, opt, pipe,checkpoint_path):
    
    foldername=os.path.splitext(os.path.basename(checkpoint_path))[0]
    os.makedirs(os.path.join(dataset.model_path, foldername), exist_ok=True)
    
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians,shuffle=False)
    # gaussians.training_setup(opt)
    # gaussians.training_setup_freeze_x0(opt)

    if checkpoint_path:
        (model_params, first_iter) = torch.load(checkpoint_path,weights_only=False,map_location="cuda")
        gaussians.restore_step3(model_params, opt)
        render_ckpt_path=checkpoint_path[:-4] + "_render.pth"
        torch.save(gaussians.capture_render(), render_ckpt_path)
        model_params = torch.load(render_ckpt_path,weights_only=False,map_location="cuda")
        gaussians.restore_render(model_params)

    torch.cuda.empty_cache()

    bg_color = [0, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    all_train_cameras = scene.getAvailableCamInfos()['train_cameras']
    num_batches = dataset.batchnum
    batch_size = max(1, len(all_train_cameras) // num_batches)
    std_name=None
    for batch_idx in range(num_batches+1):
        batch_start = batch_idx * batch_size
        if batch_idx == num_batches:
            batch_end = len(all_train_cameras)
        else:
            batch_end = batch_start + batch_size
        
        print(f"Loading batch {batch_idx + 1}/{num_batches}: cameras {batch_start} to {batch_end-1}")
        scene.loadTrainCameras(all_train_cameras[batch_start:batch_end])
        scene.loadTestCameras(scene.getAvailableCamInfos()['test_cameras'][batch_start:batch_end])


        viewpoint = scene.getTrainCameras().copy()
        if std_name is None:
            std_name=viewpoint[0].image_name
        print(f"Rendering viewpoints with name {std_name}")
        for iteration in range(len(viewpoint)):
            with torch.no_grad():
                viewpoint_cam = viewpoint[iteration]
                if viewpoint_cam.image_name != std_name:
                    continue
                save_name=str(viewpoint_cam.kid).zfill(4)#+'_'+viewpoint_cam.image_name
                bg = torch.rand((3), device="cuda") if opt.random_background else background
                codedict={}
                codedict['deformer_path']=viewpoint_cam.deformer_path
                codedict['t']=viewpoint_cam.timecode
                codedict['kid'] = viewpoint_cam.kid
                gaussians.forward_render(codedict)
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
                image = render_pkg["render"]
                image = image.clamp(0, 1)
                image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()
                save_image = image_np
                save_image = save_image[:,:,[2,1,0]]
                print(os.path.join(dataset.model_path, foldername))
                cv2.imwrite(os.path.join(dataset.model_path,foldername, f'{save_name}.png'), save_image)
                gaussians.save_ply(os.path.join(dataset.model_path,foldername, str(viewpoint_cam.kid).zfill(4)+'.ply'))
        scene.clearCameras(1.0) 



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--step3_checkpoint", type=str, default = None) 
    args = parser.parse_args(sys.argv[1:])
    args.batchnum=50
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),args.step3_checkpoint)