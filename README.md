# MoAvatar

## 环境 Environment
This code has been tested on Nvidia RTX 4090,Ubuntu 22.04

Create the envionment:
```
conda env create --file environment.yml
conda activate MoAvatar
```
Install PyTorch3D:
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```
## 数据约定 Data Convention
The data is organized in the following form：
```
dataset
├── sparse #colmap camera for SIBR view
├── deformation_graph.json
├── <frame0000>
    ├── alpha # raw alpha prediction
    ├── images # extracted video frames
    ├── transforms.json
    ├── inv_transforms.json
    ├── sparse # colmap camera【alternative】
├── <frame0001>
```
训练 Train

- train_deformer_step2.py训练标准空间，包含稠密化和剪枝，可以添加预训练的高斯ply模型作为init_ply_path（此参数为可选项）
```
train_deformer_step2.py -s <dataset path> --model_path <output path>  --init_ply_path <pretrain ply>
```
- train_deformer_step3.py训练MLP，不优化标准空间，有刚性约束和时间连续性约束，checkpoint path是step2新的训练结果
```
train_deformer_step3.py -s <dataset path> --model_path <output path> --start_checkpoint <checkpoint path>
```
- train_deformer_step3.py,如果需要继续之前的训练，可以设置step3_checkpoint
```
train_deformer_step3.py -s <dataset path> --model_path <output path> --step3_checkpoint <restore ckpt>
```
- render_ckpt.py读取checkpoint渲染每一帧的图像和ply
```
render_ckpt.py -s <dataset path> --model_path <output path> --step3_checkpoint <checkpoint path>
```