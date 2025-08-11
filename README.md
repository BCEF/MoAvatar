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
├── <frame0000>
    ├── alpha # raw alpha prediction
    ├── images # extracted video frames
    ├── sparse # colmap camera
    ├── flame.frame # flame parameters
├── <frame0001>
...

```
## 训练 Train
- train_batch_step2.py训练标准空间，包含稠密化和剪枝
```
train_batch_step2.py -s <dataset path> --model_path <output path> 
```
- train_batch_step3.py训练MLP，不优化标准空间，有刚性约束和时间连续性约束
```
train_batch_step3.py -s <dataset path> --model_path <output path> --start_checkpoint <checkpoint path>
```
- render_ckpt.py读取checkpoint渲染每一帧的图像和ply
```
render_ckpt.py -s <dataset path> --model_path <output path> --start_checkpoint <checkpoint path>
```

## 下载FLAME数据 Download FLAME data
To download the FLAME model, sign up and agree to the model license under MPI-IS/FLAME. Then run following script:
```
./fetch_FLAME.sh
```
flame数据需要存储在model文件夹下
```
model
├── flame_dynamic_embedding.npy
├── flame_static_embedding.pkl
├── FlameMesh.obj
├── generic_model.pkl

```