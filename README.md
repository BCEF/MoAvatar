# MoAvatar

## 环境
- Ubuntu 22.04

## 训练
- train_batch_step2.py训练标准空间，包含稠密化和剪枝
- train_batch_step3.py训练MLP，不优化标准空间，有刚性约束和时间连续性约束
- render_ckpt.py读取checkpoint渲染每一帧的图像和ply