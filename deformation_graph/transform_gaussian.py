def apply_deformation_to_gaussians(dg, points, transforms):
    """
    将变形图变换应用到高斯散点 - 高性能版本(不使用多进程)
    
    参数:
        dg: 变形图对象
     
        transforms: DeformationTransforms对象，包含变换信息
    
    返回:
        deformed_gaussian: 变形后的高斯数据字典
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    import numpy as np

    # 创建结果字典，复制输入高斯数据

    deformed_gaussian=points.copy()

    # 获取变换矩阵和控制节点信息
    transformations = np.array(transforms.transformations)
    node_positions = dg.node_positions
    influence_radius = dg.node_radius
    
    # 获取所有高斯点的位置和旋转
    point_count = points.shape[0]

    
    # 1. 使用KD树加速最近点查找
    kdtree = cKDTree(node_positions)
    
    # 2. 预计算控制节点的SVD分解结果
    rotation_matrices = []
    rotation_objects = []
    for t in transformations:
        R = t[:3, :3]
        u, s, vh = np.linalg.svd(R, full_matrices=False)
        orthogonal_R = u @ vh
        rotation_matrices.append(orthogonal_R)
        rotation_objects.append(Rotation.from_matrix(orthogonal_R))
    
    # 3. 批处理 + 稀疏表示
    batch_size = 30000  # 可调整
    num_batches = (point_count + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, point_count)
        batch_points = points[start_idx:end_idx]

        
        # 4. 快速查找每个点的影响节点 (半径查询)
        influence_indices = kdtree.query_ball_point(batch_points, influence_radius)
        
        # 5. 预先分配结果数组，避免重复分配内存
        batch_positions = deformed_gaussian[start_idx:end_idx].copy()

        
        # 跟踪需要处理的点
        points_to_process = []
        for i, indices in enumerate(influence_indices):
            if len(indices) > 0:
                points_to_process.append(i)
        
        # 6. 只处理有影响节点的点
        for local_idx in points_to_process:
            point_pos = batch_points[local_idx]
            
            # 有影响的节点索引
            node_indices = influence_indices[local_idx]
            
            # 计算到这些节点的距离
            node_dists = np.linalg.norm(node_positions[node_indices] - point_pos, axis=1)
            
            # 计算权重
            weights = 1.0 - node_dists / influence_radius
            weights = np.maximum(weights, 0)
            total_weight = np.sum(weights)
            
            if total_weight <= 0:
                continue
                
            weights = weights / total_weight
            
            # 7. 向量化位置变换计算
            homogeneous_pos = np.ones(4)
            homogeneous_pos[:3] = point_pos
            
            # 使用矩阵乘法一次性计算所有变换
            blend_pos = np.zeros(3)
            for j, node_idx in enumerate(node_indices):
                transformed_pos = transformations[node_idx] @ homogeneous_pos
                blend_pos += weights[j] * transformed_pos[:3]
            
            batch_positions[local_idx] = blend_pos
            
        # 更新结果数组
        deformed_gaussian[start_idx:end_idx] = batch_positions

    
    return deformed_gaussian

def apply_deformation_to_gaussians2(dg, points, transforms):
    """
    将变形图变换应用到高斯散点
    
    参数:
        dg: 变形图对象
     
        transforms: DeformationTransforms对象，包含变换信息
    
    返回:
        deformed_gaussian: 变形后的高斯数据字典
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    import numpy as np

    # 创建结果字典，复制输入高斯数据

    deformed_gaussian=points.copy()

    # 获取变换矩阵和控制节点信息
    transformations = np.array(transforms.transformations)
    node_positions = dg.node_positions
    influence_radius = dg.node_radius
    
    # 获取所有高斯点的位置和旋转
    point_count = points.shape[0]

    
    # 1. 使用KD树加速最近点查找
    kdtree = cKDTree(node_positions)
    
    # 3. 批处理 + 稀疏表示
    batch_size = 30000  # 可调整
    num_batches = (point_count + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, point_count)
        batch_points = points[start_idx:end_idx]

        
        # 4. 快速查找每个点的影响节点 (半径查询)
        influence_indices = kdtree.query_ball_point(batch_points, influence_radius)
        
        # 5. 预先分配结果数组，避免重复分配内存
        batch_positions = deformed_gaussian[start_idx:end_idx]

         # 向量化处理整个批次
        batch_size = len(batch_points)
        
        # 跟踪需要处理的点
        points_to_process = []
        for i, indices in enumerate(influence_indices):
            if len(indices) > 0:
                points_to_process.append(i)
        
        # 6. 只处理有影响节点的点
        for local_idx in points_to_process:
            point_pos = batch_points[local_idx]
            
            # 有影响的节点索引
            node_indices = influence_indices[local_idx]
            
            # 计算到这些节点的距离
            node_dists = np.linalg.norm(node_positions[node_indices] - point_pos, axis=1)
            
            # 计算权重
            weights = 1.0 - node_dists / influence_radius
            weights = np.maximum(weights, 0)
            total_weight = np.sum(weights)
            
            if total_weight <= 0:
                continue
                
            weights = weights / total_weight
            
            # 7. 向量化位置变换计算
            homogeneous_pos = np.ones(4)
            homogeneous_pos[:3] = point_pos
            
            
            # 一次性获取所有相关变换矩阵: shape (k, 4, 4)
            relevant_transforms = transformations[node_indices]

            # 一次性变换: (k, 4, 4) @ (4,) -> (k, 4)  
            transformed_positions = relevant_transforms @ homogeneous_pos

            # 加权求和: (k,) @ (k, 3) -> (3,)
            blend_pos = weights @ transformed_positions[:, :3]
            batch_positions[local_idx] = blend_pos
            
        # 更新结果数组
        deformed_gaussian[start_idx:end_idx] = batch_positions

    
    return deformed_gaussian



def get_deformation_info_fixed_influences(dg, points, num_influences=20, weight_method='inverse_distance'):
    """
    获取每个点的变形信息，每个点使用固定数量的影响节点
    Args:
        dg: 变形图对象
        points: 点位置数组 (N, 3)
        num_influences: 每个点的影响节点数量 (默认20)
        weight_method: 权重计算方法 ('inverse_distance', 'gaussian', 'linear_decay', 'uniform')
    Returns:
        influence_info: 包含影响节点索引和权重的字典
    """
    from scipy.spatial import cKDTree
    import numpy as np
    
    # 获取控制节点信息
    node_positions = dg.node_positions
    point_count = points.shape[0]
    
    # 确保影响节点数量不超过总节点数
    actual_num_influences = min(num_influences, len(node_positions))
    if actual_num_influences < num_influences:
        print(f"警告: 请求的影响节点数 {num_influences} 超过总节点数 {len(node_positions)}，调整为 {actual_num_influences}")
    
    # 使用KD树加速最近点查找
    kdtree = cKDTree(node_positions)
    
    # 创建影响信息存储结构
    influence_info = {}
    
    # 批处理
    batch_size = 30000
    num_batches = (point_count + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, point_count)
        batch_points = points[start_idx:end_idx]
        
        # 查找每个点的最近邻节点
        distances, node_indices = kdtree.query(batch_points, k=actual_num_influences)
        
        # 如果只有一个影响节点，确保distances和node_indices是2D数组
        if actual_num_influences == 1:
            distances = distances.reshape(-1, 1)
            node_indices = node_indices.reshape(-1, 1)
        
        # 计算权重
        batch_weights = []
        batch_indices = []
        
        for i in range(len(batch_points)):
            point_distances = distances[i]
            point_node_indices = node_indices[i]
            
            # 计算权重
            weights = calculate_weights(point_distances, weight_method)
            
            batch_weights.append(weights.tolist())
            batch_indices.append(point_node_indices.tolist())
        
        # 保存批次信息
        influence_info[batch_idx] = {
            'indices': batch_indices,
            'weights': batch_weights
        }
    
    return influence_info

def calculate_weights(distances, weight_method='inverse_distance'):
    """计算权重的辅助函数"""
    import numpy as np
    
    if weight_method == 'inverse_distance':
        # 避免除零，加小的epsilon
        weights = 1.0 / (distances + 1e-8)
    elif weight_method == 'gaussian':
        # 高斯权重，sigma可以调整
        sigma = np.mean(distances) if len(distances) > 1 else 1.0
        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    elif weight_method == 'linear_decay':
        # 线性衰减
        max_dist = np.max(distances) if len(distances) > 1 else 1.0
        weights = 1.0 - distances / max_dist
        weights = np.maximum(weights, 0)
    elif weight_method == 'uniform':
        # 均匀权重
        weights = np.ones_like(distances)
    else:
        raise ValueError(f"未知的权重方法: {weight_method}")
    
    # 归一化权重
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    
    return weights

def apply_deformation_to_gaussians_fix_influ(dg, points, transforms, influ=None, num_influences=20, weight_method='inverse_distance'):
    """
    将变形图变换应用到高斯散点 - 高性能版本(不使用多进程)
    参数:
    dg: 变形图对象
    points: 高斯点位置数组
    transforms: DeformationTransforms对象，包含变换信息
    influ: 预计算的影响信息字典（可选）
    num_influences: 每个点的影响节点数量 (默认20)
    weight_method: 权重计算方法
    返回:
    deformed_gaussian: 变形后的高斯数据字典
    influ: 影响信息字典
    """
    from scipy.spatial.transform import Rotation
    import numpy as np
    import time
    
    st = time.time()
    # 创建结果字典，复制输入高斯数据
    deformed_gaussian = points.copy()
    
    # 获取变换矩阵
    transformations = np.array(transforms.transformations)
    point_count = points.shape[0]
    
    # 获取或计算影响信息
    if influ is None:
        print("计算影响信息...")
        influ = get_deformation_info_fixed_influences(dg, points, num_influences, weight_method)
        print(f"影响信息计算完成，耗时 {time.time()-st:.2f} seconds")
    
    # 预计算控制节点的SVD分解结果
    rotation_matrices = []
    rotation_objects = []
    for t in transformations:
        R = t[:3, :3]
        u, s, vh = np.linalg.svd(R, full_matrices=False)
        orthogonal_R = u @ vh
        rotation_matrices.append(orthogonal_R)
        rotation_objects.append(Rotation.from_matrix(orthogonal_R))
    
    
    # 批处理应用变形
    batch_size = 30000  # 可调整
    num_batches = (point_count + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, point_count)
        batch_points = points[start_idx:end_idx]
        
        # 获取批次影响信息
        batch_influence_info = influ[batch_idx]
        influence_indices = batch_influence_info['indices']
        influence_weights = batch_influence_info['weights']
        
        # 预先分配结果数组，避免重复分配内存
        batch_positions = deformed_gaussian[start_idx:end_idx].copy()
        
        # # 处理批次中的每个点
        # for local_idx in range(len(batch_points)):
        #     point_pos = batch_points[local_idx]
            
        #     # 获取影响的节点索引和权重
        #     node_indices = np.array(influence_indices[local_idx])
        #     weights = np.array(influence_weights[local_idx])
            
        #     # 向量化位置变换计算
        #     homogeneous_pos = np.ones(4)
        #     homogeneous_pos[:3] = point_pos
            
            # # 使用矩阵乘法一次性计算所有变换
            # blend_pos = np.zeros(3)
            # for j, node_idx in enumerate(node_indices):
            #     transformed_pos = transformations[node_idx] @ homogeneous_pos
            #     blend_pos += weights[j] * transformed_pos[:3]
            
            # batch_positions[local_idx] = blend_pos

        # 向量化处理整个批次
        batch_size = len(batch_points)
        
        # 构建齐次坐标矩阵 (batch_size, 4)
        homogeneous_points = np.ones((batch_size, 4))
        homogeneous_points[:, :3] = batch_points
        
        # # 向量化处理每个点
        # for local_idx in range(batch_size):
        #     print(len(influence_indices))
        #     node_indices = np.array(influence_indices[local_idx])
        #     weights = np.array(influence_weights[local_idx])
            
        #     # 获取该点的所有变换矩阵 (num_influences, 4, 4)
        #     point_transforms = transformations[node_indices]
            
        #     # 向量化应用所有变换: (num_influences, 4, 4) @ (4,) -> (num_influences, 4)
        #     transformed_points = point_transforms @ homogeneous_points[local_idx]
            
        #     # 加权求和: (num_influences, 1) * (num_influences, 3) -> (3,)
        #     batch_positions[local_idx] = np.sum(weights[:, None] * transformed_points[:, :3], axis=0)
        
        # 将影响信息转换为NumPy数组以支持向量化
        influence_indices_array = np.array(influence_indices)  # (batch_size, num_influences)
        influence_weights_array = np.array(influence_weights)  # (batch_size, num_influences)
        
        # 展平索引并获取所有需要的变换矩阵
        flat_indices = influence_indices_array.flatten()  # (batch_size * num_influences,)
        selected_transforms = transformations[flat_indices]  # (batch_size * num_influences, 4, 4)
        
        # 重塑为批次形状 (batch_size, num_influences, 4, 4)
        batch_transforms = selected_transforms.reshape(batch_size, influence_indices_array.shape[1], 4, 4)
        
        # 批量矩阵乘法: (batch_size, num_influences, 4, 4) @ (batch_size, 4, 1) -> (batch_size, num_influences, 4)
        # 使用 einsum 进行高效的批量矩阵乘法
        transformed_points = np.einsum('bijk,bk->bij', batch_transforms, homogeneous_points)
        
        # 批量加权求和: (batch_size, num_influences, 1) * (batch_size, num_influences, 3) -> (batch_size, 3)
        batch_positions[:] = np.sum(influence_weights_array[..., None] * transformed_points[..., :3], axis=1)
        
        # 更新结果数组
        deformed_gaussian[start_idx:end_idx] = batch_positions
    
        
        # 更新结果数组
        deformed_gaussian[start_idx:end_idx] = batch_positions
    
    return deformed_gaussian, influ