import numpy as np
import json
from scipy.spatial import cKDTree


class DeformationTransforms:
    """存储和管理从A到B的变形变换信息"""
    
    def __init__(self):
        self.transformations = []  # 每个控制节点的变换矩阵 
        self.source_nodes = []     # 源网格中控制节点的索引
    
    def save(self, filename):
        """保存变换参数到文件"""
        try:
            data = {
                'source_nodes': [int(node) for node in self.source_nodes],
                'transformations': [matrix.tolist() for matrix in self.transformations]
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"变形变换已保存到文件: {filename}")
            return True
        except Exception as e:
            print(f"保存变换参数时出错: {e}")
            return False
    
    def load(self, filename):
        """从文件加载变换参数"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            self.source_nodes = data['source_nodes']
            self.transformations = [np.array(matrix) for matrix in data['transformations']]
            
            print(f"变形变换已从文件加载: {filename}")
            return True
        except Exception as e:
            print(f"加载变换参数时出错: {e}")
            return False


def compute_deformation_transforms(dg, source_vertices, target_vertices):
    """
    计算从网格A到网格B的变形变换
    
    参数:
        dg: 变形图对象
        source_vertices: 源网格A的顶点位置 
        target_vertices: 目标网格B的顶点位置
    
    返回:
        transforms: DeformationTransforms对象，包含变换信息
    """
    transforms = DeformationTransforms()
    transforms.source_nodes = dg.nodes.copy()
    
    node_count = len(dg.nodes)
    transformations = []
    
    for i in range(node_count):
        # 获取当前节点的索引和在两个网格中的位置
        node_idx = dg.nodes[i]
        source_pos = source_vertices[node_idx]
        target_pos = target_vertices[node_idx]
        
        # 获取相邻节点(在node_nodes中连接的节点)
        neighbors = []
        for j, _ in dg.node_nodes[i]:
            if j < len(dg.nodes):  # 确保索引有效
                neighbors.append(dg.nodes[j])
        
        # 如果没有足够的邻居，添加更多顶点
        if len(neighbors) < 3:
            # 找到最近的几个顶点作为邻居补充
            dists = np.linalg.norm(source_vertices - source_pos.reshape(1, 3), axis=1)
            dists[node_idx] = np.inf  # 排除自身
            for n in neighbors:
                dists[n] = np.inf  # 排除已有邻居
            
            # 获取最近的几个顶点
            closest = np.argsort(dists)[:3-len(neighbors)]
            neighbors.extend(closest)
        
        # 至少需要3个点来计算一个很好的变换
        if len(neighbors) >= 3:
            # 构建局部坐标系
            X_src = np.vstack([source_vertices[n] - source_pos for n in neighbors])
            X_tgt = np.vstack([target_vertices[n] - target_pos for n in neighbors])
            
            # 计算旋转矩阵 (使用SVD求解最佳旋转)
            H = X_src.T @ X_tgt
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # 确保是正旋转矩阵
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # 创建完整的变换矩阵
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = target_pos - R @ source_pos
            
            transformations.append(T)
        else:
            # 如果没有足够的邻居，使用简单的平移变换
            T = np.eye(4)
            T[:3, 3] = target_pos - source_pos
            transformations.append(T)
    
    transforms.transformations = transformations
    return transforms


def apply_deformation(dg, vertices, transforms):
    """
    将预先计算的变形变换应用到网格
    
    参数:
        dg: 变形图对象
        vertices: 要变形的网格的顶点位置
        transforms: DeformationTransforms对象，包含变换信息
    
    返回:
        deformed_vertices: 变形后的顶点位置
    """
    # 确认节点一致性
    if not np.array_equal(dg.nodes, transforms.source_nodes):
        print("警告: 变形图节点与变换参数中的节点不一致")
    
    transformations = transforms.transformations
    
    # 创建结果数组
    deformed_vertices = np.zeros_like(vertices)
    
    # 对每个顶点应用变形
    for i in range(len(vertices)):
        if i >= len(dg.v_nodes) or not dg.v_nodes[i]:  # 如果顶点没有关联的控制节点
            deformed_vertices[i] = vertices[i]
            continue
        
        # 使用线性混合变形(Linear Blend Skinning)
        blend_pos = np.zeros(3)
        total_weight = 0
        
        for node_idx, weight in dg.v_nodes[i]:
            if node_idx >= len(transformations):
                continue
                
            # 获取控制节点的变换矩阵
            transform = transformations[node_idx]
            
            # 将顶点从原始位置变换到新位置
            homogeneous_pos = np.ones(4)
            homogeneous_pos[:3] = vertices[i]
            transformed_pos = transform @ homogeneous_pos
            
            # 加权累加
            blend_pos += weight * transformed_pos[:3]
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            deformed_vertices[i] = blend_pos / total_weight
        else:
            deformed_vertices[i] = vertices[i]
    
    return deformed_vertices


def get_deformation_info_fixed_influences(dg, points, transforms, num_influences=20, weight_method='inverse_distance'):
    """
    获取每个点的变形信息，每个点使用固定数量的影响节点
    
    Args:
        dg: 变形图对象
        points: 点位置数组 (N, 3)
        transforms: DeformationTransforms对象，包含变换信息
        num_influences: 每个点的影响节点数量 (默认20)
        weight_method: 权重计算方法 ('inverse_distance', 'gaussian', 'linear_decay', 'uniform')
    
    Returns:
        transform_info: 变换信息字典
    """

    
    # 创建变换信息存储结构
    transform_info = {
        # 'influence_nodes': [],      # 每个点的影响节点索引列表
        'weights': [],              # 每个点的权重列表
        'RT':[]
    }
    
    # 获取变换矩阵和控制节点信息
    transformations = np.array(transforms.transformations)
    node_positions = dg.node_positions
    point_count = points.shape[0]
    
    # 确保影响节点数量不超过总节点数
    actual_num_influences = min(num_influences, len(node_positions))
    if actual_num_influences < num_influences:
        print(f"警告: 请求的影响节点数 {num_influences} 超过总节点数 {len(node_positions)}，调整为 {actual_num_influences}")
    
    # 1. 使用KD树加速最近点查找
    kdtree = cKDTree(node_positions)
    
    # 2. 批处理
    batch_size = 20000
    num_batches = (point_count + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, point_count)
        batch_points = points[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        # 3. 查找每个点的最近邻节点
        distances, node_indices = kdtree.query(batch_points, k=actual_num_influences)
        
        # 如果只有一个影响节点，确保distances和node_indices是2D数组
        if actual_num_influences == 1:
            distances = distances.reshape(-1, 1)
            node_indices = node_indices.reshape(-1, 1)
        
        # 处理批次中的每个点
        for local_idx in range(batch_size_actual):
            point_pos = batch_points[local_idx]
            point_distances = distances[local_idx]
            point_node_indices = node_indices[local_idx]
            
            # 计算权重
            weights = calculate_weights(point_distances, weight_method)
            
            # 收集变换矩阵
            point_transforms = []
            for node_idx in point_node_indices:
                transform_matrix = transformations[node_idx]
                point_transforms.append(transform_matrix.copy())
            
            # 保存变换信息
            # transform_info['influence_nodes'].append(point_node_indices.tolist())
            transform_info['weights'].append(weights.tolist())
            transform_info['RT'].append(point_transforms)
    
    return transform_info


def calculate_weights(distances, method='inverse_distance', sigma=None):
    """
    根据距离计算权重
    
    Args:
        distances: 距离数组
        method: 权重计算方法
        sigma: 高斯权重的标准差参数 (仅在method='gaussian'时使用)
    
    Returns:
        weights: 归一化后的权重数组
    """
    import numpy as np
    
    # 避免除零错误
    distances = np.maximum(distances, 1e-8)
    
    if method == 'inverse_distance':
        # 反比例权重: w = 1/d
        weights = 1.0 / distances
        
    elif method == 'inverse_distance_squared':
        # 反比例平方权重: w = 1/d^2
        weights = 1.0 / (distances ** 2)
        
    elif method == 'gaussian':
        # 高斯权重: w = exp(-d^2/(2*sigma^2))
        if sigma is None:
            sigma = np.mean(distances) / 2  # 自动设置sigma
        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        
    elif method == 'linear_decay':
        # 线性衰减权重: w = max(0, 1 - d/max_d)
        max_dist = np.max(distances)
        weights = np.maximum(0, 1.0 - distances / max_dist)
        
    elif method == 'exponential_decay':
        # 指数衰减权重: w = exp(-d/scale)
        scale = np.mean(distances)
        weights = np.exp(-distances / scale)
        
    elif method == 'uniform':
        # 均匀权重
        weights = np.ones_like(distances)
        
    else:
        raise ValueError(f"未知的权重计算方法: {method}")
    
    # 归一化权重
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    else:
        # 如果所有权重都是0，使用均匀权重
        weights = np.ones_like(weights) / len(weights)
    
    return weights

import matplotlib.pyplot as plt
def visualize_deformation_graph(mesh, dg):
    """可视化网格和变形图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制网格边缘
    edges = mesh.edges_unique
    for edge in edges:
        p1, p2 = mesh.vertices[edge]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', alpha=0.1)
    
    # 绘制控制节点
    ax.scatter(dg.node_positions[:, 0], 
               dg.node_positions[:, 1], 
               dg.node_positions[:, 2], 
               c='r', s=50, label='Control Nodes')
    
    # 绘制前10个节点的连接
    for i in range(min(10, len(dg.node_nodes))):
        for j, _ in dg.node_nodes[i]:
            p1 = dg.node_positions[i]
            p2 = dg.node_positions[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', alpha=0.5)
    
    ax.set_title(f" ({len(dg.nodes)} )")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()
