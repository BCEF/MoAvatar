import numpy as np
import json
import os

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