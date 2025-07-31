import numpy as np
import time
from .deformation_graph import DeformationGraph
from .fast_marching import FastMarching

def generate_deformation_graph(vertices, faces, 
                               node_num=100, 
                               radius_coef=2.1, 
                               node_nodes_num=4, 
                               v_nodes_num=6,
                               initial_seed=None):
    """
    生成网格的变形图
    
    参数:
        vertices: 顶点位置数组 (n x 3)
        faces: 面索引数组
        node_num: 需要的控制节点数量
        radius_coef: 节点影响半径系数
        node_nodes_num: 每个节点连接的最大节点数
        v_nodes_num: 每个顶点连接的最大节点数
        initial_seed: 初始种子点（默认随机选择）
    
    返回:
        dg: 生成的变形图对象
    """
    print("开始生成变形图...")
    start_time = time.time()
    
    # 创建变形图对象
    dg = DeformationGraph()
    
    # 创建Fast Marching工具
    fm = FastMarching()
    fm.set_mesh(vertices, faces)
    
    # 使用最远点采样选择节点
    print(f"使用最远点采样选择 {node_num} 个控制节点...")
    nodes, max_distance = fm.farthest_point_sampling(node_num, initial_point=initial_seed)
    dg.nodes = nodes
    
    # 设置节点位置
    dg.node_positions = vertices[nodes]
    
    # 设置节点影响半径
    influence_radius = max_distance * radius_coef
    dg.node_radius = influence_radius
    print(f"节点影响半径设置为 {influence_radius:.4f}")
    
    # 初始化连接数据结构
    n_nodes = len(nodes)
    n_vertices = len(vertices)
    dg.node_nodes = [[] for _ in range(n_nodes)]
    dg.v_nodes = [[] for _ in range(n_vertices)]
    
    # 节点索引映射
    node_index_map = {node: i for i, node in enumerate(nodes)}
    
    # 计算节点-节点连接
    print("计算节点-节点连接...")
    for i, node in enumerate(nodes):
        # 计算从当前节点到所有顶点的距离
        fm.compute_distance([node], max_distance=influence_radius)
        
        # 为当前节点找到邻近节点
        for other_node in nodes:
            if other_node != node and fm.distance[other_node] < influence_radius:
                dg.node_nodes[i].append((node_index_map[other_node], fm.distance[other_node]))
    
    # 计算顶点-节点连接
    print("计算顶点-节点连接...")
    for node_idx, node in enumerate(nodes):
        # 计算从当前节点到所有顶点的距离
        fm.compute_distance([node], max_distance=influence_radius)
        
        # 为每个顶点添加当前节点的影响
        for v in range(n_vertices):
            if fm.distance[v] < influence_radius:
                weight = 1.0 - (fm.distance[v] / influence_radius)  # 简单的距离权重
                dg.v_nodes[v].append((node_idx, weight))
    
    # 减少连接数量
    print("优化连接数量...")
    dg.reduce_connections(node_nodes_num, v_nodes_num)
    
    # 归一化顶点-节点权重
    dg.normalize_weights()
    
    elapsed_time = time.time() - start_time
    print(f"变形图生成完成，耗时 {elapsed_time:.2f} 秒")
    
    return dg