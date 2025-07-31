import numpy as np
import os
import json

class DeformationGraph:
    """变形图类 - 用于网格变形的稀疏控制结构"""
    
    def __init__(self):
        """初始化一个空的变形图"""
        self.nodes = []                  # 控制节点索引
        self.node_positions = None       # 节点位置 (n x 3 矩阵)
        self.node_nodes = []             # 节点-节点连接及权重
        self.v_nodes = []                # 顶点-节点连接及权重
        self.node_radius = 0.0           # 节点影响半径
    
    def save(self, filename):
        """保存变形图到文件"""
        try:
            # 创建可序列化字典
            data = {
                'vertex_count': int(len(self.v_nodes)),
                'node_count': int(len(self.nodes)),
                'node_radius': float(self.node_radius),
                # 确保转换NumPy类型到Python原生类型
                'nodes': [int(node) for node in self.nodes],
                'node_positions': self.node_positions.tolist() if self.node_positions is not None else [],
                'node_nodes': [
                    [{'node': int(n), 'weight': float(w)} for n, w in connections] 
                    for connections in self.node_nodes
                ],
                'v_nodes': [
                    [{'node': int(n), 'weight': float(w)} for n, w in connections] 
                    for connections in self.v_nodes
                ]
            }
            
            # 写入JSON文件
            with open(filename, 'w') as f:
                import json
                json.dump(data, f, indent=2)
            
            print(f"变形图已保存到文件 {filename}")
            return True
        except Exception as e:
            print(f"保存文件时出错: {e}")
            return False
    
    def load(self, filename):
        """从文件加载变形图"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
                vn = data['vertex_count']
                gn = data['node_count']
                self.node_radius = data['node_radius']
                
                self.nodes = data['nodes']
                self.node_positions = np.array(data['node_positions'])
                
                # 加载连接信息
                self.node_nodes = [[(item['node'], item['weight']) for item in connections] 
                                 for connections in data['node_nodes']]
                
                self.v_nodes = [[(item['node'], item['weight']) for item in connections] 
                              for connections in data['v_nodes']]
            
            print(f"变形图已从文件 {filename} 加载")
            return True
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return False
    
    def reduce_connections(self, node_nodes_num=4, v_nodes_num=6):
        """减少连接数量至指定上限"""
        # 减少节点-节点连接
        for i, connections in enumerate(self.node_nodes):
            if len(connections) > node_nodes_num:
                # 按距离排序（权重越小的连接越重要）
                connections.sort(key=lambda x: x[1])
                self.node_nodes[i] = connections[:node_nodes_num]
        
        # 减少顶点-节点连接
        for i, connections in enumerate(self.v_nodes):
            if len(connections) > v_nodes_num:
                connections.sort(key=lambda x: x[1])
                self.v_nodes[i] = connections[:v_nodes_num]
    
    def normalize_weights(self):
        """规范化顶点-节点连接权重，使其和为1"""
        for i, connections in enumerate(self.v_nodes):
            if not connections:
                continue
                
            total_weight = sum(w for _, w in connections)
            if total_weight > 0:
                self.v_nodes[i] = [(n, w/total_weight) for n, w in connections]