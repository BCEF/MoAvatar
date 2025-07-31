import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra

class FastMarching:
    """Fast Marching算法的Python实现，使用scipy的Dijkstra算法"""
    
    def __init__(self):
        """初始化Fast Marching工具"""
        self.vertices = None
        self.faces = None
        self.vertex_adjacency = None
        self.distance = None
        self.seed_points = []
        self.max_distance = float('inf')
        self.iter_max = 1000
    
    def set_mesh(self, vertices, faces):
        """设置用于计算的网格并手动构建邻接图"""
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int32)
        
        # 手动构建顶点邻接图
        n_vertices = len(vertices)
        # 使用字典列表表示邻接关系
        adjacency = [set() for _ in range(n_vertices)]
        
        # 从面中提取边
        for face in faces:
            for i in range(3):  # 假设是三角形网格
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
        
        # 转换为列表
        self.vertex_adjacency = [list(neighbors) for neighbors in adjacency]
        return True
    
    def compute_distance(self, seed_points, max_distance=float('inf')):
        """使用Dijkstra算法计算从种子点到所有顶点的测地线距离"""
        self.seed_points = seed_points
        self.max_distance = max_distance
        
        if len(seed_points) == 0:
            print("错误：没有提供种子点")
            return False
        
        # 创建稀疏图
        rows, cols, data = [], [], []
        
        for i, neighbors in enumerate(self.vertex_adjacency):
            for j in neighbors:
                dist = np.linalg.norm(self.vertices[i] - self.vertices[j])
                rows.append(i)
                cols.append(j)
                data.append(dist)
        
        graph = sparse.csr_matrix((data, (rows, cols)), 
                                 shape=(len(self.vertices), len(self.vertices)))
        
        # 计算最短路径
        self.distance = np.inf * np.ones(len(self.vertices))
        for seed in seed_points:
            # 为每个种子点计算最短路径
            dist_matrix = dijkstra(graph, indices=[seed], limit=max_distance)
            # 更新为最小距离
            self.distance = np.minimum(self.distance, dist_matrix[0])
        
        return True
    
    def farthest_point_sampling(self, num_samples, initial_point=None):
        """使用最远点采样算法选择顶点"""
        n_vertices = len(self.vertices)
        
        # 如果没有提供初始点，随机选择一个
        if initial_point is None:
            initial_point = np.random.randint(0, n_vertices)
        
        # 添加第一个点
        samples = [initial_point]
        
        # 计算从初始点到所有点的距离
        self.compute_distance([initial_point])
        distances = self.distance.copy()
        
        # 迭代选择最远点
        for _ in range(1, num_samples):
            # 找到距离当前样本集最远的点
            next_point = np.argmax(distances)
            samples.append(next_point)
            
            # 更新距离
            self.compute_distance([next_point])
            new_distances = self.distance.copy()
            
            # 更新每个点到最近采样点的距离
            distances = np.minimum(distances, new_distances)
        
        self.seed_points = samples
        return samples, np.max(distances)