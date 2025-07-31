import torch
import numpy as np
from scipy.spatial.distance import cdist

def subdivide_mesh(vertices, faces, subdivision_level=2):
    """
    对mesh进行细分，增加顶点密度
    
    Args:
        vertices: [N, V, 3] 原始顶点
        faces: [F, 3] 面片索引
        subdivision_level: 细分级别，每一级会增加约4倍的面数
    
    Returns:
        new_vertices: [N, V_new, 3] 细分后的顶点
    """
    batch_size = vertices.shape[0]
    device = vertices.device
    
    # 转换为numpy进行处理（批量处理每个batch）
    all_new_vertices = []
    
    for b in range(batch_size):
        verts = vertices[b].cpu().numpy()  # [V, 3]
        current_faces = faces.copy()
        current_verts = verts.copy()
        
        for level in range(subdivision_level):
            new_verts, new_faces = subdivide_once(current_verts, current_faces)
            current_verts = new_verts
            current_faces = new_faces
        
        all_new_vertices.append(torch.from_numpy(current_verts).to(device))
    
    # 确保所有batch的顶点数量一致
    max_verts = max([v.shape[0] for v in all_new_vertices])
    padded_vertices = []
    
    for verts in all_new_vertices:
        if verts.shape[0] < max_verts:
            # 用最后一个顶点填充（或者可以用其他策略）
            padding = verts[-1:].repeat(max_verts - verts.shape[0], 1)
            verts = torch.cat([verts, padding], dim=0)
        padded_vertices.append(verts)
    
    return torch.stack(padded_vertices, dim=0)

def subdivide_once(vertices, faces):
    """
    执行一次mesh细分
    """
    # 创建边到新顶点的映射
    edge_to_new_vert = {}
    new_vertices = list(vertices)
    
    # 为每条边创建中点
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edge_to_new_vert:
                # 创建边的中点
                midpoint = (vertices[v1] + vertices[v2]) / 2
                edge_to_new_vert[edge] = len(new_vertices)
                new_vertices.append(midpoint)
    
    new_vertices = np.array(new_vertices)
    
    # 创建新的面片
    new_faces = []
    for face in faces:
        v0, v1, v2 = face
        
        # 获取边中点的索引
        edge01 = edge_to_new_vert[tuple(sorted([v0, v1]))]
        edge12 = edge_to_new_vert[tuple(sorted([v1, v2]))]
        edge02 = edge_to_new_vert[tuple(sorted([v0, v2]))]
        
        # 创建4个新的三角形
        new_faces.extend([
            [v0, edge01, edge02],
            [v1, edge12, edge01], 
            [v2, edge02, edge12],
            [edge01, edge12, edge02]
        ])
    
    return new_vertices, np.array(new_faces)




# 如果有真实的FLAME obj文件，可以使用以下函数加载
def load_flame_obj(obj_path):
    """
    从obj文件加载FLAME模型的顶点和面片
    
    Args:
        obj_path: FLAME obj文件路径
    
    Returns:
        vertices: 顶点坐标
        faces: 面片索引
        uvs: UV坐标（如果存在）
    """
    vertices = []
    faces = []
    uvs = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # 顶点坐标
                coords = line.strip().split()[1:]
                vertices.append([float(x) for x in coords])
            elif line.startswith('f '):
                # 面片（需要处理可能的UV索引）
                face_data = line.strip().split()[1:]
                face_vertices = []
                for vertex_data in face_data:
                    # obj格式可能是 v/vt/vn 的形式
                    vertex_idx = int(vertex_data.split('/')[0]) - 1  # obj索引从1开始
                    face_vertices.append(vertex_idx)
                faces.append(face_vertices[:3])  # 只取前3个顶点（三角形）
            elif line.startswith('vt '):
                # UV坐标
                uv_coords = line.strip().split()[1:]
                uvs.append([float(x) for x in uv_coords[:2]])
    
    return np.array(vertices), np.array(faces), np.array(uvs) if uvs else None

# 使用示例：
# 如果有FLAME obj文件：
# vertices, faces, uvs = load_flame_obj('path/to/flame_model.obj')

# 然后在forward_geo_subdivided函数中使用真实的faces数据