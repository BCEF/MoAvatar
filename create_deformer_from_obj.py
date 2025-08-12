import numpy as np
from typing import Tuple
import os
from pathlib import Path
from deformation_graph import generate_deformation_graph,compute_deformation_transforms
from deformation_graph import DeformationGraph
from deformation_graph import apply_deformation_to_gaussians

from typing import Union, List
import torch
def read_obj(file_path: str, triangulate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取OBJ文件，返回顶点和面数据
    
    Args:
        file_path: OBJ文件路径
        triangulate: 是否将四边形面片转换为三角形
        
    Returns:
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面数组，形状为 (M, 3) 或 (M, 4)
    """
    vertices = []
    faces = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            try:
                if parts[0] == 'v':
                    # 读取顶点坐标
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
                    else:
                        print(f"警告: 第{line_num}行顶点数据不完整: {line}")
                        
                elif parts[0] == 'f':
                    # 读取面片数据
                    face_indices = []
                    for vertex_data in parts[1:]:
                        # 处理 "v", "v/vt", "v/vt/vn", "v//vn" 等格式
                        vertex_index = vertex_data.split('/')[0]
                        if vertex_index:
                            # OBJ索引从1开始，转换为从0开始
                            idx = int(vertex_index) - 1
                            face_indices.append(idx)
                    
                    if len(face_indices) >= 3:
                        if triangulate and len(face_indices) == 4:
                            # 将四边形分解为两个三角形
                            # 四边形 [0,1,2,3] -> 三角形 [0,1,2] 和 [0,2,3]
                            faces.append(np.array([face_indices[0], face_indices[1], face_indices[2]]))
                            faces.append(np.array([face_indices[0], face_indices[2], face_indices[3]]))
                        elif triangulate and len(face_indices) > 4:
                            # 将多边形扇形三角化
                            for i in range(1, len(face_indices) - 1):
                                faces.append(np.array([face_indices[0], face_indices[i], face_indices[i + 1]]))
                        else:
                            # 保持原始面片
                            faces.append(np.array(face_indices))
                    else:
                        print(f"警告: 第{line_num}行面片数据不完整: {line}")
                        
            except (ValueError, IndexError) as e:
                print(f"错误: 第{line_num}行解析失败: {line} - {e}")
                continue
    
    # 转换为numpy数组
    vertices = np.array(vertices, dtype=np.float32)
    
    if faces:
        # 检查所有面片是否有相同的顶点数
        face_lengths = [len(face) for face in faces]
        if len(set(face_lengths)) == 1:
            # 所有面片顶点数相同，可以创建规则数组
            faces = np.array(faces, dtype=np.int32)
        else:
            # 面片顶点数不同，保持为列表
            faces = np.array(faces, dtype=object)
    else:
        faces = np.array([], dtype=np.int32).reshape(0, 3)
    
    print(f"读取完成: {len(vertices)} 个顶点, {len(faces)} 个面片")
    
    return vertices, faces



def save_obj(file_path: str, vertices: np.ndarray, faces: Union[np.ndarray, List], 
             header_comment: str = None) -> bool:
    """
    保存OBJ文件
    
    Args:
        file_path: 输出OBJ文件路径
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面数组，形状为 (M, 3) 或 (M, 4)，或者包含不同长度面片的列表
        header_comment: 可选的文件头注释
        
    Returns:
        bool: 是否保存成功
    """
    try:
        # 验证输入数据
        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)
        
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"顶点数组形状应为 (N, 3)，当前为 {vertices.shape}")
        
        # 处理面片数据
        if isinstance(faces, np.ndarray) and faces.dtype == object:
            # object数组，面片长度可能不同
            face_list = faces.tolist()
        elif isinstance(faces, np.ndarray):
            # 规则数组
            if faces.ndim == 2:
                face_list = faces.tolist()
            else:
                raise ValueError(f"面片数组维度应为2，当前为 {faces.ndim}")
        elif isinstance(faces, list):
            face_list = faces
        else:
            raise ValueError("faces应为numpy数组或列表")
        
        # 检查面片索引是否有效
        max_vertex_index = len(vertices) - 1
        for i, face in enumerate(face_list):
            for vertex_idx in face:
                if vertex_idx < 0 or vertex_idx > max_vertex_index:
                    raise ValueError(f"面片 {i} 包含无效的顶点索引 {vertex_idx}，"
                                   f"有效范围为 0-{max_vertex_index}")
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as file:
            # 写入文件头注释
            if header_comment:
                file.write(f"# {header_comment}\n")
            file.write(f"# OBJ文件\n")
            file.write(f"# 顶点数: {len(vertices)}\n")
            file.write(f"# 面片数: {len(face_list)}\n\n")
            
            # 写入顶点数据
            for vertex in vertices:
                # 确保顶点坐标为浮点数并格式化输出
                x, y, z = float(vertex[0]), float(vertex[1]), float(vertex[2])
                file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            
            file.write("\n")
            
            # 写入面片数据
            for face in face_list:
                if len(face) < 3:
                    print(f"警告: 跳过不完整的面片 {face} (顶点数少于3)")
                    continue
                
                # OBJ格式索引从1开始，所以要+1
                face_indices = [str(int(idx) + 1) for idx in face]
                file.write(f"f {' '.join(face_indices)}\n")
        
        print(f"成功保存OBJ文件: {file_path}")
        print(f"  顶点数: {len(vertices)}")
        print(f"  面片数: {len(face_list)}")
        return True
        
    except Exception as e:
        print(f"保存OBJ文件失败: {e}")
        return False

def save_obj_with_normals(file_path: str, vertices: np.ndarray, faces: Union[np.ndarray, List],
                         normals: np.ndarray = None, texture_coords: np.ndarray = None,
                         header_comment: str = None) -> bool:
    """
    保存带法线和纹理坐标的OBJ文件
    
    Args:
        file_path: 输出OBJ文件路径
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面数组
        normals: 法线数组，形状为 (N, 3)，可选
        texture_coords: 纹理坐标数组，形状为 (N, 2)，可选
        header_comment: 可选的文件头注释
        
    Returns:
        bool: 是否保存成功
    """
    try:
        # 验证输入数据
        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)
        
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"顶点数组形状应为 (N, 3)，当前为 {vertices.shape}")
        
        num_vertices = len(vertices)
        
        # 验证法线数据
        if normals is not None:
            if not isinstance(normals, np.ndarray):
                normals = np.array(normals)
            if normals.shape != (num_vertices, 3):
                raise ValueError(f"法线数组形状应为 ({num_vertices}, 3)，当前为 {normals.shape}")
        
        # 验证纹理坐标数据
        if texture_coords is not None:
            if not isinstance(texture_coords, np.ndarray):
                texture_coords = np.array(texture_coords)
            if texture_coords.shape != (num_vertices, 2):
                raise ValueError(f"纹理坐标数组形状应为 ({num_vertices}, 2)，当前为 {texture_coords.shape}")
        
        # 处理面片数据
        if isinstance(faces, np.ndarray) and faces.dtype == object:
            face_list = faces.tolist()
        elif isinstance(faces, np.ndarray):
            face_list = faces.tolist()
        else:
            face_list = faces
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as file:
            # 写入文件头
            if header_comment:
                file.write(f"# {header_comment}\n")
            file.write(f"# OBJ文件\n")
            file.write(f"# 顶点数: {num_vertices}\n")
            file.write(f"# 面片数: {len(face_list)}\n\n")
            
            # 写入顶点数据
            for vertex in vertices:
                x, y, z = float(vertex[0]), float(vertex[1]), float(vertex[2])
                file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            
            # 写入纹理坐标
            if texture_coords is not None:
                file.write("\n")
                for tc in texture_coords:
                    u, v = float(tc[0]), float(tc[1])
                    file.write(f"vt {u:.6f} {v:.6f}\n")
            
            # 写入法线数据
            if normals is not None:
                file.write("\n")
                for normal in normals:
                    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
                    file.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            
            file.write("\n")
            
            # 写入面片数据
            for face in face_list:
                if len(face) < 3:
                    continue
                
                face_str_parts = []
                for idx in face:
                    vertex_idx = int(idx) + 1  # OBJ索引从1开始
                    
                    # 构建面片索引字符串 v/vt/vn
                    if texture_coords is not None and normals is not None:
                        face_str_parts.append(f"{vertex_idx}/{vertex_idx}/{vertex_idx}")
                    elif texture_coords is not None:
                        face_str_parts.append(f"{vertex_idx}/{vertex_idx}")
                    elif normals is not None:
                        face_str_parts.append(f"{vertex_idx}//{vertex_idx}")
                    else:
                        face_str_parts.append(str(vertex_idx))
                
                file.write(f"f {' '.join(face_str_parts)}\n")
        
        print(f"成功保存OBJ文件: {file_path}")
        return True
        
    except Exception as e:
        print(f"保存OBJ文件失败: {e}")
        return False

from deformation_graph import visualize_deformation_graph
import trimesh

if __name__=='__main__':
    root_folder="/home/momo/Desktop/data/ali/cc01data/"
    data_folder="/home/momo/Desktop/data/ali/cc01data/frames"
    obj_folder="/home/momo/Desktop/data/ali/cc01data/obj/"
    prefix='Frame0'
    root_path = Path(data_folder)
    
    # 检查根文件夹是否存在
    if not root_path.exists():
        print(f"错误: 根文件夹 '{data_folder}' 不存在")

    
    # 获取所有子文件夹并排序（确保索引顺序一致）
    subfolders = [f for f in root_path.iterdir() if f.is_dir() and 'sparse' not in f.name]
    subfolders.sort()

    base_path=os.path.join(obj_folder,prefix+subfolders[0].name+'.obj')
    base_vertex,faces=read_obj(base_path)
    print(base_path)
    if not os.path.exists(os.path.join(data_folder,'deformation_graph.json')):
        # dg=generate_deformation_graph(base_vertex,faces,node_num=200,radius_coef=5,node_nodes_num=16,v_nodes_num=12)
        dg=generate_deformation_graph(base_vertex,faces,node_num=200,radius_coef=5,node_nodes_num=8,v_nodes_num=12)
        dg.save(os.path.join(data_folder,'deformation_graph.json'))
        mesh=trimesh.load(base_path)
        visualize_deformation_graph(mesh,dg)
    else:
        dg = DeformationGraph()
        dg.load(os.path.join(data_folder,'deformation_graph.json'))
    # base_vertex=torch.as_tensor(base_vertex,dtype=torch.float32)
    for subfolder in subfolders:
        base_vertex,faces=read_obj(base_path)
        current_vertex,_=read_obj(os.path.join(obj_folder,prefix+subfolder.name+'.obj'))
        # current_vertex=torch.as_tensor(current_vertex,dtype=torch.float32)

        transforms=compute_deformation_transforms(dg,base_vertex,current_vertex)
        transforms.save(os.path.join(data_folder,subfolder.name,'transforms.json'))

        base_vertex,faces=read_obj(base_path)
        current_vertex,_=read_obj(os.path.join(obj_folder,prefix+subfolder.name+'.obj'))
        inv_transforms=compute_deformation_transforms(dg,current_vertex,base_vertex)
        inv_transforms.save(os.path.join(data_folder,subfolder.name,'inv_transforms.json'))

        base_vertex,faces=read_obj(base_path)
        deform_points=apply_deformation_to_gaussians(dg,base_vertex,transforms)
        os.makedirs(os.path.join(root_folder,"test"),exist_ok=True)
        save_obj(os.path.join(root_folder,"test",subfolder.name+'.obj'),deform_points['xyz'],faces)