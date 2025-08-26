
import torch
import pickle
import smplx

def create_tensor_flame_geometry(flame_path):
    from flame_pytorch import FLAME, parse_args
    payload=torch.load(flame_path,weights_only=False)
    flame_params = {}
    if 'flame' in payload:
        flame_data = payload['flame']
        flame_params = {
            'shape': flame_data.get('shape', None),
            'exp': flame_data.get('exp', None),
            'global_rotation': flame_data.get('global_rotation', None),
            'jaw': flame_data.get('jaw', None),
            'neck': flame_data.get('neck', None),
            'eyes': flame_data.get('eyes', None),
            'transl': flame_data.get('transl', None),
            'scale_factor': flame_data.get('scale_factor', None)
        }
        # Extract FLAME parameters
        shape_params = torch.as_tensor(flame_params['shape']) if 'shape' in flame_params else None
        expression_params = torch.as_tensor(flame_params['exp']) if 'exp' in flame_params else None
        
        # Process pose parameters
        global_rotation = torch.as_tensor(flame_params.get('global_rotation', torch.zeros(3))) if 'global_rotation' in flame_params else torch.zeros(3)
        jaw_pose = torch.as_tensor(flame_params.get('jaw', torch.zeros(3))) if 'jaw' in flame_params else torch.zeros(3)
        neck_pose = torch.as_tensor(flame_params.get('neck', torch.zeros(3))) if 'neck' in flame_params else torch.zeros(3)
        eye_pose = torch.as_tensor(flame_params['eyes']) if 'eyes' in flame_params else torch.zeros(6)
        transl_pose = torch.as_tensor(flame_params.get('transl', torch.zeros(3))) if 'transl' in flame_params else torch.zeros(3)
        scale_factor = torch.as_tensor(flame_params.get('scale_factor', torch.ones(1))) if 'scale_factor' in flame_params else torch.ones(1)
        pose_params = torch.cat([global_rotation, jaw_pose], dim=1)

        config = parse_args()
        flamelayer = FLAME(config)

        vertices=flamelayer.forward_geo_subdivided(
            shape_params, expression_params, pose_params, neck_pose, eye_pose, transl_pose,scale_factor,1
        )
    return vertices

def create_np_flame_geometry(flame_path):
    vertices=create_tensor_flame_geometry(flame_path)
    # Convert vertices to numpy and add to list
    if isinstance(vertices, torch.Tensor):
        vertices_np = vertices.detach().cpu().numpy()
        if vertices_np.ndim == 3:  # If has batch dimension, take first
            vertices_np = vertices_np[0]
        return vertices_np
    else:
        return vertices
            

def create_tensor_smplx_geometry(smplx_path):
    model = smplx.create("models", 
                        model_type='smplx', 
                        gender='neutral',
                        num_betas=300,
                        num_expression_coeffs=100,
                        num_pca_comps=0)
    
    with open(smplx_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract parameters and convert to tensors
    betas = torch.tensor(data['betas'], dtype=torch.float32).unsqueeze(0)
    expression = torch.tensor(data['expression'], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.tensor(data['body_pose'], dtype=torch.float32).unsqueeze(0)
    global_orient = torch.tensor(data['global_orient'], dtype=torch.float32).unsqueeze(0)
    transl = torch.tensor(data['transl'], dtype=torch.float32).unsqueeze(0)
    
    # Generate mesh
    output = model(betas=betas, 
                    expression=expression, 
                    body_pose=body_pose, 
                    global_orient=global_orient,
                    transl=transl,
                    return_verts=True)
    
    # Extract vertices
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    return vertices

def create_np_smplx_geometry(smplx_path):
    vertices=create_tensor_smplx_geometry(smplx_path)
    # Convert vertices to numpy and add to list
    if isinstance(vertices, torch.Tensor):
        vertices_np = vertices.detach().cpu().numpy()
        if vertices_np.ndim == 3:  # If has batch dimension, take first
            vertices_np = vertices_np[0]
        return vertices_np
    else:
        return vertices

def load_flame_codedict(flame_path):
    payload=torch.load(flame_path,weights_only=False)
    flame_params = {}
    if 'flame' in payload:
        flame_data = payload['flame']
        flame_params = {
            'shape': flame_data.get('shape', None),
            'exp': flame_data.get('exp', None),
            'global_rotation': flame_data.get('global_rotation', None),
            'jaw': flame_data.get('jaw', None),
            'neck': flame_data.get('neck', None),
            'eyes': flame_data.get('eyes', None),
            'transl': flame_data.get('transl', None),
            'scale_factor': flame_data.get('scale_factor', None)
        }
        shape_params = torch.as_tensor(flame_params['shape']).to("cuda")
        expression_params = torch.as_tensor(flame_params['exp']).to("cuda")
        
        # Process pose parameters
        global_rotation = torch.as_tensor(flame_params.get('global_rotation', torch.zeros(3))).to("cuda")
        jaw_pose = torch.as_tensor(flame_params.get('jaw', torch.zeros(3))).to("cuda")
        neck_pose = torch.as_tensor(flame_params.get('neck', torch.zeros(3))).to("cuda")
        eye_pose = torch.as_tensor(flame_params['eyes']).to("cuda")
        transl_pose = torch.as_tensor(flame_params.get('transl', torch.zeros(3))).to("cuda")
        scale_factor = torch.as_tensor(flame_params.get('scale_factor', torch.ones(1))).reshape(shape_params.shape[0],1).to("cuda")
        

        return {
        'shape':shape_params,
        'exp': expression_params,
        'global_rotation': global_rotation,
        'jaw': jaw_pose,
        'neck': neck_pose,
        'eyes': eye_pose,
        'transl': transl_pose,
        'scale_factor': scale_factor
        }
    else:
        return None

def load_smplx_codedict(smplx_path):
    with open(smplx_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract parameters and convert to tensors
    betas = torch.tensor(data['betas'], dtype=torch.float32).unsqueeze(0)
    expression = torch.tensor(data['expression'], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.tensor(data['body_pose'], dtype=torch.float32).unsqueeze(0)
    global_orient = torch.tensor(data['global_orient'], dtype=torch.float32).unsqueeze(0)
    transl = torch.tensor(data['transl'], dtype=torch.float32).unsqueeze(0)
    return {
        "betas":betas,
        "expression":expression,
        "body_pose":body_pose,
        "global_orient":global_orient,
        "transl":transl
    }

def generate_flame_geometry(codedict,model=None):

    if model==None:
        from flame_pytorch import FLAME, parse_args
        flame_config = parse_args()
        model = FLAME(flame_config).to("cuda")
    
    shape_param = codedict['shape'].detach()
    exp_param = codedict['exp'].detach()
    global_rotation = codedict['global_rotation'].detach()
    jaw_pose = codedict['jaw'].detach()
    neck_pose=codedict['neck'].detach()
    eyes_pose = codedict['eyes'].detach()
    transl = codedict['transl'].detach()
    scale_factor=codedict['scale_factor'].detach()

    pose_params = torch.cat((global_rotation, jaw_pose), dim=1)
    geometry =model.forward_geo(
        shape_params=shape_param,
        expression_params=exp_param,
        pose_params=pose_params,
        neck_pose=neck_pose,
        eye_pose=eyes_pose,
        transl=transl,
        scale_factor= scale_factor
    )
    return geometry.squeeze(0)

def generate_smplx_geometry(codedict,model=None):
    if model==None:
        model = smplx.create("models", 
                        model_type='smplx', 
                        gender='neutral',
                        num_betas=300,
                        num_expression_coeffs=100,
                        num_pca_comps=0)
    
    betas = codedict['betas'].detach().cpu()
    expression =codedict['expression'].detach().cpu()
    body_pose = codedict['body_pose'].detach().cpu()
    global_orient = codedict['global_orient'].detach().cpu()
    transl = codedict['transl'].detach().cpu()
    
    # Generate mesh
    output = model(betas=betas, 
                    expression=expression, 
                    body_pose=body_pose, 
                    global_orient=global_orient,
                    transl=transl,
                    return_verts=True)
    
    # Extract vertices
    # vertices = output.vertices.detach().cpu().numpy().squeeze()
    vertices=output.vertices.detach().squeeze()
    return vertices.to("cuda")

import numpy as np
from typing import Tuple

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
                            faces.append([face_indices[0], face_indices[1], face_indices[2]])
                            faces.append([face_indices[0], face_indices[2], face_indices[3]])
                        elif triangulate and len(face_indices) > 4:
                            # 将多边形扇形三角化
                            for i in range(1, len(face_indices) - 1):
                                faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
                        else:
                            # 保持原始面片
                            faces.append(face_indices)
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