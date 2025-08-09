import numpy as np
from plyfile import PlyData, PlyElement
from .deformation_graph import DeformationGraph
from .deformation_utils import DeformationTransforms
import json

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
    import time
    
    # start_time = time.time()
    # print("开始应用变形...")
    
    # 创建结果字典，复制输入高斯数据
    # deformed_gaussian = {key: value.copy() for key, value in gaussian.items()}
    deformed_gaussian={'xyz':points}
                    #    'rotations':np.zeros((points.shape[0],4))}

    # 获取变换矩阵和控制节点信息
    transformations = np.array(transforms.transformations)
    node_positions = dg.node_positions
    influence_radius = dg.node_radius
    
    # 获取所有高斯点的位置和旋转
    # points = gaussian['xyz']
    point_count = points.shape[0]
    # print(f"处理 {point_count} 个高斯点...")
    
    # 1. 使用KD树加速最近点查找
    # print("构建KD树...")
    kdtree = cKDTree(node_positions)
    
    # 2. 预计算控制节点的SVD分解结果
    # print("预计算旋转矩阵...")
    rotation_matrices = []
    rotation_objects = []
    for t in transformations:
        R = t[:3, :3]
        u, s, vh = np.linalg.svd(R, full_matrices=False)
        orthogonal_R = u @ vh
        rotation_matrices.append(orthogonal_R)
        rotation_objects.append(Rotation.from_matrix(orthogonal_R))
    
    # 3. 批处理 + 稀疏表示
    batch_size = 20000  # 可调整
    num_batches = (point_count + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, point_count)
        batch_points = points[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        # print(f"处理批次 {batch_idx+1}/{num_batches} ({batch_size_actual} 点)...")
        
        # 4. 快速查找每个点的影响节点 (半径查询)
        influence_indices = kdtree.query_ball_point(batch_points, influence_radius)
        
        # 5. 预先分配结果数组，避免重复分配内存
        batch_positions = deformed_gaussian['xyz'][start_idx:end_idx].copy()
        # batch_rotations = deformed_gaussian['rotations'][start_idx:end_idx].copy()
        
        # 跟踪需要处理的点
        points_to_process = []
        for i, indices in enumerate(influence_indices):
            if len(indices) > 0:
                points_to_process.append(i)
        
        # print(f"  批次中有 {len(points_to_process)}/{batch_size_actual} 点需要处理")
        
        # 6. 只处理有影响节点的点
        for local_idx in points_to_process:
            global_idx = start_idx + local_idx
            point_pos = batch_points[local_idx]
            # orig_quat = gaussian['rotations'][global_idx]
            # orig_quat=points_rots[global_idx]
            
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
            
            # # 8. 旋转计算 - 使用预计算的旋转对象
            # try:
            #     # 找到权重最大的节点
            #     max_weight_idx = np.argmax(weights)
            #     node_idx = node_indices[max_weight_idx]
                
            #     # 转换原始四元数为scipy格式
            #     orig_w, orig_x, orig_y, orig_z = orig_quat
            #     scipy_quat = np.array([orig_x, orig_y, orig_z, orig_w])
            #     original_rotation = Rotation.from_quat(scipy_quat)
                
            #     # 使用预计算的旋转对象进行组合
            #     combined_rotation = rotation_objects[node_idx] * original_rotation
                
            #     # 转换回w,x,y,z格式
            #     x, y, z, w = combined_rotation.as_quat()
            #     result_quat = np.array([w, x, y, z])
                
            #     # 确保符号一致
            #     if np.dot(result_quat, orig_quat) < 0:
            #         result_quat = -result_quat
                
            #     batch_rotations[local_idx] = result_quat
            # except Exception as e:
            #     if local_idx % 5000 == 0:
            #         print(f"  处理旋转出错 (点 {global_idx}): {e}")
        
        # 更新结果数组
        deformed_gaussian['xyz'][start_idx:end_idx] = batch_positions
        # deformed_gaussian['rotations'][start_idx:end_idx] = batch_rotations
        
        # batch_time = time.time() - batch_start
        # print(f"  批次处理完成，耗时: {batch_time:.2f}秒")
    
    # total_time = time.time() - start_time
    # print(f"变形处理完成 - 总耗时: {total_time:.2f}秒")
    
    return deformed_gaussian

class GaussianDeformer:
    """高斯点云变形工具"""
    
    def __init__(self, deformation_graph_path, device='cuda'):
        """初始化高斯变形器"""
        self.device = device
        
        # 加载变形图
        
        self.dg = DeformationGraph()
        self.dg.load(deformation_graph_path)
        
        print(f"变形图已加载: {len(self.dg.nodes)} 控制节点")
    
    def read_gaussian_ply(self, path):
        """
        读取高斯PLY文件并提取所有需要的数据
        
        参数:
            path: PLY文件路径
            
        返回:
            gaussian: 包含所有高斯数据的字典
        """
        print(f"读取高斯PLY: {path}")

        gaussian = {}
        plydata = PlyData.read(path)
        elem = plydata.elements[0]

        # 直接用numpy数组
        xyz = np.stack([elem["x"], elem["y"], elem["z"]], axis=1).astype(np.float32)
        opacities = elem["opacity"].astype(np.float32).reshape(-1, 1)

        features_dc = np.stack([elem["f_dc_0"], elem["f_dc_1"], elem["f_dc_2"]], axis=1).astype(np.float32)

        # 加载额外特征
        extra_f_names = sorted([p.name for p in elem.properties if p.name.startswith("f_rest_")],
                                key=lambda x: int(x.split('_')[-1]))
        features_extra = np.stack([elem[name] for name in extra_f_names], axis=1).astype(np.float32) if extra_f_names else np.empty((xyz.shape[0], 0), dtype=np.float32)

        # 加载缩放
        scale_names = sorted([p.name for p in elem.properties if p.name.startswith("scale_")],
                            key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([elem[name] for name in scale_names], axis=1).astype(np.float32) if scale_names else np.empty((xyz.shape[0], 0), dtype=np.float32)

        # 加载旋转
        rot_names = sorted([p.name for p in elem.properties if p.name.startswith("rot")],
                        key=lambda x: int(x.split('_')[-1]))
        rots = np.stack([elem[name] for name in rot_names], axis=1).astype(np.float32) if rot_names else np.empty((xyz.shape[0], 0), dtype=np.float32)
        


        gaussian['xyz'] = xyz
        gaussian['features_dc'] = features_dc
        gaussian['features_extra'] = features_extra
        gaussian['scales'] = scales
        gaussian['opacities'] = opacities
        gaussian['rotations'] = rots

        property_names = [p.name for p in elem.properties]
        if "kid" in property_names:
            # 如果存在kid属性，直接读取
            gaussian['kid'] = elem["kid"].astype(np.int32).reshape(-1, 1)
            print(f"使用PLY文件中的'kid'属性")
        else:
            # 如果不存在kid属性，使用顶点索引
            gaussian['kid'] = np.arange(gaussian['xyz'].shape[0], dtype=np.int32).reshape(-1, 1)
            print(f"PLY文件中未找到'kid'属性，使用顶点索引作为kid")
        return gaussian
    
    def save_gaussian_ply(self, gaussian, output_path):
        """
        保存高斯数据到PLY文件
        
        参数:
            gaussian: 高斯数据字典
            output_path: 输出PLY文件路径
        """
        xyz = gaussian['xyz']
        N = xyz.shape[0]
        rots=gaussian['rotations']
        features_dc = gaussian['features_dc']
        features_extra = gaussian['features_extra']
        scales = gaussian['scales']
        opacities = gaussian['opacities']
        kid=gaussian['kid']
       # 动态拼接字段
        dtype = [('kid', 'i4'),('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
        
        for i in range(features_extra.shape[1]):
            dtype.append((f'f_rest_{i}', 'f4'))
        
        for i in range(scales.shape[1]):
            dtype.append((f'scale_{i}', 'f4'))
        
        for i in range(rots.shape[1]):
            dtype.append((f'rot_{i}', 'f4'))
        
        dtype.append(('opacity', 'f4'))
        
        # 构建输出数据
        vertex_data = np.empty(N, dtype=dtype)
        vertex_data['kid'] = kid[:, 0]
        vertex_data['x'] = xyz[:, 0]
        vertex_data['y'] = xyz[:, 1]
        vertex_data['z'] = xyz[:, 2]
        vertex_data['f_dc_0'] = features_dc[:, 0]
        vertex_data['f_dc_1'] = features_dc[:, 1]
        vertex_data['f_dc_2'] = features_dc[:, 2]
        
        for i in range(features_extra.shape[1]):
            vertex_data[f'f_rest_{i}'] = features_extra[:, i]
        
        for i in range(scales.shape[1]):
            vertex_data[f'scale_{i}'] = scales[:, i]
        
        for i in range(rots.shape[1]):
            vertex_data[f'rot_{i}'] = rots[:, i]
        
        vertex_data['opacity'] = opacities[:, 0]
        
        # 写 ply 文件
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(output_path)
        print(f"变换后的PLY已保存: {output_path}")


# def process_gaussian_ply(input_gaussian_ply_path, output_gaussian_ply_path, deformation_graph_path, transform_path):
#     """
#     处理高斯PLY文件，应用变形并保存结果
    
#     参数:
#         input_gaussian_ply_path: 输入高斯PLY文件路径
#         output_gaussian_ply_path: 输出高斯PLY文件路径
#         deformation_graph_path: 变形图路径
#         transform_path: 变换参数路径
#     """
#     deformer = GaussianDeformer(deformation_graph_path)
    
#     # 读取高斯数据
#     gaussian = deformer.read_gaussian_ply(input_gaussian_ply_path)
    
#     transforms = DeformationTransforms()
#     transforms.load(transform_path)
    
#     # 应用变形
#     deformed_gaussian = apply_deformation_to_gaussians(deformer.dg, gaussian, transforms)
    
#     # 保存变形后的高斯数据
#     deformer.save_gaussian_ply(deformed_gaussian, output_gaussian_ply_path)
#     print(f"变形后的高斯PLY已保存: {output_gaussian_ply_path}")

# if __name__ == "__main__":
#     # 示例用法
#     deformation_graph_path = "/home/momo/Desktop/deform_3DGS_data/ali_data/deform_graph/deformation_graph.json"
#     gaussian_ply_path = "/home/momo/Desktop/deform_3DGS_data/ali_data/GaussianInit_speedy.ply"
#     transform_path="/home/momo/Desktop/deform_3DGS_data/ali_data/deform_graph/Frame000040.json"
#     output_gaussian_ply_path = "output_gaussian.ply"
    
#     import time

#     deformer = GaussianDeformer(deformation_graph_path)
    
#     # 读取高斯数据
#     gaussian = deformer.read_gaussian_ply(gaussian_ply_path)
#     transforms = DeformationTransforms()
#     transforms.load(transform_path)
#     # 应用变形
#     st=time.time()
#     deformed_gaussian = apply_deformation_to_gaussians(deformer.dg, gaussian, transforms)
    
#     # 保存变形后的高斯数据
#     deformer.save_gaussian_ply(deformed_gaussian, output_gaussian_ply_path)
#     print("变形完成")
#     print("耗时:",time.time()-st)

