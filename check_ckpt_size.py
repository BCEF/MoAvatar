import torch
import sys

def calculate_dict_size(obj, visited=None):
    """递归计算字典或其他对象的实际内存大小"""
    if visited is None:
        visited = set()
    
    # 避免循环引用
    obj_id = id(obj)
    if obj_id in visited:
        return 0
    visited.add(obj_id)
    
    total_size = 0
    
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    elif hasattr(obj, 'nbytes'):  # numpy数组
        return obj.nbytes
    elif isinstance(obj, dict):
        total_size += sys.getsizeof(obj)  # 字典本身的大小
        for key, value in obj.items():
            total_size += sys.getsizeof(key)  # 键的大小
            total_size += calculate_dict_size(value, visited)  # 递归计算值的大小
    elif isinstance(obj, (list, tuple)):
        total_size += sys.getsizeof(obj)
        for item in obj:
            total_size += calculate_dict_size(item, visited)
    elif isinstance(obj, str):
        total_size += sys.getsizeof(obj)
    else:
        # 对于自定义对象，检查其所有属性
        total_size += sys.getsizeof(obj)
        
        # 尝试获取对象的所有属性
        try:
            if hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    total_size += sys.getsizeof(attr_name)  # 属性名大小
                    total_size += calculate_dict_size(attr_value, visited)  # 属性值大小
            elif hasattr(obj, '__slots__'):
                for slot in obj.__slots__:
                    if hasattr(obj, slot):
                        attr_value = getattr(obj, slot)
                        total_size += sys.getsizeof(slot)  # 槽名大小
                        total_size += calculate_dict_size(attr_value, visited)  # 槽值大小
        except (AttributeError, RecursionError):
            # 如果无法访问属性，只返回对象本身的大小
            pass
    
    return total_size

def analyze_deformation_graph(dg_obj):
    """分析DeformationGraph对象的详细信息"""
    print(f"\n🔗 DeformationGraph 详细分析:")
    print(f"  - 对象类型: {type(dg_obj).__module__}.{type(dg_obj).__name__}")
    
    total_size_mb = 0
    
    if hasattr(dg_obj, '__dict__'):
        print(f"  - 属性列表:")
        for attr_name, attr_value in dg_obj.__dict__.items():
            attr_type = type(attr_value).__name__
            attr_size_mb = 0
            
            if isinstance(attr_value, torch.Tensor):
                attr_size_mb = attr_value.numel() * attr_value.element_size() / (1024**2)
                print(f"    * {attr_name}: Tensor{list(attr_value.shape)} ({attr_size_mb:.2f} MB)")
                
            elif hasattr(attr_value, 'nbytes'):  # numpy数组
                attr_size_mb = attr_value.nbytes / (1024**2)
                print(f"    * {attr_name}: {attr_type}{list(attr_value.shape)} ({attr_size_mb:.2f} MB)")
                
            elif isinstance(attr_value, (list, tuple)) and len(attr_value) > 0:
                # 计算列表的实际大小
                list_size_bytes = calculate_dict_size(attr_value)
                attr_size_mb = list_size_bytes / (1024**2)
                
                # 检查第一个元素的类型以获取更多信息
                first_elem_type = type(attr_value[0]).__name__ if len(attr_value) > 0 else "empty"
                print(f"    * {attr_name}: {attr_type}[{len(attr_value)} {first_elem_type}] ({attr_size_mb:.2f} MB)")
                
                # 如果是小列表，显示一些样本
                if len(attr_value) <= 5:
                    for i, item in enumerate(attr_value):
                        item_type = type(item).__name__
                        if hasattr(item, 'shape'):
                            print(f"      [{i}]: {item_type}{list(item.shape)}")
                        elif hasattr(item, '__len__') and not isinstance(item, str):
                            print(f"      [{i}]: {item_type}[{len(item)}]")
                        else:
                            print(f"      [{i}]: {item_type}")
                            
            elif isinstance(attr_value, dict):
                dict_size_bytes = calculate_dict_size(attr_value)
                attr_size_mb = dict_size_bytes / (1024**2)
                print(f"    * {attr_name}: {attr_type}[{len(attr_value)} items] ({attr_size_mb:.2f} MB)")
                    
            else:
                # 其他类型，使用通用方法计算大小
                obj_size_bytes = calculate_dict_size(attr_value)
                attr_size_mb = obj_size_bytes / (1024**2)
                
                size_info = ""
                if hasattr(attr_value, '__len__'):
                    try:
                        size_info = f"[{len(attr_value)}]"
                    except:
                        pass
                        
                if attr_size_mb >= 0.01:  # 只显示大于0.01MB的
                    print(f"    * {attr_name}: {attr_type}{size_info} ({attr_size_mb:.2f} MB)")
                else:
                    print(f"    * {attr_name}: {attr_type}{size_info} ({attr_value})")
            
            total_size_mb += attr_size_mb
    
    print(f"  - 总大小: {total_size_mb:.2f} MB")
    
    return total_size_mb

def quick_ckpt_info(ckpt_path):
    """
    快速查看checkpoint基本信息
    """
    print(f"🔍 快速分析: {ckpt_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    
    # 如果是元组，解包
    if isinstance(checkpoint, tuple):
        (
            features_dc,          # 修复变量名
            features_rest,        # 修复变量名  
            opacity,              # 修复变量名
            xyz_0,                # 修复变量名
            rotation_0,           # 修复变量名
            scaling_0,            # 修复变量名
            xyz_mlp_state_dict,
            rot_mlp_state_dict,
            scale_mlp_state_dict,
            dg,
            base_xyz,
            # temp_flame_vertices
        ) = checkpoint
        
        ckpt_list = {

            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacity": opacity,
            "xyz_0": xyz_0,
            "rotation_0": rotation_0,
            "scaling_0": scaling_0,
            "xyz_mlp_state_dict": xyz_mlp_state_dict,
            "rot_mlp_state_dict": rot_mlp_state_dict,
            "scale_mlp_state_dict": scale_mlp_state_dict,
            "dg": dg,
            "base_xyz": base_xyz,
        }
    else:
        # 如果是字典，直接使用
        ckpt_list = checkpoint
    
    total_params = 0
    total_size_bytes = 0
    
    print(f"{'='*60}")
    print(f"{'键名':<25} {'大小(MB)':<15} {'参数数量':<15} {'类型'}")
    print(f"{'='*60}")
    
    for key, value in ckpt_list.items():
        if isinstance(value, torch.Tensor):
            size_bytes = value.numel() * value.element_size()
            size_mb = size_bytes / (1024**2)
            total_params += value.numel()
            total_size_bytes += size_bytes
            print(f"{key:<25} {size_mb:<15.2f} {value.numel():<15,} Tensor{list(value.shape)}")
            
        elif isinstance(value, dict):
            # 检查字典中是否包含tensor
            tensor_size = 0
            tensor_params = 0
            dict_total_size = 0
            
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    sub_size = sub_value.numel() * sub_value.element_size()
                    tensor_size += sub_size
                    tensor_params += sub_value.numel()
            
            # 计算整个字典的实际大小
            dict_total_size = calculate_dict_size(value)
            
            size_mb = dict_total_size / (1024**2)
            total_params += tensor_params
            total_size_bytes += dict_total_size
            
            tensor_count = sum(1 for v in value.values() if isinstance(v, torch.Tensor))
            print(f"{key:<25} {size_mb:<15.2f} {tensor_params:<15,} Dict({tensor_count} tensors)")
            
        else:
            # 其他类型的数据
            obj_size = calculate_dict_size(value)
            size_mb = obj_size / (1024**2)
            total_size_bytes += obj_size
            
            # 特殊处理DeformationGraph对象
            if hasattr(value, '__class__') and 'DeformationGraph' in value.__class__.__name__:
                dg_tensor_size = analyze_deformation_graph(value)
                # 更新显示的大小为实际tensor大小
                print(f"{key:<25} {dg_tensor_size:<15.2f} {0:<15} {type(value).__name__}")
            else:
                print(f"{key:<25} {size_mb:<15.2f} {0:<15} {type(value).__name__}")
                if size_mb < 0.01:  # 如果很小，显示具体值
                    print(f"  └─ 值: {value}")
    
    print(f"{'='*60}")
    print(f"\n📊 总结:")
    print(f" - 总参数量: {total_params:,}")
    print(f" - 总大小: {total_size_bytes / (1024**2):.2f} MB ({total_size_bytes / (1024**3):.2f} GB)")
    print(f" - 文件路径: {ckpt_path}")

if __name__ == "__main__":  # 修复语法错误
    # 示例用法
    path = "/home/momo/Desktop/data/ali/cc01data/output/02_frames_test/chkpnt142000_render.pth"
    quick_ckpt_info(path)