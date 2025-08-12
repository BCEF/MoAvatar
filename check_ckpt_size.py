import torch

def quick_ckpt_info(ckpt_path):
    """
    快速查看checkpoint基本信息
    """
    print(f"🔍 快速分析: {ckpt_path}")
    
    (
    active_sh_degree, 
    _features_dc, 
    _features_rest, 
    _opacity,
    _xyz_0, 
    _rotation_0,
    _scaling_0,  
    xyz_mlp_state_dict,
    rot_mlp_state_dict,
    scale_mlp_state_dict,
    dg_path,
    base_xyz,
    temp_flame_vertices  
    
    ) = torch.load(ckpt_path, map_location='cpu')
    ckpt_list={
    "active_sh_degree":active_sh_degree, 
    "_features_dc":_features_dc, 
    "_features_rest":_features_rest, 
    "_opacity":_opacity,
    "_xyz_0":_xyz_0, 
    "_rotation_0":_rotation_0,
    "_scaling_0":_scaling_0,  
    "xyz_mlp_state_dict":xyz_mlp_state_dict,
    "rot_mlp_state_dict":rot_mlp_state_dict,
    "scale_mlp_state_dict":scale_mlp_state_dict,
    "dg_path":dg_path,
    "base_xyz":base_xyz,
    "temp_flame_vertices":temp_flame_vertices
    
    }
    
    
    total_params = 0
    
    for key,value in ckpt_list.items():
        if isinstance(value, torch.Tensor):
            size_mb = value.numel() * value.element_size() / (1024**2)
            total_params += value.numel()
            print(key,size_mb)
        elif isinstance(value, dict) and any(isinstance(v, torch.Tensor) for v in value.values()):

            dict_size = 0
            tensor_count = 0
            
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    size_mb = sub_value.numel() * sub_value.element_size() / (1024**2)
                    # print(f"    - {sub_key}: {list(sub_value.shape)} -> {size_mb:.2f} MB")
                    dict_size += size_mb
                    total_params += sub_value.numel()
                    tensor_count += 1
            
            print(key,dict_size)
        else:
            dict_size=value.__sizeof__()
            print(key,dict_size,'字节')

    

    
    print(f"\n📊 总结:")
    print(f"  - 总参数量: {total_params:,}")

if __name__ == "__main__":
    # 示例用法
    path="/home/momo/Desktop/data/ali/cc01data/output/01_frames/chkpnt320000_render.pth"
    quick_ckpt_info(path)