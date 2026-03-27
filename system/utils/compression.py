import torch
import numpy as np


def compress_model_for_communication(model, compression_ratio=0.5):
    """
    使用 TOPK 压缩模型参数
    
    参数:
        model: 要压缩的模型
        compression_ratio: 压缩比率（保留前 compression_ratio 比例的参数）
    
    返回:
        compressed_data: 压缩后的模型数据字典
    """
    all_params = []
    all_shapes = []
    param_names = []
    
    for name, param in model.named_parameters():
        all_params.append(param.data.view(-1))
        all_shapes.append(param.data.shape)
        param_names.append(name)
    
    all_params = torch.cat(all_params, dim=0)
    total_params = all_params.numel()
    k = int(total_params * compression_ratio)
    
    abs_params = torch.abs(all_params)
    _, topk_indices = torch.topk(abs_params, k)
    
    mask = torch.zeros_like(all_params, dtype=torch.bool)
    mask[topk_indices] = True
    
    compressed_values = all_params[mask]
    compressed_indices = torch.nonzero(mask).squeeze(-1)
    
    compressed_data = {
        'values': compressed_values,
        'indices': compressed_indices,
        'total_params': total_params,
        'shapes': all_shapes,
        'param_names': param_names,
        'compression_ratio': compression_ratio
    }
    
    return compressed_data


def decompress_model_for_communication(compressed_data, model_template):
    """
    解压缩模型参数
    
    参数:
        compressed_data: 压缩后的模型数据字典
        model_template: 模型模板（用于获取参数形状）
    
    返回:
        model: 解压缩后的模型
    """
    model = model_template
    
    values = compressed_data['values']
    indices = compressed_data['indices']
    total_params = compressed_data['total_params']
    shapes = compressed_data['shapes']
    param_names = compressed_data['param_names']
    
    all_params = torch.zeros(total_params, dtype=values.dtype, device=values.device)
    all_params[indices] = values
    
    offset = 0
    state_dict = {}
    for name, shape in zip(param_names, shapes):
        numel = shape.numel()
        param_data = all_params[offset:offset + numel].view(shape)
        state_dict[name] = param_data
        offset += numel
    
    model.load_state_dict(state_dict)
    
    return model
