import torch
import numpy as np

def calculate_dsc(pred, target, threshold=0.5):
    '''
    计算 Dice Similarity Coefficient (DSC)
    DSC = 2 * (A ∩ B) / (|A| + |B|)
    '''
    # 二值化
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    
    if union == 0:
        return 1.0 # 都是空的，认为完美匹配
    
    return (2. * intersection / union).item()

def calculate_iou(pred, target, threshold=0.5):
    '''
    计算 Jaccard Index (IoU)
    IoU = (A ∩ B) / (A ∪ B)
    '''
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    intersection = (pred_bin * target_bin).sum()
    total = (pred_bin + target_bin).sum()
    union = total - intersection
    
    if union == 0:
        return 1.0
        
    return (intersection / union).item()

def calculate_mse(pred, target):
    '''
    计算均方误差 Mean Squared Error
    '''
    criterion = torch.nn.MSELoss()
    return criterion(pred, target).item()

def generate(model, frag_input, device):
    '''
    推理辅助函数
    '''
    model.eval()
    with torch.no_grad():
        if frag_input.dim() == 3:
            frag_input = frag_input.unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64, 64)
        
        frag_input = frag_input.float().to(device)
        fake_output = model(frag_input)
        
        # 简单的后处理：因为输入部分在生成结果中应该是已知的
        # 所以我们可以强制让输出包含输入部分 (Input Consistency)
        # fake_refined = torch.max(fake_output, frag_input) 
        
    return fake_output, frag_input