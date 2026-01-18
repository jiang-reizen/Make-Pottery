import torch
from torch.utils.data import DataLoader
from utils.FragmentDataset import FragmentDataset, ToTensor3D
from utils import model_utils
import numpy as np
from tqdm import tqdm
from utils.model import Generator
import argparse
import os

def evaluate(generator, dataloader, device):
    generator.eval()
    
    dsc_scores = []
    iou_scores = []
    mse_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, leave=False, desc="Evaluating")):
            frag_input, vox_gt, labels, _ = batch
            
            frag_input = frag_input.to(device)
            vox_gt = vox_gt.to(device)
            labels = labels.to(device)
            
            # Generator now needs labels
            fake_output = generator(frag_input, labels)
            
            for j in range(fake_output.shape[0]):
                dsc = model_utils.calculate_dsc(fake_output[j], vox_gt[j])
                iou = model_utils.calculate_iou(fake_output[j], vox_gt[j])
                mse = model_utils.calculate_mse(fake_output[j], vox_gt[j])
                
                dsc_scores.append(dsc)
                iou_scores.append(iou)
                mse_scores.append(mse)
    
    avg_dsc = np.mean(dsc_scores) if dsc_scores else 0
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    avg_mse = np.mean(mse_scores) if mse_scores else 0
    
    print(f"[Result] DSC: {avg_dsc:.4f} | IoU: {avg_iou:.4f} | MSE: {avg_mse:.4f}")
    return avg_dsc, avg_iou, avg_mse

def test_pipeline(checkpoint_path, data_dir, device_name='cuda'):
    if not torch.cuda.is_available() and device_name == 'cuda':
        print("CUDA not available, switching to CPU.")
        device_name = 'cpu'
        
    device = torch.device(device_name)
    print(f"Testing on device: {device}")
    
    test_set = FragmentDataset(data_dir, 'test', transform=ToTensor3D(binary=True))
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize Conditional Generator
    generator = Generator(in_channels=4).to(device)
    
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle DataParallel loading if state_dict has 'module.' prefix
        state_dict = checkpoint['G_state_dict'] if 'G_state_dict' in checkpoint else checkpoint
        
        if isinstance(generator, torch.nn.DataParallel):
            generator.module.load_state_dict(state_dict)
        else:
            # If checkpoint has 'module.' but current model doesn't (single GPU test)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            generator.load_state_dict(new_state_dict)
            
        print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    evaluate(generator, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset root directory')
    args = parser.parse_args()
    
    test_pipeline(args.checkpoint, args.data_dir)