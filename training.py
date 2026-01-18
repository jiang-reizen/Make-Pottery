import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from utils.FragmentDataset import FragmentDataset, ToTensor3D
from utils.model import Generator, Discriminator
from utils import model_utils

# --- WGAN-GP Gradient Penalty ---
def compute_gradient_penalty(D, real_samples, fake_samples, condition_frag, labels, device):
    """
    Calculates the gradient penalty loss for WGAN GP.
    Interpolates between real and fake samples.
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(device)
    
    # Interpolation only on the Target Image part
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Pass to Discriminator (it handles concatenation inside or we do it here)
    # Our D.forward takes (img, condition, label)
    d_interpolates = D(interpolates, condition_frag, labels)
    
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    
    # Gradients w.r.t interpolates (Image)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def evaluate(generator, dataloader, device):
    generator.eval()
    dsc_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, desc="Evaluating"):
            frag_input, vox_gt, labels, _ = batch
            frag_input, vox_gt, labels = frag_input.to(device), vox_gt.to(device), labels.to(device)
            
            fake_output = generator(frag_input, labels)
            
            for j in range(fake_output.shape[0]):
                dsc = model_utils.calculate_dsc(fake_output[j], vox_gt[j])
                dsc_scores.append(dsc)
    
    avg_dsc = np.mean(dsc_scores) if dsc_scores else 0
    return avg_dsc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64) # Adjust for 2x4090 (try 128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lambda_pixel', type=float, default=100.0, help='L1 Loss Weight (k in paper)')
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint')
    args = parser.parse_args()

    # --- Device & DataParallel ---
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Data ---
    print("Initializing Datasets...")
    # Transform returns 4 channels
    train_set = FragmentDataset(args.data_dir, 'train', transform=ToTensor3D(binary=True))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
    test_set = FragmentDataset(args.data_dir, 'test', transform=ToTensor3D(binary=True))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # --- Model ---
    generator = Generator(in_channels=4).to(device)
    discriminator = Discriminator(in_channels=1).to(device) # input channels param is for generated img

    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # --- Optimizers ---
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    criterion_pixel = nn.L1Loss()

    # --- Resume Logic ---
    start_epoch = 0
    best_dsc = 0.0
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Helper for DataParallel keys
        def load_state(model, state_dict):
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

        try:
            load_state(generator, checkpoint['G_state_dict'])
            load_state(discriminator, checkpoint['D_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"Resume failed: {e}. Starting from scratch.")

    # --- Training Loop ---
    print(f"Start Training: Epoch {start_epoch} -> {start_epoch + args.epochs}")
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        generator.train()
        discriminator.train()
        
        running_d_loss = 0.0
        running_g_loss = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Ep {epoch+1}")
        
        for i, batch in pbar:
            frag_input, vox_gt, labels, _ = batch
            
            frag_input = frag_input.to(device) # (B, 4, 64, 64, 64)
            real_data = vox_gt.to(device)      # (B, 1, 64, 64, 64)
            labels = labels.to(device)         # (B)

            # 1. Train Discriminator
            optimizer_D.zero_grad()

            fake_data = generator(frag_input, labels) # G(x, y)

            # D(Real, Condition, Label)
            real_validity = discriminator(real_data, frag_input, labels)
            # D(Fake, Condition, Label)
            fake_validity = discriminator(fake_data.detach(), frag_input, labels)
            
            gp = compute_gradient_penalty(discriminator, real_data, fake_data.detach(), frag_input, labels, device)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gp
            d_loss.backward()
            optimizer_D.step()
            running_d_loss += d_loss.item()

            # 2. Train Generator
            if i % args.n_critic == 0:
                optimizer_G.zero_grad()
                
                # Forward G again
                gen_imgs = generator(frag_input, labels)
                
                # Fool Discriminator
                fake_validity = discriminator(gen_imgs, frag_input, labels)
                
                # L1 Completion Loss (Paper Eq. 1 & 3)
                pixel_loss = criterion_pixel(gen_imgs, real_data)
                
                g_loss = -torch.mean(fake_validity) + args.lambda_pixel * pixel_loss
                
                g_loss.backward()
                optimizer_G.step()
                running_g_loss += g_loss.item()

            pbar.set_postfix({'D': d_loss.item(), 'G': g_loss.item() if 'g_loss' in locals() else 0.0})

        # --- Save & Evaluate ---
        if (epoch + 1) % args.eval_interval == 0:
            avg_dsc = evaluate(generator, test_loader, device)
            print(f"Epoch {epoch+1} DSC: {avg_dsc:.4f}")

            # Get clean state dicts
            if isinstance(generator, nn.DataParallel):
                g_sd = generator.module.state_dict()
                d_sd = discriminator.module.state_dict()
            else:
                g_sd = generator.state_dict()
                d_sd = discriminator.state_dict()

            save_dict = {
                'epoch': epoch,
                'G_state_dict': g_sd,
                'D_state_dict': d_sd,
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }
            
            torch.save(save_dict, os.path.join(args.save_dir, 'latest_model.pth'))
            
            if avg_dsc > best_dsc:
                best_dsc = avg_dsc
                torch.save(save_dict, os.path.join(args.save_dir, 'best_model.pth'))
                print("--> New Best Model!")

if __name__ == "__main__":
    main()