import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Squeeze-and-Excitation Block (Paper Figure 5) ---
class SEBlock3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# --- Conditional Generator (Encoder-Decoder + Skip + SE) ---
class Generator(nn.Module):
    def __init__(self, in_channels=4, num_classes=11, latent_dim=256):
        super(Generator, self).__init__()
        
        # 类别 Embedding
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        # Helper to build Conv-BN-ReLU-SE blocks
        def enc_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
                SEBlock3D(out_c) # Paper Requirement
            )
            
        def dec_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose3d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                SEBlock3D(out_c) # Paper Requirement
            )

        # Encoder (64 -> 32 -> 16 -> 8 -> 4)
        self.enc1 = enc_block(in_channels, 32)
        self.enc2 = enc_block(32, 64)
        self.enc3 = enc_block(64, 128)
        self.enc4 = enc_block(128, 256) # Bottleneck feature size: 4x4x4

        # Decoder with Skip Connections
        # Input to dec1 will be: Bottleneck(256) + LabelEmb(expanded)
        # Note: Paper concatenates label at bottleneck.
        
        # Label embedding dimension adjustment
        # We project label to match feature map spatial size or channel size
        # Here we concat along channel: 256 + num_classes
        
        self.dec1 = dec_block(256 + num_classes, 128)
        self.dec2 = dec_block(128 + 128, 64) # Skip enc3 (128)
        self.dec3 = dec_block(64 + 64, 32)   # Skip enc2 (64)
        
        # Final Layer
        self.final = nn.Sequential(
            nn.ConvTranspose3d(32 + 32, 1, 4, 2, 1), # Skip enc1 (32)
            nn.Sigmoid() # Output [0, 1] Occupancy
        )

    def forward(self, x, labels):
        # Encoder
        e1 = self.enc1(x) # 32
        e2 = self.enc2(e1) # 16
        e3 = self.enc3(e2) # 8
        e4 = self.enc4(e3) # 4, 256 channels

        # Conditional Logic: Process Label
        # Embed label and reshape to (Batch, Class, 1, 1, 1) -> expand to (Batch, Class, 4, 4, 4)
        lbl_emb = self.label_emb(labels).view(x.size(0), -1, 1, 1, 1)
        lbl_emb = lbl_emb.expand(-1, -1, e4.size(2), e4.size(3), e4.size(4))
        
        # Concat at bottleneck
        bottleneck = torch.cat([e4, lbl_emb], dim=1) # 256 + 11

        # Decoder + Skips
        d1 = self.dec1(bottleneck) # -> 8, 128ch
        d1 = torch.cat([d1, e3], dim=1) 

        d2 = self.dec2(d1) # -> 16, 64ch
        d2 = torch.cat([d2, e2], dim=1)

        d3 = self.dec3(d2) # -> 32, 32ch
        d3 = torch.cat([d3, e1], dim=1)

        out = self.final(d3) # -> 64
        return out

# --- Conditional Discriminator (WGAN-GP + SE) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_classes=11):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        # Discriminator Input: Image (1ch) + Condition (4ch Input + Label)
        # Paper implies D receives x|y. 
        # But standard CGAN D receives (Real/Fake, Label).
        # We will follow standard CGAN: D([Image, InputFragment, LabelMap])
        # Image=1, InputFragment=4 (optional, to enforce consistency), Label=1(expanded)
        # To strictly follow "Conditional": D(x|y) usually means D(concat(x, y))
        
        # Input channels: 1 (Generated/Real) + 4 (Input Condition) + num_classes (Label)
        start_channels = 1 + 4 + num_classes

        def d_block(in_c, out_c, normalize=False):
            layers = [nn.Conv3d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(SEBlock3D(out_c)) # Paper Requirement
            return nn.Sequential(*layers)

        self.blocks = nn.Sequential(
            d_block(start_channels, 32, normalize=False),
            d_block(32, 64, normalize=True),
            d_block(64, 128, normalize=True),
            d_block(128, 256, normalize=True)
        )

        self.adv_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, 1) # WGAN outputs scalar score
        )

    def forward(self, img, condition_frag, labels):
        # 1. Expand Label
        lbl_emb = self.label_emb(labels).view(img.size(0), -1, 1, 1, 1)
        lbl_emb = lbl_emb.expand(-1, -1, img.size(2), img.size(3), img.size(4))
        
        # 2. Concat [TargetImage, InputFragment(Condition), LabelMap]
        # (B, 1+4+11, 64, 64, 64)
        d_in = torch.cat([img, condition_frag, lbl_emb], dim=1)
        
        features = self.blocks(d_in)
        validity = self.adv_layer(features)
        return validity