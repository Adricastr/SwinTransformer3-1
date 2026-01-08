import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange
import os
import cv2
from pathlib import Path

# ==================== Dataset ====================
class HandTrackingVideoDataset(Dataset):
    """Dataset for hand tracking video sequences"""
    
    def __init__(self, video_dir, num_frames=16, img_size=(224, 224)):
        """
        Args:
            video_dir: Directory containing id_X folders with video files
            num_frames: Number of frames to sample from each video
            img_size: Target image size (H, W)
        """
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.samples = []
        
        # Load all video files from id folders
        self._load_video_paths()
    
    def _load_video_paths(self):
        """Load video file paths from id folders"""
        video_dir_path = Path(self.video_dir)
        
        # Get all id folders (id_1, id_2, etc.)
        id_folders = sorted([f for f in video_dir_path.iterdir() 
                           if f.is_dir() and f.name.startswith('id_')])
        
        for id_folder in id_folders:
            # Extract activity ID from folder name (e.g., 'id_1' -> 0, 'id_2' -> 1)
            activity_id = int(id_folder.name.split('_')[1]) - 1  # 0-indexed
            
            # Get all video files in this folder
            video_files = sorted(list(id_folder.glob('*.mp4')) + list(id_folder.glob('*.MP4')))
            
            for video_path in video_files:
                self.samples.append({
                    'video_path': str(video_path),
                    'label': activity_id,
                    'video_name': video_path.name
                })
        
        print(f"Loaded {len(self.samples)} video segments from {len(id_folders)} activity classes")
    
    def _load_video(self, video_path):
        """Load video and sample frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frame indices uniformly
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # If video has fewer frames, repeat last frame
            indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            indices = np.concatenate([indices, np.repeat(indices[-1], self.num_frames - total_frames)])
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, self.img_size)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Stack frames: (T, H, W, C)
        frames = np.stack(frames, axis=0)
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video
        frames = self._load_video(sample['video_path'])
        
        if frames is None:
            # Return a dummy sample if video loading fails
            frames = np.zeros((self.num_frames, *self.img_size, 3), dtype=np.float32)
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        label = torch.LongTensor([sample['label']])
        
        return frames, label.squeeze()


# ==================== Model Components ====================
class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video data"""
    
    def __init__(self, img_size=(224, 224), patch_size=(2, 4, 4), in_channels=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', C)
        x = self.norm(x)
        return x, (T, H, W)


class WindowAttention3D(nn.Module):
    """Window-based Multi-head Self Attention for 3D windows"""
    
    def __init__(self, dim, num_heads, window_size=(8, 7, 7), qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block for 3D data"""
    
    def __init__(self, dim, num_heads, window_size=(8, 7, 7),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class VideoSwinTransformer(nn.Module):
    """Video Swin Transformer for activity recognition"""
    
    def __init__(self, 
                 img_size=(224, 224),
                 patch_size=(2, 4, 4),
                 in_channels=3,
                 num_classes=23,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            layer_blocks = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                ) for _ in range(depths[i_layer])
            ])
            self.layers.append(layer_blocks)
            
            # Patch merging (downsample) except last layer
            if i_layer < self.num_layers - 1:
                downsample = nn.Linear(dim, dim * 2)
                self.layers.append(downsample)
        
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x, (T, H, W) = self.patch_embed(x)  # (B, N, embed_dim)
        x = self.pos_drop(x)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ModuleList):
                # Transformer blocks
                for blk in layer:
                    x = blk(x)
            else:
                # Downsample
                x = layer(x)
        
        x = self.norm(x)  # (B, N, C)
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.avgpool(x)  # (B, C, 1)
        x = torch.flatten(x, 1)  # (B, C)
        x = self.head(x)  # (B, num_classes)
        
        return x


# ==================== Training ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


# ==================== Main Training Script ====================
def main():
    # Hyperparameters
    batch_size = 2  # Very small batch size due to memory constraints
    num_epochs = 50
    learning_rate = 1e-4
    num_frames = 8  # Reduced from 16
    img_size = (112, 112)  # Reduced from 224x224 to save memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = HandTrackingVideoDataset(
        video_dir='Project 3-1/splitVideos',
        num_frames=num_frames,
        img_size=img_size
    )
    
    # Split into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)  # Set to 0 for CPU
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)  # Set to 0 for CPU
    
    # Create model - Smaller configuration for CPU/low memory
    model = VideoSwinTransformer(
        img_size=img_size,
        patch_size=(2, 4, 4),  # This creates fewer patches
        in_channels=3,
        num_classes=23,  # Based on your id_1 to id_23 folders
        embed_dim=48,  # Reduced from 96
        depths=[1, 1, 3, 1],  # Reduced from [2, 2, 6, 2]
        num_heads=[2, 4, 8, 16],  # Reduced from [3, 6, 12, 24]
        window_size=(4, 7, 7),  # Reduced temporal window from 8 to 4
        drop_rate=0.1,
        attn_drop_rate=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_acc = 0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_video_swin_model.pth')
            print(f'✓ Best model saved with accuracy: {best_acc:.2f}%')
        
        print('-' * 80)
    
    print(f'\nTraining complete! Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()