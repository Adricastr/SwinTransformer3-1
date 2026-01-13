import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange
import os
import cv2
from pathlib import Path
import time
import sys
import logging

def setup_single_log_file(log_file="log.txt"):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class PrintToLogger:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip():
            self.logger.info(message.strip())

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()


def maybe_flush(logger, last_flush_time, interval_sec=20 * 60):
    if time.time() - last_flush_time >= interval_sec:
        for handler in logger.handlers:
            handler.flush()
        logger.info("---- log flushed ----")
        return time.time()
    return last_flush_time


logger = setup_single_log_file("log.txt")
sys.stdout = PrintToLogger(logger)
sys.stderr = PrintToLogger(logger)

last_flush_time = time.time()

# Set environment variables to reduce CPU usage
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

# ==================== Hardware Profiles ====================
HARDWARE_PROFILES = {
    'lowest': {
        'name': 'Lowest',
        'batch_size': 1,
        'num_frames': 8,
        'img_size': (96, 96),
        'patch_size': (4, 8, 8),
        'embed_dim': 48,
        'depths': [1, 1, 2, 1],
        'num_heads': [2, 4, 8, 16],
        'window_size': (2, 6, 6),
    },
    'low': {
        'name': 'Low ~6M params ~2.5 hours',
        'batch_size': 2,
        'num_frames': 8,
        'img_size': (112, 112),
        'patch_size': (4, 7, 7),
        'embed_dim': 64,
        'depths': [1, 2, 3, 1],
        'num_heads': [2, 4, 8, 16],
        'window_size': (2, 7, 7),
    },
    'medium': {
        'name': 'Medium 15.5M params ~5 days ;(',
        'batch_size': 2,
        'num_frames': 12,
        'img_size': (160, 160),
        'patch_size': (4, 5, 5),
        'embed_dim': 80,
        'depths': [2, 2, 4, 2],
        'num_heads': [2, 5, 10, 20],
        'window_size': (3, 8, 8),
    },
    'high': {
        'name': 'High',
        'batch_size': 8,
        'num_frames': 16,
        'img_size': (224, 224),
        'patch_size': (4, 4, 4),
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': (4, 7, 7),
    },
    'highest': {
        'name': 'Highest (40GB+ GPU / A100)',
        'batch_size': 16,
        'num_frames': 32,
        'img_size': (224, 224),
        'patch_size': (2, 4, 4),
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'window_size': (8, 7, 7),
    }
}

# ==================== Dataset ====================
class HandTrackingVideoDataset(Dataset):
    def __init__(self, video_dir, num_frames=16, img_size=(224, 224), 
                 cache_mode='buffer', buffer_size=100, augment=False):
        """
        Args:
            video_dir: Directory containing id_X folders with video files
            num_frames: Number of frames to sample from each video
            img_size: Target image size (H, W)
            cache_mode: 'none' (no cache), 'buffer' (smart buffer), 'full' (all videos)
            buffer_size: Number of videos to keep in buffer for cache_mode='buffer'
            augment: Whether to apply data augmentation (for training)
        """
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.cache_mode = cache_mode
        self.buffer_size = buffer_size
        self.augment = augment
        self.samples = []
        self.cache = {}  
        self.cache_order = [] 
        
        # Load all video files from id folders
        self._load_video_paths()
        
        # Full prefetch if requested
        if self.cache_mode == 'full':
            print("Prefetching ALL videos into RAM (this will take a moment)...")
            self._prefetch_all_videos()
            print(f"Prefetched {len(self.cache)} videos into RAM")
        elif self.cache_mode == 'buffer':
            print(f"Using smart buffer cache (max {buffer_size} videos in memory)")
        else:
            print("No caching - loading videos on demand")
    
    def _load_video_paths(self):
        """Load video file paths from id folders"""
        video_dir_path = Path(self.video_dir)
        
        # Get all id folders (id_1, id_2, etc.)
        id_folders = sorted([f for f in video_dir_path.iterdir() 
                           if f.is_dir() and f.name.startswith('id_')])
        
        for id_folder in id_folders:
            # Extract activity ID from folder name (e.g., 'id_1' -> 0, 'id_2' -> 1)
            activity_id = int(id_folder.name.split('_')[1]) - 1 
            video_files = sorted(list(id_folder.glob('*.mp4')) + list(id_folder.glob('*.MP4')))
            
            for video_path in video_files:
                self.samples.append({
                    'video_path': str(video_path),
                    'label': activity_id,
                    'video_name': video_path.name
                })
        
        print(f"Loaded {len(self.samples)} video segments from {len(id_folders)} activity classes")
    
    def _prefetch_all_videos(self):
        for i, sample in enumerate(self.samples):
            if i % 100 == 0:
                print(f"  Prefetching: {i}/{len(self.samples)}")
            frames = self._load_video(sample['video_path'])
            if frames is not None:
                self.cache[sample['video_path']] = frames
    
    def _add_to_cache(self, video_path, frames):
        if self.cache_mode != 'buffer':
            return
        
        # If already in cache, update its position
        if video_path in self.cache:
            self.cache_order.remove(video_path)
            self.cache_order.append(video_path)
            return
        
        # Add new video to cache
        self.cache[video_path] = frames
        self.cache_order.append(video_path)
        
        # Evict oldest if buffer is full
        if len(self.cache) > self.buffer_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
    
    def _get_from_cache_or_load(self, video_path):
        if video_path in self.cache:
            # Move to end of LRU queue
            if self.cache_mode == 'buffer' and video_path in self.cache_order:
                self.cache_order.remove(video_path)
                self.cache_order.append(video_path)
            return self.cache[video_path]
        
        # Load from disk
        frames = self._load_video(video_path)
        
        # Add to cache if using buffer mode
        if self.cache_mode == 'buffer' and frames is not None:
            self._add_to_cache(video_path, frames)
        
        return frames
    
    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
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
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and resize in one operation
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
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
        
        # Load video from cache or disk with smart buffering
        frames = self._get_from_cache_or_load(sample['video_path'])
        
        if frames is None:
            # Return a dummy sample if video loading fails
            frames = np.zeros((self.num_frames, *self.img_size, 3), dtype=np.float32)
        
        # Data augmentation for training
        if self.augment:
            if np.random.rand() > 0.5:
                frames = np.flip(frames, axis=2).copy()  
            
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness_factor, 0, 255)
            
            # Random temporal crop (sample different starting point)
            if frames.shape[0] > self.num_frames:
                start_idx = np.random.randint(0, frames.shape[0] - self.num_frames)
                frames = frames[start_idx:start_idx + self.num_frames]
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        label = torch.LongTensor([sample['label']])
        
        return frames, label.squeeze()
    
    def get_cache_stats(self):
        return {
            'cached_videos': len(self.cache),
            'cache_mode': self.cache_mode,
            'buffer_size': self.buffer_size if self.cache_mode == 'buffer' else 'N/A'
        }

# ==================== Model Components ====================
class PatchEmbed3D(nn.Module):
    
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
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) 
        x = self.norm(x)
        return x, (T, H, W)

class WindowAttention3D(nn.Module):
    
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
        x, (T, H, W) = self.patch_embed(x) 
        x = self.pos_drop(x)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ModuleList):
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
    global last_flush_time
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    epoch_start_time = time.time()
    batch_times = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        batch_start_time = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Clear cache after each batch on GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            avg_batch_time = np.mean(batch_times[-100:])
            elapsed = time.time() - epoch_start_time
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f'  Batch {batch_idx+1}/{len(dataloader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | '
                      f'Time: {avg_batch_time:.3f}s | GPU: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB')
            else:
                print(f'  Batch {batch_idx+1}/{len(dataloader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | '
                      f'Time: {avg_batch_time:.3f}s')
    
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = np.mean(batch_times)
    last_flush_time = maybe_flush(logger, last_flush_time)
    return total_loss / len(dataloader), 100. * correct / total, epoch_time, avg_batch_time

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    val_start_time = time.time()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    val_time = time.time() - val_start_time
    return total_loss / len(dataloader), 100. * correct / total, val_time

# ==================== Main Training Script ====================
def main(resume=False, hardware_profile='high'):
    """
    Main training function
    Args:
        resume: Whether to resume from checkpoint
        hardware_profile: One of 'lowest', 'low', 'medium', 'high', 'highest'
    """
    # Get hardware configuration
    if hardware_profile not in HARDWARE_PROFILES:
        print(f"Unknown profile '{hardware_profile}'. Using 'low'.")
        hardware_profile = 'low'
    
    config = HARDWARE_PROFILES[hardware_profile]
    print(f"\n{'='*80}")
    print(f"Hardware Profile: {config['name']}")
    print(f"{'='*80}\n")
    
    #for epoch in range(start_epoch, num_epochs):
       # print(f"\nEpoch {epoch+1}/{num_epochs}")

        #train_loss, train_acc, train_time, avg_batch_time = train_epoch(...)
       # val_loss, val_acc, val_time = validate(...)

       #last_flush_time = maybe_flush(logger, last_flush_time)

   # print("Training finished")

    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Use config values
        batch_size = config['batch_size']
        num_frames = config['num_frames']
        img_size = config['img_size']
        num_workers = 2
        pin_memory = True
        cv2.setNumThreads(2)
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("No GPU detected, using CPU (this will be slow)")
        print("To use GPU, install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        batch_size = 1
        num_frames = config['num_frames']
        img_size = config['img_size']
        num_workers = 0
        pin_memory = False
    
    # Hyperparameters
    num_epochs = 100  
    learning_rate = 5e-5 
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Frames: {num_frames}, Image size: {img_size}")
    print(f"Embed dim: {config['embed_dim']}, Depths: {config['depths']}\n")
    
    # Create dataset with smart caching
    # Cache modes:
    #   'none': No caching - load from disk every time (lowest RAM, highest CPU)
    #   'buffer': Smart buffer - keep recent videos in RAM (balanced)
    #   'full': Load all videos into RAM (highest RAM, lowest CPU)
    cache_mode = 'buffer'  # Recommended default
    buffer_size = 200  # Keep 200 videos in memory
    
    # Create training dataset WITH augmentation
    full_dataset = HandTrackingVideoDataset(
        video_dir='splitVideos',
        num_frames=num_frames,
        img_size=img_size,
        cache_mode=cache_mode,
        buffer_size=buffer_size,
        augment=False 
    )
    
    # Split into train and validation (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create separate datasets with/without augmentation
    train_dataset_aug = HandTrackingVideoDataset(
        video_dir='splitVideos',
        num_frames=num_frames,
        img_size=img_size,
        cache_mode=cache_mode,
        buffer_size=buffer_size,
        augment=True  # Augmentation for training
    )
    
    val_dataset = HandTrackingVideoDataset(
        video_dir='splitVideos',
        num_frames=num_frames,
        img_size=img_size,
        cache_mode=cache_mode,
        buffer_size=buffer_size,
        augment=False  # No augmentation for validation
    )
    
    # Use subset to maintain train/val split
    train_dataset = torch.utils.data.Subset(train_dataset_aug, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Show cache stats
    stats = full_dataset.get_cache_stats()
    print(f"Cache mode: {stats['cache_mode']}, Buffer size: {stats['buffer_size']}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    # Create model - Based on hardware profile
    model = VideoSwinTransformer(
        img_size=img_size,
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=23,
        embed_dim=config['embed_dim'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
        drop_rate=0.1,
        attn_drop_rate=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss and optimizer - FIXED learning rate
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Better learning rate schedule
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_acc = 0
    if resume and os.path.exists('best_video_swin_model.pth'):
        print("Resuming from checkpoint...")
        checkpoint = torch.load('best_video_swin_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['val_acc']
        print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
    
    # Training loop
    total_train_time = 0
    total_val_time = 0
    
    for epoch in range(start_epoch, num_epochs):
        print(f'\n{"="*80}')
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'{"="*80}')
        
        train_loss, train_acc, train_time, avg_batch_time = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_time = validate(model, val_loader, criterion, device)
        
        total_train_time += train_time
        total_val_time += val_time
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Adjust learning rate based on validation accuracy
        old_lr = current_lr
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f'  Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}')
        
        print(f'\n{"─"*80}')
        print(f'EPOCH {epoch+1} SUMMARY:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        print(f'  Train Time: {train_time:.1f}s ({train_time/60:.1f}m) | Avg Batch: {avg_batch_time:.3f}s')
        print(f'  Val Time:   {val_time:.1f}s')
        print(f'  Total Time: {(train_time + val_time)/60:.1f}m')
        
        # Save best model with full configuration
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_config': {
                    'img_size': img_size,
                    'patch_size': config['patch_size'],
                    'in_channels': 3,
                    'num_classes': 23,
                    'embed_dim': config['embed_dim'],
                    'depths': config['depths'],
                    'num_heads': config['num_heads'],
                    'window_size': config['window_size'],
                    'drop_rate': 0.1,
                    'attn_drop_rate': 0.1
                },
                'num_frames': num_frames,
            }, 'best_video_swin_model.pth')
            print(f'  ✓ Best model saved with accuracy: {best_acc:.2f}%')
        
        print(f'{"─"*80}')
    
    avg_train_time_per_epoch = total_train_time / num_epochs
    avg_val_time_per_epoch = total_val_time / num_epochs
    
    print(f'\n{"="*80}')
    print(f'TRAINING COMPLETE!')
    print(f'{"="*80}')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'Total training time: {total_train_time/60:.1f}m ({total_train_time/3600:.2f}h)')
    print(f'Total validation time: {total_val_time/60:.1f}m')
    print(f'Average time per epoch: {(avg_train_time_per_epoch + avg_val_time_per_epoch)/60:.1f}m')
    print(f'Device used: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'{"="*80}')


# ==================== Inference Functions ====================
def load_trained_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model with saved configuration
    config = checkpoint['model_config']
    model = VideoSwinTransformer(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['num_frames'], config['img_size']


def predict_video(model, video_path, num_frames, img_size, device):
    """Predict activity class for a single video"""
    # Activity class names (adjust based on your labels)
    activity_names = {
        0: "Activity_1", 1: "Activity_2", 2: "Activity_3", 3: "Activity_4",
        4: "Activity_5", 5: "Activity_6", 6: "Activity_7", 7: "Activity_8",
        8: "Activity_9", 9: "Activity_10", 10: "Activity_11", 11: "Activity_12",
        12: "Activity_13", 13: "Activity_14", 14: "Activity_15", 15: "Activity_16",
        16: "Activity_17", 17: "Activity_18", 18: "Activity_19", 19: "Activity_20",
        20: "Activity_21", 21: "Activity_22", 22: "Activity_23"
    }
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None, None
    
    # Sample frames uniformly
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
        indices = np.concatenate([indices, np.repeat(indices[-1], num_frames - total_frames)])
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frames.append(frame)
    
    cap.release()
    
    # Prepare input
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
    frames = torch.FloatTensor(frames).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
    frames = frames.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(frames)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    
    return activity_names[predicted_class], confidence_score, probabilities[0].cpu().numpy()

def real_time_video_prediction(model, video_path, num_frames, img_size, device, window_stride=8):
    """
    Process video with sliding window for real-time intention decoding
    
    Args:
        model: Trained model
        video_path: Path to video file
        num_frames: Number of frames per prediction window
        img_size: Image size for model
        device: Device to run inference on
        window_stride: How many frames to skip between predictions
    """
    activity_names = {
        0: "Activity_1", 1: "Activity_2", 2: "Activity_3", 3: "Activity_4",
        4: "Activity_5", 5: "Activity_6", 6: "Activity_7", 7: "Activity_8",
        8: "Activity_9", 9: "Activity_10", 10: "Activity_11", 11: "Activity_12",
        12: "Activity_13", 13: "Activity_14", 14: "Activity_15", 15: "Activity_16",
        16: "Activity_17", 17: "Activity_18", 18: "Activity_19", 19: "Activity_20",
        20: "Activity_21", 21: "Activity_22", 22: "Activity_23"
    }
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_buffer = []
    frame_idx = 0
    predictions = []
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames_video}, FPS: {fps}")
    print(f"Prediction window: {num_frames} frames, stride: {window_stride} frames")
    print("-" * 80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add frame to buffer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, img_size)
        frame_buffer.append(frame_resized)
        
        # When buffer reaches required size, make prediction
        if len(frame_buffer) == num_frames:
            # Prepare input
            frames_tensor = np.stack(frame_buffer, axis=0).astype(np.float32) / 255.0
            frames_tensor = torch.FloatTensor(frames_tensor).permute(3, 0, 1, 2).unsqueeze(0)
            frames_tensor = frames_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                output = model(frames_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            time_seconds = frame_idx / fps
            
            result = {
                'frame_idx': frame_idx,
                'time': time_seconds,
                'activity': activity_names[predicted_class],
                'confidence': confidence_score
            }
            predictions.append(result)
            
            print(f"Frame {frame_idx:5d} | Time {time_seconds:6.2f}s | "
                  f"Activity: {activity_names[predicted_class]:15s} | "
                  f"Confidence: {confidence_score:.2%}")
            
            # Slide window by removing first 'window_stride' frames
            frame_buffer = frame_buffer[window_stride:]
        
        frame_idx += 1
    
    cap.release()
    print("-" * 80)
    print(f"Processing complete! Total predictions: {len(predictions)}")
    
    return predictions

# Example usage for inference
def inference_example():
    """Example of how to use the trained model for inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    model, num_frames, img_size = load_trained_model('best_video_swin_model.pth', device)
    print("Model loaded successfully!")
    
    # Single video prediction
    video_path = 'path/to/your/test/video.mp4'
    activity, confidence, probs = predict_video(model, video_path, num_frames, img_size, device)
    print(f"\nPredicted Activity: {activity}")
    print(f"Confidence: {confidence:.2%}")
    
    # Real-time sliding window prediction
    predictions = real_time_video_prediction(
        model, video_path, num_frames, img_size, device, window_stride=8
    )

if __name__ == '__main__':
    # ============================================
    # SELECT YOUR HARDWARE PROFILE HERE:
    # ============================================
    # Options: 'lowest', 'low', 'medium', 'high', 'highest'
    
    PROFILE = 'low'  # <--- CHANGE THIS FOR YOUR GPU
    
    # Run training
    # Set resume=True to continue from best_video_swin_model.pth
    main(resume=False, hardware_profile=PROFILE)
    
    # Uncomment to run inference after training
    # inference_example()
