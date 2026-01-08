import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==================== Global Config ====================
WINDOW_SIZE = (2, 4, 4)


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
        id_folders = sorted(
            [f for f in video_dir_path.iterdir()
             if f.is_dir() and f.name.startswith('id_')]
        )

        for id_folder in id_folders:
            # Extract activity ID from folder name (e.g., 'id_1' -> 0, 'id_2' -> 1)
            activity_id = int(id_folder.name.split('_')[1]) - 1  # 0-indexed

            # Get all video files in this folder
            video_files = sorted(
                list(id_folder.glob('*.mp4')) +
                list(id_folder.glob('*.MP4')) +
                list(id_folder.glob('*.avi')) +
                list(id_folder.glob('*.mkv'))
            )

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
            indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            indices = np.concatenate(
                [indices, np.repeat(indices[-1], self.num_frames - total_frames)]
            )

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

        # (T, H, W, C) -> (C, T, H, W)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # float32

        # Light augmentation: random horizontal flip (same for all frames)
        if np.random.rand() < 0.5:
            frames = torch.flip(frames, dims=[3])  # flip width dimension

        label = torch.tensor(sample['label'], dtype=torch.long)
        return frames, label


# ==================== Swin 3D Utility Functions ====================

def window_partition_3d(x, window_size):
    """
    Args:
        x: (B, T, H, W, C)
        window_size: (Wt, Wh, Ww)
    Returns:
        windows: (num_windows*B, Wt*Wh*Ww, C)
    """
    B, T, H, W, C = x.shape
    Wt, Wh, Ww = window_size

    x = x.view(
        B,
        T // Wt, Wt,
        H // Wh, Wh,
        W // Ww, Ww,
        C
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, Wt * Wh * Ww, C
    )
    return windows


def window_reverse_3d(windows, window_size, B, T, H, W, C):
    """
    Args:
        windows: (num_windows*B, Wt*Wh*Ww, C)
    Returns:
        x: (B, T, H, W, C)
    """
    Wt, Wh, Ww = window_size
    x = windows.view(
        B,
        T // Wt, H // Wh, W // Ww,
        Wt, Wh, Ww,
        C
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(
        B, T, H, W, C
    )
    return x


# ==================== Swin 3D Components ====================

class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video data"""

    def __init__(self, img_size=(224, 224), patch_size=(2, 4, 4),
                 in_channels=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = T'*H'*W'
        x = self.norm(x)
        return x, (T, H, W)


class WindowAttention3D(nn.Module):
    """Window-based Multi-head Self Attention for 3D windows"""

    def __init__(self, dim, num_heads, window_size=WINDOW_SIZE,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wt, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # x: (B*nW, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, heads, N, N)

        if mask is not None:
            # mask: (nW, N, N)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block for 3D data with shifted windows"""

    def __init__(self, dim, input_resolution, num_heads,
                 window_size=WINDOW_SIZE, shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (T, H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Clamp window/shift to not exceed resolution
        T, H, W = input_resolution
        self.window_size = (
            min(window_size[0], T),
            min(window_size[1], H),
            min(window_size[2], W),
        )
        self.shift_size = (
            0 if T <= self.window_size[0] else shift_size[0],
            0 if H <= self.window_size[1] else shift_size[1],
            0 if W <= self.window_size[2] else shift_size[2],
        )

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim,
            num_heads=num_heads,
            window_size=self.window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        # Attention mask for SW-MSA
        if any(s > 0 for s in self.shift_size):
            self.attn_mask = self.calculate_mask(
                input_resolution, self.window_size, self.shift_size
            )
        else:
            self.attn_mask = None

    @staticmethod
    def calculate_mask(input_resolution, window_size, shift_size):
        T, H, W = input_resolution
        Wt, Wh, Ww = window_size
        St, Sh, Sw = shift_size

        img_mask = torch.zeros((1, T, H, W, 1))  # (1, T, H, W, 1)
        cnt = 0

        t_slices = (
            (slice(0, -Wt), slice(-Wt, -St), slice(-St, None))
            if St > 0 else (slice(0, T),)
        )
        h_slices = (
            (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None))
            if Sh > 0 else (slice(0, H),)
        )
        w_slices = (
            (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None))
            if Sw > 0 else (slice(0, W),)
        )

        for t in t_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, t, h, w, :] = cnt
                    cnt += 1

        # Partition into windows
        mask_windows = window_partition_3d(img_mask, window_size)  # (nW, Wt*Wh*Ww, 1)
        mask_windows = mask_windows.squeeze(-1)  # (nW, N)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        T, H, W = self.input_resolution
        assert N == T * H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)

        # cyclic shift
        if any(s > 0 for s in self.shift_size):
            St, Sh, Sw = self.shift_size
            x = torch.roll(x, shifts=(-St, -Sh, -Sw), dims=(1, 2, 3))

        # partition windows
        x_windows = window_partition_3d(x, self.window_size)  # (nW*B, Nw, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(
            x_windows,
            mask=self.attn_mask.to(x.device) if self.attn_mask is not None else None
        )

        # merge windows
        x = window_reverse_3d(
            attn_windows, self.window_size, B, T, H, W, C
        )

        # reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            St, Sh, Sw = self.shift_size
            x = torch.roll(x, shifts=(St, Sh, Sw), dims=(1, 2, 3))

        x = x.view(B, N, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging3D(nn.Module):
    """Patch Merging for 3D Swin (merge spatial dims, keep time constant)"""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution  # (T, H, W)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        T, H, W = self.input_resolution
        assert N == T * H * W

        # we assume H and W are even; with img_size divisible by 4 and merging 3 times this holds
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for PatchMerging3D"

        x = x.view(B, T, H, W, C)

        x0 = x[:, :, 0::2, 0::2, :]  # (B, T, H/2, W/2, C)
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, T, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)  # (B, T*H/2*W/2, 4C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, N', 2C)

        new_resolution = (T, H // 2, W // 2)
        return x, new_resolution


class BasicLayer3D(nn.Module):
    """A basic Swin Transformer layer for one stage"""

    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size=WINDOW_SIZE, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., downsample=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = (
                0 if (i % 2 == 0) else window_size[0] // 2,
                0 if (i % 2 == 0) else window_size[1] // 2,
                0 if (i % 2 == 0) else window_size[2] // 2,
            )
            block = SwinTransformerBlock3D(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
            )
            self.blocks.append(block)

        self.downsample = PatchMerging3D(input_resolution, dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x, new_res = self.downsample(x)
            return x, new_res
        else:
            return x, self.input_resolution


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
                 window_size=WINDOW_SIZE,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.depths = depths
        self.num_heads_list = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Placeholder; real layers built lazily when resolution is known
        self.layers = nn.ModuleList([nn.Identity() for _ in range(self.num_layers)])
        self.layers_built = False

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _build_layers_if_needed(self, patch_resolution, device):
        """Build BasicLayer3D modules once we know (T',H',W') from patch embedding."""
        if self.layers_built:
            return

        T, H, W = patch_resolution
        dim = self.embed_dim
        current_res = (T, H, W)
        new_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            depth = self.depths[i_layer]
            heads = self.num_heads_list[i_layer]
            downsample = (i_layer < self.num_layers - 1)

            layer = BasicLayer3D(
                dim=dim,
                input_resolution=current_res,
                depth=depth,
                num_heads=heads,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                downsample=downsample
            )
            layer.to(device)
            new_layers.append(layer)

            if downsample:
                T, H, W = current_res
                current_res = (T, H // 2, W // 2)
                dim *= 2

        self.layers = new_layers
        self.num_features = dim
        self.layers_built = True
        # update norm and head to correct feature dim
        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, self.num_classes).to(device)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x, (T, H, W) = self.patch_embed(x)  # (B, N, embed_dim)
        x = self.pos_drop(x)

        device = x.device
        self._build_layers_if_needed((T, H, W), device)

        current_res = (T, H, W)

        # Forward through stages
        for i_layer, layer in enumerate(self.layers):
            x, current_res = layer(x)

        x = self.norm(x)  # (B, N, C)
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.avgpool(x)  # (B, C, 1)
        x = torch.flatten(x, 1)  # (B, C)
        x = self.head(x)  # (B, num_classes)
        return x


# ==================== Training ====================

def train_epoch(model, dataloader, criterion, optimizer, device,
                scaler=None, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    epoch_start_time = time.time()
    batch_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        batch_start_time = time.time()

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            avg_batch_time = float(np.mean(batch_times[-100:]))
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(
                    f'  Batch {batch_idx+1}/{len(dataloader)} | '
                    f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | '
                    f'Time: {avg_batch_time:.3f}s | '
                    f'GPU: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB'
                )
            else:
                print(
                    f'  Batch {batch_idx+1}/{len(dataloader)} | '
                    f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | '
                    f'Time: {avg_batch_time:.3f}s'
                )

    epoch_time = time.time() - epoch_start_time
    avg_batch_time = float(np.mean(batch_times)) if batch_times else 0.0

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


# ==================== Model Configs ====================

def get_model_config(model_size):
    """
    Returns dict of config for tiny / small / base
    """
    if model_size == 'tiny':
        return {
            'embed_dim': 48,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
        }
    elif model_size == 'small':
        return {
            'embed_dim': 64,
            'depths': [2, 2, 18, 2],
            'num_heads': [2, 4, 8, 16],
        }
    elif model_size == 'base':
        return {
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
        }
    else:
        raise ValueError(f"Unknown model_size: {model_size}")


# ==================== Main Training Script ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='Model size: tiny | small | base')
    args = parser.parse_args()

    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        batch_size = 2
        num_frames = 8
        img_size = (128, 128)  # chosen to stay divisible under patch+merging
        num_workers = 4
        pin_memory = True
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("No GPU detected, using CPU (this will be slow)")
        batch_size = 1
        num_frames = 8
        img_size = (112, 112)
        num_workers = 0
        pin_memory = False

    # Hyperparameters
    num_epochs = 50
    learning_rate = 3e-4
    weight_decay = 0.05

    print(f"Using device: {device}")
    print(f"Model size: {args.model_size}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Model config based on size
    base_cfg = get_model_config(args.model_size)

    model = VideoSwinTransformer(
        img_size=img_size,
        patch_size=(2, 4, 4),
        in_channels=3,
        num_classes=23,
        embed_dim=base_cfg['embed_dim'],
        depths=base_cfg['depths'],
        num_heads=base_cfg['num_heads'],
        window_size=WINDOW_SIZE,
        drop_rate=0.1,
        attn_drop_rate=0.1
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == 'cuda' else None

    best_acc = 0
    total_train_time = 0
    total_val_time = 0

    for epoch in range(num_epochs):
        print(f'\n{"=" * 80}')
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'{"=" * 80}')

        train_loss, train_acc, train_time, avg_batch_time = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        val_loss, val_acc, val_time = validate(model, val_loader, criterion, device)

        total_train_time += train_time
        total_val_time += val_time

        scheduler.step()

        print(f'\n{"─" * 80}')
        print(f'EPOCH {epoch + 1} SUMMARY:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        print(f'  Train Time: {train_time:.1f}s ({train_time / 60:.1f}m) | Avg Batch: {avg_batch_time:.3f}s')
        print(f'  Val Time:   {val_time:.1f}s')
        print(f'  Total Time: {(train_time + val_time) / 60:.1f}m')

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_name = f'best_video_swin_{args.model_size}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_config': {
                    'img_size': img_size,
                    'patch_size': (2, 4, 4),
                    'in_channels': 3,
                    'num_classes': 23,
                    'embed_dim': base_cfg['embed_dim'],
                    'depths': base_cfg['depths'],
                    'num_heads': base_cfg['num_heads'],
                    'window_size': WINDOW_SIZE,
                    'drop_rate': 0.1,
                    'attn_drop_rate': 0.1
                },
                'num_frames': num_frames,
                'model_size': args.model_size
            }, ckpt_name)
            print(f'  ✓ Best model saved as {ckpt_name} with accuracy: {best_acc:.2f}%')

        print(f'{"─" * 80}')

    avg_train_time_per_epoch = total_train_time / num_epochs
    avg_val_time_per_epoch = total_val_time / num_epochs

    print(f'\n{"=" * 80}')
    print(f'TRAINING COMPLETE!')
    print(f'{"=" * 80}')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'Total training time: {total_train_time / 60:.1f}m ({total_train_time / 3600:.2f}h)')
    print(f'Total validation time: {total_val_time / 60:.1f}m')
    print(f'Average time per epoch: {(avg_train_time_per_epoch + avg_val_time_per_epoch) / 60:.1f}m')
    print(f'Device used: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
