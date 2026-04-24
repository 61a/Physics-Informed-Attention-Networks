"""
FRB检测器 V2.4 (动态DM预测 + 物理感知联合优化版) - 修复版
---------------------------------------------------------
修复：
1. ✅ 兼容 positive/negative 索引结构
2. ✅ 修复日志编码问题
3. ✅ 添加详细的数据加载日志
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
import glob
import os
from pathlib import Path
import logging
import sys
from tqdm import tqdm
import warnings
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ============ 配置 ============

class Config:
    # 路径配置
    npy_data_root = "F:/FRB/npy_data"
    dataset_index_file = "F:/FRB/npy_data/dataset_index.pkl"
    save_dir = "./experiments"
    exp_name = "frb_v2.4_dynamic_dm_0115"
    
    # 数据维度
    n_time = 15360
    n_freq = 1024
    n_pol = 1
    
    # 动态DM配置
    n_dm_trials = 32
    dm_range = (0, 3000)
    use_dynamic_dm = True
    
    # 训练配置
    batch_size = 2
    accumulation_steps = 4
    num_workers = 4
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True
    
    epochs = 100
    patience = 15
    lr = 5e-5
    weight_decay = 1e-2
    dropout_rate = 0.3
    
    use_amp = True
    gradient_clip = 5.0
    
    # 损失权重
    focal_alpha = 0.25
    focal_gamma = 2.0
    label_smoothing = 0.1
    dm_loss_weight = 0.1
    
    # 增强参数
    freq_mask_param = 150
    time_mask_param = 600
    
    log_interval = 10
    metrics_update_interval = 50
    save_interval = 10

# ============ 日志系统 ============

class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if stream is None: 
            stream = sys.stderr
        if hasattr(stream, 'reconfigure'):
            try: 
                stream.reconfigure(encoding='utf-8')
            except Exception: 
                pass
        elif hasattr(stream, 'buffer'):
            import io
            self.stream = io.TextIOWrapper(
                stream.buffer, encoding='utf-8', 
                line_buffering=True, write_through=True
            )
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception: 
            self.handleError(record)

def setup_logger(config):
    exp_dir = Path(config.save_dir) / config.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    # 文件handler
    file_handler = logging.FileHandler(exp_dir / 'training.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 控制台handler
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, exp_dir

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_score, model, epoch_loss):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ============ 数据加载（修复版） ============

def load_npy_index(index_file, logger):
    """
    从索引文件加载数据，兼容多种格式
    """
    if not os.path.exists(index_file):
        logger.warning(f"Index file not found: {index_file}")
        return None, None
    
    try:
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        logger.info(f"Index file loaded successfully")
        logger.info(f"Index structure keys: {list(index_data.keys())}")
        
        # 提取文件列表的通用函数
        def extract_files(data):
            if isinstance(data, dict):
                if 'files' in data:
                    files = data['files']
                    if isinstance(files, list):
                        return files
                    elif isinstance(files, dict):
                        return [k for k in files.keys() if k.endswith('.npy')]
                # 直接从字典keys中提取.npy文件
                return [k for k in data.keys() if isinstance(k, str) and k.endswith('.npy')]
            elif isinstance(data, list):
                return data
            return []
        
        # 尝试不同的索引结构
        pos_files = []
        neg_files = []
        
        # 结构1: {positive: {...}, negative: {...}}
        if 'positive' in index_data and 'negative' in index_data:
            logger.info("Found 'positive'/'negative' structure")
            pos_files = extract_files(index_data['positive'])
            neg_files = extract_files(index_data['negative'])
            
            logger.info(f"Extracted files:")
            logger.info(f"  - Positive: {len(pos_files)} files")
            logger.info(f"  - Negative: {len(neg_files)} files")
            
            # 打印示例文件路径
            if pos_files:
                logger.info(f"  - Example positive file: {pos_files[0]}")
            if neg_files:
                logger.info(f"  - Example negative file: {neg_files[0]}")
            
            return pos_files, neg_files
        
        # 结构2: {train: {...}, val: {...}, test: {...}}
        elif 'train' in index_data:
            logger.info("Found split-based structure (train/val/test)")
            # 合并所有split的数据
            all_pos = []
            all_neg = []
            
            for split in ['train', 'val', 'test']:
                if split in index_data:
                    split_data = index_data[split]
                    if 'positive' in split_data:
                        all_pos.extend(extract_files(split_data['positive']))
                    if 'negative' in split_data:
                        all_neg.extend(extract_files(split_data['negative']))
            
            logger.info(f"Merged files from all splits:")
            logger.info(f"  - Positive: {len(all_pos)} files")
            logger.info(f"  - Negative: {len(all_neg)} files")
            
            return all_pos, all_neg
        
        else:
            logger.warning(f"Unknown index structure: {list(index_data.keys())}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error loading index file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

# ============ 数据集 ============

class NPYFRBDataset(Dataset):
    def __init__(self, file_list, labels, config, npy_data_root, augment=True, logger=None):
        self.file_list = file_list
        self.labels = labels
        self.config = config
        self.npy_data_root = Path(npy_data_root)
        self.augment = augment
        self.logger = logger
        
        # 验证文件路径
        self.valid_indices = []
        missing_count = 0
        
        for idx, f in enumerate(self.file_list):
            full_path = self.npy_data_root / f
            if full_path.exists():
                self.valid_indices.append(idx)
            else:
                missing_count += 1
                if missing_count <= 5:  # 只打印前5个缺失文件
                    if self.logger:
                        self.logger.warning(f"File not found: {full_path}")
        
        if self.logger:
            self.logger.info(f"Dataset created: {len(self.valid_indices)} valid / {len(self.file_list)} total files")
            if missing_count > 0:
                self.logger.warning(f"Missing files: {missing_count}")

    def __len__(self): 
        return len(self.valid_indices)
    
    def apply_augmentations(self, data):
        cloned = data.clone()
        _, n_time, n_freq = cloned.shape
        
        # Freq/Time Masking
        if np.random.rand() > 0.5:
            f = np.random.randint(0, self.config.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_freq - f))
            cloned[:, :, f0:f0+f] = 0
        
        if np.random.rand() > 0.5:
            t = np.random.randint(0, self.config.time_mask_param)
            t0 = np.random.randint(0, max(1, n_time - t))
            cloned[:, t0:t0+t, :] = 0
        
        # Random Shift
        if np.random.rand() > 0.5:
            shift_t = np.random.randint(-n_time // 4, n_time // 4)
            cloned = torch.roll(cloned, shifts=shift_t, dims=1)
        
        # Gaussian Noise
        if np.random.rand() > 0.5:
            noise = torch.randn_like(cloned) * 0.05
            cloned = cloned + noise

        return cloned

    def __getitem__(self, idx):
        # 使用valid_indices映射到实际文件
        actual_idx = self.valid_indices[idx]
        npy_path = self.npy_data_root / self.file_list[actual_idx]
        
        try:
            data = np.load(npy_path, mmap_mode='r') 
            data_copy = np.array(data, dtype=np.float32)
            
            if data_copy.ndim == 2:
                data_copy = data_copy[np.newaxis, :, :]

            mean = data_copy.mean()
            std = data_copy.std()
            if std > 1e-6:
                data_copy = (data_copy - mean) / std
            else:
                data_copy -= mean
            
            data_copy = np.ascontiguousarray(data_copy)
            data_tensor = torch.from_numpy(data_copy).permute(1, 0, 2).contiguous()
            
            if self.augment:
                if np.random.rand() > 0.5: 
                    data_tensor = torch.flip(data_tensor, dims=[1])
                data_tensor = self.apply_augmentations(data_tensor)
                
            label = torch.tensor(self.labels[actual_idx], dtype=torch.long)
            return data_tensor, label
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading {npy_path}: {e}")
            # 返回零填充数据
            return torch.zeros(self.config.n_pol, self.config.n_time, self.config.n_freq), \
                   torch.tensor(self.labels[actual_idx], dtype=torch.long)

# ============ 动态DM预测网络 ============

class DynamicDMPredictor(nn.Module):
    def __init__(self, n_freq, n_time, n_dm_trials, dm_range, dropout=0.2):
        super().__init__()
        self.n_dm_trials = n_dm_trials
        self.dm_min, self.dm_max = dm_range
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 7), stride=(1, 4), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(2, 4), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((8, 16)),
        )
        
        self.dm_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_dm_trials),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        dm_normalized = self.dm_regressor(features)
        dm_values = self.dm_min + dm_normalized * (self.dm_max - self.dm_min)
        dm_values_sorted, _ = torch.sort(dm_values, dim=1)
        return dm_values_sorted

# ============ 动态去色散层 ============

class DynamicDedispersionLayer(nn.Module):
    def __init__(self, n_freq, n_time):
        super().__init__()
        self.n_freq = n_freq
        self.n_time = n_time
        
        freq_indices = torch.linspace(0, 1, n_freq)
        freq_ghz = 1.0 + freq_indices * 0.5
        dispersion_factor = (1.0 / freq_ghz**2) - (1.0 / freq_ghz.max()**2)
        dispersion_factor = dispersion_factor / (dispersion_factor.max() + 1e-8)
        dispersion_factor = dispersion_factor * (n_time * 0.2)
        
        self.register_buffer('dispersion_factor', dispersion_factor)
        
    def forward(self, x, dm_values):
        batch_size, n_pol, n_time, n_freq = x.shape
        n_dm_trials = dm_values.size(1)
        
        delays = dm_values.unsqueeze(-1) * self.dispersion_factor.unsqueeze(0).unsqueeze(0)
        dedispersed_list = []
        
        for dm_idx in range(n_dm_trials):
            current_delays = delays[:, dm_idx, :]
            batch_dedispersed = []
            
            for b in range(batch_size):
                sample_data = x[b:b+1]
                sample_delays = current_delays[b]
                shifted_data = sample_data.clone()
                chunk_size = 128
                
                for freq_start in range(0, n_freq, chunk_size):
                    freq_end = min(freq_start + chunk_size, n_freq)
                    avg_delay = int(sample_delays[freq_start:freq_end].mean().item())
                    
                    if avg_delay > 0:
                        shifted_data[:, :, :, freq_start:freq_end] = torch.roll(
                            shifted_data[:, :, :, freq_start:freq_end],
                            shifts=-avg_delay,
                            dims=2
                        )
                
                batch_dedispersed.append(shifted_data)
            
            dedispersed_list.append(torch.cat(batch_dedispersed, dim=0))
        
        dedispersed_data = torch.stack(dedispersed_list, dim=1)
        return dedispersed_data, delays

# ============ DM注意力模块 ============

class DMAttentionModule(nn.Module):
    def __init__(self, n_dm_trials, n_time, n_freq, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
        )
        self.attention_net = nn.Sequential(
            nn.Linear(32 * n_dm_trials, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_dm_trials),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, dedispersed_data):
        batch_size, n_dm_trials, n_pol, n_time, n_freq = dedispersed_data.shape
        x_flat = dedispersed_data.view(batch_size * n_dm_trials, n_pol, n_time, n_freq)
        x_pooled = self.pool(x_flat)
        features = self.feature_extractor(x_pooled).view(batch_size, n_dm_trials, 32)
        features_flat = features.flatten(1)
        attention_weights = self.attention_net(features_flat)
        attention_expanded = attention_weights.view(batch_size, n_dm_trials, 1, 1, 1)
        weighted_data = (dedispersed_data * attention_expanded).sum(dim=1)
        return weighted_data, attention_weights

# ============ Conformer组件 ============

class Swish(nn.Module):
    def forward(self, x): 
        return x * torch.sigmoid(x)

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x): 
        return x.transpose(self.dim1, self.dim2)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), Swish(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x): 
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3), qkv)
        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.to_out(out)

class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose(1, 2),
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size, padding=(kernel_size - 1) // 2, groups=dim),
            nn.BatchNorm1d(dim),
            Swish(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout),
            Transpose(1, 2)
        )
    def forward(self, x): 
        return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_dim=512, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, ffn_dim, dropout)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, conv_kernel_size, dropout)
        self.ff2 = FeedForward(dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.conv(self.norm3(x))
        x = x + 0.5 * self.ff2(self.norm4(x))
        return self.final_norm(x)

# ============ 主模型 ============

class AdvancedFRBDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.use_dynamic_dm:
            self.dm_predictor = DynamicDMPredictor(
                n_freq=config.n_freq,
                n_time=config.n_time,
                n_dm_trials=config.n_dm_trials,
                dm_range=config.dm_range,
                dropout=config.dropout_rate
            )
            self.dedispersion = DynamicDedispersionLayer(
                n_freq=config.n_freq,
                n_time=config.n_time
            )
        
        self.dm_attention = DMAttentionModule(
            n_dm_trials=config.n_dm_trials,
            n_time=config.n_time,
            n_freq=config.n_freq,
            dropout=config.dropout_rate
        )
        
        self.freq_compress = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 4), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 4), padding=1),
            nn.BatchNorm2d(256), nn.ReLU()
        )
        
        dim_model = 256
        self.projection = nn.Linear(256 * 16, dim_model)
        
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(dim=dim_model, num_heads=4, ffn_dim=1024, conv_kernel_size=15, dropout=config.dropout_rate)
            for _ in range(3)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(dim_model + config.n_dm_trials, 128),
            nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(128, 2)
        )
        
    def forward_physics(self, x):
        if self.config.use_dynamic_dm:
            dm_values = self.dm_predictor(x)
            dedispersed, delays = self.dedispersion(x, dm_values)
        
        dm_corrected, dm_attn = self.dm_attention(dedispersed)
        return dm_corrected, dm_attn, delays, dm_values

    def forward(self, x, return_interpretations=False):
        dm_corrected, dm_attn, delays, dm_values = self.forward_physics(x)

        feat = self.freq_compress(dm_corrected)
        B, C, T, F = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(B, T, C * F)
        feat = self.projection(feat)
        
        for layer in self.conformer_layers:
            feat = layer(feat)
            
        global_feat = feat.mean(dim=1)
        combined = torch.cat([global_feat, dm_attn], dim=1)
        logits = self.classifier(combined)
        
        if return_interpretations:
            return logits, {
                'dm_attention': dm_attn,
                'dm_values': dm_values,
                'delays': delays,
                'dedispersed_data': dm_corrected
            }
        return logits

# ============ 损失函数 ============

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean': 
            return torch.mean(focal_loss)
        elif self.reduction == 'sum': 
            return torch.sum(focal_loss)
        else: 
            return focal_loss

class MetricsCalculator:
    def __init__(self): 
        self.reset()
    
    def reset(self):
        self.all_preds, self.all_labels, self.all_probs = [], [], []
        self.correct = self.total = self.tp = self.tn = self.fp = self.fn = 0
    
    def update(self, preds, labels, probs=None):
        preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())
        if probs is not None:
            probs = probs.cpu().detach().numpy() if torch.is_tensor(probs) else probs
            self.all_probs.extend(probs.tolist())
        for p, l in zip(preds, labels):
            if p == 1 and l == 1: self.tp += 1
            elif p == 0 and l == 0: self.tn += 1
            elif p == 1 and l == 0: self.fp += 1
            elif p == 0 and l == 1: self.fn += 1
        self.correct += (preds == labels).sum()
        self.total += len(labels)
    
    def get_metrics(self):
        if self.total == 0: 
            return {}
        accuracy = self.correct / self.total
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
    
    def format_metrics(self, metrics, prefix=""):
        return f"{prefix}Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | Rec: {metrics['recall']:.4f}"

# ============ 训练器 ============

class Trainer:
    def __init__(self, config, logger, exp_dir):
        self.config = config
        self.logger = logger
        self.exp_dir = exp_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = AdvancedFRBDetector(config).to(self.device)
        self.logger.info(f"Initialized AdvancedFRBDetector (Dynamic DM: {config.use_dynamic_dm}) on {self.device}")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )
        self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
        self.scaler = GradScaler() if config.use_amp else None
        
        self.early_stopping = EarlyStopping(
            patience=config.patience, verbose=True, 
            path=str(exp_dir / 'best_model.pth'), trace_func=logger.info
        )
        
        self.current_epoch = 0
        self.train_losses, self.val_losses = [], []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        metrics_calc = MetricsCalculator()
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.epochs}')
        
        self.optimizer.zero_grad()
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            if self.config.use_amp:
                with autocast():
                    logits = self.model(data)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.config.accumulation_steps
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(data)
                loss = self.criterion(logits, labels)
                loss = loss / self.config.accumulation_steps
                loss.backward()
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.accumulation_steps
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            metrics_calc.update(predicted, labels, probs[:, 1])
            
            if (batch_idx + 1) % self.config.metrics_update_interval == 0:
                m = metrics_calc.get_metrics()
                pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}', 'f1': f'{m["f1_score"]:.3f}'})
        
        return total_loss / len(train_loader), metrics_calc.get_metrics()

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        metrics_calc = MetricsCalculator()
        for data, labels in tqdm(val_loader, desc='Validating'):
            data, labels = data.to(self.device), labels.to(self.device)
            with autocast(enabled=self.config.use_amp):
                logits = self.model(data)
                loss = self.criterion(logits, labels)
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            metrics_calc.update(predicted, labels, probs[:, 1])
        return total_loss / len(val_loader), metrics_calc.get_metrics(), metrics_calc.format_metrics(metrics_calc.get_metrics(), prefix="  ")

    def train(self, train_loader, val_loader):
        self.logger.info("Starting Training with Dynamic DM Prediction")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics, val_output = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step()
            
            self.logger.info(f'Epoch [{epoch+1}/{self.config.epochs}]')
            self.logger.info(f'Train Loss: {train_loss:.4f} | F1: {train_metrics["f1_score"]:.4f}')
            self.logger.info(f'Val Output:\n{val_output}')
            
            if (epoch + 1) % self.config.save_interval == 0:
                ckpt_path = self.exp_dir / f'model_epoch_{epoch+1}.pth'
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"Saved checkpoint to {ckpt_path}")

            self.early_stopping(val_metrics['f1_score'], self.model, val_loss)
            
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info("Starting Final Testing...")
        best_model_path = self.exp_dir / 'best_model.pth'
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.info(f"Loaded best model from {best_model_path}")

        self.model.eval()
        metrics_calc = MetricsCalculator()
        
        for data, labels in tqdm(test_loader, desc='Testing'):
            data, labels = data.to(self.device), labels.to(self.device)
            with autocast(enabled=self.config.use_amp):
                logits = self.model(data)
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            metrics_calc.update(predicted, labels, probs[:, 1])
        
        y_true = metrics_calc.all_labels
        y_pred = metrics_calc.all_preds
        y_prob = metrics_calc.all_probs
        
        self.logger.info("\n" + "="*50)
        self.logger.info("FINAL TEST RESULTS")
        self.logger.info("="*50)
        
        report = classification_report(y_true, y_pred, target_names=['Noise', 'FRB'], digits=4)
        self.logger.info(f"\n{report}")
        
        cm = confusion_matrix(y_true, y_pred)
        self.logger.info(f"\nConfusion Matrix:\n{cm}")
        
        try:
            auc = roc_auc_score(y_true, y_prob)
            self.logger.info(f"\nROC AUC: {auc:.4f}")
        except:
            self.logger.info("\nROC AUC: N/A")
            
        return metrics_calc.get_metrics()

# ============ 主程序 ============

def main():
    config = Config()
    logger, exp_dir = setup_logger(config)
    
    logger.info("="*60)
    logger.info("FRB Detector V2.4 - Dynamic DM Prediction (Fixed)")
    logger.info("="*60)
    
    # 1. 加载数据索引
    logger.info("\nLoading dataset index...")
    pos_files, neg_files = load_npy_index(config.dataset_index_file, logger)
    
    if pos_files is None or neg_files is None:
        logger.error("Failed to load dataset index. Trying direct file search...")
        # 备选方案：直接搜索文件
        pos_files = glob.glob(os.path.join(config.npy_data_root, 'positive', '*.npy'))
        neg_files = glob.glob(os.path.join(config.npy_data_root, 'negative', '*.npy'))
        logger.info(f"Found {len(pos_files)} positive and {len(neg_files)} negative files via direct search")
    
    if len(pos_files) == 0 or len(neg_files) == 0:
        logger.error("No data files found! Please check paths.")
        return
    
    all_files = pos_files + neg_files
    labels = [1]*len(pos_files) + [0]*len(neg_files)
    
    logger.info(f"\nDataset Summary:")
    logger.info(f"  - Positive (FRB): {len(pos_files)}")
    logger.info(f"  - Negative (Noise): {len(neg_files)}")
    logger.info(f"  - Total: {len(all_files)}")
    
    # 2. 数据划分
    logger.info("\nSplitting dataset...")
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, labels, test_size=0.15, random_state=42, stratify=labels
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=0.176, random_state=42, stratify=train_val_labels
    )
    
    logger.info(f"Data Split:")
    logger.info(f"  - Train: {len(train_files)} (FRB: {sum(train_labels)})")
    logger.info(f"  - Val: {len(val_files)} (FRB: {sum(val_labels)})")
    logger.info(f"  - Test: {len(test_files)} (FRB: {sum(test_labels)})")
    
    # 3. DataLoader
    logger.info("\nCreating datasets...")
    train_dataset = NPYFRBDataset(train_files, train_labels, config, config.npy_data_root, augment=True, logger=logger)
    val_dataset = NPYFRBDataset(val_files, val_labels, config, config.npy_data_root, augment=False, logger=logger)
    test_dataset = NPYFRBDataset(test_files, test_labels, config, config.npy_data_root, augment=False, logger=logger)
    
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty! Check file paths.")
        return
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor, persistent_workers=config.persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    logger.info(f"\nDataLoader created successfully")
    logger.info(f"  - Train batches: {len(train_loader)}")
    logger.info(f"  - Val batches: {len(val_loader)}")
    logger.info(f"  - Test batches: {len(test_loader)}")
    
    # 4. 训练与测试
    trainer = Trainer(config, logger, exp_dir)
    trainer.train(train_loader, val_loader)
    trainer.test(test_loader)

if __name__ == "__main__":
    main()