"""
FRB检测器 V3.1 - 真实DM监督学习增强版
---------------------------------------------------------
新增诊断功能：
1. ✅ DM使用统计和实时监控
2. ✅ 自适应DM损失权重
3. ✅ 详细的训练日志
4. ✅ DM标签覆盖率分析
5. ✅ 改进的Early Stopping（50 epoch后启动）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
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
    
    # DM标签文件路径
    dm_label_dirs = [
        "F:/FRB/downloads/20121102",
        "F:/FRB/downloads/20180301",
        "F:/FRB/downloads/20201124"
    ]
    
    save_dir = "./experiments"
    exp_name = "frb_v3.1_supervised_dm_enhanced"
    
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
    min_epochs_before_stop = 50
    lr = 5e-5
    weight_decay = 1e-2
    dropout_rate = 0.3
    
    use_amp = True
    gradient_clip = 5.0
    
    # 损失权重（自适应调整）
    focal_alpha = 0.25
    focal_gamma = 2.0
    label_smoothing = 0.1
    dm_regression_weight = 1.0       # 提高基础权重
    dm_ranking_weight = 0.5          # 提高排序权重
    use_adaptive_dm_weight = True    # 启用自适应权重
    
    # 增强参数
    freq_mask_param = 150
    time_mask_param = 600
    
    log_interval = 10
    metrics_update_interval = 50
    save_interval = 10
    dm_stats_interval = 50  # DM统计报告间隔

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
    
    file_handler = logging.FileHandler(exp_dir / 'training.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, exp_dir

class EarlyStopping:
    """Early Stopping with minimum epoch threshold"""
    def __init__(self, patience=7, min_delta=0, min_epochs=50, verbose=False, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.current_epoch = 0

    def __call__(self, val_score, model, epoch_loss, current_epoch=None):
        if current_epoch is not None:
            self.current_epoch = current_epoch
        
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        else:
            if self.current_epoch >= self.min_epochs:
                self.counter += 1
                if self.verbose:
                    self.trace_func(
                        f'EarlyStopping counter: {self.counter} out of {self.patience} '
                        f'(active after epoch {self.min_epochs})'
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                if self.verbose and self.counter == 0:
                    self.trace_func(
                        f'Early stopping will activate after epoch {self.min_epochs} '
                        f'(current: {self.current_epoch})'
                    )

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ============ DM标签加载 ============

def load_dm_labels(dm_label_dirs, logger):
    """
    从CSV文件加载DM标签
    Returns: dict {npy_filename: dm_value}
    """
    dm_dict = {}
    
    for label_dir in dm_label_dirs:
        label_dir = Path(label_dir)
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue
        
        csv_files = list(label_dir.glob("*.csv")) + list(label_dir.glob("*.txt"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, sep=None, engine='python')
                
                required_cols = ['file', 'dms']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"CSV file {csv_file} missing required columns")
                    continue
                
                for _, row in df.iterrows():
                    fits_file = row['file']
                    dm_value = float(row['dms'])
                    
                    if isinstance(fits_file, str):
                        npy_file = fits_file.replace('.fits', '.npy').replace('.fil', '.npy')
                        dm_dict[npy_file] = dm_value
                
                logger.info(f"Loaded {len(df)} DM labels from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
                continue
    
    logger.info(f"Total DM labels loaded: {len(dm_dict)}")
    if dm_dict:
        dm_values = list(dm_dict.values())
        logger.info(f"DM range: {min(dm_values):.1f} - {max(dm_values):.1f} pc/cm³")
        logger.info(f"DM mean: {np.mean(dm_values):.1f} ± {np.std(dm_values):.1f} pc/cm³")
    
    return dm_dict

# ============ 数据加载 ============

def load_npy_index(index_file, logger):
    if not os.path.exists(index_file):
        logger.warning(f"Index file not found: {index_file}")
        return None, None
    
    try:
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        logger.info(f"Index file loaded successfully")
        logger.info(f"Index structure keys: {list(index_data.keys())}")
        
        def extract_files(data):
            if isinstance(data, dict):
                if 'files' in data:
                    files = data['files']
                    if isinstance(files, list):
                        return files
                    elif isinstance(files, dict):
                        return [k for k in files.keys() if k.endswith('.npy')]
                return [k for k in data.keys() if isinstance(k, str) and k.endswith('.npy')]
            elif isinstance(data, list):
                return data
            return []
        
        pos_files = []
        neg_files = []
        
        if 'positive' in index_data and 'negative' in index_data:
            logger.info("Found 'positive'/'negative' structure")
            pos_files = extract_files(index_data['positive'])
            neg_files = extract_files(index_data['negative'])
            
            logger.info(f"Extracted files:")
            logger.info(f"  - Positive: {len(pos_files)} files")
            logger.info(f"  - Negative: {len(neg_files)} files")
            
            if pos_files:
                logger.info(f"  - Example positive file: {pos_files[0]}")
            if neg_files:
                logger.info(f"  - Example negative file: {neg_files[0]}")
            
            return pos_files, neg_files
        
        elif 'train' in index_data:
            logger.info("Found split-based structure (train/val/test)")
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

# ============ 数据集（带DM标签和诊断）============

class NPYFRBDataset(Dataset):
    def __init__(self, file_list, labels, config, npy_data_root, dm_dict=None, augment=True, logger=None):
        self.file_list = file_list
        self.labels = labels
        self.config = config
        self.npy_data_root = Path(npy_data_root)
        self.dm_dict = dm_dict if dm_dict is not None else {}
        self.augment = augment
        self.logger = logger
        
        # 验证文件路径
        self.valid_indices = []
        missing_count = 0
        dm_matched_count = 0
        dm_matched_files = []
        
        for idx, f in enumerate(self.file_list):
            full_path = self.npy_data_root / f
            if full_path.exists():
                self.valid_indices.append(idx)
                # 统计有DM标签的FRB样本
                if labels[idx] == 1:
                    if f in self.dm_dict:
                        dm_matched_count += 1
                        dm_matched_files.append((f, self.dm_dict[f]))
            else:
                missing_count += 1
                if missing_count <= 5:
                    if self.logger:
                        self.logger.warning(f"File not found: {full_path}")
        
        if self.logger:
            self.logger.info(f"Dataset created: {len(self.valid_indices)} valid / {len(self.file_list)} total files")
            if missing_count > 0:
                self.logger.warning(f"Missing files: {missing_count}")
            
            # ✅ 详细的DM匹配统计
            total_frb = sum(1 for idx in self.valid_indices if labels[idx] == 1)
            dm_coverage = dm_matched_count / max(total_frb, 1) * 100
            self.logger.info(f"FRB samples with DM labels: {dm_matched_count}/{total_frb} ({dm_coverage:.1f}%)")
            
            if dm_matched_count > 0:
                dm_values = [dm for _, dm in dm_matched_files]
                self.logger.info(f"  DM range in dataset: {min(dm_values):.1f} - {max(dm_values):.1f} pc/cm³")
                self.logger.info(f"  DM mean: {np.mean(dm_values):.1f} ± {np.std(dm_values):.1f} pc/cm³")
                # 显示前3个样本
                for i, (filename, dm) in enumerate(dm_matched_files[:3]):
                    self.logger.info(f"  Example {i+1}: {filename} -> DM={dm:.1f}")

    def __len__(self): 
        return len(self.valid_indices)
    
    def apply_augmentations(self, data):
        cloned = data.clone()
        _, n_time, n_freq = cloned.shape
        
        if np.random.rand() > 0.5:
            f = np.random.randint(0, self.config.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_freq - f))
            cloned[:, :, f0:f0+f] = 0
        
        if np.random.rand() > 0.5:
            t = np.random.randint(0, self.config.time_mask_param)
            t0 = np.random.randint(0, max(1, n_time - t))
            cloned[:, t0:t0+t, :] = 0
        
        if np.random.rand() > 0.5:
            shift_t = np.random.randint(-n_time // 4, n_time // 4)
            cloned = torch.roll(cloned, shifts=shift_t, dims=1)
        
        if np.random.rand() > 0.5:
            noise = torch.randn_like(cloned) * 0.05
            cloned = cloned + noise

        return cloned

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        npy_filename = self.file_list[actual_idx]
        npy_path = self.npy_data_root / npy_filename
        
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
            
            # 获取DM标签
            dm_value = self.dm_dict.get(npy_filename, -1.0)
            dm_tensor = torch.tensor(dm_value, dtype=torch.float32)
            
            return data_tensor, label, dm_tensor
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading {npy_path}: {e}")
            return torch.zeros(self.config.n_pol, self.config.n_time, self.config.n_freq), \
                   torch.tensor(self.labels[actual_idx], dtype=torch.long), \
                   torch.tensor(-1.0, dtype=torch.float32)

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

# ============ 增强的DM监督损失（带详细诊断）============

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

class SupervisedDMLoss(nn.Module):
    """增强的DM监督损失 - 带自适应权重和详细统计"""
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger
        self.focal_loss = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing
        )
        
        # ✅ 统计信息
        self.stats = {
            'total_batches': 0,
            'dm_active_batches': 0,
            'total_dm_samples': 0,
            'total_frb_samples': 0,
            'dm_loss_history': [],
            'last_report_batch': 0
        }
    
    def forward(self, logits, labels, dm_values, dm_attention, true_dm):
        self.stats['total_batches'] += 1
        
        # 1. 分类损失
        cls_loss = self.focal_loss(logits, labels)
        
        # 2. FRB样本统计
        frb_mask = (labels == 1)
        frb_count = frb_mask.sum().item()
        self.stats['total_frb_samples'] += frb_count
        
        # 3. DM标签统计
        has_dm_mask = (true_dm > 0) & frb_mask
        dm_count = has_dm_mask.sum().item()
        
        if has_dm_mask.any():
            self.stats['dm_active_batches'] += 1
            self.stats['total_dm_samples'] += dm_count
            
            # 计算加权平均DM
            weighted_dm = (dm_values * dm_attention).sum(dim=1)
            
            # ✅ DM回归损失：预测DM vs 真实DM
            dm_regression_loss = F.smooth_l1_loss(
                weighted_dm[has_dm_mask],
                true_dm[has_dm_mask]
            )
            
            # ✅ DM排序损失：鼓励注意力集中在最接近真实DM的trial上
            dm_diff = torch.abs(dm_values[has_dm_mask] - true_dm[has_dm_mask].unsqueeze(1))
            closest_idx = torch.argmin(dm_diff, dim=1)
            dm_ranking_loss = F.cross_entropy(
                dm_attention[has_dm_mask] * 100,
                closest_idx
            )
            
            # ✅ 自适应权重调整
            if self.config.use_adaptive_dm_weight:
                # 根据batch中DM样本比例动态调整权重
                dm_sample_ratio = dm_count / labels.size(0)
                # 样本越少，权重越大（补偿稀疏性）
                adaptive_factor = min(2.0, 1.0 / (dm_sample_ratio + 0.1))
                dm_reg_weight = self.config.dm_regression_weight * adaptive_factor
                dm_rank_weight = self.config.dm_ranking_weight * adaptive_factor
            else:
                dm_reg_weight = self.config.dm_regression_weight
                dm_rank_weight = self.config.dm_ranking_weight
            
            self.stats['dm_loss_history'].append({
                'regression': dm_regression_loss.item(),
                'ranking': dm_ranking_loss.item(),
                'samples': dm_count,
                'adaptive_factor': adaptive_factor if self.config.use_adaptive_dm_weight else 1.0
            })
            
        else:
            dm_regression_loss = torch.tensor(0.0).to(logits.device)
            dm_ranking_loss = torch.tensor(0.0).to(logits.device)
            dm_reg_weight = 0.0
            dm_rank_weight = 0.0
        
        # 4. DM平滑性损失（总是计算）
        dm_diff = dm_values[:, 1:] - dm_values[:, :-1]
        smoothness_loss = torch.mean(torch.abs(dm_diff))
        
        # 5. 总损失
        total_loss = (
            cls_loss +
            dm_reg_weight * dm_regression_loss +
            dm_rank_weight * dm_ranking_loss +
            0.01 * smoothness_loss
        )
        
        # ✅ 定期报告统计信息
        if self.logger and (self.stats['total_batches'] - self.stats['last_report_batch']) >= self.config.dm_stats_interval:
            self._log_statistics(dm_count, frb_count)
            self.stats['last_report_batch'] = self.stats['total_batches']
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'dm_regression': dm_regression_loss.item(),
            'dm_ranking': dm_ranking_loss.item(),
            'smoothness': smoothness_loss.item(),
            'dm_samples_in_batch': dm_count,
            'frb_samples_in_batch': frb_count,
            'dm_weight_used': dm_reg_weight
        }
    
    def _log_statistics(self, current_dm_count, current_frb_count):
        """记录详细统计信息"""
        coverage = self.stats['total_dm_samples'] / max(self.stats['total_frb_samples'], 1) * 100
        active_ratio = self.stats['dm_active_batches'] / max(self.stats['total_batches'], 1) * 100
        
        self.logger.info("="*60)
        self.logger.info("DM SUPERVISION STATISTICS")
        self.logger.info("="*60)
        self.logger.info(f"Overall Coverage: {coverage:.1f}% ({self.stats['total_dm_samples']}/{self.stats['total_frb_samples']} FRB samples)")
        self.logger.info(f"Active Batches: {active_ratio:.1f}% ({self.stats['dm_active_batches']}/{self.stats['total_batches']})")
        self.logger.info(f"Current Batch: {current_dm_count} DM samples / {current_frb_count} FRB samples")
        
        if len(self.stats['dm_loss_history']) > 0:
            recent = self.stats['dm_loss_history'][-10:]
            avg_reg = np.mean([x['regression'] for x in recent])
            avg_rank = np.mean([x['ranking'] for x in recent])
            avg_samples = np.mean([x['samples'] for x in recent])
            avg_adaptive = np.mean([x['adaptive_factor'] for x in recent])
            
            self.logger.info(f"Recent DM Losses (last 10):")
            self.logger.info(f"  - Regression: {avg_reg:.4f}")
            self.logger.info(f"  - Ranking: {avg_rank:.4f}")
            self.logger.info(f"  - Avg samples/batch: {avg_samples:.1f}")
            self.logger.info(f"  - Avg adaptive factor: {avg_adaptive:.2f}")
        self.logger.info("="*60)
    
    def get_stats(self):
        """获取统计摘要"""
        return {
            'dm_coverage': self.stats['total_dm_samples'] / max(self.stats['total_frb_samples'], 1) * 100,
            'dm_batch_ratio': self.stats['dm_active_batches'] / max(self.stats['total_batches'], 1) * 100,
            **self.stats
        }
    
    def reset_stats(self):
        """重置统计（用于每个epoch）"""
        self.stats = {
            'total_batches': 0,
            'dm_active_batches': 0,
            'total_dm_samples': 0,
            'total_frb_samples': 0,
            'dm_loss_history': [],
            'last_report_batch': 0
        }

# ============ 评估指标 ============

class MetricsCalculator:
    def __init__(self): 
        self.reset()
    
    def reset(self):
        self.all_preds, self.all_labels, self.all_probs = [], [], []
        self.all_dm_pred, self.all_dm_true = [], []
        self.correct = self.total = self.tp = self.tn = self.fp = self.fn = 0
    
    def update(self, preds, labels, probs=None, dm_pred=None, dm_true=None):
        preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())
        
        if probs is not None:
            probs = probs.cpu().detach().numpy() if torch.is_tensor(probs) else probs
            self.all_probs.extend(probs.tolist())
        
        if dm_pred is not None:
            dm_pred = dm_pred.cpu().detach().numpy() if torch.is_tensor(dm_pred) else dm_pred
            self.all_dm_pred.extend(dm_pred.tolist())
        
        if dm_true is not None:
            dm_true = dm_true.cpu().detach().numpy() if torch.is_tensor(dm_true) else dm_true
            self.all_dm_true.extend(dm_true.tolist())
        
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
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # DM预测精度
        if len(self.all_dm_pred) > 0 and len(self.all_dm_true) > 0:
            dm_pred_arr = np.array(self.all_dm_pred)
            dm_true_arr = np.array(self.all_dm_true)
            
            valid_mask = dm_true_arr > 0
            if valid_mask.any():
                dm_mae = np.mean(np.abs(dm_pred_arr[valid_mask] - dm_true_arr[valid_mask]))
                dm_rmse = np.sqrt(np.mean((dm_pred_arr[valid_mask] - dm_true_arr[valid_mask])**2))
                dm_mape = np.mean(np.abs((dm_pred_arr[valid_mask] - dm_true_arr[valid_mask]) / (dm_true_arr[valid_mask] + 1e-6))) * 100
                metrics['dm_mae'] = dm_mae
                metrics['dm_rmse'] = dm_rmse
                metrics['dm_mape'] = dm_mape
                metrics['dm_valid_samples'] = valid_mask.sum()
        
        return metrics
    
    def format_metrics(self, metrics, prefix=""):
        base_str = f"{prefix}Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | Rec: {metrics['recall']:.4f}"
        if 'dm_mae' in metrics:
            base_str += f" | DM_MAE: {metrics['dm_mae']:.1f} | DM_MAPE: {metrics.get('dm_mape', 0):.1f}%"
        return base_str

# ============ 训练器（增强版）============

class Trainer:
    def __init__(self, config, logger, exp_dir):
        self.config = config
        self.logger = logger
        self.exp_dir = exp_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = AdvancedFRBDetector(config).to(self.device)
        self.logger.info(f"Initialized AdvancedFRBDetector V3.1 (Enhanced DM) on {self.device}")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )
        self.criterion = SupervisedDMLoss(config, logger=logger)
        self.scaler = GradScaler() if config.use_amp else None
        
        self.early_stopping = EarlyStopping(
            patience=config.patience, 
            min_epochs=config.min_epochs_before_stop,
            verbose=True, 
            path=str(exp_dir / 'best_model.pth'), 
            trace_func=logger.info
        )
        
        self.current_epoch = 0
        self.train_losses, self.val_losses = [], []

    def train_epoch(self, train_loader):
        self.model.train()
        self.criterion.reset_stats()  # ✅ 重置epoch统计
        
        total_loss = 0.0
        metrics_calc = MetricsCalculator()
        loss_components = {k: 0.0 for k in ['cls_loss', 'dm_regression', 'dm_ranking', 'smoothness']}
        
        # ✅ 添加DM样本计数
        epoch_dm_samples = 0
        epoch_frb_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.epochs}')
        
        self.optimizer.zero_grad()
        for batch_idx, (data, labels, true_dm) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            true_dm = true_dm.to(self.device)
            
            # ✅ 第一个batch时打印诊断信息
            if batch_idx == 0 and self.current_epoch == 0:
                self.logger.info("="*60)
                self.logger.info("FIRST BATCH DIAGNOSIS")
                self.logger.info("="*60)
                self.logger.info(f"Batch shape: {data.shape}")
                self.logger.info(f"Labels: {labels}")
                self.logger.info(f"True DM values: {true_dm}")
                has_dm = (true_dm > 0) & (labels == 1)
                self.logger.info(f"Has DM labels: {has_dm.sum()}/{len(true_dm)}")
                if has_dm.any():
                    self.logger.info(f"DM values in batch: {true_dm[has_dm]}")
                self.logger.info("="*60)
            
            if self.config.use_amp:
                with autocast():
                    logits, interp = self.model(data, return_interpretations=True)
                    loss, loss_dict = self.criterion(
                        logits, labels,
                        interp['dm_values'],
                        interp['dm_attention'],
                        true_dm
                    )
                    loss = loss / self.config.accumulation_steps
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits, interp = self.model(data, return_interpretations=True)
                loss, loss_dict = self.criterion(
                    logits, labels,
                    interp['dm_values'],
                    interp['dm_attention'],
                    true_dm
                )
                loss = loss / self.config.accumulation_steps
                loss.backward()
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.accumulation_steps
            for k, v in loss_dict.items():
                if k in loss_components:
                    loss_components[k] += v
            
            # ✅ 统计DM使用情况
            epoch_dm_samples += loss_dict['dm_samples_in_batch']
            epoch_frb_samples += loss_dict['frb_samples_in_batch']
            
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            weighted_dm = (interp['dm_values'] * interp['dm_attention']).sum(dim=1)
            metrics_calc.update(predicted, labels, probs[:, 1], weighted_dm, true_dm)
            
            # ✅ 更新进度条，显示DM信息
            if (batch_idx + 1) % self.config.metrics_update_interval == 0:
                m = metrics_calc.get_metrics()
                dm_info = f"DM:{epoch_dm_samples}/{epoch_frb_samples}" if epoch_frb_samples > 0 else "DM:0"
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'f1': f'{m["f1_score"]:.3f}',
                    'dm_reg': f'{loss_components["dm_regression"]/(batch_idx+1):.4f}',
                    dm_info: f'{epoch_dm_samples/max(epoch_frb_samples,1)*100:.0f}%'
                })
        
        # ✅ Epoch结束时的详细报告
        avg_loss_components = {k: v/len(train_loader) for k, v in loss_components.items()}
        dm_coverage = epoch_dm_samples / max(epoch_frb_samples, 1) * 100
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"EPOCH {self.current_epoch+1} TRAINING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Loss Components: {avg_loss_components}")
        self.logger.info(f"DM Label Usage: {epoch_dm_samples}/{epoch_frb_samples} FRB samples ({dm_coverage:.1f}%)")
        
        dm_stats = self.criterion.get_stats()
        self.logger.info(f"DM Batch Coverage: {dm_stats['dm_batch_ratio']:.1f}% of batches")
        self.logger.info("="*60 + "\n")
        
        return total_loss / len(train_loader), metrics_calc.get_metrics()

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        metrics_calc = MetricsCalculator()
        
        # ✅ 验证时的DM统计
        val_dm_samples = 0
        val_frb_samples = 0
        
        for data, labels, true_dm in tqdm(val_loader, desc='Validating'):
            data, labels, true_dm = data.to(self.device), labels.to(self.device), true_dm.to(self.device)
            
            with autocast(enabled=self.config.use_amp):
                logits, interp = self.model(data, return_interpretations=True)
                loss, loss_dict = self.criterion(
                    logits, labels,
                    interp['dm_values'],
                    interp['dm_attention'],
                    true_dm
                )
            
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            weighted_dm = (interp['dm_values'] * interp['dm_attention']).sum(dim=1)
            metrics_calc.update(predicted, labels, probs[:, 1], weighted_dm, true_dm)
            
            val_dm_samples += loss_dict['dm_samples_in_batch']
            val_frb_samples += loss_dict['frb_samples_in_batch']
        
        metrics = metrics_calc.get_metrics()
        
        # ✅ 添加DM覆盖率到输出
        dm_coverage_str = ""
        if val_frb_samples > 0:
            dm_coverage = val_dm_samples / val_frb_samples * 100
            dm_coverage_str = f" | DM_Cov: {dm_coverage:.1f}%"
        
        return total_loss / len(val_loader), metrics, metrics_calc.format_metrics(metrics, prefix="  ") + dm_coverage_str

    def train(self, train_loader, val_loader):
        self.logger.info("\n" + "="*60)
        self.logger.info("STARTING TRAINING - FRB Detector V3.1")
        self.logger.info("="*60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Epochs: {self.config.epochs}")
        self.logger.info(f"  - Batch Size: {self.config.batch_size}")
        self.logger.info(f"  - DM Regression Weight: {self.config.dm_regression_weight}")
        self.logger.info(f"  - Adaptive DM Weight: {self.config.use_adaptive_dm_weight}")
        self.logger.info(f"  - Early Stopping: After epoch {self.config.min_epochs_before_stop}")
        self.logger.info("="*60 + "\n")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics, val_output = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step()
            
            self.logger.info(f'\n{"="*60}')
            self.logger.info(f'Epoch [{epoch+1}/{self.config.epochs}] Summary')
            self.logger.info(f'{"="*60}')
            self.logger.info(f'Train - Loss: {train_loss:.4f} | {MetricsCalculator().format_metrics(train_metrics, "")}')
            self.logger.info(f'Val   - {val_output}')
            self.logger.info(f'{"="*60}\n')
            
            if (epoch + 1) % self.config.save_interval == 0:
                ckpt_path = self.exp_dir / f'model_epoch_{epoch+1}.pth'
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"Saved checkpoint to {ckpt_path}")

            self.early_stopping(val_metrics['f1_score'], self.model, val_loss, current_epoch=epoch+1)
            
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info("\n" + "="*60)
        self.logger.info("STARTING FINAL TESTING")
        self.logger.info("="*60 + "\n")
        
        best_model_path = self.exp_dir / 'best_model.pth'
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.info(f"Loaded best model from {best_model_path}")

        self.model.eval()
        metrics_calc = MetricsCalculator()
        
        # ✅ 测试时的DM统计
        test_dm_samples = 0
        test_frb_samples = 0
        
        for data, labels, true_dm in tqdm(test_loader, desc='Testing'):
            data, labels, true_dm = data.to(self.device), labels.to(self.device), true_dm.to(self.device)
            
            with autocast(enabled=self.config.use_amp):
                logits, interp = self.model(data, return_interpretations=True)
            
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            weighted_dm = (interp['dm_values'] * interp['dm_attention']).sum(dim=1)
            metrics_calc.update(predicted, labels, probs[:, 1], weighted_dm, true_dm)
            
            # 统计
            has_dm = (true_dm > 0) & (labels == 1)
            test_dm_samples += has_dm.sum().item()
            test_frb_samples += (labels == 1).sum().item()
        
        y_true = metrics_calc.all_labels
        y_pred = metrics_calc.all_preds
        y_prob = metrics_calc.all_probs
        
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL TEST RESULTS")
        self.logger.info("="*60)
        
        # ✅ DM标签覆盖率
        if test_frb_samples > 0:
            dm_coverage = test_dm_samples / test_frb_samples * 100
            self.logger.info(f"\nDM Label Coverage in Test Set: {dm_coverage:.1f}% ({test_dm_samples}/{test_frb_samples} FRB samples)")
        
        report = classification_report(y_true, y_pred, target_names=['Noise', 'FRB'], digits=4)
        self.logger.info(f"\nClassification Report:\n{report}")
        
        cm = confusion_matrix(y_true, y_pred)
        self.logger.info(f"\nConfusion Matrix:\n{cm}")
        
        try:
            auc = roc_auc_score(y_true, y_prob)
            self.logger.info(f"\nROC AUC: {auc:.4f}")
        except:
            self.logger.info("\nROC AUC: N/A")
        
        metrics = metrics_calc.get_metrics()
        if 'dm_mae' in metrics:
            self.logger.info(f"\nDM Prediction Performance:")
            self.logger.info(f"  Valid Samples: {metrics.get('dm_valid_samples', 0)}")
            self.logger.info(f"  MAE: {metrics['dm_mae']:.2f} pc/cm³")
            self.logger.info(f"  RMSE: {metrics['dm_rmse']:.2f} pc/cm³")
            self.logger.info(f"  MAPE: {metrics.get('dm_mape', 0):.2f}%")
        
        self.logger.info("="*60 + "\n")
        
        return metrics

# ============ 主程序 ============

def main():
    config = Config()
    logger, exp_dir = setup_logger(config)
    
    logger.info("="*60)
    logger.info("FRB Detector V3.1 - Enhanced Supervised DM Learning")
    logger.info("="*60)
    
    # 1. 加载DM标签
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading DM Labels")
    logger.info("="*60)
    dm_dict = load_dm_labels(config.dm_label_dirs, logger)
    
    # 2. 加载数据索引
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Loading Dataset Index")
    logger.info("="*60)
    pos_files, neg_files = load_npy_index(config.dataset_index_file, logger)
    
    if pos_files is None or neg_files is None:
        logger.error("Failed to load dataset index. Trying direct file search...")
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
    
    # ✅ 检查DM标签匹配率
    frb_files = [f for f, l in zip(all_files, labels) if l == 1]
    matched_count = sum(1 for f in frb_files if f in dm_dict)
    overall_coverage = matched_count / len(frb_files) * 100 if frb_files else 0
    logger.info(f"\n⚠️  DM Label Overall Coverage: {matched_count}/{len(frb_files)} ({overall_coverage:.1f}%)")
    if overall_coverage < 10:
        logger.warning("⚠️  WARNING: Very low DM label coverage! This may limit supervised learning effectiveness.")
    
    # 3. 数据划分
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Splitting Dataset")
    logger.info("="*60)
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
    
    # 4. DataLoader
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Creating Datasets with DM Labels")
    logger.info("="*60)
    train_dataset = NPYFRBDataset(train_files, train_labels, config, config.npy_data_root, 
                                  dm_dict=dm_dict, augment=True, logger=logger)
    val_dataset = NPYFRBDataset(val_files, val_labels, config, config.npy_data_root,
                                dm_dict=dm_dict, augment=False, logger=logger)
    test_dataset = NPYFRBDataset(test_files, test_labels, config, config.npy_data_root,
                                 dm_dict=dm_dict, augment=False, logger=logger)
    
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
    
    logger.info(f"\nDataLoader Created:")
    logger.info(f"  - Train batches: {len(train_loader)}")
    logger.info(f"  - Val batches: {len(val_loader)}")
    logger.info(f"  - Test batches: {len(test_loader)}")
    
    # 5. 训练与测试
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Training and Evaluation")
    logger.info("="*60 + "\n")
    
    trainer = Trainer(config, logger, exp_dir)
    trainer.train(train_loader, val_loader)
    trainer.test(test_loader)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)

if __name__ == "__main__":
    main()