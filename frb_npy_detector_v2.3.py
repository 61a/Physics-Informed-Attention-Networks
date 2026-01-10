"""
FRB检测器 V2.3 (抗过拟合 & 全流程优化版)
---------------------------------------------------------
V2.3 更新内容：
1. [抗过拟合] 引入 Early Stopping 机制。
2. [抗过拟合] 增强数据增广 (Time/Freq Shift)。
3. [抗过拟合] 提升 Dropout (0.1->0.3) 和 Weight Decay (1e-4->1e-2)。
4. [功能] 增加中间模型定期保存功能。
5. [功能] 增加 Train/Val/Test 三集划分及最终测试环节。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
import glob
import os
from pathlib import Path
import logging
import sys
import json
from tqdm import tqdm
import warnings
import pickle
import math
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# ============ 配置 (针对过拟合调整) ============

class Config:
    # 路径配置
    npy_data_root = "F:/FRB/npy_data"
    dataset_index_file = "F:/FRB/npy_data/dataset_index.pkl"
    save_dir = "./experiments"
    exp_name = "frb_v2.3_0108"
    
    # 数据维度
    n_time = 15360
    n_freq = 1024
    n_pol = 1
    
    # 物理模型配置
    n_dm_trials = 32
    dm_range = (0, 3000)
    
    # 训练配置
    batch_size = 2       # 显存允许的话尽量大
    accumulation_steps = 4
    
    # IO 设置
    num_workers = 4
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True
    
    epochs = 100
    # [抗过拟合] 增加早停耐心值
    patience = 15 
    
    # [抗过拟合] 增大 Weight Decay
    lr = 5e-5
    weight_decay = 1e-2 
    
    # [抗过拟合] 增大 Dropout
    dropout_rate = 0.3
    
    use_amp = True
    gradient_clip = 5.0
    
    focal_alpha = 0.25
    focal_gamma = 2.0
    label_smoothing = 0.1 # [抗过拟合] 标签平滑
    
    # 增强参数
    freq_mask_param = 150 # [增强] 增大遮挡范围
    time_mask_param = 600
    
    log_interval = 10
    metrics_update_interval = 50
    save_interval = 10    # [功能] 每10个epoch保存一次

# ============ 工具类 ============

class EarlyStopping:
    """早停机制"""
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
        # 这里我们使用 F1 Score 作为指标，越大越好
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
        '''Saves model when validation score increase.'''
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 日志系统
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if stream is None: stream = sys.stderr
        if hasattr(stream, 'reconfigure'):
            try: stream.reconfigure(encoding='utf-8')
            except Exception: pass
        elif hasattr(stream, 'buffer'):
            import io
            self.stream = io.TextIOWrapper(stream.buffer, encoding='utf-8', 
                                          line_buffering=True, write_through=True)
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception: self.handleError(record)

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

# ============ 数据集 (增强版) ============

def load_npy_index(index_file):
    if not os.path.exists(index_file): raise FileNotFoundError(f"Index not found: {index_file}")
    with open(index_file, 'rb') as f: index_data = pickle.load(f)
    pos_data = index_data.get('positive', {})
    neg_data = index_data.get('negative', {})
    
    def extract_files(data):
        if isinstance(data, dict):
            if 'files' in data:
                files = data['files']
                return files if isinstance(files, list) else [k for k in files.keys() if k.endswith('.npy')]
            return [k for k in data.keys() if isinstance(k, str) and k.endswith('.npy')]
        return data if isinstance(data, list) else []
    
    return extract_files(pos_data), extract_files(neg_data)

class NPYFRBDataset(Dataset):
    def __init__(self, file_list, labels, config, augment=True):
        self.file_list = file_list
        self.labels = labels
        self.config = config
        self.augment = augment

    def __len__(self): return len(self.file_list)
    
    def apply_augmentations(self, data):
        # data: (Pol, Time, Freq)
        
        # 1. SpecAugment (Masking)
        cloned = data.clone()
        _, n_time, n_freq = cloned.shape
        
        # Freq Masking
        f = np.random.randint(0, self.config.freq_mask_param)
        f0 = np.random.randint(0, max(1, n_freq - f))
        cloned[:, :, f0:f0+f] = 0
        
        # Time Masking
        t = np.random.randint(0, self.config.time_mask_param)
        t0 = np.random.randint(0, max(1, n_time - t))
        cloned[:, t0:t0+t, :] = 0
        
        # 2. [新增] Random Shift (Rolling) - 模拟FRB出现位置的不确定性
        if np.random.rand() > 0.5:
            shift_t = np.random.randint(-n_time // 4, n_time // 4)
            cloned = torch.roll(cloned, shifts=shift_t, dims=1)
            
        # 3. [新增] Gaussian Noise Injection
        if np.random.rand() > 0.5:
            noise = torch.randn_like(cloned) * 0.05
            cloned = cloned + noise

        return cloned

    def __getitem__(self, idx):
        npy_path = self.file_list[idx]
        try:
            data = np.load(npy_path, mmap_mode='r') 
            data_copy = np.array(data, dtype=np.float32)
            
            if data_copy.ndim == 2:
                data_copy = data_copy[np.newaxis, :, :]

            # 标准化
            mean = data_copy.mean()
            std = data_copy.std()
            if std > 1e-6:
                data_copy = (data_copy - mean) / std
            else:
                data_copy -= mean
            
            # 转 Tensor
            data_copy = np.ascontiguousarray(data_copy)
            data_tensor = torch.from_numpy(data_copy).permute(1, 0, 2).contiguous() # (Pol, Time, Freq)
            
            # 增强 (Flip 放在 Tensor 转换前或后都可以，这里放在后)
            if self.augment:
                if np.random.rand() > 0.5: 
                     # Time Flip
                    data_tensor = torch.flip(data_tensor, dims=[1])
                data_tensor = self.apply_augmentations(data_tensor)
                
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return data_tensor, label
            
        except Exception as e:
            return torch.zeros(self.config.n_pol, self.config.n_time, self.config.n_freq), torch.tensor(self.labels[idx], dtype=torch.long)

# ============ 物理感知模块 ============

class EfficientDedispersionLayer(nn.Module):
    def __init__(self, n_freq, n_time, n_dm_trials=32, dm_range=(0, 3000)):
        super().__init__()
        self.n_dm_trials = n_dm_trials
        initial_dms = torch.linspace(dm_range[0], dm_range[1], n_dm_trials)
        self.dm_values = nn.Parameter(initial_dms)
        freq_indices = torch.linspace(0, 1, n_freq)
        self.register_buffer('freq_scale', (freq_indices + 0.1) ** (-2))
        
    def forward(self, x):
        batch_size, n_pol, n_time, n_freq = x.shape
        delays = self.dm_values.unsqueeze(-1) * self.freq_scale.unsqueeze(0)
        delays = torch.tanh(delays / 1000) * 500
        
        output_list = []
        chunk_size = 128
        
        for dm_idx in range(self.n_dm_trials):
            delay_per_freq = delays[dm_idx]
            shifted_data = x.clone()
            for freq_start in range(0, n_freq, chunk_size):
                freq_end = min(freq_start + chunk_size, n_freq)
                avg_delay = int(delay_per_freq[freq_start:freq_end].mean().item())
                if avg_delay != 0:
                    shifted_data[:, :, :, freq_start:freq_end] = torch.roll(
                        shifted_data[:, :, :, freq_start:freq_end], shifts=-avg_delay, dims=2
                    )
            output_list.append(shifted_data)
            
        return torch.stack(output_list, dim=1), delays

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

# ============ Conformer 组件 ============

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x): return x.transpose(self.dim1, self.dim2)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), Swish(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

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
    def forward(self, x): return self.net(x)

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
        
        self.dedispersion = EfficientDedispersionLayer(
            n_freq=config.n_freq, n_time=config.n_time,
            n_dm_trials=config.n_dm_trials, dm_range=config.dm_range
        )
        self.dm_attention = DMAttentionModule(
            n_dm_trials=config.n_dm_trials, n_time=config.n_time, n_freq=config.n_freq,
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
        dedispersed, delays = self.dedispersion(x)
        dm_corrected, dm_attn = self.dm_attention(dedispersed)
        return dm_corrected, dm_attn, delays

    def forward(self, x, return_interpretations=False):
        if self.training and x.requires_grad:
             dm_corrected, dm_attn, delays = checkpoint.checkpoint(self.forward_physics, x)
        else:
             dm_corrected, dm_attn, delays = self.forward_physics(x)

        feat = self.freq_compress(dm_corrected)
        B, C, T, F = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(B, T, C * F)
        feat = self.projection(feat)
        
        for layer in self.conformer_layers:
            if self.training:
                feat = checkpoint.checkpoint(layer, feat)
            else:
                feat = layer(feat)
            
        global_feat = feat.mean(dim=1)
        combined = torch.cat([global_feat, dm_attn], dim=1)
        logits = self.classifier(combined)
        
        if return_interpretations:
            return logits, {
                'dm_attention': dm_attn,
                'dm_values': self.dedispersion.dm_values,
                'delays': delays,
                'dedispersed_data': dm_corrected
            }
        return logits

# ============ 损失函数与评价 ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs: logits
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return torch.mean(focal_loss)
        elif self.reduction == 'sum': return torch.sum(focal_loss)
        else: return focal_loss

class MetricsCalculator:
    def __init__(self): self.reset()
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
        if self.total == 0: return {}
        accuracy = self.correct / self.total
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
        return metrics
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
        self.logger.info(f"Initialized AdvancedFRBDetector on {self.device}")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )
        # 引入 Label Smoothing
        self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
        self.scaler = GradScaler() if config.use_amp else None
        
        # 早停
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
        self.logger.info("Starting Training with Anti-Overfitting Strategy")
        
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
            
            # 中间保存
            if (epoch + 1) % self.config.save_interval == 0:
                ckpt_path = self.exp_dir / f'model_epoch_{epoch+1}.pth'
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"Saved checkpoint to {ckpt_path}")

            # 早停检查 (基于 F1 Score)
            self.early_stopping(val_metrics['f1_score'], self.model, val_loss)
            
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info("Starting Final Testing...")
        # 加载最佳模型
        best_model_path = self.exp_dir / 'best_model.pth'
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.info(f"Loaded best model from {best_model_path}")
        else:
            self.logger.warning("Best model not found, using current model weights.")

        self.model.eval()
        metrics_calc = MetricsCalculator()
        
        for data, labels in tqdm(test_loader, desc='Testing'):
            data, labels = data.to(self.device), labels.to(self.device)
            with autocast(enabled=self.config.use_amp):
                logits = self.model(data)
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            metrics_calc.update(predicted, labels, probs[:, 1])
        
        # 生成详细报告
        y_true = metrics_calc.all_labels
        y_pred = metrics_calc.all_preds
        y_prob = metrics_calc.all_probs
        
        self.logger.info("\n" + "="*30)
        self.logger.info("FINAL TEST RESULTS")
        self.logger.info("="*30)
        
        report = classification_report(y_true, y_pred, target_names=['Noise', 'FRB'], digits=4)
        self.logger.info(f"\nClassification Report:\n{report}")
        
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
    
    # 1. 加载数据索引
    try:
        pos_files, neg_files = load_npy_index(config.dataset_index_file)
    except:
        pos_files = glob.glob(os.path.join(config.npy_data_root, 'positive', '*.npy'))
        neg_files = glob.glob(os.path.join(config.npy_data_root, 'negative', '*.npy'))
    
    all_files = pos_files + neg_files
    labels = [1]*len(pos_files) + [0]*len(neg_files)
    
    # 2. 数据划分 (Train/Val/Test) - 70% / 15% / 15%
    from sklearn.model_selection import train_test_split
    
    # 先分出 Test (15%)
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    # 再从剩余的分出 Val (15% / 0.85 ≈ 17.6%)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=0.176, random_state=42, stratify=train_val_labels
    )
    
    logger.info(f"Data Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # 3. DataLoader
    train_dataset = NPYFRBDataset(train_files, train_labels, config, augment=True)
    val_dataset = NPYFRBDataset(val_files, val_labels, config, augment=False)
    test_dataset = NPYFRBDataset(test_files, test_labels, config, augment=False)
    
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
    
    # 4. 训练与测试
    trainer = Trainer(config, logger, exp_dir)
    trainer.train(train_loader, val_loader)
    trainer.test(test_loader)

if __name__ == "__main__":
    main()