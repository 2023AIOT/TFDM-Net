"""
utils.py — Only2D TF-SSM 工具集
保留原预处理、损失、指标、曲线绘图接口。
"""
import os, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import pywt
from PIL import Image


# ============================================================================
# 种子 & 目录
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ============================================================================
# CWT
# ============================================================================
# ============================================================================
# CWT
# ============================================================================
# ============================================================================
# Adaptive Multi-scale Morse Wavelet Time-Frequency Representation
# 自适应多尺度 Morse 小波时频表示
# ============================================================================

# ============================================================================
# Adaptive Multi-scale Morlet CWT
# 自适应多尺度 Morlet-CWT 时频表示
# ============================================================================

# ============================================================================
# Log-scale Morlet CWT
# 非线性尺度 Morlet-CWT 时频表示
# ============================================================================

# ============================================================================
# Synchrosqueezed CWT-enhanced Time-Frequency Representation
# 同步压缩增强 CWT 时频表示
# ============================================================================

def _resize_img(img, target_size=224):
    return np.array(
        Image.fromarray(img.astype(np.float32)).resize(
            (target_size, target_size),
            Image.BILINEAR
        )
    )


def _standardize_img(img):
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-6)


def _minmax_norm(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-6)


def _synchrosqueezed_cwt_map(signal, scales=224, target_size=224):
    """
    Synchrosqueezed CWT-enhanced TFR.

    输入:
        signal: 1D force segment, e.g. Fx / Fy / Fz

    输出:
        map: [target_size, target_size]

    说明:
        这是工程化的一阶同步压缩 CWT 表示。
        它不是严格高阶 HSST，但可以作为 CWT 的高分辨率增强版本先做实验。
    """

    x = np.asarray(signal, dtype=np.float64)

    # 不做 z-score，避免消除磨损阶段之间的幅值/能量差异
    # 只去除直流分量，减少低频偏置
    x = x - np.mean(x)

    n = len(x)
    scales_arr = np.arange(1, scales + 1)

    # 使用复 Morlet，小波系数有相位，才能估计瞬时频率
    # pywt 的 'morl' 是实 Morlet，不适合做同步压缩
    wavelet = 'cmor1.5-1.0'

    coeff, freqs = pywt.cwt(x, scales_arr, wavelet, sampling_period=1.0)

    amp = np.abs(coeff).astype(np.float64)

    # 相位展开 + 时间方向相位导数，估计瞬时频率
    phase = np.unwrap(np.angle(coeff), axis=1)
    inst_freq = np.gradient(phase, axis=1) / (2.0 * np.pi)

    f_min = float(np.min(freqs))
    f_max = float(np.max(freqs))

    # 异常值处理
    inst_freq = np.nan_to_num(inst_freq, nan=0.0, posinf=f_max, neginf=f_min)
    inst_freq = np.clip(inst_freq, f_min, f_max)

    # 将 CWT 能量重排到瞬时频率位置
    sst = np.zeros((scales, n), dtype=np.float64)

    idx = ((inst_freq - f_min) / (f_max - f_min + 1e-12) * (scales - 1))
    idx = np.clip(idx, 0, scales - 1).astype(np.int64)

    time_idx = np.tile(np.arange(n), (scales, 1))

    # 能量阈值：去掉特别弱的系数，减少噪声重排
    thr = np.percentile(amp, 45)
    mask = amp >= thr

    np.add.at(
        sst,
        (idx[mask], time_idx[mask]),
        amp[mask]
    )

    # 同步压缩图
    sst = np.log1p(sst)

    # 保留一部分原始 CWT 能量图，防止 SST 太稀疏导致 Normal 类不稳定
    cwt_map = np.log1p(amp)

    # CWT 的尺度方向和频率方向相反，这里翻转后与 SST 频率方向大致一致
    cwt_map = cwt_map[::-1, :]

    sst = _standardize_img(sst)
    cwt_map = _standardize_img(cwt_map)

    # 主要用同步压缩图，同时保留少量 CWT 稳定纹理
    fused = 0.85 * sst + 0.15 * cwt_map

    fused = _standardize_img(fused)
    fused = _resize_img(fused, target_size)
    fused = _standardize_img(fused)

    return fused.astype(np.float32)


def compute_cwt_3channel(fx, fy, fz, scales=224, wavelet='morl', target_size=224):
    """
    Fx/Fy/Fz -> Synchrosqueezed CWT-enhanced maps -> [3, target_size, target_size]

    注意:
        保留 compute_cwt_3channel 函数名，是为了兼容 preprocess.py。
        wavelet 参数保留但不使用，因为同步压缩需要复 Morlet。
    """
    fx_map = _synchrosqueezed_cwt_map(
        fx, scales=scales, target_size=target_size
    )
    fy_map = _synchrosqueezed_cwt_map(
        fy, scales=scales, target_size=target_size
    )
    fz_map = _synchrosqueezed_cwt_map(
        fz, scales=scales, target_size=target_size
    )

    return np.stack([fx_map, fy_map, fz_map], axis=0).astype(np.float32)


def compute_cwt(signal, scales=224, wavelet='morl', target_size=224):
    """
    Single-channel Synchrosqueezed CWT-enhanced map.
    """
    return _synchrosqueezed_cwt_map(
        signal,
        scales=scales,
        target_size=target_size
    ).astype(np.float32)
# ============================================================================
# 损失函数
# ============================================================================
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is not None:
            self.register_buffer('weights', torch.FloatTensor(weights))
        else:
            self.weights = None

    def forward(self, pred, target):
        return F.cross_entropy(pred, target, weight=self.weights)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', torch.FloatTensor(alpha))
        else:
            self.alpha = None

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            fl = self.alpha[target] * fl
        return fl.mean()


def feature_consistency_loss(z1, z2):
    return (1.0 - F.cosine_similarity(z1, z2, dim=-1)).mean()


def modality_consistency_loss(logits_1d, logits_2d):
    p1 = F.log_softmax(logits_1d, dim=1)
    p2 = F.log_softmax(logits_2d, dim=1)
    q1 = F.softmax(logits_1d, dim=1)
    q2 = F.softmax(logits_2d, dim=1)
    return (F.kl_div(p1, q2, reduction='batchmean') +
            F.kl_div(p2, q1, reduction='batchmean')) * 0.5


# ============================================================================
# Cosine Schedule with Warmup（epoch 级）
# ============================================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-2, float(epoch) / float(max(1, warmup_epochs)))
        prog = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 指标
# ============================================================================
def compute_metrics(y_true, y_pred):
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=['Initial', 'Normal', 'Severe'],
        digits=4,
        zero_division=0,
    )
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=labels, zero_division=0)
    return {
        'confusion_matrix': cm, 'report': report,
        'initial_precision': prec[0], 'normal_precision': prec[1], 'severe_precision': prec[2],
        'initial_recall': rec[0], 'normal_recall': rec[1], 'severe_recall': rec[2],
        'initial_f1': f1[0], 'normal_f1': f1[1], 'severe_f1': f1[2],
        'macro_precision': mp, 'macro_recall': mr, 'macro_f1': mf,
        'normal_to_severe':  int(cm[1, 2]),
        'normal_to_initial': int(cm[1, 0]),
        'severe_to_normal':  int(cm[2, 1]),
        'initial_to_normal': int(cm[0, 1]),
    }


# ============================================================================
# 可视化
# ============================================================================
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_confusion_matrix(cm, class_names=['Initial', 'Normal', 'Severe']):
    total = cm.sum()
    correct = np.trace(cm)
    print(f"\n{'='*60}\n混淆矩阵详细分析\n{'='*60}")
    print(f"总样本: {total}  正确: {correct} ({100*correct/total:.2f}%)")
    for i, name in enumerate(class_names):
        tc = cm[i, :].sum()
        cc = cm[i, i]
        print(f"\n  {name}: {cc}/{tc} ({100*cc/tc:.2f}%)")
        for j, pn in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                print(f"    -> {pn}: {cm[i, j]} ({100*cm[i, j]/tc:.2f}%)")
    if cm.shape[0] >= 3:
        print(f"\n关键: Normal->Severe={cm[1,2]}  Severe->Normal={cm[2,1]}")
    print(f"{'='*60}\n")


def plot_curves(history, save_dir):
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history['train_acc'], label='Train')
    ax.plot(history['val_acc'], label='Val')
    ax.set_title('Accuracy (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(history['val_macro_f1'], color='purple', lw=2.5)
    ax.set_title('Val Macro-F1')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history['val_initial_f1'], label='Initial')
    ax.plot(history['val_normal_f1'], label='Normal', color='green', lw=2.5)
    ax.plot(history['val_severe_f1'], label='Severe')
    ax.set_title('Per-Class F1')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history['val_normal_precision'], label='Prec', ls='--')
    ax.plot(history['val_normal_recall'], label='Rec', ls='--')
    ax.plot(history['val_normal_f1'], label='F1', color='green', lw=2.5)
    ax.set_title('Normal P/R/F1')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(history['val_severe_recall'], label='Recall', color='red')
    ax.plot(history['val_severe_f1'], label='F1', color='orange', ls='--')
    ax.set_title('Severe Metrics')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.plot(history['val_normal_to_severe'], label='N->S', color='red', marker='o', ms=2)
    ax.plot(history['val_normal_to_initial'], label='N->I', color='blue', marker='s', ms=2)
    ax.set_title('Normal Misclassification')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(history['val_severe_to_normal'], label='S->N', color='darkred', lw=2.5)
    ax.set_title('Severe Misses')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
    ax2 = ax.twinx()
    ax.plot(history['val_loss'], label='Val Loss', alpha=0.6)
    ax2.plot(history['val_macro_f1'], label='Macro-F1', color='purple', lw=2.5)
    ax.set_title('Loss vs F1')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 训练曲线已保存")
