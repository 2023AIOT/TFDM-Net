"""
train.py — TFDM-Net 训练脚本

"""

import os, json, argparse, platform, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from utils import (
    set_seed, ensure_dir, plot_curves, compute_metrics,
    WeightedCrossEntropyLoss, FocalLoss,
    get_cosine_schedule_with_warmup,
    analyze_confusion_matrix, plot_confusion_matrix,
)


def get_num_workers():
    return 0 if platform.system() == 'Windows' else 4


# ============================================================================
# 数据集：保持与你原始代码兼容，只是训练时忽略 x1d
# ============================================================================
class PHMDataset(Dataset):
    def __init__(self, index_df, stats, split='train', augment=False,
                 time_mask_param=20, freq_mask_param=20):
        self.data = index_df[index_df['split'] == split].reset_index(drop=True)
        self.stats = stats
        self.split = split
        self.augment = augment and (split == 'train')
        self.tm = time_mask_param
        self.fm = freq_mask_param

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 保留 1D 读取，保证与原 index/data 格式完全兼容；Only2D 训练时不用。
        x1d = np.load(row['path_1d'])
        x1d = torch.FloatTensor(x1d).unsqueeze(0)

        x2d = np.load(row['path_2d'])
        x2d = (x2d - self.stats['mean_2d']) / (self.stats['std_2d'] + 1e-8)

        if self.augment:
            # SpecAugment 风格：轻量时间/频率遮挡
            if np.random.rand() < 0.3:
                _, _, T = x2d.shape
                t = np.random.randint(0, min(self.tm, T))
                if t > 0:
                    t0 = np.random.randint(0, T - t + 1)
                    x2d[:, :, t0:t0 + t] = 0

            if np.random.rand() < 0.3:
                _, Fr, _ = x2d.shape
                f = np.random.randint(0, min(self.fm, Fr))
                if f > 0:
                    f0 = np.random.randint(0, Fr - f + 1)
                    x2d[:, f0:f0 + f, :] = 0

            # 轻微强度扰动，增强跨工况鲁棒性
            if np.random.rand() < 0.3:
                x2d = x2d * np.random.uniform(0.95, 1.05)

        x2d = torch.FloatTensor(x2d)
        label = torch.LongTensor([row['label']])[0]
        return x1d, x2d, label


# ============================================================================
# 公共组件
# ============================================================================
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(x.dtype)
        return x * mask / keep


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, F, T]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _init_conv(m):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _init_linear(m, gain=1.0):
    nn.init.xavier_uniform_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


# ============================================================================
# TF-SSM 2D Encoder
# ============================================================================
class SimpleLinearSSM(nn.Module):
    """
    轻量 axis-wise sequence mixer。
    论文写法可定位为：Vision SSM-inspired axis-wise temporal/frequency sequence modeling。

    输入:  [B, L, C]
    输出:  [B, L, C]
    """
    def __init__(self, dim, kernel_size=7, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        _init_linear(self.in_proj)
        _init_linear(self.out_proj, gain=0.5)
        _init_conv(self.dwconv)

    def forward(self, x):
        h = self.norm(x)
        u, v = self.in_proj(h).chunk(2, dim=-1)

        u = u.transpose(1, 2)
        u = self.dwconv(u)
        u = u.transpose(1, 2)

        h = u * torch.sigmoid(v)
        h = self.out_proj(h)
        return x + self.dropout(h)


class TFSSMBlock2D(nn.Module):
    """
    Time-Frequency Axis-wise Vision SSM Block

    x: [B, C, F, T]
    1) Local Conv 提取局部时频纹理
    2) Time-axis SSM: 每个频率 bin 上沿时间轴建模
    3) Freq-axis SSM: 每个时间片上沿频率轴建模
    4) Gated fusion
    5) Channel FFN
    """
    def __init__(self, dim, dropout=0.1, ssm_kernel=7,
                 mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.norm_local = LayerNorm2d(dim)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        _init_conv(self.local_conv[0])
        _init_conv(self.local_conv[1])

        self.time_ssm = SimpleLinearSSM(dim, kernel_size=ssm_kernel, dropout=dropout)
        self.freq_ssm = SimpleLinearSSM(dim, kernel_size=ssm_kernel, dropout=dropout)

        hidden_gate = max(dim // 4, 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_gate, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_gate, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.dp = DropPath(drop_path)

        self.norm_ffn = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        _init_conv(self.ffn[0])
        nn.init.xavier_uniform_(self.ffn[3].weight, gain=0.01)
        nn.init.zeros_(self.ffn[3].bias)

    def forward(self, x):
        B, C, Freq, Time = x.shape

        h = self.local_conv(self.norm_local(x))
        x = x + self.dp(h)

        # time-axis SSM: [B, C, F, T] -> [B*F, T, C]
        xt = x.permute(0, 2, 3, 1).reshape(B * Freq, Time, C)
        xt = self.time_ssm(xt)
        xt = xt.reshape(B, Freq, Time, C).permute(0, 3, 1, 2)

        # frequency-axis SSM: [B, C, F, T] -> [B*T, F, C]
        xf = x.permute(0, 3, 2, 1).reshape(B * Time, Freq, C)
        xf = self.freq_ssm(xf)
        xf = xf.reshape(B, Time, Freq, C).permute(0, 3, 2, 1)

        w = self.gate(x)  # [B, 2, 1, 1]
        h = w[:, 0:1] * xt + w[:, 1:2] * xf
        x = x + self.dp(h)

        x = x + self.dp(self.ffn(self.norm_ffn(x)))
        return x


class ConvStem2D(nn.Module):
    def __init__(self, in_ch=3, out_dim=64):
        super().__init__()
        mid = out_dim // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_dim, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(out_dim),
            nn.GELU(),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                _init_conv(m)

    def forward(self, x):
        return self.net(x)


class Downsample2D(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = LayerNorm2d(dim_in)
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)
        _init_conv(self.reduction)

    def forward(self, x):
        return self.reduction(self.norm(x))


class FrequencyAttentionPooling(nn.Module):
    """
    频率感知注意力池化：
    对最终时频 token 进行加权聚合，用于 only2D 分类。
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(
            nn.Linear(dim, max(dim // 4, 16)),
            nn.GELU(),
            nn.Linear(max(dim // 4, 16), 1),
        )
        _init_linear(self.score[0])
        _init_linear(self.score[2], gain=0.5)

    def forward(self, x):
        # x: [B, C, F, T]
        tok = x.flatten(2).transpose(1, 2)  # [B, F*T, C]
        tok = self.norm(tok)
        attn = self.score(tok)
        attn = torch.softmax(attn, dim=1)
        return (tok * attn).sum(dim=1)


class TFSSM2DEncoder(nn.Module):
    """
    Lightweight Time-Frequency Vision SSM Encoder.
    输出: z2 [B, out_dim]
    """
    def __init__(
        self,
        in_ch=3,
        dims=(64, 128, 256),
        depths=(2, 2, 2),
        dropout=0.15,
        drop_path=0.1,
        ssm_kernel=7,
        mlp_ratio=4,
        out_dim=256,
    ):
        super().__init__()
        C1, C2, C3 = dims
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        cur = 0

        self.stem = ConvStem2D(in_ch, C1)

        self.stage1 = nn.Sequential(*[
            TFSSMBlock2D(C1, dropout, ssm_kernel, mlp_ratio, dpr[cur + i])
            for i in range(depths[0])
        ])
        cur += depths[0]

        self.down1 = Downsample2D(C1, C2)

        self.stage2 = nn.Sequential(*[
            TFSSMBlock2D(C2, dropout, ssm_kernel, mlp_ratio, dpr[cur + i])
            for i in range(depths[1])
        ])
        cur += depths[1]

        self.down2 = Downsample2D(C2, C3)

        self.stage3 = nn.Sequential(*[
            TFSSMBlock2D(C3, dropout, ssm_kernel, mlp_ratio, dpr[cur + i])
            for i in range(depths[2])
        ])

        self.pool = FrequencyAttentionPooling(C3)
        self.proj = nn.Sequential(
            nn.LayerNorm(C3),
            nn.Linear(C3, out_dim),
        )
        _init_linear(self.proj[1], gain=0.5)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        x = self.down1(x)
        x = self.stage2(x)

        x = self.down2(x)
        x = self.stage3(x)

        z = self.pool(x)
        z = self.proj(z)
        return z


class Only2D_TFSSMNet(nn.Module):
    def __init__(
        self,
        num_classes=3,
        out_dim=256,
        dims=(64, 128, 256),
        depths=(2, 2, 2),
        dropout=0.15,
        drop_path=0.1,
        ssm_kernel=7,
        mlp_ratio=4,
    ):
        super().__init__()
        self.encoder_2d = TFSSM2DEncoder(
            in_ch=3,
            dims=dims,
            depths=depths,
            dropout=dropout,
            drop_path=drop_path,
            ssm_kernel=ssm_kernel,
            mlp_ratio=mlp_ratio,
            out_dim=out_dim,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m, gain=0.5)

    def forward(self, x2d):
        z2 = self.encoder_2d(x2d)
        logits = self.head(z2)
        return {'logits': logits, 'z2': z2}


# ============================================================================
# EMA
# ============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)


# ============================================================================
# Train / Val
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    ema=None, grad_clip=1.0):
    model.train()
    tot_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Train')

    for _, x2d, labels in pbar:
        x2d = x2d.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(x2d)
            loss = criterion(out['logits'], labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        tot_loss += loss.item() * labels.size(0)
        preds = out['logits'].argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}")

    return tot_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    tot = 0.0
    all_p, all_l = [], []

    for _, x2d, labels in tqdm(loader, desc='Val'):
        x2d = x2d.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(x2d)
        logits = out['logits']
        tot += criterion(logits, labels).item() * labels.size(0)

        all_p.extend(logits.argmax(1).cpu().numpy())
        all_l.extend(labels.cpu().numpy())

    return tot / len(all_l), compute_metrics(all_l, all_p)


# ============================================================================
# Main experiment
# ============================================================================
def train_experiment(
    exp_name,
    data_root='./data',
    output_root='./outputs_only2d_tfssm',
    seed=42,
    batch_size=16,
    epochs=60,
    lr=2e-4,
    patience=20,
    use_amp=True,
    loss_type='weighted_ce',
    severe_weight=2.0,
    dims=(64, 128, 256),
    depths=(2, 2, 2),
    out_dim=256,
    dropout=0.15,
    drop_path=0.1,
    ssm_kernel=7,
    mlp_ratio=4,
    grad_clip=1.0,
    use_ema=True,
    ema_decay=0.9999,
    warmup_epochs=3,
):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_dir = os.path.join(output_root, exp_name)
    ensure_dir(exp_dir)

    index_df = pd.read_csv(os.path.join(data_root, 'indexes', f'{exp_name}_index.csv'))
    stats = dict(np.load(os.path.join(data_root, 'stats', f'{exp_name}_norm.npz')))

    train_ds = PHMDataset(index_df, stats, 'train', augment=True)
    val_ds = PHMDataset(index_df, stats, 'val', augment=False)

    train_labels = index_df[index_df['split'] == 'train']['label'].values
    cc = np.bincount(train_labels, minlength=3)
    sw = 1.0 / (cc + 1e-6)
    sw = sw / sw.sum() * len(sw)
    sampler = WeightedRandomSampler(sw[train_labels], len(train_labels), replacement=True)

    nw = get_num_workers()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=nw, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True)

    print(f"\n{'='*80}")
    print(f"  Only2D TF-SSMNet | {exp_name}")
    print(f"  2D backbone: ConvStem + Axis-wise Time/Frequency SSM + Attention Pooling")
    print(f"  dims={dims}, depths={depths}, out_dim={out_dim}, ssm_kernel={ssm_kernel}")
    print(f"  EMA={'on' if use_ema else 'off'}, AMP={'on' if (use_amp and device.type=='cuda') else 'off'}")
    print(f"  训练: {len(train_ds)}, 验证: {len(val_ds)}, 类别: {cc.tolist()}")
    print(f"{'='*80}\n")

    model = Only2D_TFSSMNet(
        num_classes=3,
        out_dim=out_dim,
        dims=dims,
        depths=depths,
        dropout=dropout,
        drop_path=drop_path,
        ssm_kernel=ssm_kernel,
        mlp_ratio=mlp_ratio,
    ).to(device)

    print(f"  参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    cw = np.array([1.0, 1.0, severe_weight])
    if loss_type == 'weighted_ce':
        criterion = WeightedCrossEntropyLoss(cw).to(device)
    elif loss_type == 'focal':
        criterion = FocalLoss(cw, gamma=2).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs)

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    hk = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_macro_f1',
          'val_initial_precision', 'val_initial_recall', 'val_initial_f1',
          'val_normal_precision', 'val_normal_recall', 'val_normal_f1',
          'val_severe_precision', 'val_severe_recall', 'val_severe_f1',
          'val_normal_to_severe', 'val_normal_to_initial',
          'val_severe_to_normal', 'val_initial_to_normal']
    history = {k: [] for k in hk}

    best_f1 = 0.0
    best_ep = 0
    pat_cnt = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        tl, ta = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            ema=ema, grad_clip=grad_clip)

        eval_model = ema.ema if ema is not None else model
        vl, vm = validate(eval_model, val_loader, criterion, device)
        scheduler.step()

        va = 100.0 * np.trace(vm['confusion_matrix']) / vm['confusion_matrix'].sum()

        for k in hk:
            if k == 'train_loss':
                history[k].append(float(tl))
            elif k == 'train_acc':
                history[k].append(float(ta))
            elif k == 'val_loss':
                history[k].append(float(vl))
            elif k == 'val_acc':
                history[k].append(float(va))
            else:
                key2 = k.replace('val_', '')
                v = vm.get(key2, 0)
                history[k].append(
                    float(v) if isinstance(v, (float, np.floating))
                    else int(v) if isinstance(v, (int, np.integer))
                    else float(v)
                )

        print(f"  Train L={tl:.4f} A={ta:.1f}% | Val L={vl:.4f} A={va:.1f}% "
              f"| Macro-F1={vm['macro_f1']:.4f} Normal-F1={vm['normal_f1']:.4f}")

        if vm['macro_f1'] > best_f1:
            best_f1 = vm['macro_f1']
            best_ep = epoch
            pat_cnt = 0
            save_model = ema.ema if ema is not None else model

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'macro_f1': vm['macro_f1'],
                    'val_metrics': vm,
                    'config': {
                        'model_name': 'Only2D_TFSSMNet',
                        'backbone_2d': 'tf_axiswise_vision_ssm',
                        'dims': list(dims),
                        'depths': list(depths),
                        'out_dim': out_dim,
                        'dropout': dropout,
                        'drop_path': drop_path,
                        'ssm_kernel': ssm_kernel,
                        'mlp_ratio': mlp_ratio,
                    },
                },
                os.path.join(exp_dir, 'best_model.pth')
            )
            print(f"  ★ 新最佳 Macro-F1={vm['macro_f1']:.4f}")
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                print(f"  早停 @ epoch {epoch}")
                break

    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    plot_curves(history, exp_dir)

    ckpt = torch.load(os.path.join(exp_dir, 'best_model.pth'),
                      map_location=device, weights_only=False)
    cfg = ckpt['config']

    eval_model_final = Only2D_TFSSMNet(
        num_classes=3,
        out_dim=cfg.get('out_dim', 256),
        dims=tuple(cfg.get('dims', [64, 128, 256])),
        depths=tuple(cfg.get('depths', [2, 2, 2])),
        dropout=cfg.get('dropout', 0.15),
        drop_path=0.0,
        ssm_kernel=cfg.get('ssm_kernel', 7),
        mlp_ratio=cfg.get('mlp_ratio', 4),
    ).to(device)
    eval_model_final.load_state_dict(ckpt['model_state_dict'])

    _, vm = validate(eval_model_final, val_loader, criterion, device)
    print(f"\n{'='*80}")
    print(f"最佳模型 (ep {best_ep}):\n{vm['report']}")
    analyze_confusion_matrix(vm['confusion_matrix'])
    plot_confusion_matrix(
        vm['confusion_matrix'], ['Initial', 'Normal', 'Severe'],
        os.path.join(exp_dir, 'confusion_matrix_val.png'))
    print(f"完成: Macro-F1={best_f1:.4f}\n{'='*80}")


# ============================================================================
# CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(description='Only2D TF-SSMNet 训练')
    p.add_argument('--exp', default='C4C6_to_C1',
                   choices=['C1C4_to_C6', 'C1C6_to_C4', 'C4C6_to_C1'])
    p.add_argument('--data_root', default='./data')
    p.add_argument('--output_root', default='./outputs_only2d_tfssm')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--no_amp', dest='use_amp', action='store_false')
    p.set_defaults(use_amp=True)

    p.add_argument('--loss_type', default='weighted_ce',
                   choices=['ce', 'weighted_ce', 'focal'])
    p.add_argument('--severe_weight', type=float, default=2.0)

    p.add_argument('--dims', type=int, nargs=3, default=[64, 128, 256])
    p.add_argument('--depths', type=int, nargs=3, default=[2, 2, 2])
    p.add_argument('--out_dim', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--drop_path', type=float, default=0.1)
    p.add_argument('--ssm_kernel', type=int, default=7)
    p.add_argument('--mlp_ratio', type=int, default=4)

    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--warmup_epochs', type=int, default=3)
    p.add_argument('--use_ema', action='store_true', default=True)
    p.add_argument('--no_ema', dest='use_ema', action='store_false')
    p.add_argument('--ema_decay', type=float, default=0.9999)

    a = p.parse_args()

    train_experiment(
        exp_name=a.exp,
        data_root=a.data_root,
        output_root=a.output_root,
        seed=a.seed,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        patience=a.patience,
        use_amp=a.use_amp,
        loss_type=a.loss_type,
        severe_weight=a.severe_weight,
        dims=tuple(a.dims),
        depths=tuple(a.depths),
        out_dim=a.out_dim,
        dropout=a.dropout,
        drop_path=a.drop_path,
        ssm_kernel=a.ssm_kernel,
        mlp_ratio=a.mlp_ratio,
        grad_clip=a.grad_clip,
        use_ema=a.use_ema,
        ema_decay=a.ema_decay,
        warmup_epochs=a.warmup_epochs,
    )


if __name__ == '__main__':
    main()
