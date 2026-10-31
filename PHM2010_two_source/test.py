"""
test.py — TFDM-Net 测试脚本

读取 outputs_only2d_tfssm/<exp>/best_model.pth，
只使用 2D CWT 时频图进行测试。
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import PHMDataset, Only2D_TFSSMNet, get_num_workers
from utils import (
    set_seed, compute_metrics, plot_confusion_matrix,
    analyze_confusion_matrix, WeightedCrossEntropyLoss
)


def test_experiment(
    exp_name,
    data_root='./data',
    output_root='./outputs_only2d_tfssm',
    seed=42,
    batch_size=32,
):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(output_root, exp_name)

    ckpt_path = os.path.join(exp_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})

    print(f"\n{'='*80}")
    print(f"  Only2D TF-SSMNet 测试: {exp_name} (best epoch {ckpt['epoch']})")
    print(f"  2D backbone: {cfg.get('backbone_2d', 'tf_axiswise_vision_ssm')}")
    print(f"  Val Macro-F1: {ckpt['macro_f1']:.4f}")
    print(f"{'='*80}")

    index_df = pd.read_csv(os.path.join(data_root, 'indexes', f'{exp_name}_index.csv'))
    stats = dict(np.load(os.path.join(data_root, 'stats', f'{exp_name}_norm.npz')))

    test_ds = PHMDataset(index_df, stats, 'test', augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=get_num_workers(), pin_memory=True)

    model = Only2D_TFSSMNet(
        num_classes=3,
        out_dim=cfg.get('out_dim', 256),
        dims=tuple(cfg.get('dims', [64, 128, 256])),
        depths=tuple(cfg.get('depths', [2, 2, 2])),
        dropout=cfg.get('dropout', 0.15),
        drop_path=0.0,
        ssm_kernel=cfg.get('ssm_kernel', 7),
        mlp_ratio=cfg.get('mlp_ratio', 4),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("  ✓ 模型加载完成")
    print(f"  参数总量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    criterion = WeightedCrossEntropyLoss(np.array([1.0, 1.0, 2.0])).to(device)

    all_preds, all_labels = [], []
    test_loss = 0.0

    with torch.no_grad():
        for _, x2d, labels in tqdm(test_loader, desc='Testing'):
            x2d = x2d.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(x2d)
            logits = out['logits']
            test_loss += criterion(logits, labels).item() * labels.size(0)

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(all_labels)
    m = compute_metrics(all_labels, all_preds)

    print(f"\n{'='*80}\n测试结果 | Test Loss={test_loss:.4f}\n{'='*80}\n{m['report']}")
    analyze_confusion_matrix(m['confusion_matrix'])

    print(f"  Macro-F1: {m['macro_f1']:.4f}")
    print(f"  Normal  — P={m['normal_precision']:.4f} R={m['normal_recall']:.4f} "
          f"F1={m['normal_f1']:.4f}")
    print(f"  Severe  — P={m['severe_precision']:.4f} R={m['severe_recall']:.4f} "
          f"F1={m['severe_f1']:.4f}")
    print(f"  N->S={m['normal_to_severe']}  S->N={m['severe_to_normal']}")

    results = {
        'exp_name': exp_name,
        'test_samples': len(all_labels),
        'best_epoch': int(ckpt['epoch']),
        'model_name': cfg.get('model_name', 'Only2D_TFSSMNet'),
        'backbone_2d': cfg.get('backbone_2d', 'tf_axiswise_vision_ssm'),
        'test_loss': float(test_loss),
        'metrics': {
            k: (float(v) if isinstance(v, (float, np.floating))
                else int(v) if isinstance(v, (int, np.integer))
                else v)
            for k, v in m.items()
            if k not in ('confusion_matrix', 'report')
        },
        'confusion_matrix': m['confusion_matrix'].tolist(),
    }

    with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrix(
        m['confusion_matrix'], ['Initial', 'Normal', 'Severe'],
        os.path.join(exp_dir, 'confusion_matrix_test.png'))

    print(f"\n  ✓ 结果已保存至 {exp_dir}\n{'='*80}\n")


def main():
    p = argparse.ArgumentParser(description='Only2D TF-SSMNet 测试')
    p.add_argument('--exp', default='C4C6_to_C1',
                   choices=['C1C4_to_C6', 'C1C6_to_C4', 'C4C6_to_C1'])
    p.add_argument('--data_root', default='./data')
    p.add_argument('--output_root', default='./outputs_only2d_tfssm')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=32)
    a = p.parse_args()

    test_experiment(
        exp_name=a.exp,
        data_root=a.data_root,
        output_root=a.output_root,
        seed=a.seed,
        batch_size=a.batch_size,
    )


if __name__ == '__main__':
    main()
