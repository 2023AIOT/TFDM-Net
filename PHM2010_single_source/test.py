"""
test.py — TFDM-Net 测试脚本
"""

import os
import json
import argparse


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


CLASS_NAMES = ['Initial', 'Normal', 'Severe']


def metric_to_jsonable(metrics):
    """
    Convert metric dictionary to JSON-serializable format.
    """
    out = {}
    for k, v in metrics.items():
        if k in ('confusion_matrix', 'report'):
            continue
        if isinstance(v, (float, np.floating)):
            out[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def get_accuracy_from_cm(cm):
    """
    Compute accuracy from confusion matrix.
    """
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)



def print_metric_summary(title, metrics, num_samples, loss=None):
    """
    Print formatted metric summary.
    """
    print(f"\n{'=' * 80}")
    if loss is None:
        print(f"{title}")
    else:
        print(f"{title} | Test Loss={loss:.4f}")
    print(f"Samples: {num_samples}")
    print(f"Accuracy: {100.0 * get_accuracy_from_cm(metrics['confusion_matrix']):.2f}%")
    print(f"{'=' * 80}")
    print(metrics['report'])

    analyze_confusion_matrix(metrics['confusion_matrix'])

    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Initial — P={metrics['initial_precision']:.4f} "
          f"R={metrics['initial_recall']:.4f} "
          f"F1={metrics['initial_f1']:.4f}")
    print(f"  Normal  — P={metrics['normal_precision']:.4f} "
          f"R={metrics['normal_recall']:.4f} "
          f"F1={metrics['normal_f1']:.4f}")
    print(f"  Severe  — P={metrics['severe_precision']:.4f} "
          f"R={metrics['severe_recall']:.4f} "
          f"F1={metrics['severe_f1']:.4f}")
    print(f"  N->S={metrics['normal_to_severe']}  "
          f"S->N={metrics['severe_to_normal']}")


def test_experiment(
    exp_name,
    data_exp=None,
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

    print(f"\n{'=' * 80}")
    print(f"  Only2D TF-SSMNet 测试: {exp_name} (best epoch {ckpt['epoch']})")
    print(f"  2D backbone: {cfg.get('backbone_2d', 'tf_axiswise_vision_ssm')}")
    print(f"  Val Macro-F1: {ckpt['macro_f1']:.4f}")
    print(f"{'=' * 80}")

    data_exp = data_exp or ckpt.get('config', {}).get('data_exp', exp_name)

    index_path = os.path.join(data_root, 'indexes', f'{data_exp}_index.csv')

    index_df = pd.read_csv(index_path)

    # 测试时优先使用 checkpoint 训练时的 stats。
    # 如果你按上面 train.py 重算了 stats，建议把 stats 也保存进 ckpt。
    if 'stats' in ckpt:
        stats = ckpt['stats']
    else:
        stats_path = os.path.join(data_root, 'stats', f'{data_exp}_norm.npz')
        stats = dict(np.load(stats_path))

    test_ds = PHMDataset(index_df, stats, 'test', augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_num_workers(),
        pin_memory=True
    )

    # Important:
    # Your PHMDataset stores the filtered dataframe in self.data.
    # Since test_loader uses shuffle=False, sample order is exactly aligned with test_ds.data.
    if not hasattr(test_ds, 'data'):
        raise AttributeError(
            "PHMDataset does not have attribute 'data'. "
            "Please check the dataset implementation."
        )

    test_meta = test_ds.data.reset_index(drop=True).copy()

    required_cols = {'tool_id', 'cut_id', 'seg_id', 'label'}
    missing_cols = required_cols - set(test_meta.columns)
    if missing_cols:
        raise KeyError(f"测试集 metadata 缺少字段: {missing_cols}")

    if len(test_meta) != len(test_ds):
        raise RuntimeError(
            f"Metadata length mismatch: len(test_meta)={len(test_meta)}, "
            f"len(test_ds)={len(test_ds)}"
        )

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
    print(f"  参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = WeightedCrossEntropyLoss(np.array([1.0, 1.0, 2.0])).to(device)

    all_preds = []
    all_labels = []

    test_loss = 0.0
    sample_offset = 0

    with torch.no_grad():
        for _, x2d, labels in tqdm(test_loader, desc='Testing'):
            x2d = x2d.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(x2d)
            logits = out['logits']

            loss = criterion(logits, labels)
            test_loss += loss.item() * labels.size(0)

            preds_np = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            batch_n = labels.size(0)

            for i in range(batch_n):
                meta = test_meta.iloc[sample_offset + i]

                label_i = int(labels_np[i])
                pred_i = int(preds_np[i])


                # Optional safety check: dataset label should match metadata label.
                meta_label = int(meta['label'])
                if label_i != meta_label:
                    raise RuntimeError(
                        f"Label mismatch at sample {sample_offset + i}: "
                        f"dataset label={label_i}, metadata label={meta_label}"
                    )

                all_labels.append(label_i)
                all_preds.append(pred_i)




            sample_offset += batch_n

    if sample_offset != len(test_meta):
        raise RuntimeError(
            f"Processed sample number mismatch: processed={sample_offset}, "
            f"metadata={len(test_meta)}"
        )

    test_loss /= len(all_labels)

    # =========================================================================
    # 1. Segment-level evaluation
    # =========================================================================
    segment_metrics = compute_metrics(all_labels, all_preds)

    print_metric_summary(
        title="Segment-level 测试结果",
        metrics=segment_metrics,
        num_samples=len(all_labels),
        loss=test_loss,
    )


    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        'exp_name': exp_name,
        'best_epoch': int(ckpt['epoch']),
        'model_name': cfg.get('model_name', 'Only2D_TFSSMNet'),
        'backbone_2d': cfg.get('backbone_2d', 'tf_axiswise_vision_ssm'),

        'segment_level': {
            'test_samples': int(len(all_labels)),
            'test_loss': float(test_loss),
            'accuracy': get_accuracy_from_cm(
                segment_metrics['confusion_matrix']
            ),
            'metrics': metric_to_jsonable(segment_metrics),
            'confusion_matrix':
                segment_metrics['confusion_matrix'].tolist(),
        },
    }

    json_path = os.path.join(exp_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save confusion matrices separately.
    plot_confusion_matrix(
        segment_metrics['confusion_matrix'],
        CLASS_NAMES,
        os.path.join(exp_dir, 'confusion_matrix_test.png')
    )

    print(f"\n  ✓ 结果已保存至 {exp_dir}")
    print(f"  ✓ JSON: {json_path}")
    print(f"  ✓ Segment-level CM: confusion_matrix_test.png")
    print(f"{'=' * 80}\n")


def main():
    p = argparse.ArgumentParser(description='Only2D TF-SSMNet 测试')
    p.add_argument('--exp', default='C1_to_C4')
    p.add_argument('--data_exp', default='C1C6_to_C4',
                   choices=['C1C4_to_C6', 'C1C6_to_C4', 'C4C6_to_C1'])
    p.add_argument('--data_root', default='./data')
    p.add_argument('--output_root', default='./outputs_only2d_tfssm')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=32)
    a = p.parse_args()

    test_experiment(
        exp_name=a.exp,
        data_exp=a.data_exp,
        data_root=a.data_root,
        output_root=a.output_root,
        seed=a.seed,
        batch_size=a.batch_size,
    )


if __name__ == '__main__':
    main()