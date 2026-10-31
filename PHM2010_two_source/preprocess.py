"""
PHM2010 data preprocessing.

"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from utils import set_seed, ensure_dir, compute_cwt_3channel


def load_cut_signal(data_root, tool_id, cut_id):
    path = Path(data_root) / f"c{tool_id}" / f"c_{tool_id}_{cut_id:03d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(), df.iloc[:, 2].to_numpy()


def load_wear_table(data_root, tool_id):
    path = Path(data_root) / f"c{tool_id}_wear.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    wear = {}
    for _, row in df.iterrows():
        cut = int(row['cut'])
        wear[cut] = (row['flute_1'] + row['flute_2'] + row['flute_3']) / 3.0
    return wear


def get_label(cut_id):
    if 1 <= cut_id <= 50:
        return 0
    elif 51 <= cut_id <= 175:
        return 1
    elif 176 <= cut_id <= 315:
        return 2
    raise ValueError(f"Invalid cut_id: {cut_id}")


def extract_stable_window(Fx, Fy, Fz, s=90000, e=120000):
    n = len(Fx)
    if n < e:
        s, e = int(n * 0.2), int(n * 0.8)
    else:
        e = min(e, n)
    return Fx[s:e], Fy[s:e], Fz[s:e]


def segment_signal_xyz(Fx, Fy, Fz, n_seg=5, seg_len=2048):
    n = len(Fx)
    if n < seg_len:
        raise ValueError(f"Signal too short: {n}")
    max_s = n - seg_len
    starts = np.linspace(0, max_s, n_seg, dtype=int) if n_seg > 1 else [max_s // 2]
    s1d, sfx, sfy, sfz = [], [], [], []
    for st in starts:
        fx = Fx[st:st + seg_len]
        fy = Fy[st:st + seg_len]
        fz = Fz[st:st + seg_len]
        res = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
        res = (res - res.mean()) / (res.std() + 1e-6)
        s1d.append(res)
        sfx.append(fx)
        sfy.append(fy)
        sfz.append(fz)
    return s1d, sfx, sfy, sfz


def split_pass_ids(ids, ratio=0.7, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.array(ids)
    rng.shuffle(ids)
    n = int(len(ids) * ratio)
    return sorted(ids[:n].tolist()), sorted(ids[n:].tolist())


def preprocess_experiment(
    data_root, exp_name, train_tools, test_tool,
    output_root='./data', seed=42, n_passes=315,
    n_segments=5, segment_length=2048,
    stable_start=90000, stable_end=120000,
    cwt_scales=224, target_size=224, force_overwrite=False,
):
    set_seed(seed)

    dirs = {k: os.path.join(output_root, k) for k in
            ['splits', 'indexes', 'stats', 'cache']}
    cache_dir = dirs['cache']

    if os.path.exists(cache_dir) and os.listdir(cache_dir) and not force_overwrite:
        raise RuntimeError(
            f"Cache exists: {cache_dir}\nUse --force_overwrite to overwrite.")
    for d in dirs.values():
        ensure_dir(d)

    print(f"\n{'='*80}\nPreprocessing: {exp_name}")
    print(f"Train: {train_tools}  Test: {test_tool}\n{'='*80}\n")

    splits = {'train': {}, 'val': {}, 'test': {}}
    for t in train_tools:
        tr, va = split_pass_ids(list(range(1, n_passes + 1)), seed=seed)
        splits['train'][f'C{t}'] = tr
        splits['val'][f'C{t}'] = va
    splits['test'][f'C{test_tool}'] = list(range(1, n_passes + 1))

    with open(os.path.join(dirs['splits'], f'{exp_name}.json'), 'w') as f:
        json.dump(splits, f, indent=2)

    index_data = []
    train_2d = []

    for tool in train_tools + [test_tool]:
        tc = os.path.join(cache_dir, f'C{tool}')
        ensure_dir(tc)
        try:
            wear = load_wear_table(data_root, tool)
        except Exception:
            wear = {}

        for cut in tqdm(range(1, n_passes + 1), desc=f'C{tool}'):
            try:
                Fx, Fy, Fz = load_cut_signal(data_root, tool, cut)
                Fxs, Fys, Fzs = extract_stable_window(Fx, Fy, Fz,
                                                       stable_start, stable_end)
                s1d, sfx, sfy, sfz = segment_signal_xyz(
                    Fxs, Fys, Fzs, n_segments, segment_length)
                label = get_label(cut)

                if tool in train_tools:
                    sp = 'train' if cut in splits['train'][f'C{tool}'] else 'val'
                else:
                    sp = 'test'

                for si in range(n_segments):
                    p1d = os.path.join(tc, f'cut{cut:03d}_seg{si}_1d.npy')
                    np.save(p1d, s1d[si].astype(np.float32))

                    cwt = compute_cwt_3channel(sfx[si], sfy[si], sfz[si],
                                               cwt_scales, target_size=target_size)
                    p2d = os.path.join(tc, f'cut{cut:03d}_seg{si}_2d.npy')
                    np.save(p2d, cwt.astype(np.float32))

                    index_data.append({
                        'tool_id': f'C{tool}', 'cut_id': int(cut),
                        'seg_id': int(si), 'label': int(label),
                        'split': sp, 'path_1d': p1d, 'path_2d': p2d,
                        'wear_mean': wear.get(cut, -1),
                    })
                    if sp == 'train':
                        train_2d.append(cwt)
            except Exception as e:
                print(f"Error C{tool} cut{cut}: {e}")

    idf = pd.DataFrame(index_data)
    idf.to_csv(os.path.join(dirs['indexes'], f'{exp_name}_index.csv'), index=False)
    print(f"\n✓ Total samples: {len(idf)}")
    print(idf.groupby(['split', 'label']).size())

    if train_2d:
        arr = np.array(train_2d)
        stats = {'mean_2d': float(arr.mean()), 'std_2d': float(arr.std() + 1e-8),
                 'mean_1d': 0.0, 'std_1d': 1.0}
        np.savez(os.path.join(dirs['stats'], f'{exp_name}_norm.npz'), **stats)
        print(f"✓ Stats saved (2D mean={stats['mean_2d']:.6f})")

    print(f"\n{'='*80}\nDone: {exp_name}\n{'='*80}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=r'D:\PHM2010')
    p.add_argument('--output_root', default='./data')
    p.add_argument('--exp', default='C4C6_to_C1',
                   choices=['C1C4_to_C6', 'C1C6_to_C4', 'C4C6_to_C1'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_passes', type=int, default=315)
    p.add_argument('--n_segments', type=int, default=5)
    p.add_argument('--segment_length', type=int, default=2048)
    p.add_argument('--stable_start', type=int, default=90000)
    p.add_argument('--stable_end', type=int, default=120000)
    p.add_argument('--cwt_scales', type=int, default=224)
    p.add_argument('--target_size', type=int, default=224)
    p.add_argument('--force_overwrite', action='store_true')
    args = p.parse_args()

    cfgs = {'C1C4_to_C6': ([1, 4], 6), 'C1C6_to_C4': ([1, 6], 4),
            'C4C6_to_C1': ([4, 6], 1)}
    tr, te = cfgs[args.exp]

    preprocess_experiment(
        args.data_root, args.exp, tr, te,
        args.output_root, args.seed, args.n_passes,
        args.n_segments, args.segment_length,
        args.stable_start, args.stable_end,
        args.cwt_scales, args.target_size, args.force_overwrite,
    )


if __name__ == '__main__':
    main()
