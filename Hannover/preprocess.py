"""
Mendeley Milling H5 data preprocessing for experiments.

"""

import os
import re
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import set_seed, ensure_dir, compute_cwt_3channel


# -----------------------------------------------------------------------------
# Filename / metadata parsing
# Example: M2T6R30C170VB45.h5
# -----------------------------------------------------------------------------
FNAME_RE = re.compile(
    r"M(?P<machine>\d+)T(?P<tool>\d+)R(?P<run>\d+)C(?P<contact>\d+)VB(?P<wear>\d+)",
    re.IGNORECASE,
)


def _scalar(x):
    """Convert HDF5 scalar/array/bytes to a Python scalar when possible."""
    arr = np.asarray(x)
    if arr.shape == ():
        val = arr.item()
    else:
        val = arr.reshape(-1)[0]
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    return val


def _find_dataset_by_name(h5obj, target_name):
    """Fallback: recursively find a dataset whose basename equals target_name."""
    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and Path(name).name == target_name:
            found.append(name)

    h5obj.visititems(visitor)
    return found[0] if found else None


def _read_h5_dataset(h5obj, candidate_paths, fallback_name=None):
    """Read a dataset by common paths, with recursive fallback by basename."""
    for p in candidate_paths:
        if p in h5obj:
            return np.asarray(h5obj[p]).squeeze()
    if fallback_name is not None:
        hit = _find_dataset_by_name(h5obj, fallback_name)
        if hit is not None:
            return np.asarray(h5obj[hit]).squeeze()
    raise KeyError(f"Cannot find dataset. tried={candidate_paths}, fallback={fallback_name}")


def parse_meta_from_filename(path):
    """Parse M/T/R/C/VB from filename."""
    m = FNAME_RE.search(Path(path).stem)
    if not m:
        return None
    return {
        "machine": int(m.group("machine")),
        "tool": int(m.group("tool")),
        "run": int(m.group("run")),          # R
        "contact": int(m.group("contact")),  # C
        "wear": float(m.group("wear")),      # VB
        "filename": Path(path).name,
    }


def read_meta_from_h5(path):
    """Read labels from HDF5 if filename parsing is not enough."""
    with h5py.File(path, "r") as f:
        labels = {}
        mapping = {
            "machine": (["labels/machine", "/labels/machine", "machine"], "machine"),
            "tool": (["labels/tool", "/labels/tool", "tool"], "tool"),
            "run": (["labels/run", "/labels/run", "run"], "run"),
            "contact": ([
                "labels/cumulated_tool_contact_time",
                "/labels/cumulated_tool_contact_time",
                "cumulated_tool_contact_time",
            ], "cumulated_tool_contact_time"),
            "wear": (["labels/wear", "/labels/wear", "wear"], "wear"),
        }
        for key, (paths, fb) in mapping.items():
            labels[key] = _scalar(_read_h5_dataset(f, paths, fb))

    return {
        "machine": int(labels["machine"]),
        "tool": int(labels["tool"]),
        "run": int(labels["run"]),
        "contact": int(float(labels["contact"])),
        "wear": float(labels["wear"]),
        "filename": Path(path).name,
    }


def collect_h5_records(data_root, tools=(4, 5, 6), machine=None):
    """
    Collect all .h5 files belonging to selected tools.
    Metadata is primarily parsed from filename, matching the dataset convention.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    paths = sorted(root.rglob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files found under: {root}")

    records = []
    tools = set(int(t) for t in tools)
    machine = None if machine is None else int(machine)

    for p in paths:
        meta = parse_meta_from_filename(p)
        if meta is None:
            try:
                meta = read_meta_from_h5(p)
            except Exception:
                continue

        if meta["tool"] not in tools:
            continue
        if machine is not None and meta["machine"] != machine:
            continue

        meta["path"] = str(p)
        records.append(meta)

    records = sorted(records, key=lambda r: (r["machine"], r["tool"], r["run"], r["contact"]))
    if not records:
        raise RuntimeError(f"No H5 records found for tools={sorted(tools)}, machine={machine}")
    return records


# -----------------------------------------------------------------------------
# Signal loading
# -----------------------------------------------------------------------------
def load_cut_signal(data_root, tool_id=None, cut_id=None, h5_path=None):
    """
    Read force_sensor_x/y/z from an HDF5 sample file.

    The signature keeps compatibility with the old PHM-style code, but for this
    dataset h5_path is used directly.
    """
    if h5_path is None:
        raise ValueError("For the Mendeley H5 dataset, please provide h5_path.")

    with h5py.File(h5_path, "r") as f:
        fx = _read_h5_dataset(
            f,
            ["signals_sensor/force_sensor_x", "/signals_sensor/force_sensor_x", "force_sensor_x"],
            "force_sensor_x",
        )
        fy = _read_h5_dataset(
            f,
            ["signals_sensor/force_sensor_y", "/signals_sensor/force_sensor_y", "force_sensor_y"],
            "force_sensor_y",
        )
        fz = _read_h5_dataset(
            f,
            ["signals_sensor/force_sensor_z", "/signals_sensor/force_sensor_z", "force_sensor_z"],
            "force_sensor_z",
        )

    fx = np.asarray(fx, dtype=np.float32).reshape(-1)
    fy = np.asarray(fy, dtype=np.float32).reshape(-1)
    fz = np.asarray(fz, dtype=np.float32).reshape(-1)

    n = min(len(fx), len(fy), len(fz))
    if n <= 0:
        raise ValueError(f"Empty force signal: {h5_path}")

    fx = np.nan_to_num(fx[:n], nan=0.0, posinf=0.0, neginf=0.0)
    fy = np.nan_to_num(fy[:n], nan=0.0, posinf=0.0, neginf=0.0)
    fz = np.nan_to_num(fz[:n], nan=0.0, posinf=0.0, neginf=0.0)
    return fx, fy, fz


def extract_stable_window(Fx, Fy, Fz, s=-1, e=-1, ratio_start=0.20, ratio_end=0.80):
    """
    Extract the stable cutting part.

    For this dataset, each H5 file is already a single shoulder milling run.
    The middle 20%-80% is used by default to avoid tool entry/exit transients.

    If s/e are valid positive indices and e <= signal length, fixed slicing is used.
    """
    n = min(len(Fx), len(Fy), len(Fz))
    Fx, Fy, Fz = Fx[:n], Fy[:n], Fz[:n]

    if s is not None and e is not None and s >= 0 and e > s and e <= n:
        ss, ee = int(s), int(e)
    else:
        ss = int(n * ratio_start)
        ee = int(n * ratio_end)

    if ee - ss < 10:
        ss, ee = 0, n
    return Fx[ss:ee], Fy[ss:ee], Fz[ss:ee]


def segment_signal_xyz(Fx, Fy, Fz, n_seg=5, seg_len=2048):
    """
    Split one run into n_seg deterministic segments.

    path_1d stores the resultant force, normalized per segment.
    path_2d stores the 3-channel CWT image computed from Fx/Fy/Fz.
    """
    n = min(len(Fx), len(Fy), len(Fz))
    if n < seg_len:
        raise ValueError(f"Signal too short: {n}, required segment_length={seg_len}")

    max_s = n - seg_len
    starts = np.linspace(0, max_s, n_seg, dtype=int) if n_seg > 1 else np.array([max_s // 2])

    s1d, sfx, sfy, sfz = [], [], [], []
    for st in starts:
        fx = Fx[st:st + seg_len].astype(np.float32)
        fy = Fy[st:st + seg_len].astype(np.float32)
        fz = Fz[st:st + seg_len].astype(np.float32)

        res = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
        res = (res - res.mean()) / (res.std() + 1e-6)

        s1d.append(res.astype(np.float32))
        sfx.append(fx)
        sfy.append(fy)
        sfz.append(fz)
    return s1d, sfx, sfy, sfz


# -----------------------------------------------------------------------------
# Labels and split
# -----------------------------------------------------------------------------
def get_label_from_run(run, r_thr1=300, r_thr2=700.0):
    """
    3-class tool wear state label based on run index R.

    Initial: R < r_thr1
    Normal : r_thr1 <= R < r_thr2
    Severe : R >= r_thr2
    """
    run = float(run)
    if run < r_thr1:
        return 0          # Initial
    elif run < r_thr2:
        return 1          # Normal
    else:
        return 2          # Severe


def split_train_val_records(records, val_ratio=0.20):
    """
    Record-level train/val split on training tools only.

    The split is stratified by tool and wear class. Within each stratum, records
    are sorted by run R and cumulated contact time C, and validation records are
    selected evenly along the degradation trajectory. This avoids segment-level
    leakage: all segments generated from the same H5 run stay in the same split.
    """
    train, val = [], []
    df = pd.DataFrame(records)
    if len(df) == 0:
        return train, val

    for (_, _), g in df.groupby(["tool", "label"], sort=True):
        g = g.sort_values(["run", "contact"]).reset_index(drop=True)
        n = len(g)
        if n <= 1 or val_ratio <= 0:
            train.extend(g.to_dict("records"))
            continue

        n_val = int(round(n * val_ratio))
        n_val = max(1, min(n_val, n - 1))
        val_idx = set(np.linspace(0, n - 1, n_val, dtype=int).tolist())

        for i, row in g.iterrows():
            item = row.to_dict()
            if i in val_idx:
                val.append(item)
            else:
                train.append(item)

    train = sorted(train, key=lambda r: (r["tool"], r["run"], r["contact"]))
    val = sorted(val, key=lambda r: (r["tool"], r["run"], r["contact"]))
    return train, val


def _safe_stem(filename):
    return Path(filename).stem.replace(" ", "_")


def _update_stats(cwt, state):
    arr = cwt.astype(np.float64, copy=False)
    state["sum"] += float(arr.sum())
    state["sumsq"] += float((arr * arr).sum())
    state["count"] += int(arr.size)


def _finalize_stats(state):
    if state["count"] <= 0:
        return {"mean_2d": 0.0, "std_2d": 1.0, "mean_1d": 0.0, "std_1d": 1.0}
    mean = state["sum"] / state["count"]
    var = max(state["sumsq"] / state["count"] - mean * mean, 1e-12)
    std = float(np.sqrt(var) + 1e-8)
    return {"mean_2d": float(mean), "std_2d": std, "mean_1d": 0.0, "std_1d": 1.0}


def preprocess_experiment(
    data_root,
    exp_name,
    train_tools,
    test_tool,
    output_root="./data",
    seed=42,
    machine=2,
    n_segments=5,
    segment_length=2048,
    stable_start=-1,
    stable_end=-1,
    stable_ratio_start=0.20,
    stable_ratio_end=0.80,
    cwt_scales=224,
    target_size=224,
    val_ratio=0.20,
    r_thr1=300.0,
    r_thr2=700.0,
    force_overwrite=False,
):
    set_seed(seed)

    dirs = {k: os.path.join(output_root, k) for k in ["splits", "indexes", "stats", "cache"]}
    for d in dirs.values():
        ensure_dir(d)

    # Use an experiment-specific cache to avoid conflicts with PHM2010 or other folds.
    cache_dir = os.path.join(dirs["cache"], exp_name)
    if os.path.exists(cache_dir) and os.listdir(cache_dir) and not force_overwrite:
        raise RuntimeError(f"Cache exists: {cache_dir}\nUse --force_overwrite to overwrite.")
    ensure_dir(cache_dir)

    all_tools = sorted(set([int(t) for t in train_tools] + [int(test_tool)]))
    records = collect_h5_records(data_root, tools=all_tools, machine=machine)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    for r in records:
        r["label"] = get_label_from_run(r["run"], r_thr1=r_thr1, r_thr2=r_thr2)

    train_candidates = [r for r in records if int(r["tool"]) in set(train_tools)]
    test_records = [r for r in records if int(r["tool"]) == int(test_tool)]
    train_records, val_records = split_train_val_records(train_candidates, val_ratio=val_ratio)

    split_records = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    splits_json = {
        "dataset": "Mendeley zpxs87bjt8 / Multivariate time series data of milling processes",
        "exp_name": exp_name,
        "machine": f"M{machine}" if machine is not None else "all",
        "train_tools": [f"T{t}" for t in train_tools],
        "test_tool": f"T{test_tool}",
        "label_basis": "Run index R",
        "label_rule": {
            "0": f"Initial: R < {r_thr1}",
            "1": f"Normal: {r_thr1} <= R < {r_thr2}",
            "2": f"Severe: R >= {r_thr2}",
        },
        "split_policy": "cross-tool test; train/val split only from training tools; record-level; no segment leakage",
        "splits": {
            sp: [
                {
                    "filename": r["filename"],
                    "machine": int(r["machine"]),
                    "tool": int(r["tool"]),
                    "run": int(r["run"]),
                    "contact": int(r["contact"]),
                    "wear": float(r["wear"]),
                    "label": int(r["label"]),
                }
                for r in split_records[sp]
            ]
            for sp in ["train", "val", "test"]
        },
    }
    with open(os.path.join(dirs["splits"], f"{exp_name}.json"), "w", encoding="utf-8") as f:
        json.dump(splits_json, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"Preprocessing: {exp_name}")
    print(f"Data root : {data_root}")
    print(f"Machine   : M{machine}" if machine is not None else "Machine   : all")
    print(f"Train     : {[f'T{t}' for t in train_tools]}  Val from train tools")
    print(f"Test      : T{test_tool}")
    print(f"Label     : 0 Initial(R<{r_thr1}), 1 Normal([{r_thr1},{r_thr2})), 2 Severe(R>={r_thr2})")
    print(f"Segments  : {n_segments} per H5 run, segment_length={segment_length}")
    print(f"CWT       : scales={cwt_scales}, target_size={target_size}")
    print(f"Cache     : {cache_dir}")
    print(f"{'=' * 80}\n")

    for sp in ["train", "val", "test"]:
        df_sp = pd.DataFrame(split_records[sp])
        print(f"{sp:5s}: records={len(df_sp)}")
        if len(df_sp) > 0:
            print(df_sp.groupby(["tool", "label"]).size())
            print()

    index_data = []
    stats_state = {"sum": 0.0, "sumsq": 0.0, "count": 0}
    errors = []

    for sp in ["train", "val", "test"]:
        for rec in tqdm(split_records[sp], desc=f"Generating {sp}"):
            try:
                Fx, Fy, Fz = load_cut_signal(data_root, h5_path=rec["path"])
                Fxs, Fys, Fzs = extract_stable_window(
                    Fx, Fy, Fz,
                    s=stable_start,
                    e=stable_end,
                    ratio_start=stable_ratio_start,
                    ratio_end=stable_ratio_end,
                )
                s1d, sfx, sfy, sfz = segment_signal_xyz(
                    Fxs, Fys, Fzs,
                    n_seg=n_segments,
                    seg_len=segment_length,
                )

                tool_cache = os.path.join(cache_dir, f"M{rec['machine']}T{rec['tool']}")
                ensure_dir(tool_cache)
                stem = _safe_stem(rec["filename"])

                for si in range(n_segments):
                    p1d = os.path.abspath(os.path.join(tool_cache, f"{stem}_seg{si}_1d.npy"))
                    p2d = os.path.abspath(os.path.join(tool_cache, f"{stem}_seg{si}_2d.npy"))

                    np.save(p1d, s1d[si].astype(np.float32))

                    cwt = compute_cwt_3channel(
                        sfx[si], sfy[si], sfz[si],
                        scales=cwt_scales,
                        target_size=target_size,
                    ).astype(np.float32)
                    np.save(p2d, cwt)

                    if sp == "train":
                        _update_stats(cwt, stats_state)

                    index_data.append({
                        "tool_id": f"M{int(rec['machine'])}T{int(rec['tool'])}",
                        "cut_id": int(rec["run"]),       # keep old column name; here it means H5 run R
                        "seg_id": int(si),
                        "label": int(rec["label"]),
                        "split": sp,
                        "path_1d": p1d,
                        "path_2d": p2d,

                        # VB is not used for labeling in this version,
                        # but it is kept for analysis and compatibility.
                        "wear_mean": float(rec["wear"]),
                        "wear_vb": float(rec["wear"]),

                        "machine_id": f"M{int(rec['machine'])}",
                        "tool_no": int(rec["tool"]),
                        "run": int(rec["run"]),
                        "R": int(rec["run"]),
                        "contact_time": int(rec["contact"]),
                        "C": int(rec["contact"]),
                        "filename": rec["filename"],
                    })

            except Exception as e:
                msg = f"Error {rec.get('filename', rec.get('path'))}: {repr(e)}"
                errors.append(msg)
                print(msg)

    idf = pd.DataFrame(index_data)
    index_path = os.path.join(dirs["indexes"], f"{exp_name}_index.csv")
    idf.to_csv(index_path, index=False)

    stats = _finalize_stats(stats_state)
    stats_path = os.path.join(dirs["stats"], f"{exp_name}_norm.npz")
    np.savez(stats_path, **stats)

    print(f"\n✓ Index saved: {index_path}")
    print(f"✓ Stats saved: {stats_path} (2D mean={stats['mean_2d']:.6f}, std={stats['std_2d']:.6f})")
    print(f"✓ Total segment samples: {len(idf)}")
    if len(idf) > 0:
        print("\nSegment-level distribution:")
        print(idf.groupby(["split", "label"]).size())
        print("\nRecord-level distribution:")
        rec_df = idf.drop_duplicates(["split", "filename"])
        print(rec_df.groupby(["split", "label"]).size())

        print("\nRecord-level R range by split/label:")
        print(rec_df.groupby(["split", "label"])["R"].agg(["min", "max", "count"]))

        print("\nRecord-level VB range by split/label, only for reference:")
        print(rec_df.groupby(["split", "label"])["wear_vb"].agg(["min", "max", "count"]))

    if errors:
        err_path = os.path.join(dirs["indexes"], f"{exp_name}_errors.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write("\n".join(errors))
        print(f"\n⚠ {len(errors)} files failed. Error log: {err_path}")

    print(f"\n{'=' * 80}\nDone: {exp_name}\n{'=' * 80}\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Preprocess Mendeley Milling H5 dataset for cross-tool TCM experiments")

    p.add_argument(
        "--data_root",
        default=r"D:\Milling_H5\Multivariate time series data of milling processes with varying tool wear and machine tools",
        help="Folder containing filelist.csv and .h5 files",
    )
    p.add_argument("--output_root", default="./data")
    p.add_argument(
        "--exp",
        default="T5T6_to_T4",
        choices=["T4T5_to_T6", "T4T6_to_T5", "T5T6_to_T4"],
        help="Cross-tool fold among T4/T5/T6",
    )
    p.add_argument("--all_folds", action="store_true", help="Generate all 3 folds: T4T5->T6, T4T6->T5, T5T6->T4")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--machine", type=int, default=2, help="T4/T5/T6 are on machine M2; use -1 for all machines")
    p.add_argument("--val_ratio", type=float, default=0.20)

    p.add_argument("--n_segments", type=int, default=5)
    p.add_argument("--segment_length", type=int, default=2048)

    p.add_argument("--stable_start", type=int, default=-1, help="Fixed start index. -1 means use ratio crop.")
    p.add_argument("--stable_end", type=int, default=-1, help="Fixed end index. -1 means use ratio crop.")
    p.add_argument("--stable_ratio_start", type=float, default=0.20)
    p.add_argument("--stable_ratio_end", type=float, default=0.80)

    p.add_argument("--cwt_scales", type=int, default=224)
    p.add_argument("--target_size", type=int, default=224)

    # R-based label thresholds
    p.add_argument("--r_thr1", type=float, default=300.0, help="Initial/Normal threshold based on run index R")
    p.add_argument("--r_thr2", type=float, default=700.0, help="Normal/Severe threshold based on run index R")

    p.add_argument("--force_overwrite", action="store_true")
    args = p.parse_args()

    cfgs = {
        "T4T5_to_T6": ([4, 5], 6),
        "T4T6_to_T5": ([4, 6], 5),
        "T5T6_to_T4": ([5, 6], 4),
    }

    machine = None if args.machine == -1 else args.machine
    exps = list(cfgs.keys()) if args.all_folds else [args.exp]

    for exp_name in exps:
        tr, te = cfgs[exp_name]
        preprocess_experiment(
            data_root=args.data_root,
            exp_name=exp_name,
            train_tools=tr,
            test_tool=te,
            output_root=args.output_root,
            seed=args.seed,
            machine=machine,
            n_segments=args.n_segments,
            segment_length=args.segment_length,
            stable_start=args.stable_start,
            stable_end=args.stable_end,
            stable_ratio_start=args.stable_ratio_start,
            stable_ratio_end=args.stable_ratio_end,
            cwt_scales=args.cwt_scales,
            target_size=args.target_size,
            val_ratio=args.val_ratio,
            r_thr1=args.r_thr1,
            r_thr2=args.r_thr2,
            force_overwrite=args.force_overwrite,
        )


if __name__ == "__main__":
    main()