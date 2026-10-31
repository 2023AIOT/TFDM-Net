# TFDM-Net on PHM2010: Single-Source Cross-Tool Tasks

This directory contains the code used for the single-source cross-tool experiments (M1-M3) on the PHM2010 milling dataset. One tool is used for training and validation, and a different tool is used for independent testing.

## Files

- `preprocess.py`: extracts stable cutting intervals, samples five segments per cutting pass, and generates three-channel synchrosqueezing-enhanced time-frequency maps.
- `train.py`: trains TFDM-Net using one selected source tool.
- `test.py`: evaluates a trained model at the segment level on the independent test tool.
- `utils.py`: time-frequency transformation, losses, metrics, plotting, and reproducibility utilities.

The internal class names containing `TFSSM` are retained for checkpoint compatibility. They implement the time-frequency decoupled sequence mixing, adaptive gating, and attention pooling described for TFDM-Net.

## Installation

Create and activate a Python environment, install a PyTorch build compatible with your CPU/CUDA platform, and then run:

```bash
python -m pip install -r requirements.txt
```

The exact package versions of the original environment were not embedded in the supplied scripts, so `requirements.txt` lists the direct dependencies without strict version pins.

## PHM2010 data layout

The raw data are not redistributed in this repository. Arrange the downloaded PHM2010 files as follows:

```text
PHM2010_ROOT/
├── c1/
│   ├── c_1_001.csv
│   └── ...
├── c4/
│   ├── c_4_001.csv
│   └── ...
├── c6/
│   ├── c_6_001.csv
│   └── ...
├── c1_wear.csv
├── c4_wear.csv
└── c6_wear.csv
```

Each signal CSV must contain the three cutting-force channels in its first three columns. Wear CSV files are expected to contain `cut`, `flute_1`, `flute_2`, and `flute_3` columns.

## Wear-state labels

- Initial wear: cutting passes 1-50
- Normal wear: cutting passes 51-175
- Severe wear: cutting passes 176-315

The train/validation split is performed at cutting-pass level. All five segments from the same pass remain in the same subset.

## Experiment mapping

| Paper task | Source tool | Test tool | Preprocessing experiment | Training name |
|---|---:|---:|---|---|
| M11 | C1 | C4 | `C1C6_to_C4` | `M11` |
| M12 | C1 | C6 | `C1C4_to_C6` | `M12` |
| M21 | C4 | C1 | `C4C6_to_C1` | `M21` |
| M22 | C4 | C6 | `C1C4_to_C6` | `M22` |
| M31 | C6 | C1 | `C4C6_to_C1` | `M31` |
| M32 | C6 | C4 | `C1C6_to_C4` | `M32` |

## Preprocessing

Preprocess one target-tool configuration at a time. Separate output directories avoid cache conflicts and unnecessary overwriting:

```bash
# Test tool C4: supports M11 and M32
python preprocess.py --data_root /path/to/PHM2010_ROOT --output_root ./data_C4 --exp C1C6_to_C4

# Test tool C6: supports M12 and M22
python preprocess.py --data_root /path/to/PHM2010_ROOT --output_root ./data_C6 --exp C1C4_to_C6

# Test tool C1: supports M21 and M31
python preprocess.py --data_root /path/to/PHM2010_ROOT --output_root ./data_C1 --exp C4C6_to_C1
```

Preprocessing generates cached NumPy arrays, split metadata, index CSV files, and normalization statistics. Ensure sufficient disk space before running it.

## Training and testing

Example for M11 (C1 to C4):

```bash
python train.py --exp M11 --data_exp C1C6_to_C4 --train_tool C1 --data_root ./data_C4
python test.py  --exp M11 --data_exp C1C6_to_C4 --data_root ./data_C4
```

Commands for all remaining tasks:

```bash
# M12: C1 to C6
python train.py --exp M12 --data_exp C1C4_to_C6 --train_tool C1 --data_root ./data_C6
python test.py  --exp M12 --data_exp C1C4_to_C6 --data_root ./data_C6

# M21: C4 to C1
python train.py --exp M21 --data_exp C4C6_to_C1 --train_tool C4 --data_root ./data_C1
python test.py  --exp M21 --data_exp C4C6_to_C1 --data_root ./data_C1

# M22: C4 to C6
python train.py --exp M22 --data_exp C1C4_to_C6 --train_tool C4 --data_root ./data_C6
python test.py  --exp M22 --data_exp C1C4_to_C6 --data_root ./data_C6

# M31: C6 to C1
python train.py --exp M31 --data_exp C4C6_to_C1 --train_tool C6 --data_root ./data_C1
python test.py  --exp M31 --data_exp C4C6_to_C1 --data_root ./data_C1

# M32: C6 to C4
python train.py --exp M32 --data_exp C1C6_to_C4 --train_tool C6 --data_root ./data_C4
python test.py  --exp M32 --data_exp C1C6_to_C4 --data_root ./data_C4
```

The default training configuration uses seed 42, batch size 16, AdamW, an initial learning rate of `2e-4`, weight decay 0.05, and at most 60 epochs.

## Outputs

By default, each task is saved under `outputs_only2d_tfssm/<task>/`, including:

- `best_model.pth`
- `history.json`
- `training_curves.png`
- `test_results.json`
- segment-level confusion matrix

Each test tool contains 315 cutting passes, with five segments
extracted from each pass, resulting in 1,575 test samples.
Evaluation is performed at the segment level, consistent with
the experimental protocol described in the paper.

## Reproducibility notes

- Run commands from this directory so that local imports resolve to the matching `train.py` and `utils.py`.
- Keep preprocessing, training, and testing on the same generated data directory.
- The default random seed is 42. Exact GPU results can still vary with hardware, CUDA, cuDNN, and PyTorch versions.
- The repository contains code only; users must obtain the PHM2010 dataset from its original source and comply with its license.

## Citation

If this code is useful, please cite the associated manuscript:

> TFDM-Net: A Time-Frequency Decoupled Modeling Network for Tool Wear State Recognition.
