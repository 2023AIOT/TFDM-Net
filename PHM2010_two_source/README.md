# TFDM-Net on PHM2010: Two-Source Cross-Tool Tasks

This directory contains the code used for the two-source cross-tool experiments (T1-T3) on the PHM2010 milling dataset. Two tools are used for training and validation, and the remaining tool is used for independent testing.

## Files

- `preprocess.py`: extracts stable cutting intervals, samples five segments per cutting pass, and generates three-channel synchrosqueezing-enhanced time-frequency maps.
- `train.py`: trains TFDM-Net for a selected two-source task.
- `test.py`: evaluates the best validation checkpoint on the independent test tool.
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

| Paper task | Training tools | Test tool | Command value |
|---|---|---|---|
| T1 | C1, C4 | C6 | `C1C4_to_C6` |
| T2 | C4, C6 | C1 | `C4C6_to_C1` |
| T3 | C1, C6 | C4 | `C1C6_to_C4` |

## Running one task

Example for T1:

```bash
python preprocess.py --data_root /path/to/PHM2010_ROOT --output_root ./data_T1 --exp C1C4_to_C6
python train.py --data_root ./data_T1 --exp C1C4_to_C6
python test.py  --data_root ./data_T1 --exp C1C4_to_C6
```

Use a separate preprocessing directory for each fold to avoid cache conflicts:

```bash
# T2
python preprocess.py --data_root /path/to/PHM2010_ROOT --output_root ./data_T2 --exp C4C6_to_C1
python train.py --data_root ./data_T2 --exp C4C6_to_C1
python test.py  --data_root ./data_T2 --exp C4C6_to_C1

# T3
python preprocess.py --data_root /path/to/PHM2010_ROOT --output_root ./data_T3 --exp C1C6_to_C4
python train.py --data_root ./data_T3 --exp C1C6_to_C4
python test.py  --data_root ./data_T3 --exp C1C6_to_C4
```

Preprocessing generates cached NumPy arrays, split metadata, index CSV files, and normalization statistics. Ensure sufficient disk space before running it.

The default training configuration uses seed 42, batch size 16, AdamW, an initial learning rate of `2e-4`, weight decay 0.05, and at most 60 epochs.

## Outputs

By default, each experiment is saved under `outputs_only2d_tfssm/<experiment>/`, including:

- `best_model.pth`
- `history.json`
- `training_curves.png`
- `test_results.json`
- validation and test confusion matrices

## Reproducibility notes

- Run commands from this directory so that local imports resolve to the matching `train.py` and `utils.py`.
- Keep preprocessing, training, and testing on the same generated data directory.
- The default random seed is 42. Exact GPU results can still vary with hardware, CUDA, cuDNN, and PyTorch versions.
- The repository contains code only; users must obtain the PHM2010 dataset from its original source and comply with its license.
