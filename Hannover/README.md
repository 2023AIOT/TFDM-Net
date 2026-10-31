# TFDM-Net on the Hannover Milling Dataset

This directory contains the code used for the cross-tool experiments (H1-H3) on the Hannover milling dataset. The selected tools T4, T5, and T6 belong to machine M2; two tools are used for training and validation, and the remaining tool is used for independent testing.

## Files

- `preprocess.py`: reads the HDF5 records, extracts stable cutting intervals, samples five segments per record, and generates three-channel synchrosqueezing-enhanced time-frequency maps.
- `train.py`: trains TFDM-Net for a selected cross-tool task.
- `test.py`: evaluates the best validation checkpoint on the independent test tool.
- `utils.py`: time-frequency transformation, losses, metrics, plotting, and reproducibility utilities.

The internal class names containing `TFSSM` are retained for checkpoint compatibility. They implement the time-frequency decoupled sequence mixing, adaptive gating, and attention pooling described for TFDM-Net.

## Installation

Create and activate a Python environment, install a PyTorch build compatible with your CPU/CUDA platform, and then run:

```bash
python -m pip install -r requirements.txt
```

The exact package versions of the original environment were not embedded in the supplied scripts, so `requirements.txt` lists the direct dependencies without strict version pins.

## Dataset

The raw data are not redistributed in this repository. Obtain the Hannover/Mendeley dataset from its original source (dataset identifier `zpxs87bjt8`) and comply with its license.

`preprocess.py` recursively searches the supplied root directory for `.h5` files. It recognizes filenames following the dataset convention, for example:

```text
M2T6R30C170VB45.h5
```

The expected HDF5 force channels are:

```text
signals_sensor/force_sensor_x
signals_sensor/force_sensor_y
signals_sensor/force_sensor_z
```

Equivalent paths found recursively by dataset name are also supported.

## Wear-state labels

The machining run index `R` is used to define the three classes:

- Initial wear: `R < 300`
- Normal wear: `300 <= R < 700`
- Severe wear: `R >= 700`

The middle 20%-80% of each valid record is retained by default, and five segments of 2,048 samples are uniformly extracted. Train/validation splitting is performed at record level, so segments from the same HDF5 record cannot appear in both subsets.

## Experiment mapping

| Paper task | Training tools | Test tool | Command value |
|---|---|---|---|
| H1 | T4, T5 | T6 | `T4T5_to_T6` |
| H2 | T5, T6 | T4 | `T5T6_to_T4` |
| H3 | T4, T6 | T5 | `T4T6_to_T5` |

## Preprocessing

Generate all three folds:

```bash
python preprocess.py --data_root /path/to/HANNOVER_ROOT --output_root ./data --all_folds
```

Alternatively, generate one fold:

```bash
python preprocess.py --data_root /path/to/HANNOVER_ROOT --output_root ./data --exp T4T5_to_T6
```

Preprocessing generates cached NumPy arrays, split metadata, index CSV files, normalization statistics, and an error log if any records cannot be processed. Ensure sufficient disk space before running it.

## Training and testing

Example for H1:

```bash
python train.py --data_root ./data --exp T4T5_to_T6
python test.py  --data_root ./data --exp T4T5_to_T6
```

Commands for the other two folds:

```bash
# H2
python train.py --data_root ./data --exp T5T6_to_T4
python test.py  --data_root ./data --exp T5T6_to_T4

# H3
python train.py --data_root ./data --exp T4T6_to_T5
python test.py  --data_root ./data --exp T4T6_to_T5
```

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
- The repository contains code only; users must obtain the dataset from its original source.

## Citation

If this code is useful, please cite the associated manuscript:

> TFDM-Net: A Time-Frequency Decoupled Modeling Network for Tool Wear State Recognition.
