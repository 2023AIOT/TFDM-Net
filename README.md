# TFDM-Net

Official implementation of **TFDM-Net: A Time-Frequency Decoupled Modeling Network for Tool Wear State Recognition**.

TFDM-Net performs cross-tool wear state recognition using three-axis cutting-force signals and time-frequency representations. Experiments are conducted on the PHM2010 and Hannover milling datasets.

## Repository structure

| Directory | Description |
|---|---|
| `PHM2010_single_source/` | PHM2010 single-source cross-tool experiments: one tool is used for training and another unseen tool for testing. |
| `PHM2010_two_source/` | PHM2010 two-source cross-tool experiments: two tools are used for training and the remaining tool for testing. |
| `Hannover/` | Cross-tool experiments on the Hannover milling dataset. |

Each directory contains the preprocessing, training, testing, and utility scripts, together with dataset-specific instructions and dependency information.

## Usage

Enter the directory corresponding to the desired experiment and install its dependencies. For example:

```bash
cd PHM2010_single_source
python -m pip install -r requirements.txt
```

Then follow the instructions in that directory's `README.md` to prepare the data, train the model, and evaluate the results.

## Datasets

The datasets are not redistributed in this repository. Users must obtain the PHM2010 and Hannover datasets from their original sources and comply with their respective licenses.

## Citation

If you find this repository useful, please cite:

> TFDM-Net: A Time-Frequency Decoupled Modeling Network for Tool Wear State Recognition.

Complete citation information will be added after publication.
