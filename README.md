# BCI AIC3 Deep Learning Pipeline

The code in this repo is for the AIC3 competition.

The goal of this code is to serve as a full pipeline for data loading, preprocessing, model training, evaluation and inference, while being modular and giving reproducible results given the same inputs and configurations.

### Problem Statement

Given raw EEG signal readings from two tasks Motor-Imagery (MI) and Steady-State Visual Evoked Potential (SSVEP), the goal is to classify the signal to the correct class.

MI: (Left, Right)
SSVEP: (Left, Right, Forward, Backward)


### Repository Structure

```
╭──── LICENSE
├── README.md
├── checkpoints                     # model checkpoints
│   ├── MI
│   │   ├── eegnet-mi-best-f1-epoch=05-val_f1=0.5192.ckpt
│   │   └── last.ckpt
│   └── SSVEP
├── configs
│   ├── label_mapping_str_to_int.json
│   ├── label_mappings_int_to_str.json
│   ├── mi_config.yaml
│   └── ssvep_config.yaml
├── data                            # data folder structure (not uploaded to github)
│   ├── interim
│   ├── processed
│   └── raw
│       ├── MI
│       ├── README.md
│       ├── SSVEP
│       ├── mtcaic3.zip
│       ├── sample_submission.csv
│       ├── test.csv
│       ├── train.csv
│       └── validation.csv
├── models
│   ├── MI
│   └── SSVEP
├── notebooks
│   ├── aic3_test1_messy.ipynb
│   ├── lightning_trainer.ipynb
│   ├── simple_cnn_training_both.ipynb
│   └── testing_imports.ipynb
├── pyproject.toml
├── requirements-dev.lock
├── requirements.lock
├── results
│   ├── figures
│   └── logs
├── scripts
│   └── make_labels.py
├── src
│   └── bci_aic3
│       ├── __init__.py
│       ├── config.py           # config definitions and config loading
│       ├── data.py             # BCIDataset, and data loading
│       ├── inference.py        # inference methods and functions
│       └── models/             # torch models 
│            └──── eegnet.py    # EEGNet architecture
│       ├── paths.py            # paths defined with respect to PROJECT_ROOT
│       ├── preprocess.py       # filtering, ffts, down-sampling, etc...
│       ├── train.py            # pytorch lightning trainer and lightning 
│       └── util.py             # various utility scripts
├── training_stats
│   └── mi_train.pt
└── uv.lock                     # used with uv sync 
```

### Methodology

- #### Preprocessing
    TODO: filtering, ffts, down-sampling, etc...

- #### Architecture 
    TODO: EEGNet

- #### Evaluation
    TODO: Average F1 score of 2 models 


### Steps to run

```
uv sync
```

To train the MI model.
```
uv run -m bci_aic3.train --config_file mi_config.yaml
```

To train the SSVEP model.
```
uv run -m bci_aic3.train --config_file ssvep_config.yaml
```
