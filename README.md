<div align="center">
<h1>PyTorch Lightning Training Pipeline</a></h1>
by Hongnan Gao
Jan, 2023
<br>
</div>

## Introduction

This repository contains a PyTorch Lightning training pipeline for computer vision tasks.

## Workflow

### Installation

```bash
~/gaohn $ git clone https://github.com/gao-hongnan/pytorch-lightning-pipeline.git
~/gaohn $ cd pytorch-lightning-pipeline
~/gaohn/pytorch-lightning-pipeline        $ python -m venv <venv_name> && <venv_name>\Scripts\activate
~/gaohn/pytorch-lightning-pipeline (venv) $ python -m pip install --upgrade pip setuptools wheel
~/gaohn/pytorch-lightning-pipeline (venv) $ pip3 install torch torchvision torchaudio \
                                    --extra-index-url https://download.pytorch.org/whl/cu113
~/gaohn/pytorch-lightning-pipeline (venv) $ pip install -r requirements.txt
```

If you are using Conda, then you can do

```bash
~/gaohn/pytorch-lightning-pipeline (venv) $ conda install pytorch torchvision torchaudio \
                                     pytorch-cuda=11.6 -c pytorch -c nvidia
```

If your computer does not have GPU, then you can install the CPU version of PyTorch.

```bash
~/gaohn/pytorch-lightning-pipeline (venv) $ pip3 install torch torchvision torchaudio # 1.12.1
```

### Run Training

For macOS without MPS

```
python main.py --config-name rsna general.stage=train general.debug=True general.device=cpu datamodule.dataset.train_dir=./data datamodule.dataset.train_csv=./data/train/train.csv
```

For macOS with MPS

```
python main.py --config-name rsna general.stage=train general.debug=True general.device=mps datamodule.dataset.train_dir=./data datamodule.dataset.train_csv=./data/train/train.csv
```

Multirun all folds

```
python main.py --multirun --config-name rsna datamodule.fold=1,2,3,4 general.stage=train general.debug=True general.device=mps datamodule.dataset.train_dir=./data/train datamodule.dataset.train_csv=./data/train/train.csv
```