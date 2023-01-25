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

### Create Folder Structure

```bash
~/gaohn/pytorch-lightning-pipeline (venv) $ python configs/config.py
```

### Run Training

```bash
~/gaohn/pytorch-lightning-pipeline (venv) $ python main.py
```