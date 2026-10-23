# B³-Net: Controlled Posterior Bridge Learning for Multi-Task Dense Prediction

Meihua Zhou and Li Yang

[Code](https://github.com/mh-zhou/B3Net)

## Introduction

B³-Net combines a Precision Field Estimator (PFE), a Posterior Bridge Operator
(PBO), and a Contractive Dispatch Operator (CDO) for cross-task feature interaction.
The experiments in the paper were conducted on NVIDIA A100 GPUs.

![Overview of B³-Net](figures/overview.png)

## Train B³-Net

### 1. Build the environment

We follow the environment used by BridgeNet and name the environment `b3`.

```bash
conda create -n b3 python=3.8
conda activate b3
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
mim install mmsegmentation==0.30.0 mmcv==1.7.0
```

NumPy is set to 1.24.4 for Python 3.8 compatibility.

For InternImage, compile DCNv3 with a CUDA toolkit matching PyTorch (CUDA 11.7):

```bash
nvcc -V
cd models/ops_dcnv3
sh make.sh
cd ../..
python -B tests/test_dcnv3.py
```

See the [InternImage installation instructions](https://github.com/OpenGVLab/InternImage/tree/master/segmentation#installation)
for details.

### 2. Get data

We use the same preprocessed datasets as BridgeNet, ATRC, and InvPT.

```bash
mkdir -p datasets
cd datasets
wget https://data.vision.ee.ethz.ch/brdavid/atrc/NYUDv2.tar.gz
wget https://data.vision.ee.ethz.ch/brdavid/atrc/PASCALContext.tar.gz
tar xfvz NYUDv2.tar.gz
tar xfvz PASCALContext.tar.gz
cd ..
export B3_DATA_ROOT="$PWD/datasets"
```

`B3_DATA_ROOT` contains the `NYUDv2` and `PASCALContext` directories.
The dataset paths are defined in `configs/mypath.py`.

### 3. Train the model

The configuration files are in `configs/`. The default scripts train
InternImage-L on one GPU:

```bash
bash run_nyud.sh
bash run_PASCALContext.sh
```

| Dataset | Configuration | Batch size | Updates | Learning rate | Weight decay |
|:---|:---|---:|---:|---:|---:|
| NYUD-v2 | [InternImage-L (default)](configs/nyud/internimage_l.yaml) | 3 | 26K | 5e-5 | 0.03 |
| NYUD-v2 | [InternImage-L (shorter schedule)](configs/nyud/variants/internimage_l_20k.yaml) | 4 | 20K | 5e-5 | 0.03 |
| PASCAL-Context | [InternImage-L (default)](configs/pascal/internimage_l.yaml) | 4 | 45K | 3e-5 | 0.03 |
| PASCAL-Context | [InternImage-L (shorter schedule)](configs/pascal/variants/internimage_l_40k.yaml) | 4 | 40K | 3e-5 | 0.03 |

These configurations use AdamW with betas (0.9, 0.999), a polynomial learning
rate schedule with power 0.9, a validation batch size of 6, and 12 refinement
blocks per stage. Task-loss weights are fixed throughout training, with
intermediate supervision. Gradient clipping uses maximum norms of 10 for
NYUD-v2 and 5 for PASCAL-Context.

As described in the paper, the shorter 20K/40K schedules both use batch size 4.
They retain the respective default architecture, optimizer, initial learning
rate, weight decay, loss weights, and gradient clipping. The polynomial schedule
decays to zero over the selected update budget. On NYUD-v2, 20K updates at batch
size 4 give 80K training image presentations, compared with 78K for the default
26K schedule at batch size 3. On PASCAL-Context, the shorter schedule reduces
image presentations from 180K to 160K at the same batch size.

```bash
bash run_nyud.sh configs/nyud/variants/internimage_l_20k.yaml
bash run_PASCALContext.sh configs/pascal/variants/internimage_l_40k.yaml
```

To use another backbone, pass its configuration file:

```bash
bash run_nyud.sh configs/nyud/vit_b.yaml
bash run_PASCALContext.sh configs/pascal/vit_l.yaml
```

InternImage backbone weights are downloaded automatically. ViT-B training
requires the pretrained file named in `utils/common_config.py` in the PyTorch
checkpoint cache. W&B logging is disabled by default. Training checkpoints are
saved every 1,000 iterations and at the final iteration. An existing training
checkpoint resumes the run; use a new `version_name` to start a separate run.

### 4. Validation

Validation is performed during training.

For boundary detection, predictions are saved for ODS-F evaluation with
[SEISM](https://github.com/jponttuset/seism) or the
[boundary evaluation tools](https://github.com/prismformore/Boundary-Detection-Evaluation-Tools).

## Pretrained models

The checkpoints were saved at the final training iteration and are provided for visualization.

### Model downloads and paper-reported results

| Version | Dataset | Download | Segmentation (mIoU) | Human parsing (mIoU) | Saliency (maxF) | Normals (mErr) | Edge (ODS-F) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B³-Net / InternImage-L | PASCAL-Context | [Google Drive](https://drive.google.com/file/d/1dJmioAYZ9gkDn_DVhMLUXHkRerb9AeL8/view?usp=sharing) | 80.81 | 73.73 | 86.11 | 13.40 | 81.18 |

| Version | Dataset | Download | Segmentation (mIoU) | Depth (RMSE) | Normals (mErr) | Edge (ODS-F) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B³-Net / InternImage-L | NYUD-v2 | [Google Drive](https://drive.google.com/file/d/1_0z1hJs-7pTbMdXlMSw2Bi2ACajElz7z/view?usp=sharing) | 57.78 | 0.4587 | 17.22 | 83.18 |


## Visualizations

Qualitative results and mechanism visualizations from the paper.

### NYUD-v2

![NYUD-v2 qualitative comparison](vis/nyud/qualitative.png)

### PASCAL-Context

![PASCAL-Context qualitative comparison](vis/pascal/qualitative.png)

### Task outputs

![Task-output maps](figures/task_output_maps.png)

### PFE, PBO, and CDO

![Stage-3 mechanism visualization](figures/operator_mechanisms.png)

## Acknowledgments

Our code builds on [BridgeNet](https://github.com/Evergreen0929/EEMTL/tree/main/BridgeNet),
[InvPT](https://github.com/prismformore/Multi-Task-Transformer/tree/main/InvPT),
[MTI-Net](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch),
[ATRC](https://github.com/brdav/atrc), and
[InternImage](https://github.com/OpenGVLab/InternImage).
We also thank the authors of
[TaskPrompter](https://github.com/prismformore/Multi-Task-Transformer/tree/main/TaskPrompter)
for releasing their code.

We sincerely thank the reviewers for their thoughtful guidance and encouragement
throughout the peer-review process. We deeply appreciate the time and care they
devoted to our work, and their constructive feedback has greatly improved the
quality of this manuscript.


