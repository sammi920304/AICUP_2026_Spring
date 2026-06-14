# AI CUP 2026 春季賽 — 基於時序資料之桌球戰術與結果預測

> **Private Leaderboard Score:** `0.3749948`
> **Final Model:** Causal Transformer Multi-Task Model + 5-Fold Ensemble

---

## Table of Contents

* [Project Overview](#project-overview)
* [Environment](#environment)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Code Overview](#code-overview)
* [Quick Start](#quick-start)
* [Dataset](#dataset)
* [Training Arguments](#training-arguments)
* [Model Architecture](#model-architecture)
* [Derived Features](#derived-features)
* [Experiment Strategies and Final Configuration](#experiment-strategies-and-final-configuration)
* [Results](#results)
* [External Resources and References](#external-resources-and-references)
* [License](#license)

---

## Project Overview

This repository contains the solution for the **AI CUP 2026 Spring — Table Tennis Tactical and Outcome Prediction Based on Time-Series Data** competition.

The task is to predict the following targets based on the observed stroke sequence in a rally:

* `actionId`: the next stroke type
* `pointId`: the next landing point
* `serverGetPoint`: whether the server wins the point

The final solution uses a **Causal Transformer multi-task sequence prediction model** with:

* 5-fold GroupKFold ensemble
* Additional labeled old `test.csv` as extra training data
* Light data augmentation
* Dropout regularization
* Last-step loss upweighting
* Multi-task prediction heads for `actionId`, `pointId`, and `serverGetPoint`

Final best submission:

```text
submission_seed2026_v2_extraold_w15_dropout030.csv
```

Private Leaderboard score:

```text
0.3749948
```

---

## Environment

The code can be executed in a general Python deep learning environment with PyTorch installed.

| Item    | Recommended Version                    |
| ------- | -------------------------------------- |
| Python  | 3.10 or 3.11                           |
| PyTorch | >= 2.1.0                               |
| CUDA    | Optional, recommended for GPU training |
| OS      | Windows / Linux / macOS                |

GPU is not strictly required, but CUDA-enabled NVIDIA GPUs can significantly speed up training.

The experiments were developed and tested with:

```text
NVIDIA GeForce RTX 4050 Laptop GPU
```

---

## Installation

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate tabletennis
```

If a specific CUDA version is required, install the corresponding PyTorch version after creating the environment. For example, for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Option 2: pip

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

If GPU acceleration is needed, please follow the official PyTorch installation guide to install the CUDA-compatible version.

---

## Project Structure

```text
AICUP_2026_Spring/
├── train_causal_transformer_v2_extraoldtest.py   # Main training and inference script
├── train.csv                                      # Main training data
├── test.csv                                       # Old labeled test data used as extra training data
├── test_new.csv                                   # Final prediction data
├── requirements.txt                               # pip dependencies
├── environment.yml                                # conda environment
├── .gitignore
├── README.md
└── models_v2_extraold_w15_dropout030/             # Saved 5-fold model checkpoints
    ├── fold1_seed2026.pt
    ├── fold2_seed2026.pt
    ├── fold3_seed2026.pt
    ├── fold4_seed2026.pt
    └── fold5_seed2026.pt
```

---

## Code Overview

The main script is:

```text
train_causal_transformer_v2_extraoldtest.py
```

This script includes the full pipeline from data preprocessing to model training and submission generation.

| Module                | Description                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------- |
| Argument parser       | Defines paths, training parameters, model hyperparameters, and augmentation settings      |
| Data loading          | Loads `train.csv`, `test_new.csv`, and optionally old `test.csv` as extra training data   |
| Feature engineering   | Builds score, rally progress, player pair, and critical-point features                    |
| Sequence construction | Groups strokes by rally and converts them into time-series training samples               |
| Data augmentation     | Supports player masking, shot masking, score masking, random truncation, and span masking |
| Model definition      | Implements categorical embeddings, positional embeddings, and Causal Transformer Encoder  |
| Multi-task heads      | Predicts `actionId`, `pointId`, and `serverGetPoint`                                      |
| Training loop         | Uses GroupKFold, validation scoring, early stopping, and checkpoint saving                |
| Inference             | Averages predictions from all fold models to generate the final submission                |
| Checkpointing         | Saves the best model of each fold to `model_dir`                                          |

The script also keeps several optional parameters for experiments, such as `--tta_n`, `--head_type`, and `--max_folds`. In the final submitted configuration, `--tta_n 1` is used, which means Test Time Augmentation is disabled.

---

## Quick Start

### Clone the repository

```bash
git clone https://github.com/sammi920304/AICUP_2026_Spring.git
cd AICUP_2026_Spring
```

### Final training command

#### Windows PowerShell

```powershell
python train_causal_transformer_v2_extraoldtest.py `
  --train train.csv `
  --test test_new.csv `
  --extra_old_test test.csv `
  --out submission_seed2026_v2_extraold_w15_dropout030.csv `
  --sample __no_sample__.csv `
  --seeds 2026 `
  --max_folds 0 `
  --epochs 16 `
  --patience 6 `
  --batch_size 128 `
  --dropout 0.30 `
  --weight_decay 0.00005 `
  --augment `
  --repeat_aug 1 `
  --player_mask_prob 0.10 `
  --shot_mask_prob 0.01 `
  --score_mask_prob 0.01 `
  --random_truncate_prob 0.25 `
  --span_mask_prob 0.03 `
  --span_mask_max_len 3 `
  --warmup_epochs 2 `
  --last_step_weight 1.5 `
  --tta_n 1 `
  --class_weight_power 0.50 `
  --head_type linear `
  --extra_weight 1.5 `
  --extra_use_server_label 1 `
  --model_dir models_v2_extraold_w15_dropout030
```

After training and inference, the script outputs:

```text
submission_seed2026_v2_extraold_w15_dropout030.csv
```

### Quick sanity check

For a shorter test run:

```bash
python train_causal_transformer_v2_extraoldtest.py \
  --train train.csv \
  --test test_new.csv \
  --out submission_quick.csv \
  --epochs 10 \
  --batch_size 128 \
  --augment
```

---

## Dataset

| File           | Number of Rows | Usage                                             |
| -------------- | -------------: | ------------------------------------------------- |
| `train.csv`    |         84,707 | Main training data                                |
| `test.csv`     |          3,589 | Old labeled test data used as extra training data |
| `test_new.csv` |          5,668 | Final prediction data                             |

### Main columns

| Column                               | Description                       |
| ------------------------------------ | --------------------------------- |
| `rally_uid`                          | Unique rally identifier           |
| `sex`                                | Gender                            |
| `match`                              | Match identifier                  |
| `numberGame`                         | Game number                       |
| `strikeNumber`                       | Stroke number in the rally        |
| `scoreSelf` / `scoreOther`           | Current score                     |
| `serverGetPoint`                     | Whether the server wins the point |
| `gamePlayerId` / `gamePlayerOtherId` | Player IDs                        |
| `strikeId`                           | Stroke phase / type identifier    |
| `handId`                             | Handedness                        |
| `strengthId`                         | Stroke strength                   |
| `spinId`                             | Spin type                         |
| `pointId`                            | Landing point                     |
| `actionId`                           | Stroke action type                |
| `positionId`                         | Player position                   |

---

## Training Arguments

### Data arguments

| Argument                   | Default                     | Description                                                |
| -------------------------- | --------------------------- | ---------------------------------------------------------- |
| `--train`                  | `train.csv`                 | Main training data path                                    |
| `--test`                   | `test_new.csv`              | Prediction data path                                       |
| `--extra_old_test`         | `""`                        | Old `test.csv` path used as extra training data            |
| `--extra_weight`           | `1.0`                       | Extra data weight; final version uses `1.5`                |
| `--extra_use_server_label` | `1`                         | Whether to use `serverGetPoint` labels from old `test.csv` |
| `--out`                    | `submission_v2_taskavg.csv` | Output submission path                                     |
| `--model_dir`              | `models_v2_extraoldtest`    | Directory for saved checkpoints                            |

### Training settings

| Argument          | Default  | Description                               |
| ----------------- | -------- | ----------------------------------------- |
| `--seeds`         | `"2026"` | Comma-separated random seeds              |
| `--folds`         | `5`      | Number of GroupKFold splits               |
| `--max_folds`     | `0`      | Maximum folds to run; `0` means all folds |
| `--epochs`        | `16`     | Maximum number of epochs                  |
| `--patience`      | `6`      | Early stopping patience                   |
| `--batch_size`    | `128`    | Batch size                                |
| `--warmup_epochs` | `2`      | Number of warmup epochs                   |

### Model arguments

| Argument      | Default    | Description                                      |
| ------------- | ---------- | ------------------------------------------------ |
| `--emb_dim`   | `24`       | Embedding dimension for each categorical feature |
| `--model_dim` | `192`      | Transformer hidden dimension                     |
| `--n_heads`   | `6`        | Number of attention heads                        |
| `--n_layers`  | `3`        | Number of Transformer encoder layers             |
| `--ff_dim`    | `384`      | Feed-forward dimension                           |
| `--dropout`   | `0.28`     | Dropout rate; final version uses `0.30`          |
| `--head_type` | `"linear"` | Prediction head type: `linear` or `mlp`          |

### Optimization and regularization

| Argument               | Default | Description                       |
| ---------------------- | ------- | --------------------------------- |
| `--lr`                 | `8e-4`  | Initial learning rate             |
| `--weight_decay`       | `5e-5`  | AdamW weight decay                |
| `--grad_clip`          | `1.0`   | Gradient clipping                 |
| `--label_smoothing`    | `0.03`  | Label smoothing for cross entropy |
| `--class_weight_power` | `0.5`   | Class weight strength             |
| `--last_step_weight`   | `1.5`   | Last-step loss upweighting factor |

### Data augmentation

| Argument                 | Default | Description                                   |
| ------------------------ | ------- | --------------------------------------------- |
| `--augment`              | `False` | Enable data augmentation                      |
| `--repeat_aug`           | `1`     | Number of augmented copies                    |
| `--player_mask_prob`     | `0.10`  | Player feature masking probability            |
| `--shot_mask_prob`       | `0.01`  | Shot feature masking probability              |
| `--score_mask_prob`      | `0.01`  | Score feature masking probability             |
| `--random_truncate_prob` | `0.25`  | Probability of random rally prefix truncation |
| `--span_mask_prob`       | `0.03`  | Probability of span masking                   |
| `--span_mask_max_len`    | `3`     | Maximum length of span masking                |

### Inference

| Argument  | Default | Description                                               |
| --------- | ------- | --------------------------------------------------------- |
| `--tta_n` | `1`     | Number of Test Time Augmentation runs; `1` means disabled |

---

## Model Architecture

```text
Input categorical features
    ↓
Categorical Embedding
    ↓
Linear Projection → LayerNorm → Dropout
    ↓
Positional Embedding
    ↓
Causal Transformer Encoder
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Residual Connection
    - LayerNorm
    ↓
LayerNorm
    ↓
┌──────────────┬──────────────┬──────────────┐
│ action head  │  point head  │  rally head  │
│ → actionId   │ → pointId    │ → server     │
│              │              │   GetPoint   │
└──────────────┴──────────────┴──────────────┘
```

### Loss function

```text
total_loss = 0.4 × CE(actionId)
           + 0.4 × CE(pointId)
           + 0.2 × BCE(serverGetPoint)
```

The final prediction is the average of softmax / sigmoid outputs from the 5 fold models.

---

## Derived Features

| Feature               | Description                                                       |
| --------------------- | ----------------------------------------------------------------- |
| `scoreDiffBucket`     | Bucketed score difference                                         |
| `scoreSumBucket`      | Bucketed total score                                              |
| `isDeuceLike`         | Whether both sides have reached at least 20 points                |
| `isEarlyRally`        | Whether `strikeNumber <= 3`                                       |
| `isLateRally`         | Whether `strikeNumber >= 12`                                      |
| `playerPairId`        | Categorical ID for the player pair                                |
| `rallyProgressBucket` | Bucketed rally progress                                           |
| `isCriticalPoint`     | Whether score is at least 20 and the score difference is within 2 |

---

## Experiment Strategies and Final Configuration

Several strategies were tested during development. The final configuration only keeps the methods that improved or stabilized the leaderboard score.

### Tested strategies

| Strategy                                       | Description                                                    | Final Status              |
| ---------------------------------------------- | -------------------------------------------------------------- | ------------------------- |
| 5-fold GroupKFold ensemble                     | Average predictions from 5 fold models                         | Used                      |
| Old `test.csv` as extra training data          | Use the allowed old labeled test set for domain adaptation     | Used                      |
| `extra_weight` search                          | Tested different weights for the old test data                 | Used `1.5`                |
| Dropout tuning                                 | Tested different dropout values                                | Used `0.30`               |
| Light data augmentation                        | Player / shot / score masking, random truncation, span masking | Used                      |
| Last-step upweighting                          | Increase loss weight near final prediction positions           | Used                      |
| Test Time Augmentation                         | Optional inference-time augmentation                           | Not used; final `tta_n=1` |
| MLP prediction head                            | Replaced linear heads with MLP heads                           | Not used                  |
| Multi-seed ensemble                            | Ensemble multiple random seeds                                 | Not used                  |
| External CoachAI / ShuttleSet22 converted data | Tested converted external stroke forecasting data              | Not used                  |

### Final configuration

The final submitted version uses:

* Causal Transformer multi-task model
* 5-fold GroupKFold ensemble
* `train.csv` as the main training data
* Old `test.csv` as additional training data
* `extra_weight=1.5`
* `extra_use_server_label=1`
* `dropout=0.30`
* `last_step_weight=1.5`
* Light data augmentation
* Linear prediction heads
* `tta_n=1`, meaning Test Time Augmentation is disabled

---

## Results

| Setting                                        | Platform Score |
| ---------------------------------------------- | -------------: |
| Original baseline without extra data           |        0.34597 |
| Add old `test.csv`, `extra_weight=1.0`         |        0.36869 |
| Add old `test.csv`, `extra_weight=1.5`         |        0.37266 |
| **Final: `extra_weight=1.5` + `dropout=0.30`** |    **0.37499** |

The results show that the old `test.csv` provides useful distributional information for `test_new.csv`. Setting `extra_weight=1.5` gave the best balance between the original training data and the old test-domain data. Increasing dropout to `0.30` further improved generalization.

---

## External Resources and References

### External resources

* Official competition data:

  * `train.csv`
  * `test_new.csv`
  * old labeled `test.csv`
* Open-source Python libraries:

  * PyTorch
  * pandas
  * NumPy
  * scikit-learn
  * tqdm

### References

* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems.
* Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization*. International Conference on Learning Representations.

---

## License

This repository is prepared for the AI CUP 2026 competition report.
Please follow the official competition rules when using the provided data.
