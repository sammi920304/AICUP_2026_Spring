# AI CUP 2026 春季賽 — 基於時序資料之桌球戰術與結果預測

> **Private Leaderboard Score:** 0.3749948  
> **Model:** Causal Transformer (Multi-Task) + 5-Fold Ensemble

---

## 目錄

- [專案簡介](#專案簡介)
- [環境需求](#環境需求)
- [安裝方式](#安裝方式)
- [專案結構](#專案結構)
- [快速開始](#快速開始)
- [訓練參數說明](#訓練參數說明)
- [模型架構](#模型架構)
- [實驗結果](#實驗結果)
- [參考文獻](#參考文獻)

---

## 專案簡介

本專案為 AI CUP 2026 春季賽「基於時序資料之桌球戰術與結果預測」競賽的解題方案。

競賽任務要求根據一個 rally 中已發生的擊球序列，預測後續的：
- `actionId`：下一拍球種
- `pointId`：下一拍落點
- `serverGetPoint`：發球方是否得分（二元分類）

本方案採用 **Causal Transformer 多任務序列預測模型**，搭配：
- 5-fold GroupKFold ensemble
- 舊版 `test.csv` 作為額外訓練資料（domain adaptation）
- Light data augmentation（player/shot/score masking、random truncation、span masking）
- Warmup + Cosine LR Scheduler
- Last-step upweighting on rally BCE loss

---

## 環境需求

| 項目 | 建議版本 |
|------|----------|
| Python | 3.10 或 3.11 |
| PyTorch | ≥ 2.1.0 |
| CUDA（選用） | 11.8 / 12.1 |
| OS | Windows / Linux / macOS |

GPU 非必要，但有 CUDA 支援的 NVIDIA GPU 可大幅加速訓練。  
開發與測試環境：NVIDIA GeForce RTX 4050 Laptop GPU。

---

## 安裝方式

### 方式一：Conda（建議）

```bash
conda env create -f environment.yml
conda activate tabletennis
```

若要指定 CUDA 版本的 PyTorch（例如 CUDA 12.1），可在建立環境後執行：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 方式二：pip

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

若使用 GPU，請依照 [PyTorch 官方安裝指引](https://pytorch.org/get-started/locally/) 安裝對應 CUDA 版本。

---

## 專案結構

```
AICUP_2026_Spring/
├── train_causal_transformer_v2_extraoldtest.py   # 主訓練腳本
├── train.csv                                      # 主要訓練資料（84,707 筆擊球紀錄）
├── test.csv                                       # 舊版測試資料，作為額外訓練用（3,589 筆）
├── test_new.csv                                   # 最終預測資料（5,668 筆）
├── requirements.txt                               # pip 依賴清單
├── environment.yml                                # conda 環境設定
├── .gitignore
├── README.md
└── models_v2_extraold_w15_dropout030/             # 5-fold 訓練好的模型權重
    ├── fold1_seed2026.pt
    ├── fold2_seed2026.pt
    ├── fold3_seed2026.pt
    ├── fold4_seed2026.pt
    └── fold5_seed2026.pt
```

---

## 快速開始

### 最終提交版本指令（完整參數）

資料集與模型權重皆已包含於 repo，clone 後即可直接執行：

```bash
git clone https://github.com/sammi920304/AICUP_2026_Spring.git
cd AICUP_2026_Spring
```

**Windows（PowerShell）：**

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

訓練完成後會輸出 `submission_seed2026_v2_extraold_w15_dropout030.csv`。

### 快速驗證版本（僅主要訓練資料）

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

## 資料集說明

| 檔案 | 筆數 | 用途 |
|------|------|------|
| `train.csv` | 84,707 筆擊球紀錄 | 主要訓練資料 |
| `test.csv` | 3,589 筆擊球紀錄 | 舊版有標籤測試資料，作為額外訓練資料（競賽允許使用） |
| `test_new.csv` | 5,668 筆擊球紀錄 | 最終預測資料 |

**欄位說明：**

| 欄位 | 說明 |
|------|------|
| `rally_uid` | Rally 唯一識別碼 |
| `sex` | 性別 |
| `match` | 比賽場次（GroupKFold 分組依據） |
| `numberGame` | 局數 |
| `strikeNumber` | 當前拍次 |
| `scoreSelf` / `scoreOther` | 雙方比分 |
| `serverGetPoint` | 發球方是否得分（預測目標） |
| `gamePlayerId` / `gamePlayerOtherId` | 球員 ID |
| `strikeId` | 擊球方式 |
| `handId` | 慣用手 |
| `strengthId` | 擊球力道 |
| `spinId` | 旋轉類型 |
| `pointId` | 落點（預測目標） |
| `actionId` | 球種（預測目標） |
| `positionId` | 站位 |

---

## 訓練參數說明

### 資料相關

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--train` | `train.csv` | 主要訓練資料路徑 |
| `--test` | `test_new.csv` | 預測資料路徑 |
| `--extra_old_test` | `""` | 舊版 test.csv 路徑（額外訓練用） |
| `--extra_weight` | `1.0` | 額外資料重複比例（1.5 = 全部 + 隨機 50%） |
| `--extra_use_server_label` | `1` | 是否使用舊 test 的 serverGetPoint 標籤 |
| `--out` | `submission_v2_taskavg.csv` | 輸出 submission 路徑 |
| `--model_dir` | `models_v2_extraoldtest` | 模型權重儲存目錄 |

### 訓練設定

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--seeds` | `"2026"` | 逗號分隔 seed，多 seed 則做 ensemble |
| `--folds` | `5` | GroupKFold fold 數 |
| `--max_folds` | `0` | 限制 fold 數，0 = 全部 |
| `--epochs` | `16` | 最大訓練 epoch 數 |
| `--patience` | `6` | Early stopping patience |
| `--batch_size` | `128` | Batch size |
| `--warmup_epochs` | `2` | LR linear warmup epoch 數 |

### 模型架構

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--emb_dim` | `24` | 每個特徵的 embedding 維度 |
| `--model_dim` | `192` | Transformer hidden dimension |
| `--n_heads` | `6` | Multi-head attention head 數 |
| `--n_layers` | `3` | Transformer encoder 層數 |
| `--ff_dim` | `384` | Feed-forward network 維度 |
| `--dropout` | `0.28` | Dropout 比例 |
| `--head_type` | `"linear"` | 分類頭類型：`linear` 或 `mlp` |

### 正則化與優化

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--lr` | `8e-4` | 初始學習率 |
| `--weight_decay` | `5e-5` | AdamW weight decay |
| `--grad_clip` | `1.0` | Gradient clipping |
| `--label_smoothing` | `0.03` | Cross entropy label smoothing |
| `--class_weight_power` | `0.5` | 類別權重調整（0=平均，1=完全逆頻率） |
| `--last_step_weight` | `1.5` | Rally BCE loss 最後一步加權倍率 |

### 資料增強

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--augment` | `False` | 是否啟用資料增強 |
| `--repeat_aug` | `1` | 增強樣本重複次數 |
| `--player_mask_prob` | `0.10` | 球員 ID 遮蔽機率 |
| `--shot_mask_prob` | `0.01` | 擊球特徵遮蔽機率 |
| `--score_mask_prob` | `0.01` | 比分特徵遮蔽機率 |
| `--random_truncate_prob` | `0.25` | 隨機截斷 rally 前綴機率 |
| `--span_mask_prob` | `0.03` | 連續區段遮蔽觸發機率 |
| `--span_mask_max_len` | `3` | 連續遮蔽最大長度 |

### 推論

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--tta_n` | `1` | Test Time Augmentation 次數（1 = 關閉） |

---

## 模型架構

```
輸入特徵（14 個 BASE + 8 個 DERIVED = 22 維）
    ↓
Embedding Layer（每個 categorical feature 各自 embedding）
    ↓
Linear Projection → LayerNorm → Dropout
    ↓
Positional Embedding
    ↓
Causal Transformer Encoder（causal mask 確保只看過去資訊）
  × n_layers（預設 3 層）
  - Multi-Head Self-Attention（n_heads=6）
  - Feed-Forward Network（ff_dim=384）
  - Pre-LayerNorm + Residual
    ↓
LayerNorm
    ↓
┌──────────────┬──────────────┬──────────────┐
│ action head  │  point head  │  rally head  │
│ (Dropout+    │ (Dropout+    │ (MLP + BCE   │
│  Linear)     │  Linear)     │  sigmoid)    │
│ → actionId   │ → pointId    │ → server     │
│   (分類)     │   (分類)     │  GetPoint    │
└──────────────┴──────────────┴──────────────┘
```

**Loss Function：**
```
total_loss = 0.4 × CE(actionId) + 0.4 × CE(pointId) + 0.2 × BCE(serverGetPoint)
```

最終預測：5 個 fold 模型的 softmax / sigmoid 輸出取平均。

---

## 衍生特徵說明

| 特徵名稱 | 說明 |
|----------|------|
| `scoreDiffBucket` | 雙方比分差距（clip −15~15，offset +15） |
| `scoreSumBucket` | 雙方比分總和（clip 0~60） |
| `isDeuceLike` | 雙方皆達 20 分以上（deuce 狀態） |
| `isEarlyRally` | strikeNumber ≤ 3 |
| `isLateRally` | strikeNumber ≥ 12 |
| `playerPairId` | 雙方球員組合 ID（categorical） |
| `rallyProgressBucket` | strikeNumber 正規化後 bucket（0~9） |
| `isCriticalPoint` | scoreSelf ≥ 20 且雙方差距 ≤ 2 |

---

## 實驗結果

| 設定 | Platform Score |
|------|---------------|
| 原始基線（無額外資料） | 0.34597 |
| 加入舊版 test.csv（extra_weight=1.0） | 0.36869 |
| extra_weight=1.5 | 0.37266 |
| **extra_weight=1.5 + dropout=0.30（最終提交）** | **0.37499** |

---

## 參考文獻

- Vaswani, A., et al. (2017). *Attention is all you need.* NeurIPS.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR.
- Izmailov, P., et al. (2018). *Averaging Weights Leads to Wider Optima and Better Generalization.* UAI.
- Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD.* arXiv.
- Zerveas, G., et al. (2021). *A Transformer-based Framework for Multivariate Time Series Representation Learning.* KDD.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825–2830.
- Paszke, A., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS.

---

## 授權

本程式碼供 AI CUP 2026 競賽報告使用。使用競賽資料時請遵守主辦單位規定。
