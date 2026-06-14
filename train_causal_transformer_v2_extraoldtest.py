#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI CUP 2026 Spring - Table Tennis Rally Prediction
Causal Transformer Multi-Task Training Script

This script contains the full training and inference pipeline used for the
AI CUP 2026 Spring table tennis tactical and outcome prediction task.  The
input data are rally-level stroke sequences, and the model predicts three
targets at the final observed prefix of each rally:

    1. actionId: next stroke action class
    2. pointId: next landing-point class
    3. serverGetPoint: probability that the server wins the point

Main components
---------------
1. Feature processing
   - Reads train.csv, test_new.csv, and optionally an old labeled test.csv.
   - Builds categorical mappings for all base and derived features.
   - Converts each rally into a padded sequence of categorical feature IDs.
   - Derived features summarize score state, rally progress, player pairing,
     and critical-point context.

2. Causal Transformer model
   - Each categorical feature is embedded separately.
   - Embedded features are concatenated, projected to the model dimension,
     combined with positional embeddings, and passed through a Transformer
     encoder with a causal attention mask.
   - The causal mask ensures that each time step can only attend to the
     current and previous strokes, matching the time-series prediction setting.

3. Multi-task learning
   - The shared Transformer encoder is followed by three output heads:
     actionId classification, pointId classification, and serverGetPoint
     binary prediction.
   - The training loss combines action cross entropy, point cross entropy,
     and rally outcome BCE loss.

4. Cross-validation ensemble
   - GroupKFold is used so that the same group does not appear in both the
     training and validation portions of a fold.
   - The final submission averages predictions from all trained fold models.

5. Extra old-test training data
   - When --extra_old_test is provided, the old labeled test set is appended
     only to the training side of each fold.
   - Validation still uses only the official training split, preventing the
     extra data from entering validation.

6. Optional experiment switches
   - The code keeps several optional switches, such as SWA, TTA, MLP heads,
     task-specific checkpoints, and task-averaged checkpoints.
   - These switches are available for reproducibility and experimentation.
   - The final submitted configuration used linear heads and tta_n=1, meaning
     Test Time Augmentation was disabled.
"""

import argparse
import math
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

PAD_TOKEN = 0
IGNORE_INDEX = -100

TARGET_ACTION = "actionId"
TARGET_POINT = "pointId"
TARGET_RALLY = "serverGetPoint"

BASE_FEATURES = [
    "sex", "numberGame", "strikeNumber",
    "scoreSelf", "scoreOther",
    "gamePlayerId", "gamePlayerOtherId",
    "strikeId", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId",
]

DERIVED_FEATURES = [
    "scoreDiffBucket", "scoreSumBucket",
    "isDeuceLike", "isEarlyRally", "isLateRally",
    "playerPairId",
    "rallyProgressBucket",   # Discretized rally progress based on strikeNumber.
    "isCriticalPoint",       # Late-score critical point indicator.
]

FEATURES = BASE_FEATURES + DERIVED_FEATURES

PLAYER_FEATURES = ["gamePlayerId", "gamePlayerOtherId", "playerPairId"]
SHOT_FEATURES = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
SCORE_FEATURES = ["scoreSelf", "scoreOther", "scoreDiffBucket", "scoreSumBucket", "isDeuceLike"]


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# These helpers control randomness, construct table-tennis-specific features,
# build categorical vocabularies, encode/pad rally sequences, and compute class
# weights for imbalanced actionId / pointId labels.
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-stroke derived features using only information from the current row.\n\n    No future strokes are referenced here, so these features are safe for\n    causal time-series prediction.\n    """
    df = df.copy()
    diff = (df["scoreSelf"] - df["scoreOther"]).clip(-15, 15) + 15
    total = (df["scoreSelf"] + df["scoreOther"]).clip(0, 60)
    df["scoreDiffBucket"] = diff.astype(int)
    df["scoreSumBucket"] = total.astype(int)
    df["isDeuceLike"] = ((df["scoreSelf"] >= 20) & (df["scoreOther"] >= 20)).astype(int)
    df["isEarlyRally"] = (df["strikeNumber"] <= 3).astype(int)
    df["isLateRally"] = (df["strikeNumber"] >= 12).astype(int)
    a = df["gamePlayerId"].astype(str)
    b = df["gamePlayerOtherId"].astype(str)
    df["playerPairId"] = a + "_" + b

    # Rally progress bucket: discretize the current stroke number into 10 bins.
    df["rallyProgressBucket"] = (df["strikeNumber"].clip(0, 30) / 30.0 * 9).astype(int)

    # Critical point indicator: late-score state with a small score difference.
    df["isCriticalPoint"] = (
        (df["scoreSelf"] >= 20) &
        ((df["scoreSelf"] - df["scoreOther"]).abs() <= 2)
    ).astype(int)
    return df


def check_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少欄位: {missing}")


def build_maps(train_df, feature_cols):
    maps, sizes = {}, []
    for col in feature_cols:
        vals = pd.Series(train_df[col].dropna().unique()).sort_values().tolist()
        mapping = {v: i + 1 for i, v in enumerate(vals)}
        maps[col] = mapping
        sizes.append(len(mapping))
    return maps, sizes


def encode_frame(df, feature_cols, maps):
    cols = []
    for col in feature_cols:
        cols.append(df[col].map(maps[col]).fillna(PAD_TOKEN).astype(np.int64).to_numpy())
    return np.stack(cols, axis=1)


def pad_2d(arr, max_len):
    out = np.full((max_len, arr.shape[1]), PAD_TOKEN, dtype=np.int64)
    n = min(len(arr), max_len)
    out[:n] = arr[:n]
    return out


def pad_1d(arr, max_len):
    out = np.full((max_len,), IGNORE_INDEX, dtype=np.int64)
    n = min(len(arr), max_len)
    out[:n] = arr[:n]
    return out


def build_train_sequences(train_df, feature_cols, maps, max_len, group_col):
    Xs, yAs, yPs, yRs, lens, groups = [], [], [], [], [], []
    train_df = train_df.sort_values(["rally_uid", "strikeNumber"])
    for rid, g in train_df.groupby("rally_uid", sort=False):
        if len(g) < 2:
            continue
        g = g.iloc[:max_len + 1]
        x_all = encode_frame(g, feature_cols, maps)
        x = x_all[:-1]
        y_a = g[TARGET_ACTION].to_numpy(dtype=np.int64)[1:]
        y_p = g[TARGET_POINT].to_numpy(dtype=np.int64)[1:]
        y_r = float(g[TARGET_RALLY].iloc[0])
        L = min(len(x), max_len)
        Xs.append(pad_2d(x, max_len))
        yAs.append(pad_1d(y_a, max_len))
        yPs.append(pad_1d(y_p, max_len))
        yRs.append(y_r)
        lens.append(max(1, L))
        groups.append(g[group_col].iloc[0] if group_col in g.columns else rid)
    return (
        np.stack(Xs), np.stack(yAs), np.stack(yPs),
        np.asarray(yRs, dtype=np.float32), np.asarray(lens, dtype=np.int64), np.asarray(groups),
    )


def build_test_sequences(test_df, feature_cols, maps, max_len):
    Xs, lens, rids = [], [], []
    test_df = test_df.sort_values(["rally_uid", "strikeNumber"])
    for rid, g in test_df.groupby("rally_uid", sort=False):
        if len(g) > max_len:
            g = g.iloc[-max_len:]
        x = encode_frame(g, feature_cols, maps)
        Xs.append(pad_2d(x, max_len))
        lens.append(max(1, min(len(x), max_len)))
        rids.append(rid)
    return np.stack(Xs), np.asarray(lens, dtype=np.int64), np.asarray(rids)


def remap_targets(yA_raw, yP_raw, train_df):
    action_classes = np.sort(train_df[TARGET_ACTION].unique())
    point_classes = np.sort(train_df[TARGET_POINT].unique())
    action_map = {v: i for i, v in enumerate(action_classes)}
    point_map = {v: i for i, v in enumerate(point_classes)}
    yA = np.full(yA_raw.shape, IGNORE_INDEX, dtype=np.int64)
    yP = np.full(yP_raw.shape, IGNORE_INDEX, dtype=np.int64)
    for original, idx in action_map.items():
        yA[yA_raw == original] = idx
    for original, idx in point_map.items():
        yP[yP_raw == original] = idx
    return yA, yP, action_classes, point_classes


def class_weights(y, n_classes, power=0.5):
    valid = y[y != IGNORE_INDEX]
    counts = np.bincount(valid, minlength=n_classes).astype(np.float64) + 1.0
    w = (counts.sum() / counts) ** power
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def make_feature_indices(feature_cols):
    return {
        "player": [feature_cols.index(c) for c in PLAYER_FEATURES if c in feature_cols],
        "shot": [feature_cols.index(c) for c in SHOT_FEATURES if c in feature_cols],
        "score": [feature_cols.index(c) for c in SCORE_FEATURES if c in feature_cols],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# RallyDataset returns padded rally sequences.  During training it can create
# augmented copies by masking selected feature groups or by truncating the
# observed prefix, which encourages the model to be robust to partial context.
# ─────────────────────────────────────────────────────────────────────────────

class RallyDataset(Dataset):
    def __init__(self, X, L, yA=None, yP=None, yR=None,
                 augment=False, repeat_aug=0, feature_indices=None,
                 player_mask_prob=0.0, shot_mask_prob=0.0, score_mask_prob=0.0,
                 random_truncate_prob=0.0, min_truncate_len=3,
                 span_mask_prob=0.0, span_mask_max_len=3):
        self.X = torch.tensor(X, dtype=torch.long)
        self.L = torch.tensor(L, dtype=torch.long)
        self.yA = None if yA is None else torch.tensor(yA, dtype=torch.long)
        self.yP = None if yP is None else torch.tensor(yP, dtype=torch.long)
        self.yR = None if yR is None else torch.tensor(yR, dtype=torch.float32)
        self.augment = augment and self.yA is not None
        self.repeat_aug = max(0, int(repeat_aug)) if self.augment else 0
        self.feature_indices = feature_indices or {}
        self.player_mask_prob = player_mask_prob
        self.shot_mask_prob = shot_mask_prob
        self.score_mask_prob = score_mask_prob
        self.random_truncate_prob = random_truncate_prob
        self.min_truncate_len = max(1, int(min_truncate_len))
        self.span_mask_prob = span_mask_prob
        self.span_mask_max_len = max(1, int(span_mask_max_len))

    def __len__(self):
        if self.yA is None:
            return self.X.shape[0]
        return self.X.shape[0] * (1 + self.repeat_aug)

    def _mask_feature_group(self, x, length, cols, prob):
        if prob <= 0 or not cols or length <= 0:
            return
        mask = torch.rand((length, len(cols))) < prob
        if mask.any():
            sub = x[:length, cols]
            sub[mask] = PAD_TOKEN
            x[:length, cols] = sub

    def _random_truncate(self, x, yA, yP, length):
        if self.random_truncate_prob <= 0 or length <= self.min_truncate_len:
            return length
        if torch.rand(()) >= self.random_truncate_prob:
            return length
        new_len = int(torch.randint(self.min_truncate_len, length + 1, (1,)).item())
        x[new_len:] = PAD_TOKEN
        yA[new_len:] = IGNORE_INDEX
        yP[new_len:] = IGNORE_INDEX
        return new_len

    def _span_mask(self, x, length):
        if self.span_mask_prob <= 0 or length <= 1:
            return
        if torch.rand(()) >= self.span_mask_prob:
            return
        cols = self.feature_indices.get("shot", [])
        if not cols:
            return
        span_len = int(torch.randint(1, min(self.span_mask_max_len, length) + 1, (1,)).item())
        start = int(torch.randint(0, length - span_len + 1, (1,)).item())
        x[start:start + span_len, cols] = PAD_TOKEN

    def __getitem__(self, idx):
        base_idx = idx % self.X.shape[0]
        augmented_copy = self.augment and idx >= self.X.shape[0]
        x = self.X[base_idx].clone()
        L = self.L[base_idx].clone()
        if self.yA is None:
            return x, L
        yA = self.yA[base_idx].clone()
        yP = self.yP[base_idx].clone()
        yR = self.yR[base_idx].clone()
        if augmented_copy:
            length = int(L.item())
            length = self._random_truncate(x, yA, yP, length)
            L = torch.tensor(length, dtype=torch.long)
            self._mask_feature_group(x, length, self.feature_indices.get("player", []), self.player_mask_prob)
            self._mask_feature_group(x, length, self.feature_indices.get("shot", []), self.shot_mask_prob)
            self._mask_feature_group(x, length, self.feature_indices.get("score", []), self.score_mask_prob)
            self._span_mask(x, length)
        return x, L, yA, yP, yR


# ─────────────────────────────────────────────────────────────────────────────
# Model
# Categorical features are embedded independently, projected into a shared
# hidden space, and processed with a causal Transformer encoder.  Three output
# heads share the encoder representation and predict actionId, pointId, and
# serverGetPoint.
# ─────────────────────────────────────────────────────────────────────────────

class CausalTransformerMultiTask(nn.Module):
    def __init__(self, num_tokens_per_feature, n_action, n_point, max_len,
                 emb_dim=24, model_dim=192, n_heads=6, n_layers=3,
                 ff_dim=384, dropout=0.20,
                 head_type="mlp", head_hidden_ratio=0.5, head_dropout_scale=0.5):
        super().__init__()
        self.max_len = max_len
        self.embeddings = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN)
            for n in num_tokens_per_feature
        ])
        input_dim = len(num_tokens_per_feature) * emb_dim
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(max_len, model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(model_dim)

        # Optional action / point prediction heads.
        # - "linear": Dropout + Linear; used by the final submitted setting.
        # - "mlp": a two-layer non-linear head kept for ablation experiments.
        if head_type == "mlp":
            head_hidden = max(16, int(model_dim * head_hidden_ratio))
            head_dropout = min(0.8, max(0.0, dropout * head_dropout_scale))
            self.action_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, n_action),
            )
            self.point_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, n_point),
            )
        elif head_type == "linear":
            self.action_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(model_dim, n_action))
            self.point_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(model_dim, n_point))
        else:
            raise ValueError(f"Unsupported head_type: {head_type}. Use 'mlp' or 'linear'.")

        # Rally outcome head.  It predicts a logit for each prefix position;
        # the final submission uses the logit at the last valid time step.
        self.rally_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, X, lengths):
        B, T, _ = X.shape
        emb = torch.cat([layer(X[:, :, i]) for i, layer in enumerate(self.embeddings)], dim=-1)
        h = self.in_proj(emb)
        pos = torch.arange(T, device=X.device).unsqueeze(0)
        h = h + self.pos_emb(pos)
        key_padding_mask = torch.arange(T, device=X.device).unsqueeze(0) >= lengths.unsqueeze(1)
        causal_mask = torch.triu(torch.ones(T, T, device=X.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.action_head(h), self.point_head(h), self.rally_head(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Optional SWA: Stochastic Weight Averaging
# This block is kept as an experiment switch.  It is only active when --use_swa
# is provided; otherwise the ordinary best validation checkpoint is used.
# ─────────────────────────────────────────────────────────────────────────────

class SWABuffer:
    """
    Maintains a running average of model weights on CPU.

    The buffer stores a numerically stable running average of selected
    checkpoints.  It does not affect training unless --use_swa is enabled.
    """
    def __init__(self):
        self.avg_state: Optional[dict] = None
        self.n = 0

    def update(self, model: nn.Module):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self.avg_state is None:
            self.avg_state = state
            self.n = 1
        else:
            self.n += 1
            for k in self.avg_state:
                self.avg_state[k] += (state[k] - self.avg_state[k]) / self.n

    def apply_to(self, model: nn.Module, device):
        if self.avg_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.avg_state.items()})
            return True
        return False

    @property
    def has_data(self):
        return self.avg_state is not None


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# The total loss combines two token-level classification losses and one rally
# outcome BCE loss.  The rally loss can emphasize the final valid prefix because
# the submitted prediction is taken from the last observed position.
# ─────────────────────────────────────────────────────────────────────────────

def masked_bce_loss(logits_step, y, lengths, pos_weight=None, last_step_weight=2.0):
    """
    Compute masked BCE loss for the rally outcome task.

    The label y is rally-level, so it is expanded to all valid prefix positions.
    The last valid time step receives an additional weight controlled by
    last_step_weight, because the final prediction is produced from that
    position.  Samples with y < 0 are ignored for this task; this is useful when
    extra data should contribute only actionId / pointId supervision.
    """
    B, T = logits_step.shape
    sample_mask = (y >= 0).float().unsqueeze(1)
    y_clamped = y.clamp_min(0.0)
    target = y_clamped.unsqueeze(1).expand(B, T)
    step_idx = torch.arange(T, device=logits_step.device).unsqueeze(0)
    valid_mask = (step_idx < lengths.unsqueeze(1)).float() * sample_mask
    last_mask = (step_idx == (lengths - 1).unsqueeze(1)).float() * (last_step_weight - 1.0) * sample_mask
    weight_map = valid_mask + last_mask

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits_step, target, reduction="none", pos_weight=pos_weight
    )
    denom = weight_map.sum().clamp_min(1.0)
    return (loss * weight_map).sum() / denom


def compute_loss(model, batch, ce_a, ce_p, device, weights, pos_weight=None, last_step_weight=2.0):
    X, L, yA, yP, yR = batch
    X, L, yA, yP, yR = X.to(device), L.to(device), yA.to(device), yP.to(device), yR.to(device)
    la, lp, lr_step = model(X, L)
    loss_a = ce_a(la.reshape(-1, la.size(-1)), yA.reshape(-1))
    loss_p = ce_p(lp.reshape(-1, lp.size(-1)), yP.reshape(-1))
    loss_r = masked_bce_loss(lr_step, yR, L, pos_weight=pos_weight, last_step_weight=last_step_weight)
    loss = weights[0] * loss_a + weights[1] * loss_p + weights[2] * loss_r
    return loss, (loss_a.detach(), loss_p.detach(), loss_r.detach())


@torch.no_grad()
def evaluate(model, loader, ce_a, ce_p, device, weights, pos_weight=None, last_step_weight=1.0):
    model.eval()
    total_loss = 0.0
    trueA, predA, trueP, predP, trueR, probR = [], [], [], [], [], []
    for batch in loader:
        X, L, yA, yP, yR = batch
        loss, _ = compute_loss(model, batch, ce_a, ce_p, device, weights, pos_weight, last_step_weight)
        total_loss += loss.item() * X.size(0)
        X, L = X.to(device), L.to(device)
        la, lp, lr_step = model(X, L)
        pa = la.argmax(-1).cpu().numpy()
        pp = lp.argmax(-1).cpu().numpy()
        yA_np = yA.numpy()
        yP_np = yP.numpy()
        for i in range(X.size(0)):
            t = int(L[i].item()) - 1
            if yA_np[i, t] != IGNORE_INDEX:
                trueA.append(int(yA_np[i, t]))
                predA.append(int(pa[i, t]))
            if yP_np[i, t] != IGNORE_INDEX:
                trueP.append(int(yP_np[i, t]))
                predP.append(int(pp[i, t]))
        last_idx = (L - 1).view(-1, 1)
        last_logits = lr_step.gather(1, last_idx).squeeze(1)
        trueR.extend(yR.numpy().tolist())
        probR.extend(torch.sigmoid(last_logits).cpu().numpy().tolist())
    f1a = f1_score(trueA, predA, average="macro", zero_division=0) if trueA else 0.0
    f1p = f1_score(trueP, predP, average="macro", zero_division=0) if trueP else 0.0
    auc = roc_auc_score(trueR, probR) if len(set(trueR)) > 1 else 0.5
    overall = weights[0] * f1a + weights[1] * f1p + weights[2] * auc
    return {"loss": total_loss / max(1, len(loader.dataset)),
            "f1_action": f1a, "f1_point": f1p, "auc": auc, "overall": overall}


# ─────────────────────────────────────────────────────────────────────────────
# Optional TTA: Test Time Augmentation
# This function is inactive when --tta_n 1.  It remains in the script as an
# optional inference-time experiment and is not used by the final configuration.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_proba_tta(model, X_test, L_test, device, n_action, n_point,
                      feature_indices, player_mask_prob=0.08, score_mask_prob=0.02,
                      shot_mask_prob=0.04, tta_n=4, batch_size=256):
    """
    Predict probabilities with optional test-time masking.

    The first pass uses the original input without masking.  Additional passes
    randomly mask selected feature groups and average the resulting probabilities.
    When tta_n=1, this function is normally not called by the main inference
    path, and standard prediction is used instead.
    """
    model.eval()
    sum_pa = np.zeros((len(X_test), n_action), dtype=np.float64)
    sum_pp = np.zeros((len(X_test), n_point), dtype=np.float64)
    sum_pr = np.zeros((len(X_test),), dtype=np.float64)
    player_cols = feature_indices.get("player", [])
    score_cols = feature_indices.get("score", [])
    shot_cols = feature_indices.get("shot", [])

    X_tensor = torch.tensor(X_test, dtype=torch.long)
    L_tensor = torch.tensor(L_test, dtype=torch.long)

    for aug_i in range(tta_n):
        all_pa, all_pp, all_pr = [], [], []
        for start in range(0, len(X_test), batch_size):
            xb = X_tensor[start:start + batch_size].clone().to(device)
            lb = L_tensor[start:start + batch_size].to(device)
            B = xb.size(0)

            # The first view is the original sequence; only later views are masked.
            if aug_i > 0:
                lengths = lb.cpu().tolist()
                for bi in range(B):
                    length = lengths[bi]
                    if player_mask_prob > 0 and player_cols:
                        mask = torch.rand((length, len(player_cols))) < player_mask_prob
                        xb[bi, :length][:, player_cols] = xb[bi, :length][:, player_cols].masked_fill(mask.to(device), PAD_TOKEN)
                    if score_mask_prob > 0 and score_cols:
                        mask = torch.rand((length, len(score_cols))) < score_mask_prob
                        xb[bi, :length][:, score_cols] = xb[bi, :length][:, score_cols].masked_fill(mask.to(device), PAD_TOKEN)
                    if shot_mask_prob > 0 and shot_cols:
                        mask = torch.rand((length, len(shot_cols))) < shot_mask_prob
                        xb[bi, :length][:, shot_cols] = xb[bi, :length][:, shot_cols].masked_fill(mask.to(device), PAD_TOKEN)

            la, lp, lr_step = model(xb, lb)
            idx = (lb - 1).view(B, 1, 1)
            la_last = la.gather(1, idx.expand(B, 1, n_action)).squeeze(1)
            lp_last = lp.gather(1, idx.expand(B, 1, n_point)).squeeze(1)
            lr_last = lr_step.gather(1, (lb - 1).view(B, 1)).squeeze(1)
            all_pa.append(torch.softmax(la_last, dim=-1).cpu().numpy())
            all_pp.append(torch.softmax(lp_last, dim=-1).cpu().numpy())
            all_pr.append(torch.sigmoid(lr_last).cpu().numpy())

        sum_pa += np.vstack(all_pa)
        sum_pp += np.vstack(all_pp)
        sum_pr += np.concatenate(all_pr)

    return sum_pa / tta_n, sum_pp / tta_n, sum_pr / tta_n


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers
# Prefer GroupKFold when enough groups are available.  This reduces leakage by
# keeping samples from the same group out of both training and validation within
# the same fold.
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(groups, yR, n_folds, val_size, seed):
    idx = np.arange(len(groups))
    unique_groups = np.unique(groups)
    if n_folds >= 2 and len(unique_groups) >= n_folds:
        splitter = GroupKFold(n_splits=n_folds)
        return list(splitter.split(idx, yR, groups=groups))
    try:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        return [next(splitter.split(idx, yR, groups=groups))]
    except Exception:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        return [next(splitter.split(idx, yR))]


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler
# Linear warmup stabilizes early optimization.  Cosine decay then gradually
# lowers the learning rate during the rest of training.
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup followed by cosine decay.

    The returned multiplier is applied to the optimizer learning rate.  During
    warmup, the multiplier increases linearly.  After warmup, it follows a cosine
    curve down to eta_min_ratio.
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, eta_min_ratio: float = 0.05):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, epoch: int) -> float:
        # The scheduler receives a zero-based epoch index from PyTorch.
        if epoch < self.warmup_epochs:
            return (epoch + 1) / max(1, self.warmup_epochs)
        t = epoch - self.warmup_epochs
        T = max(1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * t / T))
        return self.eta_min_ratio + (1.0 - self.eta_min_ratio) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# Train one fold
# This routine trains one fold model, optionally appends old labeled test data
# to the training split, evaluates on the validation split, and returns either a
# single best model or task-specific / task-averaged model collections.
# ─────────────────────────────────────────────────────────────────────────────

def make_model(args, sizes, n_action, n_point, max_len, device):
    return CausalTransformerMultiTask(
        sizes, n_action, n_point, max_len,
        emb_dim=args.emb_dim, model_dim=args.model_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, ff_dim=args.ff_dim, dropout=args.dropout,
        head_type=args.head_type,
        head_hidden_ratio=args.head_hidden_ratio,
        head_dropout_scale=args.head_dropout_scale,
    ).to(device)


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_one_fold(fold_id, tr_idx, va_idx, arrays, sizes, n_action, n_point, max_len, args, device):
    seed_everything(args.seed + fold_id)
    feature_indices = make_feature_indices(FEATURES)

    # Training portion selected from the official train.csv split.
    train_X = arrays["X"][tr_idx]
    train_L = arrays["L"][tr_idx]
    train_yA = arrays["yA"][tr_idx]
    train_yP = arrays["yP"][tr_idx]
    train_yR = arrays["yR"][tr_idx]

    # Optional supervised data from the old labeled test.csv.
    # This data is appended only to the training side of each fold.
    # The validation fold remains part of the official train.csv split.
    if arrays.get("extra_X") is not None and len(arrays["extra_X"]) > 0 and args.extra_weight > 0:
        rng = np.random.default_rng(args.seed * 1000 + fold_id)
        n_extra = len(arrays["extra_X"])
        if args.extra_weight < 1.0:
            keep_n = max(1, int(round(n_extra * args.extra_weight)))
            ex_idx = rng.choice(n_extra, size=keep_n, replace=False)
        else:
            full_rep = int(math.floor(args.extra_weight))
            frac = args.extra_weight - full_rep
            chunks = [np.arange(n_extra) for _ in range(full_rep)]
            if frac > 1e-9:
                keep_n = max(1, int(round(n_extra * frac)))
                chunks.append(rng.choice(n_extra, size=keep_n, replace=False))
            ex_idx = np.concatenate(chunks) if chunks else np.array([], dtype=int)
        if len(ex_idx) > 0:
            train_X = np.concatenate([train_X, arrays["extra_X"][ex_idx]], axis=0)
            train_L = np.concatenate([train_L, arrays["extra_L"][ex_idx]], axis=0)
            train_yA = np.concatenate([train_yA, arrays["extra_yA"][ex_idx]], axis=0)
            train_yP = np.concatenate([train_yP, arrays["extra_yP"][ex_idx]], axis=0)
            train_yR = np.concatenate([train_yR, arrays["extra_yR"][ex_idx]], axis=0)
            print(f"[Fold {fold_id}] extra_old_test appended: {len(ex_idx)} samples (weight={args.extra_weight}, use_server={args.extra_use_server_label})")

    train_ds = RallyDataset(
        train_X, train_L, train_yA, train_yP, train_yR,
        augment=args.augment, repeat_aug=args.repeat_aug,
        feature_indices=feature_indices,
        player_mask_prob=args.player_mask_prob,
        shot_mask_prob=args.shot_mask_prob,
        score_mask_prob=args.score_mask_prob,
        random_truncate_prob=args.random_truncate_prob,
        min_truncate_len=args.min_truncate_len,
        span_mask_prob=args.span_mask_prob,
        span_mask_max_len=args.span_mask_max_len,
    )
    val_ds = RallyDataset(
        arrays["X"][va_idx], arrays["L"][va_idx],
        arrays["yA"][va_idx], arrays["yP"][va_idx], arrays["yR"][va_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")

    model = make_model(args, sizes, n_action, n_point, max_len, device)

    aw = class_weights(train_yA, n_action, power=args.class_weight_power).to(device)
    pw = class_weights(train_yP, n_point, power=args.class_weight_power).to(device)
    ce_a = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=aw, label_smoothing=args.label_smoothing)
    ce_p = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=pw, label_smoothing=args.label_smoothing)

    valid_rally = train_yR >= 0
    pos = train_yR[valid_rally].sum() if np.any(valid_rally) else 0.0
    neg = valid_rally.sum() - pos
    pos_weight = (torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
                  if args.rally_pos_weight else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        eta_min_ratio=0.05,
    )
    weights = (args.w_action, args.w_point, args.w_rally)

    best_score, best_state, best_epoch = -1.0, None, 0
    best_action, best_action_state, best_action_epoch = -1.0, None, 0
    best_point, best_point_state, best_point_epoch = -1.0, None, 0
    best_auc, best_auc_state, best_auc_epoch = -1.0, None, 0
    # For task-averaged checkpoint prediction, keep evaluated epoch states and
    # validation metrics.  This optional path is inactive unless the corresponding
    # command-line flag is set.
    epoch_records = []
    patience_count = 0

    swa = SWABuffer()
    swa_start = max(1, int(args.swa_start_ratio * args.epochs))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_loss(model, batch, ce_a, ce_p, device, weights,
                                   pos_weight=pos_weight, last_step_weight=args.last_step_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item() * batch[0].size(0)
        scheduler.step()

        if epoch >= swa_start and (epoch - swa_start) % args.swa_freq == 0:
            swa.update(model)

        metrics = evaluate(model, val_loader, ce_a, ce_p, device, weights,
                           pos_weight=pos_weight, last_step_weight=args.last_step_weight)
        train_loss /= max(1, len(train_ds))
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Fold {fold_id} Ep {epoch:02d}] lr={current_lr:.2e} "
            f"train={train_loss:.5f} val={metrics['loss']:.5f} "
            f"F1a={metrics['f1_action']:.4f} F1p={metrics['f1_point']:.4f} "
            f"AUC={metrics['auc']:.4f} Overall={metrics['overall']:.5f}"
            + (" [SWA]" if epoch >= swa_start and (epoch - swa_start) % args.swa_freq == 0 else "")
        )

        state_now = clone_state(model) if args.task_avg_checkpoints else None
        if args.task_avg_checkpoints:
            epoch_records.append({
                "epoch": epoch,
                "overall": float(metrics["overall"]),
                "f1_action": float(metrics["f1_action"]),
                "f1_point": float(metrics["f1_point"]),
                "auc": float(metrics["auc"]),
                "state": state_now,
            })
        if metrics["overall"] > best_score:
            if state_now is None:
                state_now = clone_state(model)
            best_score = metrics["overall"]
            best_epoch = epoch
            best_state = state_now
            patience_count = 0
        else:
            patience_count += 1

        # Task-specific best states. Use the same state clone when possible.
        if args.task_specific_checkpoints:
            if metrics["f1_action"] > best_action:
                if state_now is None:
                    state_now = clone_state(model)
                best_action = metrics["f1_action"]
                best_action_epoch = epoch
                best_action_state = state_now
            if metrics["f1_point"] > best_point:
                if state_now is None:
                    state_now = clone_state(model)
                best_point = metrics["f1_point"]
                best_point_epoch = epoch
                best_point_state = state_now
            if metrics["auc"] > best_auc:
                if state_now is None:
                    state_now = clone_state(model)
                best_auc = metrics["auc"]
                best_auc_epoch = epoch
                best_auc_state = state_now

        if patience_count >= args.patience:
            print(f"[Fold {fold_id}] Early stop ep={epoch}; best_ep={best_epoch}, best={best_score:.5f}")
            break

    # If SWA is enabled, compare its validation score against the best ordinary
    # checkpoint.  Without --use_swa, this block is skipped.
    if swa.has_data and args.use_swa:
        swa.apply_to(model, device)
        swa_metrics = evaluate(model, val_loader, ce_a, ce_p, device, weights,
                               pos_weight=pos_weight, last_step_weight=args.last_step_weight)
        print(f"[Fold {fold_id}] SWA Overall={swa_metrics['overall']:.5f} vs Best={best_score:.5f}")
        if swa_metrics["overall"] > best_score:
            best_score = swa_metrics["overall"]
            best_state = clone_state(model)
            print(f"[Fold {fold_id}] → 採用 SWA 權重 (score={best_score:.5f})")
        else:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    save_path = Path(args.model_dir) / f"fold{fold_id}_seed{args.seed}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)

    if args.task_avg_checkpoints:
        def select_records(metric_name, topk):
            if not epoch_records:
                return [{"epoch": best_epoch, metric_name: best_score, "state": best_state}]
            return sorted(epoch_records, key=lambda r: r[metric_name], reverse=True)[:max(1, int(topk))]

        def states_for_source(source, metric_name):
            if source == "overall":
                return [(best_epoch, best_score, best_state)]
            if source.endswith("_best"):
                recs = select_records(metric_name, 1)
            else:
                recs = select_records(metric_name, args.task_avg_topk)
            return [(r["epoch"], r[metric_name], r["state"]) for r in recs]

        action_sel = states_for_source(args.task_action_source, "f1_action")
        point_sel = states_for_source(args.task_point_source, "f1_point")
        rally_sel = states_for_source(args.task_rally_source, "auc")

        print(
            f"[Fold {fold_id}] task-avg selections | "
            f"overall={best_score:.5f}@{best_epoch} | "
            f"action_source={args.task_action_source}: "
            + ",".join([f"{v:.5f}@{e}" for e, v, _ in action_sel])
        )
        print(
            f"[Fold {fold_id}] point_source={args.task_point_source}: "
            + ",".join([f"{v:.5f}@{e}" for e, v, _ in point_sel])
            + f" | rally_source={args.task_rally_source}: "
            + ",".join([f"{v:.5f}@{e}" for e, v, _ in rally_sel])
        )

        if args.save_task_ckpts:
            for name, selected in [("action", action_sel), ("point", point_sel), ("rally", rally_sel)]:
                for rank, (ep, val, state) in enumerate(selected, start=1):
                    torch.save(state, Path(args.model_dir) / f"fold{fold_id}_seed{args.seed}_{name}_top{rank}_ep{ep}.pt")

        models = {"action": [], "point": [], "rally": []}
        for name, selected in [("action", action_sel), ("point", point_sel), ("rally", rally_sel)]:
            for ep, val, state in selected:
                m = make_model(args, sizes, n_action, n_point, max_len, device)
                m.load_state_dict({k: v.to(device) for k, v in state.items()})
                m.eval()
                models[name].append(m)
        print(f"[Fold {fold_id}] saved: {save_path} | best_overall={best_score:.5f} | task-averaged prediction enabled")
        return models, best_score

    if args.task_specific_checkpoints:
        # Fallback safety: if a task-specific checkpoint was never selected,
        # reuse the overall best checkpoint for that task.
        best_action_state = best_action_state or best_state
        best_point_state = best_point_state or best_state
        best_auc_state = best_auc_state or best_state
        print(
            f"[Fold {fold_id}] task bests | "
            f"overall={best_score:.5f}@{best_epoch} "
            f"action={best_action:.5f}@{best_action_epoch} "
            f"point={best_point:.5f}@{best_point_epoch} "
            f"auc={best_auc:.5f}@{best_auc_epoch}"
        )
        if args.save_task_ckpts:
            torch.save(best_action_state, Path(args.model_dir) / f"fold{fold_id}_seed{args.seed}_action.pt")
            torch.save(best_point_state, Path(args.model_dir) / f"fold{fold_id}_seed{args.seed}_point.pt")
            torch.save(best_auc_state, Path(args.model_dir) / f"fold{fold_id}_seed{args.seed}_rally.pt")

        models = {}
        for name, state in [("action", best_action_state), ("point", best_point_state), ("rally", best_auc_state)]:
            m = make_model(args, sizes, n_action, n_point, max_len, device)
            m.load_state_dict({k: v.to(device) for k, v in state.items()})
            m.eval()
            models[name] = m
        print(f"[Fold {fold_id}] saved: {save_path} | best_overall={best_score:.5f} | task-specific prediction enabled")
        return models, best_score

    print(f"[Fold {fold_id}] saved: {save_path} | best={best_score:.5f}")
    return model, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# Loads data, builds encoders and sequences, trains fold models for each seed,
# averages test predictions, and writes the final submission CSV.
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # Parse comma-separated random seeds.  The final submitted configuration uses
    # a single seed, but multiple seeds can be supplied for experimentation.
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    print(f"Multi-seed ensemble: seeds={seeds}, folds={args.folds}, max_folds={args.max_folds}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"裝置: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    train_df = add_features(train_df)
    test_df = add_features(test_df)

    extra_df = None
    if args.extra_old_test:
        extra_df = pd.read_csv(args.extra_old_test)
        extra_df = add_features(extra_df)

    check_columns(train_df, ["rally_uid", TARGET_RALLY, args.group_col] + FEATURES, "train")
    check_columns(test_df, ["rally_uid"] + FEATURES, "test")
    if extra_df is not None:
        check_columns(extra_df, ["rally_uid", TARGET_RALLY, args.group_col] + FEATURES, "extra_old_test")

    if args.clip_strike > 0:
        train_df["strikeNumber"] = train_df["strikeNumber"].clip(0, args.clip_strike)
        test_df["strikeNumber"] = test_df["strikeNumber"].clip(0, args.clip_strike)
        if extra_df is not None:
            extra_df["strikeNumber"] = extra_df["strikeNumber"].clip(0, args.clip_strike)

    max_candidates = [train_df.groupby("rally_uid").size().max() - 1, test_df.groupby("rally_uid").size().max()]
    if extra_df is not None:
        max_candidates.append(extra_df.groupby("rally_uid").size().max() - 1)
    obs_max = int(max(max_candidates))
    max_len = args.max_len if args.max_len > 0 else obs_max
    print(f"max_len={max_len}")

    # Include old labeled test data in categorical maps so its player/action/point
    # values are represented explicitly instead of being mapped to PAD.
    map_df = pd.concat([train_df, extra_df], axis=0, ignore_index=True) if extra_df is not None else train_df
    maps, sizes = build_maps(map_df, FEATURES)
    X, yA_raw, yP_raw, yR, L, groups = build_train_sequences(
        train_df, FEATURES, maps, max_len, args.group_col
    )

    extra_arrays = None
    if extra_df is not None:
        eX, eyA_raw, eyP_raw, eyR, eL, egroups = build_train_sequences(
            extra_df, FEATURES, maps, max_len, args.group_col
        )
        target_map_df = pd.concat([train_df[[TARGET_ACTION, TARGET_POINT]], extra_df[[TARGET_ACTION, TARGET_POINT]]], axis=0, ignore_index=True)
    else:
        target_map_df = train_df

    yA, yP, action_classes, point_classes = remap_targets(yA_raw, yP_raw, target_map_df)
    if extra_df is not None:
        eyA, eyP, _, _ = remap_targets(eyA_raw, eyP_raw, target_map_df)
        if not args.extra_use_server_label:
            eyR = np.full_like(eyR, -1.0, dtype=np.float32)
        extra_arrays = {"X": eX, "L": eL, "yA": eyA, "yP": eyP, "yR": eyR}

    print(f"特徵數={len(FEATURES)}: {FEATURES}")
    print(f"rally 數={len(X)}, group 數={len(np.unique(groups))}")
    if extra_arrays is not None:
        print(f"extra_old_test transitions={len(extra_arrays['X'])}, extra_weight={args.extra_weight}, extra_use_server_label={args.extra_use_server_label}")

    X_test, L_test, test_rids = build_test_sequences(test_df, FEATURES, maps, max_len)
    test_ds = RallyDataset(X_test, L_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=0, pin_memory=device.type == "cuda")

    n_action, n_point = len(action_classes), len(point_classes)
    sum_pa = np.zeros((len(X_test), n_action), dtype=np.float64)
    sum_pp = np.zeros((len(X_test), n_point), dtype=np.float64)
    sum_pr = np.zeros((len(X_test),), dtype=np.float64)
    all_fold_scores = []
    n_models = 0

    arrays = {"X": X, "L": L, "yA": yA, "yP": yP, "yR": yR}
    if extra_arrays is not None:
        arrays.update({
            "extra_X": extra_arrays["X"],
            "extra_L": extra_arrays["L"],
            "extra_yA": extra_arrays["yA"],
            "extra_yP": extra_arrays["yP"],
            "extra_yR": extra_arrays["yR"],
        })

    # Outer loop over seeds.  Predictions from all seed×fold models are averaged.
    for seed in seeds:
        args.seed = seed
        seed_everything(seed)
        splits = make_splits(groups, yR, args.folds, args.val_size, seed)
        if args.max_folds > 0:
            splits = splits[:args.max_folds]

        for fold_id, (tr_idx, va_idx) in enumerate(splits, start=1):
            print(f"\n===== Seed {seed} | Fold {fold_id}/{len(splits)} | train={len(tr_idx)} val={len(va_idx)} =====")
            model, score = train_one_fold(
                fold_id, tr_idx, va_idx, arrays, sizes, n_action, n_point, max_len, args, device
            )

            # Optional task-averaged checkpoint prediction:
            # within each fold, average top-k checkpoints for selected task sources.
            if args.task_avg_checkpoints:
                pa_list, pp_list, pr_list = [], [], []
                for m in model["action"]:
                    pa_i, _, _ = _predict_proba_standard(m, test_loader, device, n_action, n_point)
                    pa_list.append(pa_i)
                for m in model["point"]:
                    _, pp_i, _ = _predict_proba_standard(m, test_loader, device, n_action, n_point)
                    pp_list.append(pp_i)
                for m in model["rally"]:
                    _, _, pr_i = _predict_proba_standard(m, test_loader, device, n_action, n_point)
                    pr_list.append(pr_i)
                pa = np.mean(pa_list, axis=0)
                pp = np.mean(pp_list, axis=0)
                pr = np.mean(pr_list, axis=0)
            # Optional task-specific checkpoint prediction:
            # actionId uses the best F1_action checkpoint, pointId uses the best
            # F1_point checkpoint, and serverGetPoint uses the best AUC checkpoint.
            elif args.task_specific_checkpoints:
                pa, _, _ = _predict_proba_standard(model["action"], test_loader, device, n_action, n_point)
                _, pp, _ = _predict_proba_standard(model["point"], test_loader, device, n_action, n_point)
                _, _, pr = _predict_proba_standard(model["rally"], test_loader, device, n_action, n_point)
            else:
                # Use TTA only when args.tta_n > 1; otherwise run standard inference.
                feature_indices = make_feature_indices(FEATURES)
                if args.tta_n > 1:
                    pa, pp, pr = predict_proba_tta(
                        model, X_test, L_test, device, n_action, n_point, feature_indices,
                        player_mask_prob=args.player_mask_prob * 0.5,
                        score_mask_prob=args.score_mask_prob * 0.5,
                        shot_mask_prob=args.shot_mask_prob * 0.5,
                        tta_n=args.tta_n, batch_size=args.batch_size * 2,
                    )
                else:
                    pa, pp, pr = _predict_proba_standard(model, test_loader, device, n_action, n_point)

            sum_pa += pa
            sum_pp += pp
            sum_pr += pr
            all_fold_scores.append(score)
            n_models += 1
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    avg_pa = sum_pa / n_models
    avg_pp = sum_pp / n_models
    avg_pr = sum_pr / n_models

    pred_action = action_classes[avg_pa.argmax(axis=1)].astype(int)
    pred_point = point_classes[avg_pp.argmax(axis=1)].astype(int)

    out = pd.DataFrame({
        "rally_uid": test_rids,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": avg_pr.astype(float),
    })

    sample_path = Path(args.sample)
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if "rally_uid" in sample.columns:
            out = sample[["rally_uid"]].merge(out, on="rally_uid", how="left")

    out["actionId"] = out["actionId"].fillna(train_df[TARGET_ACTION].mode()[0]).astype(int)
    out["pointId"] = out["pointId"].fillna(train_df[TARGET_POINT].mode()[0]).astype(int)
    out["serverGetPoint"] = out["serverGetPoint"].fillna(float(train_df[TARGET_RALLY].mean())).astype(float)
    if args.binary_rally:
        out["serverGetPoint"] = (out["serverGetPoint"] >= args.rally_threshold).astype(int)

    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")

    print(f"\n============================")
    print(f"seeds={seeds}, total models={n_models}")
    print(f"CV scores: {[round(s, 5) for s in all_fold_scores]}")
    print(f"CV mean={np.mean(all_fold_scores):.5f} std={np.std(all_fold_scores):.5f}")
    print(f"輸出: {args.out}")
    print(out.head())
    print("缺值:\n", out.isna().sum())


@torch.no_grad()
def _predict_proba_standard(model, loader, device, n_action, n_point):
    model.eval()
    all_pa, all_pp, all_pr = [], [], []
    for X, L in loader:
        X, L = X.to(device), L.to(device)
        la, lp, lr_step = model(X, L)
        B = X.size(0)
        idx = (L - 1).view(B, 1, 1)
        la_last = la.gather(1, idx.expand(B, 1, n_action)).squeeze(1)
        lp_last = lp.gather(1, idx.expand(B, 1, n_point)).squeeze(1)
        lr_last = lr_step.gather(1, (L - 1).view(B, 1)).squeeze(1)
        all_pa.append(torch.softmax(la_last, dim=-1).cpu().numpy())
        all_pp.append(torch.softmax(lp_last, dim=-1).cpu().numpy())
        all_pr.append(torch.sigmoid(lr_last).cpu().numpy())
    return np.vstack(all_pa), np.vstack(all_pp), np.concatenate(all_pr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--out", default="submission_v2_taskavg.csv")
    parser.add_argument("--model_dir", default="models_v2_extraoldtest")
    parser.add_argument("--extra_old_test", default="", help="舊版 test.csv；只加入每個 fold 的 training side，不進 validation")
    parser.add_argument("--extra_weight", type=float, default=1.0, help="舊 test 訓練樣本權重/抽樣比例；1.0=全部加入，0.5=每 fold 隨機取半數，2.0=重複兩次")
    parser.add_argument("--extra_use_server_label", type=int, default=1, help="1=使用舊 test 的 serverGetPoint 作為 rally label；0=舊 test 只訓練 action/point transition")

    # Multi-seed option: comma-separated values, e.g. "42,2024,2025,2026".
    parser.add_argument("--seeds", default="2026", help="逗號分隔的 seed 列表，每個 seed 跑完整 folds")
    parser.add_argument("--seed", type=int, default=2026, help="內部用，不需手動設定")

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_folds", type=int, default=0, help="0=全部 folds")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--group_col", default="match")

    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--emb_dim", type=int, default=24)
    parser.add_argument("--model_dim", type=int, default=192)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--ff_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.28)

    # Action / point classification head option.  The final configuration uses
    # "linear"; "mlp" is kept for optional ablation experiments.
    parser.add_argument("--head_type", default="linear", choices=["mlp", "linear"],
                        help="action/point head 類型：mlp=2-layer MLP；linear=Dropout+Linear")
    parser.add_argument("--head_hidden_ratio", type=float, default=0.5,
                        help="MLP head hidden dim = model_dim * ratio")
    parser.add_argument("--head_dropout_scale", type=float, default=0.5,
                        help="MLP head 中間 dropout = dropout * scale")

    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument("--rally_pos_weight", action="store_true")

    # Learning-rate warmup
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Linear warmup 的 epoch 數，之後接 cosine decay")

    # Optional SWA
    parser.add_argument("--use_swa", action="store_true", help="啟用 SWA 平均（建議與 --epochs 40+ 搭配）")
    parser.add_argument("--swa_start_ratio", type=float, default=0.6,
                        help="從第 swa_start_ratio*epochs epoch 後開始累積 SWA")
    parser.add_argument("--swa_freq", type=int, default=2,
                        help="每幾個 epoch 累積一次 SWA 快照")

    # Rally last-step upweighting
    parser.add_argument("--last_step_weight", type=float, default=1.5,
                        help="最後一步的 rally BCE loss 額外乘數（1.0=關閉，2.0=加倍）")

    # Optional Test Time Augmentation
    parser.add_argument("--tta_n", type=int, default=1,
                        help="Test Time Augmentation 次數；1=關閉，建議 4~8")

    # Data augmentation options
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--repeat_aug", type=int, default=1)
    parser.add_argument("--player_mask_prob", type=float, default=0.10)
    parser.add_argument("--shot_mask_prob", type=float, default=0.01)
    parser.add_argument("--score_mask_prob", type=float, default=0.01)
    parser.add_argument("--random_truncate_prob", type=float, default=0.25)
    parser.add_argument("--min_truncate_len", type=int, default=3)
    parser.add_argument("--span_mask_prob", type=float, default=0.03)
    parser.add_argument("--span_mask_max_len", type=int, default=3)

    parser.add_argument("--w_action", type=float, default=0.40)
    parser.add_argument("--w_point", type=float, default=0.40)
    parser.add_argument("--w_rally", type=float, default=0.20)

    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--clip_strike", type=int, default=80)
    parser.add_argument("--binary_rally", action="store_true")
    parser.add_argument("--rally_threshold", type=float, default=0.5)
    parser.add_argument("--task_avg_checkpoints", action="store_true",
                        help="每個 fold 針對各任務選 top-k checkpoint 做機率平均，降低 single-best epoch 的 validation noise")
    parser.add_argument("--task_avg_topk", type=int, default=3,
                        help="task_avg_checkpoints 使用的 top-k checkpoint 數量")
    parser.add_argument("--task_action_source", default="overall",
                        choices=["overall", "action_best", "action_topk"],
                        help="actionId 預測來源：overall 使用 best-overall；action_best 使用單一 F1_action 最佳；action_topk 使用 top-k F1_action 平均")
    parser.add_argument("--task_point_source", default="point_topk",
                        choices=["overall", "point_best", "point_topk"],
                        help="pointId 預測來源：overall / 單一 F1_point 最佳 / top-k F1_point 平均")
    parser.add_argument("--task_rally_source", default="auc_topk",
                        choices=["overall", "auc_best", "auc_topk"],
                        help="serverGetPoint 預測來源：overall / 單一 AUC 最佳 / top-k AUC 平均")
    parser.add_argument("--task_specific_checkpoints", action="store_true",
                        help="每個 fold 分別用 val F1_action / F1_point / AUC 最佳 checkpoint 預測三個任務")
    parser.add_argument("--save_task_ckpts", action="store_true",
                        help="另外儲存 action/point/rally 各自最佳 checkpoint")
    parser.add_argument("--force_cpu", action="store_true")
    main(parser.parse_args())