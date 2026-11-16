#!/usr/bin/env python3
"""
Project 95: Lip Reading from Silent Video using Vision Transformers
===================================================================
Visual Speech Recognition (VSR) deep learning system that recognizes spoken words
from silent video sequences of a speaker's mouth region without audio input.

Key Architecture:
  1. Spatiotemporal Front-End (3D CNN): Captures localized visual micro-motion
     and downsamples 64x64 mouth frames into 4x4 spatial token grids.
  2. Factorized Space-Time Vision Transformer (ViT):
     - Spatial Multi-Head Self-Attention: Learns lip contour, aperture, and teeth geometry.
     - Temporal Multi-Head Self-Attention: Learns coarticulation and viseme transition kinetics.
  3. Multi-Task Supervised Heads:
     - Spoken Word Classification Head (Top-1 / Top-3 recognition).
     - Frame-Level Viseme Trajectory Head (Viseme alignment across 24 frames).
  4. Realistic Procedural Kinematic Lip Synthesizer: Simulates human speech motor control
     parameters (vertical aperture, lip width, puckering, teeth/tongue exposure, jaw drop).
"""

import os
import math
import time
import json
import random
from collections import Counter

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Fix random seeds for reproducible experiments
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spoken Vocabulary (10 distinct words)
VOCABULARY = [
    "HELLO",
    "STOP",
    "THANK_YOU",
    "DANGER",
    "WELCOME",
    "GOODBYE",
    "PLEASE",
    "OPEN",
    "CLOSE",
    "SILENCE"
]
NUM_WORDS = len(VOCABULARY)
WORD_TO_IDX = {w: i for i, w in enumerate(VOCABULARY)}

# Viseme Categories (Visual Phonetic Classes)
VISEMES = [
    "CLOSED",          # 0: Bilabial closure (/p/, /b/, /m/)
    "NARROW_OPEN",     # 1: Slight neutral aperture
    "WIDE_OPEN",       # 2: Broad vertical jaw drop (/a/, /ae/, /o/)
    "ROUNDED_PUCKER",  # 3: Labial rounding / protrusion (/u/, /o/, /w/)
    "STRETCHED_WIDE",  # 4: Lateral corner retraction (/i/, /e/)
    "DENTAL_CONTACT"   # 5: Teeth exposure / contact (/s/, /z/, /f/, /th/)
]
NUM_VISEMES = len(VISEMES)
VISEME_TO_IDX = {v: i for i, v in enumerate(VISEMES)}

NUM_FRAMES = 24       # ~1.0 second at 25 fps
IMG_SIZE = 64         # 64x64 mouth ROI


# ============================================================================
# 1. Procedural Kinematic Lip Video Synthesizer
# ============================================================================

def interpolate_spline(keyframes, num_frames=NUM_FRAMES):
    """
    Interpolates parameter trajectories (h, w, r, teeth, jaw, viseme)
    smoothly using cosine easing across frames.
    """
    # keyframes: list of tuples (norm_t, h, w, r, teeth, jaw, viseme_id)
    times = [kf[0] for kf in keyframes]
    h_vals = [kf[1] for kf in keyframes]
    w_vals = [kf[2] for kf in keyframes]
    r_vals = [kf[3] for kf in keyframes]
    teeth_vals = [kf[4] for kf in keyframes]
    jaw_vals = [kf[5] for kf in keyframes]
    vis_vals = [kf[6] for kf in keyframes]

    t_grid = np.linspace(0.0, 1.0, num_frames)
    h_traj = np.interp(t_grid, times, h_vals)
    w_traj = np.interp(t_grid, times, w_vals)
    r_traj = np.interp(t_grid, times, r_vals)
    teeth_traj = np.interp(t_grid, times, teeth_vals)
    jaw_traj = np.interp(t_grid, times, jaw_vals)
    
    # Nearest keyframe for discrete viseme label
    vis_traj = []
    for t in t_grid:
        closest_idx = np.argmin(np.abs(np.array(times) - t))
        vis_traj.append(vis_vals[closest_idx])

    return list(zip(h_traj, w_traj, r_traj, teeth_traj, jaw_traj, vis_traj))


WORD_KEYFRAMES = {
    "HELLO": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1), # Neutral
        (0.25, 7.5, 35.0, 0.0, 0.4, 1.0, 4), # Stretch /eh/
        (0.55, 4.0, 32.0, 0.0, 0.8, 1.0, 5), # Dental /l/
        (0.80, 6.0, 20.0, 1.2, 0.0, 1.0, 3), # Rounded /oh/
        (1.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1)  # Rest
    ],
    "STOP": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1),
        (0.20, 2.0, 34.0, 0.0, 0.9, 0.0, 5), # Teeth /s/
        (0.45, 3.0, 31.0, 0.0, 0.7, 0.0, 5), # Dental /t/
        (0.70, 14.0, 31.0, 0.0, 0.5, 4.0, 2), # Wide open /ah/
        (0.92, 0.0, 27.0, 0.0, 0.0, 0.0, 0), # Bilabial stop /p/
        (1.00, 0.0, 27.0, 0.0, 0.0, 0.0, 0)
    ],
    "THANK_YOU": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1),
        (0.20, 3.0, 33.0, 0.0, 0.9, 0.0, 5), # Tongue-teeth /th/
        (0.45, 12.0, 36.0, 0.0, 0.6, 3.0, 2), # Wide /ae/
        (0.65, 4.0, 34.0, 0.0, 0.5, 1.0, 4), # Spread /y/
        (0.85, 5.0, 19.0, 1.3, 0.0, 1.0, 3), # Puckered /oo/
        (1.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1)
    ],
    "DANGER": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1),
        (0.20, 2.0, 31.0, 0.0, 0.8, 0.0, 5), # Alveolar /d/
        (0.45, 11.0, 37.0, 0.0, 0.7, 3.0, 4), # Diphthong /ay/
        (0.70, 6.0, 25.0, 0.7, 0.4, 1.0, 3), # Protrusion /j/
        (0.90, 3.0, 28.0, 0.0, 0.2, 0.0, 1), # Central /er/
        (1.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1)
    ],
    "WELCOME": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1),
        (0.18, 4.0, 18.0, 1.3, 0.0, 0.0, 3), # Rounded /w/
        (0.40, 8.0, 34.0, 0.0, 0.5, 2.0, 4), # Stretch /eh/
        (0.65, 5.0, 30.0, 0.0, 0.3, 1.0, 2), # Neutral /uh/
        (0.90, 0.0, 27.0, 0.0, 0.0, 0.0, 0), # Bilabial /m/
        (1.00, 0.0, 27.0, 0.0, 0.0, 0.0, 0)
    ],
    "GOODBYE": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1),
        (0.20, 4.0, 23.0, 0.8, 0.0, 1.0, 3), # Rounded /oo/
        (0.45, 0.0, 27.0, 0.0, 0.0, 0.0, 0), # Bilabial /b/
        (0.75, 13.0, 35.0, 0.0, 0.6, 4.0, 2), # Wide open /eye/
        (0.95, 3.0, 36.0, 0.0, 0.8, 1.0, 4), # Stretch offglide
        (1.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1)
    ],
    "PLEASE": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1),
        (0.20, 0.0, 27.0, 0.0, 0.0, 0.0, 0), # Bilabial seal /p/
        (0.55, 5.0, 41.0, 0.0, 0.9, 1.0, 4), # Extreme wide grin /ee/
        (0.80, 2.0, 36.0, 0.0, 1.0, 0.0, 5), # Dental friction /z/
        (1.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1)
    ],
    "OPEN": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1),
        (0.25, 9.0, 21.0, 1.1, 0.0, 2.0, 3), # Rounded /oh/
        (0.65, 0.0, 27.0, 0.0, 0.0, 0.0, 0), # Bilabial seal /p/
        (0.85, 3.0, 29.0, 0.0, 0.2, 0.0, 1), # Neutral release
        (1.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1)
    ],
    "CLOSE": [
        (0.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1),
        (0.25, 4.0, 31.0, 0.0, 0.4, 0.0, 5), # Velar-dental /kl/
        (0.55, 8.0, 21.0, 1.1, 0.0, 2.0, 3), # Rounded /oh/
        (0.80, 2.0, 36.0, 0.0, 1.0, 0.0, 5), # Dental friction /z/
        (1.00, 1.0, 28.0, 0.0, 0.0, 0.0, 1)
    ],
    "SILENCE": [
        (0.00, 0.8, 28.0, 0.0, 0.0, 0.0, 1),
        (0.50, 1.2, 28.0, 0.0, 0.0, 0.0, 1),
        (1.00, 0.8, 28.0, 0.0, 0.0, 0.0, 1)
    ]
}


def render_lip_frame(h, w, r=0.0, teeth=0.0, jaw=0.0, base_skin=185,
                     dx=0, dy=0, scale=1.0, img_size=IMG_SIZE):
    """
    Renders a single 64x64 grayscale mouth ROI image with anatomical lip contours,
    oral cavity shading, teeth exposure, and facial skin texturing.
    """
    frame = np.full((img_size, img_size), base_skin, dtype=np.uint8)
    cx = int(img_size // 2 + dx)
    cy = int(img_size // 2 + jaw * 0.45 + dy)

    # Scaled parameters
    eff_w = w * scale
    eff_h = h * scale

    # Outer Vermilion Lip Contour
    lip_rx = max(6, int(eff_w // 2))
    lip_ry = max(4, int(eff_h // 2 + 5.5 * scale))
    # Lip tissue color (darker than skin)
    cv2.ellipse(frame, (cx, cy), (lip_rx, lip_ry), 0, 0, 360, 125, -1)

    # Cupid's bow subtle philtrum shading
    phil_w = max(2, int(4 * scale))
    phil_top = max(0, cy - lip_ry - 4)
    cv2.line(frame, (cx - phil_w, phil_top), (cx - phil_w, cy - lip_ry), 160, 1)
    cv2.line(frame, (cx + phil_w, phil_top), (cx + phil_w, cy - lip_ry), 160, 1)

    # Oral Cavity & Teeth
    if eff_h > 1.5:
        in_rx = max(3, int(lip_rx * 0.75 - r * 2.2 * scale))
        in_ry = max(1, int(eff_h // 2))
        # Dark oral cavity interior
        cv2.ellipse(frame, (cx, cy), (in_rx, in_ry), 0, 0, 360, 20, -1)

        # Upper Teeth
        if teeth > 0.15:
            t_w = max(2, int(in_rx * 0.82))
            t_h = max(1, int(in_ry * 0.55 * teeth))
            cv2.rectangle(frame, (cx - t_w, cy - in_ry), (cx + t_w, cy - in_ry + t_h), 235, -1)
            # Tooth center slit
            cv2.line(frame, (cx, cy - in_ry), (cx, cy - in_ry + t_h), 140, 1)

    else:
        # Tightly closed lips / oral fissure line
        cv2.line(frame, (cx - lip_rx + 2, cy), (cx + lip_rx - 2, cy), 85, 1)

    # Add subtle skin texture noise
    noise = np.random.normal(0, 2.5, (img_size, img_size)).astype(np.float32)
    final_frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return final_frame


def synthesize_word_video(word, num_frames=NUM_FRAMES, augment=True):
    """
    Synthesizes a 24-frame silent lip video sequence for a given spoken word
    with realistic random speaker variation and kinematic easing.
    """
    keyframes = WORD_KEYFRAMES[word]
    traj = interpolate_spline(keyframes, num_frames=num_frames)

    # Speaker-level anatomical variation
    if augment:
        scale = random.uniform(0.90, 1.10)
        base_skin = random.randint(165, 205)
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
    else:
        scale = 1.0
        base_skin = 185
        dx, dy = 0, 0

    video_frames = []
    viseme_labels = []

    for h, w, r, teeth, jaw, vis_id in traj:
        # Subtle frame-level micromotion jitter
        if augment:
            f_h = max(0.0, h + random.uniform(-0.3, 0.3))
            f_w = max(16.0, w + random.uniform(-0.5, 0.5))
        else:
            f_h, f_w = h, w

        frame = render_lip_frame(
            h=f_h, w=f_w, r=r, teeth=teeth, jaw=jaw,
            base_skin=base_skin, dx=dx, dy=dy, scale=scale
        )
        video_frames.append(frame)
        viseme_labels.append(int(vis_id))

    # Convert to array [T, H, W], normalized [0, 1]
    video_array = np.stack(video_frames, axis=0).astype(np.float32) / 255.0
    viseme_array = np.array(viseme_labels, dtype=np.int64)
    return video_array, viseme_array


# ============================================================================
# 2. Neural Network Architecture: Factorized Space-Time ViT
# ============================================================================

class FactorizedSpaceTimeBlock(nn.Module):
    """
    Factorized Space-Time Self-Attention Block for video sequences.
    Performs Spatial Self-Attention across 2D patches within each frame,
    followed by Temporal Self-Attention across frames at identical patch locations.
    """
    def __init__(self, d_model=192, nhead=4, dim_feedforward=384, dropout=0.1):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, B, T, P):
        # x: [B * T, P, D]
        # 1. Spatial Self-Attention within each frame
        x_norm = self.norm1(x)
        sa_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
        x = x + sa_out

        # 2. Temporal Self-Attention across frames
        # Reshape to [B * P, T, D]
        x_t = x.view(B, T, P, -1).permute(0, 2, 1, 3).reshape(B * P, T, -1)
        x_t_norm = self.norm2(x_t)
        ta_out, ta_w = self.temporal_attn(x_t_norm, x_t_norm, x_t_norm)
        x_t = x_t + ta_out

        # Reshape back to [B * T, P, D]
        x = x_t.view(B, P, T, -1).permute(0, 2, 1, 3).reshape(B * T, P, -1)

        # 3. Position-wise Feedforward Network
        x = x + self.ffn(self.norm3(x))
        return x, ta_w


class LipReadingViT(nn.Module):
    """
    End-to-End Spatio-Temporal Vision Transformer for Silent Lip Reading.
    Combines 3D Convolutional front-end patch extraction with factorized
    Space-Time Transformer layers for spoken word and viseme classification.
    """
    def __init__(self, num_classes=NUM_WORDS, num_visemes=NUM_VISEMES,
                 d_model=192, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_visemes = num_visemes

        # 3D Convolutional Front-End: [B, 1, T, 64, 64] -> [B, 64, T, 4, 4]
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), # 16x16
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((None, 4, 4)) # 4x4 spatial patch grid = 16 patches
        )

        # Linear projection to Transformer dimension
        self.patch_proj = nn.Linear(64, d_model)

        # Learnable Spatial and Temporal Positional Embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, NUM_FRAMES, d_model) * 0.02)

        # Factorized Space-Time Transformer Blocks
        self.blocks = nn.ModuleList([
            FactorizedSpaceTimeBlock(d_model=d_model, nhead=nhead, dim_feedforward=384, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Multi-Task Prediction Heads
        self.temporal_pool = nn.Linear(d_model, 1)
        self.word_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.viseme_head = nn.Linear(d_model, num_visemes)

    def forward(self, x):
        # x: [B, 1, T, H, W]
        B, C, T, H, W = x.shape
        feat = self.frontend(x) # [B, 64, T, 4, 4]
        feat = feat.permute(0, 2, 3, 4, 1).contiguous() # [B, T, 4, 4, 64]
        feat = feat.view(B, T, 16, 64)                  # [B, T, 16, 64]
        tokens = self.patch_proj(feat)                   # [B, T, 16, d_model]

        # Add spatial position embeddings
        tokens = (tokens + self.spatial_pos).view(B * T, 16, self.d_model)

        last_temporal_attn = None
        for block in self.blocks:
            tokens, ta_w = block(tokens, B, T, 16)
            last_temporal_attn = ta_w

        tokens = self.norm(tokens)                       # [B * T, 16, d_model]
        tokens = tokens.view(B, T, 16, self.d_model)

        # Frame-level token: Mean pool over 16 spatial patches
        frame_tokens = tokens.mean(dim=2)                # [B, T, d_model]
        # Viseme trajectory predictions
        viseme_logits = self.viseme_head(frame_tokens)   # [B, T, num_visemes]

        # Video-level representation: Temporal Attention Pooling
        attn_weights = F.softmax(self.temporal_pool(frame_tokens), dim=1) # [B, T, 1]
        video_token = torch.sum(frame_tokens * attn_weights, dim=1)       # [B, d_model]

        # Word classification logits
        word_logits = self.word_classifier(video_token)  # [B, num_classes]
        return word_logits, viseme_logits, last_temporal_attn


# ============================================================================
# 3. Dataset & DataLoader
# ============================================================================

class LipReadingDataset(Dataset):
    """Dataset serving synthetic silent lip video tensors and annotations."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # item['video']: [T, H, W] -> tensor [1, T, H, W]
        vid = torch.from_numpy(item["video"]).unsqueeze(0).float()
        word_label = torch.tensor(item["word_idx"], dtype=torch.long)
        vis_label = torch.from_numpy(item["visemes"]).long()
        return vid, word_label, vis_label


def generate_dataset(samples_per_word=65, is_train=True):
    """Generates balanced dataset across all 10 vocabulary words."""
    desc = "train" if is_train else "validation"
    print(f"Generating {desc} silent lip video dataset ({samples_per_word * NUM_WORDS} video clips)...")
    data = []
    for word_idx, word in enumerate(VOCABULARY):
        for _ in range(samples_per_word):
            vid, vis = synthesize_word_video(word, num_frames=NUM_FRAMES, augment=is_train)
            data.append({
                "video": vid,
                "visemes": vis,
                "word": word,
                "word_idx": word_idx
            })
    random.shuffle(data)
    return data


# ============================================================================
# 4. Evaluation Suite: Word Accuracy, Top-3, Viseme Accuracy, Confusion Matrix
# ============================================================================

def evaluate_lip_reader(model, loader, device=DEVICE):
    """Evaluates Top-1 and Top-3 word accuracy, macro F1, and viseme accuracy."""
    model.eval()
    all_word_preds = []
    all_word_targets = []
    all_word_probs = []
    all_vis_preds = []
    all_vis_targets = []

    with torch.no_grad():
        for videos, word_labels, vis_labels in loader:
            videos = videos.to(device)
            word_labels = word_labels.to(device)
            vis_labels = vis_labels.to(device)

            word_logits, vis_logits, _ = model(videos)
            probs = F.softmax(word_logits, dim=-1)

            top1 = torch.argmax(word_logits, dim=-1)
            all_word_preds.extend(top1.cpu().numpy())
            all_word_targets.extend(word_labels.cpu().numpy())
            all_word_probs.append(probs.cpu().numpy())

            vis_pred = torch.argmax(vis_logits, dim=-1)
            all_vis_preds.append(vis_pred.cpu().numpy())
            all_vis_targets.append(vis_labels.cpu().numpy())

    all_word_preds = np.array(all_word_preds)
    all_word_targets = np.array(all_word_targets)
    all_word_probs = np.concatenate(all_word_probs, axis=0)

    all_vis_preds = np.concatenate(all_vis_preds, axis=0).flatten()
    all_vis_targets = np.concatenate(all_vis_targets, axis=0).flatten()

    # Top-1 and Top-3 accuracy
    top1_acc = float(np.mean(all_word_preds == all_word_targets))
    top3_correct = 0
    for i in range(len(all_word_targets)):
        top3_indices = np.argsort(-all_word_probs[i])[:3]
        if all_word_targets[i] in top3_indices:
            top3_correct += 1
    top3_acc = float(top3_correct / len(all_word_targets))

    # Viseme accuracy
    viseme_acc = float(np.mean(all_vis_preds == all_vis_targets))

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_word_targets, all_word_preds, average="macro", zero_division=0
    )

    conf_mat = confusion_matrix(all_word_targets, all_word_preds, labels=range(NUM_WORDS))

    return {
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "viseme_accuracy": viseme_acc,
        "confusion_matrix": conf_mat.tolist()
    }


# ============================================================================
# 5. Training Pipeline
# ============================================================================

def train_lip_reading_system():
    """Trains the Spatio-Temporal Vision Transformer for lip reading."""
    print("=" * 78)
    print(" PROJECT 95: LIP READING FROM SILENT VIDEO USING VISION TRANSFORMERS")
    print("=" * 78)

    output_dir = "/home/sericsoft/Desktop/pytorch_100_projects/95_lip_reading_vision_transformer/outputs"
    model_dir = "/home/sericsoft/Desktop/pytorch_100_projects/95_lip_reading_vision_transformer/models"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Dataset Generation
    train_data = generate_dataset(samples_per_word=55, is_train=True) # 550 clips
    val_data = generate_dataset(samples_per_word=18, is_train=False)  # 180 clips

    train_loader = DataLoader(LipReadingDataset(train_data), batch_size=16, shuffle=True)
    val_loader = DataLoader(LipReadingDataset(val_data), batch_size=16, shuffle=False)

    # 2. Model Initialization
    model = LipReadingViT(
        num_classes=NUM_WORDS,
        num_visemes=NUM_VISEMES,
        d_model=192,
        nhead=4,
        num_layers=3
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized LipReadingViT on {DEVICE} ({total_params:,} trainable parameters).")

    # 3. Multi-Task Loss & Optimizer
    word_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    viseme_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_viseme_acc": []
    }

    best_acc = -1.0
    best_model_path = os.path.join(model_dir, "lip_reading_vit_best.pth")

    print("\nStarting Spatio-Temporal ViT Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        correct_words = 0
        total_words = 0
        t0 = time.time()

        for videos, word_targets, vis_targets in train_loader:
            videos = videos.to(DEVICE)
            word_targets = word_targets.to(DEVICE)
            vis_targets = vis_targets.to(DEVICE)

            optimizer.zero_grad()
            word_logits, vis_logits, _ = model(videos)

            # Losses
            loss_word = word_criterion(word_logits, word_targets)
            B, T, V = vis_logits.shape
            loss_vis = viseme_criterion(vis_logits.view(B * T, V), vis_targets.view(B * T))
            loss = loss_word + 0.45 * loss_vis

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            total_epoch_loss += loss.item()
            preds = torch.argmax(word_logits, dim=-1)
            correct_words += (preds == word_targets).sum().item()
            total_words += word_targets.size(0)

        scheduler.step()
        epoch_time = time.time() - t0
        train_loss = total_epoch_loss / len(train_loader)
        train_acc = correct_words / total_words

        # Validation
        val_eval = evaluate_lip_reader(model, val_loader, device=DEVICE)
        val_acc = val_eval["top1_accuracy"]
        val_vis_acc = val_eval["viseme_accuracy"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_viseme_acc"].append(val_vis_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Word Acc: {val_acc*100:.2f}% | Viseme Acc: {val_vis_acc*100:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocabulary": VOCABULARY,
                "visemes": VISEMES,
                "epoch": epoch,
                "val_acc": val_acc,
                "d_model": 192
            }, best_model_path)

    print(f"\nModel training completed! Best checkpoint saved to: {best_model_path}")

    # Load best checkpoint for final benchmarking
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate_lip_reader(model, val_loader, device=DEVICE)

    # 4. Latency & FPS Benchmark
    print("\nRunning Inference Latency Benchmark on CPU...")
    test_video = torch.from_numpy(val_data[0]["video"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
    latencies = []
    for _ in range(25):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(test_video)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    avg_latency_ms = float(np.mean(latencies[5:]))
    video_dur_sec = NUM_FRAMES / 25.0
    throughput_fps = NUM_FRAMES / (avg_latency_ms / 1000.0)
    rtf = (avg_latency_ms / 1000.0) / video_dur_sec
    speedup = video_dur_sec / (avg_latency_ms / 1000.0)

    final_metrics["latency_ms"] = round(avg_latency_ms, 2)
    final_metrics["throughput_fps"] = round(throughput_fps, 1)
    final_metrics["rtf"] = round(rtf, 4)
    final_metrics["speedup_realtime"] = round(speedup, 1)

    print(f"Benchmark Results:")
    print(f"  Top-1 Word Accuracy: {final_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3 Word Accuracy: {final_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Macro F1 Score:      {final_metrics['macro_f1']*100:.2f}%")
    print(f"  Viseme Accuracy:     {final_metrics['viseme_accuracy']*100:.2f}%")
    print(f"  Latency per Video:   {final_metrics['latency_ms']:.2f} ms")
    print(f"  Throughput:          {final_metrics['throughput_fps']:.1f} FPS ({speedup:.1f}x real-time speedup)")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "lip_reading_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # 5. Export Canonical Test Silent Videos
    print("\nExporting Canonical Silent MP4 Videos...")
    test_words = ["HELLO", "STOP", "THANK_YOU", "DANGER", "PLEASE"]
    saved_clips = []
    for word in test_words:
        vid, vis = synthesize_word_video(word, num_frames=NUM_FRAMES, augment=False)
        filename = f"test_silent_{word.lower()}.mp4"
        video_path = os.path.join(output_dir, filename)

        # Write MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(video_path, fourcc, 25.0, (IMG_SIZE, IMG_SIZE), isColor=False)
        for frame_idx in range(NUM_FRAMES):
            frame_uint8 = (vid[frame_idx] * 255.0).astype(np.uint8)
            out_writer.write(frame_uint8)
        out_writer.release()

        saved_clips.append({
            "word": word,
            "path": video_path,
            "filename": filename,
            "video": vid,
            "visemes": vis
        })
        print(f"  Saved: {video_path} (Word: '{word}')")

    # 6. Generate Diagnostic Dashboard
    print("\nGenerating Diagnostic Visualization Dashboard...")
    report_path = os.path.join(output_dir, "lip_reading_report.png")
    generate_diagnostic_report(
        model=model,
        history=history,
        metrics=final_metrics,
        saved_clips=saved_clips,
        report_path=report_path
    )
    print(f"Dashboard saved to: {report_path}")

    print("\n" + "=" * 78)
    print(" PROJECT 95 TRAINING AND BENCHMARKING COMPLETE!")
    print("=" * 78)


def generate_diagnostic_report(model, history, metrics, saved_clips, report_path):
    """Generates 6-panel publication-ready visualization report."""
    fig = plt.figure(figsize=(20, 14), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Panel 1: Silent Video Mouth Frame Strip (Word: HELLO)
    ax1 = fig.add_subplot(2, 3, 1)
    hello_clip = saved_clips[0] # HELLO
    vid = hello_clip["video"]
    # Pick 8 evenly spaced frames
    frame_indices = np.linspace(0, NUM_FRAMES - 1, 8, dtype=int)
    strip = np.hstack([vid[i] for i in frame_indices])
    ax1.imshow(strip, cmap="gray")
    ax1.set_title(f"1. Mouth Frame Sequence: '{hello_clip['word']}' (8 of 24 Frames)", fontsize=11, fontweight="bold")
    ax1.axis("off")
    # Annotate frame numbers
    for idx_pos, f_i in enumerate(frame_indices):
        ax1.text(idx_pos * IMG_SIZE + IMG_SIZE // 2, IMG_SIZE + 8, f"F{f_i+1}",
                 color="black", fontsize=8.5, ha="center")

    # 2. Panel 2: Confusion Matrix Heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm.astype('float') / np.maximum(1, cm.sum(axis=1)[:, np.newaxis])
    im2 = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax2.set_xticks(range(NUM_WORDS))
    ax2.set_yticks(range(NUM_WORDS))
    ax2.set_xticklabels([w[:4] for w in VOCABULARY], fontsize=8, rotation=45)
    ax2.set_yticklabels(VOCABULARY, fontsize=8)
    ax2.set_xlabel("Predicted Spoken Word", fontsize=9.5)
    ax2.set_ylabel("True Spoken Word", fontsize=9.5)
    ax2.set_title(f"2. Word Confusion Matrix (Top-1 Acc: {metrics['top1_accuracy']*100:.1f}%)",
                  fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Panel 3: Viseme Trajectory Alignment
    ax3 = fig.add_subplot(2, 3, 3)
    test_tensor = torch.from_numpy(hello_clip["video"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, vis_logits, _ = model(test_tensor)
        vis_preds = torch.argmax(vis_logits[0], dim=-1).cpu().numpy()
    gt_vis = hello_clip["visemes"]
    frames = np.arange(1, NUM_FRAMES + 1)
    ax3.plot(frames, gt_vis, "s-", color="#1f77b4", linewidth=2.0, label="Ground Truth Viseme")
    ax3.plot(frames, vis_preds, "o--", color="#d62728", linewidth=1.8, label="Predicted Viseme")
    ax3.set_yticks(range(NUM_VISEMES))
    ax3.set_yticklabels(VISEMES, fontsize=8)
    ax3.set_xlabel("Video Frame Index (t)", fontsize=9.5)
    ax3.set_ylabel("Viseme Class", fontsize=9.5)
    ax3.set_title(f"3. Frame-Level Viseme Trajectory: '{hello_clip['word']}'", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=8.5)

    # 4. Panel 4: Spatiotemporal Self-Attention Heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    with torch.no_grad():
        _, _, temp_attn = model(test_tensor)
    if temp_attn is not None:
        # temp_attn: [B * P, T, T] -> average across patches
        attn_matrix = temp_attn.view(-1, 16, NUM_FRAMES, NUM_FRAMES).mean(dim=1)[0].cpu().numpy()
        im4 = ax4.imshow(attn_matrix, cmap="viridis", origin="lower")
        ax4.set_xlabel("Key Video Frame (Source)", fontsize=9.5)
        ax4.set_ylabel("Query Video Frame (Target)", fontsize=9.5)
        ax4.set_title("4. Temporal Attention Matrix Across Frames", fontsize=11, fontweight="bold")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 5. Panel 5: Training Loss & Accuracy Convergence
    ax5 = fig.add_subplot(2, 3, 5)
    ep_range = range(1, len(history["train_loss"]) + 1)
    ax5.plot(ep_range, [a * 100 for a in history["train_acc"]], "o-", color="#2ca02c", label="Train Word Acc (%)")
    ax5.plot(ep_range, [a * 100 for a in history["val_acc"]], "^--", color="#1f77b4", label="Val Word Acc (%)")
    ax5.plot(ep_range, [a * 100 for a in history["val_viseme_acc"]], "s:", color="#ff7f0e", label="Viseme Acc (%)")
    ax5.set_xlabel("Epoch", fontsize=9.5)
    ax5.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax5.set_ylim(0, 105)
    ax5.set_title("5. ViT Multi-Task Training Convergence", fontsize=11, fontweight="bold")
    ax5.grid(True, linestyle=":", alpha=0.5)
    ax5.legend(loc="lower right", fontsize=8.5)

    # 6. Panel 6: Lip Reading Predictions Showcase
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    title_text = "6. Silent Video Recognition Test Predictions\n"
    lines = [title_text]

    for item in saved_clips:
        t_vid = torch.from_numpy(item["video"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            w_logits, _, _ = model(t_vid)
            probs = F.softmax(w_logits, dim=-1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            pred_word = VOCABULARY[pred_idx]
            conf = probs[pred_idx] * 100

        status = "✓ CORRECT" if pred_word == item["word"] else "✗ ERROR"
        lines.append(f"• Clip: {item['filename']}")
        lines.append(f"  Target: '{item['word']}'  →  Pred: '{pred_word}' ({conf:.1f}%)  [{status}]")
        lines.append("")

    lines.append(f"• CPU Pipeline Latency: {metrics['latency_ms']} ms ({metrics['throughput_fps']} FPS)")
    lines.append(f"• Real-Time Factor (RTF): {metrics['rtf']} ({metrics['speedup_realtime']}x real-time speedup)")

    ax6.text(0.02, 0.95, "\n".join(lines), transform=ax6.transAxes,
             fontsize=9.5, verticalalignment="top", fontfamily="sans-serif",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95))

    fig.suptitle("Project 95: Lip Reading from Silent Video using Vision Transformers",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(report_path, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main Execution Entrypoint
# ============================================================================

def main():
    train_lip_reading_system()


if __name__ == "__main__":
    main()

