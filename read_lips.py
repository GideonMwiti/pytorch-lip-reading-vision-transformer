#!/usr/bin/env python3
"""
CLI Tool: read_lips.py
======================
Predicts spoken words and visual speech viseme trajectories from silent video
recordings of a speaker's mouth region using the trained LipReadingViT model.

Usage Examples:
  python read_lips.py --video outputs/test_silent_hello.mp4
  python read_lips.py --video outputs/test_silent_stop.mp4 --top-k 5
  python read_lips.py --video outputs/test_silent_danger.mp4 --show-visemes
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from main import (
    LipReadingViT,
    VOCABULARY,
    VISEMES,
    NUM_WORDS,
    NUM_VISEMES,
    NUM_FRAMES,
    IMG_SIZE
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_prob_bar(prob, length=24):
    filled = int(round(prob * length))
    bar = "█" * filled + "░" * (length - filled)
    return bar


def load_lip_reading_model(checkpoint_path, device=DEVICE):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint.get("vocabulary", VOCABULARY)
    visemes = checkpoint.get("visemes", VISEMES)

    model = LipReadingViT(
        num_classes=len(vocab),
        num_visemes=len(visemes),
        d_model=checkpoint.get("d_model", 192),
        nhead=4,
        num_layers=3
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, visemes


def process_video_file(video_path, target_frames=NUM_FRAMES, target_size=IMG_SIZE):
    """Loads video frames via OpenCV, converts to grayscale, and resizes/pads to [1, 1, T, H, W]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
        raw_frames.append(resized)
    cap.release()

    if not raw_frames:
        raise ValueError(f"No valid video frames found in: {video_path}")

    orig_count = len(raw_frames)

    # Resample or pad/truncate to target_frames
    if len(raw_frames) == target_frames:
        sampled_frames = raw_frames
    elif len(raw_frames) > target_frames:
        indices = np.linspace(0, len(raw_frames) - 1, target_frames, dtype=int)
        sampled_frames = [raw_frames[i] for i in indices]
    else:
        # Loop or pad with last frame
        sampled_frames = list(raw_frames)
        while len(sampled_frames) < target_frames:
            sampled_frames.append(raw_frames[-1])

    # Convert to normalized float tensor [1, 1, T, H, W]
    arr = np.stack(sampled_frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor, orig_count, fps


def run_lip_reading_inference(video_path, checkpoint_path, top_k=5, show_visemes=True):
    t_start_total = time.perf_counter()

    print("\n" + "=" * 78)
    print(" SILENT LIP READING RECOGNITION: VISION TRANSFORMER (VSR)")
    print("=" * 78)

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        sys.exit(1)

    # 1. Video Preprocessing
    t_load_0 = time.perf_counter()
    video_tensor, frame_count, fps = process_video_file(video_path)
    t_load_ms = (time.perf_counter() - t_load_0) * 1000.0

    dur_sec = frame_count / max(1.0, fps)
    print(f"Input Video File:   {os.path.basename(video_path)}")
    print(f"Path:               {os.path.abspath(video_path)}")
    print(f"Video Metadata:     {frame_count} frames | {fps:.1f} FPS | Duration: {dur_sec:.2f} s")
    print(f"Mouth ROI Resized:  {IMG_SIZE}x{IMG_SIZE} (Model Input: {NUM_FRAMES} frames)")
    print(f"Inference Device:   {DEVICE}")

    # 2. Model Loading
    model, vocab, visemes = load_lip_reading_model(checkpoint_path, device=DEVICE)

    # 3. Model Inference
    t_infer_0 = time.perf_counter()
    with torch.no_grad():
        video_tensor = video_tensor.to(DEVICE)
        word_logits, vis_logits, _ = model(video_tensor)
        probs = F.softmax(word_logits, dim=-1)[0].cpu().numpy()
        vis_preds = torch.argmax(vis_logits[0], dim=-1).cpu().numpy()
    t_infer_ms = (time.perf_counter() - t_infer_0) * 1000.0

    # 4. Spoken Word Ranking
    ranked_indices = np.argsort(-probs)
    best_idx = ranked_indices[0]
    best_word = vocab[best_idx]
    best_conf = probs[best_idx] * 100.0

    print("\n" + "-" * 78)
    print(" RECOGNIZED SPOKEN WORD CANDIDATES (TOP PREDICTIONS)")
    print("-" * 78)
    print(f" {'Rank':<4}  {'Spoken Word':<18}  {'Probability':<12}  {'Confidence Level'}")
    print(" " + "-" * 74)

    for rank in range(min(top_k, len(vocab))):
        idx = ranked_indices[rank]
        word = vocab[idx]
        p = probs[idx]
        bar = format_prob_bar(p, length=24)
        marker = "★ [PREDICTED]" if rank == 0 else ""
        print(f"  {rank+1:<3}  {word:<18}  {p*100:>6.2f}%       {bar}  {marker}")

    print("\n" + "=" * 78)
    print(f" >>> FINAL PREDICTED SPOKEN WORD:  \"{best_word}\"  ({best_conf:.2f}% Confidence)")
    print("=" * 78)

    # 5. Frame-by-Frame Viseme Breakdown
    if show_visemes:
        print("\n" + "-" * 78)
        print(" FRAME-BY-FRAME VISEME (VISUAL SPEECH) TRAJECTORY")
        print("-" * 78)
        print(f" {'Frame':<8}  {'Timestamp':<12}  {'Viseme Class':<18}  {'Phonetic Movement Description'}")
        print(" " + "-" * 74)

        viseme_descriptions = {
            "CLOSED": "Tight bilabial lip closure (/p/, /b/, /m/)",
            "NARROW_OPEN": "Neutral resting oral posture",
            "WIDE_OPEN": "Broad vertical jaw drop & open oral cavity (/a/, /ae/)",
            "ROUNDED_PUCKER": "Forward labial protrusion & circular rounding (/u/, /o/, /w/)",
            "STRETCHED_WIDE": "Extreme lateral mouth corner stretch (/ee/, /ay/)",
            "DENTAL_CONTACT": "Incisor exposure with dental friction (/s/, /z/, /th/)"
        }

        for f_idx in range(NUM_FRAMES):
            t_sec = f_idx / fps
            vis_id = vis_preds[f_idx]
            vis_name = visemes[vis_id]
            desc = viseme_descriptions.get(vis_name, "Articulation in progress")
            print(f"  F{f_idx+1:<7}  {t_sec:5.2f}s        {vis_name:<18}  {desc}")

    # 6. Performance & Speed Benchmarks
    t_total_ms = (time.perf_counter() - t_start_total) * 1000.0
    throughput_fps = NUM_FRAMES / (t_infer_ms / 1000.0)
    rtf = (t_infer_ms / 1000.0) / dur_sec
    speedup = dur_sec / max(1e-4, t_infer_ms / 1000.0)

    print("\n" + "-" * 78)
    print(" INFERENCE PERFORMANCE & REAL-TIME FACTOR")
    print("-" * 78)
    print(f"  • Video Preprocessing:    {t_load_ms:6.2f} ms")
    print(f"  • ViT Neural Inference:   {t_infer_ms:6.2f} ms")
    print(f"  • Total Pipeline Latency: {t_total_ms:6.2f} ms")
    print(f"  • Video Processing Rate:  {throughput_fps:6.1f} FPS")
    print(f"  • Real-Time Factor (RTF): {rtf:6.4f} ({speedup:.1f}x real-time speedup)")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict spoken words from silent video recordings of a speaker's mouth."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="/home/sericsoft/Desktop/pytorch_100_projects/95_lip_reading_vision_transformer/outputs/test_silent_hello.mp4",
        help="Path to input silent video file (.mp4, .avi)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/sericsoft/Desktop/pytorch_100_projects/95_lip_reading_vision_transformer/models/lip_reading_vit_best.pth",
        help="Path to trained LipReadingViT checkpoint."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predicted words to display."
    )
    parser.add_argument(
        "--no-visemes",
        action="store_true",
        help="Disable frame-level viseme trajectory breakdown table."
    )

    args = parser.parse_args()
    run_lip_reading_inference(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        show_visemes=not args.no_visemes
    )


if __name__ == "__main__":
    main()

