#!/usr/bin/env python3
"""
Compute grayscale mean/std over train+valid images and update config/model.yaml.

Usage:
  python scripts/compute_mean_std.py \
    --data_dir /datasets/PAR/CODE/F-Clip/IQIdata/processed \
    --config config/model.yaml
"""

import argparse
import math
import re
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute mean/std for grayscale images.")
    parser.add_argument("--data_dir", required=True, help="Processed dataset root containing train/valid.")
    parser.add_argument("--config", default="config/model.yaml", help="Model config to update.")
    return parser.parse_args()


def collect_images(data_dir: Path):
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    images = []
    if train_dir.exists():
        images.extend(sorted(train_dir.glob("*.png")))
    if valid_dir.exists():
        images.extend(sorted(valid_dir.glob("*.png")))
    return images


def compute_mean_std(images):
    total_sum = 0.0
    total_sumsq = 0.0
    total_n = 0
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skip unreadable image: {p}")
            continue
        arr = img.astype(np.float64)
        total_sum += arr.sum()
        total_sumsq += (arr * arr).sum()
        total_n += arr.size

    if total_n == 0:
        raise ValueError("No valid images found to compute mean/std.")

    mean = total_sum / total_n
    var = total_sumsq / total_n - mean * mean
    std = math.sqrt(max(var, 0.0))
    return mean, std, total_n


def update_base_yaml(config_path: Path, mean: float, std: float):
    text = config_path.read_text(encoding="utf-8")
    mean_str = f"[{mean:.6f}]"
    std_str = f"[{std:.6f}]"

    def _replace(key, value, src):
        pattern = rf"(^\s*{key}\s*:\s*)\[(.*?)\]\s*$"
        repl = rf"\1{value}"
        new_text, n = re.subn(pattern, repl, src, flags=re.MULTILINE)
        return new_text, n

    text, n1 = _replace("mean", mean_str, text)
    text, n2 = _replace("stddev", std_str, text)
    if n1 == 0 or n2 == 0:
        raise ValueError("Failed to locate mean/stddev in config. Please update manually.")
    config_path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    config_path = Path(args.config)

    images = collect_images(data_dir)
    print(f"Found {len(images)} images.")
    mean, std, total_n = compute_mean_std(images)
    print(f"Total pixels: {total_n}")
    print(f"Mean: {mean:.6f}, Std: {std:.6f}")

    update_base_yaml(config_path, mean, std)
    print(f"Updated {config_path} with mean/stddev.")


if __name__ == "__main__":
    main()
