#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""筛选 nocrack 数据集中未检测到像质计 ROI 的图像。

使用 run_iqi_grade_infer.py 中像质计区域识别的算法接口（YOLO OBB gauge 模型），
遍历 datasets/images/nocrack 中的所有图像，将未检测到像质计 ROI 区域的图像
复制到 outputs/候选双丝像质计 目录中。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gauge.pipeline_utils import collect_images, ensure_dir, resize_long_side
from gauge.roi_stage import extract_best_obb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="筛选 nocrack 数据集中未检测到像质计 ROI 的图像，"
        "保存到 outputs/候选双丝像质计。使用 run_iqi_grade_infer.py "
        "中相同的 YOLO OBB gauge 算法接口。"
    )
    parser.add_argument(
        "--image-dir",
        default="datasets/images/nocrack",
        help="输入图像目录（默认: datasets/images/nocrack）",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/候选双丝像质计",
        help="未检测到 ROI 的图像输出目录（默认: outputs/候选双丝像质计）",
    )
    parser.add_argument(
        "--gauge-weights",
        default="local/gauge.pt",
        help="OBB gauge 检测器权重文件路径（默认: local/gauge.pt）",
    )
    parser.add_argument(
        "--gauge-conf",
        type=float,
        default=0.25,
        help="OBB 置信度阈值（默认: 0.25）",
    )
    parser.add_argument(
        "--gauge-iou",
        type=float,
        default=0.45,
        help="OBB IoU 阈值（默认: 0.45）",
    )
    parser.add_argument(
        "--gauge-imgsz",
        type=int,
        default=640,
        help="OBB 推理图像尺寸（默认: 640）",
    )
    parser.add_argument(
        "--gauge-device",
        default=None,
        help="OBB 推理设备，如 0 / cuda:0 / cpu（默认: None，自动选择）",
    )
    parser.add_argument(
        "--gauge-select",
        choices=["conf", "area"],
        default="conf",
        help="多候选 ROI 选择策略: conf（按置信度）/ area（按面积）（默认: conf）",
    )
    parser.add_argument(
        "--gauge-class",
        type=int,
        default=None,
        help="可选的 YOLO 类别 ID 过滤",
    )
    parser.add_argument(
        "--resize-long-side",
        type=int,
        default=960,
        help="送入 gauge 模型前图像长边缩放目标值（默认: 960）",
    )
    return parser.parse_args()


def _resolve_path(raw: Optional[str], cwd: Path) -> Optional[Path]:
    """Resolve a path against cwd, returning None if input is None."""
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return path


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    image_dir = _resolve_path(args.image_dir, cwd)
    if image_dir is None or not image_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {args.image_dir}")

    output_dir = _resolve_path(args.output_dir, cwd)
    ensure_dir(output_dir)

    gauge_weights = _resolve_path(args.gauge_weights, cwd)
    if gauge_weights is None or not gauge_weights.is_file():
        raise SystemExit(f"Gauge 权重文件不存在: {args.gauge_weights}")
    print(f"Gauge 权重: {gauge_weights}")

    # 收集输入图像
    image_paths = collect_images(image_dir=image_dir, image_list=None)
    if not image_paths:
        raise SystemExit(f"在 {image_dir} 中未找到支持的图像文件")
    print(f"待处理图像: {len(image_paths)} 张")

    # 延迟导入 YOLO 避免过早初始化 GPU
    from ultralytics import YOLO

    gauge_model = YOLO(str(gauge_weights))

    no_roi_paths: list[Path] = []
    has_roi_count = 0
    error_count = 0

    try:
        for i, image_path in enumerate(image_paths, 1):
            status = f"[{i}/{len(image_paths)}] {image_path.name}"
            print(f"{status} ...", end=" ", flush=True)

            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    print("✗ 无法读取图像")
                    error_count += 1
                    continue

                # 与 run_iqi_grade_infer.py 相同的预处理：缩放长边
                sampled, _scale = resize_long_side(image, args.resize_long_side)

                # 与 run_iqi_grade_infer.py 相同的 ROI 检测算法
                results = gauge_model.predict(
                    source=sampled,
                    conf=args.gauge_conf,
                    iou=args.gauge_iou,
                    imgsz=args.gauge_imgsz,
                    device=args.gauge_device,
                    verbose=False,
                )

                if not results:
                    roi_detected = False
                else:
                    roi_info = extract_best_obb(
                        results[0],
                        select=args.gauge_select,
                        class_filter=args.gauge_class,
                    )
                    roi_detected = roi_info is not None

                if roi_detected:
                    print("✓ 已检测到 ROI，跳过")
                    has_roi_count += 1
                else:
                    print("→ 未检测到 ROI，复制到输出目录")
                    no_roi_paths.append(image_path)
                    dest = output_dir / image_path.name
                    shutil.copy2(image_path, dest)

            except Exception as exc:
                print(f"✗ 异常: {exc}")
                error_count += 1
                continue

    finally:
        del gauge_model

    # 输出汇总
    print()
    print("=" * 50)
    print("汇总")
    print("=" * 50)
    print(f"  输入目录:     {image_dir}")
    print(f"  输出目录:     {output_dir}")
    print(f"  图像总数:     {len(image_paths)}")
    print(f"  检测到 ROI:   {has_roi_count}")
    print(f"  未检测到 ROI: {len(no_roi_paths)} (已保存至 {args.output_dir})")
    if error_count:
        print(f"  处理异常:     {error_count}")
    print("=" * 50)

    # 列出所有未检测到 ROI 的图像路径
    if no_roi_paths:
        print()
        print("未检测到 ROI 的图像:")
        for p in no_roi_paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
