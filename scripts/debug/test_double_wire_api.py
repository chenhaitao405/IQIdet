#!/usr/bin/env python3
"""自测 double_wire_api — 批量提取真实 strip，调 API，存档输入/输出供对比。

从 annotate 保存的 profile JSON 中读取 image_path + profile_line，
提取真实 strip → base64 → 调 API → 保存 strip PNG + API 结果 JSON。

Usage:
    # 默认：扫描 outputs/double_wire_demo/ 下所有 _profile.json
    python scripts/debug/test_double_wire_api.py

    # 指定目录
    python scripts/debug/test_double_wire_api.py --profile-dir outputs/double_wire_demo

    # 单个 JSON
    python scripts/debug/test_double_wire_api.py --profile outputs/.../xxx_profile.json

    # 自定义输出目录
    python scripts/debug/test_double_wire_api.py --output-dir outputs/api_test_package
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gauge.imaging.profile import extract_profile_strip
from double_wire_api import (
    DoubleWireRequest,
    compute_double_wire,
    init_double_wire_api,
    close_double_wire_api,
)


def find_profile_jsons(profile_dir: str) -> List[Path]:
    """递归扫描目录下所有 _profile.json 文件。"""
    root = Path(profile_dir)
    paths = sorted(root.rglob("*_profile.json"))
    # 排除 inverted 变体（只测 ori）
    paths = [p for p in paths if "/inver/" not in str(p) and "\\inver\\" not in str(p)]
    return paths


def extract_real_strip(profile_json_path: Path) -> Tuple[np.ndarray, dict]:
    """从原始图像重新提取 strip，返回 (strip_2d, metadata)。

    metadata 包含: image_stem, band_width, expand, profile_line
    """
    with open(profile_json_path) as f:
        pdata = json.load(f)

    image_path = pdata.get("image_path")
    band_width = pdata.get("band_width", 21)
    expand = pdata.get("expand", 60)
    profile_line = pdata.get("profile_line") or pdata.get("profile_midline")

    if not image_path or not profile_line:
        raise ValueError(f"{profile_json_path.name}: 缺少 image_path 或 profile_line")

    # 解析 image_path（可能是相对路径，相对于 repo root）
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = REPO_ROOT / img_path

    if not img_path.exists():
        # 尝试只用文件名搜索
        candidates = list(Path("outputs").rglob(img_path.name))
        if candidates:
            img_path = candidates[0]
        else:
            raise FileNotFoundError(f"找不到原始图像: {img_path}")

    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"无法读取: {img_path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    start = (float(profile_line["start"][0]), float(profile_line["start"][1]))
    end = (float(profile_line["end"][0]), float(profile_line["end"][1]))
    line_length = pdata.get("line_length_px")
    if line_length is None:
        line_length = max(1, int(np.ceil(np.hypot(
            end[0] - start[0], end[1] - start[1]))))

    full_strip = extract_profile_strip(
        raw, start, end, expand=expand, num_samples=line_length,
    )

    # 窄条带：只取中线附近 band_width 行（与 annotate 一致）
    half_h = full_strip.shape[0] // 2
    half_bw = band_width // 2
    strip = full_strip[half_h - half_bw : half_h + half_bw + 1, :]

    meta = {
        "image_stem": profile_json_path.stem.replace("_profile", ""),
        "source_profile": str(profile_json_path),
        "source_image": str(img_path),
        "band_width": band_width,
        "strip_height": strip.shape[0],
        "expand": expand,
        "line_length_px": line_length,
    }
    return strip, meta


def strip_to_png_bytes(strip: np.ndarray) -> bytes:
    """将 strip 编码为 PNG 字节流（8-bit 归一化）。"""
    s_min, s_max = strip.min(), strip.max()
    if s_max > s_min:
        strip_8u = ((strip - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        strip_8u = np.zeros_like(strip, dtype=np.uint8)
    _, buf = cv2.imencode(".png", strip_8u)
    return buf.tobytes()


async def test_one(profile_path: Path, output_dir: Path) -> dict:
    """测试一个 profile JSON，保存 strip PNG + API 结果。"""
    stem = profile_path.stem.replace("_profile", "")
    print(f"\n{'='*70}")
    print(f"[test] {stem}")
    print(f"[test]   profile: {profile_path}")

    # 1. 提取真实 strip
    try:
        strip, meta = extract_real_strip(profile_path)
    except Exception as e:
        print(f"[test]   SKIP: {e}")
        return {"stem": stem, "ok": False, "error": str(e)}

    print(f"[test]   strip: {strip.shape}, dtype={strip.dtype}")

    # 2. 保存 strip PNG（输入存档）
    strips_dir = output_dir / "strips"
    strips_dir.mkdir(parents=True, exist_ok=True)
    png_bytes = strip_to_png_bytes(strip)
    strip_path = strips_dir / f"{stem}_strip.png"
    strip_path.write_bytes(png_bytes)
    print(f"[test]   saved strip: {strip_path}")

    # 3. 编码 base64
    b64 = base64.b64encode(png_bytes).decode("ascii")

    # 4. 调 API
    req = DoubleWireRequest(image_base64=b64)
    resp = await compute_double_wire(req)

    # 5. 打印摘要
    print(f"[test]   ok: {resp.ok}  |  {resp.result_name}  |  {resp.timings_ms}")
    if resp.result:
        r = resp.result
        print(f"[test]   film={r['film_type']}  pairs={r['num_pairs']}  "
              f"unresolved=D{r['first_unresolved_group']}" if r['first_unresolved_group']
              else f"[test]   film={r['film_type']}  pairs={r['num_pairs']}  "
              f"all resolved")
        for p in r["pairs"][:3]:
            print(f"[test]     D{p['group']}: dip={p['dip_percent']:.1f}%  "
                  f"w1={p['wire_a_idx']} g={p['gap_idx']} w2={p['wire_b_idx']}")
        if len(r["pairs"]) > 3:
            print(f"[test]     ... ({len(r['pairs'])} pairs total)")

    # 6. 保存结果 JSON
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "api_response": resp.model_dump(),
    }
    result_path = results_dir / f"{stem}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[test]   saved result: {result_path}")

    return {
        "stem": stem,
        "ok": resp.ok,
        "result_code": resp.result_code,
        "num_pairs": resp.result["num_pairs"] if resp.result else 0,
        "film_type": resp.result["film_type"] if resp.result else None,
        "first_unresolved": resp.result["first_unresolved_group"] if resp.result else None,
    }


def write_report(results: list, output_dir: Path) -> None:
    """写测试报告。"""
    lines = []
    lines.append("双丝像质计分析 API 自测报告")
    lines.append("=" * 60)
    lines.append(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"总数: {len(results)}")
    passed = sum(1 for r in results if r.get("ok"))
    lines.append(f"成功: {passed}  |  失败: {len(results) - passed}")
    lines.append("")
    lines.append(f"{'图像':<45} {'OK':>5} {'pairs':>6} {'film':>9} {'unresolved':>12}")
    lines.append("-" * 80)
    for r in results:
        unresolved = f"D{r['first_unresolved']}" if r.get("first_unresolved") else "none"
        lines.append(
            f"{r['stem']:<45} "
            f"{'PASS' if r.get('ok') else 'FAIL':>5} "
            f"{r.get('num_pairs', 0):>6} "
            f"{r.get('film_type', 'N/A'):>9} "
            f"{unresolved:>12}"
        )
    if any(not r.get("ok") for r in results):
        lines.append("")
        lines.append("失败详情:")
        for r in results:
            if not r.get("ok"):
                lines.append(f"  {r['stem']}: {r.get('error', 'result_code=' + str(r.get('result_code')))}")

    report_path = output_dir / "test_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{'='*70}")
    print("\n".join(lines))


def write_readme(output_dir: Path) -> None:
    """写交付说明。"""
    readme = output_dir / "README.md"
    readme.write_text("""# 双丝像质计分析 API 自测数据包

## 目录结构

```
├── README.md              ← 本文件
├── strips/                ← API 输入：从原始图像提取的 strip PNG
│   └── *_strip.png
├── results/               ← API 输出：完整 JSON 结果
│   └── *_result.json
└── test_report.txt        ← 批量测试摘要
```

## 复现方式

```bash
conda activate weld-gpu
cd /path/to/IQIdet

# 运行自测（使用本数据包的 strip 图像）
python scripts/debug/test_double_wire_api.py \\
    --profile-dir outputs/double_wire_demo \\
    --output-dir outputs/api_test_package
```

## API 调用示例

```python
import asyncio
import base64
import json
from double_wire_api import (
    init_double_wire_api,
    compute_double_wire,
    DoubleWireRequest,
    close_double_wire_api,
)

async def example():
    init_double_wire_api()

    # 读取 strip PNG → base64
    with open("strips/xxx_strip.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    req = DoubleWireRequest(image_base64=b64)
    resp = await compute_double_wire(req)

    # 保存结果
    with open("result.json", "w") as f:
        json.dump(resp.model_dump(), f, indent=2, ensure_ascii=False)

    close_double_wire_api()

asyncio.run(example())
```

## 输出字段说明

详见 `docs/contract/DOUBLE_WIRE_ANALYSIS_API.md`

核心字段：
- `result.profile` — 灰度曲线（可画图）
- `result.peaks/valleys` — 峰/谷坐标
- `result.pairs[]` — 每组丝对：wire_a, gap, wire_b 位置及 dip 对比度
- `result.film_type` — 正片/负片
- `result.first_unresolved_group` — 首个不可分辨组号 (Dn)
""", encoding="utf-8")


async def main():
    parser = argparse.ArgumentParser(description="双丝 API 自测")
    parser.add_argument("--profile", help="单个 profile JSON")
    parser.add_argument("--profile-dir", default="outputs/double_wire_demo",
                        help="批量扫描目录")
    parser.add_argument("--output-dir", default="outputs/double_wire_api_test",
                        help="测试输出目录")
    args = parser.parse_args()

    # 收集 profile JSONs
    if args.profile:
        profiles = [Path(args.profile)]
    else:
        profiles = find_profile_jsons(args.profile_dir)

    if not profiles:
        print(f"Error: 未找到 _profile.json 文件", file=sys.stderr)
        sys.exit(1)

    print(f"[test] 找到 {len(profiles)} 个 profile JSON")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 API 服务
    init_double_wire_api()

    results = []
    try:
        for profile_path in profiles:
            r = await test_one(profile_path, output_dir)
            results.append(r)
    finally:
        close_double_wire_api()

    # 写报告和 README
    write_report(results, output_dir)
    write_readme(output_dir)

    print(f"\n[test] 数据包已生成: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
