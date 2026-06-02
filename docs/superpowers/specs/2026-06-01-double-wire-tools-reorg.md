# 双丝工具脚本重组设计

**日期**: 2026-06-01
**状态**: 已批准

## 目标

将 `scripts/debug/` 下的三个双丝像质计脚本重组到 `scripts/double_wire/`，合并 OBB 采集和标注为单一工具，提炼通用函数到架构层。

## 文件迁移

### 新建目录

```
scripts/double_wire/
├── annotate.py              # OBB选择 + 剖面展示 + GT标注（合并）
├── validate_bam_gt.py       # GT验证（移动）
└── src/
    ├── __init__.py
    ├── obb_ui.py            # OBBSelector — OpenCV窗口交互
    ├── profile_view.py      # ProfilePlot — matplotlib剖面展示
    ├── annotation.py        # 标注交互 — matplotlib事件处理
    └── io_utils.py          # 图像加载、文件保存、路径工具
```

### 删除

- `scripts/debug/double_wire_demo.py`
- `scripts/debug/annotate_profile.py`
- `scripts/debug/validate_bam_gt.py`（移动后删除原文件）

## 函数归属

### 提升到 `src/gauge/imaging/profile.py`（架构通用层）

| 函数 | 来源 | 理由 |
|------|------|------|
| `normalize_profile_obb()` | double_wire_demo | OBB剖面方向归一化，通用几何 |
| `bam_pair_marker_indices()` | double_wire_demo | 从BAM配对提取丝/隙索引 |
| `build_groundtruth_payload()` | annotate_profile | 标注→GT构建，验证流程通用 |
| `pair_wire_markers()`（原`_pair_markers`） | annotate_profile | 丝-隙配对算法 |

### 留在 `scripts/double_wire/src/`（UI/IO专用）

| 模块 | 内容 |
|------|------|
| `obb_ui.py` | `OBBSelector` — OpenCV GUI，鼠标/键盘/叠加绘制 |
| `profile_view.py` | `ProfilePlot` — matplotlib figure管理，剖面+BAM可视化 |
| `annotation.py` | GT标注交互 — matplotlib click/key handlers |
| `io_utils.py` | `load_image()`, `save_profile_json()`, `default_groundtruth_path()` 等 |

### `validate_bam_gt.py` 内部保留

- `_fmt_err()` — 格式化helper
- `_pair_metrics()` — 配对指标计算
- 验证报告生成和可视化逻辑

## 合并后 `annotate.py` 交互流程

```
OBB选择 (OpenCV窗口)              标注模式 (matplotlib figure)
─────────────────────              ────────────────────────────
L-click → 加点 (4点→确认态)
Enter   → 锁定OBB, 显示剖面图
Trackbar → 调整offset, 实时更新BAM
A       → 进入标注模式  ────────→  p/v   → 切换峰/谷模式
                                  L-click → 加标记
                                  R-click → 删除最近标记
                                  u       → 撤销
                                  A/ESC   → 退出标注模式, 返回OBB窗口
S       → 保存 profile.json + groundtruth.json (两者同时)
Q/ESC   → 退出
```

## 状态机

```
IDLE → COLLECTING → CONFIRM → LOCKED ⇄ ANNOTATING
  ↑                                    │
  └──────────── R (reset) ─────────────┘
```

- **IDLE**: 等待首次点击
- **COLLECTING**: 收集OBB角点（1-3个点）
- **CONFIRM**: 4点完成，显示拟合OBB，等待Enter确认
- **LOCKED**: OBB锁定，剖面+BAM实时显示，trackbar可用
- **ANNOTATING**: 标注模式（matplotlib figure活跃），p/v/u 键可用

## 需更新的引用

### 测试文件

- `tests/test_double_wire_demo.py`:
  - `from scripts.debug.double_wire_demo import bam_pair_marker_indices, normalize_profile_obb`
  - → `from gauge.imaging.profile import bam_pair_marker_indices, normalize_profile_obb`
- `tests/test_annotate_profile.py`:
  - `from scripts.debug import annotate_profile`
  - → `from gauge.imaging.profile import build_groundtruth_payload`（函数级导入）
  - `annotate_profile.default_groundtruth_path` → `from scripts.double_wire.src.io_utils import default_groundtruth_path`
  - `annotate_profile.__doc__` → CLI doc测试需调整

### SOP文档

- `docs/sop/BAM-algorithm-dev-workflow.md`: 更新脚本路径
- `docs/sop/BAM-algorithm-validation.md`: 更新脚本路径
- `docs/技术路线/BAM双丝算法详解——validate_bam_gt实例.md`: 更新脚本路径

## 不在此次范围

- `generate_bam_flow_diagram.py`（文件已不存在）
- 历史设计文档 `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 中的旧路径引用（历史快照，不改）
