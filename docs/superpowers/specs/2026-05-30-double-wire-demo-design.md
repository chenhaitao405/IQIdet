# 双丝像质计分辨率 Demo 可视化工具设计规格

**日期：** 2026-05-30  
**状态：** 草案  
**前置背景：** 已解读 JBT 7902-2025 双丝型像质计标准、已确认客户需求（`docs/需求/双丝像质计分辨率.md`），设计规格见 `PROGRESS.md` Phase 2。

## 目标

开发一个可视化交互工具，用于验证双丝像质计剖面线提取和对比度计算的正确性，为后续自动判定算法打好基础。

## 非目标

- 不实现自动定位双丝各组（由 OBB 检测模型负责，本 demo 阶段跳过）。
- 不实现自动判定第一个不可分辨组的逻辑（留到可视化验证通过后再写）。
- 不涉及单丝型 IQI 逻辑。
- 不集成到 `run_iqi_grade_infer.py` 交付管线。

## 用户操作流程

```
1. 打开双丝像质计图像（16-bit TIFF）
2. 鼠标左键逆时针点击 OBB 四个顶点（p0→p1→p2→p3），框选整个双丝像质计
   - 右键可撤销最后一个顶点
   - 第 4 点点击后自动闭合 OBB
3. 系统沿 OBB 长边中点画一条剖面线（横穿所有线对），弹出 matplotlib 灰度曲线图
4. 拖动 Trackbar 滑块调节剖面线位置（0%~100%），曲线实时更新
5. 按 F 键切换正/负片模式（适配不同成像类型）
6. 按 S 键保存结果（叠加图 + 剖面数据 JSON）
7. 按 R 键重置，重新框选下一张图
```

## OBB 与剖面线的几何关系

```
       短边（沿丝方向）
  ┌──────────────────────┐
  │  ═══  ═══    ← D1   │  ↑
  │  ═══  ═══    ← D2   │  长
  │  ═══  ═══    ← ...  │  边
  │  ═══  ═══    ← Dn   │  │   剖面线沿长边方向 →
  └──────────────────────┘  ↓
```

- 金属丝与 OBB **短边**平行。
- OBB **长边**垂直于金属丝，横跨 D1 → Dn 所有线对。
- 在长边方向取一条直线，即为**剖面线**。
- 剖面线一次性切过所有线对的"两根丝+间隙"结构，灰度曲线呈现一串"峰-谷-峰"序列。

## 交互设计

### 交互流程与状态机

基于 OpenCV HighGUI 的交互方案，三种状态自动流转：

```
                    左键点击 p0
    ┌──────────────────────────────────┐
    ↓                                  │
  IDLE ──左键点击 p0──→ COLLECTING ──p3 点击──→ LOCKED
  (等待框选)            (收集顶点 1~4)  (自动闭合)  (OBB锁定,剖面计算完成)
    ↑                      │     ↑                    │
    │       右键：撤销最后顶点     │                    │
    │                      └─────┘                    │
    └──────────── R 键：重置 ─────────────────────────┘
```

**状态说明：**

| 状态 | 触发条件 | 行为 |
|------|----------|------|
| `IDLE` | 启动 / R 键重置 | 鼠标十字准星，等待框选 |
| `COLLECTING` | 左键点击 p0 | 实时显示已点顶点（黄点）和连线（黄虚线），左下角显示 "点 N/4" |
| `LOCKED` | p3 点击后自动闭合 | OBB 绿实线 → 计算剖面 → 更新 matplotlib 曲线 → Trackbar 可调 |

**键盘绑定：**

| 按键 | 可用状态 | 功能 |
|------|----------|------|
| 左键点击 | IDLE / COLLECTING | 添加 OBB 顶点（最多 4 个），第 4 点闭合后自动触发剖面计算 |
| 右键点击 | COLLECTING | 撤销最后一个顶点 |
| `R` | 任意 | 重置 OBB，回到 IDLE |
| `S` | LOCKED | 保存当前结果（叠加图 + 剖面数据 JSON） |
| `Q` / `ESC` | 任意 | 退出程序 |
| `H` | 任意 | 显示/隐藏帮助覆盖层（半透明按键说明叠加在原图上） |
| `F` | LOCKED | 切换正片/负片模式，反转波峰/波谷检测方向 |

**Trackbar：**

| 控件 | 范围 | 功能 |
|------|------|------|
| 剖面偏移滑块 | 0% ~ 100% | 沿 OBB 短边方向调整剖面线位置，拖动时 matplotlib 曲线实时更新 |

### OBB 框选方式

**方案**：4 点点击模式（更精确控制任意角度）

```
     p0 ●──────────● p1      ← 短边（沿金属丝方向）
        │  ═══╪═══╪═══  │
        │  ═══╪═══╪═══  │     金属丝与短边平行
        │  ═══╪═══╪═══  │
     p3 ●──────────● p2      ← 短边
        └─ 长边（剖面方向）─┘

点击顺序: p0 → p1 → p2 → p3（逆时针）
```

- **点击顺序**：p0（左上）→ p1（右上）→ p2（右下）→ p3（左下），逆时针
- **实时预览**：每点击一个点显示黄色圆点（r=5px）和黄色虚线连线
- **撤销操作**：右键点击撤销最后一个顶点
- **完成条件**：点击第 4 个点后自动闭合 OBB，触发剖面线计算
- **适用角度**：天然支持 OBB 旋转任意角度，无需额外旋转控件

### 剖面线位置

- **默认位置**：OBB 长边中点（垂直于金属丝方向，对应 Trackbar 50%）
- **调整方式**：OpenCV `createTrackbar("偏移%", ...)` 滑块，沿短边方向 0% ~ 100% 连续调节
- **实时更新**：拖动 Trackbar 时自动重提取剖面线并更新 matplotlib 曲线图
- **用途**：验证不同位置的信号一致性、寻找最佳信号区域

### 双窗口布局

**主窗口（OpenCV）**：原图 + 叠加标注

| 元素 | 样式 | 出现状态 |
|------|------|----------|
| OBB 已点击顶点 | 🟡 黄色圆点（r=5px） | COLLECTING |
| OBB 顶点连线 | 黄色虚线（1px） | COLLECTING |
| OBB 边框 | 🟢 绿色实线（2px） | LOCKED |
| 剖面线 | 🔴 红色实线（1px），沿长边方向 | LOCKED |
| 当前鼠标位置 | ➕ 十字准星 | IDLE / COLLECTING |
| 状态指示文字 | 左下角："就绪" / "点 N/4" / "已锁定" | 所有状态 |
| 帮助覆盖层 | 半透明黑色背景 + 白色按键说明 | H 键触发 |
| Trackbar | 窗口底部 `偏移%: 0 ──●── 100` | 始终显示 |

**副窗口（matplotlib）**：灰度曲线图
- X 轴：沿剖面线的像素位置（0 ~ 剖面线长度）
- Y 轴：灰度值归一化到 [0, 1]，剖面提取保持 16-bit 精度
- 曲线：蓝色实线（`#4C78A8`），亚像素插值采样
- 自动检测标注：
  - 波峰：红色 ▲ 标记 + 数值标签（如 `a=0.72`）
  - 波谷：蓝色 ▼ 标记 + 数值标签（如 `c=0.32`）
- 标题（纯英文/数字，无需中文字体）：`{stem} | OBB: {w}x{h} @ {angle:.1f}° | offset:{pct}% | {film_type}`
- 支持缩放和平移（matplotlib 内置工具栏）
- 更新策略：`ax.clear()` + 重绘（清空重绘，简单可靠，10ms 内完成）

### 16-bit 图像处理

双丝像质计图像通常是 16-bit TIFF，需要特殊处理：

```python
# 读取 16-bit 图像
image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 保持原始位深

if image.dtype == np.uint16:
    # 显示用：归一化到 8-bit（窗宽窗位调整）
    display_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 剖面提取用：保持 16-bit 精度
    profile = extract_profile_16bit(image, line_coords)
```

## 接口预留

为后续自动判定逻辑预留以下函数签名，本次不实现：

```python
def compute_contrast(profile: np.ndarray) -> float:
    """根据剖面灰度曲线计算 Contrast = (a + b - 2c) / (a + b)"""
    pass

def find_first_unresolved_group(
    profiles: list[np.ndarray], threshold: float = 0.2
) -> int | None:
    """从粗到细遍历，返回第一个 Contrast < threshold 的组编号（1-indexed）"""
    pass
```

## 代码落盘方案

### 文件结构

```
scripts/debug/double_wire_demo.py          # 主脚本（交互入口）
src/gauge/imaging/profile.py              # 剖面线提取工具函数（可复用）
tests/test_double_wire_profile.py         # 单元测试（可选）
```

### 核心模块职责

#### 1. `src/gauge/imaging/profile.py` - 剖面线提取（可复用层）

纯函数工具模块，不依赖交互逻辑，便于后续集成到自动化管线。

**核心函数**：

```python
def extract_profile_along_line(
    image: np.ndarray,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    num_samples: Optional[int] = None
) -> np.ndarray:
    """
    沿直线提取灰度剖面，支持亚像素插值。
    
    Args:
        image: 输入图像（支持 8-bit 或 16-bit）
        start_point: 起点坐标 (x, y)
        end_point: 终点坐标 (x, y)
        num_samples: 采样点数（None 则按像素距离自动计算）
        
    Returns:
        一维灰度剖面数组
        
    Implementation:
        使用 cv2.remap() 或 scipy.ndimage.map_coordinates() 进行亚像素插值
    """
    
def get_obb_long_edge_midline(
    obb_points: np.ndarray  # shape (4, 2)
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    从 OBB 四点计算长边中点连线（剖面线）。
    
    Args:
        obb_points: OBB 四个顶点坐标，逆时针顺序
        
    Returns:
        (start_point, end_point): 剖面线的起点和终点
        
    Logic:
        1. 计算四条边的长度
        2. 找到最长的两条对边（长边）
        3. 计算两条长边的中点
        4. 返回连接两个中点的直线
    """
    
def detect_peaks_valleys(
    profile: np.ndarray,
    min_distance: int = 10,
    prominence: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    检测波峰和波谷位置。
    
    Args:
        profile: 一维灰度剖面
        min_distance: 相邻峰值的最小间距（像素）
        prominence: 峰值显著性阈值（相对于局部基线的高度）
        
    Returns:
        (peak_indices, valley_indices): 波峰和波谷的索引数组
        
    Implementation:
        使用 scipy.signal.find_peaks()
        波谷检测：对 -profile 应用 find_peaks()
    """
```

**接口预留**（本次不实现）：

```python
def compute_contrast(
    profile: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray
) -> float:
    """
    根据剖面灰度曲线计算 Contrast = (a + b - 2c) / (a + b)。
    
    Args:
        profile: 一维灰度剖面
        peak_indices: 波峰位置索引
        valley_indices: 波谷位置索引
        
    Returns:
        对比度值（0~1）
        
    Note:
        Phase 2 实现。需要处理：
        - 正片/负片判定（峰谷顺序）
        - a、b 取邻域均值还是峰值点
        - 异常情况处理（峰谷数量不匹配）
    """
    raise NotImplementedError("Phase 2 implementation")

def find_first_unresolved_group(
    profiles: List[np.ndarray],
    threshold: float = 0.2
) -> Optional[int]:
    """
    从粗到细遍历，返回第一个 Contrast < threshold 的组编号（1-indexed）。
    
    Args:
        profiles: 每组线对的剖面灰度曲线列表（D1 → Dn）
        threshold: 对比度阈值（默认 0.2 即 20%）
        
    Returns:
        第一个不可分辨组的编号（1-indexed），None 表示全部可分辨
        
    Note:
        Phase 2 实现。
    """
    raise NotImplementedError("Phase 2 implementation")
```

#### 2. `scripts/debug/double_wire_demo.py` - 交互主脚本

**类设计**：

```python
class DoubleWireDemo:
    """双丝像质计剖面线交互可视化工具"""
    
    # ── 状态常量 ──
    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_LOCKED = "locked"
    
    def __init__(self, image_path: str, output_dir: Optional[str] = None):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir) if output_dir else None
        
        # 图像数据
        self.image_raw = None       # 16-bit 原始图像（剖面提取用）
        self.image_display = None   # 8-bit 归一化显示图像
        
        # 交互状态
        self.state = self.STATE_IDLE
        self.obb_points = []        # 当前 OBB 顶点 (0~4 个)
        
        # 剖面参数
        self.profile_offset_pct = 50        # Trackbar 偏移 0~100
        self.film_type = "positive"         # "positive" | "negative"
        self.show_help = False              # 帮助覆盖层开关
        
        # 剖面数据
        self.profile_line = None            # (start_point, end_point)
        self.profile = None                 # 一维灰度剖面数组
        self.peak_indices = None            # 波峰索引
        self.valley_indices = None          # 波谷索引
        
        # 窗口与图形
        self.window_name = "Double Wire Demo"
        self.fig = None                     # matplotlib Figure
        self.ax = None                      # matplotlib Axes
        self.profile_line_artist = None     # Line2D artist
        
    def load_image(self):
        """加载图像，处理 16-bit，生成显示用 8-bit"""
        
    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调。
        
        IDLE / COLLECTING:
            - 左键 (cv2.EVENT_LBUTTONDOWN): 添加顶点
            - 右键 (cv2.EVENT_RBUTTONDOWN): 撤销最后顶点
        
        COLLECTING 第 4 个点:
            - 自动闭合 → self.state = LOCKED → self.update_profile()
        """
        
    def on_trackbar(self, value):
        """Trackbar 回调：更新 self.profile_offset_pct，若 LOCKED 则重新提取剖面"""
        
    def update_profile(self):
        """
        根据当前 OBB + offset_pct 更新剖面线。
        
        1. 调用 get_obb_long_edge_midline() + offset 偏移计算剖面线端点
        2. 调用 extract_profile_along_line() 在 image_raw 上提取灰度剖面
        3. 调用 detect_peaks_valleys() 检测波峰波谷（根据 film_type 调整方向）
        4. 调用 plot_profile() 更新 matplotlib 图窗
        """
        
    def draw_overlay(self) -> np.ndarray:
        """
        在 image_display 上绘制叠加标注。
        
        按状态绘制：
            - 顶点黄点 + 黄虚线 (COLLECTING)
            - OBB 绿实线 + 剖面线红实线 (LOCKED)
            - 左下角状态文字
            - 帮助覆盖层 (show_help=True 时)
        """
        
    def plot_profile(self):
        """
        matplotlib 清空重绘灰度曲线图。
        
        - ax.clear()
        - ax.plot(profile, color='#4C78A8')
        - 波峰: ax.plot(peaks, profile[peaks], 'r^') + 数值标注
        - 波谷: ax.plot(valleys, profile[valleys], 'bv') + 数值标注
        - 标题: f"{stem} | OBB: {w}x{h} @ {angle:.1f}° | offset:{pct}% | {film_type}"
        - fig.canvas.draw()
        """
        
    def save_results(self):
        """
        保存当前结果到输出目录。
        
        Outputs:
            - {stem}_overlay.png: 叠加标注的图像
            - {stem}_profile.json: 剖面数据（JSON 格式）
              {
                "image_path": str,
                "obb_points": [[x, y], ...],
                "profile_line": {"start": [x, y], "end": [x, y]},
                "profile_offset_pct": int,
                "film_type": "positive" | "negative",
                "profile_values": [float, ...],
                "peak_indices": [int, ...],
                "valley_indices": [int, ...]
              }
        """
        
    def run(self):
        """
        主循环。
        
        初始化:
            - cv2.namedWindow() + setMouseCallback()
            - cv2.createTrackbar("offset%", ...)
            - plt.ion() + 创建 Figure
        
        主循环:
            while True:
                overlay = self.draw_overlay()
                cv2.imshow(self.window_name, overlay)
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('r'):      重置 OBB
                elif key == ord('s'):    保存结果 (LOCKED)
                elif key == ord('q') or key == 27:  退出
                elif key == ord('h'):    切换帮助覆盖层
                elif key == ord('f'):    切换正/负片 (LOCKED)
        """
```

**命令行接口**：

```python
"""
Usage:
    double_wire_demo.py <image_path> [options]
    
Arguments:
    <image_path>              双丝像质计图像路径（支持 16-bit TIFF）
    
Options:
    -h --help                 显示帮助信息
    --output-dir <dir>        输出目录 [default: outputs/double_wire_demo]
    --window-size <size>      显示窗口最大尺寸 [default: 1200]
"""
```

### 依赖关系

```python
# scripts/debug/double_wire_demo.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from docopt import docopt

from gauge.imaging.profile import (
    extract_profile_along_line,
    get_obb_long_edge_midline,
    detect_peaks_valleys,
)
```

**新增依赖**：
- `scipy`：波峰检测（`scipy.signal.find_peaks`）
- `matplotlib`：灰度曲线可视化
- `docopt`：命令行参数解析（已在 `fclip_valid.py` 中使用）

### 复用现有代码

- `src/gauge/imaging/preprocess.py` 中的 `order_points()` 可用于 OBB 顶点排序
- `src/gauge/imaging/geometry.py` 中的坐标变换函数可供参考（但本 demo 不需要透视变换）

## 实现优先级

### Phase 1（本次 demo）

- [ ] 图像加载（支持 16-bit TIFF，`cv2.IMREAD_UNCHANGED`）
- [ ] 16-bit → 8-bit 显示归一化（`cv2.normalize`）
- [ ] OBB 四点交互框选（鼠标左键点击，逆时针 p0→p1→p2→p3）
- [ ] 右键撤销最后顶点
- [ ] 第 4 点自动闭合 OBB 并触发剖面计算
- [ ] 剖面线提取（沿 OBB 长边中点，`scipy.ndimage.map_coordinates` 亚像素插值）
- [ ] Trackbar 滑块调节剖面线短边偏移（0% ~ 100%，实时更新）
- [ ] matplotlib 灰度曲线可视化（清空重绘，纯英文标题）
- [ ] 波峰/波谷自动标注（`scipy.signal.find_peaks`，正片：峰-谷-峰）
- [ ] `F` 键正/负片切换（反转峰值检测方向）
- [ ] `H` 键帮助覆盖层（半透明按键说明）
- [ ] `R` 键重置 OBB 框选
- [ ] `S` 键保存结果（叠加图 PNG + 剖面数据 JSON）
- [ ] `Q` / `ESC` 退出程序
- [ ] 状态指示文字（左下角："就绪" / "点 N/4" / "已锁定"）

### Phase 2（后续自动判定）

- [ ] 实现 `compute_contrast()` 函数（波峰两波谷法）
- [ ] 实现 `find_first_unresolved_group()` 函数（遍历判定）
- [ ] 正/负片自动判定逻辑（基于 D1 信号方向）
- [ ] 集成 YOLO-OBB 自动检测双丝 ROI
- [ ] 查表输出不清晰度物理值（mm）

## 验证标准

### 功能验证

- [ ] 能正常打开 16-bit TIFF 图像（保持原始位深用于剖面提取）
- [ ] OBB 四点框选交互流畅，顶点顺序正确（逆时针）
- [ ] 右键撤销最后一个顶点功能正常
- [ ] 剖面线自动计算正确（沿长边方向，垂直于金属丝）
- [ ] Trackbar 滑块调节剖面线偏移，曲线实时更新
- [ ] F 键正/负片切换，峰值检测方向正确反转
- [ ] H 键帮助覆盖层显示/隐藏正常
- [ ] 灰度曲线与实际图像肉眼观察一致
- [ ] 能同时看到原图（带 OBB + 剖面线叠加）和灰度曲线图
- [ ] 波峰/波谷自动检测准确（正片："峰-谷-峰"，负片："谷-峰-谷"）
- [ ] S 键保存：叠加图 PNG 和剖面数据 JSON 均正确输出

### 数据验证

使用 `outputs/候选双丝像质计/` 中的 311 张图像进行验证：

- [ ] 至少在 5 张不同图像上验证剖面提取正确性
- [ ] 验证正片和负片图像的信号模式差异
- [ ] 验证不同角度 OBB 的剖面线计算正确性
- [ ] 保存的 JSON 数据可被后续脚本正确读取

### 代码质量

- [ ] `src/gauge/imaging/profile.py` 中的函数有完整的 docstring
- [ ] 剖面提取函数支持亚像素插值（避免锯齿）
- [ ] 16-bit 图像处理全程保持精度（不提前转 8-bit）
- [ ] 异常情况有清晰的错误提示（如图像加载失败、OBB 点数不足）

## 测试数据

- **来源**：`outputs/候选双丝像质计/`（311 张图像）
- **筛选脚本**：`scripts/tool/filter_nocrack_no_roi.py`（已完成）
- **图像特征**：从 nocrack 数据集中筛选出未检测到单丝 ROI 的图像，大概率为双丝像质计

## 后续集成路径

本 demo 验证通过后，剖面提取和对比度计算逻辑将集成到主管线：

1. **新增 Stage**：`DoubleWireStage` 继承 `PipelineStage`
2. **触发条件**：图像文件名以 `GB`/`ISO` 开头，或 OCR 识别到 "JB D" 标识
3. **输入**：YOLO-OBB 检测到的双丝 ROI 多边形
4. **输出**：不清晰度值（mm）+ 第一个不可分辨组编号
5. **交付字段**：在 `iqi_grade_batch_v1` schema 中新增 `double_wire_resolution` 字段

详细集成方案见 `PROGRESS.md` Phase 2。
