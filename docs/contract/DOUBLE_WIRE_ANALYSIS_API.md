# 双丝像质计分析 API 契约

本文约束 `analyze_double_wire()` 的输出 JSON schema，供集成方参考。

## 1. 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_base64` | `string` | strip 条带图像 base64 编码，支持 `data:image/png;base64,` 前缀。图像为剖面线±expand 区域沿线的子像素采样结果，灰度图。 |

## 2. 输出 JSON Schema

Schema 标识：`double_wire_analysis_v1`

### 2.1 顶层

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema` | `string` | 固定值 `"double_wire_analysis_v1"` |
| `ok` | `bool` | 分析是否成功 |
| `meta` | `object` | 元信息 |
| `result` | `object \| null` | 分析结果，`ok=false` 时为 `null` |

### 2.2 `meta`

| 字段 | 类型 | 说明 |
|------|------|------|
| `created_at` | `string` | ISO 8601 UTC 时间戳 |
| `strip_shape` | `[int, int]` | strip 图像尺寸 `[height, width]` |
| `algorithm` | `object` | 算法参数 |
| `algorithm.min_distance` | `int` | 峰谷检测最小间距 (px) |
| `algorithm.prominence` | `float` | 峰谷检测相对 prominience |

### 2.3 `result`

| 字段 | 类型 | 说明 |
|------|------|------|
| `film_type` | `string` | `"positive"`（丝=亮峰）或 `"negative"`（丝=暗谷） |
| `num_pairs` | `int` | 检测到的丝对数量 |
| `first_unresolved_group` | `int \| null` | 首个不可分辨组号（1-indexed），`null` 表示全部可分辨 |
| `profile` | `float[]` | 1D 灰度曲线，长度 = strip 宽度 |
| `background` | `float[]` | 背景拟合值，与 `profile` 等长 |
| `peaks` | `object[]` | 检测到的峰（局部极大值） |
| `valleys` | `object[]` | 检测到的谷（局部极小值） |
| `pairs` | `object[]` | 丝对详情 |

### 2.4 `peaks[]` / `valleys[]`

| 字段 | 类型 | 说明 |
|------|------|------|
| `idx` | `int` | profile 数组中的索引位置 |
| `gray` | `float` | 该位置的灰度值 |

### 2.5 `pairs[]`

| 字段 | 类型 | 说明 |
|------|------|------|
| `group` | `int` | 丝对组号 D1~Dn（1-indexed，从粗到细） |
| `wire_a_idx` | `int` | 第一根丝的 profile 索引 |
| `gap_idx` | `int` | 间隙的 profile 索引 |
| `wire_b_idx` | `int` | 第二根丝的 profile 索引 |
| `wire_a_gray` | `float` | 第一根丝的灰度值 |
| `gap_gray` | `float` | 间隙的灰度值 |
| `wire_b_gray` | `float` | 第二根丝的灰度值 |
| `dip_percent` | `float` | 对比度百分比 `100*(A+B-2C)/(A+B)` |

## 3. 片型约定

| `film_type` | 丝对应 | 间隙对应 |
|-------------|--------|----------|
| `"positive"` | `peaks`（亮峰） | `valleys`（暗谷） |
| `"negative"` | `valleys`（暗谷） | `peaks`（亮峰） |

## 4. 分辨率判定

- 从 D1（最粗）遍历到 Dn（最细）
- 第一个 `dip_percent < 20%` 的组即为 `first_unresolved_group`
- 若所有组 `dip_percent >= 20%`，`first_unresolved_group = null`

## 5. 错误响应

`ok=false` 时：

| `result_code` | `result_name` | 说明 |
|---------------|---------------|------|
| `5001` | `invalid_image` | 图像解码失败或为空 |
| `5002` | `profile_too_short` | strip 宽度 < 3，无法分析 |
| `5003` | `no_extrema_found` | 未检出峰或谷 |
| `5999` | `internal_error` | 内部异常 |

## 6. 示例

### 正片 7 对丝

```json
{
  "schema": "double_wire_analysis_v1",
  "ok": true,
  "meta": {
    "created_at": "2026-06-04T12:00:00Z",
    "strip_shape": [42, 519],
    "algorithm": { "min_distance": 5, "prominence": 0.03 }
  },
  "result": {
    "film_type": "positive",
    "num_pairs": 7,
    "first_unresolved_group": 8,
    "profile": [245.1, 244.8, 244.2, "..."],
    "background": [246.0, 245.9, 245.7, "..."],
    "peaks": [
      {"idx": 125, "gray": 208.5},
      {"idx": 163, "gray": 207.1}
    ],
    "valleys": [
      {"idx": 106, "gray": 245.3},
      {"idx": 144, "gray": 217.2}
    ],
    "pairs": [
      {
        "group": 1,
        "wire_a_idx": 125, "gap_idx": 144, "wire_b_idx": 163,
        "wire_a_gray": 208.5, "gap_gray": 217.2, "wire_b_gray": 207.1,
        "dip_percent": 45.2
      }
    ]
  }
}
```
