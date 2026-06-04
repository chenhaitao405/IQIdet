# 双丝像质计分析 API 集成验证

参考输入/输出，供集成方验证 API 调用链是否正确。

## 目录结构

```
├── strips/           ← 7 张 strip PNG（API 输入）
├── results/          ← 对应的 API 输出 JSON（参考结果）
└── test_report.txt   ← 批量测试摘要
```

## 验证方法

### 1. 写集成脚本

用你的语言，对每张 strip PNG 调用 API。

Python 最小示例：

```python
import asyncio, base64, json, os
from double_wire_api import (
    init_double_wire_api, compute_double_wire, DoubleWireRequest,
    close_double_wire_api,
)

async def test_one(strip_path, output_path):
    with open(strip_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    req = DoubleWireRequest(image_base64=b64)
    resp = await compute_double_wire(req)
    with open(output_path, "w") as f:
        json.dump(resp.model_dump(), f, indent=2, ensure_ascii=False)

async def main():
    init_double_wire_api()
    strips_dir = "test_data/double_wire_api_test/strips"
    out_dir = "your_output"
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(strips_dir)):
        if fname.endswith("_strip.png"):
            stem = fname.replace("_strip.png", "")
            await test_one(f"{strips_dir}/{fname}", f"{out_dir}/{stem}_result.json")
    close_double_wire_api()

asyncio.run(main())
```

### 2. 对比结果

你的输出与 `results/` 下对应 JSON 对比。允许误差：

| 字段 | 允许误差 |
|------|----------|
| `film_type`, `num_pairs`, `first_unresolved_group` | 完全一致 |
| `pairs[].wire_a/gap/wire_b_idx` | ±2 px |
| `pairs[].dip_percent` | ±2% |
| `profile`, `background` | ±5 灰度值 |

### 3. 全部匹配 = 集成通过
