# IQI ROI存在即执行像质丝识别 设计说明

## 背景

当前 IQI 主流程中，像质丝识别 `FClip` 的执行前提不仅要求 ROI 检出和裁剪成功，还要求像质计标识解析成功。这样会导致以下问题：

- ROI 已经有效，但因为铅字牌/标识 OCR 失败，像质丝识别被直接跳过。
- 在“标识失败但 ROI 有效”的样本中，无法保留像质丝识别结果，影响调试和交付分析。

本次变更仅调整像质丝识别的执行门槛，不改变 IQI 主任务成功/失败的判定原则。

## 目标

- 只要 ROI 检出且裁剪成功，就执行像质丝识别。
- 当标识失败但 ROI 有效时，整体结果仍报错。
- 当标识失败但 ROI 有效时，仍保留真实的 `wire_count`。
- 当标识失败但 ROI 有效时，仍在 JSON 的 `visualization` 字段中保留识别到的 `wire_lines`。

## 非目标

- 不修改 ROI 检测与裁剪逻辑。
- 不修改像质计标识解析规则。
- 不修改等级计算规则。
- 不引入新的命令行参数或兼容开关。

## 现状问题

当前 `gauge/iqi_inferencer.py` 在选定 `selected_plate` 后，若 `selected_plate.ok == false`，会直接：

- 将 `wire` 置为 `skipped_marker_not_found`
- 记录 marker 错误
- 立即返回

因此 `FClip` 没有执行，`wire_count` 与 `visualization.wire_lines` 都不会产出。

## 设计方案

### 1. 触发条件调整

将像质丝识别的执行条件从：

- ROI 有效
- 且标识识别成功

调整为：

- 只要 ROI 有效

这里的 ROI 有效定义不变：

- 已检测到 ROI
- ROI 透视裁剪成功

若 ROI 检测失败或裁剪失败，仍沿用现有提前返回逻辑，像质丝识别不执行。

### 2. 标识失败场景的处理

当 ROI 有效但标识失败时：

- 仍执行 `FClip`
- `record["wire"]` 保存真实像质丝识别结果
- 主结果仍保持失败
- 主错误码仍以 marker 失败为主，不因为识别出像质丝而转为成功

也就是说，像质丝识别结果在该场景下是“附带诊断信息”，不是“主任务成功依据”。

### 3. 等级计算

等级计算仍依赖合法的像质计标识和像质丝数量。

因此当标识失败时：

- 不输出有效 `grade`
- 不将仅有的 `wire_count` 作为主任务成功条件

实现上可继续复用现有 `compute_iqi_grade` 规则，但最终主错误码优先保留 marker 错误。

### 4. 输出字段语义

本次变更后，以下字段语义明确如下：

- `ok/result_code/result_name/result_message`
  - 仍表示 IQI 主任务是否成功
  - 标识失败时，即使 `wire_count` 存在，整体仍失败

- `wire`
  - 在 ROI 有效时保存真实 `FClip` 输出
  - 不再因为 marker 失败而强制写入 `skipped_marker_not_found`

- `wire_count`
  - 继续从 `wire` 结果回填
  - 标识失败时允许非空

- `visualization.wire_lines`
  - 只要 `FClip` 输出了线段，就写入 JSON
  - 即使主任务失败也保留

### 5. 可视化要求

可视化逻辑保持当前数据结构不变，不新增字段。只需确保在“ROI 有效 + 标识失败”路径上：

- `record["wire"]` 为真实结果
- `_attach_visualization_payload()` 仍被调用

这样 `visualization.wire_lines` 会按当前逻辑自然生成。

## 实现边界

核心改动集中在 `gauge/iqi_inferencer.py`：

- 调整 marker 失败时的提前返回分支，避免在 ROI 有效情况下提前跳过 `FClip`
- 保留 ROI 无效时的提前返回
- 在 ROI 有效场景下，无论 marker 成功与否都执行 `FClip`
- 在最终错误聚合时，marker 失败仍优先作为主错误来源

不要求修改：

- `run_iqi_grade_infer.py`
- `gauge/iqi_rules.py`
- 交付 JSON schema

## 测试设计

新增一组针对 `IQIInferencer.infer_image_path()` 的行为测试，至少覆盖以下场景：

### 场景 A

ROI 有效，marker 失败，FClip 成功。

期望：

- 整体结果失败
- `result_code` 为 marker 类错误码
- `wire.status == "ok"`
- `wire_count` 为 FClip 实际结果
- `visualization.wire_lines` 被保留
- 不出现 `skipped_marker_not_found`
- `grade is None`

### 场景 B

ROI 无效。

期望：

- 保持当前行为
- 不执行 FClip
- 仍返回 ROI 错误

## 风险与兼容性

- 下游如果假设“主任务失败时 `wire_count` 一定为空”，本次变更会打破该假设。
- 但本次不改变字段名和结构，只改变失败场景下字段是否保留，兼容成本相对可控。

## 验收标准

- ROI 有效时总会执行像质丝识别。
- marker 失败时整体结果仍失败。
- marker 失败但 FClip 成功时，返回中能看到真实 `wire_count`。
- marker 失败但 FClip 成功时，JSON `visualization.wire_lines` 中能看到识别到的线段。
- ROI 无效场景行为不变。
