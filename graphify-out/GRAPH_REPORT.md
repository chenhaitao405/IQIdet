# Graph Report - IQIdet  (2026-05-30)

## Corpus Check
- 1684 files · ~39,794,216 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 28 nodes · 27 edges · 4 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `34d705a2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]

## God Nodes (most connected - your core abstractions)
1. `Gauge 职责分层重排设计规格` - 13 edges
2. `分层职责` - 8 edges
3. `风险与控制` - 5 edges
4. `测试要求` - 4 edges
5. `目标` - 1 edges
6. `非目标` - 1 edges
7. `当前问题` - 1 edges
8. `目标目录结构` - 1 edges
9. ``app/`` - 1 edges
10. ``pipeline/`` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities (4 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.18
Nodes (10): Gauge 职责分层重排设计规格, 后续计划, 当前问题, 文档更新要求, 目标, 目标目录结构, 直接迁移策略, 迁移顺序 (+2 more)

### Community 1 - "Community 1"
Cohesion: 0.25
Nodes (8): `app/`, `domain/`, `imaging/`, `pipeline/`, `runtime/`, `services/`, `services/` 现状与优化, 分层职责

### Community 2 - "Community 2"
Cohesion: 0.40
Nodes (5): 风险：import churn 过大, 风险：循环导入, 风险：旧脚本 import 断裂, 风险：输出字段漂移, 风险与控制

### Community 3 - "Community 3"
Cohesion: 0.50
Nodes (4): 导入边界测试, 测试要求, 行为回归测试, 输出回归

## Knowledge Gaps
- **23 isolated node(s):** `目标`, `非目标`, `当前问题`, `目标目录结构`, ``app/`` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Gauge 职责分层重排设计规格` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`?**
  _High betweenness centrality (0.875) - this node is a cross-community bridge._
- **Why does `分层职责` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.459) - this node is a cross-community bridge._
- **Why does `风险与控制` connect `Community 2` to `Community 0`?**
  _High betweenness centrality (0.279) - this node is a cross-community bridge._
- **What connects `目标`, `非目标`, `当前问题` to the rest of the system?**
  _23 weakly-connected nodes found - possible documentation gaps or missing edges._