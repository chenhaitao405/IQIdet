本文档面向开发者，临时记录待办事项。

 - [ ] SOP编写
 - [ ]
修复 BAM 双丝算法，直到 `  python scripts/debug/validate_bam_gt.py \
    "outputs/double_wire_demo_3/wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_profile.json" \
    "outputs/double_wire_demo_3/wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_groundtruth.json"` 输出 PASS；PASS 定义为 film_type 与 GT 一致、算法 pair 数量等于 GT、8/8 GT pairs 一一匹配、无 extra/missing pair、每组 wire_a/
  gap/wire_b 最大位置误差 <=5px、全部点位平均误差 <=3px；第一阶段只优化配对位置，不把 dip 阈值作为硬指标；每一轮优化都以点位平均误差提交一次。
  修复方案可参考docs/需求/方案调研.md docs/技术路线/BAM双丝算法原理.md 