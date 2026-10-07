# Current Best Reconstruction

日期：2026-10-05。人工复核前版本；不指定canonical、不改模型、不修报告。

## 如果今天必须解释这个项目

这个项目用SQLite中的电机端口幅相谱拟合三相灰箱等效电路，目标是100Hz至约100MHz的端口响应。早期的11参数电路无法充分表现UHF尾部；研究者试验了附加电感与多种RC/LC结构。12参数版本成为已保存的优化消融benchmark；14参数`Lad ∥ (Rad + Cad)`版本可以精确重现报告主图。这一数值对应关系现在有直接验证证据，具体历史运行文件/决策仍是工作假设。

## 已闭合的数值证据（CONFIRMED）

- 数据：`D:\Desktop\EE5003\data\AP_1p5.db / exp_10`，A1–B1C1星形短接；Freq/Hz、Zabs/Ω、Phase/deg。
- 1601原点筛为1590点，100至99967334.06Hz；N_SAMPLES=2000使该例不下采样。
- 复制的try/try.py，以原14参数初值、原bounds、seed0、DE60×10、多起点120/top10、soft_l1、MAD与0.3–4权重运行。
- 得到RMSE_raw=130.8719570241928、SSE_raw=54465351.850388095、相对幅值RMSE=0.2164369083981353、best_cost=1528.2956920576398。
- 与原外部`curver_gp_residual.csv`恢复曲线差不超过1.46e-14Ω；输出图与报告`fig_curver.png`全部RGB像素一致。
- try.py、try2.py、NUTSVerTry.py的NumPy模型/优化关键函数相同；不能据此确定当时点击的文件。

详见[受控验证数值记录](</C:/Users/35789/Documents/ChatGPT/LinkCodex/Phase2A_audit/checks/controlled_results.json>)、[像素、AST、数据质量与历史拓扑记录](</C:/Users/35789/Documents/ChatGPT/LinkCodex/Phase2A_audit/checks/supplemental_results.json>)及[恢复参数表](</C:/Users/35789/Documents/ChatGPT/LinkCodex/Phase2A_audit/checks/parameter_register_v3.csv>)。这些新参数是本轮重建结果，不是历史原件或唯一物理解。AIC/BIC与报告文字最后一位有约0.0005差别；PNG元数据不同，不能称文件hash一致。

## 最可信的运行图景

```text
多接线测量SQLite → 特征/解析工作与手工先验
                              ↓
             历史脚本中的固定p0 + 参数边界
                              ↓
    11参数 / 12参数 / 两种不同14参数拓扑的探索
                              ↓
                DE粗搜 + 多起点鲁棒LS
                              ↓
     幅相图、raw/relative指标、GP残差及NUTS展示

12参数线 → 六组方法×10种子 → stdout日志 → CSV统计 → 消融图
12参数线 → 当前workflow → 自动提取/拟合/同表验证
14参数L∥(R+C)线 → 报告主图（本轮可精确重现）
```

## 不能写进正式叙事的过度结论

1. **不能说只有一个V3。** 正文R+(L∥C)在tmp里确有实现；图/主数值为另一拓扑。正文混版是MEDIUM解释，不能直接宣布笔误。
2. **不能说消融严格验证了主模型。** 已保存热图/日志是12参数；GA/LS/DE预算及起点数也不完全对齐。混合阶段解释HIGH。
3. **不能说NUTS证明LS全局最优。** raw打印和GP仍用LS参数；NUTS是Gaussian而非soft_l1，bounds围绕LS解重建，Mean是Z(exp(E[log p]))。
4. **不能说所有参数物理确定。** Rrs/Rsf触边，nLls与原etaLls差8个数量级；曲线重复不代表唯一物理参数。
5. **不能说SQLite就是原始仪器文件。** 它是已检查范围内最早可用输入；设备、校准、导入和工况未闭合。
6. **不能说30HP已成功验证。** 它明确沿用1.5HP默认输入，stage1多项失败，已有结果只证明试跑过。
7. **不能把同表validation当独立验证。** 主脚本DO_VAL=False；多配置已参与参数提取。

## 重新进入项目时先看什么

先读[HUMAN_REVIEW.md](</C:/Users/35789/Documents/ChatGPT/LinkCodex/Phase2A_audit/HUMAN_REVIEW.md>)；需要详细反证时查[OPEN_QUESTIONS.md](</C:/Users/35789/Documents/ChatGPT/LinkCodex/Phase2A_audit/OPEN_QUESTIONS.md>)；模型分支看[MODEL_EVOLUTION.md](</C:/Users/35789/Documents/ChatGPT/LinkCodex/Phase2A_audit/MODEL_EVOLUTION.md>)。完整调查操作记录在[PROJECT_AUDIT.md](</C:/Users/35789/Documents/ChatGPT/LinkCodex/Phase2A_audit/PROJECT_AUDIT.md>)。

下一步应先完成历史意图/工况复核，再进入技术重建或restart设计。所有修正建议仍未实施。

**This is a working hypothesis, not a confirmed conclusion.** 已验证事实与历史/物理解释分开记录，人工回答也只作为线索，不能覆盖相反文件证据。
