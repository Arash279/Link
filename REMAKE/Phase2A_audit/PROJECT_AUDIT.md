# PROJECT AUDIT — Phase 2A

> 2026-10-06 更新：已收到11项人工回答，见[人工回答整合](HUMAN_REVIEW_INTEGRATION.md)。下文原调查文字保留；涉及人工意图与待回答状态时，以本次整合说明为准。

本次操作：读取用户指定回答文件、保存字节一致副本与SHA-256，备份5份审计文档，新增整合表并为15个相关问题加入修订注记。仅写入审计目录；未运行研究脚本或修改历史项目。此前 final_validation.json 对应原审计时点；本次验证另存 human_review_validation.json。

日期：2026-10-05（Asia/Singapore）。本目录是调查工作区，不是最终restart。

## 操作边界

- 原项目`D:\Desktop\LinkCodex`只读；数据库用SQLite URI `mode=ro`，并启用`PRAGMA query_only=ON`。
- 新建内容只在`C:\Users\35789\Documents\ChatGPT\LinkCodex\Phase2A_audit`。
- 未删除、移动、重命名、修复历史文件；未修改报告；未指定canonical；未创建`D:\Desktop\LinkCodex_restart`。
- 独立数值验证只复制4个候选脚本，保留原内容并在新目录导入。没有运行历史`main()`，避免硬编码写入。`-B`禁用字节码写入，Matplotlib配置和输出指向新目录。
- 未访问外网。学术研究/PDF/演示文稿技能用于审计方法和本地资料读取，未上传材料、未调用其他模型、未委派子代理。

## 调查日志

| 步骤 | 检查内容 | 发现 / 推断 / 下一步 |
|---|---|---|
| 01 | 读取Phase 2A要求及工作区 | 四个主交付+日志，历史目录只读；新目录不存在既有文件冲突 |
| 02 | 源文件SHA-256、Git状态/差异/路径历史 | 282个项目/数据侧文件建立初始清单；报告/MD/workflow未跟踪，不能只依赖Git |
| 03 | 81个Python文件静态浏览、32个模型相关脚本AST索引 | 11/12/14参数不能唯一识别拓扑；tmp另有R+(L∥C)；旧Git另有端口并Cad |
| 04 | PNG元数据与报告电路图/热图 | 主图是Snipaste Screenshot；热图12参数；正文拓扑在历史tmp确有对应 |
| 05 | 3份PPTX文字及notes、4份PDF文本 | 1月已有eta问题和拓扑探索；本地参考论文是背景资料，不作为本项目仪器证据；重建PDF首个参数页已渲染查看 |
| 06 | 代码引用的D:\Desktop\tmp输出目录 | 找到curver/nutsver历史残差，hash相同；相关统计/报告草稿候选清单已登记。未扫描其他无关目录 |
| 07 | SQLite只读查询与全部63表基本检查 | 主AP1.5/exp10筛后1590点；其他32表有重复频率；无非有限/非正值或倒序记录（所查三列） |
| 08 | 现有残差+测量恢复预测 | GP实/虚残差符号为data−sim，按此恢复出历史预测；残差可核对SSE/RMSE |
| 09 | 复制try家族与旧12参数脚本，受控DE+LS一次 | 14参数seed0解逐点匹配历史残差；raw与相对幅值指标匹配报告显示精度；新的全精度参数已保存 |
| 10 | 生成主图并查看、比较RGB数组 | 与报告主图1800×1200所有RGB像素相同；PNG文件元数据不同，未宣称二进制相同 |
| 11 | 三个try确定性函数与NUTS内部代数 | 16个关键函数AST相同；4组随机参数输出相同；NUTS内部实虚代数用NumPy替身求值，与主模型最优点最大差2.77e-11Ω。未执行PyTensor编译/采样 |
| 12 | 初始化、Rs来源、HP30、消融/目标/GP | Rs外推可恢复；HP30显式使用默认输入；GA/DE/LS预算不完全一致；NUTS打印LS指标；Mean不是平均预测 |
| 13 | 写28题、模型树、短重建说明、11项人工复核 | 每题包括支持、冲突、备选、置信度、验证和证伪条件；未将猜测升为正式叙事 |
| 14 | 最终文档结构/链接及源文件校验 | 结果见evidence/final_validation.json；不靠聊天记录维持结论 |

## 受控验证的精确约束

脚本：`scripts/controlled_check.py`。源副本：`checks/source_copies/`。输入仍从外部SQLite只读读取；复制不包括数据库。运行环境为已存在的common环境：Python3.9.25、NumPy1.26.4、SciPy1.12.0、Matplotlib3.9.4。没有安装或升级研究依赖。

复现设置：`exp_10`、`0<f<=1e8`、seed0、n_starts120/top_k10、DE maxiter60/popsize10、LS max_nfev200、soft_l1、MAD Re/Im缩放、梯度权重0.3–4、f_scale从初值残差MAD得到。该例1590点少于N_SAMPLES=2000，因此未实际抽样；绘图4000个对数模型点。

结果：SSE=54465351.850388095；RMSE=130.8719570241928；相对幅值SSE=74.48344715393908；相对RMSE=0.2164369083981353；best_cost=1528.2956920576398。新曲线与外部残差恢复曲线最大差1.46e-14Ω。AIC/BIC与报告三位小数存在约0.0005差异，已在问题登记保留。

新参数是本轮计算结果，不是历史原件。没有把无附加支路、旧tmp拓扑、60次消融或30HP重新全部运行；没有重新执行NUTS/GP拟合。主图像素一致和残差一致足以支持此次有限复现，不能外推所有实验均可信。

## 证据文件

- `evidence/source_manifest_before.json`：282个原文件路径、大小、hash、mtime；时间只作辅助。
- `evidence/external_manifest.json`：按代码线索追踪的外部残差、统计、草稿候选hash。
- `evidence/code_inventory.json`：模型参数表、关键函数源码/行号/AST hash。
- `evidence/git_history.txt`、`git_status.txt`、`git_diff.txt`及3个Git源码快照。
- `evidence/png_metadata.json`：图像尺寸、软件元数据、像素hash。
- `evidence/*.pptx.txt`、`*.pdf.txt`：文本提取副本，含slide/page定位；仅作为检索辅助，公式提取不视为完整数学校验。
- `checks/controlled_results.json`、`supplemental_results.json`：计算证据；后者的Git/AST/数据库检查以本轮只读审计生成，主复现由保存脚本可重跑。
- `checks/parameter_register_v3.csv`、`v3_rerun_curve.csv`、`recovered_*`、`v3_rerun_plot.png`：新计算产物。
- `evidence/final_validation.json`：原文件不变、链接/文档完整性、主图RGB一致验证。

## 技术性中断与处理

文档生成器首次在Python3.9遇到f-string表达式反斜杠语法限制；只修改新工作区生成器为正斜杠路径后成功。隔离绘图出现Agg不可show提示，但保存图片成功；该提示不影响像素验证。历史模型/报告均未为消除这些提示而修改。

## 人工复核之后

先把HUMAN_REVIEW答案记为HUMAN_RECOLLECTION，优先验证具体路径线索。没有外部证据的记忆保持与CONFIRMED分离。明确模型身份、拓扑叙述与验证范围后，再设计restart；本阶段不实施修复或迁移。
