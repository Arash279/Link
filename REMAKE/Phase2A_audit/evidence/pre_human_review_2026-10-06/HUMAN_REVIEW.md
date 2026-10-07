# Human Review — 只复核意图与外部线索

你不需要重新审计代码，也不需要批准某个模型为canonical。请按记忆回答；不确定就选Not sure。回答作为下一轮线索保存，不会自动变成事实或覆盖文件证据。

本轮已经数值验证主图，不再问你“图是否能复现”。真正需要你帮助的是最终意图、版本拼接、测量工况和缺失记录的位置。建议先答HR-01、03、05、07、08。

## HR-01

你最终打算采用的是哪种端口附加结构？

Evidence suggests: 主图/代码支持L∥(R+C)；正文与旧tmp支持R+(L∥C)。

Confidence: HIGH（主图身份）；MEDIUM（最终研究意图）。对应OPEN_QUESTIONS：Q01/Q11。

Your answer:

- [ ] L∥(R+C)
- [ ] R+(L∥C)
- [ ] 只加L
- [ ] 当时尚未定稿
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-02

报告正文是否可能保留了更早R+(L∥C)方案的文字？

Evidence suggests: 这种旧拓扑确有代码；不是凭空猜测的笔误。

Confidence: MEDIUM。对应OPEN_QUESTIONS：Q11。

Your answer:

- [ ] Yes
- [ ] No
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-03

你是否把12参数消融与14参数主拟合结果合并进同一份报告？

Evidence suggests: 热图12行、消融日志p=12，主图14参数精确重现。

Confidence: HIGH。对应OPEN_QUESTIONS：Q05。

Your answer:

- [ ] Yes
- [ ] No
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-04

当前workflow的创建意图是什么？

Evidence suggests: 它调用12参数，并非报告14参数模型的模块化实现。

Confidence: MEDIUM。对应OPEN_QUESTIONS：Q04。

Your answer:

- [ ] 整理已有12参数baseline
- [ ] 有意回退并替代14参数
- [ ] 只为临时演示/试跑
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-05

nLls从约1.7806e-2 H变成1.7806e-10 H，当时是否有意调整？

Evidence suggests: 尾数保留、指数差8位；原公式被计算但不使用。

Confidence: LOW（修改动机）。对应OPEN_QUESTIONS：Q14。

Your answer:

- [ ] 有意调整，有物理/数值理由
- [ ] 可能误录后沿用
- [ ] 记得另有定义/单位
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-06

exp_10作为主案例的选择理由是什么？

Evidence suggests: 代码反复选它，多配置用于提取；未找到质量优选记录。

Confidence: MEDIUM。对应OPEN_QUESTIONS：Q07。

Your answer:

- [ ] 代表性三相端口
- [ ] 数据质量最好
- [ ] 最早跑通，沿用案例
- [ ] 其他已记录理由
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-07

原始测量的仪器、校准/夹具记录或建库源文件是否还在别处？

Evidence suggests: 本次找到SQLite和接线表，未找到原始仪器文件。

Confidence: UNKNOWN。对应OPEN_QUESTIONS：Q09。

Your answer:

- [ ] Yes，可提供本地路径
- [ ] No，可能未保存
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-08

阻抗测量时电机的实际运行状态是什么？

Evidence suggests: fit_rs注释称no-load/turning，但无独立测量记录支持。

Confidence: LOW。对应OPEN_QUESTIONS：Q15。

Your answer:

- [ ] 静止且未供电
- [ ] 旋转/空载
- [ ] 不同配置工况不同
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-09

30HP分支沿用1.5HP先验是否属于有意压力测试？

Evidence suggests: 代码明确打印“stage1仅诊断、fit用defaults”，不是仅靠猜测。

Confidence: MEDIUM（意图）。对应OPEN_QUESTIONS：Q10。

Your answer:

- [ ] Yes
- [ ] No，本应按30HP配置
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-10

是否曾保存NUTS的原始trace/MAP参数/链诊断，或另一套14参数消融？

Evidence suggests: 当前脚本打印LS raw指标，未发现绑定主报告的采样归档。

Confidence: UNKNOWN。对应OPEN_QUESTIONS：Q05/Q22/Q23。

Your answer:

- [ ] Yes，可提供路径
- [ ] No
- [ ] Not sure

补充说明 / 本地文件路径：

## HR-11

报告April2026封面与五月Git归档的关系是什么？

Evidence suggests: 首次入库时间不等于开发时间，不能用它推翻封面。

Confidence: LOW。对应OPEN_QUESTIONS：Q28。

Your answer:

- [ ] 完成后延迟提交代码
- [ ] 封面是模板/未更新
- [ ] 报告后续又改过
- [ ] Not sure

补充说明 / 本地文件路径：

## 回答之后的处理

- 有具体文件线索：优先核对文件，再更新证据等级。
- 只有记忆：标记HUMAN_RECOLLECTION，与CONFIRMED分开。
- 记忆与文件冲突：保留双方，新增需要验证的解释，不强行消解。
- 本清单不授权修改历史代码、修报告或创建restart。
