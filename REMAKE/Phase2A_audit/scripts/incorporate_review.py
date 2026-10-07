from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parents[1]
source = Path('C:/Users/35789/Documents/Obsidian Vault/HUMAN_REVIEW_Answered.md')
raw = source.read_bytes()
snapshot = root / 'evidence/HUMAN_REVIEW_Answered_2026-10-06.md'
assert not snapshot.exists(), 'Do not overwrite a previous review snapshot'
snapshot.write_bytes(raw)
link = '[人工回答整合](HUMAN_REVIEW_INTEGRATION.md)'
updates = {
 'Q01': 'HR-01：作者说明当时尚未定稿；暂以 L∥(R+C) 称 latest experimental version，不指定 canonical。主图身份的数值证据不变。',
 'Q04': 'HR-04：workflow 意图是整理稳定、尚可的12参数 baseline；14参数仍为实验性。这是 HUMAN_RECOLLECTION，不能据此推定精确开发日期。',
 'Q05': 'HR-03 勾选 Yes，但补充为“很有可能”：支持报告混合12参数消融与14参数主图的解释，具体编排历史仍非确定。HR-10 未提供另一套14参数消融归档。',
 'Q07': 'HR-06：选择 exp_10 因端口有代表性、复杂度适中且惯性沿用；没有“数据质量最好”的依据，也不意味着其他接线等价。',
 'Q08': 'HR-05：nLls 小数量级来自有意的经验调整，解析依据未解决；不能把整个初始化链称为已由理论推导。',
 'Q09': 'HR-07：作者报告目前只有 SQLite 和接线表，无实验细节；未提供新路径。当前已知资料边界明确，原始仪器来源仍 UNKNOWN。',
 'Q10': 'HR-09：作者否认有意压力测试；应按30HP配置，实际因时间限制不严谨地套用默认值。代码中的显式选择与此动机不冲突，不构成有效跨功率验证。',
 'Q11': 'HR-01/02：拓扑当时未定稿，作者认可正文可能保留旧方案。旧结构真实存在的文件证据不变；不可把冲突直接删作笔误。',
 'Q14': 'HR-05：作者明确回忆有意把约1.7806e-2 H 调到1.7806e-10 H量级，以改善拟合。误录假说优先级下降；物理意义、解析公式与可辨识性仍开放。',
 'Q15': 'HR-08：虽勾选旋转/空载，补充明确是“推测”。归类 HUMAN_SPECULATION，不是已确认工况；转速、供电状态、滑差仍 UNKNOWN。',
 'Q22': 'HR-10：作者称应该未特别保存 trace/MAP/链诊断；未新增采样证据。重跑可作为新实验，但不能补成历史证据，本轮未执行。',
 'Q23': 'HR-10 没有补充后验归档；现有 mean 计算语义的代码判断不变，不能由人工回答验证历史后验预测。',
 'Q24': 'HR-01 提及接口/线材作用，但作者明确称模糊怀疑。记为 HUMAN_SPECULATION，不能据此把附加元件解释为已识别的夹具或线缆参数。',
 'Q26': 'HR-05 的经验调参动机不能解决参数物理可辨识性；HR-01 也明确附加结构尚缺理论支撑。',
 'Q28': 'HR-11：作者将报告定位为研究背景与主要任务参考，非终稿；报告后可能继续试验。精确修改/提交时间未新增独立证据。',
}

integration = '''# 人工回答整合 — 2026-10-06

来源：[用户提供的回答](</C:/Users/35789/Documents/Obsidian Vault/HUMAN_REVIEW_Answered.md>)；[逐字节归档副本](evidence/HUMAN_REVIEW_Answered_2026-10-06.md)。

本次只吸收回答并更新审计文档。回答中的后续实验建议、模板说明与当前执行请求分开处理；没有启动新实验、修复历史代码、修改报告或创建 restart。

## 证据分层

- CONFIRMED：此前文件核对与受控数值复现，结论保持不变。
- HUMAN_RECOLLECTION：作者对当时意图、过程和资料保存情况的回忆；与文件证据并列。
- HUMAN_SPECULATION：作者明确以“推测”“模糊的怀疑”表述的内容；不升格为历史事实。
- CURRENT_USER_POSITION：作者目前愿意采用的工作定位；不等于过去已经定稿，也不等于物理验证。

## 11项回答与处理结果

| 回答 | 内容与证据性质 | 对审计的影响 |
|---|---|---|
| HR-01 | 回忆当时未定稿；目前暂以 L∥(R+C) 为 latest；接口/线材来源只是怀疑 | 区分最新实验候选与最终模型；结构阶数和物理来源仍开放 |
| HR-02 | 认可报告可能保留早期方案文字 | 支持混版解释，不能据此断定每处文字的修改时间 |
| HR-03 | 勾选 Yes，补充“很有可能”；回忆14参数需频繁试探谐振 | 支持混阶段报告；“未作为最终结构写入”不否定其输出出现在主图 |
| HR-04 | 12参数稳定、尚可；workflow用于整理该基线 | 整理意图有作者回忆支持，不再仅凭文件布局推断 |
| HR-05 | 有意将 nLls 降到1e-10量级改善拟合，尚无解析解释 | 偶然误录假说降级；保留经验参数定位和物理解释问题 |
| HR-06 | exp_10代表性适中、便于综合检验，惯性沿用 | 不宣称质量优选、统计代表性已验证或各接线等价 |
| HR-07 | 当前只有SQLite与接线表，无实验细节 | 未新增外部文件线索；记录资料缺口，不宣称已穷尽所有存储位置 |
| HR-08 | 勾选旋转/空载，但解释为推测 | HUMAN_SPECULATION；测量工况保持 UNKNOWN |
| HR-09 | 并非压力测试；时间限制导致30HP套用 | 解释实现意图；原跨功率验证不足的判断保持 |
| HR-10 | 应该没有特别保存诊断；认为重跑一小时内可接受 | 没有新增历史采样证据；时长为作者估计，重跑仅列后续选项 |
| HR-11 | 报告后可能继续实验；报告不是终稿 | 日期不能作为单一版本冻结依据；确切时间线仍开放 |

## 更新后的项目定位

**12参数：稳定基线；14参数 L∥(R+C)：最新实验候选；报告：多阶段研究记录。** 前两项中的“稳定”和“最新”是作者给出的工作定位，不是独立验证或 canonical 判定。

主图与14参数确定性拟合的对应关系已由数值和像素验证；12参数消融只能支撑其自身实验范围。作者关于14参数未定稿的回答与这两项文件事实可以同时成立。

## 仍需技术验证的问题

1. 附加结构是否必要、应该几阶、是否与接口/线材有关，以及能否跨接线解释数据。
2. nLls 的定义、数量级、解析来源与可辨识性；拟合改善本身不是物理证明。
3. 同一拓扑、同一目标和可比计算预算下的优化/消融比较。
4. NUTS 的似然、参数支持域、链诊断与真正后验预测；重跑将是新实验。
5. 30HP适配初始化和独立验证；现有套用结果不能充当验证成功。
6. 原始测量工况和仪器记录仍缺失；不能凭通用测试印象推定滑差或旋转状态。

原始调查正文保留作历史记录，相关问题顶部已加本次修订注记。没有重跑文档生成器，以免覆盖人工回答及本轮增补。
'''
(root / 'HUMAN_REVIEW_INTEGRATION.md').write_text(integration, encoding='utf-8')

names = ['OPEN_QUESTIONS.md', 'MODEL_EVOLUTION.md', 'CURRENT_BEST_RECONSTRUCTION.md', 'HUMAN_REVIEW.md', 'PROJECT_AUDIT.md']
backups = root / 'evidence/pre_human_review_2026-10-06'
backups.mkdir(exist_ok=False)
for name in names:
    path = root / name
    (backups / name).write_bytes(path.read_bytes())
    body = path.read_text(encoding='utf-8')
    heading, rest = body.split('\n', 1)
    note = '\n\n> 2026-10-06 更新：已收到11项人工回答，见' + link + '。下文原调查文字保留；涉及人工意图与待回答状态时，以本次整合说明为准。\n'
    if name == 'OPEN_QUESTIONS.md':
        for q, update in updates.items():
            pattern = r'(^## ' + q + r' — [^\n]+\n)'
            body, count = re.subn(pattern, lambda m: m.group(1) + '\n> **人工复核更新（2026-10-06）**：' + update + '\n', body, count=1, flags=re.M)
            assert count == 1, q
        heading, rest = body.split('\n', 1)
    if name in ['MODEL_EVOLUTION.md', 'CURRENT_BEST_RECONSTRUCTION.md']:
        note += '\n**当前工作定位：** 12参数是作者认可的稳定基线；14参数 L∥(R+C) 暂称最新实验候选，尚未定稿。nLls 为有意经验调整，解析解释未闭合。工况回答属推测，保持 UNKNOWN。报告用于研究背景参考，不代表冻结版本。\n'
    if name == 'HUMAN_REVIEW.md':
        note += '\n答案原文已归档，原空白问卷保留，不需要重复填写。HR-08 的勾选与补充文字须合并理解，不能仅凭勾选确认工况。\n'
    if name == 'PROJECT_AUDIT.md':
        note += '\n本次操作：读取用户指定回答文件、保存字节一致副本与SHA-256，备份5份审计文档，新增整合表并为15个相关问题加入修订注记。仅写入审计目录；未运行研究脚本或修改历史项目。此前 final_validation.json 对应原审计时点；本次验证另存 human_review_validation.json。\n'
    path.write_text(heading + note + rest, encoding='utf-8')

assert source.read_bytes() == raw == snapshot.read_bytes()
assert len(re.findall(r'^## Q\d\d —', (root / 'OPEN_QUESTIONS.md').read_text(encoding='utf-8'), re.M)) == 28
record = {'date': '2026-10-06', 'source': str(source), 'snapshot': str(snapshot), 'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_and_snapshot_byte_equal': True, 'source_unchanged': True, 'question_count': 28, 'annotated_questions': list(updates), 'updated_documents': names, 'new_document': 'HUMAN_REVIEW_INTEGRATION.md', 'original_documents_backup': str(backups), 'research_experiments_run': False}
(root / 'evidence/human_review_validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(record, ensure_ascii=False, indent=2))
