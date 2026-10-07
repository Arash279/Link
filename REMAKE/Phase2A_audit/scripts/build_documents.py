from pathlib import Path
import json, csv, math
B=Path(__file__).resolve().parents[1]
R=Path(r'D:\Desktop\LinkCodex')
def link(rel,line=None,label=None):
    p=(R/rel).as_posix();return f'[{label or rel}](</{p}{":"+str(line) if line else ""}>)'
def local(rel,label=None):
    return f'[{label or rel}](</{(B/rel).as_posix()}>)'
def ext(path,line=None,label=None):
    return f'[{label or path}](</{Path(path).as_posix()}{":"+str(line) if line else ""}>)'
C=json.loads((B/'checks/controlled_results.json').read_text(encoding='utf-8'))
S=json.loads((B/'checks/supplemental_results.json').read_text(encoding='utf-8'))
E=local('checks/controlled_results.json','受控验证数值记录')
P=local('checks/supplemental_results.json','像素、AST、数据质量与历史拓扑记录')
G=local('evidence/git_history.txt','完整本地 Git 路径历史')
SUM=link('MD_documents/Project Summary; Wide-to-Ultra-High-Frequency Grey-Box Modeling of Motor Impedance.md',184,'研究总结中的14参数指标')
V3=link('try/try.py',147)
Q=[]
def q(title,why,evidence,ha,sa,ca,hb,sb,cb,best,confidence,verify,falsify):
    Q.append(dict(title=title,why=why,evidence=evidence,ha=ha,sa=sa,ca=ca,hb=hb,sb=sb,cb=cb,best=best,confidence=confidence,verify=verify,falsify=falsify))
q('EE5003 report 主结果究竟对应哪个模型？','决定未来复现对象，防止把数值相近的12参数结果误认成报告结果。',
  f'{V3} 的端口支路是 L∥(R+C)，参数数目14。{SUM} 明写 p=14。{E}：隔离重跑 SSE=54465351.850388095、RMSE=130.8719570241928、相对幅值 RMSE=0.2164369083981353；与外部历史残差恢复曲线的最大差为1.46e-14 Ω。{P}：重跑主图与报告图全部RGB像素一致。',
  '报告主数值来自 try 系列14参数并联串联RC模型。','模型、原始/相对指标、历史残差及整张主图共同吻合。','无法区分共用相同函数的 try.py、try2.py 和 NUTSVerTry.py 的LS阶段；没有原始执行清单。',
  '报告主数值来自12参数模型，14参数只是配图。','报告和日志都使用 CurVer 名称，12参数RMSE很接近。','12参数已有结果和14参数复现曲线、像素及相对指标不相同；研究笔记明确p=14。',
  '14参数模型家族能够精确重现主图，这是本轮验证事实；“当时具体运行 try.py”仍只是历史解释。不要指定canonical。','HIGH',
  '保留数据库和源码hash，查当时stdout、IDE历史或原始参数文件以锁定文件身份。复现脚本只调用复制文件的函数，未调用历史main。',
  '若找到绑定主图的原始执行记录，证明来自不同模型且能解释同一曲线，则必须修改文件级历史归属。')
q('fig_curver.png 如何生成？','图像文件名无法直接搜索到时，仍需区别截图与脚本导出。',
  f'{link("EE5003report__Copy_/main.tex",385)}引用 fig_Curver.png，磁盘文件小写。PNG元数据为 Software=Snipaste、User Comment=Screenshot。{link("try/try.py",731)}附近的plot_compare含同样标题、图例、坐标；{P}确认1800×1200 RGB逐像素相同。',
  '用try系列的plot_compare显示拟合结果，再以Snipaste截图保存/改名。','截图元数据、plot_compare只show而不以fig_curver命名保存，重跑图像完全吻合。','没有截图按键记录、原始窗口状态或文件改名记录。',
  '由另一个脚本复用同样绘图函数或复制已有图。','相同绘图实现存在于多个脚本；复制不会改变内容。','没有独立生成脚本命中最终文件名；该解释不改变已验证的数值来源。',
  '可确认图像像素可重现；最可能经历了显示→截图→报告引用，具体操作历史未确认。','HIGH',
  '人工确认截图方式，查本地其他源文件或IDE执行记录；无需再凭肉眼拟合曲线。',
  '带执行日志的独立图像生成器能证明是直接savefig或来自其他文件，而非截图流程。')
q('报告最终参数在哪里，是否已经恢复？','复现一条曲线与证明参数是历史原件、物理唯一解是不同问题。',
  f'报告参数表只列物理意义和方法，无完整最终数字。{link("try/try.py",870)}附近仅打印6位有效数字。现有JSON是12参数；{E}保存了本轮14参数全精度新拟合向量，逐点匹配历史残差；外部CSV未含参数列。',
  '历史完整14参数向量未显式归档，本轮已恢复一个与历史结果等效的向量。','当前范围中未找到14参数序列化最终文件；新向量重现历史曲线及主图。','仍可能有未纳入范围的stdout、IDE记录或截图参数。',
  '最终向量已保存在某个笔记或外部文件，只是尚未定位。','12参数stdout被复制到多篇笔记，存在类似保存习惯。','目前找到的完整14参数数值是本轮新输出，不能冒充历史原件。',
  '已恢复“可复现参数”，尚未证明是“历史原始参数”或唯一参数。见文末参数表。','HIGH',
  '用新向量复算历史曲线已经完成；后续只需查原始stdout并测试参数可辨识性。',
  '发现当时14参数完整输出，可将对应字段升级为历史参数；若多组参数给出同曲线，唯一性假设被否定。')
q('11、12、14参数与later workflow的真实演化关系？','目录名和提交日期不能单独决定最终版本。',
  f'{G}显示11参数电路在早期提交中，12参数CurVer于9cf0d55（2026-02-09）进入Git，消融在fc8ad7f（2026-03-28），try家族在44cd9ba（2026-05-27）。workflow当前未跟踪；{link("baseline1/workflow/model.py",3)}导入12参数CurVer。',
  '12参数benchmark与14参数拓扑探索并行，后来的workflow整理选择了12参数线。','导入关系及输出p=12直接支持；14参数数值和报告吻合。','workflow首次创建时间无Git记录；提交时间不等于实际开发时间。',
  'workflow早于14参数，或者代表研究者后来主动回退的最终决定。','没有带日期的决策记录，存在这种可能。','没有明确回退声明；workflow README只说当前clean workflow，不证明论文模型被否定。',
  '按分支图描述，不写成无分叉V1→V2→V3。later cleanup的时间/意图是MEDIUM假设。','MEDIUM',
  '查workflow创建记录、历史备份和人类意图；保留工作树与Git版本的差异。',
  '有早期workflow快照或明确回退决策，会推翻目前的先后/整理动机解释。')
q('报告消融实验是否与主模型严格一致？','决定性能比较能否作为14参数模型的直接验证。',
  f'{link("baseline1/socket_run_log.txt",22)}明确p=12；六个baseline1脚本参数表均12项。报告{link("EE5003report__Copy_/fig4_param_cv_heatmap.png")}可见12行，无Cad/Rad；主图由14参数模型重现。No-L代码11参数；旧baselines/NUTSVer是12参数，try/NUTSVerTry是14参数。',
  '报告组合了11参数结构对照、12参数算法消融和14参数主模型/NUTS展示。','代码、日志、热图行标签与主图数值共同支持阶段混用。','无所有图的原始导出清单；部分图的确切脚本仍未确认。',
  '另有完整14参数消融，热图仅故意省略两个参数。','原则上可选择性展示参数。','本地只有12参数60次运行日志，图中其余参数模式与这些统计相符；未找到14参数消融证据。',
  '混用阶段是最合理解释；至少不能将现有12参数消融自动宣传为14参数的严格消融。','HIGH',
  '找五幅消融图的原始绘图数据/脚本；按每张图记录参数数目、bounds、objective。',
  '找到与图绑定的14参数消融日志且证明同拓扑、同边界，才可替代当前解释。')
q('1.5×Z_ad 的来源和适用条件？','错误移植到对地或非对称配置会改变测量端口模型。',
  f'{link("try/try.py",208)}返回Z_core+Z_ad+0.5Z_ad；旧{link("baselines/CurVer.py",204)}注释说明BC短接。{ext("D:/Desktop/EE5003/data/README.md",89)}给出exp_10为A1–B1C1。未找到独立的历史完整推导。',
  '三相对称且B/C内端等势时，A侧串联一个Z_ad，B/C返回侧两个相同支路并联为Z_ad/2。','图中每相相同支路、BC端短接、代码系数均一致；解释符合串并联代数。','这是本轮条件性电路解释，不是找到的旧项目证明；测量三相并不数值相同。',
  '1.5是经验缩放，注释只是事后解释。','缺历史推导且参数本身可吸收系数。','结构与接线确实自然产生1+1/2，未找到经验调系数记录。',
  '对称端口化简最合理，但必须附“相同支路、BC对称等势、测量边界一致”的条件。','MEDIUM',
  '后续对原三相电路做独立节点分析，分别验证exp_10和对地配置；不要把本轮推导归因于旧笔记。',
  '完整节点分析表明BC内端不等势或存在绕过支路的电流，或历史说明系数为经验拟合，会推翻该化简解释。')
q('为何选exp_10，exp_11/12是否等价？','决定代表性、相对称假设和训练/验证边界。',
  f'{link("docs/EE5003.md",51)}讨论先单相后混合配置，并明确否定简单1–9训练/10–21测试划分。{link("Parameter_Fitting/fr_fa.py",10)}将10/11/12标记训练/定参。{E}：11/12与10同频率网格，但相对复阻抗RMS差约20.74%/12.97%（逐频点按|Z10|归一）。',
  'exp_10是三相对称假设下的代表端口，参数提取综合多组，最终拟合选单组。','接线表、参数提取列表、main默认值及旧笔记一致。','为什么恰好选第10而非11/12，没有明确选择准则。',
  'exp_10数据质量最佳，因而被筛选。','单表拟合和反复使用与这种策略相容。','无测量噪声、重复性或质量排名证据；差异不能直接解释为谁更好。',
  '代表案例解释较强，质量优选解释证据不足；三相理论对称不等于实测相同。','MEDIUM',
  '让研究者确认选择理由；后续做按频段的相间差异与重复测量分析。',
  '找到当时质量筛选表或相位故障记录，可改为有意质量筛选；物理不对称证据会限制等价假设。')
q('参数初始化、固定值、优化变量和bounds如何衔接？','初始化的来源决定“解析引导”究竟指自动数据链还是手工固定参数盒。',
  f'{link("try/try.py",308)}返回14个常数，main不调用Parameter_Fitting；{link("baseline1/workflow/fitting.py",31)}重新提取数据后构造p0；{link("baseline1/workflow/initial_params.py",93)}计算eta但不传入nLls；Rs是模块固定常数。完整参数表见本文附录A。',
  '报告脚本使用先前解析工作得到/调整的常数，workflow才将部分提取流程自动接入。','调用链、固定常数、笔记和当前stage1 JSON支持。','常数逐个来自哪次提取尚不全；nLls、Csf0有明显口径差异。',
  '报告运行时自动提取了全部参数，只是代码后来被替换。','报告文字描述完整hierarchical workflow。','现存可精确复现主图的main不自动提取；没有当时自动链记录。',
  '将报告脚本称为“固定先验及解析来源的初值/边界”，不要称每次运行全自动测量提取。','HIGH',
  '逐参数对照tools/Calculate.py、早期输出及workflow保存输入；详见附录A/B。',
  '发现当时完整自动执行链且数值与固定常数一致，可补充其上游来源，但不改变现存脚本行为。')
q('SQLite是否原始仪器数据？','决定restart的数据层级和物理可信度。',
  f'{ext("D:/Desktop/EE5003/data/README.md",43)}给出列和接线；SQLite只有Freq/Zabs/Phase，缺仪器和采集元数据。tools/Export_CSV.py是数据库向CSV导出，不是导入。项目、研究笔记、组会文本及Git文件历史中未定位原始仪器文件/建库脚本。',
  'SQLite是本次范围内最早可用测量数据层。','代码真正读取它；三个电机数据库存在且可查询。','这是“当前可用”的结论，不是采集历史起点。',
  'SQLite就是原始仪器无损导出。','列名与阻抗仪常用量相容。','没有导出文件、校准、日期或转换脚本证明无损；不能凭列名断言。',
  'SQLite database is the earliest available project data within the inspected scope；不标raw instrument data。','HIGH',
  '人工提供仪器导出位置/采集记录线索，再核对转换和校准；不向网络上传数据。',
  '找到更早Touchstone/CSV/仪器工程文件后，应更新最早可用层级与来源链。')
q('30 HP/其他电机验证是否有效？','避免把一次有输出的运行当作跨电机有效性。',
  f'{link("baseline1/hp30_workflow/fitting.py",90)}明确fit_inputs=DEFAULT_INITIAL_INPUTS，并打印stage1仅供诊断；该输入hp=1.5。stage1漏感多项NO_WINDOW、Lm为NO_BAND、Rrs非有限值失败；{link("baseline1/hp30_workflow/outputs/stage2_fit_params_exp_10_seed0.json")}仍使用1.5HP默认输入，RMSE约68424.99。',
  '这是刻意沿用主配置的试跑/诊断分支，尚非通过验证的跨电机模型。','日志打印说明是显式选择，并非仅隐藏继承；失败提取和巨大误差可见。','README称只换数据库，与实际诊断失败/输入选择的含义没有解释清楚。',
  '这是完成的跨电机验证，误差只反映尺度不同。','确实读取AP_30数据且执行拟合，绝对误差跨电机不宜直接比较。','未给相对指标和验证门槛；关键物理先验未适配，多参数触边。',
  '确认执行过跨数据集试跑；不能确认科学意义的跨电机有效性。是否属于bug或有意压力测试需人工确认。','HIGH',
  '确认试跑意图；以后再评估独立的30HP先验和相对指标，本阶段不修复。',
  '若有独立30HP输入、重复运行及通过的验证指标，可改评，但现有这组结果身份不变。')
q('拓扑冲突是笔误、旧版本残留还是图/代码后改？','这是模型身份最直接的反证来源。',
  f'{link("EE5003report__Copy_/main.tex",326)}写R+(L∥C)，图和try代码是L∥(R+C)。关键新增事实：{link("tmp/tmp.py",214)}及tmp1–3确实实现前者；Git fc8ad7f同样如此，而ee0a7f1的tmp另有端口并联Cad。{P}保存历史函数。',
  '报告正文残留了真实存在的R+(L∥C)试验表述，主图后来采用L∥(R+C)。','两版均有代码，不必假想；主图精确匹配后一版。','没有正文修改记录可证明复制残留；也可能独立措辞错误。',
  '正文只是纯文字笔误，与旧tmp版本无关。','电路图、总结笔记、主图一致支持后一版。','存在完全对应正文的历史代码，不能忽略它；缺作者说明。',
  '不能只说“笔误”；最合理是跨版本文字未同步，具体原因保留MEDIUM。数值使用后一拓扑的可复现证据为HIGH。','MEDIUM',
  '查报告草稿修订/人工确认，比较历史拓扑创建与图替换记录。',
  '作者可定位独立措辞错误或找到正文版本对应的正式结果，将改变原因归属；不能改变已观察的两版代码。')
q('try.py、try2.py、NUTSVerTry.py 是否同一个deterministic model？','决定能否通过数值识别脚本，及NUTS是否比较了相同电路。',
  f'{P}：16个关键函数AST相同，包括电路、初值、bounds、DE+LS、weights/residual。{E}：4组边界内随机参数在1590频点的NumPy输出差均为0。NUTS内部实/虚代数以NumPy解释的最优点差2.77e-11 Ω。try2只改评价为相对幅值。',
  '三者NumPy确定性核心相同，只是评价/推断阶段不同。','AST与数值检查相互支持。','未用真正PyTensor编译器进行全域检验；含epsilon的内部代数不能称逐位相同。',
  '三者电路有细小差异导致不同主图。','NUTS另写了一份实虚运算，理论上可能分歧。','已检查的NumPy核心相同，有限样本验证不支持此说。',
  '确定性NumPy模型相同已确认；PyTensor实现是数值相容的候选，未验证采样器整体。','HIGH',
  '后续如需贝叶斯复现，再做编译张量与NumPy的随机参数全频比较；当前无需重跑昂贵NUTS。',
  '实际PyTensor后端在有效参数内产生不可忽略差异，将推翻其实现相容性，但不影响三份NumPy函数相同的事实。')
q('是否还有V1/V2/V3之外的模型？','参数数目相同并不代表拓扑或端口相同。',
  f'{local("evidence/code_inventory.json")}列32个模型/相关脚本；tmp系列14参数使用另一拓扑；ee0a7f1的tmp是Z_meas∥Cad；单相exp_1和三相exp_10都可有11参数；Git早期还有已删除的脚本与MATLAB文件。',
  '真实历史是多个拓扑与端口分支的探索树。','Z_total函数不同、接线不同及Git历史直接支持。','不能断言这些版本都被实际运行或全部进入报告。',
  '它们只是相同模型的重命名/备份。','大量函数重复，少数文件确实hash相同。','端口并联Cad与每相附加谐振单元的行为不同，不能统一。',
  '用“端口+拓扑+参数化+目标函数+输入”标识模型，V编号仅是调查标签。','HIGH',
  '逐个保存行为差异表与Git blob，不从文件名推断弃用；MODEL_EVOLUTION已列分支。',
  '独立代数证明某两版在所有有效条件等价，可合并那两版；不能由参数数目直接合并。')
q('nLls 与 eta*Lls 的八个数量级差异意味着什么？','直接影响物理解释与“解析初始化”真实性。',
  f'{link("docs/EE5003.md",11)}列etaLls=1.7806e-2 H；{link("try/try.py",318)}nLls=1.7806e-10 H。代码直接使用jω*nLls，不是无量纲系数乘Lls。{link("baseline1/workflow/initial_params.py",93)}明确不使用eta公式。1月12日组会文本已讨论eta问题。',
  'nLls已变成独立小电感的经验/数值自由度，不能再等同原论文eta*Lls。','代码单位、显式不映射注释、巨大差异一致。','旧图/报告仍沿用首匝漏感物理名称。',
  '最初只是指数录入错误，后来被沿用。','1.7806尾数保留而指数相差8位，非常可疑。','没有修改动机记录；组会确实讨论了原公式不适用，可能有意缩小。',
  '当前实现是独立H单位参数；为何变小仍需人工复核，不能自动修成0.017806。','MEDIUM',
  '查最早修改该指数的Git差异和手稿；以后用敏感性/节点分析检验物理映射。',
  '若找到明确的单位换算或误录修正记录，可解释起因；若完整推导证明新参数另有尺度，也可消除混淆。')
q('Rs=8.703 Ω 的来源、转速/滑差假设可靠吗？','固定定子电阻影响低频和转子支路解释。',
  f'{link("Parameter_Fitting/fit_rs.py",4)}注释称数据为no-load/turning，使用exp7–9的100–500Hz Re(Z)线性外推。{E}本轮按相同操作得到8.7028583428 Ω，四舍五入即8.703。旧笔记明确仍需DC/热态试验确认。无独立运行状态记录。',
  'Rs常数来自这一低频外推，运行状态只是当时建模假设。','数字匹配，笔记承认来源受限。','没有该脚本原始stdout，也没原始测量工况证明。',
  'Rs来自独立电阻测试，外推只是复核。','仍可能存在未归档实验。','未找到测量记录；数值精确吻合外推更支持A。',
  'Rs的计算来源已高度可解释，但不应称为独立测量的DC电阻；运行状态UNKNOWN。','HIGH',
  '人工确认电机是否静止/旋转、供电/接地状态；定位直流电阻测量记录。',
  '找到明确仪器电阻值/温度/工况记录可改写来源；不能由脚本注释证实实际转速。')
q('频带、抽样、插值、排序是否一致？','改变频点可使曲线相近而指标不同。',
  f'{link("try/try.py",223)}默认0<f<=1e8并排序；原exp10为1601点100–110MHz，筛后1590点最大99967334.06Hz。N_SAMPLES=2000，因此抽样函数直接返回全部点；N_PLOT=4000只用于绘图模型网格。{E}记录实际频点。',
  '报告拟合实际使用全部1590测量点，所谓log_uniform未在该案例下采样。','样本阈值和实际数量直接决定代码分支。','报告泛述准均匀log抽样，容易误读为重新插值测量。',
  '报告主结果先插值为2000点再拟合。','配置写N_SAMPLES=2000。','实际返回逻辑、n=3180和复现结果反驳。',
  '未发现主拟合测量插值；图上4000点是模型求值，不是新增观测。','HIGH',
  '后续固定SQL行、频率mask和排序hash，不能仅写100MHz；其他实验单独审核。',
  '另一个图/指标运行清单证明插值或不同mask，只能改变该结果的归属，不能改变现存主图复现链。')
q('重复频率及坏数据会不会污染提取/权重？','np.gradient对重复自变量可能出现除零；跨配置不能盲用相同流程。',
  f'{P}检查3库63表：未见非有限值、非正Freq/Zabs或逆序，但32表各有7个重复频率额外行。主AP_1p5/exp10无重复。{link("Parameter_Fitting/fr_fa.py",215)}和CurVer权重使用gradient；workflow/data排序但不去重。',
  '部分跨配置运行存在梯度/提取数值风险，主exp10本次不受此重复问题影响。','数据库重复与代码未去重直接可见。','重复点是否进入特定计算窗口尚未逐表追踪，不能把全部失败归因于它。',
  '重复点完全位于忽略频段，不影响实际流程。','提取有局部窗口，不是所有点都参与梯度。','仍需逐窗口证据；通用权重函数没有防护。',
  '记录为已确认的数据特征和待验证的影响；本轮没有去重或改数据库。','MEDIUM',
  '输出每个重复频率及参与窗口，隔离测试梯度是否有限，再由研究者选择保留/聚合策略。',
  '证明所有重复均在掩码外且未进入数值运算，可排除特定流程风险；否则需要显式处理。')
q('Hz/rad/s、度/弧度、相位wrap和误差单位是否正确？','避免把特征频率或残差尺度混成物理改善。',
  f'{link("try/try.py",62)}使用deg2rad复数重建；simulate_complex用2πf；权重相位先angle再unwrap，单位rad；phase_diff输出deg并wrap到[-180,180)。Delta解析公式用4πfr。曲线纵轴实际log10(|Z|)，标签带Ohm。',
  '主复现链单位转换一致，仍需区别权重的弧度梯度和图/相位残差的度。','代码与精确复现主图支持，Csf/Llr提取亦显式deg2rad和2πf。','不能因此确认所有历史公式/拓扑换算物理正确；文档wrap区间注释与实现边界不完全一致。',
  '主要失配来自全局Hz/rad或度/rad误用。','历史公式多、Delta双倍频率容易误解。','已追踪主链没有这种全局错用证据；不应凭UHF失配猜测。',
  '主链无已证实单位转换错误；eta/nLls、Delta适用关系另列，不能泛化“全部正确”。','MEDIUM',
  '后续以已知R/L/C单元和端口化简核对单位；补写log幅值及相位残差定义。',
  '找到被主链实际调用而省略deg2rad/2π的分支，或独立端口推导否定Delta换算，会推翻对应判断。')
q('analytic initialization是否真的作为DE/LS起点？','决定算法贡献的措辞与消融设计。',
  f'{link("try/try.py",429)}计算u0但完整DE流程未将u0加入候选；DE在由p0缩放的bounds全盒搜索，再加120随机点中top10做LS。LS-only从u0直接出发。bounds本身依赖p0。',
  '解析/手工p0主要通过bounds和鲁棒f_scale影响完整算法，而非直接作为LS起点。','调用和候选构造明确如此。','报告称“initialized around analytical estimates”，可作宽泛描述而非严格起点声明。',
  'DE population围绕p0局部集中初始化。','bounds围绕p0数量级；文字有这一印象。','未传init以集中p0；V3多项bounds跨400或10000倍。',
  '应称先验限定搜索范围；不能把完整方法与LS-only的起点安排说成完全相同。','HIGH',
  '以后若做初始化消融，须独立改变bounds、起点及f_scale，避免一次改变多个因素。',
  '找到当时不同DE调用并显式传入p0邻域population，可修订历史版本的起点说法。')
q('消融是否一次只移除一个因素，算法预算是否公平？','防止把实现差别全部归因于DE优于GA或全局搜索必要。',
  f'{link("baseline1/ablation_ls_only.py",431)}单起点；{link("baseline1/ablation_ga_ls.py",506)}GA后只做一次LS；CurVer为DE候选加top10随机候选；{link("baseline1/ablation_de_only.py",760)}DE-only预算160×20，完整60×10；No-Scaling实际去Re/Im缩放而非取消log参数化。',
  '现有比较是实用配置比较，非严格单因素、等预算消融。','起点数、全局预算、缩放标签的行为差异均确认。','六个baseline1版本拓扑/多数初值bounds仍相同，不能说比较完全无意义。',
  '所有比较严格只改变图例对应因素。','报告描述layer-by-layer，配置字段名称相似。','实际执行路径反驳：N_STARTS/TOP_K字段在某些分支未用。',
  '保留历史结果但限定结论，不以此证明算法的一般优劣。','HIGH',
  '记录每个方法真实model_eval、LS起点数、损失和预算；后续若获授权再设计匹配预算比较。',
  '找到另一套与图绑定的等预算单因素脚本，可改变图的解释；不能改变已存脚本差异。')
q('RMSE、AIC/BIC与优化cost是什么关系？','不同目标数值不可直接比较，信息准则不能脱离其假设。',
  f'{link("try/try.py",528)}raw SSE=sum(Re误差²+Im误差²)，n=2N，RMSE=sqrt(SSE/(2N))；不是sqrt(mean(|误差|²))。AIC=n ln(SSE/n)+2p，BIC=n ln(SSE/n)+p ln(n)。DE用加权缩放平方和，LS用soft_l1，try2相对指标只看幅值。',
  '报告raw指标是共同描述性评价，与用于选择解的鲁棒cost不同。','代码明确两套计算；相对SSE=74.483447/1590对应0.2164369。','报告用AIC/BIC宣称结构统计合理性，但拟合并非该raw同方差高斯似然的MLE，残差相关性也未证明。',
  'best_cost较小就等于raw RMSE较小，AIC/BIC证明新结构优越。','同一目标下cost一般能反映拟合变化。','No-Weights均值raw更低；12参数保存AIC低于14参数代表值；跨目标比较不成立。',
  '保留原计算定义与n=2N，AIC/BIC暂作同口径描述性数字；不能据此证明14参数优于12参数。','HIGH',
  '后续统一raw/relative/robust指标和样本数；验证噪声模型后再讨论信息准则解释。',
  '如证明各模型均在同一raw似然下达到MLE且同数据/假设成立，信息准则解释可增强。')
q('NUTS是否真正在同目标上证明LS近最优？','报告对剩余误差归因依赖这一论证。',
  f'{link("try/NUTSVerTry.py",555)}从LS解重建bounds，非原初值bounds；Uniform prior在log空间。{link("try/NUTSVerTry.py",668)}Normal(sigma=1)似然，不是soft_l1；{link("try/NUTSVerTry.py",1137)}raw评价仍用p_opt(LS)，GP亦如此；curver/nutsver残差CSV字节hash相同。',
  '相同raw数字/残差来自同一LS向量，不能作为MAP/NUTS不改善的独立证据。','变量流和CSV同hash直接解释“完全相同”。','图里可能真的显示LS/MAP接近；这一事实需单独读取或重算MAP参数。',
  '采样独立证实相同最优值且消除了优化问题。','报告写MAP/LS接近，代码确实调用sample和find_MAP。','没有保存trace、Rhat/ESS/divergence证据；目标、prior支持域不同；打印的数字不是MAP。',
  'Bayesian检查的确定性模型基本相容，但现有量化论证不成立；不能称已排除优化不足。','HIGH',
  '找保存的MAP/trace；分别计算LS、MAP、样本预测的raw及相对误差，检查链诊断。',
  '存在独立MAP指标与合格采样诊断可支持更窄结论，但不能让原脚本打印LS指标变成MAP指标。')
q('图中的posterior mean是否真是后验平均预测？','非线性映射下不同平均方式会改变平滑/不确定性解释。',
  f'{link("try/NUTSVerTry.py",681)}先求E[u]，再p_mean=exp(E[u])，最后Z(p_mean)；不是E[p]，也不是E[Z(p)]。图例为Mean，报告解释为posterior averaging。',
  '这是log参数均值处的插件预测，标签/解释过宽。','返回值与后续simulate_on_freq调用直接证明。','可以作为代表参数曲线，只是不能据此宣称平均预测平滑。',
  '它近似后验平均预测，差异可忽略。','如果后验很窄且模型局部近线性，可能接近。','未保存样本统计以检验该近似，弱辨识参数恰可能不满足。',
  '应描述为Z(exp(E[log p]))；近似程度UNKNOWN。','HIGH',
  '后续用trace逐样本计算Z并平均，比较插件预测和区间。',
  '若样本预测平均与插件预测足够接近，可接受数值近似，但数学身份仍不同。')
q('GP残差能否判定结构不足、随机性或物理来源？','防止将诊断工具输出升级为因果结论。',
  f'{link("try/try.py",591)}Re/Im为data−sim，phase为sim−data；先按Re/Im robust zscore<=5筛点，再拟合RBF+WhiteKernel，phase也沿用同mask。历史log有核length_scale触下界警告。报告{link("EE5003report__Copy_/main.tex",401)}与405对系统性/随机性表述不一致。',
  'GP只能定位剩余频段结构，不能区分夹具、漏建动态、优化不足或采集噪声。','没有干预或重复测量；mask和核约束也会塑造结果。','结构峰确实重复出现，具有诊断价值。',
  'GP峰证明电路结构错误，UHF尾部则证明随机噪声。','报告如此解释，峰位与异常频段吻合。','同一诊断不足以证明两种因果归属；目标不同、mask及kernel警告未处理。',
  '保留频段定位，物理/随机归因属于LOW证据解释；不可自动作为结构真假的裁决。','HIGH',
  '对残差定义、剔除点、核超参做敏感性记录；以后用重复采集/边界干预验证来源。',
  '独立测量干预能分离夹具/机器效应时，物理解释可增强，GP自身仍不是因果实验。')
q('是否存在独立验证集，泛化结论覆盖到哪里？','拟合误差、随机种子稳定与外推/跨配置验证不能混用。',
  f'{link("try/try.py",779)}DO_VAL=False；workflow validation在同一exp_10和相同频带重算；{link("docs/EE5003.md",51)}曾提出训练/测试划分又否定。提取阶段已用多个所谓check配置。',
  '主报告证据主要是同数据拟合和优化稳定性，不是独立泛化验证。','执行开关、读取表和已有输出支持。','不同配置图与特征提取提供辅助一致性线索，但不是完全隔离测试。',
  '10–21构成独立测试集或GP就是验证集。','早期笔记提到这个设想，文件名含validation/check。','笔记自己否定隔离；参数提取用到了这些配置，GP同样基于拟合残差。',
  '不把文件名validation翻译成已通过独立验证；报告自身也要求更多电机/边界验证。','HIGH',
  '建立每个表对提取、拟合、选择、展示的使用矩阵后再划验证集。',
  '找到未参与提取/选择的独立测量及预先确定的验证结果，可补充泛化证据。')
q('触边、稳定性与可辨识性如何影响参数物理解释？','低RMSE或低跨seed CV不意味着唯一物理参数。',
  f'{E}14参数复现Rrs≈2800到上界、Rsf≈27.4到下界；12参数日志同样触边。热图显示Csf0/nLls高CV。不同12参数解的弱参数变化显著而RMSE近似。',
  '部分参数受边界/耦合约束，应该按等效拟合量解释。','触边与CV是直接证据；nLls量级偏离原公式。','未做profile likelihood或灵敏度矩阵，不能断言所有参数不可辨识。',
  '跨seed稳定足以确认物理值准确。','大多数参数CV很低，主曲线可重复。','边界固定会制造稳定，数值稳定与物理真实性不等价。',
  '先标Rrs/Rsf为边界敏感，Csf0/nLls为弱辨识候选；不自动修值或改变bounds。','HIGH',
  '后续做边界扩展、剖面和多配置联合约束；需与历史baseline分开保存。',
  '独立测量或局部/全局可辨识分析证实唯一值且不依赖边界，可增强物理解释。')
q('输出覆盖、旧CSV和当前代码能否混读？','关系到每项结果的身份与长期可追溯性。',
  f'{link("try/NoLad.py",947)}和try主模型都可写同一D:/Desktop/tmp/curver_gp_residual.csv；多脚本也写相同相对PNG名。报告/MD/workflow未跟踪，CurVer已改initialization；输出JSON不记录源码/database hash。当前12参数JSON重新求值与其指标相同。',
  '外部文件是最后一次写入的快照，文件名不足以证明整个历史。','共享路径和未版本化输出明确存在；主残差已靠数值匹配确认当前内容。','没有证据证明所有CSV均过时或被错误覆盖，不能一概否定。',
  '同名文件始终对应同一模型，所以可直接复用。','命名CurVer容易产生这种理解。','NoLad也写curver文件，直接反驳。',
  '以当前内容hash+曲线匹配识别；本轮检查未证明整个输出目录新鲜。','HIGH',
  '保存本轮manifest及副本；每个重要输出绑定脚本、数据、参数、设置。',
  '找到不可变run manifest可恢复特定历史身份，但不会消除现有覆盖风险。')
q('运行时间、方法优势与报告日期能否当作可靠历史？','防止用不同计时口径、单个异常运行或提交时间编排故事。',
  f'{link("EE5003report__Copy_/main.tex",377)}表为8.399s，正文7.996s；表global/local=1.880/6.321。历史summary的CurVer均值20.8039s，外层duration含GP等；GA均值111.7701s而中位数1.7375s，最大1101.318s。报告封面April2026，14参数首次可见提交May27。',
  '报告整合多次运行/阶段，GA平均耗时被异常长一次强烈影响；提交时间只能证明当时已入库。','表、日志统计和Git事件支持。','无法判断异常时间原因，也不能据此认定封面日期虚假。',
  '所有数字是同一次同环境，GA必然比DE慢很多；May提交即May开发。','报告叙述有这种统一印象。','计时口径和统计分布直接不同；提交可晚于开发/报告。',
  '将时间视为具体run的记录，方法一般优劣和真实提交/完成日期暂不确认。','HIGH',
  '人工确认封面与归档时序；将fit-only/global/local/GP/NUTS/process时间分开。',
  '原始run manifest及资源监控可解释异常，并可能支持更严格的耗时比较；不能靠均值替代。')

header='''# Phase 2A — Open Questions Register

日期：2026-10-05（Asia/Singapore）。用途：人类复核前的研究调查记录；不是restart的正式技术叙事，不指定canonical。

## Material Passport / 范围与证据约束

- 历史项目：`D:\\Desktop\\LinkCodex`；只读。
- 依据代码路径向外追踪：`D:\\Desktop\\EE5003\\data`；以及`D:\\Desktop\\tmp`内被模型引用的残差、关联统计及报告草稿候选。未遍历整盘或个人其他项目。
- 新文件仅写入当前工作区`Phase2A_audit`；没有创建`D:\\Desktop\\LinkCodex_restart`。
- FACT/CONFIRMED：文件可观察内容或本轮记录的计算结果。INFERENCE：证据支持的解释。SPECULATION：缺直接支持的可能性。
- HIGH/MEDIUM/LOW是定性置信度，不是概率。每题即使某些事实闭合，历史解释仍保留工作假设和反证条件。
- 本地报告/笔记可以证明作者写过什么，不能自动证明物理结论。Git首次出现只是入库时间上界。未找到不等于不存在。
- 对3个电机63张表做基本数据检查，对32个模型/相关脚本做AST索引；PPTX文字/notes与4份PDF文本已提取。未完成全部图表、参考论文公式和全部Git blob的穷尽数学审查。
- 主图的确定性重跑成功；没有重跑NUTS采样、全部60次消融或30HP拟合。

## 本轮比Phase 1新增/修正

1. 找到外部历史残差，可恢复历史主曲线；14参数隔离重跑逐点吻合，报告主图RGB像素完全吻合。
2. 正文的另一拓扑确实存在于tmp系列及Git中，不能简单归为凭空文字笔误。
3. Rs=8.703的数值可由exp7–9低频外推恢复，但工况仍无独立证明。
4. 30HP明确选择默认1.5HP输入，不能只说是隐式配置继承bug。
5. NUTS的raw数字/残差实际仍使用LS参数；posterior mean实际为log参数均值处预测。
6. 主exp10无重复频率，但其他多配置有重复；不能假设整个数据库满足gradient条件。
7. Phase 1“未发现MATLAB”只指当前工作树；Git历史含已删除figure1.m/figure2.m，未据此认定是当前模型实现。

## 问题索引

'''
for i,x in enumerate(Q,1):header+=f'- Q{i:02d} — {x["title"]}\n'
text=header
for i,x in enumerate(Q,1):
    text+=f'''\n## Q{i:02d} — {x['title']}

### Why it matters

{x['why']}

### Confirmed evidence

{x['evidence']}

### Candidate hypothesis A

{x['ha']}

Supporting evidence:
- {x['sa']}

Conflicting evidence:
- {x['ca']}

### Candidate hypothesis B

{x['hb']}

Supporting evidence:
- {x['sb']}

Conflicting evidence:
- {x['cb']}

### Current best interpretation / Best current answer

{x['best']}

Confidence: **{x['confidence']} confidence**.

**This is a working hypothesis, not a confirmed conclusion.** 此句限定历史/物理解释；Confirmed evidence中的文件和验证事实不因此降为猜测。

### How to verify

{x['verify']}

### What would falsify this hypothesis

{x['falsify']}
'''

meaning={'Lls':'定子漏感','Csw':'绕组/匝间等效电容','Rsw':'绕组高频损耗','Llr':'转子漏感','Rrs':'Rr/s等效项（工况待证）','Rcore':'铁损等效电阻','Lm':'励磁电感','nLls':'独立首段小电感；不等同已证etaLls','Csf':'高频绕组-机壳等效电容','Rsf':'机壳路径损耗','Csf0':'中性点/地等效电容','Lad':'端口附加电感','Cad_lad':'与Rad串联后并联Lad的电容','Rad_lad':'附加RC支路阻尼'}
sources={'Lls':'历史常数；workflow多配置L_sigma/2','Llr':'历史常数；workflow多配置L_sigma/2','Csw':'历史常数；workflow由fr/CsfHF/Lls解析','Rsw':'历史常数；workflow由Zmax/Rcore/Llr解析','Rrs':'历史低频拟合28；workflow fit_rr或回退','Rcore':'6300 hp^(-0.6958)初值，随后优化','Lm':'历史估计0.055；workflow Lm提取或回退','nLls':'手工小初值，eta公式未注入','Csf':'历史CsfHF；workflow单相平台中位数','Rsf':'(2/3)Zanti初值，随后优化','Csf0':'历史固定7.38e-10；workflow LF−3HF','Lad':'手工1.3e-7','Cad_lad':'手工1e-11','Rad_lad':'手工120'}
rows=[]
for k,v in C['rerun']['at_bounds'].items():
    unit='Ω' if k.startswith('R') else ('F' if k.startswith('C') else 'H')
    rows.append([k,meaning[k],unit,sources[k],'见来源；V3运行时不提取','否','是',f"{v['initial']:.10g}",f"[{v['lo']:.10g}, {v['hi']:.10g}]",f"{v['value']:.12g}", '触边' if v['near_bound'] else '未触边'])
with (B/'checks/parameter_register_v3.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['Parameter','meaning','unit','source','extracted','fixed','optimized','initial','bounds','recovered_value_not_original','boundary']);w.writerows(rows)
text+='''
## 附录A — 参数表（以可复现主图的14参数版本为主）

下表数值是**2026-10-05本轮新恢复参数**，不是找到的历史原件，也不宣称物理唯一性。所有L以H、C以F、R以Ω；`nLls`是H单位。`fixed`指拟合期间是否固定，而不是初始值是否硬编码。

| Parameter | physical meaning | 单位 | source | extracted? | fixed? | optimized? | initial | bounds | recovered/report-compatible value | 边界 |
|---|---|---|---|---|---|---|---:|---|---:|---|
'''
for row in rows:text+='| '+' | '.join(row)+' |\n'
text+='| Rs | 定子串联电阻 | Ω | exp7–9低频Re外推与常数吻合 | 独立脚本估计，主拟合不重新提取 | 是 | 否 | 8.703 | 不适用 | 8.703 | 固定 |\n'
text+=f'''
证据：{link('try/try.py',308)}的p0、327起bounds、114的Rs；新结果见{E}。DE/LS在log参数空间内优化上述全部14项；不要把“解析初值”误写成“拟合中固定的解析参数”。

### 同一参数在不同版本的状态

| 版本 | 拟合变量 | 初值来源 | bounds | 固定项 / 特殊项 |
|---|---|---|---|---|
| 11参数exp10/No-L | Lls至Csf0的11项 | 多为内嵌常数；Anly_Meth另有解析路径 | exp_10_fit/CurVer_no_L：一般0.1–10×，R项0.01–100×；Anly_Meth另查 | Rs固定，无Lad/Cad_lad/Rad_lad |
| 历史12参数CurVer与六组消融 | 前11项+Lad | 内嵌p0 | 一般0.1–10×p0；Rsw/Rrs/Rcore/Rsf为0.01–100× | Rs固定；无Cad/Rad |
| 当前12参数workflow | 同上12项 | stage1提取+initial_params公式+回退/先验 | 沿用12参数倍率，但p0变了，绝对边界亦变 | nLls=1.8e-10初值，Lad=1.3e-7初值；都参与优化 |
| 14参数try家族 | 上表14项 | 内嵌常数 | L与C为0.05–20×；R为0.01–100× | Rs固定 |
| 14参数tmp另一拓扑 | Cad/Rad与前12项 | 脚本常数 | 存在Cad专用边界等差异，不能共用try边界 | 附加网络为R+(L∥C)，不同模型 |
| HP30 workflow | 同12参数 | 明确强制DEFAULT_INITIAL_INPUTS，stage1仅诊断 | 围绕1.5HP默认p0 | hp仍1.5，Rs仍8.703 |

### 提取到初值的实际路径

- `exp1–6` → Csf平台；`exp10–12/14–16/18–20` → 漏感候选及质量筛选 → Lls=Llr=L_sigma/2。
- `exp10–12` → fr/fa、峰谷中位数；低频100–500Hz结合漏感估Rrs；Lm优先10–12，必要时参考18–20。
- `hp`、connection、nLls、Lad来自配置；Rs不在主workflow自动提取链内。
- Csw、Rsw、Rsf、Csf0、Rcore经initial_params生成；eta公式被计算后丢弃，不赋给nLls。
- 特征缺失时`_prefer`回退历史默认；回退不是成功测量提取。
- `run_fit`重新调用提取并写stage1输出，不保证使用先前手工检查过的stage1 JSON。

## 附录B — 初始化口径的具体差异

| 输入/参数 | 历史笔记 / 14参数主图初值 | 当前workflow或Calculate |
|---|---|---|
| fr / fa | 笔记27.49k / 76.03k Hz | Calculate和DEFAULT为36311.2 / 66734.6，注释exp_1；当前stage1为27492.377 / 76030.726 |
| Csf0 | 主图p0=7.38e-10 F | 1.516e-9−3×2.461e-10=7.777e-10 F，笔记已指出差别 |
| etaLls vs nLls | 笔记约1.7806e-2 H vs p0=1.7806e-10 H | 不能当作同一变量的简单单位换算 |
| Rs | 主模型8.703 Ω | 本轮外推8.7028583428 Ω；独立物理测量仍未证实 |
| Rrs / Rsf | 主图恢复解2800 / 27.4 Ω | 分别为14参数原界上限/下限，不是初始化28 / 2740 Ω |

## 附录C — 复现链、精度和未闭合边界

`AP_1p5.db/exp_10 → dropna/sort/(0,1e8] → 1590点 → try家族p0/weights/MAD/soft_l1/seed0 → 14参数解 → 4000点模型图 → fig_curver.png RGB像素一致`。

历史`curver_gp_residual.csv`采用data−sim符号，因此可从数据库减去res_re+j res_im恢复预测；恢复曲线与本轮模型最大差1.46e-14 Ω。它与nutsver残差字节相同，不能作为两次独立实验。

报告SSE/RMSE/相对指标在其显示精度内吻合。AIC/BIC文字分别31028.036/31112.941，本轮为31028.035497916/31112.940408577，约5.0e-4/5.9e-4差异；不要声称三位小数严格一致。可能来自运行微差或转录，仍保留。像素完全相同不是PNG二进制hash相同：原图是Snipaste截图，新图由Matplotlib写入，元数据不同。

此链证明现有材料可重现该数值/图像，不证明：哪个同构脚本当时被点击、参数唯一、物理支路真实、原始仪器数据未处理，或所有报告图均已复现。
'''
(B/'OPEN_QUESTIONS.md').write_text(text,encoding='utf-8')

evolution=f'''# Model Evolution — evidence-led working history

日期：2026-10-05。所有箭头表示工作假设中的关系，不表示研究者已确认的决策。未指定canonical。

## 可确认的模型家族

| 调查标签 | 端口/拓扑 | 代表文件 | 证据状态 |
|---|---|---|---|
| M0-single | 单相配置、11参数 | {link('Circuit_Fitting/exp_1_fit.py')} | CONFIRMED，与同为11参数的三相模型不能混同 |
| M1 | exp10三相化简，11参数，无附加支路 | {link('Circuit_Fitting/exp_10_fit.py')}、{link('baselines/CurVer_no_L.py')} | CONFIRMED |
| M2 | M1核心+1.5jωLad，12参数 | {link('baselines/CurVer.py')}、{link('baseline1/CurVer.py')} | CONFIRMED；六组消融属此家族 |
| M2-shunt | 加Lad后在测量端并Cad | ee0a7f1:tmp/tmp.py，保存于evidence | CONFIRMED存在，实际报告用途UNKNOWN |
| M3-Rseries | 每相R+(L∥C)，14参数 | {link('tmp/tmp.py',214)}、tmp1–3 | CONFIRMED存在，报告正文同类拓扑 |
| M3-RCparallel | 每相L∥(R+C)，14参数 | {V3}、try2、NUTSVerTry | CONFIRMED可复现主图/历史曲线 |
| W2 | M2+模块化提取/拟合/验证 | {link('baseline1/workflow/run_workflow.py')} | CONFIRMED仍12参数；不是M3自动化版 |
| W2-ReIm / W2-magphase | 相同12参数电路，残差实验分支 | experimental_reim_workflow / mag_phase_workflow | CONFIRMED；前者本身不是与原CurVer不同的ReIm创新 |
| W2-HP30 | 换AP_30输入，保留默认先验 | {link('baseline1/hp30_workflow/fitting.py',90)} | CONFIRMED试跑；有效性未确认 |

```mermaid
flowchart TD
    S[单相和三相基础电路] --> M1[11参数三相端口]
    M1 --> M2[12参数 附加Lad]
    M2 --> AB[12参数 六组消融]
    M2 --> SH[端口并联Cad探索]
    M2 --> R[14参数 R加并联LC]
    M2 --> RC[14参数 L并联串联RC]
    RC --> FIG[本轮精确重现报告主图]
    RC --> N[14参数NUTS候选]
    M2 -. 整理关系推断 .-> W[12参数workflow]
    W --> WM[幅相与ReIm分支]
    W --> HP[HP30试跑]
```

关系/意图总体MEDIUM confidence。两种14参数并不等价；不能把它们合并为唯一V3。

## 时间证据（Git入库时间，不等于编写时间）

| 事件 | 可确认内容 | 可推断与限制 |
|---|---|---|
| ac82e9a 2025-10-14 | 初始提交已有Csf/Llr/fr_fa、参考论文及figure1.m | 最早仓库材料，不代表研究起点 |
| 4a21dfa 2026-01-12 | exp_10_fit、Ztotal_result.csv等进入；旧MATLAB文件删除 | 当前Python主体不能抹去更早绘图历史 |
| 1月组会材料 | 讨论eta异常、Cad位置尝试、支路遮蔽、小附加电感 | 内容支持探索分支，但文件名日期不单独证明执行时间 |
| 9cf0d55 2026-02-09 | baselines/CurVer、IniVer、NUTSVer进入；CurVer是纯Lad版 | M2至少当时已归档 |
| ee0a7f1 2026-03-11 | Anly/GA/No-L及其他比较；tmp有端口并联Cad | 优化与拓扑探索并行 |
| fc8ad7f 2026-03-28 | baseline1六组消融及3月27日日志；tmp为R+(L∥C) | 12参数benchmark与14参数实验同时存在 |
| 44cd9ba 2026-05-27 | try/try、try2、NUTSVerTry及PaperFigure进入 | 后一拓扑首次可见入库；不能断言五月才开发 |
| 当前未提交工作树 | workflow/MD/report未跟踪，CurVer改为import initial_params | clean workflow的意图/创建日仍需人工确认 |

来源：{G}、{local('evidence/git_diff.txt')}、{P}。报告封面April2026与五月入库不构成造假证据；有延迟归档的合理替代解释。

## 本轮最合理历史解释

**HIGH：** 无附加支路→附加纯电感→多种端口修正方案确实共存；12参数用于已保存消融，14参数L∥(R+C)能重现主结果。报告热图是12参数，主图是可复现的14参数输出。

**MEDIUM：** 作者很可能将多个开发阶段汇入报告，正文留有R+(L∥C)文字；后来workflow围绕稳定的12参数baseline整理。没有证据说明作者正式否定14参数或指定12参数最终canonical。

**LOW/UNKNOWN：** 每次改版的确切动机、正式提交报告日期、最终采用决策、原始仪器边界。应由HUMAN_REVIEW确认并回查证据。

**This is a working hypothesis, not a confirmed conclusion.**

若出现早期workflow快照、正式回退说明、或与图绑定的另一套运行记录，需修订相应箭头；不删除现有分支。
'''
(B/'MODEL_EVOLUTION.md').write_text(evolution,encoding='utf-8')

recon=f'''# Current Best Reconstruction

日期：2026-10-05。人工复核前版本；不指定canonical、不改模型、不修报告。

## 如果今天必须解释这个项目

这个项目用SQLite中的电机端口幅相谱拟合三相灰箱等效电路，目标是100Hz至约100MHz的端口响应。早期的11参数电路无法充分表现UHF尾部；研究者试验了附加电感与多种RC/LC结构。12参数版本成为已保存的优化消融benchmark；14参数`Lad ∥ (Rad + Cad)`版本可以精确重现报告主图。这一数值对应关系现在有直接验证证据，具体历史运行文件/决策仍是工作假设。

## 已闭合的数值证据（CONFIRMED）

- 数据：`D:\\Desktop\\EE5003\\data\\AP_1p5.db / exp_10`，A1–B1C1星形短接；Freq/Hz、Zabs/Ω、Phase/deg。
- 1601原点筛为1590点，100至99967334.06Hz；N_SAMPLES=2000使该例不下采样。
- 复制的try/try.py，以原14参数初值、原bounds、seed0、DE60×10、多起点120/top10、soft_l1、MAD与0.3–4权重运行。
- 得到RMSE_raw=130.8719570241928、SSE_raw=54465351.850388095、相对幅值RMSE=0.2164369083981353、best_cost=1528.2956920576398。
- 与原外部`curver_gp_residual.csv`恢复曲线差不超过1.46e-14Ω；输出图与报告`fig_curver.png`全部RGB像素一致。
- try.py、try2.py、NUTSVerTry.py的NumPy模型/优化关键函数相同；不能据此确定当时点击的文件。

详见{E}、{P}及{local('checks/parameter_register_v3.csv','恢复参数表')}。这些新参数是本轮重建结果，不是历史原件或唯一物理解。AIC/BIC与报告文字最后一位有约0.0005差别；PNG元数据不同，不能称文件hash一致。

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

先读{local('HUMAN_REVIEW.md')}；需要详细反证时查{local('OPEN_QUESTIONS.md')}；模型分支看{local('MODEL_EVOLUTION.md')}。完整调查操作记录在{local('PROJECT_AUDIT.md')}。

下一步应先完成历史意图/工况复核，再进入技术重建或restart设计。所有修正建议仍未实施。

**This is a working hypothesis, not a confirmed conclusion.** 已验证事实与历史/物理解释分开记录，人工回答也只作为线索，不能覆盖相反文件证据。
'''
(B/'CURRENT_BEST_RECONSTRUCTION.md').write_text(recon,encoding='utf-8')

hr=[
('你最终打算采用的是哪种端口附加结构？','主图/代码支持L∥(R+C)；正文与旧tmp支持R+(L∥C)。','HIGH（主图身份）；MEDIUM（最终研究意图）',['L∥(R+C)','R+(L∥C)','只加L','当时尚未定稿','Not sure'],'Q01/Q11'),
('报告正文是否可能保留了更早R+(L∥C)方案的文字？','这种旧拓扑确有代码；不是凭空猜测的笔误。','MEDIUM',['Yes','No','Not sure'],'Q11'),
('你是否把12参数消融与14参数主拟合结果合并进同一份报告？','热图12行、消融日志p=12，主图14参数精确重现。','HIGH',['Yes','No','Not sure'],'Q05'),
('当前workflow的创建意图是什么？','它调用12参数，并非报告14参数模型的模块化实现。','MEDIUM',['整理已有12参数baseline','有意回退并替代14参数','只为临时演示/试跑','Not sure'],'Q04'),
('nLls从约1.7806e-2 H变成1.7806e-10 H，当时是否有意调整？','尾数保留、指数差8位；原公式被计算但不使用。','LOW（修改动机）',['有意调整，有物理/数值理由','可能误录后沿用','记得另有定义/单位','Not sure'],'Q14'),
('exp_10作为主案例的选择理由是什么？','代码反复选它，多配置用于提取；未找到质量优选记录。','MEDIUM',['代表性三相端口','数据质量最好','最早跑通，沿用案例','其他已记录理由','Not sure'],'Q07'),
('原始测量的仪器、校准/夹具记录或建库源文件是否还在别处？','本次找到SQLite和接线表，未找到原始仪器文件。','UNKNOWN',['Yes，可提供本地路径','No，可能未保存','Not sure'],'Q09'),
('阻抗测量时电机的实际运行状态是什么？','fit_rs注释称no-load/turning，但无独立测量记录支持。','LOW',['静止且未供电','旋转/空载','不同配置工况不同','Not sure'],'Q15'),
('30HP分支沿用1.5HP先验是否属于有意压力测试？','代码明确打印“stage1仅诊断、fit用defaults”，不是仅靠猜测。','MEDIUM（意图）',['Yes','No，本应按30HP配置','Not sure'],'Q10'),
('是否曾保存NUTS的原始trace/MAP参数/链诊断，或另一套14参数消融？','当前脚本打印LS raw指标，未发现绑定主报告的采样归档。','UNKNOWN',['Yes，可提供路径','No','Not sure'],'Q05/Q22/Q23'),
('报告April2026封面与五月Git归档的关系是什么？','首次入库时间不等于开发时间，不能用它推翻封面。','LOW',['完成后延迟提交代码','封面是模板/未更新','报告后续又改过','Not sure'],'Q28'),
]
h='''# Human Review — 只复核意图与外部线索

你不需要重新审计代码，也不需要批准某个模型为canonical。请按记忆回答；不确定就选Not sure。回答作为下一轮线索保存，不会自动变成事实或覆盖文件证据。

本轮已经数值验证主图，不再问你“图是否能复现”。真正需要你帮助的是最终意图、版本拼接、测量工况和缺失记录的位置。建议先答HR-01、03、05、07、08。

'''
for i,(question,evidence,confidence,options,refs) in enumerate(hr,1):
    h+=f'## HR-{i:02d}\n\n{question}\n\nEvidence suggests: {evidence}\n\nConfidence: {confidence}。对应OPEN_QUESTIONS：{refs}。\n\nYour answer:\n\n'
    for opt in options:h+=f'- [ ] {opt}\n'
    h+='\n补充说明 / 本地文件路径：\n\n'
h+='''## 回答之后的处理

- 有具体文件线索：优先核对文件，再更新证据等级。
- 只有记忆：标记HUMAN_RECOLLECTION，与CONFIRMED分开。
- 记忆与文件冲突：保留双方，新增需要验证的解释，不强行消解。
- 本清单不授权修改历史代码、修报告或创建restart。
'''
(B/'HUMAN_REVIEW.md').write_text(h,encoding='utf-8')
print('Built',len(Q),'questions and',len(hr),'human review items')



