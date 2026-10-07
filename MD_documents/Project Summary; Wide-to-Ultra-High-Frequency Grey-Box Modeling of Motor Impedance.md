## 1. Background

现代电机驱动系统中，快速开关器件与三相 PWM 会引发反射波、端口过电压、高频泄漏电流与 EMI 等复杂电磁现象。传统单线 DM/CM 分析在某些情况下会忽略三相调制、线缆、电机端口边界与接地路径之间的耦合效应，因此需要更完整、宽频且具有物理可解释性的电机阻抗模型。已有高频感应电机建模研究中，grey-box 等效电路方法兼顾了物理意义与参数可识别性，是工程上较有吸引力的一类路线。

现有 traditional grey-box motor impedance models 通常以低频 T 型等效为骨架，再加入有限个高频寄生支路，以覆盖从低频到 MHz 级别的宽频阻抗行为。这类方法在常规 EMI 频段内可以有效工作，但当关键异常特征推进到 `10^7–10^8 Hz` 时，其结构自由度、参数获取方式和边界建模能力开始受到挑战。

---

## 2. Problem

本项目面对的核心问题，不是一般意义上的“宽频拟合”，而是：

1. **目标异常推进到了更高频段**：在本文数据中，关键异常行为主要出现在 `10^7–10^8 Hz`，表现为超高频相位异常上翻与末端幅值回升。
2. **传统 grey-box 基线在 LF–MF 区间有效，但在 UHF 出现系统失配**：说明问题不是 baseline 从头错误，而是其在更高频边界上的表达能力不足。
3. **参数识别流程存在复现门槛**：传统流程常依赖 locked-rotor / no-load / dedicated CM/DM measurements、厂家数据、几何信息或人工读取谐振点；对 boundary-sensitive 的 UHF terminal modeling 而言，这种多实验拼接未必比“同一测量边界下的一致性识别”更稳。
4. **剩余误差更像结构性而非纯优化器问题**：通过参数扫描、branch hierarchy 分析与后续 optimizer-side check，可以判断 UHF 失配主要来自主导通路和拓扑刚性导致的结构不敏感，而不是“还没搜到更优参数”。

一句话概括：  
**本文要解决的是：如何在保持 LF–MF 物理可解释性的同时，构建一个可复现的 grey-box 框架，用于表达 `10^7–10^8 Hz` 的 UHF anomaly。**

---

## 3. Method

### 3.1 Overall philosophy

方法主线是：

**measured impedance spectrum → spectral feature extraction → parameter classification and analytical initialization → constrained fitting refinement → UHF failure diagnosis → port-side augmentation**

整体上属于一种 **feature-driven, reproducible, boundary-aware grey-box modeling framework**。其目标不是完全摆脱物理，而是尽可能从 measured impedance spectrum 本身提取信息，减少对额外专门实验和边界不一致辅助数据的依赖。

---

### 3.2 Reproducible parameter identification pipeline

参数识别流程分三层。

#### (a) 直接由阻抗谱提取的观测量

从 magnitude/phase 曲线中自动提取：

- `fr`, `fa`
- `|Z|max`, `|Z|anti-r`
- `Csf_HF`, `Csf_LF`
- `Lσ` 及派生的 `Lls`, `Llr`

方法包括：

- phase-assisted extrema
- LF/HF plateau detection
- local smoothing
- simple robustness constraints

代表公式包括：

- `Z = |Z|(cosθ + j sinθ)`
- `Cs = -1 / (2π f Im(Z))`
- `Leq = Im(Z)/(2πf)`  
    这些量构成后续解析初始化的输入。

#### (b) 由解析关系初始化的高频参数

基于上述观测量，通过 approximate analytical relations 初始化：

- `Csw`
- `ηLls`
- `Rsw`
- `Rsf`
- `Csf0`

对应逻辑是：

- `fr` + `Csf` → `Csw`
- `fa` + `Csf` → `ηLls`
- `|Z|max` + `Rcore` + inductive reactance → `Rsw`
- `|Z|anti-r` → `Rsf`
- `Csf_LF` + `Csf_HF` → `Csf0`

这一步的意义在于：baseline 参数不是完全由黑箱优化器“猜”出来，而是有一条从谱特征到解析初始化的物理路径。

#### (c) 弱可辨识项的拟合修正

对无法稳定唯一确定的参数，采用 constrained fitting 或 engineering approximation，例如：

- `Rs`
- `Rr/s`
- frame-related terms
- residual correction terms
- 部分 empirical prior（如 `Rcore`）
- `Lm` 在高频可弱化或忽略

核心思想是：**不是所有参数都同等可观测，因此必须区分 directly observable、analytically initialized、weakly identifiable 三类参数。**

---

### 3.3 Reduced reliance on auxiliary tests

本项目并非简单“省实验”，而是强调：  
对于 UHF terminal modeling，**同一测量边界下的内部一致性** 往往比更多但条件敏感的附加实验更重要。传统辅助测试可能受 wiring、grounding、fixture、thermal state 和 operating condition 影响，导致参数跨边界拼接，从而削弱统一 terminal model 的闭合性与复现性。本文流程因此优先从 measured impedance spectrum 本身提取信息，只把极少数弱可辨识项交给 constrained refinement。

---

### 3.4 Baseline grey-box circuit

基线模型属于 **conventional physically interpretable grey-box motor impedance formulation**，特点是：

- 低频骨架保留 classical motor equivalent circuit：`Rs`, `Rr`, leakage inductances, magnetizing branch, core-loss branch；
- 高频扩展加入 winding-to-frame capacitive path、inter-turn capacitive path、damping/loss terms、leakage partitioning effects 以及必要的 frame-return path；
- 网络经 structured equivalent transformations 化简为可直接扫频的 terminal impedance expression，便于在拟合循环中高效评估。

该 baseline 在本文中不是最终答案，而是一个 **physically interpretable reference formulation**，用于后续分析 UHF failure mechanism。

---

### 3.5 UHF failure analysis

对 `10^7–10^8 Hz` 失配的分析表明：

- 在 LF–MF，high-frequency parasitic paths 基本近似开路，terminal response 主要由传统低频骨架决定；
- 随频率升高，`Zbra`, `Zcsf0` 等 bypass / return paths 快速降低，branch hierarchy 被重写；
- 到 UHF，低阻抗 capacitive / return paths 主导 terminal response，而局部 inductive ingredients（如 `Z_nLls`）虽仍存在，却在 terminal level 被淹没；
- 因而失配不是“缺少感性成分”本身，而是这些成分在当前拓扑中无法控制终端输出；
- 结论是：**UHF mismatch is structural, caused by dominant-path locking and topological rigidity.**

---

### 3.6 Port-side high-frequency augmentation

为补足 baseline 在 UHF 的结构自由度，引入端口侧高频修正单元：

- `Lad || (Cad + Rad)`

其定位不是“已被证实的新的内部电机元件”，而是：

- **a boundary-aware port-side UHF augmentation**
- 用于表示 terminal / lead / fixture / frame-return / grounding-related parasitic dynamics 在 measured port 处的等效表现

其物理动机来自经验事实：

- UHF phase-transition tendency 在多组实验设置中反复出现；
- 出现频段相对稳定；
- 与内部接线方式变化关系不强；
- 因而更像 measurement boundary / external parasitics 的共同表现，而非某个额外内部 lumped branch 的重排。

使用谐振块而非纯串联电感的原因是：目标现象不是单调感性抬升，而是一个局部频率选择性的 phase rise + magnitude recovery，因此需要一个最小的 localized pole-zero mechanism。

---

### 3.7 Fitting algorithm

拟合采用 **global-to-local optimization**：

- 解析初始化给出 plausible neighborhood；
- `DE` 用于 lightweight global coarse search；
- `least_squares` 用于 local refinement；
- objective 为 complex impedance residual；
- `Re/Im` 分别缩放；
- 使用 `soft_l1` robust loss；
- 采用 `frequency weighting` 自动强调 resonance / antiresonance / rapid phase transition 区域；
- 加入 `log-bounds scaling` 约束参数空间。

选择 `DE + LS` 而非 `GA` 的原因：

- 参数连续；
- 已有解析先验；
- 全频模型评估代价较高；
- 因而“轻量全局粗搜 + 高效局部精修”优于完全基于群体演化的重型全局搜索。

---

## 4. Result

### 4.1 Main modeling result

当前方法 `CurVer` 在 full-band fitting 上实现了三个核心目标：

1. LF–MF 区间主 resonance / antiresonance 结构被保留；
2. 全频 magnitude / phase 主趋势与实验一致；
3. `10^7–10^8 Hz` 的 UHF phase rise 与末端 magnitude recovery 被恢复。

代表性指标：

- `p = 14`
- `N_freq_fit = 1590`
- `N_residual_dim = 3180`
- `SSE_raw = 5.44654e+07`
- `RMSE_raw = 130.872`
- `AIC_raw = 31028.036`
- `BIC_raw = 31112.941`  
    表明该方法不是只在局部修曲线，而是在可控复杂度下完成了 full-band modeling。

---

### 4.2 Computational complexity

CurVer 的拟合代价为：

- `model_eval = 22801`
- `objective_calls = 8660`
- `residual_calls = 22801`
- `T_fit_total = 7.996 s`
- `T_global = 1.637 s`
- `T_local = 5.955 s`

这说明方法不是“精度换时间”的极端方案，而是在 accuracy–efficiency 上取得平衡。

---

### 4.3 Residual diagnosis

GP residual analysis 表明，当前模型虽已显著改善 UHF anomaly，但仍存在 frequency-local structured mismatch。主要 residual peaks 为：

- `Re`: `2.34e5`, `3.86e4`, `1.19e4 Hz`
- `Im`: `1.96e5`, `3.48e4`, `3.08e5 Hz`
- `Phase`: `1.83e7`, `3.19e7`, `7.58e7 Hz`

含义是：

- `Re/Im` 的主要剩余系统误差集中在 `10^4–10^5 Hz` 的 resonance/antiresonance neighborhood，提示 motor core baseline 仍有改进空间；
- `Phase` 的主要 residual peaks 已推到 `10^7–10^8 Hz`，说明当前模型已把“主要剩余失配”压缩到局部 UHF 子区间。

---

### 4.4 Ablation study

消融实验围绕三类问题展开：

1. **Optimization framework**：是否必须 global-to-local；
2. **Objective-function design**：frequency weights 与 separate Re/Im scaling 是否必要；
3. **Algorithm choice**：DE 是否优于 GA。

对比变体包括：

- `LS only`
- `DE only`
- `No frequency weights`
- `No Re/Im separate scaling`
- `GA + LS`
- `Current Version`（完整流程）

主要结论：

- **CurVer 是综合最优方案**：accuracy 近最优且跨 seed variance 极低；
- `LS only` 很快，但明显不够准确；
- `DE only` 昂贵且不准，说明全局搜索 alone 不足；
- `GA + LS` 相比 `DE + LS` 更慢且更不稳，说明 DE 更适合作为 global stage；
- `No frequency weights` 可能略降 mean RMSE，但 variance 显著增大，说明 frequency weighting 主要提升鲁棒性与一致性；
- 去掉 `Re/Im separate scaling` 会同时恶化精度与稳定性，说明 residual balancing 是必要设计。

---

### 4.5 Robustness and identifiability

多 seed 统计与 parameter CV heatmap 表明：

- CurVer 不仅拟合更好，而且参数收敛更稳定；
- `DE only` 和 `GA + LS` 的 parameter variation 明显更大；
- `Csf0` 与 `nLls` 等参数跨 seed 波动显著，说明其 weakly identifiable 或与邻近参数强耦合。

因此，本文的贡献不只是降低误差，更在于：

- 提供了更稳定的参数识别流程；
- 同时明确了 grey-box model 的可解释性边界。

---

## One-sentence takeaway

本文建立了一套 **wide-to-ultra-high-frequency, reproducible, boundary-aware grey-box motor impedance modeling framework**：它通过阻抗谱特征提取、解析初始化、受约束拟合和端口侧高频修正，在保持 LF–MF 物理一致性的同时，恢复了 `10^7–10^8 Hz` 的 UHF anomaly，并通过消融与稳定性分析证明改进主要来自结构扩展与方法链设计，而非单纯更强优化器。