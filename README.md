# 「今晚看什么」——面向小团体观影决策的推荐 Agent

> 《机器学习》课程期末项目。输入人数、可看时长以外的硬约束（类型、年代、已看）与各成员口味，
> Agent 在 3 轮内给出 3—5 部**带理由**的候选片，并说明“为什么这一群人都会满意”。

项目对应的课程模块：**用户分群或推荐**（协同过滤、矩阵分解、结合用户约束的可解释推荐，评估排序效果）。

---

## 1. 快速开始

```powershell
cd mlagent
.\setup_env.bat                 # 建 .venv 并安装依赖（也可手动执行下面三行）
# py -3.12 -m venv .venv
# .\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

.\.venv\Scripts\python.exe -m data.download     # 下载 MovieLens latest-small（约 1 MB）
.\.venv\Scripts\python.exe -m src.agent --demo  # 脚本化演示，3 轮对话
.\.venv\Scripts\python.exe -m src.agent --interactive --users 1,2,3,4   # 真人输入
.\.venv\Scripts\python.exe eval\run_all.py      # 复现全部实验（约 40 s）
.\.venv\Scripts\python.exe tests\test_metrics.py
```

依赖：Python 3.12、numpy、pandas、scikit-learn、matplotlib（见 `requirements.txt`），不需要 GPU。

## 2. 目录结构

```
mlagent/
├── data/download.py        下载并解压 MovieLens（带 certifi 证书链）
├── src/
│   ├── config.py           路径与超参
│   ├── data.py             读取、留一法切分、类型特征
│   ├── baselines.py        热门榜、ItemKNN
│   ├── mf.py               带偏置的矩阵分解（SGD，numpy 实现）
│   ├── constraints.py      中文指令 → 硬/软约束（词典 + 否定作用域 + 年代正则）
│   ├── rank.py             约束掩码、候选池、池内效用、max-min 群体重排
│   ├── explain.py          推荐理由生成
│   └── agent.py            多轮对话（demo / interactive）
├── eval/run_all.py         一键复现：排序指标、RMSE、群体重排对照
├── eval/tune.py            超参与“最少评分条数”调参
├── tests/test_metrics.py   指标单元测试（6 条）
└── results/                summary.json + run.log
```

## 3. 方法

**① 约束解析（规则）**：词典映射类型词、`不要/除了/别` 的否定作用域、`2010 年以后`/`近五年`/`90 年代` 的年代正则、
`N 个人` 的人数抽取。解析结果是一个 `Constraints` 对象，直接编译成布尔掩码。选择规则而非分类器，
是因为约束解析错一条整轮推荐就废掉，可解释、可单测比“多一个模型”更重要。

**② 召回**：ItemKNN（余弦相似度 + shrink 惩罚）为每位成员取 Top-200，取并集作为候选池。
**取 Top-N 之前先按硬约束过滤**，否则会出现“池子里全是老片、按年份过滤后为空”的坑。

**③ 群体重排**：先去掉全局热度（`S - mean_u S`），再在池内算每位成员的**相对效用**（池内排名分位），
按 `mean` / `min`（max-min 公平）/ `blend` 三种策略聚合。去热度是必须的：不去的话所有人的榜单
都长得一样，公平策略等于没做（见第 4 节消融）。

**④ 解释**：只引用模型真正用到的量——类型、每位成员在候选池内的分位，不说无法验证的“大家都喜欢”。

## 4. 实验结果

> 图由脚本生成、可复现：`eval/fig_pipeline.py`（流程图）、`eval/figures.py`（Recall@K 曲线、
> 公平–效率权衡、去热度消融）、`eval/export_tables.py`（导出 `ranking.csv` / `group.csv`），
> 图片在 `results/figs/`。

数据：MovieLens latest-small，610 用户 / 9724 部片 / 100 226 条训练评分，留一法（每人最后一条评分作测试）。
363 位用户有 1 条 ≥4 分的测试项。统一协议：只推荐训练集中**评分人数 ≥ 20** 的条目。

### 4.1 排序与评分

| 模型 | Recall@10 | NDCG@10 | 覆盖率 | RMSE |
|---|---|---|---|---|
| 热门榜 | 0.0634 | 0.0299 | 1.0% | — |
| ItemKNN | **0.0854** | **0.0412** | 1.4% | — |
| 矩阵分解（32 维，30 轮） | 0.0331 | 0.0190 | 4.1% | **0.9688** |
| 全局均值 | — | — | — | 1.1169 |

> 一个值得写进报告的发现：MF 在**评分预测**上明显更好（0.9688 vs 1.1169），但在**Top-10 排序**上
> 输给 ItemKNN。原因是 MF 优化的是均方误差，不直接优化排序；热门片对 Recall 的贡献被 MF 的
> 个性化偏好“稀释”了。这也解释了为什么真实系统常用“召回多路 + 排序模型”而不是单一模型。

### 4.2 群体重排（50 组，25 组随机 + 25 组口味冲突）

| 组型 | 策略 | 首推·最差成员 | 列表·最差成员 | 列表·平均 |
|---|---|---|---|---|
| 随机 | mean | 0.485 | 0.460 | **0.772** |
| 随机 | min | **0.682** | **0.648** | 0.697 |
| 随机 | blend | 0.666 | 0.628 | 0.729 |
| 冲突 | mean | 0.644 | 0.554 | **0.808** |
| 冲突 | min | **0.754** | **0.708** | 0.754 |
| 冲突 | blend | 0.742 | 0.693 | 0.779 |

max-min 策略把“最不满意那位成员”的满意度提高 17—20 个百分点，代价是平均满意度下降 5—7 个百分点——
这正是群体推荐里经典的公平性/效率权衡。

### 4.3 消融：去掉“去热度”会怎样

若直接在原始 KNN 分数上做群体重排，`mean` 与 `min` 的首推最差成员分别为 0.986 与 0.985，
**两者几乎无差别**。因为 KNN 分数被条目热度主导，所有人的 Top 列表高度重叠，
“公平”没有可优化的空间。去热度后差异才显现（0.485 vs 0.682）。

## 5. 复现

- 随机种子固定为 `src/config.py: SEED = 2026`；`eval/run_all.py` 一条命令产出 `results/summary.json`。
- 数据版本：MovieLens latest-small（GroupLens，下载于 2026-11，`data/cache/ml-latest-small.zip` 可核对）。
- 运行环境：Python 3.12 + numpy 2.5 / pandas 3.1 / scikit-learn 1.9。

## 6. 数据、许可与合规

| 资源 | 用途 | 许可 |
|---|---|---|
| MovieLens latest-small（GroupLens Research） | 评分与电影元数据 | 研究用途，需署名（https://grouplens.org/datasets/movielens/） |
| numpy / pandas | 数值与数据处理 | BSD-3-Clause |
| scikit-learn | 聚类、对照模型 | BSD-3-Clause |
| matplotlib | 图表 | PSF/BSD 兼容 |

只使用公开匿名数据集，不含任何个人信息；“成员”均由 `user_id` 模拟，不代表真实用户。

## 7. 已知限制

1. 数据集**没有片长字段**，所以“90 分钟以内”这类约束暂时无法执行（已预留 `max_runtime`），
   需要接 TMDB 元数据才能补上。
2. 群体满意度是**代理指标**（池内分位），不是真实用户的打分；上表只能说明相对优劣。
3. MF 只做了 RMSE 目标，没有做 BPR 之类的排序目标，排序指标偏弱。
4. 新用户冷启动目前靠“从数据集里挑成员”模拟，没有实现问卷 → 聚类先验的在线路径。
5. 约束解析是规则，覆盖不到“像《疯狂的石头》那种”这类类比表达。

## 8. 后续改进

- 用 TMDB API 补片长、海报、简介，接上片长约束与更自然的解释；
- 把 MF 换成 BPR / 轻量双塔，提升 Top-K；
- 冷启动问卷 → KMeans 群组先验（`src/` 已预留接口思路）；
- 用 20 组真人小规模 A/B 替换代理指标。
