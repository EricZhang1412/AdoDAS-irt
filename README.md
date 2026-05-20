# AdoDAS-irt

AdoDAS 2026 ACM MM Grand Challenge 参赛工程。基于 **IRT/factor-structured joint head + 树模型专家 hybrid stack** 路线,替换原 `backup/` 的纯深度学习 baseline。

## 核心思想

1. **IRT 联合头**:21 个 DASS item 共享 3 维潜在严重度 θ_DAS,放大有效样本量。a_j 是 item discrimination,b_j 是 difficulty,CORAL 累积阈值用 `cumsum(softplus)` 保单调。
2. **Hybrid stack**:LightGBM/XGBoost/HGBT/ExtraTrees per item + IRT 联合头 + (可选) 序列分支,subject-disjoint 5 折 OOF。
3. **NNLS 元融合**:per-item / per-target 受约束 ridge,A1 用多输出共享耦合。
4. **后处理**:A2 per-item ordinal threshold optimization,A1 isotonic OR logit-shift(二选一)。
5. **终训**:在 train+val 全量数据 refit 每个 base learner(stage10 模式),用 OOF 学到的权重融合。

## 目录结构

```
adodas/
├── data/         # feature_io, pooled_table, ssl_pca, views, folds, labels
├── models/       # tree_experts, irt_joint, irt_trainer, heads
├── stack/        # meta_blend, thresholds, calibration
├── submit/       # build_submission, refit_all_labeled
└── utils/        # metrics, seed, io, dass21

configs/          # paths/features/trees/irt/stack yaml
scripts/          # 01-07 流水线 + 99_smoke_test
tests/            # unit tests
run.sh            # 统一入口脚本
```

## 安装(推荐 uv)

[uv](https://github.com/astral-sh/uv) 是 Rust 写的 Python 包管理器,比 pip 快 10-100×。

```bash
# 一次性安装 uv(macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在项目目录里创建 .venv 并安装依赖(读 pyproject.toml)
./run.sh setup
# 等价于:
#   uv venv
#   uv pip install -e ".[dev]"
```

`run.sh` 会自动检测 `.venv/bin/python`,所以**不需要手动 `source .venv/bin/activate`**,直接 `./run.sh <command>` 就行。

也可以走传统 pip 路线:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## 使用

所有命令统一通过 `run.sh` 调用:

```bash
./run.sh setup        # 安装依赖(uv)
./run.sh smoke        # 端到端 smoke test(合成数据,5 分钟)
./run.sh test         # pytest 单元测试

# 流水线(每一步独立运行)
./run.sh features     # 01_build_features.py
./run.sh folds        # 02_make_folds.py
./run.sh trees        # 03_train_trees_oof.py
./run.sh irt          # 04_train_irt_oof.py
./run.sh meta         # 06_meta_blend.py
./run.sh refit        # 07_refit_all_and_predict.py

# 或者一把梭
./run.sh pipeline     # 顺序跑 features → folds → trees → irt → meta → refit
```

### 配置覆盖

`run.sh` 通过环境变量读 5 个 yaml 配置,可以覆盖:

```bash
PATHS_CFG=configs/paths_server.yaml ./run.sh pipeline
IRT_CFG=configs/irt_bifactor.yaml ./run.sh irt
```

也可以直接给底层脚本透传参数:

```bash
./run.sh trees --families lightgbm --views fused      # 只跑 lightgbm × fused
./run.sh smoke --n-subjects 20 --d-feat 64
```

### 在远端服务器上跑

1. clone 仓库 + `./run.sh setup`
2. 改 `configs/paths.yaml` 指向服务器上的 `feature_root` 和 `manifest_dir`
3. `./run.sh pipeline`
4. 提交 CSV 在 `output/submission/` 下

## 提交格式

- **A1**: `anon_school, anon_class, anon_pid, p_D, p_A, p_S`(概率 ∈ [0,1])
- **A2**: `anon_school, anon_class, anon_pid, d01, ..., d21`(整数 ∈ {0,1,2,3})
- 默认 participant-level。

## 不做的事

- MTCN/Cross-modal attention/AuxFiLM(被替换对象)
- TabPFN / AutoGluon(留作 ablation)
- 序列分支默认关(stretch goal,见 `scripts/05_train_seq_oof.py` 注释)
