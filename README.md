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
└── utils/        # metrics, seed, io

configs/          # paths/features/trees/irt/stack yaml
scripts/          # 01-07 流水线 + 99_smoke_test
```

## 快速开始

1. 改 `configs/paths.yaml` 指向远端服务器上的 `feature_root` 和 `manifest_dir`。
2. 安装依赖:`pip install -e .`(或 `uv pip install -e .`)。
3. 顺序执行:

```bash
python scripts/01_build_features.py --config configs/paths.yaml
python scripts/02_make_folds.py     --config configs/paths.yaml --n-folds 5
python scripts/03_train_trees_oof.py --config configs/trees.yaml
python scripts/04_train_irt_oof.py  --config configs/irt.yaml
python scripts/06_meta_blend.py     --config configs/stack.yaml
python scripts/07_refit_all_and_predict.py --config configs/stack.yaml
```

4. 提交 CSV 在 `output/submission/` 下。

## 端到端 smoke 测试

```bash
python scripts/99_smoke_test.py --n-subjects 10
```

5 分钟内跑完整个 pipeline(dummy 数据)。

## 提交格式

- **A1**: `anon_school, anon_class, anon_pid, p_D, p_A, p_S`(概率 ∈ [0,1])
- **A2**: `anon_school, anon_class, anon_pid, d01, ..., d21`(整数 ∈ {0,1,2,3})
- 默认 participant-level。

## 不做的事

- MTCN/Cross-modal attention/AuxFiLM(被替换对象)
- TabPFN / AutoGluon(留作 ablation)
- 序列分支默认关(stretch goal)
