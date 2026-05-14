# AGENTS

## 项目概览

- 本仓库是 TAAC2026 的 PCVR 训练基线，核心模型为 `PCVRHyFormer`。
- 训练入口：`train.py`，推荐通过 `run.sh` 启动。
- 数据规范以 [industrial_data/README.md](industrial_data/README.md) 为准，`schema.json` 必须与 Parquet 布局一致。

## 运行方式

- 直接运行 `run.sh`（默认 RankMixer 模式）。
- 关键环境变量（覆盖 CLI 参数）：
  - `TRAIN_DATA_PATH`（Parquet 目录 + schema.json）
  - `TRAIN_CKPT_PATH`
  - `TRAIN_LOG_PATH`
  - `TRAIN_TF_EVENTS_PATH`

## 重要约定与风险点

- `ns_groups.json` 仅在 `group` NS tokenizer 下需要；两种模式参见 `run.sh`。
- `dataset.py` 采用文件系统共享张量，`num_workers` 过大可能导致稳定性问题。
- `trainer.py` 采用双优化器：Embedding 使用 Adagrad，非 Embedding 使用 AdamW。
- 根目录暂无测试用例；修改 `dataset.py` / `model.py` / `trainer.py` / `ns_groups.json` 时需谨慎。
- 修改笔记录在相关文件中，每次修改后建议更新笔记以保持同步。

## 关键文件

- [train.py](train.py)：训练入口、参数解析、模型构建
- [run.sh](run.sh)：推荐运行配置
- [dataset.py](dataset.py)：数据加载与批次构建
- [model.py](model.py)：PCVRHyFormer 架构实现
- [trainer.py](trainer.py)：训练循环与评估
- [utils.py](utils.py)：日志、早停、focal loss
- [ns_groups.json](ns_groups.json)：NS 分组示例
- [industrial_data/README.md](industrial_data/README.md)：数据集 schema 与说明
