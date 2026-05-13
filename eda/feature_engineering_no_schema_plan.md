# TAAC2026 特征工程整理版：不修改 schema.json 的执行方案

本文基于官方 flat column 数据说明、项目 `dataset.py` / `model.py` 的 schema 消费方式，以及 EDA 报告中的缺失、高相关、长尾和序列信号，整理一套 **不修改 `schema.json`** 的特征工程执行方案。

## 1. 官方字段结构核对

官方 demo 数据是 1000 行、120 列的 flat column parquet。字段分为：

| 类别 | 数量 | 字段模式 | 处理要点 |
|---|---:|---|---|
| ID & Label | 5 | `user_id`, `item_id`, `label_type`, `label_time`, `timestamp` | ID 不当连续数值；`label_time` 默认进 leakage watchlist；`timestamp` 可做历史相对时间。 |
| User Int | 46 | `user_int_feats_*` | 标量 + list；高缺失和 multi-hot 需要 mask/长度/去重。 |
| User Dense | 10 | `user_dense_feats_*` | list<float>；要标准化、clip、missing mask；与同 fid int 对齐。 |
| Item Int | 14 | `item_int_feats_*` | 标量 + `item_int_feats_11` list；高基数/高缺失要单独处理。 |
| Domain Sequence | 45 | `domain_a/b/c/d_seq_*` | 多域行为序列；要校验 timestamp 顺序、截断方向、长度和 recency。 |

项目当前通过 `schema.json` 描述 `user_int`、`item_int`、`user_dense` 和 `seq`，训练时据此构造 embedding、dense projection 和序列 embedding。因此本文所有新增特征默认以 **sidecar parquet / EDA baseline / runtime extra features** 方式实现，不直接改 schema。

## 2. 不能改 schema 时的三种落地方式

### 2.1 Sidecar 特征表（推荐第一步）

生成一个与原 parquet 行顺序对齐的 `sidecar_features.parquet`：

```bash
python eda/no_schema_feature_engineering.py \
  --input-dir /path/to/train_parquet_dir \
  --schema-path /path/to/schema.json \
  --output-path /tmp/taac_sidecar_features.parquet
```

适用：EDA、GBDT 单模型、特征 ablation、生成 feature triage table。

优点：不触碰原 parquet 和 schema；可以快速验证哪些特征家族有效。

### 2.2 训练外部 GBDT baseline

将 sidecar 与 label 合并，训练 LightGBM/CatBoost 单模型作为特征强度评估器。注意这只是单模型 baseline，不与 PCVRHyFormer 做 ensemble。

### 2.3 Runtime extra dense（第二阶段）

若某些 sidecar 特征稳定有效，可以在 `dataset.py` 运行时生成 `extra_dense_feats` 并在模型里单独加一个 extra dense projection。这样仍不改 schema，但需要改代码和 checkpoint 配置。建议等 sidecar ablation 明确收益后再做。

## 3. 数据预处理执行方案

### 3.1 缺失值：missing / empty / OOV / padding 分离

当前训练数据加载会把 int 特征中的 null 和 `<=0` 都转成 0，0 又是 embedding padding。这样稳定但会混淆缺失、空列表、OOV 和真实 padding。

执行：

1. 对所有 `user_int_feats_*`、`item_int_feats_*`、`user_dense_feats_*` 生成 `__missing`。
2. 对用户侧、item 侧生成缺失聚合：`missing_cnt`、`missing_ratio`。
3. 对 list 字段额外生成 `len==0` 的 empty indicator。
4. OOV 由训练日志中的 OOB stats 单独监控，不混入 missing。

代码入口：`_missing_features()`。

### 3.2 高缺失字段：不直接删除，先做四组 ablation

对缺失率高的字段分别测试：

| 实验 | 输入 |
|---|---|
| raw only | 原值按当前模型进入 embedding/dense。 |
| mask only | 只用 missing indicator。 |
| raw + mask | 原值和 mask 都用。 |
| drop | 删除或不使用该字段。 |

判定：如果 mask only 有收益，说明缺失模式本身有信号；如果 raw + mask 稳定优于 raw only，应保留 mask。

### 3.3 高基数 ID：频次化和 hash 化优先

`user_id` 与 `item_id` 不应作为连续数值。第一阶段先生成：

- `fe_user_id__freq_in_input`
- `fe_user_id__log1p_freq_in_input`
- `fe_item_id__freq_in_input`
- `fe_item_id__log1p_freq_in_input`

注意：脚本里的 `*_freq_in_input` 只适合 EDA。正式验证/测试时要用训练期 frozen count table transform holdout/test，避免未来信息泄漏。

### 3.4 Dense 特征：标准化与 mask 分开做

不修改 schema 时，标准化不要覆盖原 parquet。建议生成 sidecar dense statistics：

- 每个 dense fid 的 missing。
- `valid_len`、`sum`、`mean`、`std`、`min`、`max`、`norm`。
- train split 上拟合 clip/standardize 参数，保存 JSON，然后 transform valid/test。

当前脚本先提供 missing；dense summary 可作为下一步扩展。

### 3.5 序列截断方向：先校验再改训练参数

官方只说明 sequence 是 list<int64>，并没有在 README 中保证排序方向。schema 里有每个 domain 的 `ts_fid`，因此应：

1. 读取 schema 中每个 domain 的 timestamp 列。
2. 对每行检查 timestamp list 单调递增/递减/无序比例。
3. 若旧到新排列，训练截断应保留 tail；若新到旧，保留 head；若无序，先按 timestamp 对所有 sideinfo 同步排序。

在未确认前，不建议盲目增大 `seq_max_lens`。

## 4. 特征构建执行方案

### 4.1 序列长度 / 活跃度 / 覆盖度

生成：

- 每个 sideinfo 列的 `__len`、`__non_empty`、`__unique_cnt`。
- 每个 domain 的 `fe_seq_*__len`、`log1p_len`、`non_empty`。
- 全域 `total_len`、`max_len`、`non_empty_domain_cnt`、各 domain 长度占比。

代码入口：`_sequence_features()`。

### 4.2 Recency / 时间窗口行为

若提供 `--schema-path`，脚本会读取 `seq.*.ts_fid`，对每个 domain 生成：

- `last_gap`
- `mean_gap`
- `std_gap`
- `cnt_300s`
- `cnt_3600s`
- `cnt_21600s`
- `cnt_86400s`
- `cnt_259200s`
- `cnt_604800s`
- `cnt_2592000s`
- `burst_1h_over_7d`

代码入口：`_add_time_window_features()`。

### 4.3 多值离散 list 特征

对 `user_int_feats_{15,60,62-66,80,89-91}` 和 `item_int_feats_11` 生成：

- `len`
- `non_empty`
- `unique_cnt`
- `dup_ratio`
- `first`
- `top1`
- 用户/item 多值总长度与非空列数

代码入口：`_multi_value_features()`。

### 4.4 Popularity / Activity / Cold-start

第一阶段仅做 user/item 频次。第二阶段建议补：

- item 近期频次：必须用 train-only rolling/frozen table。
- user 活跃等级：从历史序列长度和窗口行为计数派生。
- cold item/user bucket：基于 frozen frequency threshold。

### 4.5 Item-history match

不修改 schema 时先在 sidecar 中做 match 特征。执行步骤：

1. 用 schema 和 EDA 找 item fid 与 sequence fid 的候选对应关系。
2. 对候选 fid 构造 exact match、match count、recent match、last match gap。
3. 所有 match 必须只用 `timestamp` 之前的历史序列。
4. 先用 GBDT/单特征 AUC 验证，再决定是否进入 runtime extra dense。

## 5. 特征选择 / 治理方案

### 5.1 先建 leakage watchlist

默认 blacklist：

- `label_time`
- conversion 后才生成的字段
- post-event action count
- full-session summary
- full-history target rate
- test 合并 popularity

Graylist：

- global count
- user-item pair stat
- item popularity
- target encoding
- session-level aggregate
- full long-history aggregate

Whitelist：

- raw categorical
- missing indicator
- sequence length
- strictly-past recency
- recent window count
- train-only frequency
- cheap hash cross

### 5.2 单特征 AUC + strict split 稳定性

每个 feature family 记录：random CV、time CV、group CV、main/stress delta、adversarial validation importance。

如果 random 强但 strict 崩，降级为 graylist 或 blacklist。

### 5.3 高相关特征组

EDA 报告提示存在强相关字段组。处理优先级：

1. 保留缺失率低的。
2. 保留 strict CV 稳定的。
3. 保留单特征收益高的。
4. 保留低 cardinality / 低 latency 的。
5. 深度模型中优先放同一 NS group，而不是机械删除。

### 5.4 Latency-aware selection

每组特征都记录：AUC gain、logloss gain、preprocess time、batch inference time、embedding 参数量、显存、OOV/unseen 比例、复现风险。

## 6. 推荐实验顺序

1. 生成 sidecar cheap features。
2. 跑 GBDT 或轻量 DNN 单模型 ablation：`+missing`、`+seq_stat`、`+recency`、`+frequency`。
3. 校验 sequence timestamp 排序方向和 truncation 信息损失。
4. 做 item-history match 候选。
5. 在 main/stress CV 稳定后尝试 OOF target encoding。
6. 将稳定有效的 sidecar 特征改造成 runtime extra dense，而不是直接改 schema。

## 7. 特征治理表模板

| feature family | example | leakage | latency | reproducibility | action |
|---|---|---:|---:|---:|---|
| missing mask | `fe_user_int_feats_101__missing` | low | low | low | use |
| sequence length | `fe_seq_c__len` | low | low | low | use |
| recency | `fe_seq_d__last_gap` | low-mid | low | low | use |
| frequency | `fe_item_id__freq_train` | mid | low | mid | train-only frozen table |
| match | `category_match_7d` | mid | mid | mid | past-only, validate |
| target encoding | `item_cvr_oof` | high | mid | high | graylist |
| full-history stat | `user_item_full_ctr` | very high | mid-high | high | default deny |
| long sequence full | `seq_d_512+` | low | high | low | latency ablation |
