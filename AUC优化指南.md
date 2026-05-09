# AUC 优化指南

本文档聚焦于本项目（PCVRHyFormer 训练基线）提升 AUC 的可行路径，按“数据 → 模型 → 训练 → 评估 → 工程”分层给出思路与操作要点。

## 1. 数据层优化

### 1.1 验证集切分方式
- 当前按 Row Group 尾部切分验证集，若数据按时间/来源排序，易出现分布偏移，AUC 不稳定。
- 建议：在数据生成阶段打散 Row Group；或按时间滑窗/分层抽样生成验证集。

### 1.2 样本不平衡处理
- 任务为二分类（label_type==2 为正样本）。正负样本严重不平衡时 AUC 易波动。
- 建议：
  - 训练侧使用 `focal` loss 并调 `focal_alpha`/`focal_gamma`。
  - 数据侧做负采样（控制负样本比例）。

### 1.3 高基数特征处理
- `emb_skip_threshold` 会直接跳过高基数 embedding，可能丢失有效信号。
- 建议：
  - 优先降低 `emb_skip_threshold`，观察 AUC 变化。
  - 若显存紧张，可对高基数特征做 hash/截断，而非直接置零。

### 1.4 序列域质量
- 序列长度与时间桶依赖数据字段完整性，缺失会弱化时序建模。
- 建议：排查序列字段的空值比例与长度分布，保证时间戳列与 side-info 对齐。

## 2. 模型结构优化

### 2.1 NS Tokenizer 策略
- `group` 模式：每组一个 token，语义清晰但 token 数固定。
- `rankmixer` 模式：可调整 token 数，但需满足 $d\_model % T == 0$。
- 建议：
  - 若 AUC 无提升，可尝试切换模式或调 `user_ns_tokens`/`item_ns_tokens`。

### 2.2 序列编码器选择
- `swiglu`：轻量，可能欠拟合。
- `transformer`：表达强，但可能过拟合。
- `longer`：适合长序列与稀疏信号。
- 建议：对三种编码器做小规模网格对比（固定其他超参）。

### 2.3 RoPE 与时间桶
- `use_rope` 和 `use_time_buckets` 可增强时序建模。
- 建议：
  - 若序列长度较长或时间间隔信息重要，启用 RoPE。
  - 时间桶需要时间戳列完整，缺失会降低收益。

### 2.4 RankMixer 结构约束
- `rank_mixer_mode=full` 要求 $d\_model % T == 0$。
- 建议：调参前先计算 T，避免因约束导致训练失败。

### 2.5 非序列与序列特征融合原理（基于实现）
- **融合入口**：`MultiSeqQueryGenerator` 先生成每个序列域的 Q tokens，Q 来自 NS tokens 与该序列的 MeanPool，全局信息在此汇聚。
- **融合方式**：`CrossAttention` 用 Q 作为 Query，序列 token 作为 Key/Value，实现“非序列 → 序列”的跨注意力注入。
- **融合收敛**：所有域的 Q tokens 与 NS tokens 拼接后进入 `RankMixerBlock`，进行 token mixing + FFN，完成跨域融合。

基于 [model.py](model.py) 中 `MultiSeqQueryGenerator`、`CrossAttention`、`RankMixerBlock` 的实现，这套流程的核心是“**以 Q tokens 为桥梁，让非序列信息参与每个序列的注意力计算**”。

可优化方向：
- **Q tokens 数量**：调整 `num_queries`，避免 Q 太少导致表达不足或太多导致噪声。
- **融合强度**：尝试 `rank_mixer_mode=ffn_only` 或增加 `hidden_mult`，观察 AUC 是否更稳。
- **序列池化策略**：目前是 MeanPool，可尝试基于时间衰减/Top-K 的池化（需改模型代码）。

### 2.6 基于 model.py 的可优化点（不改数据即可尝试）

- **序列 RoPE 缓存上限**：`RotaryEmbedding` 默认 `max_seq_len=2048` 且不扩容。若序列长度 $L$ 超过 2048，会截断 RoPE 位置编码，影响长序列表达。
  - 建议：根据数据的真实最大长度提高 `max_seq_len`，或在模型初始化时用数据统计值设置。
- **LongerEncoder 的 top-k 选择策略**：当前取“最新 top_k”，可能忽视高价值历史行为。
  - 建议：尝试 “最新 + 高权重” 混合采样，或对 `top_k` 做 20/50/100 网格对比。
- **Q 侧 RoPE 的稀疏位置**：`LongerEncoder` 在交叉注意力中用 `q_pos_indices` 对 RoPE 做 gather，若序列被压缩后位置跨度过大，RoPE 可能过度稀疏。
  - 建议：启用 `use_rope` 时配合更小的 `seq_top_k` 或更平滑的时间桶。
- **序列 ID Embedding Dropout**：`seq_id_threshold` 以上特征会额外 `Dropout`，过高阈值可能导致高基数 ID 被过度随机化。
  - 建议：结合高基数统计调 `seq_id_threshold`，观察 AUC 是否更稳。
- **NS Tokenizer 的 token 数量**：`rankmixer` 模式下 `user_ns_tokens` / `item_ns_tokens` 可以独立调节。
  - 建议：先固定 `d_model`，再在满足 $d\_model % T == 0$ 的前提下微调 token 数。
- **RankMixerBlock 的模式切换**：`full` 含 token mixing，`ffn_only` 只做逐 token FFN，后者更稳但表达弱。
  - 建议：在 AUC 波动时先尝试 `ffn_only`，再回到 `full` 做对比。
- **RoPE 基频与 head_dim**：`rope_base` 与 `head_dim` 共同决定位置编码频率覆盖范围。
  - 建议：在长序列任务上尝试更大的 `rope_base`（如 1e5），与 `num_heads` 联动测试。
- **Q 生成器的 FFN 宽度**：`MultiSeqQueryGenerator` 使用 `hidden_mult` 扩展全局信息。
  - 建议：在表达不足时增大 `hidden_mult`，在过拟合时降低。
- **序列编码器 dropout 策略**：`seq_id_emb_dropout` 是针对高基数 ID 的额外正则。
  - 建议：在 ID 过拟合时调高该 dropout，在稀疏场景则调低。
- **输出投影压缩**：`output_proj` 直接将所有 Q tokens 展平后线性压缩。
  - 建议：若多域信息过度“挤压”，可尝试增大 `d_model` 或 `num_queries`，并配合更小 `dropout_rate`。

### 2.7 按模块拆分的优化空间（与算法相关）

- **`RotaryEmbedding`**：固定 `max_seq_len` 可能截断长序列位置信息。
  - 建议：根据真实最大序列长度调大 `max_seq_len`，或在初始化时设置为统计值。
- **`RoPEMultiheadAttention`**：Q/K 旋转与门控 `W_g` 会影响注意力稳定性。
  - 建议：在过拟合时增大 `dropout` 或降低 `num_heads`；在欠拟合时反向调整。
- **`MultiSeqQueryGenerator`**：全局信息通过 `hidden_mult` 放大，可能引入噪声。
  - 建议：AUC 波动时降低 `hidden_mult` 或减少 `num_queries`。
- **`CrossAttention`**：Q 与序列对齐强度不足会削弱融合效果。
  - 建议：对比 `ln_mode=pre` 与 `post`，并配合 `num_queries` 做小网格。
- **`RankMixerBlock`**：`full` 模式表达强但约束多，`ffn_only` 更稳但弱。
  - 建议：先用 `ffn_only` 作为稳定基线，再回到 `full` 做提升。
- **`GroupNSTokenizer`**：组内池化可能掩盖高价值 fid。
  - 建议：对关键 fid 单独成组，或调大 `emb_dim` 提升容量。
- **`RankMixerNSTokenizer`**：padding 造成信息稀释，分段过多会稀疏。
  - 建议：控制 `num_ns_tokens` 与 `d_model` 的配比，避免 chunk 过小。

## 3. 训练策略优化

### 3.1 学习率与优化器
- Embedding 用 Adagrad、非 embedding 用 AdamW。
- 建议：
  - 逐步降低 `lr`（AdamW）与 `sparse_lr`（Adagrad），观察 AUC 曲线是否更平滑。
  - 可尝试更大的 `batch_size` 降低梯度噪声。

### 3.2 正则化
- 提高 `dropout_rate` 可缓解过拟合，但过高会损失表示能力。
- 建议：在验证 AUC 下降明显时，适当提升 dropout 或加入权重衰减。

### 3.3 早停与评估频率
- 训练支持 `eval_every_n_steps` 进行 step-level 验证。
- 建议：在 AUC 震荡时缩短评估间隔，避免过拟合后才发现。

## 4. 评估与诊断

### 4.1 AUC 波动分析
- 若 AUC 起伏大，优先排查：
  - 验证集切分是否稳定
  - 学习率是否过高
  - 数据是否存在分布偏移

### 4.2 NaN 预测
- `trainer.py` 会过滤 NaN，但这通常是训练不稳定信号。
- 建议：
  - 降低学习率
  - 增加梯度裁剪
  - 缩小模型规模或提高正则

## 5. 工程与算力策略

### 5.1 训练稳定性
- 多 worker + 大 batch 时可能引发 OOM 或死锁。
- 建议：逐步提升 `num_workers`，并结合 `--shm-size` 调整容器共享内存。

### 5.2 复现实验
- 固定随机种子，并记录 `train_config.json` + `schema.json` 以保证可追溯。

## 6. 基于当前字段的特征工程清单

以下基于 [industrial_data/README.md](industrial_data/README.md) 中的字段分组给出落地建议。

### 6.1 适合分桶的字段

- `timestamp`：按小时/天/周分桶，或转为“距离当前样本时间”的相对时间桶。
- `label_time`：与 `timestamp` 的差值分桶（0-1h、1-6h、6-24h、>1d）。
- `user_dense_feats_{61-66,87,89-91}`：先做统计（均值/最大/最近值）再分桶。
- 若某些 `user_int_feats_*` / `item_int_feats_*` 为计数或等级字段，可做 log 分桶。

### 6.2 适合序列衰减的字段

- `domain_a_seq_{38-46}` / `domain_b_seq_{67-79,88}` / `domain_c_seq_{27-37,47}` / `domain_d_seq_{17-26}`：
  - 取近 N 条作为短期兴趣，配合时间衰减权重。
  - 长序列做 Top-K 截断（如 20/50/100）并对比 AUC。
- 如果 `schema.json` 配置了序列时间戳列（`ts_fid`），可用 `timestamp - seq_ts` 做衰减或时间桶。

### 6.3 适合交叉与聚合的字段

- `user_id` × `item_id`：可离线统计历史交互次数或点击率（注意避免泄漏）。
- `user_int_feats_*` × `item_int_feats_*`：对同类字段做交叉（如区域/类目相关字段）。
- `user_dense_feats_*`：做均值/方差/最大/最小等聚合，形成额外连续特征。

### 6.4 序列统计特征

- 序列长度（已有 `*_len`）。
- 不同 ID 数量（去重数）。
- 最近一次出现的位置（recency）。
- 与当前 `item_id` 的匹配次数（命中统计）。

### 6.5 用户与物品侧统计特征

- **用户频率**：用户历史曝光/点击/转化计数、最近活跃天数、活跃跨度。
- **物品热度**：物品全局曝光/点击/转化计数、最近窗口内热度。
- **用户-物品交互强度**：同用户对同物品的历史交互次数、最近一次交互时间间隔。

### 6.6 目标编码与平滑统计（离线）

- 对高基数离散特征（`user_id`、`item_id`、关键 `*_int_feats_*`）构造 CTR/CVR 统计。
- 对统计特征加 Beta 平滑：$\hat{p} = \frac{c + \alpha}{n + \alpha + \beta}$，避免小样本噪声。
- 训练/验证切分严格按时间做统计，避免泄漏。

### 6.7 时间与窗口特征

- **相对时间**：`timestamp` 与 `label_time` 的差值分桶。
- **窗口统计**：过去 1h/6h/1d 的点击/转化/曝光次数。
- **周期性编码**：小时/星期几可做周期特征（sin/cos），用于捕捉周期性行为。

### 6.8 序列内结构特征

- **去重比例**：$\text{unique} / \text{len}$ 反映兴趣分散程度。
- **重访强度**：最近 K 条中重复 ID 的占比。
- **序列一致性**：序列中与当前 `item_id` 同类目/同标签的命中率。

### 6.9 Cross 特征与交互

- **用户侧 × 物品侧**：同字段或同语义字段做交叉（地域 × 类目，价格段 × 品类）。
- **序列 × 物品**：序列中是否出现同类目/同品牌/同属性的 item。
- **序列 × 时间**：序列事件的时间间隔分桶与序列统计相结合。

### 6.10 数值特征变换与归一化

- **长尾截断**：计数/金额类字段做截断或 log1p 变换。
- **分段归一**：按人群/类目分组做归一化，减少尺度差异。
- **缺失指示**：为关键 dense 特征添加 missing flag，区分真实 0 与缺失 0。

## 7. 推荐的调参起点

- 序列编码器：`transformer` / `longer` 做对比。
- `rank_mixer_mode`：`full` 或 `ffn_only`。
- `num_queries`：1 或 2（注意 T 约束）。
- `dropout_rate`：0.01 → 0.05。
- `loss_type`：优先尝试 `focal`（不平衡数据时）。
- `eval_every_n_steps`：先设较小值，观察趋势。

## 8. 总结

AUC 提升通常需要“数据分布稳定 + 模型表示增强 + 训练策略稳健”三者协同。建议从**验证集切分 + 损失函数 + 编码器选择**这三条主线开始，先做小规模对比，再逐步扩展到参数与结构的系统调优。

## 9. 结构级优化补充：围绕统一 Token 路由提升 AUC

### 9.1 Query Generator 优化

当前 Query Generator 是 AUC 提升优先级最高的模块，因为它决定模型从序列中“问什么”。

建议：

- **MeanPool → AttentionPool**：用当前 `NS/item` 上下文生成 pooling query，对序列 token 加权汇总。
- **独立 FFN → shared FFN**：使用 `domain embedding + query slot embedding + shared MLP` 降低参数量并增强共享。
- **控制 `num_queries`**：从 1、2、4 做小网格；query 过多可能增加噪声与过拟合。
- **监控 query 多样性**：计算不同 query token 的余弦相似度，避免多个 query 学成同一个槽位。

### 9.2 Sequence Pooling 与 Encoder 优化

- 短序列：`transformer` 通常表达更强。
- 长序列：优先尝试 `longer` 或 Recent-K/Top-K 预压缩。
- 超长序列：可探索 Mamba、Hyena、Linear Attention。
- 时间敏感行为：加入 time gap bucket、时间衰减 bias、最近行为加权。

### 9.3 RankMixer 与跨域融合优化

- `ffn_only` 更稳，适合作为 baseline。
- `full` 表达更强，但受 `d_model % T == 0` 约束，且更容易过拟合。
- 增加 domain gate，让模型按样本动态分配搜索、视频、电商等不同 domain 权重。
- 对 RankMixer FFN 引入轻量 MoE，可适配不同用户群体与行为分布。

### 9.4 NS Token 优化

- 关键高价值 fid 单独成 token，避免被组内 pooling 稀释。
- 弱字段或长尾字段进入 RankMixer 切块 token，减少噪声。
- 对高基数字段使用 hash/OOV/frequency cutoff，降低 embedding 过拟合。
- dense 特征拆成多个语义 token：活跃度、热度、价格/统计、时间上下文。

### 9.5 Retrieval Memory 方向

在当前架构中，Query token 已经像检索请求。进一步可加入外部 memory：

```text
query token / ns summary
  -> ANN retrieval
  -> memory tokens
  -> CrossAttention / RankMixer
  -> prediction head
```

可检索内容包括：

- 用户长期兴趣原型；
- 相似 item 的历史转化样本；
- 用户簇 prototype；
- 高转化行为路径；
- session summary。

注意必须按时间构造 memory，避免验证集或未来信息泄漏。

## 10. 评估与诊断补充

### 10.1 不要只看全局 AUC

建议同时记录：

- 按用户活跃度分桶 AUC；
- 按 item 热度分桶 AUC；
- 按 sequence length 分桶 AUC；
- 按 domain 是否为空分桶 AUC；
- 按时间窗口分桶 AUC；
- logloss / calibration error。

### 10.2 AUC 与 loss 分歧时如何判断

- **AUC 上升，logloss 下降**：最理想。
- **AUC 上升，logloss 上升**：排序更好但概率校准变差，可后处理校准。
- **AUC 下降，logloss 下降**：概率更保守但排序变差，推荐排序场景通常不优先。
- **训练 AUC 上升，验证 AUC 下降**：结构过拟合，优先降低 query 数、dropout/weight decay、简化 RankMixer。

### 10.3 推荐消融顺序

```text
固定数据切分与 seed
  -> BCE baseline
  -> focal loss
  -> transformer vs longer
  -> num_queries 1/2/4
  -> MeanPool vs AttentionPool
  -> ffn_only vs full RankMixer
  -> NS 分组调整
  -> domain gate / MoE / retrieval memory
```

每轮只改一个变量，记录 best AUC、best logloss、early stopping 步数、训练耗时和显存，避免多个改动叠加后无法定位收益来源。
