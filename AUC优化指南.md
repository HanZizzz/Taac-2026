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

### 2.5.1 Cross Attention / LongerEncoder / RankMixer 专项优化空间

这三块是当前模型最像“检索式推荐模型”的部分，建议单独做结构化消融。

#### 2.5.1.1 调用链定位

- `PCVRHyFormer.forward()` 先构造 NS tokens、序列 tokens 和 padding mask，再调用 `MultiSeqQueryGenerator` 生成每个序列域的 Q tokens。
- `_run_multi_seq_blocks()` 会逐层调用 `MultiSeqHyFormerBlock`；每层内部先做序列编码，再用 Q 对对应序列做 `CrossAttention`，最后把所有 Q 与 NS tokens 拼接进 `RankMixerBlock`。
- 因此该模块的优化目标不是单纯“加深 Transformer”，而是提升：
  1. Q token 的检索意图质量；
  2. 长序列压缩后的信息保真度；
  3. 多序列域与 NS tokens 的跨域融合效率。

#### 2.5.1.2 Q tokens 优化

- **当前方式**：NS tokens 展平后与当前序列 MeanPool 拼接，再由每个序列域、每个 query 独立 FFN 生成 Q。
- **风险**：MeanPool 会平均掉少数高价值行为；多个 query 可能学成重复语义；独立 FFN 参数量随 `num_sequences * num_queries` 增长。
- **建议**：
  - 先调 `num_queries=1/2/4`，同时检查 `rank_mixer_mode=full` 下的 $d\_model % T$ 约束。
  - 将 MeanPool 升级为 time-decay pooling：近期行为权重大，旧行为权重小。
  - 进一步可做 item-aware / ns-aware attention pooling：用候选 item 或 NS summary 作为 query 生成序列摘要。
  - 若参数量过大，可改为 shared FFN + sequence embedding + query type embedding。

#### 2.5.1.3 Cross Attention 优化

- **当前方式**：Q tokens 作为 Query，sequence tokens 作为 Key/Value，输出仍为少量 Q tokens，相当于从历史行为中检索相关证据。
- **优势**：复杂度为 $O(Nq \cdot L)$，比把所有 token 做 full self-attention 更适合长序列。
- **建议**：
  - 用 `num_queries` 控制检索槽位数量：1 更稳，2/4 更适合多兴趣但更容易冗余。
  - 对 CrossAttention 增加诊断：attention entropy、不同 query 的关注位置重叠率、各序列域 Q norm。
  - 如发现 Q 过度集中在最近 token，可加入 attention dropout 或 time bucket smoothing。
  - 如发现 Q 无法聚焦，可增强 Q 生成器，而不是盲目加深序列 encoder。

#### 2.5.1.4 LongerEncoder 优化

- **当前方式**：当 `L > seq_top_k` 时，用最新 `seq_top_k` token 作为 Query，对全量序列做 Cross Attention 压缩；后续长度不超过 `seq_top_k` 后转为 self-attention 精炼。
- **优势**：第一层完成“长历史读取”，后续层只在短序列上演化，显著降低复杂度。
- **风险**：latest top-k 是强 recency 先验，但可能丢失远期高价值行为。
- **建议**：
  - 对 `seq_top_k=20/50/100/200` 做网格，对比整体 AUC 与长序列用户分桶 AUC。
  - 将 latest top-k 升级为 hybrid selection：`latest top_m + relevance top_n + diversity top_r`。
  - relevance scorer 可使用轻量 MLP：输入 sequence token、item/NS summary、time bucket，输出保留分数。
  - 如果启用 RoPE，联动测试 `rope_base=10000/100000`，避免超长位置旋转过快。

#### 2.5.1.5 RankMixer 优化

- **当前方式**：所有 decoded Q 与 NS tokens 拼接后，`RankMixerBlock(full)` 通过 `view(B,T,T,d_sub) + transpose` 做参数无关 token mixing，再接共享 FFN。
- **优势**：几乎零成本实现跨 token 子空间交换，适合 token 数不大但交互重要的推荐场景。
- **约束**：`full` 模式要求 $d\_model % T == 0$，其中 $T=num\_queries*num\_sequences+num\_ns$。
- **建议**：
  - 先用 `rank_mixer_mode=ffn_only` 建立稳定基线，再切 `full` 观察收益。
  - 若 `full` 失败或 AUC 波动，优先调整 `num_queries`、`user_ns_tokens`、`item_ns_tokens` 或 `d_model`，保证合理的 `d_sub=d_model/T`。
  - 如果固定置换表达不足，可尝试 `RankMixer + low-rank token mixer`：在 RankMixer 后追加小型低秩 attention 或 gated token mixing。
  - 记录 RankMixer 前后 token cosine similarity，若混合后 token 过度同质化，可提高 dropout 或降低 token 数。

#### 2.5.1.6 推荐实验顺序

1. **配置级消融**：`seq_encoder_type in {transformer,longer,swiglu}`、`rank_mixer_mode in {ffn_only,full}`、`seq_top_k in {20,50,100}`。
2. **Q 质量优化**：MeanPool → time-decay pooling → item-aware attention pooling。
3. **Top-K 策略优化**：latest top-k → latest + relevance hybrid top-k。
4. **融合层优化**：RankMixer full → RankMixer + gated/low-rank learnable mixer。
5. **长期方向**：外部 memory bank / ANN retrieval，把超长历史先离线召回为少量 memory tokens，再接入现有 CrossAttention。

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


### 3.4 SwiGLU / 双优化器 / Loss 的专项优化空间

这一节专门补充 `seq_encoder_type=swiglu`、Embedding Adagrad、Dense AdamW、BCE/Focal Loss 这几部分的调优空间。

#### 3.4.1 `seq_encoder_type=swiglu` 的优化空间

`swiglu` 的优势是轻量、稳定、无 `L^2` attention 成本；风险是序列 token 之间缺少显式交互。

建议按以下路径优化：

- **先把 `swiglu` 当吞吐基线**：在相同 `batch_size`、`d_model`、`num_layers` 下记录 samples/sec、显存峰值、AUC，作为和 `transformer` / `longer` 对比的基准。
- **调 `hidden_mult`**：
  - 欠拟合：`hidden_mult=4 -> 6/8`，增强通道非线性。
  - 过拟合或延迟敏感：`hidden_mult=4 -> 2/3`，降低容量和计算。
- **调 `num_layers`**：SwiGLU 单层表达弱于 attention，可以尝试用更多轻量层弥补；但层数增加会放大残差噪声，建议从 1/2/3 层小网格开始。
- **调 `dropout_rate`**：
  - AUC 训练高、验证低：提高到 `0.03~0.08`。
  - AUC 整体上不去：降低到 `0.0~0.01`，避免轻量模型被过度正则。
- **与 `num_queries` 联动**：SwiGLU 不做序列内部 attention，因此 Q tokens 的读取能力更关键。可尝试 `num_queries=1/2/4`，但要同步检查 RankMixer 的 token 数约束。
- **与时间特征联动**：SwiGLU 本身不建模相对位置，建议保留 `use_time_buckets`；若切到 `transformer/longer` 再比较 `use_rope`。
- **可改造方向**：在 `SwiGLUEncoder` 内加入 depthwise 1D conv、time-decay gate、target-aware gate，让轻量编码器具备局部序列交互或候选 item 感知能力。

推荐实验矩阵：

```text
swiglu + hidden_mult=2/4/6
swiglu + num_layers=1/2/3
swiglu + num_queries=1/2/4
swiglu + dropout_rate=0.0/0.01/0.03/0.05
```

#### 3.4.2 Embedding Adagrad 的优化空间

Embedding 的主要矛盾是“高频 ID 稳定”和“长尾 ID 学得动”。当前用 Adagrad 是合理默认，但仍有优化空间：

- **调 `sparse_lr`**：
  - AUC 抖动、embedding norm 变大：降低 `sparse_lr`，如 `0.05 -> 0.02 -> 0.01`。
  - 长尾特征无收益、训练 AUC 上升慢：尝试 `0.05 -> 0.08/0.1`。
- **谨慎使用 `sparse_weight_decay`**：Embedding 通常不建议大 weight decay；如果要加，先从很小值开始。
- **监控高基数特征**：`emb_skip_threshold` 过低会直接损失高价值 ID；显存允许时优先“hash/截断”而不是跳过。
- **重初始化策略**：`reinit_sparse_after_epoch` 和 `reinit_cardinality_threshold` 适合作为去噪实验，不建议在没有对照实验时默认打开。
- **更工程化的替代**：超大 embedding 可考虑 row-wise Adagrad、TorchRec/FBGEMM optimizer、分层 embedding table、冷热 ID 分表。

建议实验顺序：

```text
固定 dense lr，先扫 sparse_lr
再评估 emb_skip_threshold
最后测试 sparse reinit 或 row-wise optimizer
```

#### 3.4.3 Dense AdamW 的优化空间

Dense 参数包括 projection、attention、SwiGLU/FFN、RankMixer、output head。AdamW 是合理默认，但建议继续细化：

- **调 `lr`**：AUC 抖动优先降 dense lr；欠拟合且 loss 下降慢可小幅升 lr。
- **加入 warmup**：训练初期不稳时，建议加入 `3%~10%` steps 的 linear warmup。
- **加入 decay schedule**：固定 lr 后期容易震荡，可用 cosine decay 或 step decay。
- **参数组 weight decay**：LayerNorm/bias 通常不做 weight decay；Linear/FFN/Attention 可保留 weight decay。
- **Head 使用更大学习率**：如果 backbone 已稳定但分类边界不够，可让 classifier/output head 使用 `1.5x~3x` lr。
- **EMA/SWA**：验证 AUC 波动大时，可尝试 EMA 权重用于评估和保存 best checkpoint。

推荐实验顺序：

```text
lr=1e-4/3e-4/5e-4
warmup=0/3%/5%
cosine decay vs fixed lr
head lr multiplier=1/2
```

#### 3.4.4 BCE / Focal Loss 的优化空间

Loss 选择需要和样本比例、负采样方式、评估指标一起看。

- **BCE 作为稳定基线**：任何结构改动前先跑 BCE，确认模型和数据 pipeline 没有问题。
- **Focal 主要用于 hard sample**：正负极不平衡、easy negative 过多时再打开。
- **调 `focal_gamma`**：
  - `gamma=0` 近似 BCE；
  - `gamma=1` 温和关注 hard sample；
  - `gamma=2` 常用默认；
  - `gamma>=3` 可能过度关注噪声样本。
- **调 `focal_alpha`**：当前实现中 positive 权重为 `alpha`，negative 权重为 `1-alpha`；原始正样本极少时要尝试 `alpha>0.5`，负采样后正样本比例较高时可尝试 `alpha<0.5`。
- **观察 AUC 与 logloss 的 trade-off**：Focal 可能提升 AUC，但损害概率校准；如果线上需要 calibrated CVR，需要额外做 calibration。
- **更进一步**：尝试 `pos_weight BCE`、sample-weighted BCE、pairwise AUC surrogate、listwise loss 或 CTR->CVR 多任务辅助。

推荐实验矩阵：

```text
bce
focal: gamma=1/2/3, alpha=0.25/0.5/0.75
pos_weight BCE: pos_weight=neg_count/pos_count 的平滑版本
```

#### 3.4.5 联合调参优先级

不要一次同时改太多。建议优先级：

1. **数据与验证集稳定性**：先保证 AUC 可信。
2. **BCE + swiglu 吞吐基线**：建立速度/显存/AUC baseline。
3. **编码器对比**：`swiglu` vs `transformer` vs `longer`。
4. **优化器学习率**：先 `lr`，再 `sparse_lr`，最后 schedule。
5. **Loss 网格**：BCE 稳定后再试 Focal。
6. **结构改造**：Conv-SwiGLU、target-aware Q、pairwise/listwise loss。

一个推荐的低成本起点：

```text
--seq_encoder_type swiglu
--hidden_mult 4
--dropout_rate 0.01
--loss_type bce
--lr 3e-4
--sparse_lr 0.05
```

如果验证 AUC 稳但偏低：

```text
增加 num_queries / hidden_mult
尝试 focal gamma=1~2
切 transformer 或 longer 对比
```

如果验证 AUC 抖动明显：

```text
降低 lr / sparse_lr
提高 dropout_rate
缩短 eval_every_n_steps
尝试 rank_mixer_mode=ffn_only
```

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
