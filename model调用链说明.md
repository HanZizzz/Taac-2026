# model.py 调用链与原理说明

本文档聚焦 `model.py` 的主要调用链、关键模块原理与可优化方向，便于理解模型的特征融合与序列建模机制。

## 1. 顶层调用链（训练 / 推理）

### 1.1 训练前向（PCVRHyFormer.forward）

1) **NS tokens 构造**
- `GroupNSTokenizer` 或 `RankMixerNSTokenizer` 对 user/item 离散特征做 embedding + pooling + projection，得到 `user_ns`、`item_ns`。
- 若存在 dense 特征，分别通过 `user_dense_proj` / `item_dense_proj` 生成补充 token。
- 拼接为 `ns_tokens`。

2) **序列域 embedding**
- `_embed_seq_domain` 对每个 domain 的 side-info 序列做 embedding 拼接并投影到 `d_model`。
- 可叠加时间桶 embedding（`time_embedding`）。
- `_make_padding_mask` 根据 `seq_len` 生成 mask。

3) **Q tokens 生成**
- `MultiSeqQueryGenerator` 用 `ns_tokens` 和每个序列的 MeanPool 生成每域独立的 Q tokens。

4) **多域 HyFormer block 堆叠**
- `_run_multi_seq_blocks` 依次调用 `MultiSeqHyFormerBlock`：
  - 序列编码（`SwiGLUEncoder` / `TransformerEncoder` / `LongerEncoder`）
  - CrossAttention：Q tokens 读取序列 token
  - RankMixer：拼接所有 Q + NS 做 token mixing + FFN

5) **输出投影与分类器**
- 拼接所有 Q tokens → `output_proj` → `clsfier` 输出 logits。

### 1.2 推理前向（PCVRHyFormer.predict）

- 逻辑与 `forward` 相同，但 `_run_multi_seq_blocks(..., apply_dropout=False)`，最终输出 logits 与中间表征。

## 2. 关键模块原理与作用

### 2.1 RoPEMultiheadAttention
- 手工投影 Q/K/V，注入 RoPE，再用 `scaled_dot_product_attention`。
- `rope_on_q` 控制是否对 Q 侧使用 RoPE。
- 通过 `W_g` 的门控（sigmoid）对输出做调制。

### 2.2 CrossAttention
- Q 来自全局 token（Q tokens），K/V 来自序列 token。
- 允许非序列信息通过注意力直接影响序列表示。

### 2.3 MultiSeqQueryGenerator
- 以 `ns_tokens` + 每序列 MeanPool 作为全局信息，生成多个 Q tokens。
- 本质是“序列域的查询锚点”，用于跨注意力读序列。

### 2.4 RankMixerBlock
- 将所有域的 Q tokens 与 NS tokens 拼接后做 token mixing + FFN。
- `mode=full` 需要 `d_model % T == 0`（T 为总 token 数）。

### 2.5 序列编码器
- `SwiGLUEncoder`：无注意力，速度快但表达弱。
- `TransformerEncoder`：标准自注意力，表达强但更易过拟合。
- `LongerEncoder`：Top-K 压缩，用于长序列。

## 3. 可优化方案

### 3.1 Q tokens 与融合策略
- **调整 `num_queries`**：Q 太少→表达不足；太多→噪声和计算开销增加。
- **改进序列池化**：MeanPool 可替换为加权池化或时间衰减池化。
- **RankMixer 模式**：`full` → `ffn_only` 可减少约束与不稳定性。

### 3.2 序列建模增强
- **切换编码器**：`transformer` 与 `longer` 对比，长序列优先 `longer`。
- **启用 RoPE**：若序列位置重要，开启 `use_rope`。
- **时间桶优化**：保证时间戳列质量，避免无效 time bucket。

### 3.3 特征表示优化
- **高基数特征处理**：降低 `emb_skip_threshold` 或采用 hash/截断替代直接置零。
- **NS 分组调整**：合理划分 `ns_groups.json`，提升 token 语义一致性。

### 3.4 训练稳定性优化
- **学习率分组**：调整 `lr` 与 `sparse_lr`，避免 embedding 训练过慢或过快。
- **正则化**：适当提高 `dropout_rate` 或使用更强权重衰减。
- **损失函数**：不平衡数据可切换 `focal` 并调参。

## 4. 推荐排查顺序

1) 验证集分布是否稳定。
2) Q tokens 数量与 RankMixer 约束是否合理。
3) 编码器类型与 RoPE/时间桶是否匹配数据特点。
4) 学习率与正则化是否导致过拟合或欠拟合。

---

如需我进一步把调用链画成结构图（Mermaid）或补充“参数范围建议”，告诉我你的训练预算与显存限制即可。

## 5. 细化版调用链：从 `PCVRHyFormer.forward` 到 logits

```text
PCVRHyFormer.forward(inputs)
  ├─ user_ns = user_ns_tokenizer(inputs.user_int_feats)
  ├─ item_ns = item_ns_tokenizer(inputs.item_int_feats)
  ├─ user_dense_token = user_dense_proj(inputs.user_dense_feats)  # optional
  ├─ item_dense_token = item_dense_proj(inputs.item_dense_feats)  # optional
  ├─ ns_tokens = concat(user_ns, user_dense_token, item_ns, item_dense_token)
  ├─ for domain in seq_domains:
  │    ├─ seq_tokens = _embed_seq_domain(domain, seq_feats, seq_times)
  │    └─ seq_mask = _make_padding_mask(seq_len, max_len)
  ├─ q_tokens_list = MultiSeqQueryGenerator(ns_tokens, seq_tokens_list, seq_masks)
  ├─ for block in hyformer_blocks:
  │    ├─ seq_tokens_list = sequence_encoder(seq_tokens_list, seq_masks)
  │    ├─ q_tokens_list = CrossAttention(q_tokens_list, seq_tokens_list)
  │    └─ q_tokens_list, ns_tokens = RankMixer(concat(q_tokens_list, ns_tokens))
  ├─ out = concat(q_tokens_list)
  ├─ out = output_proj(out)
  └─ logits = classifier(out)
```

这条链路的设计意图是：用少量 `NS + Query` 压缩 token 作为信息枢纽，避免把所有非序列字段和所有历史行为都放入 full attention，从而控制复杂度并提升目标相关性。

## 6. 关键模块的“为什么”与可替代方案

### 6.1 NS Tokenizer

- **作用**：把 user/item 非序列稀疏特征压缩为全局上下文 token。
- **为什么这样用**：推荐特征多且稀疏，逐字段 token 化会使 token 数过多；NS token 是一种结构化压缩。
- **替代方案**：字段级 attention pooling、field gate、hash embedding、共享 embedding + field embedding、关键字段单独成 token。

### 6.2 Query Generator

- **作用**：根据 `ns_tokens + sequence summary` 动态生成每个序列域的 query token。
- **为什么这样用**：CVR 预测需要围绕当前 user-item 场景检索历史行为证据，动态 query 比静态 latent 更 target-aware。
- **主要风险**：独立 FFN 参数量随 `num_sequences * num_queries` 增长；MeanPool summary 可能丢失少数关键行为。
- **替代方案**：shared MLP + domain/query embedding、attention pooling、top-k pooling、recent-k pooling、Mamba/RWKV state pooling。

### 6.3 Sequence Encoder

- **作用**：把原始行为 token 更新为上下文化行为 token。
- **为什么这样用**：Cross Attention 读取上下文化行为，比读取孤立事件更能捕捉兴趣演化。
- **替代方案**：Mamba/Hyena/Linear Attention、session-aware encoder、多尺度 recent/long/top-k encoder。

### 6.4 Cross Attention

- **作用**：让少量 query token 从长序列中读取相关证据。
- **为什么这样用**：复杂度约为 `O(num_queries * seq_len)`，比所有 token full attention 更适合长序列推荐。
- **替代方案**：top-k sparse attention、带时间衰减 bias 的 attention、domain-specific head、可解释 attention 导出。

### 6.5 RankMixer

- **作用**：在所有 domain 的 query token 与 NS token 之间做跨域全局融合。
- **为什么这样用**：多域行为不是孤立信号，RankMixer 在压缩 token 层面完成跨域路由，成本低于跨原始序列 full attention。
- **替代方案**：domain gate、Sparse MoE FFN、shared memory tokens、LayerScale/residual scaling。

### 6.6 Output Head

- **作用**：把融合后的 query representation 转为 CVR logit。
- **为什么这样用**：query token 已经过 NS 条件化、序列检索和跨域融合，是最目标相关的最终表征。
- **替代方案**：拼接 NS summary、多任务 head、校准 head、不确定性 head。

## 7. 优化空间优先级

1. **Query token 质量**：优先尝试 attention pooling、shared FFN、`num_queries` 小网格。
2. **长序列效率**：对比 `transformer`、`longer`，进一步可尝试 Mamba/Linear Attention。
3. **跨域交互**：增加 domain gate 或 shared memory tokens。
4. **训练稳定性**：学习率分组、梯度裁剪、dropout、分桶 AUC 诊断。
5. **外部记忆**：引入用户长期兴趣原型、item 相似样本、ANN/Faiss retrieval memory。
6. **辅助目标**：加入 CTR/CVR 多任务、InfoNCE、masked behavior modeling。
