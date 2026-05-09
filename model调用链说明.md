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
