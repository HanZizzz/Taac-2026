# Phase 5 行动清单与命令速查

**当前阶段**: 立即启动 Baseline 对照实验  
**目标**: 确认 v2 summary token 相对 baseline 的真实增益  
**预期完成时间**: 本周内 (约 1-2 小时运行 + 1 小时分析)

---

## 🔴 P0 优先级：Baseline 对照 (3 seeds × 2 configs = 6 次运行)

### 配置对比
| 配置 | 类型 | 目的 | 预期 AUC |
|------|------|------|----------|
| `engineered_dense_dim 0` | Baseline | 无 summary token | ~0.76 (推测) |
| `engineered_dense_dim 66 + engineered_summary_mode topk + engineered_summary_topk 8` | v2 | 轻量版 summary token | 0.7509 (已测) |

---

## ✅ 运行命令汇总

### Baseline (engineered_dense_dim=0) 运行

**Seed 42:**
```bash
./run.sh --seed 42 --d_model 64 --rank_mixer_mode ffn_only \
  --engineered_dense_dim 0 \
  --batch_size 64
```

**Seed 52:**
```bash
./run.sh --seed 52 --d_model 64 --rank_mixer_mode ffn_only \
  --engineered_dense_dim 0 \
  --batch_size 64
```

**Seed 62:**
```bash
./run.sh --seed 62 --d_model 64 --rank_mixer_mode ffn_only \
  --engineered_dense_dim 0 \
  --batch_size 64
```

---

### v2 Summary Token (已完成，对比参考)

**Seed 42:**
```bash
./run.sh --seed 42 --d_model 64 --rank_mixer_mode ffn_only \
  --engineered_dense_dim 66 \
  --engineered_summary_mode topk \
  --engineered_summary_topk 8 \
  --engineered_summary_bottleneck_dim 4 \
  --no_engineered_summary_gate \
  --batch_size 64
```

**Seed 52:**
```bash
./run.sh --seed 52 --d_model 64 --rank_mixer_mode ffn_only \
  --engineered_dense_dim 66 \
  --engineered_summary_mode topk \
  --engineered_summary_topk 8 \
  --engineered_summary_bottleneck_dim 4 \
  --no_engineered_summary_gate \
  --batch_size 64
```

**Seed 62:**
```bash
./run.sh --seed 62 --d_model 64 --rank_mixer_mode ffn_only \
  --engineered_dense_dim 66 \
  --engineered_summary_mode topk \
  --engineered_summary_topk 8 \
  --engineered_summary_bottleneck_dim 4 \
  --no_engineered_summary_gate \
  --batch_size 64
```

---

## 📊 预期结果收集与分析

### 步骤 1: 运行 Baseline (3 seeds)

```bash
# 连续运行 3 个 seed，每个约 1-2 分钟
for seed in 42 52 62; do
  echo "Running baseline seed=$seed"
  ./run.sh --seed $seed --d_model 64 --rank_mixer_mode ffn_only \
    --engineered_dense_dim 0 --batch_size 64
  sleep 30
done
```

**预期输出**: 3 个 seed 的完整训练日志，记录每个 seed 的最优 AUC @ epoch

### 步骤 2: 数据收集

从运行日志中提取并填表：

| Seed | Config | Best AUC | @ Epoch | Best LogLoss | Early Stop |
|------|--------|----------|---------|--------------|-----------|
| 42   | baseline | ? | ? | ? | ? |
| 52   | baseline | ? | ? | ? | ? |
| 62   | baseline | ? | ? | ? | ? |
| 42   | v2 topk | 0.7386 | 6 | 0.3351 | 11 |
| 52   | v2 topk | 0.7528 | 4 | 0.3296 | 9 |
| 62   | v2 topk | 0.7613 | 5 | 0.3302 | 10 |

### 步骤 3: 生成对比统计

```
Baseline (engineered_dense_dim=0) 统计:
  seed_42: ? AUC
  seed_52: ? AUC
  seed_62: ? AUC
  ├─ 平均值: ? AUC
  ├─ 标准差: ?
  └─ 结论: [待填]

v2 Summary Token (topk=8) 统计:
  seed_42: 0.7386 AUC
  seed_52: 0.7528 AUC
  seed_62: 0.7613 AUC
  ├─ 平均值: 0.7509 AUC
  ├─ 标准差: 0.0114
  └─ 结论: 已测

Baseline vs v2 差异:
  seed_42: (v2 - baseline) = ? 
  seed_52: (v2 - baseline) = ?
  seed_62: (v2 - baseline) = ?
  ├─ 平均差异: ?
  ├─ 改善比例: ?%
  └─ 统计显著性: [显著 / 中等 / 不显著]
```

---

## 🎯 结果解读与决策

### 决策框架

#### 情景 A: 基线 AUC > 0.75，v2 vs baseline ≤ 0%

**结论**: v2 summary token 相对 baseline 无改善甚至有害

**决策**: 转向 **路线 B**
- 放弃"额外 token"方案
- 改用 **Dense Fusion** 方案：
  - engineered_dense 投影后 **相加** 到 NS token embedding
  - 不新增序列长度，减少过拟合风险
  - 代码改动：在 model.py forward() 中修改融合方式

**示例代码 (Dense Fusion)**:
```python
# 原始方式（当前）: 新增 token
ns_parts = [ns_token1, ns_token2, ..., summary_token]  # T 增加

# Dense Fusion 方式: 投影后相加
summary_proj = self.engineered_proj(engineered_dense_feats)  # (B, d_model)
ns_parts = [ns_token1 + summary_proj, ns_token2, ...]  # T 不变
```

---

#### 情景 B: 基线 AUC ~0.76-0.77，v2 vs baseline > +0.5%

**结论**: v2 summary token 有显著改善

**决策**: 转向 **路线 A** - 继续优化 v2
- P1 (立即): 增加正则化 + 激进早停 + 诊断不稳定性
- P2 (后续): 学习式特征权重 + 更大 bottleneck_dim
- 目标: 达到 0.77+ AUC，种子方差 <1%

**改进清单**:
1. 增加 dropout (0.01 → 0.05)
2. 调整 patience (5 → 2)
3. 诊断 embedding 重初始化逻辑 (是否导致 Epoch 4-5 跳变)
4. 尝试 bottleneck_dim 8 (from 4)
5. 尝试启用 gate (--engineered_summary_gate)

---

#### 情景 C: 基线 AUC ~0.74-0.75，v2 vs baseline -0.5% to +0%

**结论**: v2 与 baseline 基本相当，稳定性有所下降

**决策**: 折衷方案
- 如果关注 **排序性能 (AUC)**：保留 baseline（v2 无优势）
- 如果关注 **校准性能 (LogLoss)**：比较两者 LogLoss，选较优者
- 可选：尝试混合方案（v2 + 更强正则化，看能否稳定性改善）

---

## 📈 预期时间表

| 阶段 | 任务 | 预期时间 |
|------|------|----------|
| 执行 | 运行 3 seeds baseline | 20-30 分钟 |
| 分析 | 提取数据、生成对比表 | 10-15 分钟 |
| 决策 | 阅读对比结果、选择路线 | 5 分钟 |
| **总计** | | **35-50 分钟** |

---

## 🔧 故障排除

### 问题 1: `./run.sh: command not found`
**解决**:
```bash
cd /path/to/workspace  # 确保在项目根目录
chmod +x run.sh
./run.sh --seed 42 ...
```

### 问题 2: `CUDA out of memory`
**解决**:
```bash
# 减少 batch_size
./run.sh --seed 42 --batch_size 32 ...

# 或减少 d_model
./run.sh --seed 42 --d_model 32 --batch_size 64 ...
```

### 问题 3: 日志中找不到最优 AUC
**查看命令**:
```bash
# 在日志中搜索 "Best AUC"
grep "Removed old best_model" /logs/train_*.log

# 或查看 TensorBoard events
tensorboard --logdir tfevents
```

---

## 📝 记录模板

将实验结果填入此表并保存到 `baseline_vs_v2_comparison.md`:

```markdown
# Baseline vs v2 Summary Token 对比报告
生成时间: [DATE]

## 实验配置
- d_model: 64
- rank_mixer_mode: ffn_only
- batch_size: 64

## 结果汇总

### Baseline (engineered_dense_dim=0)
| Seed | Best AUC | @ Epoch | Best LogLoss | Early Stop |
|------|----------|---------|--------------|-----------|
| 42   | [填入]   | [填入]  | [填入]       | [填入]    |
| 52   | [填入]   | [填入]  | [填入]       | [填入]    |
| 62   | [填入]   | [填入]  | [填入]       | [填入]    |

平均 AUC: [计算]
标准差: [计算]

### v2 Summary Token (topk=8)
| Seed | Best AUC | @ Epoch | Best LogLoss | Early Stop |
|------|----------|---------|--------------|-----------|
| 42   | 0.7386   | 6       | 0.3351       | 11        |
| 52   | 0.7528   | 4       | 0.3296       | 9         |
| 62   | 0.7613   | 5       | 0.3302       | 10        |

平均 AUC: 0.7509
标准差: 0.0114

### 对比与结论
v2 vs Baseline 平均差异: [计算]
改善百分比: [计算]%

**决策**: [选择路线 A/B/C 并说明理由]
```

---

## ✨ 后续路线详解

### 如果进入路线 A (继续优化 v2)

**P1 改进** (本周执行):
```bash
# 尝试配置 1: 增加 dropout
./run.sh --seed 42/52/62 --engineered_dense_dim 66 \
  --dropout_rate 0.05 \
  --engineered_summary_mode topk --engineered_summary_topk 8 \
  --engineered_summary_bottleneck_dim 4 \
  --no_engineered_summary_gate

# 尝试配置 2: 更激进的早停
./run.sh --seed 42/52/62 --engineered_dense_dim 66 \
  --patience 2 \
  --engineered_summary_mode topk --engineered_summary_topk 8 \
  --engineered_summary_bottleneck_dim 4 \
  --no_engineered_summary_gate

# 尝试配置 3: 启用 gate
./run.sh --seed 42/52/62 --engineered_dense_dim 66 \
  --engineered_summary_mode topk --engineered_summary_topk 8 \
  --engineered_summary_bottleneck_dim 4 \
  --engineered_summary_gate
```

**P2 改进** (下周执行，如果 P1 有改善):
```bash
# 尝试更大的瓶颈维度
./run.sh --seed 42/52/62 --engineered_dense_dim 66 \
  --engineered_summary_mode topk --engineered_summary_topk 8 \
  --engineered_summary_bottleneck_dim 8 \
  --engineered_summary_gate
```

---

### 如果进入路线 B (Dense Fusion)

**代码改动清单**:
1. 修改 `model.py` forward() 方法，改为相加融合而不是新增 token
2. 运行同样的 3 seed 对照，与新 baseline 对比
3. 生成"Dense Fusion vs 旧 baseline"对比报告

**预期代码行数**: ~20 行改动

---

## 🎓 学习要点

通过这个实验，你将理解：

1. **为什么需要 baseline 对照**: 了解特征改动的真实增益
2. **A/B 实验的设计原则**: 同一个维度改变，控制其他变量不变
3. **统计显著性**: 多 seed 的平均和方差如何判断改善的真实性
4. **决策下沉**: 基于数据做出是"继续迭代"vs"改方案"的选择

---

**下一步**: 点击运行命令，开始 Baseline 对照实验！

有任何问题或进度更新，请在 issue / discussion 中反馈。
