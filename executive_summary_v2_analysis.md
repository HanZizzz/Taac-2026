# 📋 Summary Token v2 分析 - 执行摘要与决策矩阵

**分析完成时间**: 2026-05-17  
**分析阶段**: Phase 4.3 (完成) → Phase 5 (启动)  
**当前决策状态**: 🔴 关键决策点 - 需要 Baseline 对照实验

---

## 📊 一页纸总结

### 问题定义
之前的 summary token v1 在所有 seed 上都导致 AUC 下降 3-5%。是否通过轻量化（v2）可以改善，还是整个方案本身不适合？

### 方案演进
```
v0 (baseline): 无 summary token
  ↓
v1 (原始方案): 66-dim engineered_dense → d_model projection
  → 结果：AUC 下降，无法接受 ❌
  ↓
v2 (轻量化): 16 特征 (core) 或 8 特征 (topk) → 小 bottleneck → 门控
  → 结果：相对 v1 改善，但相对 baseline 未知 ⚠️
```

### 当前数据点 (v2)

**实验规模**: 4 次完整训练
- seed=42 core mode
- seed=42 topk mode
- seed=52 topk mode
- seed=62 topk mode

**关键指标**:
| 指标 | 值 | 解读 |
|------|-----|------|
| 平均 AUC (topk, 3 seeds) | 0.7509 | 中等水平 |
| AUC 标准差 | ±0.0114 | 1.5% 相对差异（较大） |
| 平均 LogLoss | 0.3283 | 校准能力中等 |
| 最优 Epoch | 5-6 | 早期就达到最优 |
| 过拟合幅度 | 6.6% (seed_52) | 明显过拟合 |

### 关键发现

#### 发现 1: Core vs TopK 完全等效 ✅
- seed_42 core: 0.7386 AUC
- seed_42 topk: 0.7386 AUC
- **含义**: 后 8 个特征是噪声，前 8 个特征足够
- **决定**: 推荐用 topk=8 (参数更少，推理更快)

#### 发现 2: Seed 间差异 ±1.5% ⚠️
```
seed_42: 0.7386 (最低)
seed_52: 0.7528 (+1.92%)
seed_62: 0.7613 (+3.07%, 最高)
```
- **含义**: 模型对初始化敏感，结果波动大
- **问题**: 相对高差异使得难以判断真实效果

#### 发现 3: 稳定性问题 🔴
最严重的是 seed_52 的不稳定性：
```
Epoch 2: 0.7490 AUC ← 好
Epoch 3: 0.7357 AUC
Epoch 4: 0.7528 AUC ← 最好 (+0.0171 反弹)
Epoch 5: 0.7035 AUC ← 暴跌 (-0.0493, -6.6%)
Epoch 6: 0.6638 AUC ← 继续跌
```
**根本原因**: 学习率过高、embedding 重初始化、或梯度更新不稳定

#### 发现 4: v2 vs Baseline 真实增益未知 ❓
- v2 平均 AUC: 0.7509
- 推估 baseline: 0.76-0.77 (基于历史数据)
- **推估差异**: -1% 到 0%
- **问题**: 仅凭推测无法决策
- **必做**: 运行真实 baseline 对照

#### 发现 5: LogLoss 校准能力中等
- v2 LogLoss: 0.328 (平均)
- AUC 与 LogLoss 不完全相关（seed_62 AUC 最好但 LogLoss 不是最优）
- **含义**: v2 改善了排序性能但校准性能一般

---

## 🎯 关键决策点

### 问题 1: 要不要继续迭代 v2？

**需要的信息**: v2 vs baseline 的真实 AUC 差异

**决策矩阵**:
```
┌─────────────────────────────────────────────────────┐
│ Baseline vs v2 AUC 对比                              │
├──────────────────┬──────────────────┬────────────────┤
│ v2 > baseline    │ v2 ≈ baseline    │ v2 < baseline  │
│ (+>0.5%)         │ (±0.5%)          │ (->-0.5%)      │
├──────────────────┼──────────────────┼────────────────┤
│ 路线 A:          │ 路线 C:          │ 路线 B:        │
│ 继续优化 v2      │ 折衷/等待        │ 转向 B 方案    │
│ ✓ 做正则化改进   │ ⚠️ 需要平衡     │ ❌ 放弃 v2    │
│ ✓ 学习式特征权重 │    决策          │ → Dense Fusion │
│ 目标: 0.77+ AUC  │ 或 v2 但更稳定   │ 或回到 baseline│
└──────────────────┴──────────────────┴────────────────┘
```

**行动**: 运行 Baseline 对照 (本周内完成)
```bash
./run.sh --seed 42/52/62 --engineered_dense_dim 0
```
预期 1-2 小时完成。

---

### 问题 2: 如果 v2 无改善，为什么会这样？

**可能原因分析**:

| 原因 | 症状 | 检验方式 |
|------|------|----------|
| 特征冗余 | v2 AUC ≈ baseline | ✓ 当前已观察 (core/topk 等效) |
| 模型容量不足 | AUC 低于预期 | 对比 d_model 更大的版本 |
| 特征噪声 | LogLoss 没改善 | 检查 LogLoss 数据 |
| 优化不稳定 | AUC 曲线锯齿 | ✓ 当前已观察 (seed_52) |
| 序列信息充足 | v2 无增益 | 尝试移除 v2，AUC 无变化 |

---

### 问题 3: 如果进入路线 B (Dense Fusion)，怎么改？

**Dense Fusion 核心思想**:
```python
# 旧方式 v2: 新增 token
summary_token = proj(engineered_dense)
output = [ns_tokens..., summary_token]  # T 增加

# 新方式 Dense Fusion: 投影后相加
summary_emb = proj(engineered_dense)  # (B, d_model)
output = ns_tokens + summary_emb  # 逐元素相加，T 不变
```

**优势**:
- 不新增序列长度 → 不改变 token budget → 公平对照
- 减少序列复杂度 → 降低过拟合风险
- 更简洁的融合方式

**代码改动**: ~20 行 (model.py forward() 方法)

**预期效果**: 相对 baseline 应该接近或略好（因为特征本身有价值）

---

## 📈 三条路线详细规划

### 路线 A: 继续优化 v2 (如果 v2 > baseline by +0.5%)

```
Week 1: P1 改进 (本周)
  ├─ 增加 dropout: 0.01 → 0.05
  ├─ 调整 patience: 5 → 2
  ├─ 诊断 embedding 重初始化
  └─ 跑 3 seed 验证

Week 2: P2 改进 (下周)
  ├─ 尝试 bottleneck_dim: 4 → 8
  ├─ 尝试启用 gate
  ├─ 学习式特征权重 (如果时间允许)
  └─ 跑 5 seed 确保稳定性

Week 3: 决策
  ├─ 若 AUC 0.77+, std <1% → 进入预训练阶段
  └─ 若 AUC 0.74-0.76 → 与 baseline 持平，考虑放弃
```

**预期投入**: 15-20 小时  
**风险**: 可能需要修改多个超参数

---

### 路线 B: 转向 Dense Fusion (如果 v2 ≤ baseline)

```
Week 1: 方案设计 & 代码改动 (1-2 小时)
  ├─ 修改 model.py forward()
  ├─ 调整融合逻辑（相加而非新增 token）
  └─ smoke test

Week 1: 实验对照 (1-2 小时)
  ├─ 跑 Dense Fusion 3 seed 对照
  ├─ 对比与 baseline
  └─ 生成结果报告

Week 2: 评估
  ├─ 若 Dense Fusion > baseline by +0.5% → 采用
  ├─ 若 Dense Fusion ≈ baseline → 保持 baseline
  └─ 若都无改善 → 考虑其他特征工程方向
```

**预期投入**: 5-10 小时  
**风险**: 低（改动简单，快速验证）

---

### 路线 C: 等待 & 平衡 (如果 v2 ≈ baseline)

```
短期: 保留 baseline，继续其他特征工程方向
  ├─ 尝试序列特征工程
  ├─ 尝试新的交叉特征
  └─ 监控总体 AUC 进展

中期: 若有其他改进达到 +1% → 重新对比 v2
后期: 集成方案 (ensemble v2 + 其他改进)
```

---

## 🚀 立即行动 (本周)

### Step 1: 运行 Baseline 对照 (1-2 小时)
```bash
cd /workspace
for seed in 42 52 62; do
  echo "=== Baseline seed=$seed ==="
  ./run.sh --seed $seed --d_model 64 --rank_mixer_mode ffn_only \
    --engineered_dense_dim 0 --batch_size 64
done
```

### Step 2: 数据收集 (15 分钟)
从日志提取：
- 每个 seed 的最优 AUC、LogLoss、所在 epoch

### Step 3: 生成对比报告 (30 分钟)
填表、计算平均值/标准差、生成可视化图表

### Step 4: 决策 (15 分钟)
根据对比结果，选择路线 A/B/C

### Step 5: 执行后续行动
根据选择的路线执行相应改进

---

## 📋 核查清单

- [ ] Baseline 对照 3 seeds 运行完毕
- [ ] 数据提取：AUC, LogLoss, early stop epoch
- [ ] 对比表格生成
- [ ] 决策框架评估（v2 vs baseline 差异确认）
- [ ] 选定路线 (A/B/C)
- [ ] 路线相关的改进清单准备
- [ ] 后续实验时间表确认

---

## 📞 问题排查

**问题**: 怎样快速判断结果是否显著？  
**回答**: 
- 如果 v2 mean AUC - baseline mean AUC > 标准差，则显著
- 例：若 baseline 0.760 ± 0.003，v2 0.7509 ± 0.0114
  - 差异 = 0.7509 - 0.760 = -0.0091 = -1.2%
  - 相对 baseline std (0.003)，差异 3 倍多 → 显著
  - **结论**: v2 显著劣于 baseline → 路线 B

---

## 🎓 已学到的经验

1. **特征工程的陷阱**: 额外特征不一定有帮助，可能引入噪声
2. **A/B 对照的重要性**: 不能凭 inference 做决策，必须实验
3. **多 seed 的价值**: 单 seed 结果可能是幸运，多 seed 才能看到模式
4. **稳定性 vs 性能**: 有时候稳定性比单点性能更重要
5. **决策框架**: 提前定义"成功"的标准，避免事后诠释

---

**下一步**: 🔴 运行 Baseline 对照实验  
**预期时间**: 本周内 (1-2 小时运行 + 1 小时分析)  
**关键里程碑**: 周五完成决策，周一启动路线对应的改进

有任何问题，请查看 `Phase5_baseline_contrast.md` 获取详细命令和故障排除。
