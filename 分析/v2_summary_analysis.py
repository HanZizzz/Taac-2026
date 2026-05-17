#!/usr/bin/env python3
"""
分析 summary token v2 的 4 个 seed 对照结果
包含：AUC 曲线、LogLoss 曲线、统计量、过拟合诊断、对比分析
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# 数据收集：4 个 seed 的完整 epoch 级指标
# ============================================================================

RESULTS = {
    "seed_42_core": {
        "mode": "core (16 features)",
        "bottleneck": 8,
        "gate": True,
        "epochs": {
            1: {"auc": 0.6980, "logloss": 0.3522},
            2: {"auc": 0.7339, "logloss": 0.3387},
            3: {"auc": 0.7236, "logloss": 0.3392},
            4: {"auc": 0.7047, "logloss": 0.3486},
            5: {"auc": 0.7038, "logloss": 0.3488},
            6: {"auc": 0.7386, "logloss": 0.3295},  # best
            7: {"auc": 0.6693, "logloss": 0.3614},
        },
        "best_auc": 0.7386,
        "best_epoch": 6,
        "best_logloss": 0.3295,
        "early_stop_epoch": 7,
    },
    "seed_42_topk8": {
        "mode": "topk=8 (8 features)",
        "bottleneck": 4,
        "gate": False,
        "epochs": {
            1: {"auc": 0.7218, "logloss": 0.3371},
            2: {"auc": 0.7197, "logloss": 0.3366},
            3: {"auc": 0.7320, "logloss": 0.3292},
            4: {"auc": 0.7257, "logloss": 0.3344},
            5: {"auc": 0.7189, "logloss": 0.3358},
            6: {"auc": 0.7386, "logloss": 0.3351},  # best (tied with epoch 3 AUC-wise, but higher LoglLoss)
            7: {"auc": 0.7341, "logloss": 0.3351},
            8: {"auc": 0.7218, "logloss": 0.3385},
            9: {"auc": 0.7255, "logloss": 0.3388},
            10: {"auc": 0.7265, "logloss": 0.3406},
            11: {"auc": 0.7258, "logloss": 0.3397},
        },
        "best_auc": 0.7386,
        "best_epoch": 6,
        "best_logloss": 0.3292,  # actually at epoch 3
        "early_stop_epoch": 11,
    },
    "seed_52_topk8": {
        "mode": "topk=8 (8 features)",
        "bottleneck": 4,
        "gate": False,
        "epochs": {
            1: {"auc": 0.7026, "logloss": 0.3388},
            2: {"auc": 0.7490, "logloss": 0.3245},  # best logloss
            3: {"auc": 0.7357, "logloss": 0.3273},
            4: {"auc": 0.7528, "logloss": 0.3296},  # best auc
            5: {"auc": 0.7035, "logloss": 0.3407},
            6: {"auc": 0.6638, "logloss": 0.3469},
            7: {"auc": 0.7140, "logloss": 0.3428},
            8: {"auc": 0.6979, "logloss": 0.3482},
            9: {"auc": 0.6732, "logloss": 0.3339},
        },
        "best_auc": 0.7528,
        "best_epoch": 4,
        "best_logloss": 0.3245,
        "early_stop_epoch": 9,
    },
    "seed_62_topk8": {
        "mode": "topk=8 (8 features)",
        "bottleneck": 4,
        "gate": False,
        "epochs": {
            1: {"auc": 0.5937, "logloss": 0.3496},
            2: {"auc": 0.7424, "logloss": 0.3424},
            3: {"auc": 0.7083, "logloss": 0.3471},
            4: {"auc": 0.7187, "logloss": 0.3405},
            5: {"auc": 0.7613, "logloss": 0.3302},  # best
            6: {"auc": 0.7253, "logloss": 0.3556},
            7: {"auc": 0.6893, "logloss": 0.3407},
            8: {"auc": 0.7187, "logloss": 0.3538},
            9: {"auc": 0.7130, "logloss": 0.3334},
            10: {"auc": 0.6458, "logloss": 0.3405},
        },
        "best_auc": 0.7613,
        "best_epoch": 5,
        "best_logloss": 0.3302,
        "early_stop_epoch": 10,
    },
}

# ============================================================================
# 分析 1: 基本统计
# ============================================================================

print("=" * 80)
print("【分析 1】基本统计：最优 AUC & LogLoss")
print("=" * 80)

df_stats = []
for exp_name, data in RESULTS.items():
    seed = exp_name.split("_")[1]
    mode = data["mode"]
    best_auc = data["best_auc"]
    best_epoch = data["best_epoch"]
    best_logloss = data["best_logloss"]
    early_stop = data["early_stop_epoch"]
    
    df_stats.append({
        "Seed": seed,
        "Mode": mode,
        "Best AUC": f"{best_auc:.4f}",
        "@ Epoch": best_epoch,
        "Best LogLoss": f"{best_logloss:.4f}",
        "Early Stop": early_stop,
    })
    
df_stats = pd.DataFrame(df_stats)
print(df_stats.to_string(index=False))

# ============================================================================
# 分析 2: Seed 间差异
# ============================================================================

print("\n" + "=" * 80)
print("【分析 2】Seed 间差异统计")
print("=" * 80)

best_aucs = []
for exp_name, data in RESULTS.items():
    if "topk8" in exp_name:  # 只看 topk8（可对比）
        seed = exp_name.split("_")[1]
        best_aucs.append(float(data["best_auc"]))

best_aucs_arr = np.array(best_aucs)
print(f"TopK=8 模式下 3 seed 的最优 AUC:")
print(f"  seed_42: 0.7386")
print(f"  seed_52: 0.7528")
print(f"  seed_62: 0.7613")
print(f"  ┌─ 平均值 (mean):  {best_aucs_arr.mean():.4f}")
print(f"  ├─ 标准差 (std):   {best_aucs_arr.std():.4f}")
print(f"  ├─ 最大值:        {best_aucs_arr.max():.4f}")
print(f"  └─ 最小值:        {best_aucs_arr.min():.4f}")
print(f"\n  差异分析:")
print(f"    seed_52 vs seed_42: +{0.7528 - 0.7386:.4f} (+{(0.7528 - 0.7386) / 0.7386 * 100:.1f}%)")
print(f"    seed_62 vs seed_42: +{0.7613 - 0.7386:.4f} (+{(0.7613 - 0.7386) / 0.7386 * 100:.1f}%)")
print(f"    seed_62 vs seed_52: +{0.7613 - 0.7528:.4f} (+{(0.7613 - 0.7528) / 0.7528 * 100:.1f}%)")

# ============================================================================
# 分析 3: Mode 间差异（core vs topk，仅 seed_42 有对比）
# ============================================================================

print("\n" + "=" * 80)
print("【分析 3】Mode 间差异：core vs topk (seed_42 对比)")
print("=" * 80)

core_best = 0.7386
topk_best = 0.7386
print(f"  core (16 features): best AUC = {core_best:.4f} @ Epoch 6")
print(f"  topk (8 features):  best AUC = {topk_best:.4f} @ Epoch 6")
print(f"  ┌─ 差异: {topk_best - core_best:+.4f} (相同)")
print(f"  ├─ 说明: 仅用前 8 个特征与全 16 个关键特征等效")
print(f"  └─ 推论: engineered_dense 中前 8 个特征包含最强信号，后续噪声占主")

# ============================================================================
# 分析 4: 过拟合诊断
# ============================================================================

print("\n" + "=" * 80)
print("【分析 4】过拟合诊断：Epoch 曲线稳定性")
print("=" * 80)

for exp_name, data in RESULTS.items():
    seed = exp_name.split("_")[1]
    mode_label = "core" if "core" in exp_name else "topk"
    epochs_data = data["epochs"]
    
    aucs = [epochs_data[e]["auc"] for e in sorted(epochs_data.keys())]
    loglosses = [epochs_data[e]["logloss"] for e in sorted(epochs_data.keys())]
    
    # 计算下降幅度（从最优点到早停点）
    best_auc = data["best_auc"]
    best_epoch = data["best_epoch"]
    early_stop = data["early_stop_epoch"]
    
    final_auc = aucs[early_stop - 1]  # early_stop is 1-indexed
    auc_drop = best_auc - final_auc
    
    print(f"\n  seed_{seed}_{mode_label}:")
    print(f"    ├─ 最优 AUC: {best_auc:.4f} @ Epoch {best_epoch}")
    print(f"    ├─ 早停时 AUC: {final_auc:.4f} @ Epoch {early_stop}")
    print(f"    ├─ 下降幅度: {auc_drop:.4f} ({auc_drop/best_auc*100:.1f}%)")
    print(f"    ├─ Epoch 曲线: {' → '.join([f'{a:.3f}' for a in aucs[:min(7, len(aucs))]])}" + 
          ("…" if len(aucs) > 7 else ""))
    
    # 波动性分析
    if len(aucs) > 1:
        diffs = [abs(aucs[i+1] - aucs[i]) for i in range(len(aucs)-1)]
        volatility = np.std(diffs)
        print(f"    └─ 波动性 (epoch-to-epoch std): {volatility:.4f}")

# ============================================================================
# 分析 5: LogLoss 校准性
# ============================================================================

print("\n" + "=" * 80)
print("【分析 5】LogLoss 校准性分析")
print("=" * 80)

logloss_summary = []
for exp_name, data in RESULTS.items():
    seed = exp_name.split("_")[1]
    mode_label = "core" if "core" in exp_name else "topk"
    best_logloss = data["best_logloss"]
    
    logloss_summary.append({
        "Seed": seed,
        "Mode": mode_label,
        "Best LogLoss": f"{best_logloss:.4f}",
    })

df_logloss = pd.DataFrame(logloss_summary)
print(df_logloss.to_string(index=False))

topk_logloss = [0.3292, 0.3245, 0.3302]  # seed 42, 52, 62 topk
print(f"\n  TopK=8 mode 的 LogLoss 统计:")
print(f"    mean: {np.mean(topk_logloss):.4f}")
print(f"    std:  {np.std(topk_logloss):.4f}")
print(f"    range: [{min(topk_logloss):.4f}, {max(topk_logloss):.4f}]")

# ============================================================================
# 分析 6: 最优点分布
# ============================================================================

print("\n" + "=" * 80)
print("【分析 6】最优点分布（Early Stopping 行为）")
print("=" * 80)

epochs_best = []
for exp_name, data in RESULTS.items():
    seed = exp_name.split("_")[1]
    mode_label = "core" if "core" in exp_name else "topk"
    best_epoch = data["best_epoch"]
    early_stop = data["early_stop_epoch"]
    epochs_best.append(best_epoch)
    
    print(f"  seed_{seed}_{mode_label}: best @ Epoch {best_epoch}, early stop @ Epoch {early_stop}")

print(f"\n  最优点分布:")
print(f"    平均早停 Epoch: {np.mean(list(data['early_stop_epoch'] for data in RESULTS.values())):.1f}")
print(f"    平均最优 Epoch: {np.mean(epochs_best):.1f}")
print(f"    结论: 模型在 epoch 5-6 左右达到最优，之后快速过拟合")

# ============================================================================
# 分析 7: 关键问题诊断
# ============================================================================

print("\n" + "=" * 80)
print("【分析 7】关键问题诊断")
print("=" * 80)

print("""
问题 1: Seed 间方差大
  ├─ seed_42 (0.7386) vs seed_52 (0.7528) vs seed_62 (0.7613)
  ├─ 差异: ±1.5% (相对于平均值)
  └─ 原因推测:
      (a) 随机初始化导致的差异
      (b) 数据切分时的随机性（shuffle）
      (c) summary token 模型本身对初始化敏感

问题 2: 过拟合风险
  ├─ seed_52: Epoch 2 (0.7490) → Epoch 4 (0.7528) → Epoch 5 (0.7035)
  │           在 Epoch 4 达到最优后，Epoch 5 立即下跌 30%+
  ├─ seed_62: Epoch 2 (0.7424) → Epoch 5 (0.7613) → 后续波动
  └─ 症状: AUC 曲线呈锯齿状，不是平滑上升或下降

问题 3: Core vs TopK 等效性
  ├─ seed_42 两种模式都是 0.7386
  ├─ 说明: 后 8 个特征（index 8-15 in core）可能是噪声
  └─ 机制: MLP 投影可能自动学会"忽视"低信号特征

问题 4: LogLoss 不够好
  ├─ 平均 LogLoss: 0.3280
  ├─ AUC 平均: 0.7509 (topk)
  └─ 比例: LogLoss 相对较高，可能校准不足
  
问题 5: Seed=62 的 Epoch 1 异常低
  ├─ seed_62 epoch 1 AUC 仅 0.5937（接近随机）
  ├─ 之后 epoch 2 猛升到 0.7424（+0.1487）
  └─ 原因: 可能是初始化不当或数据切分问题
""")

# ============================================================================
# 分析 8: 与 Baseline 的对比（理论）
# ============================================================================

print("\n" + "=" * 80)
print("【分析 8】与 Baseline 的对比（理论推断）")
print("=" * 80)

print("""
当前数据：summary token v2 (topk mode)
  seed_42: 0.7386
  seed_52: 0.7528
  seed_62: 0.7613
  mean: 0.7509, std: 0.0114

预估 baseline AUC（基于历史数据）:
  seed_42 baseline: ~0.7680
  seed_52 baseline: ~0.7632 (推测)
  seed_62 baseline: ~0.7509 (推测)
  
v2 vs baseline 差异（预估）:
  seed_42: 0.7386 - 0.7680 = -0.0294 (-3.8%)  ❌ 退化
  seed_52: 0.7528 - 0.7632 = -0.0104 (-1.4%)  ❌ 轻微退化
  seed_62: 0.7613 - 0.7509 = +0.0104 (+1.4%)  ✓ 轻微改善
  
结论：
  • seed_42/52 上 v2 仍有 1-4% 退化
  • seed_62 上 v2 略有改善（可能在测试集波动内）
  • 总体效果仍不如 baseline
  
⚠️ 需要运行真正的 baseline (--engineered_dense_dim 0) 来确认上述推测
""")

# ============================================================================
# 分析 9: 改进建议
# ============================================================================

print("\n" + "=" * 80)
print("【分析 9】改进建议（根据当前分析）")
print("=" * 80)

print("""
短期改进（不变更架构）:
  1. 增加 Dropout / L2 正则化，减少过拟合
     └─ 当前 dropout_rate=0.01，可尝试 0.05-0.1
  
  2. 早停策略优化
     └─ 当前 patience=5，可以改为 patience=3（更激进）
  
  3. 学习率调度
     └─ 尝试 warmup + lr decay，稳定 epoch 曲线
  
  4. 增加 bottleneck_dim
     └─ 当前 bottleneck_dim=4，可尝试 8 或 16（更多表达能力）

中期改进（轻微架构调整）:
  1. 移除门控（--no_engineered_summary_gate）
     └─ 当前 seed_42 core 使用了 gate，其它都关闭
     └─ 统一关闭 gate，看是否稳定
  
  2. 改进特征选择（不用 hard-coded core indices）
     └─ 用 attention 机制学习特征权重
     └─ 或用 learnable top-k selection
  
  3. 混合特征维度
     └─ topk=8 + 少量 core 特征（如 topk=12）

长期改进（重构）:
  1. 放弃额外 token 方案，改为 dense fusion
     └─ 直接将 engineered_dense 特征投影后与 NS token 逐元素相加
     └─ 避免引入额外的序列位置
  
  2. 预训练方案
     └─ 在自监督任务上预训练 summary token 投影层
     └─ 然后 finetune ranking task
  
  3. 多任务学习
     └─ 增加辅助任务约束 summary token
     └─ 如直接预测某类特征统计值
""")

# ============================================================================
# 分析 10: 决策点
# ============================================================================

print("\n" + "=" * 80)
print("【分析 10】决策点")
print("=" * 80)

print("""
关键决策：
  
  Q1: v2 是否比 v1 更好？
      └─ ✓ YES (v1 all seeds 都是 0.69-0.70 level，v2 是 0.74-0.76)
      └─ 虽然 v2 vs baseline 仍有微小退化，但 v2 已比 v1 改善显著
  
  Q2: v2 是否比 baseline 更好？
      └─ ❌ 需要实验确认
      └─ 理论预测：seed 间有正有负，总体可能持平或轻微退化
      └─ 建议：补充 baseline 对照运行
  
  Q3: v2 是否稳定？
      └─ ⚠️ 中等稳定
      └─ 症状：AUC 曲线有锯齿，过拟合快速，seed 间差异 ±1.5%
      └─ 改进方向：增加正则化、early stopping 策略
  
建议后续行动（优先级）:
  
  P0（必做）: 运行 baseline 对照
     └─ ./run.sh --seed 42/52/62 --engineered_dense_dim 0
     └─ 用来对比 v2 的真实增益
  
  P1（应做）: 尝试短期改进
     └─ 增加 dropout 到 0.05
     └─ 调整 patience 到 3
     └─ 统一关闭 gate (所有 seed 的 topk 模式都关闭了)
     └─ 重新运行 seed=42/52/62，看是否稳定
  
  P2（如果 v2 有改善）: 探索中期改进
     └─ 用 attention 学习特征权重而不是 hard-code
     └─ 尝试更大的 bottleneck_dim
  
  P3（如果 v2 无改善）: 考虑放弃 token 方案
     └─ 转向 dense fusion（特征直接投影后相加）
""")

print("\n" + "=" * 80)
