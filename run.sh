#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 运行约定：
# - 代码放在当前目录（或容器内挂载到 /workspace）
# - 数据/日志/ckpt 位置通过环境变量控制
# - 本地直接运行时保留默认目录，平台上建议显式覆盖这些环境变量
: "${TRAIN_DATA_PATH:=${SCRIPT_DIR}/industrial_data}"
: "${TRAIN_CKPT_PATH:=${SCRIPT_DIR}/ckpt}"
: "${TRAIN_LOG_PATH:=${SCRIPT_DIR}/logs}"
: "${TRAIN_TF_EVENTS_PATH:=${SCRIPT_DIR}/tfevents}"
export TRAIN_DATA_PATH TRAIN_CKPT_PATH TRAIN_LOG_PATH TRAIN_TF_EVENTS_PATH

# 腾讯平台 / Docker / 本地统一入口：
# 默认走 RankMixer，不依赖 ns_groups.json；如需切 group tokenizer，
# 可以在命令行里覆盖这些参数。
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"

# 平台建议的线上参数示例：
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --reinit_sparse_after_epoch 999 \
#     --reinit_cardinality_threshold 1000000 \
#     "$@"