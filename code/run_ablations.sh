#!/usr/bin/env bash
# 消融训练队列：等 frag RL 跑完后自动依次训练
#
# 三个变体（其余超参完全一致，单变量）：
#   A  nomamba : n_mamba_layers=0  -> 纯 GCN（13K 参数），回答「Mamba 到底有没有贡献」
#   B  dfs     : Mamba + DFS 遍历序（相邻即邻居 77%，序列即一条图上游走）
#   C  core    : Mamba + k-core 剥壳序（core number 是全局量，GCN 难以推断）
#
# 关键设计推论：GCN 对节点置换等变，因此**排序只通过 Mamba 起作用**——
# 「无 Mamba + 任意排序」是同一个模型，A 的结果与排序无关。
# 于是「Mamba+结构排序 是否优于 GCN-only」就成为该方法核心主张的直接检验。
set -u

cd "$(dirname "$0")"
PY=venv312/Scripts/python.exe
LOG_DIR=logs/ablation
mkdir -p "$LOG_DIR"

RL_LOG="${RL_LOG:-}"
if [ -n "$RL_LOG" ] && [ -f "$RL_LOG" ]; then
  echo "[queue] 等待 frag RL 训练结束..."
  for _ in $(seq 1 600); do
    grep -q "训练完成" "$RL_LOG" 2>/dev/null && break
    sleep 30
  done
  echo "[queue] frag RL 已结束，开始消融训练"
fi

COMMON="--n-gnn-layers 3 --epochs 100 --patience 15 --batch-size 4 \
        --select-by auc --auc-subset 5 --seed 42"

run() {
  local name=$1; shift
  echo "=============================================================="
  echo "[queue] 开始训练: $name   ($(date '+%H:%M:%S'))"
  echo "=============================================================="
  $PY -u run_gnn_train.py "$@" > "$LOG_DIR/$name.log" 2>&1
  echo "[queue] $name 结束 (exit=$?)，日志 $LOG_DIR/$name.log"
  tail -6 "$LOG_DIR/$name.log"
}

# A. 纯 GCN（无 Mamba）—— 参数少、无 Mamba 开销，应当很快
run nomamba --dataset-dir datasets/mixed_gnn --checkpoint-dir checkpoints/gnn_nomamba \
    --n-mamba-layers 0 --order degree $COMMON

# B. Mamba + DFS 遍历序
run dfs --dataset-dir datasets/mixed_dfs --checkpoint-dir checkpoints/gnn_mamba_dfs \
    --n-mamba-layers 2 --order dfs $COMMON

# C. Mamba + k-core 剥壳序
run core --dataset-dir datasets/mixed_core --checkpoint-dir checkpoints/gnn_mamba_core \
    --n-mamba-layers 2 --order core $COMMON

echo "[queue] 全部完成 ($(date '+%H:%M:%S'))"
