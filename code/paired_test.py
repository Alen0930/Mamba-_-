"""
配对显著性检验（读实验结果用）

读 eval_multitopo.py 产出的 CSV，对同一批测试图上的两个方法做**配对**检验。

为什么必须配对：所有方法跑在**同一张图**上，"这张图本身好不好拆"这个巨大的
图间方差可以被配对消掉，检验力比独立样本高一个量级。用独立检验会严重低估
显著性和高估所需样本量。

样本量硬约束（本脚本会自动检查并警告）：
    Wilcoxon signed-rank 在 k 个配对样本下，最小双侧 p = 2/2^k
      k=5  -> 0.0625 > 0.05   **数学上不可能显著，跑了白跑**
      k=6  -> 0.03125         勉强（要求所有差值同号）
      k=8  -> 0.0078          可用
      k=30 -> <0.001          功效 ~0.99

用法:
    python paired_test.py --csv results/multitopo_eval.csv
    python paired_test.py --csv results/multitopo_eval.csv --ref rl-greedy --metric auc
    python paired_test.py --csv results/multitopo_eval.csv --topology ba --n 500
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def min_achievable_p(k: int) -> float:
    """k 个配对样本下 Wilcoxon 双侧检验的最小可能 p 值"""
    if k <= 0:
        return 1.0
    return min(1.0, 2.0 / (2 ** k))


def paired_report(df, ref: str, metric: str, topo: str, n: int) -> None:
    """对一个 (topology, n) 子集，输出参考方法 vs 其余方法的配对检验"""
    sub = df[(df["topology"] == topo) & (df["n"] == n)]
    ref_col = f"{ref}_{metric}"
    if ref_col not in sub.columns:
        print(f"  [跳过] 缺少列 {ref_col}")
        return

    # 只保留两组都齐全的行（配对要求成对出现）
    others = [c[: -(len(metric) + 1)] for c in sub.columns
              if c.endswith(f"_{metric}") and c != ref_col]
    rows = []
    for m in others:
        col = f"{m}_{metric}"
        pair = sub[[ref_col, col]].dropna()
        k = len(pair)
        if k == 0:
            continue
        a, b = pair[ref_col].to_numpy(), pair[col].to_numpy()
        diff = a - b
        try:
            p = stats.wilcoxon(a, b, alternative="two-sided").pvalue if np.any(diff != 0) else 1.0
        except Exception:
            p = float("nan")
        p_min = min_achievable_p(k)
        # 符号一致性：k 很小时只有差值几乎全同号才可能显著
        same_sign = max((diff > 0).sum(), (diff < 0).sum()) / k if k else 0.0
        rows.append({
            "对比": f"{ref} vs {m}",
            "Δ均值": diff.mean(),
            "k": k,
            "同号比例": same_sign,
            "p": p,
            "最小可达p": p_min,
            "裁判": ("显著" if p < 0.05 else
                     ("样本不足" if p_min > 0.05 else "不显著")),
        })

    if not rows:
        return
    print(f"\n【{topo.upper()}  n={n}】 参考 = {ref}，指标 = {metric}（Δ = {ref} − 对手，负值表示 {ref} 更优）")
    print(f"  {'对比':<34} {'Δ均值':>9} {'k':>4} {'同号':>7} {'p':>9} {'最小p':>8}  裁判")
    print("  " + "-" * 84)
    for r in sorted(rows, key=lambda x: x["p"]):
        print(f"  {r['对比']:<34} {r['Δ均值']:>9.4f} {r['k']:>4} "
              f"{r['同号比例']:>7.0%} {r['p']:>9.5f} {r['最小可达p']:>8.4f}  {r['裁判']}")


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
        )

    ap = argparse.ArgumentParser(description="配对显著性检验（读实验 CSV）")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--ref", type=str, default="rl-greedy",
                    help="参考方法（要检验它是否优于别人）")
    ap.add_argument("--metric", type=str, default="auc", choices=["auc", "stop", "fc"])
    ap.add_argument("--topology", type=str, default=None, help="只跑某个拓扑")
    ap.add_argument("--n", type=int, default=None, help="只跑某个规模")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV 不存在: {path}")
    df = pd.read_csv(path)

    topo_list = [args.topology] if args.topology else sorted(df["topology"].unique())
    n_list = [args.n] if args.n else sorted(df["n"].unique())

    print("=" * 88)
    print(f"配对 Wilcoxon 检验　参考={args.ref}　指标={args.metric}")
    print(f"注：Δ = {args.ref} − 对手。{'越小越好（AUC/stop/fc 都是）' if args.metric != 'auc' else 'AUC 越小越好'}")
    print("=" * 88)

    for topo in topo_list:
        for n in n_list:
            paired_report(df, args.ref, args.metric, topo, n)

    print()
    print("=" * 88)
    print("提醒：")
    print("  1. 「不显著」≠「无差别」——样本量不够时也会不显著。看『最小可达p』。")
    print("  2. 「A 上显著、B 上不显著」**不能**推出「A 与 B 有显著差异」，")
    print("     要下这种结论必须直接检验交互作用（对 Δ 做组间比较）。")
    print("  3. 不要跨拓扑或跨规模取平均——AUC 拓扑间差近一倍，样本数也不同。")


if __name__ == "__main__":
    main()
