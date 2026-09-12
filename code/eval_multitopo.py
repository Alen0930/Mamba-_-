"""
多拓扑正式评测（AUC 主指标 + stop_step + fc_value）

在 BA / ER / WS 三种拓扑上，把各模型与基线放在**同条件**下对比：

    - 所有方法都通过 unified_interface.dismantle() 拿到**长度 n 的完整序列**
      （内部用 _fill_remaining 按 degree 降序补齐），尾部形状一致，指标可比
    - 所有随机性方法（CoreHD/GND/EI/random）固定 seed=0，可复现
    - CoreHD 报多档 Sthreshold：它是 CoreHD 自己的核心旋钮，只跟 stop_condition=1
      比是不公平的——对齐到 fc 目标后 CoreHD 的 fc 会显著变好

指标口径见 evaluate.py 的 lcc_trace/calc_auc：auc = ∫₀¹ S(q)dq，越小越好。

## 方法命名

    degree / random / pagerank        静态启发式
    CoreHD                            stop_condition=1
    CoreHD@0.01n / CoreHD@0.1n        按比例缩放 Sthreshold
    gnn-static:<alias>                监督模型静态打分（一次前向）
    gnn-dyn-b25:<alias>               监督模型动态重算，批粒度 25（b 可任意指定）
    rl-greedy:<alias>                 RL 模型逐步贪心（批粒度 1）

`<alias>` 由 --gnn-ckpt / --rl-ckpt 以 `名称=路径` 指定。同一个 alias 的模型
只加载一次、跨图复用。

## 样本量（重要）

**Wilcoxon signed-rank 在 k 个配对样本下最小双侧 p = 2/2^k**：
    k ≤ 5  ->  ≥0.0625  **数学上不可能显著，跑了也白跑**
    k = 8  ->  0.0078   可用
    k = 30 ->  <0.001   功效 ~0.99

而模型方法的每图成本随 n 急升（n=1500 的 rl-greedy ≈ n=500 的 3 倍），
但**检验力只取决于配对样本数、与 n 无关** → 用 --graphs-per-size 把预算
堆在便宜的规模上换样本量，例如 `--n-sizes 500,1000,1500 --graphs-per-size 30,10,8`。

**per-topology 分别报告，不做跨拓扑平均**（BA 的 AUC ~0.19 与 WS ~0.33 差近一倍，
混合平均会被拓扑配比支配）。

用法:
    # 廉价基线（秒级）
    python eval_multitopo.py --methods degree,random,pagerank,CoreHD,CoreHD@0.01n

    # 对比两个监督模型 + RL（n=500 取 30 图/拓扑，配对检验才有功效）
    python eval_multitopo.py --n-sizes 500 --graphs-per-size 30 \
        --methods degree,CoreHD@0.01n,gnn-dyn-b25:ba,gnn-dyn-b25:mixed,rl-greedy:v4 \
        --gnn-ckpt ba=checkpoints/gnn_mamba/best_model.pth \
                    mixed=checkpoints/gnn_mamba_mixed/best_model.pth \
        --rl-ckpt  v4=checkpoints/gnn_mamba_rl500v4/best_model.pth \
        --device cpu --output results/main_n500.csv
"""
import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch

from evaluate import calc_all_metrics
from network_dismantling.Mamba.graph_factory import build_graph
from network_dismantling.Mamba.model_io import unregister_model
from network_dismantling.unified_interface import dismantle

# 各拓扑的评测图参数（与 graph_factory 的默认范围一致）
TOPOLOGY_PARAMS = {
    "ba": {"m": 3},
    "er": {"k_avg": 6.0},
    "ws": {"k": 6, "beta": 0.2},
}

# 拓扑 -> 测试图种子偏移量。
# 不用 Python 内置 hash()：它对字符串每进程随机加盐（PYTHONHASHSEED），
# 会让同一命令两次运行生成不同的图。
_TOPO_SEED_OFFSET = {"ba": 0, "er": 1_000_000, "ws": 2_000_000}

# 不需要模型的方法
_PLAIN_METHODS = {"degree", "random", "pagerank", "CoreHD"}

_DYN_RE = re.compile(r"^gnn-dyn-b(\d+)$")


# 别名分隔符。
# 不能用 '@'——CoreHD 的 Sthreshold 写法 CoreHD@0.01n 里也有 '@'，
# 用 partition('@') 会把它拆成 base='CoreHD' + alias='0.01n'，于是命中
# _PLAIN_METHODS 分支、**静默退化成 stop_condition=1 的普通 CoreHD**。
# 这个坑真的踩过：主实验里 CoreHD@0.01n 的 fc 全是 sc=1 的值，导致
# 「RL 在 fc 上显著优于 CoreHD」的错误结论。
_ALIAS_SEP = ":"


def split_spec(spec: str):
    """'gnn-dyn-b25:mixed' -> ('gnn-dyn-b25', 'mixed')；无分隔符时 alias 为 None"""
    base, _, alias = spec.partition(_ALIAS_SEP)
    return base, (alias or None)


def spec_to_key(spec: str) -> str:
    """方法名 -> CSV 列名前缀（去掉可能残留的分隔符）"""
    return spec.replace(_ALIAS_SEP, "-").replace("@", "-")


def build_test_graph(n: int, topology: str, seed: int) -> nx.Graph:
    """构造单张评测图（拓扑参数固定，保证同拓扑跨 seed 可比）"""
    return build_graph(n, topology, seed=seed, **TOPOLOGY_PARAMS[topology])


def validate_methods(methods: List[str]) -> None:
    for spec in methods:
        base, alias = split_spec(spec)
        if base in _PLAIN_METHODS or base.startswith("CoreHD@"):
            continue
        if base == "gnn-static" or _DYN_RE.match(base):
            if alias is None:
                raise SystemExit(f"方法 '{spec}' 需要 '{_ALIAS_SEP}别名' 指定用哪个检查点，"
                                 f"例如 '{base}{_ALIAS_SEP}ba'")
            continue
        if base == "rl-greedy":
            if alias is None:
                raise SystemExit(f"方法 '{spec}' 需要 '{_ALIAS_SEP}别名'，"
                                 f"例如 'rl-greedy{_ALIAS_SEP}v4'")
            continue
        raise SystemExit(
            f"未知方法 '{spec}'。可用：degree/random/pagerank/CoreHD/CoreHD@0.01n/"
            f"gnn-static{_ALIAS_SEP}别名/gnn-dyn-b<N>{_ALIAS_SEP}别名/"
            f"rl-greedy{_ALIAS_SEP}别名"
        )


def parse_alias_paths(pairs: Optional[List[str]], kind: str) -> Dict[str, str]:
    """['ba=path1', 'mixed=path2'] -> {'ba': 'path1', ...}"""
    out: Dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--{kind} 需要 名称=路径 形式，收到 '{p}'")
        name, _, path = p.partition("=")
        out[name.strip()] = path.strip()
    return out


def min_achievable_p(k: int) -> float:
    """k 个配对样本下 Wilcoxon 双侧最小 p 值（= 2/2^k）"""
    return min(1.0, 2.0 / (2 ** k)) if k > 0 else 1.0


def _self_check() -> None:
    """
    启动自检：方法名解析。

    为什么需要：曾经把别名分隔符设成 '@'，于是 `CoreHD@0.01n` 被
    `partition('@')` 拆成 base='CoreHD' + alias='0.01n'，命中 _PLAIN_METHODS
    分支后**静默退化成 stop_condition=1 的普通 CoreHD**——不报错、不崩溃，
    只是结果变成另一个方法，导致整轮主实验的 fc 结论完全反了。
    这类"静默串味"必须靠断言拦住。
    """
    cases = [
        ("CoreHD@0.01n", ("CoreHD@0.01n", None)),   # '@' 是 Sthreshold，不是别名
        ("CoreHD@0.1n", ("CoreHD@0.1n", None)),
        ("CoreHD", ("CoreHD", None)),
        ("degree", ("degree", None)),
        ("gnn-dyn-b25:mixed", ("gnn-dyn-b25", "mixed")),
        ("gnn-static:ba", ("gnn-static", "ba")),
        ("rl-greedy:v4", ("rl-greedy", "v4")),
    ]
    for spec, want in cases:
        got = split_spec(spec)
        assert got == want, f"方法名解析错误：{spec!r} -> {got}，期望 {want}"
    assert spec_to_key("CoreHD@0.01n") == "CoreHD-0.01n"
    assert spec_to_key("gnn-dyn-b25:mixed") == "gnn-dyn-b25-mixed"


def resolve_stop_condition(method: str, n_nodes: int) -> int:
    """
    CoreHD 的 Sthreshold 解析（自检用，也方便调用方复核）。
    'CoreHD@0.01n' + n=500 -> 5
    """
    if not method.startswith("CoreHD@"):
        return 1
    frac = float(method.split("@", 1)[1].rstrip("n"))
    return max(1, int(frac * n_nodes))


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
        )

    ap = argparse.ArgumentParser(description="多拓扑正式评测（AUC 主指标）")
    ap.add_argument("--topologies", type=str, default="ba,er,ws")
    ap.add_argument("--n-sizes", type=str, default="500,1000,1500")
    ap.add_argument("--seeds-per-size", type=int, default=3,
                    help="不指定 --graphs-per-size 时所有规模统一用这个数")
    ap.add_argument("--graphs-per-size", type=str, default=None,
                    help="按规模分别指定图数，与 --n-sizes 一一对应，如 '30,10,8'。"
                         "理由是模型每图成本随 n 急升，而检验力只取决于配对样本数")
    ap.add_argument("--methods", type=str,
                    default="degree,random,pagerank,CoreHD,CoreHD@0.01n")
    ap.add_argument("--gnn-ckpt", type=str, nargs="+", default=None,
                    help="监督模型检查点，'名称=路径'，如 ba=checkpoints/gnn_mamba/best_model.pth")
    ap.add_argument("--rl-ckpt", type=str, nargs="+", default=None,
                    help="RL 模型检查点，'名称=路径'")
    ap.add_argument("--device", type=str, default="cpu",
                    help="推理设备，默认 cpu（Mamba use_fast_path=False 下 CPU 反而快 ~1.5x）")
    ap.add_argument("--stop-ratio", type=float, default=0.1)
    ap.add_argument("--fc-threshold", type=float, default=0.01)
    ap.add_argument("--output", type=str, default="results/multitopo_eval.csv")
    args = ap.parse_args()

    _self_check()  # 方法名解析自检（防止 CoreHD@<frac> 被误当别名）

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    n_sizes = [int(x) for x in args.n_sizes.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    validate_methods(methods)

    if args.graphs_per_size:
        gps = [int(x) for x in args.graphs_per_size.split(",")]
        if len(gps) != len(n_sizes):
            raise SystemExit(f"--graphs-per-size 有 {len(gps)} 项，"
                             f"--n-sizes 有 {len(n_sizes)} 项，必须一一对应")
    else:
        gps = [args.seeds_per_size] * len(n_sizes)
    n_per_size = dict(zip(n_sizes, gps))

    gnn_ckpts = parse_alias_paths(args.gnn_ckpt, "gnn-ckpt")
    rl_ckpts = parse_alias_paths(args.rl_ckpt, "rl-ckpt")

    print("=" * 88)
    print("多拓扑正式评测（AUC 主指标）")
    print("=" * 88)
    print(f"拓扑: {topologies}   规模/图数: {n_per_size}")
    print(f"方法: {methods}")
    print(f"监督模型: {gnn_ckpts or '(未使用)'}")
    print(f"RL 模型:  {rl_ckpts or '(未使用)'}")
    print(f"设备: {args.device}")
    for n, k in zip(n_sizes, gps):
        flag = "  [样本数不足，无法判定显著性]" if k < 6 else ""
        print(f"  n={n:<6} 图数={k:<4} Wilcoxon 最小双侧 p = {min_achievable_p(k):.4f}{flag}")
    print("=" * 88, flush=True)

    device = args.device
    unregister_model()

    # ---- 按需懒加载模型 ----
    gnn_models: Dict[str, object] = {}
    for spec in methods:
        base, alias = split_spec(spec)
        if alias and (base == "gnn-static" or _DYN_RE.match(base)):
            if alias in gnn_models:
                continue
            if alias not in gnn_ckpts:
                raise SystemExit(f"方法 '{spec}' 用了别名 '{alias}'，"
                                 f"但 --gnn-ckpt 里没有定义")
            gnn_models[alias] = gnn_ckpts[alias]  # 存路径，dismantle 内部按需加载
            print(f"监督模型别名 {alias}: {gnn_ckpts[alias]}")

    rl_models: Dict[str, tuple] = {}
    rl_specs = [s for s in methods if split_spec(s)[0] == "rl-greedy"]
    if rl_specs:
        from network_dismantling.Mamba.gnn_mamba_actor_critic import (
            create_actor_critic_from_supervised,
        )
        from eval_rl import greedy_sequence
        for spec in rl_specs:
            alias = split_spec(spec)[1]
            if alias in rl_models:
                continue
            if alias not in rl_ckpts:
                raise SystemExit(f"方法 '{spec}' 用了别名 '{alias}'，"
                                 f"但 --rl-ckpt 里没有定义")
            m = create_actor_critic_from_supervised(rl_ckpts[alias], device)
            m.eval()
            rl_models[alias] = (m, greedy_sequence)
            print(f"RL 模型别名 {alias}: {rl_ckpts[alias]}")

    def run_method(spec: str, G: nx.Graph, stop_condition: int) -> List[int]:
        base, alias = split_spec(spec)
        if base in _PLAIN_METHODS:
            return dismantle(G, method=base)
        if base.startswith("CoreHD@"):
            return dismantle(G, method="CoreHD",
                             stop_condition=resolve_stop_condition(base, G.number_of_nodes()))
        if base == "gnn-static":
            return dismantle(G, method="mamba_gnn", model_path=gnn_models[alias])
        m = _DYN_RE.match(base)
        if m:
            return dismantle(G, method="mamba_gnn", model_path=gnn_models[alias],
                             batch_size=int(m.group(1)))
        if base == "rl-greedy":
            model, greedy_fn = rl_models[alias]
            return greedy_fn(model, G, device, stop_condition)
        raise ValueError(spec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # n = 请求的节点数（**分组用**）；n_actual = 取 LCC 后的实际节点数
    # （AUC/stop/fc 的分母用它，已在 calc_all_metrics 内部处理）。
    # 二者必须分开：ER 取 LCC 后会丢几个节点，若分组用 n_actual，
    # 同一批评测会被拆成 n=496/n=497 等碎组，配对检验直接失效。
    fields = ["topology", "n", "n_actual", "seed"] + [
        f"{spec_to_key(m)}_{k}" for m in methods for k in ("auc", "stop", "fc")
    ]

    rows: List[Dict] = []
    t_start = time.time()

    for topo in topologies:
        for n in n_sizes:
            for s in range(n_per_size[n]):
                # 确定性种子：同拓扑同规模跨 seed 可比，且跨进程可复现
                seed = 1000 * n + _TOPO_SEED_OFFSET.get(topo, 3_000_000) + s
                G = build_test_graph(n, topo, seed)
                n_actual = G.number_of_nodes()
                stop_cond = max(1, int(0.01 * n_actual))

                row = {"topology": topo, "n": n, "n_actual": n_actual, "seed": seed}
                print(f"\n=== {topo.upper()} n={n} (实际 {n_actual}) seed={seed} ===",
                      flush=True)

                for m in methods:
                    t0 = time.time()
                    try:
                        seq = run_method(m, G, stop_cond)
                    except Exception as e:  # 单方法失败不中断整轮评测
                        print(f"  {m:<26} 失败: {type(e).__name__}: {e}", flush=True)
                        continue

                    met = calc_all_metrics(G, seq)
                    key = spec_to_key(m)
                    row[f"{key}_auc"] = round(met["auc"], 4)
                    row[f"{key}_stop"] = met["stop_step"]
                    row[f"{key}_fc"] = round(met["fc_value"], 4)
                    print(f"  {m:<26} AUC={met['auc']:.4f}  stop={met['stop_step']:<5} "
                          f"fc={met['fc_value']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

                rows.append(row)

                # 增量落盘，避免长评测中途崩溃丢数据
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: r.get(k, "") for k in fields})

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 88)
    print("按拓扑汇总（AUC / stop_step / fc_value 越小越好）")
    print("=" * 88)
    for topo in topologies:
        sub = [r for r in rows if r["topology"] == topo]
        if not sub:
            continue
        print(f"\n【{topo.upper()}】 {len(sub)} 张图")
        print(f"  {'方法':<26} {'AUC':<10} {'stop':<9} {'fc':<9} {'rel_auc':<9}")
        print("  " + "-" * 68)

        hd_key, rand_key = "CoreHD-0.01n_auc", "random_auc"
        auc_hd = (np.mean([r[hd_key] for r in sub])
                  if any(hd_key in r for r in sub) else None)
        auc_rand = (np.mean([r[rand_key] for r in sub])
                    if any(rand_key in r for r in sub) else None)

        for m in methods:
            key = spec_to_key(m)
            vals = [r[f"{key}_auc"] for r in sub if f"{key}_auc" in r]
            if not vals:
                continue
            auc = float(np.mean(vals))
            stop = np.mean([r[f"{key}_stop"] for r in sub if f"{key}_stop" in r])
            fc = np.mean([r[f"{key}_fc"] for r in sub if f"{key}_fc" in r])
            rel = ""
            if auc_rand is not None and auc_hd is not None and auc_rand != auc_hd:
                rel = f"{(auc - auc_hd) / (auc_rand - auc_hd):.3f}"
            print(f"  {m:<26} {auc:<10.4f} {stop:<9.1f} {fc:<9.4f} {rel:<9}")

    print(f"\n结果已保存: {out_path}")
    print(f"总耗时: {(time.time()-t_start)/60:.1f} 分钟")
    print("\n下一步：python paired_test.py --csv " + str(out_path)
          + " --ref rl-greedy-<别名>")


if __name__ == "__main__":
    main()
