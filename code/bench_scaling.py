"""
序列模型扩展性对比：双向 Mamba (O(N)) vs 自注意力 (O(N²)) vs 无序列模型

这是方法核心论断的直接检验——「MAMBA 以线性复杂度处理长序列，天然适配大规模
网络；相比依赖 O(N²) 注意力、难以扩展到上千节点的 Transformer」。

测量同一图规模 N 下：
    - 前向+反向墙钟时间
    - 峰值内存增量（若 psutil 可用）
    - 注意力矩阵的理论内存（B·heads·N²·4 字节，解析值）

模型配置对齐：相同的 GCN 层数/d_model/层数，dim_feedforward=2·d_model 与
Mamba 的 expand=2 量级对齐，保证两者参数量与计算量可比。

用法:
    python bench_scaling.py                      # 默认 N=500..4000
    python bench_scaling.py --sizes 500,1000,2000,4000,8000
    python bench_scaling.py --device cpu
"""
import argparse
import io
import sys
import time

import numpy as np
import torch

from network_dismantling.Mamba.gnn_mamba_model import GNNMambaModel


def build_inputs(N: int, batch: int = 1, seed: int = 0, density: float = 0.01):
    """造一个稠密邻接 + 度特征的输入（模拟真实调用形状）"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, N, 1, generator=g)
    a = (torch.rand(batch, N, N, generator=g) < density).float()
    a = ((a + a.transpose(1, 2)) > 0).float()
    mask = torch.ones(batch, N, dtype=torch.bool)
    return x, a, mask


def _rss_mb():
    """当前进程 RSS（MB）。Windows 用 ctypes 直接读，无需 psutil。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        ctr = PROCESS_MEMORY_COUNTERS()
        ctr.cb = ctypes.sizeof(ctr)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(ctr), ctr.cb
        )
        return ctr.WorkingSetSize / 1e6
    except Exception:
        return None


def measure(model, x, a, mask, device, repeats: int = 3, backward: bool = True):
    """返回 (中位耗时秒, 峰值内存增量MB 或 None)"""
    x = x.to(device); a = a.to(device); mask = mask.to(device)
    times, mems = [], []
    for _ in range(repeats):
        m0 = _rss_mb()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(x, a, mask)
        if backward and model.training:
            out.sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        m1 = _rss_mb()
        if m0 is not None and m1 is not None:
            mems.append(m1 - m0)
        model.zero_grad(set_to_none=True)
    return float(np.median(times)), (float(np.median(mems)) if mems else None)


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
        )

    ap = argparse.ArgumentParser(description="序列模型扩展性对比")
    ap.add_argument("--sizes", type=str, default="500,1000,2000,4000")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-gnn-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", type=str, default="results/scaling.csv")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    variants = [("mamba", "双向 Mamba (O(N))"),
                ("attention", "自注意力 (O(N²))"),
                ("none", "无序列模型")]

    print("=" * 92)
    print(f"序列模型扩展性对比　设备={args.device}　batch={args.batch}　"
          f"d_model={args.d_model}　层数={args.n_layers}")
    print("=" * 92)
    print(f"  {'N':>6} {'变体':<20} {'参数':>9} {'前向+反向(s)':>13} "
          f"{'峰值内存增量(MB)':>17} {'注意力矩阵(MB)':>15}")
    print("  " + "-" * 88)

    rows = []
    for N in sizes:
        x, a, mask = build_inputs(N, batch=args.batch)
        for sm, tag in variants:
            nl = 0 if sm == "none" else args.n_layers
            # 'none' 变体用 n_mamba_layers=0 表达（模型不接受 seq_model='none'）
            model = GNNMambaModel(
                input_dim=1, hidden_dim=64, d_model=args.d_model,
                n_gnn_layers=args.n_gnn_layers, n_mamba_layers=nl,
                seq_model=("mamba" if sm == "none" else sm),
            ).to(args.device)
            model.train()
            n_par = sum(p.numel() for p in model.parameters())
            try:
                t, mem = measure(model, x, a, mask, args.device, args.repeats)
            except (RuntimeError, MemoryError) as e:
                print(f"  {N:>6} {tag:<20} {n_par:>9,}  OOM/失败: {type(e).__name__}")
                rows.append({"N": N, "variant": sm, "params": n_par,
                             "time_s": None, "mem_mb": None})
                del model
                continue
            # 注意力矩阵：B · heads · N² · 4B（float32）
            attn_mb = (args.batch * 4 * N * N * 4 / 1e6) if sm == "attention" else 0.0
            print(f"  {N:>6} {tag:<20} {n_par:>9,} {t:>13.3f} "
                  f"{(f'{mem:.1f}' if mem is not None else 'n/a'):>17} "
                  f"{attn_mb:>15.1f}")
            rows.append({"N": N, "variant": sm, "params": n_par,
                         "time_s": round(t, 4),
                         "mem_mb": (round(mem, 1) if mem is not None else None),
                         "attn_matrix_mb": round(attn_mb, 2)})
            del model
        print()

    # ---- 增长阶估计 ----
    print("=" * 92)
    print("增长阶（用最小二乘拟合 log t ~ k·log N，k≈1 线性、k≈2 二次）")
    print("=" * 92)
    for sm, tag in variants:
        pts = [(r["N"], r["time_s"]) for r in rows
               if r["variant"] == sm and r["time_s"]]
        if len(pts) < 2:
            continue
        Ns = np.log([p[0] for p in pts]); Ts = np.log([p[1] for p in pts])
        k = np.polyfit(Ns, Ts, 1)[0]
        print(f"  {tag:<22} 时间增长指数 k = {k:.2f}")

    import csv
    from pathlib import Path
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["N", "variant", "params", "time_s",
                                          "mem_mb", "attn_matrix_mb"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n结果已保存: {out}")
    print(
        "\n注意：\n"
        "  · 本机的 mamba_ssm 已强制回退到 selective_scan_ref（纯 PyTorch 参考实现），\n"
        "    因为预编译 CUDA kernel 不支持 RTX 5060 的 sm_120。因此**这里的 Mamba 耗时\n"
        "    不代表其真实性能**，只反映回退路径。O(N) 的论断针对优化过的 kernel，\n"
        "    在本机无法直接验证。\n"
        "  · 「峰值内存增量」为进程 RSS 差值的 median；torch 会复用缓存分配，\n"
        "    小规模下常常低于测量灵敏度而显示 0.0。可信的是「注意力矩阵(MB)」列——\n"
        "    它是精确解析值 B·heads·N²·4B，即注意力先天需要的 O(N²) 中间量\n"
        "    （N=4000 时单层 256MB，N=8000 时 1GB，N=16000 时 4GB）。\n"
        "  · 时间增长指数用 4 个点做最小二乘拟合，只作趋势指示，不是严格定阶。"
    )


if __name__ == "__main__":
    main()
