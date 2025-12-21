#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_objective_correlation.py

真正诊断 Cost 与 Carbon 目标相关性的脚本（避免被负荷大小驱动的假相关性误导）

输出内容：
1) raw: 直接在不同工况样本上算 corr(cost, carbon) —— 仅作参考
2) per-MW: 用总有功负荷归一化后 corr(cost/Pd, carbon/Pd) —— 更接近目标本质关系
3) residual/partial: 先用 Pd_total 回归掉 cost 与 carbon 的趋势，再看残差 corr —— 更强去趋势
4) within-bin: 在相似 Pd 区间内算相关性并加权平均 —— 最直观“同负荷”目标关系
5) generator-level: (边际成本/成本系数) 与 GCI 的相关 —— 粗但对“是否存在权衡”很敏感
6) 图：raw/perMW/residual 散点图 + 回归线（保存到 results/ 目录）

依赖：numpy, scipy, matplotlib
工程依赖：config/get_config, data_loader/load_all_data, unified_eval/get_genload/build_ctx_from_supervised
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from data_loader import load_all_data
from unified_eval import get_genload, build_ctx_from_supervised
from utils import get_carbon_emission_vectorized


# -----------------------------
# Utils
# -----------------------------
def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, len(x)
    if method == "pearson":
        r, p = pearsonr(x, y)
    elif method == "spearman":
        r, p = spearmanr(x, y)
    else:
        raise ValueError("method must be pearson or spearman")
    return float(r), float(p), len(x)


def linear_residual(y, X):
    """
    y: (N,)
    X: (N, d)  (will add intercept automatically)
    return residual: y - (a + Xb)
    """
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    N = X.shape[0]
    X_aug = np.concatenate([np.ones((N, 1)), X], axis=1)
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_hat = X_aug @ beta
    resid = (y - y_hat).reshape(-1)
    return resid


def weighted_mean(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# -----------------------------
# Objective computation
# -----------------------------
def compute_objectives_from_real_solution(ctx, n_samples=2000):
    """
    用真实最优电压（如果 ctx 有）计算：
    - total cost ($/h)
    - total carbon (tCO2/h 或你的实现单位)
    同时返回 Pd_total (MW) 作为控制变量
    """
    if not (hasattr(ctx, "Real_Vm_full") and hasattr(ctx, "Real_Va_full")):
        raise RuntimeError("ctx 缺少 Real_Vm_full / Real_Va_full，无法基于真实最优解诊断。")

    Pd = to_numpy(ctx.Pdtest)[:n_samples]
    Qd = to_numpy(ctx.Qdtest)[:n_samples]
    Real_Vm = to_numpy(ctx.Real_Vm_full)[:n_samples]
    Real_Va = to_numpy(ctx.Real_Va_full)[:n_samples]

    # total demand (MW)
    baseMVA = float(ctx.baseMVA)
    Pd_total_MW = np.sum(Pd, axis=1) * baseMVA  # Pd 是 p.u. 的话
    Pd_total_MW = np.clip(Pd_total_MW, 1e-6, None)

    Real_V = Real_Vm * np.exp(1j * Real_Va)

    Real_Pg, _, _, _ = get_genload(
        Real_V, Pd, Qd,
        ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )

    gencost = to_numpy(ctx.gencost_Pg)  # 期望形状：(Ng, 2/3) 或已裁剪
    Pg_MW = Real_Pg * baseMVA

    # 兼容 2列/3列成本系数（若你的 gencost 已经是纯系数矩阵）
    if gencost.ndim != 2 or gencost.shape[0] != Pg_MW.shape[1]:
        raise ValueError(f"gencost_Pg shape 异常：{gencost.shape}，Pg_MW shape={Pg_MW.shape}，请检查 ctx.gencost_Pg 是否为纯系数矩阵")

    if gencost.shape[1] >= 3:
        c2, c1, c0 = gencost[:, 0], gencost[:, 1], gencost[:, 2]
        cost_each = c2 * (Pg_MW ** 2) + c1 * Pg_MW + c0
    elif gencost.shape[1] == 2:
        c2, c1 = gencost[:, 0], gencost[:, 1]
        cost_each = c2 * (Pg_MW ** 2) + c1 * Pg_MW
    else:
        raise ValueError(f"不支持的 gencost 列数：{gencost.shape[1]}")

    cost_total = np.sum(cost_each, axis=1)

    # carbon
    gci_values = to_numpy(ctx.gci_values)
    carbon_total = get_carbon_emission_vectorized(Real_Pg, gci_values, baseMVA)

    return cost_total, carbon_total, Pd_total_MW, Pg_MW, gencost, gci_values


# -----------------------------
# Analyses
# -----------------------------
def report_correlation_suite(cost, carbon, Pd_total_MW, title_prefix=""):
    """
    输出四种相关性：
    1) raw
    2) per-MW
    3) residual (remove Pd trend)
    4) within-bin (same-load bins)
    """
    print("\n" + "=" * 90)
    print(f"{title_prefix}相关性诊断（重点看 per-MW / residual / within-bin）")
    print("=" * 90)

    # Raw
    r_p, p_p, n = safe_corr(cost, carbon, "pearson")
    r_s, p_s, _ = safe_corr(cost, carbon, "spearman")
    print(f"[Raw] Pearson r={r_p:.4f} (p={p_p:.2e}, n={n}) | Spearman ρ={r_s:.4f} (p={p_s:.2e})")

    # Correlation with load
    r_c_load, p_c_load, _ = safe_corr(cost, Pd_total_MW, "pearson")
    r_e_load, p_e_load, _ = safe_corr(carbon, Pd_total_MW, "pearson")
    print(f"[Load drive] corr(cost, Pd_total_MW)={r_c_load:.4f} (p={p_c_load:.2e})")
    print(f"[Load drive] corr(carbon, Pd_total_MW)={r_e_load:.4f} (p={p_e_load:.2e})")
    if np.isfinite(r_c_load) and np.isfinite(r_e_load) and (abs(r_c_load) > 0.7 and abs(r_e_load) > 0.7):
        print("  -> 警告：cost 与 carbon 都被 Pd 强烈驱动，Raw 的 cost-carbon 高相关很可能是“负荷趋势”导致。")

    # per-MW normalize
    cost_pm = cost / Pd_total_MW
    carbon_pm = carbon / Pd_total_MW
    r_pm, p_pm, n_pm = safe_corr(cost_pm, carbon_pm, "pearson")
    r_pm_s, p_pm_s, _ = safe_corr(cost_pm, carbon_pm, "spearman")
    print(f"[Per-MW] Pearson r={r_pm:.4f} (p={p_pm:.2e}, n={n_pm}) | Spearman ρ={r_pm_s:.4f} (p={p_pm_s:.2e})")

    # residual/partial: remove linear trend vs Pd_total_MW
    cost_res = linear_residual(cost, Pd_total_MW)
    carbon_res = linear_residual(carbon, Pd_total_MW)
    r_res, p_res, n_res = safe_corr(cost_res, carbon_res, "pearson")
    r_res_s, p_res_s, _ = safe_corr(cost_res, carbon_res, "spearman")
    print(f"[Residual|Pd] Pearson r={r_res:.4f} (p={p_res:.2e}, n={n_res}) | Spearman ρ={r_res_s:.4f} (p={p_res_s:.2e})")

    # within-bin: bin by Pd_total_MW and compute correlation inside each bin
    nbins = 10
    edges = np.quantile(Pd_total_MW, np.linspace(0, 1, nbins + 1))
    bin_rs = []
    bin_ns = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        mask = (Pd_total_MW >= lo) & (Pd_total_MW <= hi if i == nbins - 1 else Pd_total_MW < hi)
        if mask.sum() < 20:
            continue
        r_bin, _, n_bin = safe_corr(cost[mask], carbon[mask], "pearson")
        if np.isfinite(r_bin):
            bin_rs.append(r_bin)
            bin_ns.append(n_bin)
    if len(bin_rs) > 0:
        r_within = weighted_mean(bin_rs, bin_ns)
        print(f"[Within-bin|Pd] 加权平均 Pearson r={r_within:.4f}  (bins_used={len(bin_rs)}/{nbins})")
    else:
        print("[Within-bin|Pd] bin 内样本不足，跳过")

    # return for plotting
    return {
        "cost_pm": cost_pm,
        "carbon_pm": carbon_pm,
        "cost_res": cost_res,
        "carbon_res": carbon_res,
    }


def report_generator_level_tradeoff(Pg_MW, gencost, gci_values):
    """
    发电机层面的“成本 vs 排放强度”关系（粗诊断，但很有效）：
    - 相关：c1 vs gci、(2*c2*E[Pg]+c1) vs gci
    - 如果显著负相关：便宜机组更脏 -> 强权衡
      如果显著正相关：便宜机组更干净 -> 目标更一致
    """
    print("\n" + "=" * 90)
    print("发电机层面 trade-off 诊断（粗但很有指向性）")
    print("=" * 90)

    Ng = gencost.shape[0]
    gci = np.asarray(gci_values).reshape(-1)
    if gci.shape[0] != Ng:
        print(f"  gci_values 长度({gci.shape[0]}) != 发电机数({Ng})，跳过发电机层分析")
        return

    # cost coefficients
    if gencost.shape[1] >= 2:
        c2 = gencost[:, 0]
        c1 = gencost[:, 1]
    else:
        print("  gencost 列数不足，跳过")
        return

    # correlation: c1 vs gci
    r1, p1, _ = safe_corr(c1, gci, "pearson")
    rs1, ps1, _ = safe_corr(c1, gci, "spearman")
    print(f"  corr(c1, GCI): Pearson r={r1:.4f} (p={p1:.2e}) | Spearman ρ={rs1:.4f} (p={ps1:.2e})")

    # mean dispatch marginal cost at mean Pg: mc = 2*c2*Pg + c1
    Pg_mean = np.mean(Pg_MW, axis=0)  # (Ng,)
    mc_mean = 2.0 * c2 * Pg_mean + c1
    r2, p2, _ = safe_corr(mc_mean, gci, "pearson")
    rs2, ps2, _ = safe_corr(mc_mean, gci, "spearman")
    print(f"  corr(mean marginal cost, GCI): Pearson r={r2:.4f} (p={p2:.2e}) | Spearman ρ={rs2:.4f} (p={ps2:.2e})")

    def interpret(r):
        if not np.isfinite(r):
            return "无法判断"
        if r < -0.5:
            return "强负相关：便宜机组更脏（典型权衡）"
        if r > 0.5:
            return "强正相关：便宜机组更净（目标更一致）"
        return "弱/中等相关：关系复杂（可能存在局部权衡）"

    print(f"  结论(按 c1): {interpret(r1)}")
    print(f"  结论(按 mc_mean): {interpret(r2)}")


# -----------------------------
# Plotting
# -----------------------------
def plot_scatter(x, y, xlabel, ylabel, title, outpath):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x, y, s=10, alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)

    # add linear fit line
    if len(x) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 200)
        ys = a * xs + b
        plt.plot(xs, ys, linewidth=2)

        r, p, _ = safe_corr(x, y, "pearson")
        plt.text(0.02, 0.98, f"Pearson r={r:.3f}\np={p:.2e}",
                 transform=plt.gca().transAxes, va="top")

    ensure_dir(os.path.dirname(outpath))
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000, help="用于统计的样本数（来自 test set 前 n 个）")
    parser.add_argument("--out_dir", type=str, default=None, help="结果输出目录（默认: 当前脚本同级 results/）")
    args = parser.parse_args()

    print("=" * 90)
    print("Objective Correlation Diagnostics (Cost vs Carbon) - 去除负荷驱动的假相关")
    print("=" * 90)

    config = get_config()
    sys_data, dataloaders, BRANFT = load_all_data(config)

    device = config.device
    ctx = build_ctx_from_supervised(config, sys_data, dataloaders, BRANFT, device)

    n_samples = args.n_samples
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(out_dir)

    # Compute objectives from real optimal voltages
    cost, carbon, Pd_total_MW, Pg_MW, gencost, gci_values = compute_objectives_from_real_solution(ctx, n_samples=n_samples)

    print("\n" + "=" * 90)
    print("目标范围（基于真实最优解）")
    print("=" * 90)
    print(f"  n={len(cost)}")
    print(f"  Cost:   mean={np.mean(cost):.2f}, std={np.std(cost):.2f}, "
          f"min={np.min(cost):.2f}, max={np.max(cost):.2f}")
    print(f"  Carbon: mean={np.mean(carbon):.4f}, std={np.std(carbon):.4f}, "
          f"min={np.min(carbon):.4f}, max={np.max(carbon):.4f}")
    print(f"  Pd_total(MW): mean={np.mean(Pd_total_MW):.2f}, std={np.std(Pd_total_MW):.2f}, "
          f"min={np.min(Pd_total_MW):.2f}, max={np.max(Pd_total_MW):.2f}")

    # Correlation suite
    suite = report_correlation_suite(cost, carbon, Pd_total_MW, title_prefix="Cost-Carbon ")

    # Generator-level tradeoff
    report_generator_level_tradeoff(Pg_MW, gencost, gci_values)

    # Plots
    print("\n" + "=" * 90)
    print("生成可视化（raw / per-MW / residual）")
    print("=" * 90)
    plot_scatter(cost, carbon,
                 xlabel="Cost ($/h)", ylabel="Carbon",
                 title="Raw: Cost vs Carbon (cross-load samples)",
                 outpath=os.path.join(out_dir, "scatter_raw_cost_vs_carbon.png"))

    plot_scatter(suite["cost_pm"], suite["carbon_pm"],
                 xlabel="Cost / Pd_total ($/MWh approx.)", ylabel="Carbon / Pd_total",
                 title="Per-MW: (Cost/Pd) vs (Carbon/Pd)  (less load-driven)",
                 outpath=os.path.join(out_dir, "scatter_perMW_cost_vs_carbon.png"))

    plot_scatter(suite["cost_res"], suite["carbon_res"],
                 xlabel="Cost residual after regressing on Pd_total", ylabel="Carbon residual after regressing on Pd_total",
                 title="Residual: remove Pd trend then correlate",
                 outpath=os.path.join(out_dir, "scatter_residual_cost_vs_carbon.png"))

    print("\n" + "=" * 90)
    print("Done.")
    print("=" * 90)
    print("解读建议：")
    print("  - 如果 Raw r 很大，但 Per-MW / Residual / Within-bin r 很小：说明高相关主要来自负荷趋势，不代表目标天然一致。")
    print("  - 如果 Per-MW / Residual / Within-bin 依然很大（尤其 >0.7）：才说明目标在“同负荷”意义下高度一致（权衡空间可能很小）。")
    print("  - 如果发电机层面 corr(marginal cost, GCI) 显著负相关：典型 cost-carbon 权衡，模型应当能学到偏好差异。")


if __name__ == "__main__":
    main()
