#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展训练数据集 - 为不同偏好生成多目标 OPF 解

使用方法:
    # 从头生成 1000 个样本，偏好范围 0~100
    python expand_training_data_multi_preference.py --generate --nsamples 1000

    # 增强现有数据集
    python expand_training_data_multi_preference.py --data_mat main_part/data/XY_case300real.mat

    # 使用 case118
    python expand_training_data_multi_preference.py --case_m main_part/data/case118_ieee_modified.m --generate

    # 自定义偏好范围
    python expand_training_data_multi_preference.py --generate --lambda_min 0 --lambda_max 50 --lambda_step 5

依赖: pypower, scipy, numpy, tqdm
"""

import os
import argparse
import numpy as np
from typing import Dict, Optional
import json
from tqdm import tqdm

from opf_by_pypower import load_data, PyPowerOPFSolver


# =============================================================================
# 核心功能
# =============================================================================

def expand_with_preferences(
    case_m_path: str,
    data_mat_path: Optional[str] = None,
    n_samples: int = 100,
    lambda_carbon_values: np.ndarray = None,
    output_dir: str = "saved_data/multi_preference_solutions",
    delta: float = 0.1,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    为不同偏好生成多目标 OPF 解
    
    Args:
        case_m_path: MATPOWER .m 文件路径
        data_mat_path: 训练数据 .mat 文件路径。None = 自动生成负荷场景
        n_samples: 样本数量
        lambda_carbon_values: 碳排放权重数组，默认 [0, 2, 4, ..., 100]
        output_dir: 输出目录
        delta: 自动生成时的负荷变化范围 (±delta)
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        {"total": int, "success": int, "failed": int, "files": list}
    """
    if lambda_carbon_values is None:
        lambda_carbon_values = np.arange(0, 101, 2.0)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载/生成负荷数据
    mode = "增强现有数据" if data_mat_path else "自动生成负荷"
    print(f"模式: {mode}")
    
    data = load_data(case_m_path, data_mat_path, n_samples, delta, seed)
    x_load_pu = data["x_load_pu"]
    nbus = data["nbus"]
    n_actual = x_load_pu.shape[0]
    
    print(f"样本数: {n_actual}, 节点数: {nbus}")
    
    # 创建求解器
    solver = PyPowerOPFSolver(
        case_m_path,
        use_multi_objective=True,
        lambda_carbon=0.0,
        verbose=verbose
    )
    
    # 输出维度: Va (nbus-1, 去掉slack) + Vm (nbus)
    output_dim = 2 * nbus - 1
    
    # 统计
    stats = {"total": 0, "success": 0, "failed": 0, "files": []}
    
    # 总样本数 = 偏好数 * 每个偏好的样本数
    total_tasks = len(lambda_carbon_values) * n_actual
    
    # 为每个偏好求解
    with tqdm(total=total_tasks, desc="OPF Solving", ncols=100, unit="sample") as pbar:
        for lam_c in lambda_carbon_values:
            solutions = np.zeros((n_actual, output_dim), dtype=np.float32)
            success_mask = np.zeros(n_actual, dtype=bool)
            costs = np.zeros(n_actual, dtype=np.float32)
            carbons = np.zeros(n_actual, dtype=np.float32)
            
            # 更新进度条描述
            pbar.set_postfix({"λ": f"{lam_c:.1f}", "ok": stats["success"], "fail": stats["failed"]})
            
            for i in range(n_actual):
                stats["total"] += 1
                
                try:
                    result = solver.forward(x_load_pu[i], lambda_carbon=lam_c)
                except Exception as e:
                    stats["failed"] += 1
                    if verbose:
                        tqdm.write(f"[ERR] i={i}, λ={lam_c:.2f}: {e}")
                    pbar.update(1)
                    continue
                
                if not result["success"]:
                    stats["failed"] += 1
                    pbar.update(1)
                    continue
                
                # 提取结果
                Va_rad = result["bus"]["Va_rad"]
                Vm = result["bus"]["Vm"]
                
                # 去掉 slack 节点的相角
                mask = np.ones(nbus, dtype=bool)
                mask[solver.slack_row] = False
                Va_noslack = Va_rad[mask]
                
                solutions[i] = np.concatenate([Va_noslack, Vm])
                success_mask[i] = True
                costs[i] = result["summary"]["economic_cost"]
                carbons[i] = result["summary"]["carbon_emission"]
                stats["success"] += 1
                pbar.update(1)
            
            # 保存该偏好的结果
            filename = f"y_train_lc{lam_c:.2f}.npz"
            filepath = os.path.join(output_dir, filename)
            np.savez_compressed(
                filepath,
                solutions=solutions,
                success_mask=success_mask,
                costs=costs,
                carbons=carbons,
                lambda_carbon=lam_c,
            )
            stats["files"].append(filename)
            
            # 打印进度
            n_success = np.sum(success_mask)
            if verbose:
                tqdm.write(f"  λ_c={lam_c:.1f}: {n_success}/{n_actual} 成功")
    
    # 保存输入数据和元数据
    np.savez_compressed(
        os.path.join(output_dir, "x_train.npz"),
        x_load_pu=x_load_pu,
        nbus=nbus,
        baseMVA=data["baseMVA"],
    )
    
    metadata = {
        "case_m": case_m_path,
        "data_mat": data_mat_path,
        "mode": mode,
        "n_samples": n_actual,
        "nbus": nbus,
        "output_dim": output_dim,
        "lambda_carbon_values": lambda_carbon_values.tolist(),
        "n_preferences": len(lambda_carbon_values),
        "stats": {k: v for k, v in stats.items() if k != "files"},
        "files": stats["files"],
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return stats


def load_expanded_data(output_dir: str, lambda_carbon: float) -> Dict:
    """
    加载指定偏好的扩展数据
    
    Args:
        output_dir: 扩展数据目录
        lambda_carbon: 碳排放权重
    
    Returns:
        {"x_load_pu": array, "solutions": array, "success_mask": array, ...}
    """
    # 加载输入
    x_data = np.load(os.path.join(output_dir, "x_train.npz"))
    
    # 加载指定偏好的输出
    y_file = os.path.join(output_dir, f"y_train_lc{lambda_carbon:.2f}.npz")
    if not os.path.exists(y_file):
        raise FileNotFoundError(f"找不到偏好文件: {y_file}")
    
    y_data = np.load(y_file)
    
    return {
        "x_load_pu": x_data["x_load_pu"],
        "nbus": int(x_data["nbus"]),
        "solutions": y_data["solutions"],
        "success_mask": y_data["success_mask"],
        "costs": y_data["costs"],
        "carbons": y_data["carbons"],
        "lambda_carbon": float(y_data["lambda_carbon"]),
    }


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="扩展训练数据集 - 为不同偏好生成多目标 OPF 解",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从头生成 1000 个样本
  python expand_training_data_multi_preference.py --generate --nsamples 1000

  # 增强现有数据集
  python expand_training_data_multi_preference.py --data_mat main_part/data/XY_case300real.mat

  # 使用 case118
  python expand_training_data_multi_preference.py --case_m main_part/data/case118_ieee_modified.m --generate
        """
    )
    
    # 数据源
    parser.add_argument("--case_m", default="main_part/data/case300_ieee_modified.m",
                        help="MATPOWER .m 文件路径")
    parser.add_argument("--data_mat", default=None,
                        help="训练数据 .mat 文件 (None = 自动生成)")
    parser.add_argument("--generate", action="store_true",
                        help="强制自动生成负荷场景 (忽略 --data_mat)")
    
    # 样本参数
    parser.add_argument("--nsamples", type=int, default=4000,
                        help="样本数量 (default: 4000)")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="负荷变化范围 ±delta (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    
    # 偏好参数
    parser.add_argument("--lambda_min", type=float, default=0.0,
                        help="最小 lambda_carbon (default: 0.0)")
    parser.add_argument("--lambda_max", type=float, default=50.0,
                        help="最大 lambda_carbon (default: 50.0)")
    parser.add_argument("--lambda_step", type=float, default=5.0,
                        help="lambda_carbon 步长 (default: 5.0)")
    
    # 输出
    parser.add_argument("--output_dir", default="saved_data/multi_preference_solutions",
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    
    args = parser.parse_args()
    
    # 确定数据来源
    data_mat = None if args.generate else args.data_mat
    
    # 生成偏好值 (使用 round 避免浮点累积误差)
    lambda_values = np.round(np.arange(
        args.lambda_min, 
        args.lambda_max + args.lambda_step / 2,  # 包含最大值
        args.lambda_step
    ), 2)
    
    # 打印配置
    print("=" * 60)
    print("扩展训练数据集 - 多目标 OPF")
    print("=" * 60)
    print(f"案例文件: {args.case_m}")
    print(f"数据来源: {'自动生成' if data_mat is None else data_mat}")
    print(f"样本数量: {args.nsamples}")
    print(f"偏好数量: {len(lambda_values)} (λ_carbon: {args.lambda_min} ~ {args.lambda_max}, step={args.lambda_step})")
    print(f"输出目录: {args.output_dir}")
    print("-" * 60)
    
    # 执行扩展
    stats = expand_with_preferences(
        case_m_path=args.case_m,
        data_mat_path=data_mat,
        n_samples=args.nsamples,
        lambda_carbon_values=lambda_values,
        output_dir=args.output_dir,
        delta=args.delta,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # 打印结果
    print("-" * 60)
    success_rate = 100 * stats["success"] / stats["total"] if stats["total"] > 0 else 0
    print(f"完成: {stats['success']}/{stats['total']} 成功 ({success_rate:.1f}%)")
    print(f"生成文件: {len(stats['files'])} 个偏好文件 + x_train.npz + metadata.json")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
