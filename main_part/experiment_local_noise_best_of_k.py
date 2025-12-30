"""
实验：监督信号附近局部加噪 + Best-of-K 选择

目的：验证在已知可行解附近做局部探索是否能够生成可行样本

实验设计：
1. 从监督数据 y_sup 出发（已知的可行解）
2. 采样高斯噪声 Δy ~ N(0, σ²I)
3. 生成 K 个候选 y = y_sup + Δy
4. 用 NGT loss 评估每个候选的约束违反程度
5. 选择约束违反最小的样本（Best-of-K）
6. 观察可行率和目标值随 K 的变化

结论解读：
- 如果局部加噪 + Best-of-K 能提升可行率 → 局部探索可行，应考虑 Anchor-Residual 方案
- 如果不能提升 → NGT 可行性度量可能与评估口径不一致，需要先对齐

Usage:
    conda activate pdp_cp
    python main_part/experiment_local_noise_best_of_k.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_part.config import get_config
from main_part.data_loader import load_multi_preference_dataset, load_ngt_training_data
from main_part.deepopf_ngt_loss import DeepOPFNGTLoss
from flow_model.generative_vae_utils import (
    make_pref_tensors, lambda_to_key, compute_ngt_loss_chunked_differentiable
)


def evaluate_samples_with_ngt(
    y_samples: torch.Tensor,
    x_input: torch.Tensor,
    pref_raw: torch.Tensor,
    ngt_loss_fn: DeepOPFNGTLoss,
    chunk_size: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 NGT loss 评估样本的约束违反和目标值
    
    Args:
        y_samples: [K, B, output_dim] 或 [K*B, output_dim] 候选样本
        x_input: [B, input_dim] 输入场景
        pref_raw: [B, 2] 偏好权重 [λ_cost, λ_carbon]
        ngt_loss_fn: NGT loss 函数
        chunk_size: 分块大小
    
    Returns:
        constraint_violation: [K, B] 约束违反程度
        objective_value: [K, B] 目标函数值
    """
    # Handle different input shapes
    if y_samples.dim() == 3:
        K, B, output_dim = y_samples.shape
        y_flat = y_samples.reshape(K * B, output_dim)
    else:
        # Assume already flattened
        y_flat = y_samples
        K = y_flat.shape[0] // x_input.shape[0]
        B = x_input.shape[0]
    
    device = y_samples.device
    
    # Expand inputs
    x_expanded = x_input.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)
    pref_expanded = pref_raw.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)
    
    # Get NGT parameters
    ngt_params = ngt_loss_fn.params
    carbon_scale = getattr(ngt_params, 'carbon_scale', 30.0)
    
    # Compute NGT loss
    loss_dict = compute_ngt_loss_chunked_differentiable(
        ngt_params, y_flat, x_expanded, pref_expanded,
        chunk_size=chunk_size, carbon_scale=carbon_scale
    )
    
    # Reshape results
    constraint_flat = loss_dict['constraint_scaled']
    objective_flat = loss_dict['objective_per_sample']
    
    constraint_kb = constraint_flat.reshape(K, B)
    objective_kb = objective_flat.reshape(K, B)
    
    return constraint_kb, objective_kb


def run_local_noise_experiment(
    config,
    multi_pref_data: Dict,
    sys_data: Dict,
    ngt_loss_fn: DeepOPFNGTLoss,
    device: torch.device,
    sigma_values: List[float] = [0.001, 0.005, 0.01, 0.02, 0.05],
    K_values: List[int] = [1, 5, 10, 20, 50, 100],
    n_test_samples: int = 50,
    feas_threshold: float = 0.01
):
    """
    运行局部加噪 + Best-of-K 实验
    
    Args:
        sigma_values: 噪声标准差列表
        K_values: Best-of-K 的 K 值列表
        n_test_samples: 测试样本数
        feas_threshold: 可行性阈值（constraint_scaled < threshold 视为可行）
    """
    print("\n" + "=" * 70)
    print("实验：监督信号附近局部加噪 + Best-of-K 选择")
    print("=" * 70)
    
    # Extract data
    x_train = multi_pref_data['x_train'].to(device)
    y_train_by_pref = multi_pref_data['y_train_by_pref']
    lambda_carbon_values = multi_pref_data['lambda_carbon_values']
    output_dim = multi_pref_data['output_dim']
    
    y_train_by_pref_device = {lambda_to_key(lc): y.to(device) for lc, y in y_train_by_pref.items()}
    lambda_max = max(lambda_carbon_values)
    
    # Use a subset of data
    n_samples = min(n_test_samples, x_train.shape[0])
    x_test = x_train[:n_samples]
    
    # Select middle preference for testing
    test_lc = lambda_carbon_values[len(lambda_carbon_values) // 2]
    pref_norm, pref_raw = make_pref_tensors(test_lc, lambda_max, n_samples, device)
    
    lc_key = lambda_to_key(test_lc)
    y_sup = y_train_by_pref_device[lc_key][:n_samples]  # 监督信号（anchor）
    
    print(f"\n实验配置:")
    print(f"  测试样本数: {n_samples}")
    print(f"  测试偏好: λ_carbon = {test_lc}")
    print(f"  输出维度: {output_dim}")
    print(f"  可行性阈值: {feas_threshold}")
    print(f"  噪声标准差: {sigma_values}")
    print(f"  K 值列表: {K_values}")
    
    # First, evaluate the original supervised signal
    print("\n" + "-" * 70)
    print("Step 1: 评估原始监督信号的可行性")
    print("-" * 70)
    
    with torch.no_grad():
        # Evaluate y_sup directly (K=1)
        constraint_sup, objective_sup = evaluate_samples_with_ngt(
            y_sup.unsqueeze(0),  # [1, B, output_dim]
            x_test, pref_raw, ngt_loss_fn
        )
        constraint_sup = constraint_sup.squeeze(0)  # [B]
        objective_sup = objective_sup.squeeze(0)    # [B]
        
        feas_rate_sup = (constraint_sup < feas_threshold).float().mean().item()
        avg_constraint_sup = constraint_sup.mean().item()
        avg_objective_sup = objective_sup.mean().item()
        
        print(f"\n原始监督信号:")
        print(f"  可行率: {feas_rate_sup * 100:.1f}%")
        print(f"  平均约束违反: {avg_constraint_sup:.6f}")
        print(f"  平均目标值: {avg_objective_sup:.4f}")
        print(f"  约束违反分布: min={constraint_sup.min().item():.6f}, "
              f"max={constraint_sup.max().item():.6f}, "
              f"median={constraint_sup.median().item():.6f}")
    
    # Store results
    results = {
        'sigma': [],
        'K': [],
        'feas_rate': [],
        'avg_constraint': [],
        'avg_objective': [],
        'best_constraint': [],
    }
    
    print("\n" + "-" * 70)
    print("Step 2: 局部加噪 + Best-of-K 实验")
    print("-" * 70)
    
    for sigma in sigma_values:
        print(f"\n>>> σ = {sigma}")
        print(f"{'K':>6} | {'可行率':>10} | {'平均约束':>12} | {'最佳约束':>12} | {'平均目标':>12}")
        print("-" * 60)
        
        for K in K_values:
            with torch.no_grad():
                # Sample K candidates for each test sample
                # y = y_sup + Δy, where Δy ~ N(0, σ²I)
                noise = torch.randn(K, n_samples, output_dim, device=device) * sigma
                y_candidates = y_sup.unsqueeze(0) + noise  # [K, B, output_dim]
                
                # Evaluate all candidates
                constraint_kb, objective_kb = evaluate_samples_with_ngt(
                    y_candidates, x_test, pref_raw, ngt_loss_fn
                )  # [K, B]
                
                # Best-of-K selection: choose sample with minimum constraint violation
                best_idx = constraint_kb.argmin(dim=0)  # [B]
                
                # Get best constraint and objective for each sample
                best_constraint = constraint_kb[best_idx, torch.arange(n_samples, device=device)]
                best_objective = objective_kb[best_idx, torch.arange(n_samples, device=device)]
                
                # Compute metrics
                feas_rate = (best_constraint < feas_threshold).float().mean().item()
                avg_constraint = best_constraint.mean().item()
                avg_objective = best_objective.mean().item()
                min_constraint = constraint_kb.min(dim=0)[0].mean().item()
                
                # Store results
                results['sigma'].append(sigma)
                results['K'].append(K)
                results['feas_rate'].append(feas_rate)
                results['avg_constraint'].append(avg_constraint)
                results['avg_objective'].append(avg_objective)
                results['best_constraint'].append(min_constraint)
                
                print(f"{K:>6} | {feas_rate*100:>9.1f}% | {avg_constraint:>12.6f} | {min_constraint:>12.6f} | {avg_objective:>12.4f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("实验结果分析")
    print("=" * 70)
    
    # Check if Best-of-K helps
    print("\n1. Best-of-K 效果分析（与原始监督信号对比）:")
    print("-" * 50)
    
    best_results_by_sigma = {}
    for sigma in sigma_values:
        sigma_results = [(r['K'], r['feas_rate'], r['avg_constraint']) 
                         for r in [dict(zip(results.keys(), vals)) 
                                   for vals in zip(*results.values())]
                         if r['sigma'] == sigma]
        
        # Find best K for this sigma
        best_k_result = max(sigma_results, key=lambda x: x[1])  # by feas_rate
        best_results_by_sigma[sigma] = best_k_result
        
        k, feas, constr = best_k_result
        improvement = (feas - feas_rate_sup) * 100
        print(f"  σ={sigma:.3f}: 最佳 K={k}, 可行率={feas*100:.1f}% "
              f"({'+'if improvement>=0 else ''}{improvement:.1f}% vs 原始)")
    
    # Overall conclusion
    print("\n2. 总体结论:")
    print("-" * 50)
    
    max_feas_rate = max(results['feas_rate'])
    best_sigma = results['sigma'][results['feas_rate'].index(max_feas_rate)]
    best_k = results['K'][results['feas_rate'].index(max_feas_rate)]
    
    if max_feas_rate > feas_rate_sup + 0.05:  # 5% improvement threshold
        print(f"  [POSITIVE] 局部加噪 + Best-of-K 有效提升可行率!")
        print(f"    最佳配置: σ={best_sigma}, K={best_k}")
        print(f"    可行率: {feas_rate_sup*100:.1f}% → {max_feas_rate*100:.1f}%")
        print(f"\n  → 建议: 应该考虑 Anchor-Residual 或修复器方案")
    elif max_feas_rate >= feas_rate_sup - 0.02:  # Within 2%
        print(f"  [NEUTRAL] 局部加噪 + Best-of-K 效果有限")
        print(f"    可行率基本持平: {feas_rate_sup*100:.1f}% vs {max_feas_rate*100:.1f}%")
        print(f"\n  → 建议: 可能需要更精细的噪声控制或检查 NGT 口径")
    else:
        print(f"  [NEGATIVE] 局部加噪 + Best-of-K 无法提升可行率!")
        print(f"    可行率下降: {feas_rate_sup*100:.1f}% → {max_feas_rate*100:.1f}%")
        print(f"\n  → 建议: NGT 可行性度量可能与评估口径不一致，需要先对齐尺度")
    
    # Check if original supervised signal is already feasible
    print("\n3. 监督信号质量检查:")
    print("-" * 50)
    
    if feas_rate_sup > 0.95:
        print(f"  [OK] 原始监督信号可行率很高 ({feas_rate_sup*100:.1f}%)")
        print(f"       说明监督数据质量良好")
    elif feas_rate_sup > 0.5:
        print(f"  [WARN] 原始监督信号可行率一般 ({feas_rate_sup*100:.1f}%)")
        print(f"         可能存在一些数据质量问题")
    else:
        print(f"  [ERROR] 原始监督信号可行率很低 ({feas_rate_sup*100:.1f}%)")
        print(f"          需要检查 NGT 评估与监督数据生成的口径是否一致!")
    
    return results, feas_rate_sup


def plot_results(results: Dict, feas_rate_sup: float, save_path: str = None):
    """
    绘制实验结果
    """
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get unique sigma values
    sigma_values = sorted(set(results['sigma']))
    K_values = sorted(set(results['K']))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_values)))
    
    # Plot 1: Feasibility rate vs K
    ax1 = axes[0]
    for i, sigma in enumerate(sigma_values):
        mask = [s == sigma for s in results['sigma']]
        ks = [results['K'][j] for j in range(len(mask)) if mask[j]]
        feas = [results['feas_rate'][j] for j in range(len(mask)) if mask[j]]
        ax1.plot(ks, [f*100 for f in feas], 'o-', color=colors[i], label=f'sigma={sigma}')
    
    ax1.axhline(y=feas_rate_sup*100, color='red', linestyle='--', label='Original y_sup')
    ax1.set_xlabel('K (number of candidates)')
    ax1.set_ylabel('Feasibility Rate (%)')
    ax1.set_title('Feasibility Rate vs K')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Average constraint violation vs K
    ax2 = axes[1]
    for i, sigma in enumerate(sigma_values):
        mask = [s == sigma for s in results['sigma']]
        ks = [results['K'][j] for j in range(len(mask)) if mask[j]]
        constr = [results['avg_constraint'][j] for j in range(len(mask)) if mask[j]]
        ax2.plot(ks, constr, 'o-', color=colors[i], label=f'sigma={sigma}')
    
    ax2.set_xlabel('K (number of candidates)')
    ax2.set_ylabel('Avg Constraint Violation')
    ax2.set_title('Constraint Violation vs K')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Plot 3: Objective value vs K
    ax3 = axes[2]
    for i, sigma in enumerate(sigma_values):
        mask = [s == sigma for s in results['sigma']]
        ks = [results['K'][j] for j in range(len(mask)) if mask[j]]
        obj = [results['avg_objective'][j] for j in range(len(mask)) if mask[j]]
        ax3.plot(ks, obj, 'o-', color=colors[i], label=f'sigma={sigma}')
    
    ax3.set_xlabel('K (number of candidates)')
    ax3.set_ylabel('Avg Objective Value')
    ax3.set_title('Objective Value vs K')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.close()


def main():
    """Main experiment function."""
    print("=" * 70)
    print("局部加噪 + Best-of-K 可行性实验")
    print("=" * 70)
    
    # Load config
    config = get_config()
    device = config.device
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading NGT training data...")
    ngt_data, sys_data = load_ngt_training_data(config)
    
    print("\nLoading multi-preference dataset...")
    multi_pref_data, _ = load_multi_preference_dataset(config, sys_data)
    print(f"Training samples: {multi_pref_data['n_train']}")
    
    # Create NGT loss function
    print("Creating NGT loss function...")
    ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
    
    # Run experiment
    results, feas_rate_sup = run_local_noise_experiment(
        config=config,
        multi_pref_data=multi_pref_data,
        sys_data=sys_data,
        ngt_loss_fn=ngt_loss_fn,
        device=device,
        sigma_values=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        K_values=[1, 5, 10, 20, 50, 100],
        n_test_samples=100,
        feas_threshold=0.01
    )
    
    # Plot results
    save_dir = getattr(config, 'results_dir', 'main_part/results')
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'local_noise_best_of_k_experiment.png')
    plot_results(results, feas_rate_sup, plot_path)
    
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
