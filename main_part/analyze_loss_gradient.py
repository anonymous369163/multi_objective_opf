"""
分析脚本：诊断为什么高权重的负荷约束仍被忽略

可能的原因：
1. 梯度量级不平衡：即使权重高，如果 L_d 的梯度本身很小，总贡献仍小
2. 损失数值范围差异：L_objective 和 L_d 的数值范围差异大
3. 自适应权重问题：权重更新有滞后或计算错误
4. 损失函数设计：负荷偏差损失可能导致梯度消失

运行方式：
    conda activate pdp && python main_part/analyze_loss_gradient.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

# 导入必要模块
from main_part.multi_objective_loss import MultiObjectiveOPFLoss
from main_part.data_loader import load_all_data
from main_part.config import get_config


def compute_gradient_norms(loss_fn, Vm_pred, Va_pred, Pd, Qd, lambda_cost=0.9, lambda_carbon=0.1):
    """
    分别计算各损失分量对参数的梯度范数
    """
    results = {}
    device = Vm_pred.device
    
    # 1. 计算完整损失并获取分量
    Vm = Vm_pred.detach().clone().requires_grad_(True)
    Va = Va_pred.detach().clone().requires_grad_(True)
    
    _, loss_dict = loss_fn(Vm, Va, Pd, Qd, lambda_cost=lambda_cost, lambda_carbon=lambda_carbon, return_details=True)
    
    # 获取各分量（未加权）
    L_cost = loss_dict['cost']
    L_carbon = loss_dict['carbon']
    L_g = loss_dict['gen_vio']
    L_Sl = loss_dict['branch_pf_vio']
    L_theta = loss_dict['branch_ang_vio']
    L_d = loss_dict['load_dev']
    load_satisfy_pct = loss_dict.get('load_satisfy_pct', 0)
    
    weights = loss_dict['weights']
    k_g = weights['k_g']
    k_Sl = weights['k_Sl']
    k_theta = weights['k_theta']
    k_d = weights['k_d']
    
    results['loss_values'] = {
        'L_cost': L_cost,
        'L_carbon': L_carbon,
        'L_g': L_g,
        'L_Sl': L_Sl,
        'L_theta': L_theta,
        'L_d': L_d,
        'load_satisfy_pct': load_satisfy_pct,
    }
    
    results['weights'] = weights
    
    # 计算加权后的损失值
    L_objective = lambda_cost * L_cost + lambda_carbon * L_carbon
    L_constraints = k_g * L_g + k_Sl * L_Sl + k_theta * L_theta + k_d * L_d
    
    results['weighted_loss'] = {
        'L_objective': L_objective,
        'k_g * L_g': k_g * L_g,
        'k_Sl * L_Sl': k_Sl * L_Sl,
        'k_theta * L_theta': k_theta * L_theta,
        'k_d * L_d': k_d * L_d,
        'L_constraints': L_constraints,
    }
    
    return results


def analyze_gradient_ratio(loss_fn, Vm_pred, Va_pred, Pd, Qd, lambda_cost=0.9, lambda_carbon=0.1):
    """
    分析各损失分量的梯度贡献比例
    
    关键问题：即使 k_d 很大，如果 ∂L_d/∂V 很小，k_d * ∂L_d/∂V 仍可能被其他项主导
    """
    results = {}
    device = Vm_pred.device
    
    # 保存原始权重
    orig_k_g = loss_fn.k_g
    orig_k_Sl = loss_fn.k_Sl
    orig_k_theta = loss_fn.k_theta
    orig_k_d = loss_fn.k_d
    
    # 如果使用自适应权重
    has_scheduler = hasattr(loss_fn, 'weight_scheduler') and loss_fn.weight_scheduler is not None
    if has_scheduler:
        sched_k_g = loss_fn.weight_scheduler.k_g
        sched_k_Sl = loss_fn.weight_scheduler.k_Sl
        sched_k_theta = loss_fn.weight_scheduler.k_theta
        sched_k_d = loss_fn.weight_scheduler.k_d
    
    components = ['objective', 'gen_vio', 'branch_vio', 'load_dev']
    
    for component in components:
        Vm = Vm_pred.detach().clone().requires_grad_(True)
        Va = Va_pred.detach().clone().requires_grad_(True)
        
        # 临时修改权重，只计算单项
        if component == 'objective':
            loss_fn.k_g = 0
            loss_fn.k_Sl = 0
            loss_fn.k_theta = 0
            loss_fn.k_d = 0
            lc, lcarbon = lambda_cost, lambda_carbon
        elif component == 'gen_vio':
            loss_fn.k_g = orig_k_g
            loss_fn.k_Sl = 0
            loss_fn.k_theta = 0
            loss_fn.k_d = 0
            lc, lcarbon = 0, 0
        elif component == 'branch_vio':
            loss_fn.k_g = 0
            loss_fn.k_Sl = orig_k_Sl
            loss_fn.k_theta = orig_k_theta
            loss_fn.k_d = 0
            lc, lcarbon = 0, 0
        elif component == 'load_dev':
            loss_fn.k_g = 0
            loss_fn.k_Sl = 0
            loss_fn.k_theta = 0
            loss_fn.k_d = orig_k_d
            lc, lcarbon = 0, 0
        
        if has_scheduler:
            if component == 'objective':
                loss_fn.weight_scheduler.k_g = 0
                loss_fn.weight_scheduler.k_Sl = 0
                loss_fn.weight_scheduler.k_theta = 0
                loss_fn.weight_scheduler.k_d = 0
            elif component == 'gen_vio':
                loss_fn.weight_scheduler.k_g = sched_k_g
                loss_fn.weight_scheduler.k_Sl = 0
                loss_fn.weight_scheduler.k_theta = 0
                loss_fn.weight_scheduler.k_d = 0
            elif component == 'branch_vio':
                loss_fn.weight_scheduler.k_g = 0
                loss_fn.weight_scheduler.k_Sl = sched_k_Sl
                loss_fn.weight_scheduler.k_theta = sched_k_theta
                loss_fn.weight_scheduler.k_d = 0
            elif component == 'load_dev':
                loss_fn.weight_scheduler.k_g = 0
                loss_fn.weight_scheduler.k_Sl = 0
                loss_fn.weight_scheduler.k_theta = 0
                loss_fn.weight_scheduler.k_d = sched_k_d
        
        try:
            loss, _ = loss_fn(Vm, Va, Pd, Qd, lambda_cost=lc, lambda_carbon=lcarbon, 
                             return_details=True, update_weights=False)
            
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                grad_Vm = Vm.grad.clone() if Vm.grad is not None else torch.zeros_like(Vm)
                grad_Va = Va.grad.clone() if Va.grad is not None else torch.zeros_like(Va)
                results[component] = {
                    'grad_Vm_norm': torch.norm(grad_Vm).item(),
                    'grad_Va_norm': torch.norm(grad_Va).item(),
                    'grad_total_norm': (torch.norm(grad_Vm)**2 + torch.norm(grad_Va)**2).sqrt().item(),
                    'loss_value': loss.item(),
                }
            else:
                results[component] = {
                    'grad_Vm_norm': 0,
                    'grad_Va_norm': 0,
                    'grad_total_norm': 0,
                    'loss_value': float(loss) if loss is not None else 0,
                }
        except Exception as e:
            print(f"  计算 {component} 梯度时出错: {e}")
            results[component] = {
                'grad_Vm_norm': 0,
                'grad_Va_norm': 0,
                'grad_total_norm': 0,
                'loss_value': 0,
            }
    
    # 恢复权重
    loss_fn.k_g = orig_k_g
    loss_fn.k_Sl = orig_k_Sl
    loss_fn.k_theta = orig_k_theta
    loss_fn.k_d = orig_k_d
    if has_scheduler:
        loss_fn.weight_scheduler.k_g = sched_k_g
        loss_fn.weight_scheduler.k_Sl = sched_k_Sl
        loss_fn.weight_scheduler.k_theta = sched_k_theta
        loss_fn.weight_scheduler.k_d = sched_k_d
    
    # 计算梯度贡献比例
    total_grad = sum(r['grad_total_norm'] for r in results.values())
    if total_grad > 0:
        for component in results:
            results[component]['contribution_ratio'] = results[component]['grad_total_norm'] / total_grad
    
    return results


def main():
    print("=" * 80)
    print("负荷约束忽略问题诊断分析")
    print("=" * 80)
    
    # 加载配置和数据
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载系统数据
    print("\n加载系统数据...")
    sys_data, dataloaders, _ = load_all_data(config)
    
    # 加载 GCI 值
    gci_path = Path(config.data_path) / 'bus_gci.csv'
    if gci_path.exists():
        import pandas as pd
        gci_df = pd.read_csv(gci_path)
        gci_values = torch.tensor(gci_df['gci'].values, dtype=torch.float32, device=device)
    else:
        gci_values = torch.ones(config.Nbus, device=device) * 0.5
    
    # 创建损失函数
    print("\n创建损失函数...")
    loss_fn = MultiObjectiveOPFLoss(sys_data, config, gci_values, use_adaptive_weights=True)
    loss_fn.to(device)
    
    # 准备测试数据
    print("\n准备测试数据...")
    train_loader_vm = dataloaders['train_vm']
    train_loader_va = dataloaders['train_va']
    
    # 获取一个 batch
    for (batch_x_vm, batch_y_vm), (batch_x_va, batch_y_va) in zip(train_loader_vm, train_loader_va):
        batch_x = batch_x_vm.to(device)
        Vm_true = batch_y_vm.to(device)
        Va_true = batch_y_va.to(device)
        break
    
    batch_size = batch_x.shape[0]
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {batch_x.shape}")
    
    # 分离 Vm 和 Va 标签（模拟预测）
    output_dim_vm = sys_data.yvm_train.shape[1]
    output_dim_va = sys_data.yva_train.shape[1]
    
    # 使用真实标签加噪声作为"预测"
    Vm_pred = Vm_true.clone()
    Va_pred = Va_true.clone()
    
    # 添加噪声模拟预测误差
    noise_scale = 0.1
    Vm_pred = Vm_pred + torch.randn_like(Vm_pred) * noise_scale
    Va_pred = Va_pred + torch.randn_like(Va_pred) * noise_scale
    
    print(f"  Vm_pred shape: {Vm_pred.shape}")
    print(f"  Va_pred shape: {Va_pred.shape}")
    
    # 提取负荷数据
    num_pd = len(sys_data.idx_Pd)
    num_qd = len(sys_data.idx_Qd)
    Pd = batch_x[:, :num_pd]
    Qd = batch_x[:, num_pd:num_pd + num_qd]
    
    print(f"  Pd shape: {Pd.shape}")
    print(f"  Qd shape: {Qd.shape}")
    
    # ==================== 分析 1: 损失分量数值 ====================
    print("\n" + "=" * 60)
    print("分析 1: 损失分量数值")
    print("=" * 60)
    
    results = compute_gradient_norms(loss_fn, Vm_pred, Va_pred, Pd, Qd)
    
    print("\n【未加权损失值】")
    for name, value in results['loss_values'].items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")
    
    print("\n【权重】")
    for name, value in results['weights'].items():
        print(f"  {name}: {value:.2f}")
    
    print("\n【加权后损失值】")
    for name, value in results['weighted_loss'].items():
        print(f"  {name}: {value:.6f}")
    
    # 分析关键比例
    L_obj = results['weighted_loss']['L_objective']
    L_d_weighted = results['weighted_loss']['k_d * L_d']
    L_cons = results['weighted_loss']['L_constraints']
    
    print("\n【关键比例分析】")
    total = L_obj + L_cons
    if total > 0:
        print(f"  L_objective / Total: {L_obj / total * 100:.2f}%")
        print(f"  L_constraints / Total: {L_cons / total * 100:.2f}%")
    if L_cons > 0:
        print(f"  k_d * L_d / L_constraints: {L_d_weighted / L_cons * 100:.2f}%")
    
    # ==================== 分析 2: 梯度贡献 ====================
    print("\n" + "=" * 60)
    print("分析 2: 梯度贡献分析")
    print("=" * 60)
    
    grad_results = analyze_gradient_ratio(loss_fn, Vm_pred, Va_pred, Pd, Qd)
    
    print("\n【各损失分量的梯度范数】")
    for component, data in grad_results.items():
        print(f"  {component}:")
        print(f"    损失值: {data['loss_value']:.6f}")
        print(f"    梯度范数: {data['grad_total_norm']:.6f}")
        if 'contribution_ratio' in data:
            print(f"    贡献比例: {data['contribution_ratio'] * 100:.2f}%")
    
    # ==================== 分析 3: 问题诊断 ====================
    print("\n" + "=" * 60)
    print("分析 3: 问题诊断")
    print("=" * 60)
    
    # 检查 L_d 的梯度是否太小
    load_dev_grad = grad_results.get('load_dev', {}).get('grad_total_norm', 0)
    objective_grad = grad_results.get('objective', {}).get('grad_total_norm', 0)
    
    print(f"\n  目标梯度范数: {objective_grad:.6f}")
    print(f"  负荷偏差梯度范数: {load_dev_grad:.6f}")
    
    if load_dev_grad > 0 and objective_grad > 0:
        grad_ratio = load_dev_grad / objective_grad
        print(f"\n  负荷偏差梯度 / 目标梯度 = {grad_ratio:.4f}")
        
        if grad_ratio < 0.1:
            print("  ⚠️  问题：负荷偏差的梯度贡献不足目标的 10%！")
            print("     即使权重很高，梯度仍被目标主导。")
        elif grad_ratio < 1.0:
            print("  ⚠️  注意：负荷偏差的梯度贡献小于目标。")
        else:
            print("  ✓  负荷偏差梯度贡献充足。")
    elif load_dev_grad == 0:
        print("  ❌ 严重问题：负荷偏差梯度为 0！")
        print("     可能是 L_d 计算方式导致梯度消失。")
    
    # 检查损失值比例
    L_d_raw = results['loss_values']['L_d']
    k_d = results['weights']['k_d']
    
    print(f"\n  原始 L_d = {L_d_raw:.6f}")
    print(f"  权重 k_d = {k_d:.2f}")
    print(f"  加权 k_d * L_d = {L_d_raw * k_d:.6f}")
    
    if L_d_raw < 1e-6:
        print("  ⚠️  问题：L_d 值非常小，可能是计算方式导致梯度消失！")
    
    # ==================== 分析 4: 数值范围检查 ====================
    print("\n" + "=" * 60)
    print("分析 4: 数值范围检查")
    print("=" * 60)
    
    print(f"\n  Vm_pred 范围: [{Vm_pred.min().item():.4f}, {Vm_pred.max().item():.4f}]")
    print(f"  Va_pred 范围: [{Va_pred.min().item():.4f}, {Va_pred.max().item():.4f}]")
    print(f"  Pd 范围: [{Pd.min().item():.4f}, {Pd.max().item():.4f}]")
    print(f"  Qd 范围: [{Qd.min().item():.4f}, {Qd.max().item():.4f}]")
    
    # ==================== 建议 ====================
    print("\n" + "=" * 60)
    print("建议")
    print("=" * 60)
    
    print("""
    1. 如果 L_d 的梯度贡献很小：
       - 检查 L_d 的计算方式，可能需要放大数值范围
       - 考虑使用相对误差而非绝对误差
       - 考虑对 L_d 取 log 或使用 Huber Loss
    
    2. 如果 L_d 值本身很小：
       - 负荷偏差计算可能有问题
       - 单位可能不一致（p.u. vs MW）
    
    3. 考虑使用更激进的策略：
       - 两阶段训练：先只优化约束，再优化目标
       - 约束满足时才计算目标梯度
       - 使用 Lagrangian 方法或约束裁剪
    """)
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()
