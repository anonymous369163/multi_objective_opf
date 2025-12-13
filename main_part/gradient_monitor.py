"""
梯度监控工具：实时分析损失函数各分量的梯度贡献

在训练脚本中使用：
    from gradient_monitor import GradientMonitor
    
    monitor = GradientMonitor()
    
    # 在 backward 之前
    monitor.compute_component_gradients(loss_fn, Vm_pred, Va_pred, Pd, Qd, ...)
    
    # 定期打印
    if step % 100 == 0:
        monitor.print_summary()
"""

import torch
import numpy as np


class GradientMonitor:
    """监控各损失分量的梯度贡献"""
    
    def __init__(self):
        self.history = {
            'objective_grad': [],
            'gen_vio_grad': [],
            'branch_vio_grad': [],
            'load_dev_grad': [],
            'L_d_raw': [],
            'k_d': [],
        }
        
    def compute_component_gradients(self, loss_fn, Vm_pred, Va_pred, Pd, Qd, 
                                    lambda_cost=0.9, lambda_carbon=0.1):
        """
        分别计算各损失分量的梯度范数
        
        注意：这会增加计算开销，建议只在诊断时使用
        """
        device = Vm_pred.device
        
        # 保存原始权重
        orig_k_g = loss_fn.k_g
        orig_k_Sl = loss_fn.k_Sl
        orig_k_theta = loss_fn.k_theta
        orig_k_d = loss_fn.k_d
        
        # 如果使用自适应权重，从 scheduler 获取
        if hasattr(loss_fn, 'weight_scheduler') and loss_fn.weight_scheduler is not None:
            weights = loss_fn.weight_scheduler.get_weights()
            orig_k_g = weights['k_g']
            orig_k_Sl = weights['k_Sl']
            orig_k_theta = weights['k_theta']
            orig_k_d = weights['k_d']
        
        results = {}
        
        # 1. 目标损失梯度
        Vm = Vm_pred.detach().clone().requires_grad_(True)
        Va = Va_pred.detach().clone().requires_grad_(True)
        
        # 临时设置权重
        loss_fn.k_g = 0
        loss_fn.k_Sl = 0
        loss_fn.k_theta = 0
        loss_fn.k_d = 0
        if hasattr(loss_fn, 'weight_scheduler') and loss_fn.weight_scheduler is not None:
            loss_fn.weight_scheduler.k_g = 0
            loss_fn.weight_scheduler.k_Sl = 0
            loss_fn.weight_scheduler.k_theta = 0
            loss_fn.weight_scheduler.k_d = 0
        
        try:
            loss, _ = loss_fn(Vm, Va, Pd, Qd, lambda_cost=lambda_cost, 
                             lambda_carbon=lambda_carbon, return_details=True, update_weights=False)
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                grad_norm = torch.norm(torch.cat([Vm.grad.flatten(), Va.grad.flatten()])).item()
            else:
                grad_norm = 0
        except:
            grad_norm = 0
        results['objective'] = grad_norm
        
        # 2. 负荷偏差梯度（k_d * L_d）
        Vm = Vm_pred.detach().clone().requires_grad_(True)
        Va = Va_pred.detach().clone().requires_grad_(True)
        
        loss_fn.k_g = 0
        loss_fn.k_Sl = 0
        loss_fn.k_theta = 0
        loss_fn.k_d = orig_k_d
        if hasattr(loss_fn, 'weight_scheduler') and loss_fn.weight_scheduler is not None:
            loss_fn.weight_scheduler.k_g = 0
            loss_fn.weight_scheduler.k_Sl = 0
            loss_fn.weight_scheduler.k_theta = 0
            loss_fn.weight_scheduler.k_d = orig_k_d
        
        try:
            loss, loss_dict = loss_fn(Vm, Va, Pd, Qd, lambda_cost=0, lambda_carbon=0, 
                                      return_details=True, update_weights=False)
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                grad_norm = torch.norm(torch.cat([Vm.grad.flatten(), Va.grad.flatten()])).item()
            else:
                grad_norm = 0
            L_d_raw = loss_dict.get('load_dev', 0)
        except Exception as e:
            grad_norm = 0
            L_d_raw = 0
        results['load_dev'] = grad_norm
        
        # 恢复权重
        loss_fn.k_g = orig_k_g
        loss_fn.k_Sl = orig_k_Sl
        loss_fn.k_theta = orig_k_theta
        loss_fn.k_d = orig_k_d
        if hasattr(loss_fn, 'weight_scheduler') and loss_fn.weight_scheduler is not None:
            loss_fn.weight_scheduler.k_g = orig_k_g
            loss_fn.weight_scheduler.k_Sl = orig_k_Sl
            loss_fn.weight_scheduler.k_theta = orig_k_theta
            loss_fn.weight_scheduler.k_d = orig_k_d
        
        # 记录历史
        self.history['objective_grad'].append(results['objective'])
        self.history['load_dev_grad'].append(results['load_dev'])
        self.history['L_d_raw'].append(L_d_raw if isinstance(L_d_raw, (int, float)) else L_d_raw.item() if hasattr(L_d_raw, 'item') else 0)
        self.history['k_d'].append(orig_k_d)
        
        return results
    
    def print_summary(self, last_n=10):
        """打印最近 n 步的梯度统计"""
        print("\n" + "=" * 50)
        print("梯度监控摘要")
        print("=" * 50)
        
        if len(self.history['objective_grad']) == 0:
            print("  无数据")
            return
        
        obj_grad = self.history['objective_grad'][-last_n:]
        load_grad = self.history['load_dev_grad'][-last_n:]
        L_d_raw = self.history['L_d_raw'][-last_n:]
        k_d = self.history['k_d'][-last_n:]
        
        print(f"\n最近 {len(obj_grad)} 步统计:")
        print(f"  目标梯度范数:     avg={np.mean(obj_grad):.4f}, max={np.max(obj_grad):.4f}")
        print(f"  负荷偏差梯度范数: avg={np.mean(load_grad):.4f}, max={np.max(load_grad):.4f}")
        print(f"  L_d 原始值:       avg={np.mean(L_d_raw):.6f}")
        print(f"  k_d 权重:         avg={np.mean(k_d):.1f}")
        
        # 梯度比例
        if np.mean(obj_grad) > 0:
            ratio = np.mean(load_grad) / np.mean(obj_grad)
            print(f"\n  负荷梯度/目标梯度 = {ratio:.4f}")
            if ratio < 0.1:
                print("  ⚠️ 警告：负荷偏差梯度贡献不足 10%！")
        
        print("=" * 50)
    
    def get_latest_ratio(self):
        """获取最新的梯度比例"""
        if len(self.history['objective_grad']) == 0:
            return None
        obj = self.history['objective_grad'][-1]
        load = self.history['load_dev_grad'][-1]
        if obj > 0:
            return load / obj
        return None


def analyze_loss_landscape(loss_fn, Vm_base, Va_base, Pd, Qd, 
                          lambda_cost=0.9, lambda_carbon=0.1,
                          direction='load_dev', num_points=20):
    """
    分析损失函数在某个方向上的景观
    
    Args:
        direction: 'load_dev' 或 'objective' - 分析哪个方向
        num_points: 采样点数
    """
    import matplotlib.pyplot as plt
    
    device = Vm_base.device
    
    # 计算梯度方向
    Vm = Vm_base.detach().clone().requires_grad_(True)
    Va = Va_base.detach().clone().requires_grad_(True)
    
    loss, loss_dict = loss_fn(Vm, Va, Pd, Qd, lambda_cost=lambda_cost, 
                             lambda_carbon=lambda_carbon, return_details=True)
    loss.backward()
    
    grad_Vm = Vm.grad.clone()
    grad_Va = Va.grad.clone()
    
    # 归一化梯度方向
    grad_norm = torch.sqrt(torch.sum(grad_Vm**2) + torch.sum(grad_Va**2))
    grad_Vm = grad_Vm / grad_norm
    grad_Va = grad_Va / grad_norm
    
    # 在梯度方向上采样
    alphas = np.linspace(-0.5, 0.5, num_points)
    losses_total = []
    losses_obj = []
    losses_load = []
    
    for alpha in alphas:
        Vm_new = Vm_base.detach() + alpha * grad_Vm
        Va_new = Va_base.detach() + alpha * grad_Va
        
        with torch.no_grad():
            _, ld = loss_fn(Vm_new, Va_new, Pd, Qd, lambda_cost=lambda_cost, 
                           lambda_carbon=lambda_carbon, return_details=True, update_weights=False)
        
        losses_total.append(ld['cost'] + ld['gen_vio'] + ld['load_dev'])
        losses_obj.append(ld['cost'])
        losses_load.append(ld['load_dev'])
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(alphas, losses_total, 'b-', label='Total Loss')
    axes[0].set_xlabel('Step along gradient')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss Landscape')
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    axes[1].plot(alphas, losses_obj, 'g-', label='Objective')
    axes[1].set_xlabel('Step along gradient')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Objective Loss Landscape')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    axes[2].plot(alphas, losses_load, 'r-', label='Load Dev')
    axes[2].set_xlabel('Step along gradient')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Load Deviation Landscape')
    axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('loss_landscape_analysis.png', dpi=150)
    plt.close()
    
    print("\n损失景观分析已保存到 loss_landscape_analysis.png")
    
    return {
        'alphas': alphas,
        'losses_total': losses_total,
        'losses_obj': losses_obj,
        'losses_load': losses_load,
    }

