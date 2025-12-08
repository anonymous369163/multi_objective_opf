"""
对比实验: 原版 vs 改进版
验证使用anchor作为起点是否能提升流模型性能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch 
from load_opf_data_v2 import OPF_Flow_Dataset_V2, filter_training_data
from utiles_v2 import train_all_v2
from default_args import modify_args 
import numpy as np
from models.actor import Actor
from post_processing import apply_post_processing
# from feasibility_corrector import load_feas_corrector  # 暂时注释，模块可能缺失


def get_unified_config(debug_mode=False):
    """
    统一的训练配置参数（合并了 get_training_config 和 opf_flow_args）
    
    Args:
        debug_mode: 是否为调试模式
    
    Returns:
        config: 统一的配置字典
    """
    # 模型目录
    model_dir = 'models/h512_l5_b512_lr0.001_wd1e-06_wc0.1_cg[True]_ctF_tmst'
    
    config = {
        # ==================== 数据配置 ====================
        # 'data_path': '../saved_data/training_data_case118_40k_preferences.npz',         
        'data_path': '../saved_data/training_data_case118_40k.npz',         # 单目标下（不考虑碳税率下）的数据集路径
        # ==================== 模型配置 ====================
        # 可选模型类型: 'vae', 'rectified', 'boosted_rectified', 'simple', 'latent_flow_vae', 'riemannian', 'deepopf_mlp'
        # riemannian: Riemannian Flow Matching，在训练时将目标向量投影到切空间
        # deepopf_mlp: 与 DeepOPV-V.ipynb 完全一致的双网络 MLP
        # 'model_list': ['simple', 'vae', 'rectified'],  # 评估三个模型
        # 'networks': ['mlp', 'mlp', 'mlp'],  # 对应每个模型的网络类型  
        # 'model_list': ['riemannian'],  # 测试 RFM 训练
        # 'model_list': ['rectified', 'reflow'],  # 对比 Rectified Flow 和 Reflow
        # 'model_list': ['reflow'],  # 仅测试改进后的 Reflow
        # 'model_list': ['simple', 'vae', 'rectified', 'deepopf_mlp'],  # 对比 MLP, VAE, RectifiedFlow 和 DeepOPF 双网络
        # 'networks': ['mlp', 'mlp', 'mlp', 'deepopf_mlp'],  # 对应每个模型的网络类型
        # 'model_list': ['deepopf_mlp'],  # 单独测试 DeepOPF 双网络
        # 'networks': ['deepopf_mlp'],  # 对应每个模型的网络类型  
        'model_list': ['simple', 'vae', 'rectified'],  # 对比 MLP, VAE, RectifiedFlow (带约束引导)
        'networks': ['mlp', 'mlp', 'mlp'],  # 对应每个模型的网络类型  
        
        'add_carbon_tax': False,  # 表示数据包含碳税作为输入
        # 'constraints_guided': [False, False, True],  # 各模型的约束引导配置
        # 'constraints_guided': [False, False, False, False],  # 与 DeepOPF 对齐，不使用约束引导（纯 MSE）
        # 'constraints_guided': [False],  # 与 DeepOPF 对齐，不使用约束引导（纯 MSE）
        'constraints_guided': [True, True, True],  # 启用约束引导
        'train_': True,  # 训练模式
        'test_': True,
        'sample_num': 1,  # 生成模型产生的样本数
        
        # ==================== 训练参数 ====================
        'num_iteration': 10000,  # 与论文一致，充分训练
        'test_freq': 100,  # 多长时间打印下loss
        'batch_dim': 512,
        'hidden_dim': 512,
        'num_layer': 5,
        'output_act': None,
        'w_constraints': 1.0,  # 约束损失权重 (由于移除了*100放大因子，从0.01调整为1.0保持相同有效权重)
        
        # ==================== 优化器参数 ====================
        'learning_rate': 1e-3,
        'learning_rate_decay': [1000, 0.9],    # 0.95
        'weight_decay': 1e-6,
        
        # ==================== VAE/Flow latent维度 ====================
        'latent_dim_vae': 32,      # VAE模型使用
        'latent_dim_flow': 64,    # Latent Flow VAE的潜在空间维度 (推荐64-128)
        'vae_beta': 1,            # VAE的KL散度权重
        
        # ==================== 模型特定参数 ====================
        'num_cluster': 4,
        'update_generator_freq': 5,
        'output_norm': False,
        'test_dim': 1024,
        'time_step': 1000,
        'inf_step': 100,
        'eta': 0.5,
        'ode_solver': 'Euler',
        
        # ==================== 碳税和模型路径配置 ====================  
        'pretrain_model_path': f'{model_dir}/vae_mlp_separate_training_add_carbon_tax_False_20251205_153826_best.pth',  # 单目标的VAE模型
        'first_stage_model_path': f'{model_dir}/rectified_mlp_separate_training_add_carbon_tax_False_20251205_154927_best.pth',  # 用于评估的 rectified 模型
        
        # ==================== 各模型评估路径 ====================
        'model_paths': {
            'simple': f'{model_dir}/simple_mlp_separate_training_add_carbon_tax_False_20251205_153351_best.pth',
            'vae': f'{model_dir}/vae_mlp_separate_training_add_carbon_tax_False_20251205_153826_best.pth',
            'rectified': f'{model_dir}/rectified_mlp_separate_training_add_carbon_tax_False_20251205_154927_best.pth',
            'reflow': f'{model_dir}/reflow_mlp_separate_training_add_carbon_tax_False_20251207_212658_best.pth',
        },
        
        # ==================== 引导配置 ====================
        'guidance_config': {
            'enabled': False,    # todo: 这块注意我开启了梯度引导策略了
            'scale': 1,        # 目前效果最好的情况是设置为0.1
            'perp_scale': 0.01,
            'start_time': 0.7
        },

        'evolutionary_config': {
            'enabled': False,
            'method': 'DE',
            'start_time': 0.70,
            'de_F': 0.4,
            'de_CR': 0.4,
            'de_strategy': 'current-to-best/1',
            'verbose': False
        },
        
        # ==================== Drift-Correction 流形稳定化配置 ====================
        # 支持三种模式:
        #   - 'jacobian': 每步使用 Jacobian 计算修正（精确但慢，174ms/样本）
        #   - 'sparse_jacobian': 每隔 N 步使用 Jacobian（推荐，40ms/样本，快4x）
        #   - 'learned': 使用 v_feas 网络（当前方向精度不足，暂不可用）
        'projection_config': {
            'enabled': True,              # 启用 Drift-Correction
            'mode': 'sparse_jacobian',    # 推荐：稀疏 Jacobian（速度与精度的最佳平衡）
            'start_time': 0.7,            # 从 t=0.7 开始应用
            'lambda_cor': 1.5,            # 法向修正增益
            'verbose': False,             # 是否打印调试信息
            # sparse_jacobian 模式参数:
            'sparse_interval': 5,         # 每5步使用一次 Jacobian（约束违反 1.4 vs 0.27）
            # learned 模式额外参数（当前不推荐）:
            'feas_model': None,
            'feas_model_path': 'models/feas_corrector/feas_corrector_best.pth',
            'max_correction_norm': 10.0,
            'fallback_to_jacobian': True,
        },
        
        # ==================== Riemannian Flow Matching (RFM) 训练配置 ====================
        # RFM 训练的核心改进：从"中性稳定"变为"渐进稳定"系统
        # 训练目标：v = P_tan @ (y-z) + Clip(λ*F^+@f(y_t), -C, C)
        #         = 切向（去终点） + 法向（回流形）
        # 使用方法：将 model_list 中的 'rectified' 替换为 'riemannian'
        'rfm_config': {
            'enabled': True,              # 是否启用 RFM 训练
            'freeze_interval': 10,        # 每 10 步更新一次 P_tan（减少 Jacobian 计算）
            'soft_weight': 1.0,           # 软约束权重（1.0=纯投影+修正）
            
            # === 法向修正参数（核心改进）===
            'lambda_cor': 5.0,            # 法向修正增益（建议 5.0-10.0）
            'add_perturbation': True,     # 训练时添加扰动模拟漂移
            'perturbation_scale': 0.05,   # 扰动幅度（相对于状态范围）
            'max_correction_norm': 10.0,  # Clip 上限，防止数值爆炸
            
            # === 损失函数参数 ===
            'mse_weight': 1.0,            # MSE 损失权重
            'direction_weight': 0.0,      # 方向损失权重（可选）
            'normal_weight': 0.0,         # 法向惩罚权重（可选）
        },
        
        # ==================== Reflow (带 Jacobian 后处理的自蒸馏) 配置 ====================
        # Reflow 是一种两阶段训练方法：
        #   1. 第一阶段：用已训练好的 Rectified Flow 模型 + Jacobian 修正生成轨迹
        #   2. 第二阶段：用这些轨迹训练 Student 模型，使其学会"内化"修正能力
        # 推理时 Student 模型不需要 Jacobian 后处理！
        # 使用方法：将 model_list 中的 'rectified' 替换为 'reflow'
        'reflow_config': {
            'enabled': True,
            
            # === Teacher 模型路径（第一阶段训好的 Rectified Flow）===
            'teacher_model_path': 'models/h512_l5_b512_lr0.001_wd1e-06_wc0.1_cg[True]_ctF_tmst/rectified_mlp_separate_training_add_carbon_tax_False_20251205_154927_best.pth',
            
            # === 轨迹生成参数（增强版）===
            'ode_step': 0.02,             # ODE 步长（0.02 = 50步）
            'correction_interval': 5,     # 每 5 步进行一次 Jacobian 修正（更频繁）
            'lambda_cor': 1.5,            # 法向修正增益
            'save_interval': 5,           # 每 5 步保存一个轨迹点（更密集）
            'start_correction_t': 0.2,    # 从 t=0.2 开始应用修正（更早）
            'trajectory_epochs': 1,       # 生成轨迹时遍历数据集的次数
            'max_samples': 16000,         # 使用 16000 个样本（增加一倍）
            
            # === Student 训练参数 ===
            'use_pure_trajectory': False, # False=混合训练（推荐，保持分布 + 学习修正）
            'lambda_reflow': 2.0,         # Reflow 损失权重（增大以强调轨迹学习）
            
            # === 数据路径 ===
            'trajectory_data_path': 'data/reflow_trajectories_v2.pt',
            'regenerate_trajectories': True,  # 重新生成带扰动的轨迹
        },
        
        # ==================== 训练模式和实例配置 ==================== 
        'train_mode': 'separate_training',  # joint_training, separate_training
        # ==================== 其他配置 ====================
        'debug_mode': debug_mode,
    }
    
    # ==================== 单目标配置 ====================
    config['single_target'] = True if 'preferences' not in config['data_path'] else False
    config['guidance_config']['single_target'] = config['single_target']
    config['evolutionary_config']['single_target'] = config['single_target']
    config['projection_config']['single_target'] = config['single_target']
    
    print(f"\n参数配置:")
    print(f"  - 批次大小: {config['batch_dim']}")
    print(f"  - 隐藏层维度: {config['hidden_dim']}")
    print(f"  - 网络层数: {config['num_layer']}")
    print(f"  - 训练迭代数: {config['num_iteration']}")
    
    return config


def evaluate_model(model, model_type, x_test, y_test, args, objective_fn=None, guidance_config=None, 
                   sample_num=1, device='cuda', apply_post_process=False, env=None, 
                   cost_fn=None, return_details=False, projection_config=None):
    """
    评估模型在测试集上的性能
    
    Args:
        model: 训练好的模型
        model_type: 模型类型 (simple, rectified, gan, etc.)
        x_test: 测试输入 (test_dim, input_dim)
        y_test: 测试目标 (test_dim, output_dim)
        args: 参数字典
        sample_num: 采样数量（用于生成式模型）
        device: 计算设备
        apply_post_process: 是否应用后处理修正
        env: 电网环境对象（后处理时需要）
        cost_fn: 经济成本计算函数
        return_details: 是否返回详细评估结果
        projection_config: 约束切空间投影配置（用于 rectified 模型）
    
    Returns:
        如果 return_details=False:
            test_loss: 测试损失（标量）
            constraint_loss: 约束损失（标量）
            constraint_loss_post: 后处理后的约束损失（如果apply_post_process=True）
        如果 return_details=True:
            返回包含所有详细指标的字典
    """
    import time
    timing_info = {}  # 用于存储各阶段耗时
    
    model.eval()
    test_dim = x_test.shape[0]
    output_dim = y_test.shape[1]  # 从y_test获取output_dim
    
    # with torch.no_grad():
    # 根据模型类型生成anchor（如果需要）
    if model_type in ['rectified', 'riemannian', 'reflow']:
        # 使用VAE模型生成锚点
        with torch.no_grad():
            x_test_pretrain = x_test[:, :-1] if 'True' not in args['pretrain_model_path'] and not args['single_target'] else x_test
            y_anchor_test = model.pretrain_model(x_test_pretrain).to(device)
    else:
        y_anchor_test = None
    
    # 根据不同模型类型生成预测
    t_inference_start = time.time()
    
    if model_type == 'simple':
        with torch.no_grad():
            y_pred = model(x_test)
    
    elif model_type == 'deepopf_mlp':
        with torch.no_grad():
            y_pred_deepopf = model(x_test)
            # 转换回标准格式
            n_bus = output_dim // 2
            Vm_pred_deepopf = y_pred_deepopf[:, :n_bus]
            Va_pred_deepopf = y_pred_deepopf[:, n_bus:]
            # 转换 Vm: [0,10] -> [-1,1]
            Vm_pred = (Vm_pred_deepopf / model.scale_vm) * 2 - 1
            # 转换 Va: Va_deepopf -> [-1,1]
            Va_pred = Va_pred_deepopf / model.scale_va
            y_pred = torch.cat([Vm_pred, Va_pred], dim=1)
        
    elif model_type in ['cluster', 'hindsight']:
        with torch.no_grad():
            y_pred = model(x_test).view(x_test.shape[0], -1, args['num_cluster'])
        
    elif model_type in ['gan', 'wgan', 'vae']:
        with torch.no_grad():
            x_test_repeated = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
            z_test = torch.randn(size=[x_test_repeated.shape[0], args['latent_dim']]).to(device)
            y_pred = model(x_test_repeated, z_test)
        
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation', 'riemannian', 'reflow']:
        # 流模型从anchor开始（添加高斯噪声）
        x_test_repeated = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
        z_test = torch.repeat_interleave(y_anchor_test, repeats=sample_num, dim=0)
        # z_test = z_test_mean + torch.randn_like(z_test_mean).to(device) * 0.1
        # 将锚点拼接到条件x中: x_test_repeated现在包含[负荷, 碳税, 锚点]
        # x_test_repeated = torch.concat([x_test_repeated, z_test], dim=1)
        y_pred, _ = model.flow_backward(x_test_repeated, z_test, step=1/args['inf_step'], method='Euler',
            objective_fn=objective_fn, guidance_config=guidance_config, projection_config=projection_config)
        
    elif model_type == 'diffusion':
        with torch.no_grad():
            x_test_repeated = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
            z_test = torch.randn(size=[x_test_repeated.shape[0], output_dim]).to(device)
            y_pred = model.diffusion_backward(x_test_repeated, z_test, args['inf_step'], eta=args['eta'])
        
    elif model_type in ['potential']:
        with torch.enable_grad():
            x_test_repeated = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
            y_pred = model.flow_backward(x_test_repeated, 1/args['inf_step'], method=args['ode_solver'])
            
    elif model_type in ['consistency_training', 'consistency_distillation']:
        x_test_repeated = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
        y_pred = model.sampling(x_test_repeated, inf_step=1) 
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented")
    
    t_inference_end = time.time()
    timing_info['inference_time'] = t_inference_end - t_inference_start
    timing_info['inference_time_per_sample'] = timing_info['inference_time'] / test_dim
    
    # 计算测试损失
    if sample_num > 1 and model_type not in ['cluster', 'hindsight']:
        # 从多个采样中选择最好的
        # 正确的重塑方式：先(test_dim, sample_num, output_dim)，再转置为(test_dim, output_dim, sample_num)
        y_pred = y_pred.view(test_dim, sample_num, -1).transpose(1, 2)  # (test_dim, output_dim, sample_num)
        y_test_expanded = y_test.unsqueeze(-1).expand_as(y_pred)  # 扩展到相同形状
        # 计算每个采样的MSE: (test_dim, sample_num)
        sample_losses = torch.nn.functional.mse_loss(y_pred, y_test_expanded).mean(dim=1)
        # 每个测试样本选最好的采样，然后求平均
        test_loss = sample_losses.min(dim=1)[0].mean().item()
        best_indices = sample_losses.min(dim=1)[1]  # (test_dim,)
        y_pred_best = y_pred[torch.arange(test_dim), :, best_indices]  # (test_dim, output_dim)
        Vm, Va = y_pred_best[:, :output_dim//2], y_pred_best[:, output_dim//2:]
        constraint_loss, _ = objective_fn(Vm, Va, x_test[:, :-1], 'mean')
    else:
        test_loss = torch.nn.functional.mse_loss(y_pred, y_test).item()

        # 另外我想要计算预测的电压和相交，最终得到的功率值是否满足约束条件：
        Vm, Va = y_pred[:, :output_dim//2], y_pred[:, output_dim//2:]
        if not args['add_carbon_tax']:
            constraint_loss, _ = objective_fn(Vm, Va, x_test, 'mean')
        else:
            constraint_loss, _ = objective_fn(Vm, Va, x_test[:, :-1], 'mean')
    
    # 确保 constraint_loss 是标量
    if isinstance(constraint_loss, torch.Tensor):
        constraint_loss = constraint_loss.item()
    
    # 获取详细约束信息
    x_input = x_test[:, :-1] if args['add_carbon_tax'] else x_test
    _, constraint_details = objective_fn(Vm, Va, x_input, 'mean')
    
    # 计算经济成本
    economic_cost = None
    if cost_fn is not None:
        economic_cost = cost_fn(Vm, Va, x_input, env, 'mean')
        if isinstance(economic_cost, torch.Tensor):
            economic_cost = economic_cost.item()

    # 应用后处理修正（如果启用）
    constraint_loss_post = None
    economic_cost_post = None
    constraint_details_post = None
    
    if apply_post_process and env is not None:
        # ==================== 多次迭代后处理 ====================
        t_post_start = time.time()
        
        constraint_loss_original = constraint_loss
        max_iterations = 10  # 最大迭代次数
        convergence_threshold = 0.01  # 收敛阈值：约束损失变化小于1%则停止
        target_constraint = 0.5  # 目标约束损失
        
        Vm_current, Va_current = Vm.clone(), Va.clone()
        
        print(f"\n[多次迭代后处理] 最大迭代: {max_iterations}, 目标约束: {target_constraint}")
        print(f"  Iteration 0: 约束损失 = {constraint_loss_original:.6f}")
        
        iteration_times = []  # 记录每次迭代的耗时
        for iteration in range(max_iterations):
            t_iter_start = time.time()
            # 应用后处理
            Vm_corrected, Va_corrected, correction_info = apply_post_processing(
                Vm_current, Va_current, x_input, env, k_dV=1.0, verbose=False, debug_mode=0
            )
            t_iter_end = time.time()
            iteration_times.append(t_iter_end - t_iter_start)
            
            # 计算新的约束损失
            constraint_loss_new, _ = objective_fn(Vm_corrected, Va_corrected, x_input, 'mean')
            if isinstance(constraint_loss_new, torch.Tensor):
                constraint_loss_new = constraint_loss_new.item()
            
            # 计算改善幅度
            if iteration == 0:
                prev_loss = constraint_loss_original
            improvement = (prev_loss - constraint_loss_new) / prev_loss * 100 if prev_loss > 0 else 0
            
            print(f"  Iteration {iteration + 1}: 约束损失 = {constraint_loss_new:.6f} (改善 {improvement:+.2f}%)")
            
            # 更新电压
            Vm_current, Va_current = Vm_corrected, Va_corrected
            
            # 检查是否达到目标
            if constraint_loss_new < target_constraint:
                print(f"  [达到目标] 约束损失 {constraint_loss_new:.6f} < {target_constraint}")
                break
            
            # 检查是否收敛
            if abs(improvement) < convergence_threshold:
                print(f"  [收敛] 改善幅度 {abs(improvement):.4f}% < {convergence_threshold}%")
                break
            
            prev_loss = constraint_loss_new
        
        constraint_loss_post = constraint_loss_new
        
        t_post_end = time.time()
        
        # 获取后处理后的详细约束信息
        _, constraint_details_post = objective_fn(Vm_corrected, Va_corrected, x_input, 'mean')
        
        # 计算后处理后的经济成本
        if cost_fn is not None:
            economic_cost_post = cost_fn(Vm_corrected, Va_corrected, x_input, env, 'mean')
            if isinstance(economic_cost_post, torch.Tensor):
                economic_cost_post = economic_cost_post.item()
        
        # ==================== 时间统计 ====================
        timing_info['post_processing_total'] = t_post_end - t_post_start
        timing_info['post_processing_iterations'] = iteration + 1
        timing_info['post_processing_per_iteration'] = sum(iteration_times) / len(iteration_times)
        timing_info['post_processing_per_sample'] = timing_info['post_processing_total'] / test_dim
        
        # ==================== 结果总结 ====================
        print(f"\n[Post-Processing Summary]")
        print(f"  总迭代次数: {iteration + 1}")
        print(f"  约束损失变化: {constraint_loss_original:.6f} -> {constraint_loss_post:.6f} ({(constraint_loss_post - constraint_loss_original)/constraint_loss_original * 100:+.2f}%)")
        if economic_cost is not None and economic_cost_post is not None:
            print(f"  经济成本变化: {economic_cost:.2f} -> {economic_cost_post:.2f} ($/h)")
        
        # ==================== 时间统计输出 ====================
        print(f"\n[Timing Statistics]")
        print(f"  推理时间 (含Drift-Correction): {timing_info['inference_time']:.4f}s ({timing_info['inference_time_per_sample']*1000:.2f}ms/样本)")
        print(f"  后处理总时间: {timing_info['post_processing_total']:.4f}s ({timing_info['post_processing_per_sample']*1000:.2f}ms/样本)")
        print(f"  后处理每次迭代: {timing_info['post_processing_per_iteration']*1000:.2f}ms")
        print(f"  总时间: {timing_info['inference_time'] + timing_info['post_processing_total']:.4f}s")
    else:
        # 如果不使用后处理，也打印基本的时间统计
        print(f"\n[Timing Statistics]")
        print(f"  推理时间 (含Drift-Correction): {timing_info['inference_time']:.4f}s ({timing_info['inference_time_per_sample']*1000:.2f}ms/样本)")

    # 计算 DeepOPF 风格的评估指标
    from models.actor import Actor as ActorClass
    if isinstance(model, ActorClass) or hasattr(model, 'compute_deepopf_metrics'):
        # 如果模型是 Actor 或有 compute_deepopf_metrics 方法
        actor_for_metrics = model
    else:
        # 否则使用外部 actor_helper (通过 objective_fn 获取)
        actor_for_metrics = None
    
    # 通过 objective_fn 的包装获取 actor_helper
    deepopf_metrics = None
    if env is not None:
        try:
            # 创建临时 Actor 用于计算 DeepOPF 指标
            temp_actor = Actor(input_dim=x_input.shape[1], env=env, output_dim=output_dim//2)
            temp_actor.eval()
            with torch.no_grad():
                deepopf_metrics = temp_actor.compute_deepopf_metrics(Vm, Va, x_input, env)
        except Exception as e:
            print(f"[Warning] 计算 DeepOPF 指标时出错: {e}")
    
    # 返回结果
    if return_details:
        result = {
            'test_loss': test_loss,
            'constraint_loss': constraint_loss,
            'constraint_details': constraint_details,
            'economic_cost': economic_cost,
            'timing_info': timing_info,  # 添加时间统计信息
            'deepopf_metrics': deepopf_metrics,  # DeepOPF 风格的评估指标
        }
        if apply_post_process:
            result['constraint_loss_post'] = constraint_loss_post
            result['constraint_details_post'] = constraint_details_post
            result['economic_cost_post'] = economic_cost_post
        return result
    else:
        if apply_post_process:
            return test_loss, constraint_loss, constraint_loss_post
        else:
            return test_loss, constraint_loss 




def main(env=None, Actor=None, debug_mode=False):
    """
    主函数，用于训练或评估模型
    
    Args:
        env: 电网环境
        Actor: Actor类
        debug_mode: 是否为调试模式
    """
    import pandas as pd
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}\n")
    
    # 获取统一的训练配置
    config = get_unified_config(debug_mode=debug_mode)
    
    # 应用 modify_args 进行必要的修改
    args = modify_args(config)
    args['env'] = env
    
    # 解包配置
    data_path = config['data_path']
    model_list = config['model_list']
    networks = config['networks']
    constraints_guideds = config['constraints_guided']
    sample_num = config['sample_num']
    guidance_config = config['guidance_config']
    projection_config = config['projection_config']
    model_paths = config.get('model_paths', {})
    
    # 根据模型类型设置latent_dim
    if 'vae' in model_list:
        args['latent_dim'] = config['latent_dim_vae']
    else:
        args['latent_dim'] = config['latent_dim_flow']
        
    # 加载数据 
    add_carbon_tax = args['add_carbon_tax'] 
    single_target = True if 'preferences' not in data_path else False        
    args['single_target'] = single_target
    data_v2 = OPF_Flow_Dataset_V2(data_path, device=DEVICE, test_ratio=0.2, random_seed=42, 
                                   add_carbon_tax=add_carbon_tax, single_target=single_target)
    
    print(f"\n数据分割信息:")
    print(f"  - 训练集样本: {data_v2.num_train_samples}")
    print(f"  - 测试集样本: {data_v2.num_test_samples}")

    # 创建 Actor 实例用于计算约束和成本
    actor_helper = Actor(input_dim=189, env=args['env'], output_dim=118).to(DEVICE)  
    actor_helper.eval()
    
    # 定义约束损失函数
    def objective_fn_train(Vm, Va, x_input, reduction):
        """训练用的约束损失函数，不返回 details"""
        return actor_helper.compute_constraint_loss(Vm, Va, x_input, env, reduction=reduction, return_details=False)
    
    def objective_fn(Vm, Va, x_input, reduction):
        """评估用的约束损失函数，返回 details"""
        return actor_helper.compute_constraint_loss(Vm, Va, x_input, env, reduction=reduction, return_details=True)
    
    # 定义经济成本计算函数
    def cost_fn(Vm, Va, x_input, env, reduction):
        """计算经济成本"""
        return actor_helper.compute_economic_cost(Vm, Va, x_input, env, reduction=reduction)

    # 设置 projection_config 的 env 引用
    projection_config['env'] = env
    
    # 加载 v_feas 模型（如果使用 learned 模式）
    if projection_config.get('mode') == 'learned':
        feas_model_path = projection_config.get('feas_model_path')
        if feas_model_path and os.path.exists(feas_model_path):
            try:
                from feasibility_corrector import load_feas_corrector
                print(f"\n[v_feas] 加载可行性修正网络: {feas_model_path}")
                # 加载检查点获取配置
                checkpoint = torch.load(feas_model_path, map_location=DEVICE, weights_only=False)
                feas_config = checkpoint.get('config', {
                    'state_dim': data_v2.output_dim,
                    'cond_dim': data_v2.input_dim,
                    'hidden_dim': 256,
                    'num_layers': 4,
                    'use_time_emb': True,
                })
                # 加载模型
                feas_model = load_feas_corrector(feas_model_path, feas_config, DEVICE)
                projection_config['feas_model'] = feas_model
                print(f"[v_feas] 模型加载成功，Val Cosine: {checkpoint.get('val_cosine', 'N/A')}")
            except ImportError:
                print(f"\n[v_feas] 警告: feasibility_corrector 模块不可用，回退到 jacobian 模式")
                projection_config['mode'] = 'jacobian'
        else:
            print(f"\n[v_feas] 警告: 未找到模型文件 {feas_model_path}，回退到 jacobian 模式")
            projection_config['mode'] = 'jacobian'

    import datetime
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"当前时间: {time_str}")
    
    # 使用固定的测试集进行评估
    torch.manual_seed(42)  # 确保可复现
    if data_v2.num_test_samples > 500:
        test_indices = torch.randperm(data_v2.num_test_samples)[:500]
        x_test = data_v2.x_test[test_indices].to(DEVICE)
        y_test = data_v2.y_test[test_indices].to(DEVICE)
    else:
        x_test = data_v2.x_test.to(DEVICE)
        y_test = data_v2.y_test.to(DEVICE) 
    
    # 初始化结果字典
    results_v2 = {}
    
    for ith, model_type in enumerate(model_list): 
        print(f"\n{'='*70}")
        print(f"评估模型: {model_type}")
        print(f"{'='*70}")
        
        args['network'] = networks[ith]
        args['constraints_guided'] = constraints_guideds[ith]
        args['latent_dim'] = config['latent_dim_vae'] if model_type == 'vae' else config['latent_dim_flow']
        
        if args['train_']:
            # 训练模式
            
            # Reflow 特殊处理：需要先生成矫正轨迹
            reflow_loader = None
            if model_type == 'reflow':
                from reflow_utils import ReflowTrajectoryGenerator, ReflowDataset, create_reflow_dataloader
                from torch.utils.data import TensorDataset, DataLoader
                
                reflow_config = args.get('reflow_config', {})
                trajectory_path = reflow_config.get('trajectory_data_path', 'data/reflow_trajectories.pt')
                regenerate = reflow_config.get('regenerate_trajectories', True)
                
                # 检查是否需要生成轨迹
                if regenerate or not os.path.exists(trajectory_path):
                    print("\n[Reflow] Stage 1: Generating corrected trajectories...")
                    
                    # 加载 Teacher 模型
                    teacher_path = reflow_config.get('teacher_model_path', args['first_stage_model_path'])
                    print(f"  Loading Teacher model: {teacher_path}")
                    teacher_model = torch.load(teacher_path, weights_only=False).to(DEVICE)
                    teacher_model.eval()
                    
                    # 获取优化参数
                    ode_step = reflow_config.get('ode_step', 0.02)
                    correction_interval = reflow_config.get('correction_interval', 10)
                    save_interval = reflow_config.get('save_interval', 10)
                    max_samples = reflow_config.get('max_samples', None)
                    
                    print(f"  ODE step: {ode_step} ({int(1/ode_step)} steps)")
                    print(f"  Jacobian correction interval: every {correction_interval} steps")
                    print(f"  Save interval: every {save_interval} steps")
                    
                    # 创建轨迹生成器
                    trajectory_generator = ReflowTrajectoryGenerator(
                        teacher_model=teacher_model,
                        env=env,
                        device=DEVICE,
                        correction_interval=correction_interval,
                        lambda_cor=reflow_config.get('lambda_cor', 1.5),
                        save_interval=save_interval,
                        start_correction_t=reflow_config.get('start_correction_t', 0.3),
                    )
                    
                    # 创建数据加载器（支持 max_samples 限制，包含 y_train）
                    if max_samples is not None and max_samples < data_v2.x_train.shape[0]:
                        print(f"  Using {max_samples} samples (out of {data_v2.x_train.shape[0]})")
                        indices = torch.randperm(data_v2.x_train.shape[0])[:max_samples]
                        x_train_subset = data_v2.x_train[indices]
                        y_train_subset = data_v2.y_train[indices]
                    else:
                        x_train_subset = data_v2.x_train
                        y_train_subset = data_v2.y_train
                        
                    # 包含 x 和 y 以便计算终点信息
                    train_tensor = TensorDataset(x_train_subset, y_train_subset)
                    train_loader = DataLoader(train_tensor, batch_size=args['batch_dim'], shuffle=False)
                    
                    # 生成轨迹
                    os.makedirs('data', exist_ok=True)
                    trajectory_data = trajectory_generator.generate_dataset(
                        dataloader=train_loader,
                        anchor_generator=teacher_model.pretrain_model,
                        num_epochs=reflow_config.get('trajectory_epochs', 1),
                        step=ode_step,
                        save_path=trajectory_path,
                        verbose=True,
                    )
                else:
                    print(f"\n[Reflow] Using existing trajectory data: {trajectory_path}")
                    trajectory_data = torch.load(trajectory_path)
                
                # 创建 Reflow 数据集和加载器
                reflow_dataset = ReflowDataset(data_dict=trajectory_data)
                reflow_loader = create_reflow_dataloader(
                    reflow_dataset, 
                    batch_size=args['batch_dim'],
                    shuffle=True
                )
                
                # 将 reflow_loader 传递给 args，以便在 train_all_v2 中使用
                args['reflow_loader'] = reflow_loader
                
                print(f"  Reflow dataset size: {len(reflow_dataset)}")
                print("\n[Reflow] Stage 2: Training Student model...")
            
            model, _ = train_all_v2(data_v2, args, model_type, time_str, objective_fn_train)
        else:
            # 评估模式：加载已训练的模型
            model_path = model_paths.get(model_type, args['first_stage_model_path'])
            print(f"加载模型: {model_path}")
            model = torch.load(model_path, weights_only=False)
            model.eval()

        print('开始评估...')
        
        # 根据模型类型决定是否使用投影
        # 流模型类型使用约束切空间投影（推理时的 sparse_jacobian）
        # 注意：'reflow' 模型的目标是不需要推理时修正，所以默认不启用
        use_projection = model_type in ['rectified', 'gaussian', 'conditional', 'interpolation', 'riemannian']
        current_projection_config = projection_config if use_projection else None
        
        # 对于 reflow 模型，可以选择是否使用投影来对比效果
        if model_type == 'reflow':
            # reflow 模型的核心价值是不需要推理时修正
            # 如果需要对比，可以设置 reflow_use_projection=True
            reflow_use_projection = args.get('reflow_config', {}).get('use_projection_at_inference', False)
            current_projection_config = projection_config if reflow_use_projection else None
        
        # 使用详细评估模式
        result = evaluate_model(
            model, model_type, x_test, y_test, args, 
            objective_fn=objective_fn, guidance_config=None, 
            sample_num=sample_num, device=DEVICE,
            apply_post_process=True, env=env,
            cost_fn=cost_fn, return_details=True,
            projection_config=current_projection_config
        )
        
        # 存储结果
        results_v2[model_type] = result
        
        # 打印详细结果
        print(f"\n[{model_type}] 评估结果:")
        print(f"  MSE 损失:           {result['test_loss']:.6f}")
        print(f"  约束违反 (原始):    {result['constraint_loss']:.6f}")
        print(f"  约束违反 (后处理):  {result.get('constraint_loss_post', 'N/A')}")
        if result['economic_cost'] is not None:
            print(f"  经济成本 (原始):    {result['economic_cost']:.2f} $/h")
        if result.get('economic_cost_post') is not None:
            print(f"  经济成本 (后处理):  {result['economic_cost_post']:.2f} $/h")
        
        # 打印详细约束信息
        if result.get('constraint_details'):
            print(f"\n  约束违反详情 (原始):")
            for key, value in result['constraint_details'].items():
                print(f"    {key}: {value:.6f}")
        
        if result.get('constraint_details_post'):
            print(f"\n  约束违反详情 (后处理后):")
            for key, value in result['constraint_details_post'].items():
                print(f"    {key}: {value:.6f}")
        
        # 打印 DeepOPF 风格的评估指标
        if result.get('deepopf_metrics'):
            metrics = result['deepopf_metrics']
            print(f"\n  DeepOPF Metrics:")
            print(f"    Total Violation (p.u.):    {metrics['total_violation_pu']:.4f}")
            print(f"    Pg Satisfy Rate:           {metrics['pg_satisfy_rate']:.2f}%")
            print(f"    Qg Satisfy Rate:           {metrics['qg_satisfy_rate']:.2f}%")
            print(f"    Branch Satisfy Rate:       {metrics['branch_satisfy_rate']:.2f}%")
            print(f"    Avg Pg_max Violation Num:  {metrics['avg_pg_max_vio_count']:.2f}/{metrics['num_generators']}")
            print(f"    Max Pg Violation (p.u.):   {metrics['max_pg_violation']:.4f}")
    
    # ==================== 结果对比表格 ====================
    print("\n" + "="*100)
    print("模型性能对比总结")
    print("="*100)
    
    # 构建对比表格数据
    table_data = []
    for model_type in model_list:
        result = results_v2.get(model_type, {})
        row = {
            'Model': model_type,
            'MSE Loss': result.get('test_loss', float('inf')),
            'Constraint (Original)': result.get('constraint_loss', float('inf')),
            'Constraint (Post)': result.get('constraint_loss_post', None),
            'Cost (Original) $/h': result.get('economic_cost', None),
            'Cost (Post) $/h': result.get('economic_cost_post', None),
        }
        # 添加详细约束信息
        if result.get('constraint_details'):
            for key, value in result['constraint_details'].items():
                row[f'{key}_orig'] = value
        if result.get('constraint_details_post'):
            for key, value in result['constraint_details_post'].items():
                row[f'{key}_post'] = value
        table_data.append(row)
    
    # 打印主要对比表格
    print(f"\n{'Model':<12} {'MSE Loss':<12} {'Constr(Orig)':<14} {'Constr(Post)':<14} {'Cost(Orig)':<12} {'Cost(Post)':<12}")
    print("-"*80)
    for row in table_data:
        mse = f"{row['MSE Loss']:.6f}" if row['MSE Loss'] < float('inf') else "N/A"
        c_orig = f"{row['Constraint (Original)']:.4f}" if row['Constraint (Original)'] < float('inf') else "N/A"
        c_post = f"{row['Constraint (Post)']:.4f}" if row['Constraint (Post)'] is not None else "N/A"
        cost_orig = f"{row['Cost (Original) $/h']:.2f}" if row['Cost (Original) $/h'] is not None else "N/A"
        cost_post = f"{row['Cost (Post) $/h']:.2f}" if row['Cost (Post) $/h'] is not None else "N/A"
        print(f"{row['Model']:<12} {mse:<12} {c_orig:<14} {c_post:<14} {cost_orig:<12} {cost_post:<12}")
    
    # 打印详细约束对比表格
    print(f"\n{'Model':<12} {'Pg_max':<10} {'Pg_min':<10} {'Qg_max':<10} {'Qg_min':<10} {'Sf':<10} {'St':<10}")
    print("-"*80)
    for row in table_data:
        g1 = f"{row.get('g1_pmax_orig', 0):.4f}"
        g2 = f"{row.get('g2_pmin_orig', 0):.4f}"
        g5 = f"{row.get('g5_qmax_orig', 0):.4f}"
        g6 = f"{row.get('g6_qmin_orig', 0):.4f}"
        g9 = f"{row.get('g9_sf_orig', 0):.4f}"
        g10 = f"{row.get('g10_st_orig', 0):.4f}"
        print(f"{row['Model']:<12} {g1:<10} {g2:<10} {g5:<10} {g6:<10} {g9:<10} {g10:<10}")
    
    # 保存结果到CSV
    df = pd.DataFrame(table_data)
    csv_path = f'evaluation_results_{time_str}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")
    
    # 找到最优模型
    best_model_type = None
    best_loss = float('inf')
    for model_type in model_list:
        result = results_v2.get(model_type, {})
        loss = result.get('test_loss', float('inf'))
        if loss < best_loss:
            best_loss = loss
            best_model_type = model_type

    if best_model_type is not None and best_loss < float('inf'):
        print(f"\n[结论] MSE最优模型: {best_model_type}，测试损失: {best_loss:.6f}")
        
        # 约束违反最优
        best_constraint_model = min(model_list, 
                                     key=lambda m: results_v2.get(m, {}).get('constraint_loss_post', float('inf')) or float('inf'))
        best_constraint = results_v2.get(best_constraint_model, {}).get('constraint_loss_post', float('inf'))
        print(f"[结论] 约束违反最优模型: {best_constraint_model}，后处理约束损失: {best_constraint:.6f}")
        
        # 经济成本最优
        best_cost_model = min(model_list, 
                               key=lambda m: results_v2.get(m, {}).get('economic_cost_post', float('inf')) or float('inf'))
        best_cost = results_v2.get(best_cost_model, {}).get('economic_cost_post', float('inf'))
        if best_cost < float('inf'):
            print(f"[结论] 经济成本最优模型: {best_cost_model}，后处理经济成本: {best_cost:.2f} $/h")
    else:
        print("\n没有成功评估的模型。")
    
    return results_v2


if __name__ == "__main__":

    # 导入电网相关的类
    from env import PowerGridEnv

    num_timesteps = 288
    case_name = "case118"
    random_load = True
    run_pp = True
    consider_renewable_generation = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.actor import PowerSystemConfig
    ps_config = PowerSystemConfig(device=device, case_file_path='../saved_data/pglib_opf_case118.mat')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PowerGridEnv(num_timesteps=num_timesteps, case_name=case_name, random_load=random_load, run_pp=run_pp,
                        consider_renewable_generation=consider_renewable_generation, PowerSystemConfig=ps_config,
                        device=device, carbon_tax=0.0)

    
    results_v2 = main(env=env, Actor=Actor)

