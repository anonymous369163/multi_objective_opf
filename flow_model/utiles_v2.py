"""
修改版训练函数 - 为从anchor到target的流模型优化 
"""
import torch
import numpy as np
import math
from models.actor import Actor
from torch.utils.tensorboard import SummaryWriter 

def train_all_v2(data, args, model_type, time_str, objective_fn=None, pretrain_model=None):
    """
    改进的训练函数，专门为从anchor到target的流模型设计
    
    关键区别:
    - 原版: z_batch = torch.randn_like(y_batch)  (随机噪声)
    - V2版: z_batch = y_anchor_batch  (使用真实的anchor作为起点)
    """
    from net_utiles import Simple_NN, GMM, VAE, GAN, WGAN, DM, FM, AM, CM, CD, LatentFlowVAE, DeepOPF_MLP
    import os

    add_carbon_tax = args['add_carbon_tax']
    
    # Unpack arguments
    input_dim = data.x_train.shape[-1]
    output_dim = data.y_train.shape[1]
    data_dim = data.x_train.shape[0]
    
    # Set training parameters
    instance = args['instance'] 
    num_epochs = args['num_iteration']
    batch_dim = args['batch_dim']
    hidden_dim = args['hidden_dim']
    num_layers = args['num_layer']
    output_act = args['output_act']
    network = args['network']
    pred_type = 'node'    # 'edge' if args['data_set'] == 'tsp' else 
    
    # Additional parameters
    num_cluster = args['num_cluster']
    latent_dim = args['latent_dim']
    time_step = args['time_step']
    output_norm = args['output_norm']
    noise_type = 'gaussian'
    train_mode = args['train_mode']
    vae_beta = args.get('vae_beta', 0.01)  # VAE的KL散度权重，默认0.01
    
    # 调整 input_dim：对于 carbon_tax_aware_mlp_v2，需要包含锚点维度
    # 因为实际训练时会拼接 x_batch + z_batch（锚点） 
    actual_input_dim = input_dim
    
    # 创建模型
    if model_type == 'simple':
        model = Simple_NN(network, actual_input_dim, output_dim, hidden_dim, num_layers, output_act, pred_type).to(data.device)
    elif model_type == 'deepopf_mlp':
        # DeepOPF 风格的双网络 MLP
        # 获取环境参数用于设置 Vm 边界
        env = args.get('env', None)
        if env is not None:
            VmLb = torch.tensor(env.Vm_min, dtype=torch.float32) if hasattr(env, 'Vm_min') else None
            VmUb = torch.tensor(env.Vm_max, dtype=torch.float32) if hasattr(env, 'Vm_max') else None
        else:
            VmLb = None
            VmUb = None
        
        model = DeepOPF_MLP(
            input_dim=actual_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layer=num_layers,
            scale_vm=10.0,  # 与 DeepOPF 一致
            scale_va=10.0,
            slack_bus_idx=0,  # IEEE 118 的 slack bus 索引
            VmLb=VmLb,
            VmUb=VmUb
        ).to(data.device)
        print(f"[OK] DeepOPF MLP model initialized (dual network, no output activation)")
    elif model_type in ['cluster','hindsight']:
        model = GMM(network, actual_input_dim, output_dim, hidden_dim, num_layers, num_cluster, output_act, pred_type).to(data.device)
    elif model_type == 'vae':
        model = VAE(network, actual_input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type).to(data.device)
    elif model_type == 'gan':
        model = GAN(network, actual_input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type).to(data.device)
    elif model_type == 'wgan':
        model = WGAN(network, actual_input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type).to(data.device)
    elif model_type == 'diffusion':
        model = DM(network, actual_input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
    elif model_type in ['gaussian', 'rectified', 'interpolation', 'conditional', 'ours', 'boosted_rectified', 'riemannian', 'reflow']:
        model = FM(network, actual_input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
        if model_type in ['rectified', 'riemannian', 'reflow']:    
            if pretrain_model is not None:
                model.pretrain_model = pretrain_model
            else:
                model_path = args['pretrain_model_path'] 
                model.pretrain_model = torch.load(model_path, weights_only=False).to(data.device) 
            print("[OK] 已加载预训练的VAE模型（完整模型）作为锚点生成器") 
            if train_mode == 'separate_training':
                model.pretrain_model.eval()
            else:   # joint_training
                model.optimizer_vae = torch.optim.Adam(model.pretrain_model.parameters(), lr=1e-4, weight_decay=args['weight_decay'])
                model.scheduler_vae = torch.optim.lr_scheduler.StepLR(model.optimizer_vae, step_size=2000, gamma=0.95)
                model.pretrain_model.train()
        
        # RFM 专用：初始化 RFMTrainingHelper（增强版）
        if model_type == 'riemannian':
            from rfm_utils import RFMTrainingHelper, get_default_rfm_config
            
            rfm_config = args.get('rfm_config', get_default_rfm_config())
            env = args.get('env', None)
            if env is None:
                raise ValueError("RFM 训练模式需要提供 env（电网环境对象）在 args['env'] 中")
            
            model.rfm_helper = RFMTrainingHelper(
                env=env,
                freeze_interval=rfm_config.get('freeze_interval', 10),
                soft_weight=rfm_config.get('soft_weight', 1.0),
                lambda_cor=rfm_config.get('lambda_cor', 5.0),
                add_perturbation=rfm_config.get('add_perturbation', True),
                perturbation_scale=rfm_config.get('perturbation_scale', 0.05),
                max_correction_norm=rfm_config.get('max_correction_norm', 10.0),
                device=data.device,
            )
            print(f"[OK] 初始化 RFM 训练助手 (freeze={rfm_config.get('freeze_interval', 10)}, "
                  f"lambda={rfm_config.get('lambda_cor', 5.0)}, "
                  f"perturb={rfm_config.get('add_perturbation', True)})")
        
        # Reflow 专用：附加 reflow_loader
        if model_type == 'reflow':
            reflow_loader = args.get('reflow_loader', None)
            if reflow_loader is not None:
                model.reflow_loader = reflow_loader
                model.reflow_iter = iter(reflow_loader)
                print(f"[OK] 初始化 Reflow 训练 (loader batches={len(reflow_loader)})")
            else:
                model.reflow_loader = None
                model.reflow_iter = None
                print("[Warning] Reflow 训练未提供轨迹数据，将只使用线性插值损失")
    elif model_type == 'potential':
        model = AM(network, actual_input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
    elif model_type == 'consistency_training':
        model = CM(network, actual_input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
        vector_model = torch.load(f'models/improved_version/rectified_{network}.pth', weights_only=False)
        vector_model.eval()
    elif model_type == 'consistency_distillation':
        model = CD(network, actual_input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
        vector_model = torch.load(f'models/improved_version/rectified_{network}.pth', weights_only=False)
        vector_model.eval() 
    else:
        raise NotImplementedError 
    
    model.single_target = data.single_target
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['learning_rate_decay'][0], gamma=args['learning_rate_decay'][1])
    
    # 创建TensorBoard writer
    # 避免重复名称 (如 deepopf_mlp_deepopf_mlp)
    if model_type == network:
        log_dir = f'runs/{instance}/{model_type}_{train_mode}_add_carbon_tax_{add_carbon_tax}_{time_str}'
    else:
        log_dir = f'runs/{instance}/{model_type}_{network}_{train_mode}_add_carbon_tax_{add_carbon_tax}_{time_str}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard日志目录: {log_dir}")
    
    # 最佳模型跟踪
    best_test_loss = float('inf')
    best_model_state = None
    test_interval = max(args['test_freq'], 100)  # 测试间隔，至少100个epoch
    
    # Train the model
    loss_record = []
    print(f"\n开始训练 {model_type} 模型...")
    print(f"数据集大小: {data_dim}, 批次大小: {batch_dim}")
    print(f"测试间隔: 每 {test_interval} 个epoch")
    
    for epoch in range(num_epochs):
        batch_indices = np.random.choice(data_dim, batch_dim)
        x_batch = data.x_train[batch_indices].to(data.device)
        y_batch = data.y_train[batch_indices].to(data.device)
        
        t_batch = torch.rand([batch_dim, 1]).to(data.device)
        
        optimizer.zero_grad()
        if model_type in ['rectified', 'riemannian', 'reflow'] and hasattr(model, 'pretrain_model') and train_mode == 'joint_training': 
            model.optimizer_vae.zero_grad()

        # 构建不同的噪声
        if noise_type == 'gaussian':
            z_batch = torch.randn_like(y_batch).to(data.device)
        elif noise_type == 'uniform_fixed':
            z_batch = torch.rand_like(y_batch).to(data.device) * 2 - 1
        else:
            NotImplementedError
        
        # 初始化损失组件
        supervision_loss = 0.0
        constraint_loss = 0.0
        kl_loss = 0.0
        
        if model_type =='simple':
            y_pred = model(x_batch)
            loss = model.loss(y_pred, y_batch)
            supervision_loss = loss.item()
        elif model_type == 'deepopf_mlp':
            # DeepOPF MLP 使用特殊的数据转换
            # 输入: 标准化的 Vm [-1,1], Va [-1,1]
            # 需要先转换 y_batch 到 DeepOPF 格式:
            #   Vm: [-1,1] -> [0,10] (scale_vm=10)
            #   Va: [-1,1] -> [-scale_va, scale_va] 
            n_bus = output_dim // 2
            Vm_batch = y_batch[:, :n_bus]
            Va_batch = y_batch[:, n_bus:]
            
            # 转换 Vm: [-1,1] -> [0,10]
            # 原始: Vm_pu = Vm_norm * 0.06 + 1 (范围 [0.94, 1.06])
            # DeepOPF: Vm_scaled = ((Vm_pu - VmLb) / (VmUb - VmLb)) * scale_vm
            # 简化: Vm_scaled = ((Vm_norm + 1) / 2) * 10
            Vm_deepopf = ((Vm_batch + 1) / 2) * model.scale_vm
            
            # 转换 Va: [-1,1] -> Va_rad / scale_va * scale_va = Va_rad
            # 原始: Va_rad = Va_norm * pi/6
            # DeepOPF: Va_scaled = Va_rad * scale_va / (pi/6) ??? 
            # 简化: 保持相同的转换比例
            Va_deepopf = Va_batch * model.scale_va
            
            y_target_deepopf = torch.cat([Vm_deepopf, Va_deepopf], dim=1)
            
            y_pred = model(x_batch)
            loss = model.loss(y_pred, y_target_deepopf)
            supervision_loss = loss.item()
        elif model_type in ['cluster', 'hindsight']:
            y_pred = model(x_batch)
            if model_type == 'cluster':
                loss = model.loss(x_batch, y_pred, y_batch)
            else:  # hindsight
                loss = model.hindsight_loss(x_batch, y_pred, y_batch)
            supervision_loss = loss.item()
        elif model_type == 'vae':
            y_pred, mean, logvar = model.encoder_decode(x_batch)
            # 分离VAE损失组件
            recon_loss = torch.nn.functional.mse_loss(y_pred, y_batch)
            kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
            kl_div = torch.mean(kl_div)
            loss = recon_loss + vae_beta * kl_div
            supervision_loss = recon_loss.item()
            kl_loss = kl_div.item()
        elif model_type in ['gan', 'wgan']:
            y_pred =  model(x_batch, z_batch)
            loss = model.loss_d(x_batch, y_batch, y_pred)
            if epoch % args['update_generator_freq'] == 0:
                loss += model.loss_g(x_batch, y_pred)
            supervision_loss = loss.item()
        elif model_type == 'diffusion':
            noise_pred = model.predict_noise(x_batch, y_batch, t_batch, z_batch)
            loss = model.loss(z_batch, noise_pred)
            supervision_loss = loss.item()
            ################################流模型部分##############################################
        elif model_type in ['gaussian', 'rectified', 'interpolation', 'conditional']:
            # **关键改进——起点的选择**: 使用生成模型VAE的近似最优解作为起始点，流模型学习的传统模型的预测误差
            if model_type == 'rectified':
                # 使用VAE模型生成锚点
                x_batch_pretrain = x_batch[:, :-1] if 'True' not in args['pretrain_model_path'] and not data.single_target else x_batch
                if train_mode == 'separate_training':
                    with torch.no_grad():
                        y_anchor_batch = model.pretrain_model(x_batch_pretrain, use_mean=True).to(data.device) 
                else:   # joint_training
                    y_anchor_batch = model.pretrain_model(x_batch_pretrain).to(data.device)  
            else:
                # 如果没有y_anchor，回退到随机噪声
                y_anchor_batch = torch.randn_like(y_batch).to(data.device)
            z_batch = y_anchor_batch      
            vec_type = model_type
            yt, vec_target = model.flow_forward(y_batch, t_batch, z_batch, vec_type) 
            # 将锚点拼接到条件x中: x_batch现在包含[负荷, 碳税, 锚点]
            # x_batch_original = torch.concat([x_batch, z_batch], dim=1)
            x_batch_original = x_batch
            vec_pred = model.predict_vec(x_batch_original, yt, t_batch)  # 这里x_batch是[负荷, 碳税]的信息，yt是当前位置，t_batch是时间步长
            loss = model.loss(y_batch, z_batch, vec_pred, vec_target, vec_type)
            supervision_loss = loss.item()
            y_pred = vec_pred + z_batch
        
        ################################ Riemannian Flow Matching 部分（增强版）##############################################
        elif model_type == 'riemannian':
            """
            Riemannian Flow Matching (RFM) 训练 - 渐进稳定版
            
            训练目标：
            v_target = P_tan @ (y - z) + Clip(correction, -C, C)
                     = 切向（去终点）  +  法向（回流形）
            
            关键改进：
            - 法向修正项让误差指数衰减，主动拉回流形
            - 扰动训练让模型学会修正偏离的状态
            - Clip截断防止数值爆炸
            """
            # 1. 生成锚点（与 rectified 相同）
            x_batch_pretrain = x_batch[:, :-1] if 'True' not in args['pretrain_model_path'] and not data.single_target else x_batch
            if train_mode == 'separate_training':
                with torch.no_grad():
                    y_anchor_batch = model.pretrain_model(x_batch_pretrain, use_mean=True).to(data.device) 
            else:   # joint_training
                y_anchor_batch = model.pretrain_model(x_batch_pretrain).to(data.device)  
            z_batch = y_anchor_batch
            
            # 2. 使用 RFMTrainingHelper 计算投影目标（增强版）
            #    - yt: 插值点（可能加了扰动）
            #    - vec_target: 切向 + 法向修正
            #    - P_tan: 切空间投影矩阵
            x_batch_for_jac = x_batch[:, :-1] if add_carbon_tax else x_batch
            yt, vec_target, P_tan = model.rfm_helper.compute_rfm_target(
                y_batch, z_batch, t_batch, x_batch_for_jac, epoch
            )
            
            # 3. 预测速度并计算损失
            vec_pred = model.predict_vec(x_batch, yt, t_batch)
            loss = model.loss(y_batch, z_batch, vec_pred, vec_target, 'riemannian')
            supervision_loss = loss.item()
            y_pred = vec_pred + z_batch
            
            # 4. 每隔一定 epoch 打印 RFM 统计信息
            if epoch % args['test_freq'] == 0 and epoch > 0:
                rfm_stats = model.rfm_helper.get_stats()
                print(f"  [RFM] cache_hit={rfm_stats['cache_hit_rate']:.1%}, "
                      f"jac_calls={rfm_stats['jacobian_computations']}, "
                      f"clips={rfm_stats.get('clip_events', 0)}")
        ################################ Riemannian Flow Matching 部分结束 ##########################################
        
        ################################ Reflow (带 Jacobian 后处理的自蒸馏) ##############################################
        elif model_type == 'reflow':
            """
            Reflow 训练 - 带 Jacobian 后处理的自蒸馏（改进版）
            
            核心思想：
            1. Teacher = Rectified Flow + Jacobian 修正，生成矫正后的轨迹
            2. Student 直接在 Teacher 的轨迹点上学习目标速度
            3. 推理时 Student 不需要 Jacobian 后处理！
            
            改进关键：
            - 不再使用线性插值（linear_loss），避免分布不匹配
            - 直接使用 Teacher 轨迹点作为训练数据
            - Student 学习在轨迹点上预测修正后的速度
            """
            # 获取 reflow 配置
            reflow_config = args.get('reflow_config', {})
            use_pure_trajectory = reflow_config.get('use_pure_trajectory', True)  # 是否只用轨迹训练
            lambda_reflow = reflow_config.get('lambda_reflow', 1.0)
            
            # 1. 从 reflow_loader 获取轨迹数据
            if hasattr(model, 'reflow_loader') and model.reflow_loader is not None:
                try:
                    reflow_batch = next(model.reflow_iter)
                except (StopIteration, AttributeError):
                    model.reflow_iter = iter(model.reflow_loader)
                    reflow_batch = next(model.reflow_iter)
                
                z_reflow, x_reflow, t_reflow, v_reflow = reflow_batch
                z_reflow = z_reflow.to(data.device)
                x_reflow = x_reflow.to(data.device)
                t_reflow = t_reflow.to(data.device)
                v_reflow = v_reflow.to(data.device)
                
                # 2. 在 Teacher 轨迹点上预测速度
                v_pred_reflow = model.model(x_reflow, z_reflow, t_reflow)
                loss_reflow = model.criterion(v_pred_reflow, v_reflow)
                
                if use_pure_trajectory:
                    # 纯轨迹训练：只学习 Teacher 的轨迹
                    loss = loss_reflow
                    loss_linear = torch.tensor(0.0)
                else:
                    # 混合训练：同时学习线性插值和轨迹
                    x_batch_pretrain = x_batch[:, :-1] if 'True' not in args['pretrain_model_path'] and not data.single_target else x_batch
                    with torch.no_grad():
                        y_anchor_batch = model.pretrain_model(x_batch_pretrain, use_mean=True).to(data.device)
                    z_batch = y_anchor_batch
                    
                    yt = t_batch * y_batch + (1 - t_batch) * z_batch
                    vec_target_linear = y_batch - z_batch
                    vec_pred = model.predict_vec(x_batch, yt, t_batch)
                    loss_linear = model.criterion(vec_pred, vec_target_linear)
                    
                    loss = loss_linear + lambda_reflow * loss_reflow
            else:
                # 没有轨迹数据，回退到普通 Rectified Flow
                x_batch_pretrain = x_batch[:, :-1] if 'True' not in args['pretrain_model_path'] and not data.single_target else x_batch
                with torch.no_grad():
                    y_anchor_batch = model.pretrain_model(x_batch_pretrain, use_mean=True).to(data.device)
                z_batch = y_anchor_batch
                
                yt = t_batch * y_batch + (1 - t_batch) * z_batch
                vec_target_linear = y_batch - z_batch
                vec_pred = model.predict_vec(x_batch, yt, t_batch)
                loss_linear = model.criterion(vec_pred, vec_target_linear)
                loss = loss_linear
                loss_reflow = torch.tensor(0.0)
            
            supervision_loss = loss.item()
            
            # 用于约束损失计算的 y_pred（使用原始数据）
            x_batch_pretrain = x_batch[:, :-1] if 'True' not in args['pretrain_model_path'] and not data.single_target else x_batch
            with torch.no_grad():
                y_anchor_batch = model.pretrain_model(x_batch_pretrain, use_mean=True).to(data.device)
            z_batch = y_anchor_batch
            yt = t_batch * y_batch + (1 - t_batch) * z_batch
            vec_pred = model.predict_vec(x_batch, yt, t_batch)
            y_pred = vec_pred + z_batch
            
            # 4. 打印统计信息
            if epoch % args['test_freq'] == 0 and epoch > 0:
                linear_val = loss_linear.item() if isinstance(loss_linear, torch.Tensor) else loss_linear
                reflow_val = loss_reflow.item() if isinstance(loss_reflow, torch.Tensor) else loss_reflow
                mode = "pure_traj" if use_pure_trajectory else "mixed"
                print(f"  [Reflow-{mode}] linear={linear_val:.4f}, traj={reflow_val:.4f}")
        ################################ Reflow 部分结束 ##########################################
        
        elif model_type == 'potential':
            loss = model.loss(x_batch, y_batch, z_batch, t_batch)
            supervision_loss = loss.item()
        elif model_type in ['consistency_training']:
            N = math.ceil(math.sqrt((epoch * (1000**2 - 4) / num_epochs) + 4) - 1) + 1
            boundaries = model.kerras_boundaries(1.0, 0.002, N, 1).to(device=data.device)
            t = torch.randint(0, N - 1, (x_batch.shape[0], 1), device=data.device)
            t_1 = boundaries[t+1]
            t_2 = boundaries[t]
            loss = model.loss(x_batch, y_batch, z_batch, t_1, t_2, data, vector_model)
            supervision_loss = loss.item()
        elif model_type == "consistency_distillation":
            forward_step = 10
            N = math.ceil(1000*(epoch/num_epochs) + 4) + forward_step
            boundaries = torch.linspace(0,1-1e-3,N).to(device=data.device)
            # model.kerras_boundaries(0.5, 0.001, N, 1).to(device=data.device)
            t = torch.randint(0, N - forward_step, (x_batch.shape[0], 1), device=data.device)
            t_1 = boundaries[t] 
            loss = model.loss(x_batch, y_batch, z_batch, t_1, 1/N, forward_step, data, vector_model)
            supervision_loss = loss.item()
        
        ################################ Latent Flow VAE 部分 ##############################################
        elif model_type == 'latent_flow_vae':
            # 三阶段训练策略
            # 阶段1 (epoch 0 - 40%): 预训练VAE
            # 阶段2 (epoch 40% - 70%): 训练Flow
            # 阶段3 (epoch 70% - 100%): 联合微调
            
            vae_epochs = int(num_epochs * 0.4)
            flow_epochs = int(num_epochs * 0.7)
            
            # 自动切换训练阶段
            if epoch == 0:
                model.set_training_stage('vae')
                print(f"\n=== 阶段1: 预训练VAE (epoch 0 - {vae_epochs}) ===")
            elif epoch == vae_epochs:
                model.set_training_stage('flow')
                print(f"\n=== 阶段2: 训练Latent Flow (epoch {vae_epochs} - {flow_epochs}) ===")
            elif epoch == flow_epochs:
                model.set_training_stage('joint')
                print(f"\n=== 阶段3: 联合微调 (epoch {flow_epochs} - {num_epochs}) ===")
            
            current_stage = model.training_stage
            
            if current_stage == 'vae':
                # 阶段1: 训练VAE (可选约束感知)
                # 在VAE预训练阶段也可以加入约束损失，让decoder从一开始就学习满足约束
                constraint_fn_for_vae = objective_fn if args.get('constraints_guided', False) else None
                constraint_weight_vae = args.get('w_constraints', 0.1) * 0.1 if constraint_fn_for_vae else 0.0  # VAE阶段用较小权重
                
                loss, loss_dict, y_pred, mean, logvar = model.vae_loss(
                    x_batch, y_batch, 
                    beta=vae_beta,
                    constraint_fn=constraint_fn_for_vae,
                    constraint_weight=constraint_weight_vae
                )
                supervision_loss = loss_dict['recon_loss']
                kl_loss = loss_dict['kl_div']
                
            elif current_stage == 'flow':
                # 阶段2: 固定decoder, 只训练flow
                loss, loss_dict = model.flow_loss(x_batch, y_batch)
                supervision_loss = loss_dict['flow_loss']
                # 用flow采样结果作为y_pred进行约束评估
                with torch.no_grad():
                    z_sampled = model.flow_backward(x_batch, num_steps=20)
                    y_pred = model.decode(x_batch, z_sampled)
                
            else:  # joint
                # 阶段3: 联合训练，加入约束损失
                constraint_fn_for_joint = objective_fn if args.get('constraints_guided', False) else None
                constraint_weight = args.get('w_constraints', 0.1) if constraint_fn_for_joint else 0.0
                
                loss, loss_dict, y_pred = model.joint_loss(
                    x_batch, y_batch, 
                    beta=vae_beta, 
                    flow_weight=1.0,
                    constraint_fn=constraint_fn_for_joint,
                    constraint_weight=constraint_weight
                )
                supervision_loss = loss_dict['recon_loss']
                kl_loss = loss_dict['kl_div']
        ################################ Latent Flow VAE 部分结束 ##############################################
        
        else:
            raise NotImplementedError
        
        # 这块再加入潮流方程计算后的功率的违反程度，进而得到额外的损失
        # 注意: latent_flow_vae 已经在内部处理了约束损失，不需要再计算
        if objective_fn is not None and args['constraints_guided'] and model_type != 'latent_flow_vae': 
            # 【修复】直接使用 mean 形式的约束损失，与 model_comparison_experiment.py 保持一致
            # 之前的问题：
            # 1. g2 * 200 无端放大了 Pg_min 约束
            # 2. w_gi * gi = gi²/sum 不是简单的加权和，导致梯度不稳定
            if not add_carbon_tax:
                constraint_loss = objective_fn(y_pred[:, :output_dim//2], y_pred[:, output_dim//2:], x_batch, 'mean')
            else:
                constraint_loss = objective_fn(y_pred[:, :output_dim//2], y_pred[:, output_dim//2:], x_batch[:, :-1], 'mean')
            
            # 直接使用总约束损失
            loss += constraint_loss * args['w_constraints']
        else:
            constraint_loss = torch.tensor(0.0)
        ################################流模型部分##############################################
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 新加入的，裁剪梯度
        optimizer.step()
        scheduler.step()
        if model_type in ['rectified', 'riemannian', 'reflow'] and hasattr(model, 'pretrain_model') and train_mode == 'joint_training':  
            model.optimizer_vae.step()
            model.scheduler_vae.step() 
        loss_record.append(loss.item())
        
        # 记录训练损失到TensorBoard
        writer.add_scalar('Loss/total', loss.item(), epoch)
        writer.add_scalar('Loss/supervision', supervision_loss, epoch)
        if isinstance(constraint_loss, torch.Tensor) and constraint_loss.item() > 0:
            writer.add_scalar('Loss/constraint', constraint_loss.item(), epoch)
        if kl_loss > 0:
            writer.add_scalar('Loss/kl_divergence', kl_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 定期打印训练进度
        if epoch % args['test_freq'] == 0:
            cons_val = constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss
            print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {loss.item():.6f}, "
                  f"Supervision Loss: {supervision_loss:.6f}, Constraint Loss: {cons_val:.6f}")
        
        # 定期在测试集上评估模型
        if epoch % test_interval == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                # 使用测试集的一个子集进行快速评估
                test_batch_size = min(512, data.x_test.shape[0])
                test_indices = np.random.choice(data.x_test.shape[0], test_batch_size, replace=False)
                x_test_batch = data.x_test[test_indices].to(data.device)
                y_test_batch = data.y_test[test_indices].to(data.device)
                
                # 根据模型类型进行预测
                if model_type == 'simple':
                    y_test_pred = model(x_test_batch)
                elif model_type == 'deepopf_mlp':
                    # DeepOPF MLP 输出需要转换回标准格式
                    y_pred_deepopf = model(x_test_batch)
                    n_bus = output_dim // 2
                    Vm_pred_deepopf = y_pred_deepopf[:, :n_bus]
                    Va_pred_deepopf = y_pred_deepopf[:, n_bus:]
                    # 转换 Vm: [0,10] -> [-1,1]
                    Vm_pred = (Vm_pred_deepopf / model.scale_vm) * 2 - 1
                    # 转换 Va: Va_deepopf -> [-1,1]
                    Va_pred = Va_pred_deepopf / model.scale_va
                    y_test_pred = torch.cat([Vm_pred, Va_pred], dim=1)
                elif model_type == 'vae':
                    y_test_pred = model(x_test_batch, use_mean=True)
                elif model_type in ['rectified', 'boosted_rectified', 'riemannian', 'reflow']:
                    # 对于流模型，需要生成锚点并进行推理
                    if model_type in ['rectified', 'riemannian', 'reflow']:
                        x_test_pretrain = x_test_batch[:, :-1] if 'True' not in args['pretrain_model_path'] and not data.single_target else x_test_batch
                        y_anchor_test = model.pretrain_model(x_test_pretrain).to(data.device) 
                    # 流模型反向传播
                    # x_test_with_anchor = torch.concat([x_test_batch, y_anchor_test], dim=1)
                    x_test_with_anchor = x_test_batch
                    y_test_pred, _ = model.flow_backward(
                        x_test_with_anchor, y_anchor_test,
                        step=1/args['inf_step'], method='Euler'
                    )
                elif model_type == 'latent_flow_vae':
                    # Latent Flow VAE: 通过flow采样生成
                    num_flow_steps = args.get('inf_step', 50)
                    y_test_pred = model(x_test_batch, num_steps=num_flow_steps)
                else:
                    # 对于其他模型类型，使用简单预测
                    y_test_pred = model(x_test_batch) if model_type == 'simple' else model(x_test_batch)
                
                # 计算测试损失
                test_loss = torch.nn.functional.mse_loss(y_test_pred, y_test_batch).item()
                
                # 计算测试集上的约束损失
                test_constraint_loss = 0.0
                test_constraint_details = None
                if objective_fn is not None:
                    x_test_input = x_test_batch if not add_carbon_tax else x_test_batch[:, :-1]
                    result = objective_fn(y_test_pred[:, :output_dim//2], y_test_pred[:, output_dim//2:], x_test_input, 'mean')
                    # 处理两种返回格式：有 details 和无 details
                    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                        test_constraint_loss, test_constraint_details = result
                        test_constraint_loss = test_constraint_loss.item()
                    else:
                        # 无 details 的情况
                        test_constraint_loss = result.item() if hasattr(result, 'item') else float(result)
                        test_constraint_details = None
                # 记录测试指标到TensorBoard
                writer.add_scalar('Test/loss', test_loss, epoch)
                writer.add_scalar('Test/constraint_loss', test_constraint_loss, epoch)
                if test_constraint_details is not None:
                    writer.add_scalar('Test/g1_pmax', test_constraint_details['g1_pmax'], epoch)
                    writer.add_scalar('Test/g2_pmin', test_constraint_details['g2_pmin'], epoch)
                    writer.add_scalar('Test/g5_qmax', test_constraint_details['g5_qmax'], epoch)
                    writer.add_scalar('Test/g6_qmin', test_constraint_details['g6_qmin'], epoch)
                    writer.add_scalar('Test/g9_sf', test_constraint_details['g9_sf'], epoch)
                    writer.add_scalar('Test/g10_st', test_constraint_details['g10_st'], epoch)
                
                # 保存最佳模型
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'test_loss': test_loss,
                        'test_constraint_loss': test_constraint_loss
                    }
                    print(f"  [Best] 新的最佳模型! Test Loss: {test_loss:.6f}, Constraint Loss: {test_constraint_loss:.6f}")
                else:
                    print(f"  → Test Loss: {test_loss:.6f}, Constraint Loss: {test_constraint_loss:.6f}")
            
            model.train()
            if model_type in ['rectified', 'riemannian', 'reflow'] and hasattr(model, 'pretrain_model') and train_mode == 'separate_training':
                model.pretrain_model.eval()  # 确保预训练模型保持eval模式
    
    # 关闭TensorBoard writer
    writer.close()
    
    # 保存前清理不可序列化的属性
    if hasattr(model, 'reflow_loader'):
        del model.reflow_loader
    if hasattr(model, 'reflow_iter'):
        del model.reflow_iter
    
    # 保存最终模型
    os.makedirs(f'models/{instance}', exist_ok=True)
    save_path = f'models/{instance}/{model_type}_{network}_{train_mode}_add_carbon_tax_{add_carbon_tax}_{time_str}.pth'
    torch.save(model, save_path)
    print(f"\n最终模型已保存: {save_path}")
    
    # 如果有最佳模型，也保存它
    if best_model_state is not None:
        best_save_path = f'models/{instance}/{model_type}_{network}_{train_mode}_add_carbon_tax_{add_carbon_tax}_{time_str}_best.pth'
        # 加载最佳状态
        model.load_state_dict(best_model_state['model_state_dict'])
        torch.save(model, best_save_path)
        print(f"最佳模型已保存: {best_save_path}")
        print(f"  - 训练轮次: {best_model_state['epoch']}")
        print(f"  - 测试损失: {best_model_state['test_loss']:.6f}")
        print(f"  - 约束损失: {best_model_state['test_constraint_loss']:.6f}")
    
    return model, loss_record


def model_forward(model, model_type, x_test, args, objective_fn=None, guidance_config=None,
                  evolutionary_config=None, sample_num=1, device='cuda',
                  apply_post_process=False, env=None, max_iterations=5):
    """
    评估模型在测试集上的性能
    
    Args:
        model: 训练好的模型
        model_type: 模型类型 (simple, rectified, gan, etc.)
        x_test: 测试输入 (test_dim, input_dim)
        args: 参数字典
        objective_fn: 目标函数（用于引导和投影）
        guidance_config: 引导配置字典
        projection_config: 投影配置字典（可选）
        sample_num: 采样数量（用于生成式模型）
        device: 计算设备
        apply_post_process: 是否应用后处理修正
        env: 电网环境对象（后处理时需要）
        max_iterations: 后处理最大迭代次数
    
    Returns:
        y_pred: 预测结果
    """
    model.eval()
    test_dim = x_test.shape[0]
    
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
    if model_type == 'simple':
        with torch.no_grad():
            y_pred = model(x_test)
    
    elif model_type == 'deepopf_mlp':
        with torch.no_grad():
            y_pred_deepopf = model(x_test)
            # 转换回标准格式
            output_dim = y_pred_deepopf.shape[1]
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
        # 流模型从anchor开始
        x_test_repeated = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
        
        # 【两种anchor生成策略】
        # 策略A（推荐）：充分利用VAE的随机性，为每个采样生成不同的anchor
        # 策略B（当前）：生成1个anchor，然后在其周围添加小噪声进行局部探索
        
        use_vae_diversity = True  # 设为True使用策略A，False使用策略B    
        
        if use_vae_diversity and model_type in ['rectified', 'riemannian', 'reflow'] and sample_num > 1:
            # 策略A：让VAE为每个采样都生成不同的随机anchor（充分利用VAE的latent space）
            with torch.no_grad():
                x_test_repeated_pretrain = x_test_repeated[:, :-1] if 'True' not in args['pretrain_model_path'] and not args['single_target'] else x_test_repeated
                z_test = model.pretrain_model(x_test_repeated_pretrain).to(device)  # VAE内部会自动采样不同的z 
        else:
            # 策略B：生成1个anchor，然后在周围局部探索（原始方式）
            z_test = torch.repeat_interleave(y_anchor_test, repeats=sample_num, dim=0)
            z_test = z_test + torch.randn_like(z_test).to(device) * 0.01  # 添加小噪声
        
        # 将锚点拼接到条件x中: x_test_repeated现在包含[负荷, 碳税, 锚点]
        # x_test_repeated = torch.concat([x_test_repeated, z_test], dim=1)
        y_pred, constraint_violation = model.flow_backward(
            x_test_repeated, z_test, step=1/args['inf_step'], method='Euler',
            objective_fn=objective_fn, guidance_config=guidance_config,
             evolutionary_config=evolutionary_config)

        if sample_num > 1:
            # 这里根据constraint_violation违反程度从sample_num里选择最好的作为最后的y_pred 
            y_pred = y_pred.view(test_dim, sample_num, -1)  # (test_dim, sample_num, output_dim)
            constraint_violation = constraint_violation.view(test_dim, sample_num)  # (test_dim, sample_num)
            
            # 为每个测试样本选择约束违反最小的采样
            best_indices = constraint_violation.min(dim=1)[1]  # (test_dim,)
            y_pred = y_pred[torch.arange(test_dim), best_indices, :]  # (test_dim, output_dim)
        
    elif model_type == 'diffusion':
        with torch.no_grad():
            x_test_repeated = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
            z_test = torch.randn(size=[x_test_repeated.shape[0], args['output_dim']]).to(device)
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

    # ==================== 后处理修正 ====================
    if apply_post_process and env is not None:
        from post_processing import apply_post_processing
        
        # 获取 output_dim
        output_dim = y_pred.shape[1]
        batch_size_pred = y_pred.shape[0]
        batch_size_input = x_test.shape[0]
        
        # 检查批次大小是否匹配（如果模型类型没有内置选择逻辑）
        # 对于没有选择逻辑的模型类型，需要先选择最佳样本
        if batch_size_pred != batch_size_input and sample_num > 1:
            # 需要选择最佳样本
            y_pred_reshaped = y_pred.view(batch_size_input, sample_num, -1)  # (test_dim, sample_num, output_dim)
            
            # 计算每个样本的约束损失用于选择
            if objective_fn is not None:
                # 准备用于约束计算的输入
                if args.get('add_carbon_tax', False):
                    x_input_for_selection = x_test[:, :-1]
                else:
                    x_input_for_selection = x_test
                
                # 扩展输入以匹配样本数
                x_input_expanded = x_input_for_selection.unsqueeze(1).expand(-1, sample_num, -1)  # (test_dim, sample_num, input_dim)
                x_input_flat = x_input_expanded.reshape(-1, x_input_expanded.shape[-1])  # (test_dim * sample_num, input_dim)
                
                # 计算所有样本的约束损失
                Vm_all = y_pred[:, :output_dim//2]
                Va_all = y_pred[:, output_dim//2:]
                constraint_losses = objective_fn(Vm_all, Va_all, x_input_flat, 'none')
                if isinstance(constraint_losses, tuple):
                    constraint_losses = constraint_losses[0]
                
                constraint_losses = constraint_losses.view(batch_size_input, sample_num)
                best_indices = constraint_losses.argmin(dim=1)  # (test_dim,)
                y_pred = y_pred_reshaped[torch.arange(batch_size_input, device=y_pred.device), best_indices, :]
            else:
                # 没有目标函数，选择第一个样本
                y_pred = y_pred_reshaped[:, 0, :]
        
        # 分割 Vm 和 Va
        Vm = y_pred[:, :output_dim//2]
        Va = y_pred[:, output_dim//2:]
        
        # 准备输入数据（去掉碳税列如果有的话）
        if args.get('add_carbon_tax', False):
            x_input = x_test[:, :-1]
        else:
            x_input = x_test
        
        # 迭代后处理
        Vm_current, Va_current = Vm.clone(), Va.clone()
        convergence_threshold = 0.01  # 收敛阈值
        target_constraint = 0.5  # 目标约束损失
        prev_loss = None
        
        for iteration in range(max_iterations):
            # 应用后处理
            Vm_corrected, Va_corrected, correction_info = apply_post_processing(
                Vm_current, Va_current, x_input, env, k_dV=1.0, verbose=False, debug_mode=0
            )
            
            # 计算约束损失（如果有目标函数）
            if objective_fn is not None:
                constraint_loss_new = objective_fn(Vm_corrected, Va_corrected, x_input, 'mean')
                if isinstance(constraint_loss_new, tuple):
                    constraint_loss_new = constraint_loss_new[0]
                if isinstance(constraint_loss_new, torch.Tensor):
                    constraint_loss_new = constraint_loss_new.item()
                
                # 检查是否达到目标或收敛
                if constraint_loss_new < target_constraint:
                    Vm_current, Va_current = Vm_corrected, Va_corrected
                    break
                
                if prev_loss is not None:
                    improvement = abs(prev_loss - constraint_loss_new) / prev_loss * 100
                    if improvement < convergence_threshold:
                        Vm_current, Va_current = Vm_corrected, Va_corrected
                        break
                
                prev_loss = constraint_loss_new
            
            Vm_current, Va_current = Vm_corrected, Va_corrected
        
        # 合并回 y_pred
        y_pred = torch.cat([Vm_current, Va_current], dim=1)

    return y_pred