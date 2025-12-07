"""
PPO训练测试脚本 - 用于微调Flow模型

核心思想：
1. 使用PPO算法直接微调Flow模型参数
2. 添加KL散度约束防止策略偏离原始模型
3. 设置熵系数为0，不强制探索
4. 使用clip机制限制策略更新幅度

参考：RLfinetuning_Diffusion_Bioseq项目的PPO微调方法

作者：基于rl_opf_flow项目
日期：2025
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime

# 添加项目路径并切换工作目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 导入必要模块（使用相对路径）
from ode_rl_env import ODE_RL_Env
from ppo_flow_model import PPOFlowModel, RolloutBuffer

# ==================== 设置随机种子，保证实验可复现 ====================
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
#                    训练配置类
# ============================================================
class PPOTrainingConfig:
    """
    PPO训练配置
    
    集中管理所有训练参数，便于调整和实验
    """
    def __init__(self):
        # ===== 场景选择配置 =====
        self.use_fixed_scenario = True      # 是否使用固定场景
        self.fixed_scenario_idx = 0         # 固定场景索引
        
        # ===== 训练轮次配置 =====
        self.num_iterations = 1000           # 总迭代次数
        self.rollout_steps = 100            # 每次rollout的步数（等于一个episode）
        self.n_rollouts_per_update = 3      # 每次更新前收集的rollout数
        
        # ===== 评估配置 =====
        self.num_eval_samples = 10          # 评估时使用的样本数
        
        # ===== PPO超参数 =====
        self.actor_lr = 1e-5                # Actor学习率（小一些以保护预训练知识）
        self.critic_lr = 3e-4               # Critic学习率
        self.gamma = 0.99                   # 折扣因子
        self.gae_lambda = 0.95              # GAE的lambda参数
        self.clip_epsilon = 0.1             # PPO clip参数（小一些以限制更新幅度）
        self.entropy_coef = 0.0             # 熵系数（设为0！）
        self.value_coef = 0.5               # 价值损失系数
        self.max_grad_norm = 0.5            # 梯度裁剪
        self.n_epochs = 5                   # 每次更新的epoch数
        self.batch_size = 64                # mini-batch大小
        
        # ===== KL散度约束 =====
        self.kl_coef = 0.2                  # KL散度约束系数
        self.kl_target = 0.01               # KL散度目标值
        self.adaptive_kl = True             # 是否自适应调整KL系数
        
        # ===== 策略网络配置 =====
        self.log_std_init = -3.0            # 初始log_std（std约0.05）
        self.learn_std = True               # 是否学习std
        self.log_std_min = -5.0             # log_std最小值
        self.log_std_max = -2.0             # log_std最大值
        
        # ===== 奖励配置 =====
        self.reward_scale = 0.01            # 奖励缩放因子
        self.use_dense_reward = False        # 使用密集奖励
        self.reward_clip = (-10.0, 10.0)    # 奖励裁剪
        self.use_flow_baseline = True       # 使用Flow模型作为基线
        
        # ===== 模型配置 =====
        self.hidden_dim = 256               # 隐藏层维度
        self.eval_interval = 20              # 每隔多少次迭代评估一次


def create_tensorboard_writer(config):
    """
    创建TensorBoard writer
    
    Args:
        config: 训练配置
    
    Returns:
        writer: SummaryWriter对象
        log_dir: 日志目录
    """
    from torch.utils.tensorboard import SummaryWriter
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/ppo_finetune_{timestamp}"
    
    # 创建配置描述
    config_str = (f"ppo_lr{config.actor_lr}_clip{config.clip_epsilon}_"
                  f"kl{config.kl_coef}_ent{config.entropy_coef}")
    log_dir = f"runs/{config_str}_{timestamp}"
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志目录: {log_dir}")
    
    return writer, log_dir


def compute_flow_baseline_violation(flow_model, ode_env, x_input, z_init, single_target, device):
    """
    计算原始Flow模型的约束违反值（作为基线）
    
    使用原始Flow模型进行完整的ODE采样，然后计算最终的约束违反值。
    这个值作为基线，用于计算PPO模型的相对改进。
    
    Args:
        flow_model: 预训练的Flow模型
        ode_env: ODE_RL_Env 环境
        x_input: [batch_size, input_dim] 输入条件
        z_init: [batch_size, output_dim] 初始潜在变量
        single_target: 是否单目标模式
        device: 计算设备
    
    Returns:
        violation: float 约束违反值
        z_final: tensor 最终的潜在变量
    """
    with torch.no_grad():
        # 使用原始Flow模型进行ODE采样
        z_flow = z_init.clone()
        for step in range(ode_env.inf_step):
            t = torch.tensor([[step / ode_env.inf_step]], device=device)
            # Flow模型预测速度向量
            velocity = flow_model.model(x_input, z_flow, t)
            # 更新z
            z_flow = z_flow + velocity * ode_env.step_size
        
        # 计算约束违反
        half_dim = ode_env.output_dim // 2
        Vm = z_flow[:, :half_dim]
        Va = z_flow[:, half_dim:]
        x_condition = x_input if single_target else x_input[:, :-1]
        violation = ode_env.objective_fn(Vm, Va, x_condition, reduction='mean')
        
    return violation.item(), z_flow


def evaluate_model_comparison(
    agent, 
    flow_model, 
    ode_env, 
    data, 
    config, 
    device,
    single_target=True
):
    """
    评估PPO模型与原始Flow模型的性能对比
    
    通过比较两个模型在相同输入下生成解的约束违反程度，
    来判断PPO微调是否真正改善了模型性能。
    
    Args:
        agent: PPOFlowModel PPO智能体
        flow_model: 原始预训练的Flow模型
        ode_env: ODE_RL_Env 强化学习环境
        data: OPF_Flow_Dataset_V2 数据集
        config: PPOTrainingConfig 配置对象
        device: 计算设备
        single_target: 是否为单目标模式
    
    Returns:
        eval_results: dict 包含评估结果的字典
    """
    # 设置为评估模式
    agent.actor.eval()
    flow_model.eval()
    
    # 存储评估结果
    ppo_violations = []      # PPO模型的约束违反值
    flow_violations = []     # 原始Flow模型的约束违反值
    
    # 确定评估样本的索引
    if config.use_fixed_scenario:
        # 固定场景模式：只评估固定的那个场景
        eval_indices = [config.fixed_scenario_idx]
    else:
        # 多样场景模式：随机选择测试样本
        num_samples = min(config.num_eval_samples, data.num_test_samples)
        eval_indices = np.random.choice(
            data.num_test_samples, 
            size=num_samples, 
            replace=False
        )
    
    for idx in eval_indices:
        # 获取测试样本（固定场景用训练集，多样场景用测试集）
        if config.use_fixed_scenario:
            x_test = data.x_train[idx:idx+1].to(device)
        else:
            x_test = data.x_test[idx:idx+1].to(device)
        
        # 获取初始z（从预训练模型）
        with torch.no_grad():
            if hasattr(flow_model, 'pretrain_model'):
                x_pretrain = x_test if single_target else x_test[:, :-1]
                z_init = flow_model.pretrain_model(x_pretrain, use_mean=True)
            else:
                z_init = torch.randn(1, ode_env.output_dim).to(device)
        
        # ===== 方法1: 使用原始Flow模型生成解 =====
        flow_violation, _ = compute_flow_baseline_violation(
            flow_model, ode_env, x_test, z_init.clone(), single_target, device
        )
        flow_violations.append(flow_violation)
        
        # ===== 方法2: 使用PPO微调后的策略生成解 =====
        with torch.no_grad():
            # 重置环境（传入基线violation用于相对奖励计算）
            state = ode_env.reset(x_test, z_init.clone(), baseline_violation=flow_violation)
            
            # 使用PPO策略进行ODE采样
            # 注意：评估时使用确定性策略（deterministic=True）
            while not ode_env.done:
                action, _, _ = agent.take_action(state, deterministic=True)
                next_state, reward, done, info = ode_env.step(action)
                state = next_state
            
            # 获取最终的z并计算约束违反
            z_ppo = ode_env.current_z
            half_dim = ode_env.output_dim // 2
            Vm_ppo = z_ppo[:, :half_dim]
            Va_ppo = z_ppo[:, half_dim:]
            x_input = x_test if single_target else x_test[:, :-1]
            ppo_violation = ode_env.objective_fn(Vm_ppo, Va_ppo, x_input, reduction='mean')
            ppo_violations.append(ppo_violation.item())
    
    # 切换回训练模式
    agent.actor.train()
    
    # 计算统计结果
    eval_results = {
        'ppo_violation_mean': np.mean(ppo_violations),      # PPO模型平均约束违反
        'ppo_violation_std': np.std(ppo_violations),        # PPO模型约束违反标准差
        'flow_violation_mean': np.mean(flow_violations),    # Flow模型平均约束违反
        'flow_violation_std': np.std(flow_violations),      # Flow模型约束违反标准差
        'improvement': np.mean(flow_violations) - np.mean(ppo_violations),  # 绝对改进量
        'improvement_ratio': (np.mean(flow_violations) - np.mean(ppo_violations)) / (np.mean(flow_violations) + 1e-8)  # 相对改进比例
    }
    
    return eval_results


def collect_rollout(env, agent, x_condition, z_init, rollout_buffer, baseline_violation=None):
    """
    收集一个完整episode的rollout数据
    
    Args:
        env: RL环境
        agent: PPO agent
        x_condition: 条件输入
        z_init: 初始状态
        rollout_buffer: RolloutBuffer
        baseline_violation: 基线约束违反值
    
    Returns:
        episode_reward: episode总奖励
        episode_length: episode长度
    """
    state = env.reset(x_condition, z_init, baseline_violation)
    
    episode_reward = 0
    episode_length = 0
    
    while not env.done:
        action, log_prob, value = agent.take_action(state, deterministic=False)
        next_state, reward, done, info = env.step(action)
        
        # 添加到buffer
        rollout_buffer.add(
            state=state.squeeze(0),
            action=action.squeeze(0),
            reward=reward.item() if isinstance(reward, torch.Tensor) else reward,
            done=float(done),
            log_prob=log_prob.item(),
            value=value.item()
        )
        
        episode_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
        episode_length += 1
        state = next_state
    
    return episode_reward, episode_length


def main():
    """
    PPO训练主函数
    """
    print("="*60)
    print("PPO Flow Model微调测试")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建配置
    config = PPOTrainingConfig()
    
    # 打印配置信息
    print("\n[Configuration]")
    print(f"  Training Mode: {'Fixed Scenario' if config.use_fixed_scenario else 'Multi Scenario'}")
    if config.use_fixed_scenario:
        print(f"  Fixed Scenario Index: {config.fixed_scenario_idx}")
    print(f"  Number of Iterations: {config.num_iterations}")
    print(f"  Evaluation Interval: {config.eval_interval}")
    print(f"  [Reward] scale={config.reward_scale}, dense={config.use_dense_reward}, "
          f"clip={config.reward_clip}, baseline={config.use_flow_baseline}")
    print(f"  [PPO] actor_lr={config.actor_lr}, clip={config.clip_epsilon}, "
          f"kl_coef={config.kl_coef}, entropy={config.entropy_coef}")
    
    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("\n[Step 1] Loading data...")
    from load_opf_data_v2 import OPF_Flow_Dataset_V2
    from train_main import get_unified_config
    
    # 获取统一配置
    unified_config = get_unified_config(debug_mode=True)
    data_path = unified_config['data_path']
    add_carbon_tax = unified_config['add_carbon_tax']
    single_target = True if 'preferences' not in data_path else False
    
    # 加载OPF数据集
    data = OPF_Flow_Dataset_V2(
        data_path,
        device=device,
        test_ratio=0.2,
        random_seed=42,
        add_carbon_tax=add_carbon_tax,
        single_target=single_target
    )
    print(f"  Train samples: {data.num_train_samples}")
    print(f"  Test samples: {data.num_test_samples}")
    
    # ============================================================
    # 2. 加载预训练Flow模型
    # ============================================================
    print("\n[Step 2] Loading pre-trained flow model...")
    model_path = 'models/h512_l6_b512_lr0.001_wd1e-06_wc1_cg[False]_ctF_tmst/rectified_mlp_separate_training_add_carbon_tax_False_20251114_163919_best.pth'
    flow_model = torch.load(model_path, map_location=device, weights_only=False)
    flow_model.eval()
    print(f"  Model loaded: {model_path}")
    
    # ============================================================
    # 3. 创建电网环境（用于计算约束违反）
    # ============================================================
    print("\n[Step 3] Creating power grid environment...")
    from env import PowerGridEnv
    from models.actor import Actor, PowerSystemConfig
    
    # 电力系统配置
    ps_config = PowerSystemConfig(
        device=device, 
        case_file_path='../saved_data/pglib_opf_case118.mat'
    )
    
    # 电网环境（用于评估约束）
    pg_env = PowerGridEnv(
        num_timesteps=288,
        case_name="case118",
        random_load=False,
        run_pp=True,
        consider_renewable_generation=False,
        PowerSystemConfig=ps_config,
        device=device,
        carbon_tax=0.0
    )
    
    # Actor模型（用于计算约束损失）
    actor = Actor(input_dim=189, env=pg_env, output_dim=118).to(device)
    actor.eval()
    
    # 定义目标函数（约束违反计算）- 与SAC相同
    def objective_fn(Vm, Va, x_input, reduction='mean'):
        """
        计算约束违反程度
        
        Args:
            Vm: 电压幅值
            Va: 电压相角
            x_input: 输入条件（负荷等）
            reduction: 归约方式 ('mean', 'none')
        
        Returns:
            constraint_loss: 约束违反损失值
        """
        constraint_loss, _ = actor.compute_constraint_loss(
            Vm, Va, x_input, pg_env,
            reduction=reduction,
            return_details=True
        )
        return constraint_loss
    
    # ============================================================
    # 4. 创建ODE强化学习环境
    # ============================================================
    print("\n[Step 4] Creating ODE RL environment...")
    
    # 环境参数配置（包含奖励配置）
    env_args = {
        'inf_step': unified_config['inf_step'],           # ODE求解步数
        'output_dim': data.y_train.shape[1],              # 输出维度
        'input_dim': data.x_train.shape[1],               # 输入维度
        'single_target': single_target,                    # 是否单目标
        'output_norm': unified_config['output_norm'],      # 是否输出归一化
        
        # ===== 奖励配置 =====
        'reward_scale': config.reward_scale,               # 奖励缩放因子
        'use_dense_reward': config.use_dense_reward,       # 是否使用密集奖励
        'reward_clip': config.reward_clip,                 # 奖励裁剪范围
        'use_flow_baseline': config.use_flow_baseline      # 是否使用基线
    }
    
    # 创建ODE环境
    env = ODE_RL_Env(
        flow_model=flow_model,
        objective_fn=objective_fn,
        args=env_args,
        device=device
    )
    
    # ============================================================
    # 5. 创建PPO Agent
    # ============================================================
    print("\n[Step 5] Creating PPO Agent...")
    
    ppo_args = {
        'input_dim': env_args['input_dim'],
        'output_dim': env_args['output_dim'],
        'hidden_dim': config.hidden_dim,
        'actor_lr': config.actor_lr,
        'critic_lr': config.critic_lr,
        'gamma': config.gamma,
        'gae_lambda': config.gae_lambda,
        'clip_epsilon': config.clip_epsilon,
        'entropy_coef': config.entropy_coef,
        'value_coef': config.value_coef,
        'max_grad_norm': config.max_grad_norm,
        'n_epochs': config.n_epochs,
        'batch_size': config.batch_size,
        'kl_coef': config.kl_coef,
        'kl_target': config.kl_target,
        'adaptive_kl': config.adaptive_kl,
        'log_std_init': config.log_std_init,
        'learn_std': config.learn_std,
        'log_std_min': config.log_std_min,
        'log_std_max': config.log_std_max,
    }
    
    agent = PPOFlowModel(flow_model, env.state_dim, env.action_dim, ppo_args, device)
    
    # ============================================================
    # 6. 创建TensorBoard和RolloutBuffer
    # ============================================================
    print("\n[Step 6] 初始化训练组件...")
    
    writer, log_dir = create_tensorboard_writer(config)
    
    # Rollout buffer大小 = rollout_steps * n_rollouts_per_update
    buffer_size = config.rollout_steps * config.n_rollouts_per_update
    rollout_buffer = RolloutBuffer(
        buffer_size=buffer_size,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda
    )
    
    # ============================================================
    # 7. 初始评估
    # ============================================================
    print("\n[Step 7] Initial evaluation (before training)...")
    
    initial_eval = evaluate_model_comparison(
        agent, flow_model, env, data, config, device, single_target
    )
    print(f"  [Initial] Flow Model Violation: {initial_eval['flow_violation_mean']:.6f}")
    print(f"  [Initial] PPO Model Violation:  {initial_eval['ppo_violation_mean']:.6f}")
    
    # 记录初始评估到TensorBoard
    writer.add_scalar('Evaluation/Flow_Violation', initial_eval['flow_violation_mean'], 0)
    writer.add_scalar('Evaluation/PPO_Violation', initial_eval['ppo_violation_mean'], 0)
    writer.add_scalar('Evaluation/Improvement', initial_eval['improvement'], 0)
    
    # ============================================================
    # 8. 训练循环
    # ============================================================
    print("\n" + "="*60)
    print(f"Starting training ({config.num_iterations} iterations)...")
    print("="*60)
    
    # 全局步数计数器
    global_step = 0
    
    # 记录最佳模型
    best_improvement = float('-inf')
    best_iteration = 0
    
    for iteration in range(config.num_iterations):
        rollout_buffer.clear()
        
        total_reward = 0
        total_length = 0
        
        # 收集多个rollout
        for rollout_idx in range(config.n_rollouts_per_update):
            # 选择场景
            if config.use_fixed_scenario:
                idx = config.fixed_scenario_idx
            else:
                idx = np.random.randint(0, data.num_train_samples)
            
            x_train = data.x_train[idx:idx+1].to(device)
            
            # 生成初始潜在变量z
            if hasattr(flow_model, 'pretrain_model'):
                with torch.no_grad():
                    x_pretrain = x_train if single_target else x_train[:, :-1]
                    z_init = flow_model.pretrain_model(x_pretrain, use_mean=True)
            else:
                z_init = torch.randn(1, env_args['output_dim']).to(device)
            
            # 计算基线（如果使用）
            baseline_violation = None
            if config.use_flow_baseline:
                baseline_violation, _ = compute_flow_baseline_violation(
                    flow_model, env, x_train, z_init.clone(), single_target, device
                )
            
            # 收集rollout
            ep_reward, ep_length = collect_rollout(
                env, agent, x_train, z_init, rollout_buffer, baseline_violation
            )
            
            total_reward += ep_reward
            total_length += ep_length
        
        # 计算GAE
        with torch.no_grad():
            last_value = agent.critic(
                torch.cat([env.current_x_condition, 
                          torch.ones(1, 1, device=device), 
                          env.current_z], dim=1)
            ).squeeze(-1).item()
        
        rollout_buffer.compute_gae(last_value, float(env.done))
        
        # 更新PPO
        update_info = agent.update(rollout_buffer)
        
        global_step += rollout_buffer.ptr
        
        # 记录到TensorBoard
        avg_reward = total_reward / config.n_rollouts_per_update
        avg_length = total_length / config.n_rollouts_per_update
        
        writer.add_scalar('Training/Episode_Reward', avg_reward, iteration)
        writer.add_scalar('Training/Episode_Length', avg_length, iteration)
        writer.add_scalar('Loss/Actor_Loss', update_info['actor_loss'], iteration)
        writer.add_scalar('Loss/Critic_Loss', update_info['critic_loss'], iteration)
        writer.add_scalar('PPO/Entropy', update_info['entropy'], iteration)
        writer.add_scalar('PPO/KL_Divergence', update_info['kl_divergence'], iteration)
        writer.add_scalar('PPO/KL_Coef', update_info['kl_coef'], iteration)
        
        # 打印进度
        if (iteration + 1) % 5 == 0 or iteration < 5:
            print(f"Iter {iteration+1}/{config.num_iterations}: "
                  f"reward={avg_reward:.4f}, length={avg_length:.0f}, "
                  f"actor_loss={update_info['actor_loss']:.4f}, "
                  f"kl={update_info['kl_divergence']:.6f}")
        
        # 定期评估
        if (iteration + 1) % config.eval_interval == 0:
            print(f"\n[Evaluation at Iteration {iteration+1}]")
            
            eval_results = evaluate_model_comparison(
                agent, flow_model, env, data, config, device, single_target
            )
            
            print(f"  Flow Model Violation: {eval_results['flow_violation_mean']:.6f} ± {eval_results['flow_violation_std']:.6f}")
            print(f"  PPO Model Violation:  {eval_results['ppo_violation_mean']:.6f} ± {eval_results['ppo_violation_std']:.6f}")
            print(f"  Improvement: {eval_results['improvement']:.6f} ({eval_results['improvement_ratio']*100:.2f}%)")
            print()
            
            # 记录评估结果到TensorBoard
            writer.add_scalar('Evaluation/Flow_Violation', eval_results['flow_violation_mean'], iteration + 1)
            writer.add_scalar('Evaluation/PPO_Violation', eval_results['ppo_violation_mean'], iteration + 1)
            writer.add_scalar('Evaluation/Improvement', eval_results['improvement'], iteration + 1)
            writer.add_scalar('Evaluation/Improvement_Ratio', eval_results['improvement_ratio'], iteration + 1)
            
            # 保存最佳模型
            if eval_results['improvement'] > best_improvement:
                best_improvement = eval_results['improvement']
                best_iteration = iteration + 1
                best_save_path = os.path.join(log_dir, 'best_ppo_model.pth')
                os.makedirs(os.path.dirname(best_save_path), exist_ok=True)
                agent.save(best_save_path)
                print(f"  [Best Model Saved] Improvement: {best_improvement:.6f}")
    
    # ============================================================
    # 9. 最终评估
    # ============================================================
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    final_eval = evaluate_model_comparison(
        agent, flow_model, env, data, config, device, single_target
    )
    
    print(f"\n[Final Results]")
    print(f"  Flow Model Violation: {final_eval['flow_violation_mean']:.6f} ± {final_eval['flow_violation_std']:.6f}")
    print(f"  PPO Model Violation:  {final_eval['ppo_violation_mean']:.6f} ± {final_eval['ppo_violation_std']:.6f}")
    print(f"  Final Improvement: {final_eval['improvement']:.6f} ({final_eval['improvement_ratio']*100:.2f}%)")
    print(f"  Best Improvement: {best_improvement:.6f} at Iteration {best_iteration}")
    
    # 判断PPO是否有效改善了约束违反
    if final_eval['improvement'] > 0:
        print(f"\n  ✓ PPO training IMPROVED constraint violation!")
    else:
        print(f"\n  ✗ PPO training did NOT improve constraint violation.")
        print(f"    Suggestions:")
        print(f"    - Try smaller actor_lr (current: {config.actor_lr})")
        print(f"    - Try smaller clip_epsilon (current: {config.clip_epsilon})")
        print(f"    - Try larger kl_coef (current: {config.kl_coef})")
        print(f"    - Try more training iterations")
    
    # 保存最终模型
    print("\n[Step 10] Saving trained model...")
    final_save_path = os.path.join(log_dir, 'final_ppo_model.pth')
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    agent.save(final_save_path)
    print(f"  Model saved: {final_save_path}")
    
    # 关闭TensorBoard写入器
    writer.close()
    print(f"\n[TensorBoard] To view logs, run:")
    print(f"  tensorboard --logdir={log_dir}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    main()

