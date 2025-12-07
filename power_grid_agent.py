"""
电力系统智能体模型定义
基于DDPG (Deep Deterministic Policy Gradient) 算法设计的电力系统智能体
集成了电压相角预测和功率计算功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np  
# from agent2_pq import agent2
from env import BranchCurrentLayer
from flow_model.models.actor import Actor, ActorFlow, Critic

class QValueNet(nn.Module):
    """Critic网络，用于评估状态-动作对的价值"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, a):
        """评估状态x和动作a的价值"""
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class PowerGridAgent:
    """基于DDPG架构的电力系统智能体"""
    def __init__(self, env, hidden_dim=256, actor_lr=1e-5, critic_lr=1e-3, gamma=0.99, 
                 tau=0.005, sigma=0.2, device='cpu', writer=None, updated_v=False,
                 critic_loss_threshold=0.1, enable_actor_threshold=True,
                 use_flow_model=False, flow_model_type='rectified', flow_model_path=None,
                 flow_args=None):
        """
        初始化智能体
        
        Args:
            env: 电力系统环境
            hidden_dim: 隐藏层维度
            actor_lr: Actor网络学习率
            critic_lr: Critic网络学习率
            gamma: 折扣因子
            tau: 目标网络软更新参数
            sigma: 探索噪声标准差
            device: 计算设备
            writer: TensorBoard SummaryWriter对象，用于记录训练指标
            critic_loss_threshold: Critic损失阈值，当损失低于此值时才更新Actor
            enable_actor_threshold: 是否启用Actor更新阈值机制
            use_flow_model: 是否使用流模型（ActorFlow）
            flow_model_type: 流模型类型 (rectified, simple, vae等)
            flow_model_path: 预训练流模型的路径
            flow_args: 流模型的参数字典
        """
        self.env = env
        self.gamma = gamma
        self.tau = tau  # 软更新参数
        self.sigma = sigma  # 高斯噪声的标准差
        self.device = device
        self.update_count = 0  # 用于累积更新计数
        self.writer = writer  # TensorBoard writer 
        
        # 环境相关参数
        self.num_buses = len(env.net.bus)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.num_gen + env.num_pg
        
        # 保存流模型配置
        self.use_flow_model = use_flow_model
        self.flow_model_type = flow_model_type
        self.flow_model_path = flow_model_path

        # 根据配置选择Actor网络类型
        if use_flow_model:
            print(f"使用 ActorFlow 网络，模型类型: {flow_model_type}") 
            
            self.actor = ActorFlow(
                input_dim=self.state_dim, 
                env=env, 
                output_dim=self.num_buses,
                norm=False,
                args=flow_args,
                model_type=flow_model_type,
                device=device
            ).to(device) 
            
            # 如果提供了预训练模型路径，加载模型
            if flow_model_path is not None:
                print(f" 加载预训练流模型: {flow_model_path}")
                self.actor.load_model(flow_model_path) 
        else:
            print(" 使用标准 Actor 网络")
            self.actor = Actor(input_dim=self.state_dim, env=env, output_dim=self.num_buses).to(device)
            self.target_actor = Actor(input_dim=self.state_dim, env=env, output_dim=self.num_buses).to(device) 
        
        # region # 创建Critic网络 (价值网络)
        # if updated_v:
        #     self.critic = Critic(self.state_dim, self.action_dim).to(device)
        #     self.target_critic = Critic(self.state_dim, self.action_dim).to(device)
        # else:
        #     self.critic = QValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        #     self.target_critic = QValueNet(self.state_dim, hidden_dim, self.action_dim).to(device) 
        
        # # 初始化目标网络参数与原网络相同
        # self.target_actor.load_state_dict(self.actor.state_dict())
        # self.target_critic.load_state_dict(self.critic.state_dict())
        
        # # 设置目标网络为评估模式
        # self.target_actor.eval()
        # self.target_critic.eval()
        
        # # 定义优化器
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)  
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # # 定义学习率调度器
        # # 根据总训练步数(300*288≈86400)设计学习率衰减策略 
        # # self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.9)
        # # self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.9)
        # if not updated_v:
        #     self.branch_current_layer = BranchCurrentLayer(self.env.Ybus, self.env.Yf, self.env.Yt, device=self.device)
        #     # 预计算线路电流限制的pu值（这些参数在训练过程中不会改变）
        #     self._precompute_thermal_limits()
        # else:
        #     self.branch_current_layer = None
        # endregion
    
    def act(self, state, out_ma=False):
        # 这个函数只有在与环境交互的时候用的到 
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 增加批次维度

        if out_ma:
            action, Vm, Va = self.actor.act(state, out_ma=True)
        else:
            action = self.actor.act(state) 

        action[:, self.env.num_gen:] = action[:, self.env.num_gen:] * 100  # 单位转换为MW

        # 检查action是否在cuda上，只有在cuda上才调用.cpu()，否则直接numpy()
        if action.device.type == 'cuda':
            action = action[0].detach().cpu().numpy()
        else:
            action = action[0].detach().numpy()
        if out_ma:  
            return action, Vm, Va
        else:
            return action
    
    def _create_default_flow_args(self, env):
        """
        创建默认的流模型参数配置
        
        返回推理时所需的参数字典。某些参数仅在特定模型类型时使用。
        详细的参数使用情况请参考 flow_model_args_usage_analysis.md
        """
        args = { 
            # === 网络结构参数（必需） ===
            'hidden_dim': 512,           # 神经网络隐藏层维度
            'num_layer': 3,              # 神经网络层数
            'network': 'carbon_tax_aware_mlp',  # 网络架构类型
            'output_act': None,          # 输出层激活函数
            
            # === 流模型核心参数（必需） ===
            'latent_dim': self.num_buses * 2,   # VAE/GAN模型的潜在空间维度
            'inf_step': 100,             # 流模型推理步数（关键参数，影响精度和速度）
            'time_step': 1000,           # 时间步数（模型结构参数，需与训练时一致）
            
            # === 模型配置（必需） ===
            'instance': 'improved_version',     # 模型实例名称（用于加载预训练模型）
            'data_set': 'opf_flow',      # 数据集类型（决定pred_type）
            'output_norm': False,        # 是否对输出进行归一化
            'env': env,                  # 电网环境（用于约束计算）
            
            # === 特定模型类型参数（按需使用） ===
            'ode_solver': 'Euler',       # ODE求解器类型（potential模型使用）
            'eta': 0.5,                  # 噪声控制参数（diffusion模型使用）
        }
        return args
    
    def print_model_summary(self):
        """打印模型参数摘要"""
        def count_parameters(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params, trainable_params
        
        print("="*60)
        print("模型参数摘要:")
        print("="*60)
        
        # Actor网络参数统计
        actor_total, actor_trainable = count_parameters(self.actor)
        actor_type = "ActorFlow" if self.use_flow_model else "Actor"
        print(f"{actor_type}网络:")
        print(f"  总参数数量: {actor_total:,}")
        print(f"  可训练参数: {actor_trainable:,}")
        if self.use_flow_model:
            print(f"  流模型类型: {self.flow_model_type}")
        
        # Critic网络参数统计
        critic_total, critic_trainable = count_parameters(self.critic)
        print(f"Critic网络:")
        print(f"  总参数数量: {critic_total:,}")
        print(f"  可训练参数: {critic_trainable:,}")
        
        print(f"\n总计:")
        print(f"  总参数数量: {actor_total + critic_total:,}")
        print(f"  可训练参数: {actor_trainable + critic_trainable:,}")
        
        # 打印层结构
        print(f"\n{actor_type}网络层结构:")
        print("-" * 60)
        for name, param in self.actor.named_parameters():
            print(f"{name:40} | {str(param.shape):20} | {param.numel():8,}")
        
        print(f"\nCritic网络层结构:")
        print("-" * 60)
        for name, param in self.critic.named_parameters():
            print(f"{name:40} | {str(param.shape):20} | {param.numel():8,}")
    
    def _precompute_thermal_limits(self):
        """
        预计算线路电流限制的pu值
        
        这些参数是电网的固有属性，在训练过程中不会改变：
        - 线路电流限制 (max_i_ka): 导线的物理热限制
        - 基准功率 (sn_mva): 标幺化基准
        - 电压等级 (vn_kv): 母线额定电压等级
        """
        # 获取电流限制 (kA)
        Imax_f_ka = self.env.net.line.max_i_ka.values  # From端电流限制 (kA)
        Imax_t_ka = self.env.net.line.max_i_ka.values  # To端电流限制 (kA)
        
        # 获取基准功率和电压等级
        sn_mva = self.env.net.sn_mva  # 基准功率 (MVA)
        from_bus_vn = self.env.net.bus.loc[self.env.net.line.from_bus].vn_kv.values  # From端电压等级 (kV)
        to_bus_vn = self.env.net.bus.loc[self.env.net.line.to_bus].vn_kv.values     # To端电压等级 (kV)
        
        # 转换为 pu 值并存储为实例变量
        # 电流基准值: I_base = S_base / (sqrt(3) * V_base)
        # 电流 pu = 电流 kA / I_base = 电流 kA * sqrt(3) * V_base / S_base
        sqrt_3 = np.sqrt(3)
        self.Imax_f_pu = Imax_f_ka * sqrt_3 * from_bus_vn / sn_mva  # From端电流限制 pu
        self.Imax_t_pu = Imax_t_ka * sqrt_3 * to_bus_vn / sn_mva    # To端电流限制 pu
        
        print(f"预计算完成: 线路电流限制转换为pu值")
        print(f"From端电流限制范围: [{self.Imax_f_pu.min():.4f}, {self.Imax_f_pu.max():.4f}] pu")
        print(f"To端电流限制范围: [{self.Imax_t_pu.min():.4f}, {self.Imax_t_pu.max():.4f}] pu")

    def thermal_limit_loss(self, If, It, Imax_f=None, Imax_t=None, p=1):
        """
        If, It: (B, nl) 复数电流
        Imax_f, Imax_t: (nl,) 或 (B, nl) 的实数上限（pu 或按同一标幺）
        p: 惩罚幂（2 表示平方）
        返回: 标量 loss
        """
        # 电流幅值 (实数，保留梯度)
        If_mag = torch.abs(If)  # (B, nl)
        It_mag = torch.abs(It)

        loss = 0.0
        if Imax_f is not None:
            Imax_f = torch.as_tensor(Imax_f, device=If.device, dtype=If_mag.dtype)
            if Imax_f.ndim == 1:
                Imax_f = Imax_f.unsqueeze(0).expand_as(If_mag)
            over_f = torch.relu(If_mag - Imax_f)
            loss = loss + torch.mean(over_f**p)

        if Imax_t is not None:
            Imax_t = torch.as_tensor(Imax_t, device=It.device, dtype=It_mag.dtype)
            if Imax_t.ndim == 1:
                Imax_t = Imax_t.unsqueeze(0).expand_as(It_mag)
            over_t = torch.relu(It_mag - Imax_t)
            loss = loss + torch.mean(over_t**p)

        return loss

    # def calc_constraint_violations(self, p_g, q_g, If, It):
    #     """
    #     计算约束违反程度
        
    #     参数:
    #         vm_pu: 电压幅值 (标幺值) - tensor
    #         va_deg: 电压相角 (度) - tensor
    #         p_g: 发电机有功功率 - tensor
    #         q_g: 发电机无功功率 - tensor
    #         load_p: 负荷有功功率 - tensor
    #         load_q: 负荷无功功率 - tensor
    #         I_mag: 线路电流幅值 - tensor，与self.env.net.line对应的电流幅值
            
    #     返回:
    #         dict: 包含各类约束违反程度的tensor字典
    #     """
    #     # 初始化各类约束违反为tensor
    #     p_violation = torch.tensor(0.0, device=self.device)
    #     q_violation = torch.tensor(0.0, device=self.device)
    #     v_violation = torch.tensor(0.0, device=self.device)
    #     i_violation = torch.tensor(0.0, device=self.device) 
        
    #     # 1. 发电机有功功率限制违反惩罚
    #     k = 100   # todo: origin 100
    #     for i, gen in enumerate(self.env.net.gen.itertuples()):
    #         bus_idx = gen.bus
    #         p_min = gen.min_p_mw  
    #         p_max = gen.max_p_mw  
            
    #         if len(p_g.shape) > 1:
    #             # 批处理模式
    #             gen_p = p_g[:, bus_idx]
    #             # 使用torch.clamp替代max函数
    #             p_violation += torch.sum(constraint_violation_smooth(gen_p, torch.tensor(p_min, device=self.device), torch.tensor(p_max, device=self.device),
    #                                                                  k=k))
    #         else:
    #             # 单个样本模式
    #             gen_p = p_g[bus_idx]
    #             p_violation += constraint_violation_smooth(gen_p, torch.tensor(p_min, device=self.device), torch.tensor(p_max, device=self.device),
    #                                                        k=k) 

    #     # 2. 发电机无功功率限制违反惩罚
    #     for i, gen in enumerate(self.env.net.gen.itertuples()):
    #         bus_idx = gen.bus
    #         q_min = gen.min_q_mvar if hasattr(gen, 'min_q_mvar') else 0
    #         q_max = gen.max_q_mvar if hasattr(gen, 'max_q_mvar') else float('inf')
            
    #         if len(q_g.shape) > 1:
    #             # 批处理模式
    #             gen_q = q_g[:, bus_idx] 
    #             q_violation = torch.sum(constraint_violation_smooth(gen_q, torch.tensor(q_min, device=self.device), torch.tensor(q_max, device=self.device),
    #                                                                 k=k))
    #         else:
    #             # 单个样本模式
    #             gen_q = q_g[bus_idx]
    #             q_violation = constraint_violation_smooth(gen_q, torch.tensor(q_min, device=self.device), torch.tensor(q_max, device=self.device),
    #                                                       k=k)

    #     # 3. 线路电流限制违反惩罚
    #     # 使用预计算的电流限制pu值（这些参数在训练过程中不变）
    #     i_violation = self.thermal_limit_loss(If, It, Imax_f=self.Imax_f_pu, Imax_t=self.Imax_t_pu, p=1)

    #     # 加权惩罚项
    #     w1, w2, w3 = 1.0, 1.0, 1.0  # 有功/无功/电压违反惩罚系数 (1/MW)
    #     w4 = 100.0  # 电流违反惩罚系数 αv (1/p.u.)
    #     total_penalty = (w1 * p_violation +  # MW * (1/MW) = 1
    #                     w2 * q_violation +   # Mvar * (1/MW) = 1  
    #                     w3 * v_violation +   # p.u. * (1/MW) = p.u./MW  todo: 这块的约束违反是不用考虑的
    #                     w4 * i_violation)    # p.u. * (1/p.u.) = 1
        
    #     return {
    #         'p_violation': p_violation,
    #         'q_violation': q_violation,
    #         'v_violation': v_violation,
    #         'i_violation': i_violation,
    #         'total_penalty': total_penalty
    #     }
    def actor_forward(self, state, actor):
        """
        Actor网络前向传播，处理DDP包装的情况
        
        Args:
            state: 输入状态
            actor: Actor网络（可能是DDP包装的）
            
        Returns:
            action: 预测的动作
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 增加批次维度

        # 处理DDP包装的模型
        with torch.no_grad():
            if hasattr(actor, 'module'):
                action = actor.module(state)
            else:
                action = actor(state)

        # 还原电压归一化：先乘以缩放因子，再从[0,1]映射回实际电压范围
        scale_vm = 10.0  # 与训练时相同的缩放因子
        action[:, :self.num_buses] = action[:, :self.num_buses] / scale_vm  # 先除以缩放因子
        action[:, :self.num_buses] = action[:, :self.num_buses] * (self.env.voltage_high[0] - self.env.voltage_low[0]) + self.env.voltage_low[0]
        
        # 还原相角归一化：先乘以缩放因子，再从弧度转回角度
        scale_va = 10.0  # 与训练时相同的缩放因子
        action[:, self.num_buses:] = action[:, self.num_buses:] / scale_va  # 先除以缩放因子
        action[:, self.num_buses:] = action[:, self.num_buses:] * 180 / np.pi  # 从弧度转回角度
        return action

    # def take_action(self, state, test=False, target_actor=False, add_noise=False):
    #     """
    #     根据状态选择动作
        
    #     Args:
    #         state: 环境状态 
    #         target_actor: 是否调用目标网络进行动作选择          
    #         add_noise: 是否加入噪声
    #     Returns:
    #         action: 智能体动作
    #         vm_pu: 标幺电压
    #         va_deg: 相角（度）
    #     """
    #     # 将输入转换为tensor格式并移至指定设备
    #     if not isinstance(state, torch.Tensor):
    #         state = torch.tensor(state, dtype=torch.float32).to(self.device)
    #     if len(state.shape) == 1:
    #         state = state.unsqueeze(0)  # 增加批次维度

    #     # 选择使用的Actor网络（目标网络或当前网络）
    #     actor_network = self.target_actor if target_actor else self.actor
        
    #     if test:
    #         # 测试模式：不计算梯度
    #         actor_network.eval()  # 设置为评估模式
    #         with torch.no_grad():
    #             # 处理DDP包装的模型
    #             if hasattr(actor_network, 'module'):
    #                 action = actor_network.module(state)
    #             else:
    #                 action = actor_network(state)
    #         actor_network.train()  # 恢复为训练模式
    #     else:
    #         # 训练模式：计算梯度
    #         # 处理DDP包装的模型
    #         if hasattr(actor_network, 'module'):
    #             action = actor_network.module(state)
    #         else:
    #             action = actor_network(state)  # 输出的action=[电压, 相角] 电压需要归一化，相角需要转换为弧度

    #     # 根据测试模式和是否添加噪声处理动作
    #     if test and add_noise:
    #         action = action.cpu().numpy().flatten()   # 这块有个问题，如果是目标网络过来对一个批次的数据，然后还是测试模型的话，那么岂不是action也要抹平，这就不对了啊
        
    #     # 还原电压归一化
    #     scale_vm = 10.0
    #     # 还原相角归一化
    #     scale_va = 10.0
        
    #     # 根据动作形状选择不同的处理方式
    #     if len(action.shape) == 1:  # 一维数组(测试模式)
    #         action[:self.num_buses] = action[:self.num_buses] / scale_vm
    #         action[:self.num_buses] = action[:self.num_buses] * (self.env.voltage_high[0] - self.env.voltage_low[0]) + self.env.voltage_low[0]
            
    #         action[self.num_buses:] = action[self.num_buses:] / scale_va
    #         action[self.num_buses:] = action[self.num_buses:] * 180 / np.pi
    #     else:  # 二维数组(批处理模式)
    #         action[:, :self.num_buses] = action[:, :self.num_buses] / scale_vm
    #         action[:, :self.num_buses] = action[:, :self.num_buses] * (self.env.voltage_high[0] - self.env.voltage_low[0]) + self.env.voltage_low[0]
            
    #         action[:, self.num_buses:] = action[:, self.num_buses:] / scale_va
    #         action[:, self.num_buses:] = action[:, self.num_buses:] * 180 / np.pi
        
    #     # 添加噪声进行探索
    #     if len(action.shape) == 1:
    #         if not target_actor and add_noise:
    #             noise = self.sigma * np.random.randn(*action.shape) 
    #             # 确保动作在合理范围内 加噪声用于提升actor网络的探索能力
    #             voltage = np.clip(
    #                 action[:self.num_buses] + noise[:self.num_buses], 
    #                 self.env.voltage_low[0], 
    #                 self.env.voltage_high[0]
    #             )
    #             angle = np.clip(
    #                 action[self.num_buses:] + noise[self.num_buses:], 
    #                 self.env.angle_low[0], 
    #                 self.env.angle_high[0]
    #             )
    #         else:
    #             voltage = np.clip(
    #                 action[:self.num_buses], 
    #                 self.env.voltage_low[0], 
    #                 self.env.voltage_high[0]
    #             )
    #             angle = np.clip(
    #                 action[self.num_buses:], 
    #                 self.env.angle_low[0], 
    #                 self.env.angle_high[0]
    #             )
    #     else:
    #         voltage = action[:, :self.num_buses]
    #         angle = action[:, self.num_buses:]
        
    #     # 合并动作
    #     # 从状态中提取负荷信息，而不是直接从环境中获取
    #     load_p = state[:, :len(self.env.net.load)]
    #     load_q = state[:, len(self.env.net.load):2*len(self.env.net.load)]

    #     if len(voltage.shape) == 1:
    #         load_p = load_p.flatten()
    #         load_q = load_q.flatten()

    #     p_g, q_g, _, _, _ = agent2(self.env.net, vm_pu=voltage, va_deg=angle, Ybus=self.env.Ybus,
    #                            load_p=load_p, load_q=load_q,
    #                            gen_buses=self.env.net.gen.bus.values, load_buses=self.env.net.load.bus.values)

    #     if isinstance(voltage, torch.Tensor) and isinstance(p_g, torch.Tensor):
    #         action = torch.cat([voltage, p_g], dim=1)
    #     else:
    #         action = np.concatenate([voltage, p_g])
    #     return action, voltage, angle, p_g, q_g
    
    
    def soft_update(self, net, target_net):
        """软更新目标网络参数: θ' = (1-τ)θ' + τθ"""
        # 处理DDP包装过的模型
        if hasattr(net, 'module'):
            source_net = net.module
        else:
            source_net = net
            
        for param, param_target in zip(source_net.parameters(), target_net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )
    # startregion 这块的代码是原来DDPG的代码，现在失效了，先注释掉
    # def _should_update_actor(self, current_critic_loss):
    #     """
    #     判断是否应该更新Actor网络
        
    #     Args:
    #         current_critic_loss: 当前的Critic损失值
            
    #     Returns:
    #         bool: 是否应该更新Actor
    #     """
    #     # 如果未启用阈值机制，始终更新Actor
    #     if not self.enable_actor_threshold:
    #         return True
        
    #     # 如果已经达到阈值，之后始终更新Actor
    #     if self.actor_threshold_reached:
    #         return True
        
    #     # 记录当前损失到历史记录
    #     self.critic_loss_history.append(current_critic_loss)
        
    #     # 保持滑动窗口大小
    #     if len(self.critic_loss_history) > self.history_window:
    #         self.critic_loss_history.pop(0)
        
    #     # 需要有足够的历史数据才能判断
    #     if len(self.critic_loss_history) < self.history_window:
    #         return False
        
    #     # 计算滑动平均损失
    #     avg_critic_loss = sum(self.critic_loss_history) / len(self.critic_loss_history)
        
    #     # 检查是否达到阈值
    #     if avg_critic_loss <= self.critic_loss_threshold:
    #         self.actor_threshold_reached = True
    #         if self.writer is not None:
    #             self.writer.add_text('Training/threshold_reached', 
    #                                f'Critic loss threshold reached at update {self.update_count}, '
    #                                f'avg_loss: {avg_critic_loss:.6f}, threshold: {self.critic_loss_threshold}',
    #                                self.update_count)
    #         return True
        
    #     return False
    
    # def to_tensor(self, data, dtype=torch.float32):
    #     if isinstance(data, torch.Tensor):
    #         return data.to(dtype).to(self.device)
    #     else:
    #         return torch.tensor(np.array(data), dtype=dtype).to(self.device)


    def update(self, transition_dict):
        """使用DDPG算法更新网络参数 - 优化版本""" 
        
        # 1. 数据预处理 - 批量转换到GPU
        states = self.to_tensor(transition_dict['states'])
        actions = self.to_tensor(transition_dict['actions'])
        rewards = self.to_tensor(transition_dict['rewards']).view(-1, 1)
        next_states = self.to_tensor(transition_dict['next_states'])
        dones = self.to_tensor(transition_dict['dones'], torch.float32).view(-1, 1)

        # 2. 缓存模型引用以减少重复检查
        critic_model = self.critic.module if hasattr(self.critic, 'module') else self.critic
        target_critic_model = self.target_critic.module if hasattr(self.target_critic, 'module') else self.target_critic 

        # 3. 更新Critic网络
        # 计算目标Q值 - 使用无梯度上下文
        with torch.no_grad():
            next_actions = self.take_action(next_states, test=True, target_actor=True)[0]
            next_q_values = target_critic_model(next_states, next_actions)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算当前Q值和Critic损失
        q_values = critic_model(states, actions)
        critic_loss = F.mse_loss(q_values, q_targets)
        
        # Critic网络更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        # self.critic_scheduler.step()

        # 软更新Critic的目标网络
        self.soft_update(self.critic, self.target_critic)

        # 4. 条件更新Actor网络
        current_critic_loss = critic_loss.item()
        should_update_actor = self._should_update_actor(current_critic_loss)
        should_update_actor = True   # to-do： IEEE 118案例过大，这个critic的loss就没法很小，所以这里先设置为True

        actor_loss = torch.tensor(0.0, device=self.device)
        constraint_loss = torch.tensor(0.0, device=self.device)
        total_actor_loss = torch.tensor(0.0, device=self.device)
        
        if should_update_actor:
            # 计算当前动作和约束
            current_actions, vm_pu, va_deg, p_g, q_g = self.take_action(states, test=False)
            
            # 并行计算电流和约束违反
            Ibus, If, It, _, _ = self.branch_current_layer(vm_pu, va_deg)
            If = If.index_select(dim=1, index=self.env.line_rows)
            It = It.index_select(dim=1, index=self.env.line_rows)
            violations = self.calc_constraint_violations(p_g, q_g, If, It)
            
            # Actor损失计算
            actor_loss = -torch.mean(critic_model(states, current_actions))
            constraint_loss = violations['total_penalty']
            total_actor_loss = actor_loss + constraint_loss 
            
            # Actor网络更新
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            # self.actor_scheduler.step()
            
            # 软更新Actor的目标网络
            self.soft_update(self.actor, self.target_actor)
        
        # 5. 记录指标 (减少频率，每10次更新记录一次)
        if self.writer is not None and self.update_count % 10 == 0:
            self.writer.add_scalar('Loss/actor_objective', actor_loss.item(), self.update_count)
            self.writer.add_scalar('Loss/constraint_penalty', constraint_loss.item(), self.update_count)
            self.writer.add_scalar('Loss/total_actor_loss', total_actor_loss.item(), self.update_count)
            self.writer.add_scalar('Loss/critic', critic_loss.item(), self.update_count)
            self.writer.add_scalar('Training/should_update_actor', float(should_update_actor), self.update_count)
            
            # 记录约束违反详情（仅在更新Actor时有效）
            if should_update_actor:
                self.writer.add_scalar('Constraints/p_violation', violations['p_violation'].item(), self.update_count)
                self.writer.add_scalar('Constraints/q_violation', violations['q_violation'].item(), self.update_count)
                self.writer.add_scalar('Constraints/v_violation', violations['v_violation'].item(), self.update_count)
                if 'i_violation' in violations:
                    self.writer.add_scalar('Constraints/i_violation', violations['i_violation'].item(), self.update_count)
        
        self.update_count += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'should_update_actor': should_update_actor,
            'actor_threshold_reached': self.actor_threshold_reached
        }
    # endregion
    
    def save(self, path):
        """保存完整模型，处理DDP包装的情况"""
        # 获取原始模型（非DDP包装）
        actor = self.actor.module if hasattr(self.actor, 'module') else self.actor
        critic = self.critic.module if hasattr(self.critic, 'module') else self.critic
        
        torch.save({
            'actor': actor.state_dict(),
            'critic': critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def save_actor_only(self, path):
        """只保存Actor模型，与预训练模型格式兼容"""
        # 获取原始模型（非DDP包装）
        actor = self.actor.module if hasattr(self.actor, 'module') else self.actor
        torch.save(actor.state_dict(), path)
    
    def load(self, path):
        """加载完整模型，处理DDP包装的情况"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 获取原始模型（非DDP包装）
        actor = self.actor.module if hasattr(self.actor, 'module') else self.actor
        critic = self.critic.module if hasattr(self.critic, 'module') else self.critic
        
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    
    def load_actor_only(self, path):
        """只加载Actor模型，用于加载预训练模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 获取原始模型（非DDP包装）
        actor = self.actor.module if hasattr(self.actor, 'module') else self.actor
        
        if isinstance(checkpoint, dict) and 'actor' in checkpoint:
            # 新格式 - 包含多个组件的字典
            actor.load_state_dict(checkpoint['actor'])
            self.target_actor.load_state_dict(checkpoint['actor'])
        else:
            # 旧格式 - 直接是actor的state_dict
            actor.load_state_dict(checkpoint)
            self.target_actor.load_state_dict(checkpoint) 

    def check_model_parameters(self, model, name="model"):
        """检查模型参数是否包含NaN或无穷大值"""
        has_problem = False
        for param_name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                # print(f"{name}的参数{param_name}包含NaN或无穷大值")
                has_problem = True
        return has_problem 
    


import torch

def constraint_violation(r: torch.Tensor, p: torch.Tensor, p_: torch.Tensor) -> torch.Tensor:
    """
    计算 r 的约束违反程度（可微）：
    - 如果 r < p: 违反 = p - r
    - 如果 r > p_: 违反 = r - p_
    - 否则违反 = 0
    """
    violation_lower = torch.relu(p - r)  # 当 r < p 时，p - r > 0
    violation_upper = torch.relu(r - p_)  # 当 r > p_ 时，r - p_ > 0
    total_violation = violation_lower + violation_upper  # 总违反程度
    return total_violation

def constraint_violation_smooth(r: torch.Tensor, p: torch.Tensor, p_: torch.Tensor, 
                               k=10.0, method='relu') -> torch.Tensor:
    """
    计算约束违反，支持两种方法：
    
    Args:
        r: 待约束的变量
        p: 约束下界
        p_: 约束上界
        k: sigmoid 方法的陡峭度参数（越大越接近阶跃函数）
        method: 'sigmoid' 或 'relu'，选择约束违反计算方法
        
    Returns:
        约束违反程度（非负值）
        
    Methods:
        - 'sigmoid': 使用 sigmoid 函数，提供平滑的约束违反计算
                    在约束边界处可微，有利于梯度优化
        - 'relu': 使用 ReLU 函数，提供硬约束违反计算
                 计算更简单快速，但在边界处不可微
    """
    if method.lower() == 'sigmoid':
        # Sigmoid 版本：平滑约束
        violation_lower = (p - r) * torch.sigmoid(k * (p - r))  # 当 r < p 时接近 p - r
        violation_upper = (r - p_) * torch.sigmoid(k * (r - p_))  # 当 r > p_ 时接近 r - p_
        return violation_lower + violation_upper
        
    elif method.lower() == 'relu':
        # ReLU 版本：硬约束
        violation_lower = torch.relu(p - r)  # max(0, p - r)
        violation_upper = torch.relu(r - p_)  # max(0, r - p_)
        return violation_lower + violation_upper
        
    else:
        raise ValueError(f"未支持的方法 '{method}'。请选择 'sigmoid' 或 'relu'")


if __name__ == "__main__":
    # 验证电流计算的是否正确
    pass