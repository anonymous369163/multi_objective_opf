"""
PPO (Proximal Policy Optimization) 算法用于微调流模型
直接微调Flow模型参数，添加KL散度约束，熵系数为0

核心思想：
1. 直接微调Flow模型参数（而非添加残差网络）
2. 使用PPO的clip机制限制每次策略更新幅度
3. 添加KL散度约束防止策略偏离原始模型太远
4. 设置熵系数为0，不强制探索（预训练模型已经足够好）

参考：RLfinetuning_Diffusion_Bioseq项目的PPO实现

作者：基于rl_opf_flow项目
日期：2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import copy


class PPOActor(nn.Module):
    """
    PPO策略网络 - 直接使用/微调Flow模型
    
    与SAC不同，这里：
    1. 直接微调flow_network的参数
    2. 使用固定的小方差进行采样（可配置）
    3. 不使用残差模式
    """
    
    def __init__(self, state_dim, action_dim, flow_model, args):
        """
        初始化PPO Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（速度向量维度）
            flow_model: 预训练的FM流模型
            args: 参数字典
        """
        super(PPOActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 直接复制flow_model.model并允许微调
        self.flow_network = copy.deepcopy(flow_model.model)
        
        # 确保flow_network参数可训练
        for param in self.flow_network.parameters():
            param.requires_grad = True
        
        # 保存维度信息
        self.x_condition_dim = args.get('input_dim', 189)
        self.output_dim = args.get('output_dim', 236)
        
        # ===== 固定的log_std（可学习或固定） =====
        # PPO通常使用可学习的log_std
        self.log_std_init = args.get('log_std_init', -3.0)  # 初始std约0.05
        self.learn_std = args.get('learn_std', True)
        
        if self.learn_std:
            # 可学习的log_std参数
            self.log_std = nn.Parameter(torch.ones(action_dim) * self.log_std_init)
        else:
            # 固定的log_std
            self.register_buffer('log_std', torch.ones(action_dim) * self.log_std_init)
        
        # std的范围限制
        self.log_std_min = args.get('log_std_min', -5.0)  # std最小约0.007
        self.log_std_max = args.get('log_std_max', -1.0)  # std最大约0.37
        
    def forward(self, state):
        """
        前向传播，输出动作的均值
        
        Args:
            state: [batch_size, state_dim] 状态张量
        
        Returns:
            mu: [batch_size, action_dim] 动作均值（速度向量）
        """
        # 从state中提取各个组件
        # state = [x_condition, t, z_t]
        x_condition = state[:, :self.x_condition_dim]
        t = state[:, self.x_condition_dim:self.x_condition_dim+1]
        z_t = state[:, self.x_condition_dim+1:]
        
        # 使用flow_network计算速度
        # flow_network的输入是 (x, z, t)
        mu = self.flow_network(x_condition, z_t, t)
        
        return mu
    
    def get_action_and_log_prob(self, state, deterministic=False):
        """
        获取动作和对数概率
        
        Args:
            state: [batch_size, state_dim] 状态
            deterministic: 是否使用确定性策略
        
        Returns:
            action: [batch_size, action_dim] 采样的动作
            log_prob: [batch_size] 对数概率
        """
        mu = self.forward(state)
        
        # 限制log_std范围
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mu
            # 对于确定性动作，计算在该点的log_prob
            dist = Normal(mu, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            # 采样动作
            dist = Normal(mu, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, state, action):
        """
        评估给定动作的对数概率和熵
        
        Args:
            state: [batch_size, state_dim] 状态
            action: [batch_size, action_dim] 动作
        
        Returns:
            log_prob: [batch_size] 对数概率
            entropy: [batch_size] 熵
        """
        mu = self.forward(state)
        
        # 限制log_std范围
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class PPOCritic(nn.Module):
    """
    PPO价值网络 - 评估状态价值V(s)
    """
    
    def __init__(self, state_dim, hidden_dim=256):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super(PPOCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        # 最后一层使用较小的初始化
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
    
    def forward(self, state):
        """
        前向传播，输出状态价值
        
        Args:
            state: [batch_size, state_dim] 状态
        
        Returns:
            value: [batch_size, 1] 状态价值
        """
        return self.network(state)


class RolloutBuffer:
    """
    PPO的Rollout Buffer - 存储一个epoch的轨迹数据
    
    与Replay Buffer不同，Rollout Buffer：
    1. 每次收集完整的轨迹后进行一次更新
    2. 更新后清空buffer（on-policy）
    3. 存储额外的信息如log_prob、value、advantage等
    """
    
    def __init__(self, buffer_size, state_dim, action_dim, device='cuda', gamma=0.99, gae_lambda=0.95):
        """
        初始化Rollout Buffer
        
        Args:
            buffer_size: buffer大小
            state_dim: 状态维度
            action_dim: 动作维度
            device: 设备
            gamma: 折扣因子
            gae_lambda: GAE的lambda参数
        """
        self.buffer_size = buffer_size
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 初始化存储张量
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        
        # GAE计算后填充
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False
    
    def add(self, state, action, reward, done, log_prob, value):
        """
        添加一条转移记录
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
    
    def compute_gae(self, last_value, last_done):
        """
        计算广义优势估计 (GAE)
        
        Args:
            last_value: 最后一个状态的价值估计
            last_done: 最后一个状态是否结束
        """
        last_gae_lam = 0
        
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # 计算returns
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
    
    def get(self, batch_size=None):
        """
        获取所有数据（用于PPO更新）
        
        Args:
            batch_size: 如果指定，返回随机采样的mini-batch
        
        Returns:
            字典形式的数据
        """
        if batch_size is None or batch_size >= self.ptr:
            indices = np.arange(self.ptr)
        else:
            indices = np.random.choice(self.ptr, batch_size, replace=False)
        
        # 标准化优势
        advantages = self.advantages[indices]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'old_log_probs': self.log_probs[indices],
            'advantages': advantages,
            'returns': self.returns[indices],
            'values': self.values[indices]
        }
    
    def clear(self):
        """
        清空buffer
        """
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False


class PPOFlowModel:
    """
    PPO算法用于微调Flow模型
    
    关键特点：
    1. 直接微调Flow模型参数
    2. 使用clip机制限制策略更新幅度
    3. 添加KL散度约束
    4. 熵系数为0
    """
    
    def __init__(self, flow_model, state_dim, action_dim, args, device='cuda'):
        """
        初始化PPO Flow Model
        
        Args:
            flow_model: 预训练的FM流模型
            state_dim: 状态维度
            action_dim: 动作维度
            args: 参数字典，包含：
                - actor_lr: Actor学习率（默认3e-4）
                - critic_lr: Critic学习率（默认3e-4）
                - gamma: 折扣因子（默认0.99）
                - gae_lambda: GAE的lambda参数（默认0.95）
                - clip_epsilon: PPO clip参数（默认0.2）
                - entropy_coef: 熵系数（默认0.0）
                - value_coef: 价值损失系数（默认0.5）
                - max_grad_norm: 梯度裁剪（默认0.5）
                - kl_target: KL散度目标值（默认0.01）
                - kl_coef: KL散度约束系数（默认0.0，设为>0启用）
                - n_epochs: 每次更新的epoch数（默认10）
                - batch_size: mini-batch大小（默认64）
            device: 计算设备
        """
        self.device = device
        
        # ===== 超参数 =====
        self.gamma = args.get('gamma', 0.99)
        self.gae_lambda = args.get('gae_lambda', 0.95)
        self.clip_epsilon = args.get('clip_epsilon', 0.2)
        self.entropy_coef = args.get('entropy_coef', 0.0)  # 设为0！
        self.value_coef = args.get('value_coef', 0.5)
        self.max_grad_norm = args.get('max_grad_norm', 0.5)
        self.kl_target = args.get('kl_target', 0.01)
        self.kl_coef = args.get('kl_coef', 0.2)  # KL散度约束系数
        self.n_epochs = args.get('n_epochs', 10)
        self.batch_size = args.get('batch_size', 64)
        
        # ===== Actor网络（可训练） =====
        self.actor = PPOActor(state_dim, action_dim, flow_model, args).to(device)
        
        # ===== 原始Actor（冻结，用于KL约束） =====
        self.original_actor = PPOActor(state_dim, action_dim, flow_model, args).to(device)
        for param in self.original_actor.parameters():
            param.requires_grad = False
        self.original_actor.eval()
        
        # ===== Critic网络 =====
        hidden_dim = args.get('hidden_dim', 256)
        self.critic = PPOCritic(state_dim, hidden_dim).to(device)
        
        # ===== 优化器 =====
        actor_lr = args.get('actor_lr', 3e-4)
        critic_lr = args.get('critic_lr', 3e-4)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # ===== 自适应KL系数（可选） =====
        self.adaptive_kl = args.get('adaptive_kl', True)
        
        print(f"PPOFlowModel 初始化完成:")
        print(f"  clip_epsilon: {self.clip_epsilon}")
        print(f"  entropy_coef: {self.entropy_coef}")
        print(f"  kl_coef: {self.kl_coef}")
        print(f"  n_epochs: {self.n_epochs}")
        print(f"  actor_lr: {actor_lr}, critic_lr: {critic_lr}")
    
    def take_action(self, state, deterministic=False):
        """
        选择动作
        
        Args:
            state: 状态
            deterministic: 是否使用确定性策略
        
        Returns:
            action: 动作
            log_prob: 对数概率
            value: 状态价值
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(state, deterministic)
            value = self.critic(state).squeeze(-1)
        
        return action, log_prob, value
    
    def compute_kl_divergence(self, states):
        """
        计算原始策略与当前策略之间的KL散度
        
        标准PPO使用 KL(π_old || π_new)，即惩罚新策略在旧策略有概率的地方变成零概率
        这样可以保护预训练知识，实现保守更新
        
        Args:
            states: [batch_size, state_dim] 状态
        
        Returns:
            kl: 平均KL散度
        """
        # 当前策略的输出
        mu_new = self.actor.forward(states)
        log_std_new = torch.clamp(self.actor.log_std, self.actor.log_std_min, self.actor.log_std_max)
        std_new = torch.exp(log_std_new)
        
        # 原始策略的输出
        with torch.no_grad():
            mu_old = self.original_actor.forward(states)
            log_std_old = torch.clamp(self.original_actor.log_std, 
                                      self.original_actor.log_std_min, 
                                      self.original_actor.log_std_max)
            std_old = torch.exp(log_std_old)
        
        # 计算KL散度：KL(π_old || π_new)  [标准PPO方向]
        # 公式：KL(old||new) = log(σ_new/σ_old) + (σ_old² + (μ_old - μ_new)²) / (2σ_new²) - 0.5
        var_ratio = (std_old / std_new) ** 2  # σ_old² / σ_new²
        mu_diff_sq = ((mu_old - mu_new) / std_new) ** 2  # (μ_old - μ_new)² / σ_new²
        kl = 0.5 * (var_ratio + mu_diff_sq - 1 - torch.log(var_ratio))
        
        return kl.sum(dim=-1).mean()
    
    def update(self, rollout_buffer):
        """
        使用PPO算法更新策略
        
        Args:
            rollout_buffer: RolloutBuffer对象
        
        Returns:
            info: 训练信息字典
        """
        # 获取所有数据
        all_data = rollout_buffer.get()
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(rollout_buffer.ptr)
            
            for start in range(0, rollout_buffer.ptr, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                if len(batch_indices) < 2:
                    continue
                
                # 获取mini-batch数据
                states = all_data['states'][batch_indices]
                actions = all_data['actions'][batch_indices]
                old_log_probs = all_data['old_log_probs'][batch_indices]
                advantages = all_data['advantages'][batch_indices]
                returns = all_data['returns'][batch_indices]
                
                # 重新标准化advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # ===== 计算新的log_prob和entropy =====
                new_log_probs, entropy = self.actor.evaluate_actions(states, actions)
                
                # ===== 计算PPO的ratio =====
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # ===== 计算clip后的目标 =====
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # ===== 添加KL散度约束 =====
                if self.kl_coef > 0:
                    kl = self.compute_kl_divergence(states)
                    actor_loss = actor_loss + self.kl_coef * kl
                    total_kl += kl.item()
                
                # ===== 添加熵正则化（系数为0） =====
                entropy_loss = -entropy.mean()
                actor_loss = actor_loss + self.entropy_coef * entropy_loss
                
                # ===== 更新Actor =====
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # ===== 计算Critic损失 =====
                values = self.critic(states).squeeze(-1)
                critic_loss = F.mse_loss(values, returns)
                
                # ===== 更新Critic =====
                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # ===== 自适应调整KL系数 =====
        if self.adaptive_kl and self.kl_coef > 0 and n_updates > 0:
            avg_kl = total_kl / n_updates
            if avg_kl > 1.5 * self.kl_target:
                self.kl_coef *= 1.5
            elif avg_kl < 0.5 * self.kl_target:
                self.kl_coef *= 0.5
            self.kl_coef = max(0.01, min(10.0, self.kl_coef))
        
        info = {
            'actor_loss': total_actor_loss / max(n_updates, 1),
            'critic_loss': total_critic_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'kl_divergence': total_kl / max(n_updates, 1),
            'kl_coef': self.kl_coef,
            'n_updates': n_updates
        }
        
        return info
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'kl_coef': self.kl_coef
        }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if 'kl_coef' in checkpoint:
            self.kl_coef = checkpoint['kl_coef']
        print(f"模型已从 {path} 加载")

