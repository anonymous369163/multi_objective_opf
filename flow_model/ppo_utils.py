"""
PPO工具模块 - 用于RL微调流模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValueCritic(nn.Module):
    """
    价值网络：估计状态价值V(s)
    
    输入: (x, z, t) - 条件、当前ODE状态、时间步
    输出: V(s) - 状态价值
    """
    def __init__(self, condition_dim, state_dim, hidden_dim=256):
        super().__init__()
        # condition_dim: x的维度 (例如189)
        # state_dim: z的维度 (例如236)
        input_dim = condition_dim + state_dim + 1  # +1 for time
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, z, t):
        """
        Args:
            x: 条件输入 (batch_size, condition_dim)
            z: ODE状态 (batch_size, state_dim)
            t: 时间步 (batch_size, 1) 或标量
        """
        if isinstance(t, (int, float)):
            t = torch.tensor([[t]], device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # 扩展t到batch维度
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0], -1)
        
        state = torch.cat([x, z, t], dim=1)
        return self.net(state)


def compute_gae(values, rewards, gamma=0.99, gae_lambda=0.95):
    """
    计算广义优势估计 (Generalized Advantage Estimation)
    
    Args:
        values: 每步的价值估计列表 [V(s_0), V(s_1), ..., V(s_T)]
        rewards: 每步的奖励列表 [r_0, r_1, ..., r_T]
        gamma: 折扣因子
        gae_lambda: GAE lambda参数
    
    Returns:
        advantages: 优势估计列表
        returns: 回报估计列表
    """
    T = len(rewards)
    values = values + [0.0]  # 添加终态价值V(s_T+1) = 0
    
    advantages = []
    gae = 0
    
    for i in reversed(range(T)):
        delta = rewards[i] + gamma * values[i+1] - values[i]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    
    return advantages, returns


def setup_finetune(flow_model, num_finetune_layers=2):
    """
    设置流模型的微调：冻结大部分层，只微调最后几层
    
    Args:
        flow_model: FM类实例
        num_finetune_layers: 要微调的最后几个Linear层数
    
    Returns:
        finetune_params: 可训练参数列表
    """
    # 首先冻结所有参数
    for param in flow_model.model.parameters():
        param.requires_grad = False
    
    finetune_params = []
    
    # 找到net中的所有Linear层
    linear_indices = []
    for i, module in enumerate(flow_model.model.net):
        if isinstance(module, nn.Linear):
            linear_indices.append(i)
    
    # 解冻最后num_finetune_layers个Linear层
    if len(linear_indices) >= num_finetune_layers:
        layers_to_unfreeze = linear_indices[-num_finetune_layers:]
        for idx in layers_to_unfreeze:
            for param in flow_model.model.net[idx].parameters():
                param.requires_grad = True
                finetune_params.append(param)
        print(f"[RL Finetune] 解冻层索引: {layers_to_unfreeze}")
    else:
        # 如果Linear层数不够，解冻所有
        for param in flow_model.model.net.parameters():
            param.requires_grad = True
            finetune_params.append(param)
        print(f"[RL Finetune] 解冻所有net层")
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in flow_model.model.parameters())
    trainable_params = sum(p.numel() for p in finetune_params)
    print(f"[RL Finetune] 总参数: {total_params}, 可训练参数: {trainable_params} ({100*trainable_params/total_params:.2f}%)")
    
    return finetune_params


class RolloutBuffer:
    """
    存储一个episode的轨迹数据
    """
    def __init__(self):
        self.states_x = []
        self.states_z = []
        self.states_t = []
        self.actions = []  # 向量场v
        self.values = []
        self.rewards = []
        self.advantages = []
        self.returns = []
    
    def add(self, x, z, t, v, value):
        self.states_x.append(x)
        self.states_z.append(z)
        self.states_t.append(t)
        self.actions.append(v)
        self.values.append(value)
    
    def compute_returns(self, final_reward, gamma=0.99, gae_lambda=0.95):
        """计算优势和回报"""
        # 稀疏奖励：只有最后一步有奖励
        self.rewards = [0.0] * (len(self.values) - 1) + [final_reward]
        
        # 提取values为列表
        values_list = [v.item() if isinstance(v, torch.Tensor) else v for v in self.values]
        
        # 计算GAE
        self.advantages, self.returns = compute_gae(
            values_list, self.rewards, gamma, gae_lambda
        )
    
    def get_batch(self, device):
        """获取批量数据用于PPO更新"""
        x = torch.cat(self.states_x, dim=0).to(device)
        z = torch.cat(self.states_z, dim=0).to(device)
        t = torch.cat(self.states_t, dim=0).to(device)
        v = torch.cat(self.actions, dim=0).to(device)
        
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=device)
        
        return x, z, t, v, advantages, returns
    
    def clear(self):
        self.__init__()


def ppo_update(flow_model, critic, optimizer, buffer, config, device):
    """
    执行PPO更新
    
    由于直接微调流模型，使用简化的策略梯度：
    - 正优势：保持/强化当前动作方向
    - 负优势：削弱当前动作方向
    """
    x, z, t, v_old, advantages, returns = buffer.get_batch(device)
    
    # 标准化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_actor_loss = 0
    total_critic_loss = 0
    
    for _ in range(config['ppo_epochs']):
        # 分批次处理（如果数据太大）
        batch_size = len(advantages)
        
        for start in range(0, batch_size, config.get('mini_batch_size', batch_size)):
            end = min(start + config.get('mini_batch_size', batch_size), batch_size)
            
            x_batch = x[start:end]
            z_batch = z[start:end]
            t_batch = t[start:end]
            adv_batch = advantages[start:end]
            ret_batch = returns[start:end]
            
            # 重新计算向量场
            v_new = flow_model.model(x_batch, z_batch, t_batch)
            
            # 重新计算价值
            value_new = critic(x_batch, z_batch, t_batch).squeeze()
            
            # Actor损失：策略梯度
            # 简化形式：让向量场在优势为正时保持，优势为负时缩小
            # 使用L2正则化形式
            actor_loss = -(adv_batch.unsqueeze(1) * v_new).mean()
            
            # Critic损失
            critic_loss = F.mse_loss(value_new, ret_batch)
            
            # 总损失
            loss = actor_loss + config['value_coef'] * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(filter(lambda p: p.requires_grad, flow_model.model.parameters())) + 
                list(critic.parameters()),
                config['max_grad_norm']
            )
            
            optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
    
    return total_actor_loss / config['ppo_epochs'], total_critic_loss / config['ppo_epochs']

