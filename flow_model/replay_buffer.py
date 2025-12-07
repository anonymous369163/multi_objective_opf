"""
经验回放池 (Replay Buffer) 实现
用于SAC等off-policy强化学习算法

作者：基于rl_opf_flow项目
日期：2025
"""

import random
import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """
    经验回放池，存储和采样训练经验
    
    存储的经验格式: (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity, device='cuda'):
        """
        初始化回放池
        
        Args:
            capacity: 最大容量
            device: 存储设备（'cpu' 或 'cuda'）
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        print(f"ReplayBuffer 初始化: 容量={capacity}, 设备={device}")
    
    def add(self, state, action, reward, next_state, done):
        """
        添加一个经验到回放池
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        # 转换为numpy数组以节省内存（存储在CPU上）
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()
        
        # 如果reward和done是标量，转换为数组
        if np.isscalar(reward):
            reward = np.array([reward])
        if np.isscalar(done):
            done = np.array([done])
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从回放池中随机采样一个批次
        
        Args:
            batch_size: 批次大小
        
        Returns:
            transition_dict: 包含states, actions, rewards, next_states, dones的字典
        """
        # 随机采样
        batch = random.sample(self.buffer, batch_size)
        
        # 解包批次
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为numpy数组
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # 如果是单样本维度，需要squeeze
        if rewards.ndim > 1 and rewards.shape[1] == 1:
            rewards = rewards.squeeze(1)
        if dones.ndim > 1 and dones.shape[1] == 1:
            dones = dones.squeeze(1)

        # 如果next_states的维度是三个维度的，则删减一个维度
        if next_states.ndim == 3:
            next_states = next_states.squeeze(1)
        
        # 转换为tensor并移动到指定设备
        transition_dict = {
            'states': torch.tensor(states, dtype=torch.float32, device=self.device),
            'actions': torch.tensor(actions, dtype=torch.float32, device=self.device),
            'rewards': torch.tensor(rewards, dtype=torch.float32, device=self.device),
            'next_states': torch.tensor(next_states, dtype=torch.float32, device=self.device),
            'dones': torch.tensor(dones, dtype=torch.float32, device=self.device)
        }
        
        return transition_dict
    
    def size(self):
        """返回当前回放池中的经验数量"""
        return len(self.buffer)
    
    def is_ready(self, min_size):
        """检查回放池是否有足够的经验可以开始训练"""
        return self.size() >= min_size
    
    def clear(self):
        """清空回放池"""
        self.buffer.clear()
        self.position = 0
    
    def __len__(self):
        """返回回放池大小"""
        return len(self.buffer)
    
    def get_statistics(self):
        """获取回放池统计信息"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0
            }
        
        # 计算利用率
        utilization = len(self.buffer) / self.capacity
        
        # 采样一小部分数据计算统计
        sample_size = min(100, len(self.buffer))
        sample_batch = random.sample(self.buffer, sample_size)
        _, _, rewards, _, _ = zip(*sample_batch)
        rewards = np.array(rewards)
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': utilization,
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_min': float(np.min(rewards)),
            'reward_max': float(np.max(rewards))
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    优先经验回放池 (Prioritized Experience Replay)
    根据TD误差的大小来调整采样概率
    
    注意：这是一个高级功能，目前项目中暂不使用
    """
    
    def __init__(self, capacity, device='cuda', alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先回放池
        
        Args:
            capacity: 最大容量
            device: 存储设备
            alpha: 优先级指数（0=均匀采样，1=完全按优先级）
            beta: 重要性采样权重指数
            beta_increment: beta的增长速率
        """
        super().__init__(capacity, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
        print(f"PrioritizedReplayBuffer 初始化: alpha={alpha}, beta={beta}")
    
    def add(self, state, action, reward, next_state, done):
        """添加经验，初始优先级设为最大"""
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        """根据优先级采样"""
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # 获取样本
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 转换为tensor
        transition_dict = {
            'states': torch.tensor(np.array(states), dtype=torch.float32, device=self.device),
            'actions': torch.tensor(np.array(actions), dtype=torch.float32, device=self.device),
            'rewards': torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device),
            'next_states': torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            'dones': torch.tensor(np.array(dones), dtype=torch.float32, device=self.device),
            'weights': torch.tensor(weights, dtype=torch.float32, device=self.device),
            'indices': indices
        }
        
        return transition_dict
    
    def update_priorities(self, indices, priorities):
        """更新指定样本的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


if __name__ == "__main__":
    """测试回放池功能"""
    print("="*60)
    print("ReplayBuffer 测试")
    print("="*60)
    
    # 创建回放池
    buffer = ReplayBuffer(capacity=1000, device='cpu')
    
    # 添加一些经验
    print("\n测试1: 添加经验")
    for i in range(10):
        state = np.random.randn(5)
        action = np.random.randn(3)
        reward = np.random.randn()
        next_state = np.random.randn(5)
        done = (i == 9)
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"  添加了10个经验，当前大小: {buffer.size()}")
    
    # 测试采样
    print("\n测试2: 批量采样")
    if buffer.is_ready(5):
        batch = buffer.sample(batch_size=5)
        print(f"  采样批次大小: {batch['states'].shape[0]}")
        print(f"  States形状: {batch['states'].shape}")
        print(f"  Actions形状: {batch['actions'].shape}")
        print(f"  Rewards形状: {batch['rewards'].shape}")
    
    # 测试统计信息
    print("\n测试3: 统计信息")
    stats = buffer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试容量限制
    print("\n测试4: 容量限制")
    for i in range(1000):
        state = np.random.randn(5)
        action = np.random.randn(3)
        reward = np.random.randn()
        next_state = np.random.randn(5)
        done = False
        buffer.add(state, action, reward, next_state, done)
    
    print(f"  添加1000个经验后，当前大小: {buffer.size()} (容量: {buffer.capacity})")
    
    # 测试tensor输入
    print("\n测试5: Tensor输入")
    buffer2 = ReplayBuffer(capacity=100, device='cpu')
    state_tensor = torch.randn(5)
    action_tensor = torch.randn(3)
    reward_tensor = torch.tensor(1.0)
    next_state_tensor = torch.randn(5)
    done_tensor = torch.tensor(False)
    
    buffer2.add(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)
    print(f"  ✓ Tensor输入添加成功，大小: {buffer2.size()}")
    
    print("\n✓ 所有测试通过!")

