"""
Reflow 训练工具 - 带 Jacobian 后处理的自蒸馏

核心思想（参考 reflow_idea.md）：
1. 第一阶段：标准 Rectified Flow 训练，得到 v_θ^(0)
2. 第二阶段：用 v_θ^(0) + Jacobian 后处理 生成矫正轨迹
3. 第三阶段：用矫正轨迹训练 Student 模型 v_θ^(1)

推理时只用 v_θ^(1)，不需要 Jacobian 后处理！

这是一个知识蒸馏框架：
Teacher = Flow Model + Jacobian Correction
Student = 纯 Flow Model（学会了 Teacher 的能力）
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

# 导入 Jacobian 计算函数
from post_processing import compute_drift_correction_batch, apply_drift_correction


class ReflowTrajectoryGenerator:
    """
    Reflow 轨迹生成器
    
    用 Teacher 模型（v_θ^(0) + Jacobian 修正）生成矫正后的轨迹，
    用于训练 Student 模型。
    
    Args:
        teacher_model: 第一阶段训练好的 Rectified Flow 模型
        env: 电网环境对象，用于计算 Jacobian
        device: 计算设备
        correction_interval: 每隔多少步进行一次 Jacobian 修正
        lambda_cor: 法向修正增益
        save_interval: 每隔多少步保存一个轨迹点
    """
    
    def __init__(
        self,
        teacher_model,
        env,
        device: str = 'cuda',
        correction_interval: int = 5,
        lambda_cor: float = 1.5,
        save_interval: int = 10,
        start_correction_t: float = 0.3,
    ):
        self.teacher_model = teacher_model
        self.env = env
        self.device = device
        self.correction_interval = correction_interval
        self.lambda_cor = lambda_cor
        self.save_interval = save_interval
        self.start_correction_t = start_correction_t
        
        # 确保模型在评估模式
        self.teacher_model.eval()
    
    def generate_trajectory(
        self,
        x_batch: torch.Tensor,
        z_start: torch.Tensor,
        y_target: torch.Tensor,
        step: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        生成单个批次的矫正轨迹（改进版 v2）
        
        核心改进：
        1. Teacher+Jacobian 先走完整个 ODE 轨迹，得到最终点 y_final
        2. 目标速度 = (y_final - z_t) / (1 - t)，指向最终目标
        3. 这样 Student 学习的是"从任意点指向修正后终点的速度"
        4. 即使 Student 推理时漂移了，它仍然知道如何到达目标
        
        Args:
            x_batch: 条件输入 (batch_size, input_dim)
            z_start: 起始点/锚点 (batch_size, output_dim)
            y_target: Ground Truth 目标 (batch_size, output_dim)
            step: ODE 步长
        
        Returns:
            trajectory_data: 包含轨迹点和目标速度的字典
        """
        batch_size = x_batch.shape[0]
        output_dim = z_start.shape[1]
        
        # =========== 阶段 1：Teacher+Jacobian 走完轨迹，得到 y_final ===========
        z = z_start.clone()
        t = 0.0
        num_steps = int(1.0 / step)
        step_count = 0
        jacobian_count = 0
        
        # 保存轨迹点和时间
        trajectory_points = []
        trajectory_times = []
        
        with torch.no_grad():
            for step_idx in range(num_steps):
                t_tensor = torch.ones(batch_size, 1, device=self.device) * t
                
                # 保存轨迹点（按 save_interval 间隔）
                if step_idx % self.save_interval == 0:
                    trajectory_points.append(z.clone())
                    trajectory_times.append(t)
                
                # 计算模型预测的速度
                v_pred = self.teacher_model.model(x_batch, z, t_tensor)
                
                # 是否应用 Jacobian 修正
                apply_correction = (
                    t >= self.start_correction_t and 
                    step_count % self.correction_interval == 0
                )
                
                if apply_correction:
                    P_tan, correction = compute_drift_correction_batch(
                        z, x_batch, self.env, self.lambda_cor
                    )
                    v_corrected = apply_drift_correction(v_pred, P_tan, correction)
                    jacobian_count += 1
                else:
                    v_corrected = v_pred
                
                # 更新状态
                z = z + v_corrected * step
                t += step
                step_count += 1
        
        # 最终状态：Teacher+Jacobian 到达的终点
        y_final = z.clone()
        
        # =========== 阶段 2：计算指向 y_final 的目标速度（带扰动增强）===========
        z_list = []
        t_list = []
        v_list = []
        
        # 添加扰动来模拟推理时的漂移
        perturbation_scale = 0.05  # 扰动幅度
        num_perturbations = 3      # 每个轨迹点生成的扰动版本数
        
        for z_t, t_val in zip(trajectory_points, trajectory_times):
            t_tensor = torch.ones(batch_size, 1, device=self.device) * t_val
            
            # 1. 原始轨迹点
            if t_val < 0.99:
                v_target = (y_final - z_t) / (1.0 - t_val)
            else:
                v_target = torch.zeros_like(z_t)
            
            z_list.append(z_t)
            t_list.append(t_tensor)
            v_list.append(v_target)
            
            # 2. 添加扰动版本（模拟漂移）
            if t_val > 0.1 and t_val < 0.9:  # 只在中间时间段添加扰动
                for _ in range(num_perturbations):
                    # 生成扰动
                    noise = torch.randn_like(z_t) * perturbation_scale
                    z_perturbed = z_t + noise
                    
                    # 扰动点指向同一个终点
                    v_perturbed = (y_final - z_perturbed) / (1.0 - t_val)
                    
                    z_list.append(z_perturbed)
                    t_list.append(t_tensor.clone())
                    v_list.append(v_perturbed)
        
        # 保存最终状态（终点速度为0）
        t_final = torch.ones(batch_size, 1, device=self.device)
        z_list.append(y_final)
        t_list.append(t_final)
        v_list.append(torch.zeros_like(y_final))
        
        # 合并数据
        trajectory_data = {
            'z': torch.cat(z_list, dim=0),       # (N, output_dim) - 轨迹点
            't': torch.cat(t_list, dim=0),       # (N, 1) - 时间步
            'v': torch.cat(v_list, dim=0),       # (N, output_dim) - 指向终点的速度
            'x': x_batch.repeat(len(z_list), 1), # (N, input_dim) - 条件
            'y_final': y_final,                   # (batch_size, output_dim) - 修正后终点
            'y_target': y_target,                 # (batch_size, output_dim) - GT
            '_jacobian_count': jacobian_count,
        }
        
        return trajectory_data
    
    def generate_dataset(
        self,
        dataloader,
        anchor_generator,
        num_epochs: int = 1,
        step: float = 0.01,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        从数据加载器生成完整的 Reflow 训练数据集（改进版）
        
        Args:
            dataloader: 包含 (x_input, y_target) 的数据加载器
            anchor_generator: 生成锚点的模型（如 VAE）
            num_epochs: 遍历数据集的次数
            step: ODE 步长
            save_path: 保存路径
            verbose: 是否显示进度
        
        Returns:
            dataset: 合并后的完整数据集
        """
        import time
        
        all_data = {
            'z': [],
            't': [],
            'v': [],
            'x': [],
        }
        
        # 额外统计
        y_finals = []
        y_targets = []
        
        anchor_generator.eval()
        
        # 计算预估时间
        num_steps = int(1.0 / step)
        start_step = int(self.start_correction_t / step)
        num_jacobian_per_batch = (num_steps - start_step) // self.correction_interval
        total_batches = len(dataloader) * num_epochs
        
        if verbose:
            print(f"\n  Trajectory generation config:")
            print(f"    - ODE steps: {num_steps}")
            print(f"    - Jacobian calls per batch: {num_jacobian_per_batch}")
            print(f"    - Total batches: {total_batches}")
            print(f"    - Estimated Jacobian calls: {num_jacobian_per_batch * total_batches}")
        
        total_jacobian_calls = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            desc = f"Generating trajectories"
            iterator = tqdm(dataloader, desc=desc) if verbose else dataloader
            
            for batch in iterator:
                # 获取 x_input 和 y_target
                if isinstance(batch, (list, tuple)):
                    x_input = batch[0].to(self.device)
                    if len(batch) > 1:
                        y_target = batch[1].to(self.device)
                    else:
                        # 获取 output_dim
                        output_dim = getattr(self.teacher_model, 'output_dim', x_input.shape[1])
                        y_target = torch.zeros(x_input.shape[0], output_dim, device=self.device)
                else:
                    x_input = batch.to(self.device)
                    output_dim = getattr(self.teacher_model, 'output_dim', x_input.shape[1])
                    y_target = torch.zeros(x_input.shape[0], output_dim, device=self.device)
                
                # 生成锚点
                with torch.no_grad():
                    z_start = anchor_generator(x_input, use_mean=True)
                
                # 生成这批数据的轨迹
                batch_data = self.generate_trajectory(x_input, z_start, y_target, step)
                
                # 累积 Jacobian 统计
                total_jacobian_calls += batch_data.get('_jacobian_count', 0)
                
                # 累积数据（排除内部统计字段）
                for key in all_data:
                    if key in batch_data:
                        all_data[key].append(batch_data[key].cpu())
                
                # 保存终点信息
                if 'y_final' in batch_data:
                    y_finals.append(batch_data['y_final'].cpu())
                if 'y_target' in batch_data:
                    y_targets.append(batch_data['y_target'].cpu())
        
        elapsed = time.time() - start_time
        
        # 合并所有数据
        dataset = {
            key: torch.cat(tensors, dim=0) 
            for key, tensors in all_data.items()
        }
        
        # 保存终点统计
        if y_finals:
            dataset['_y_final'] = torch.cat(y_finals, dim=0)
        if y_targets:
            dataset['_y_target'] = torch.cat(y_targets, dim=0)
        
        if verbose:
            print(f"\n  Generation completed in {elapsed:.1f}s")
            print(f"  Total Jacobian calls: {total_jacobian_calls}")
            
            # 分析 Teacher+Jacobian 的效果
            if y_finals and y_targets:
                y_final_cat = dataset['_y_final']
                y_target_cat = dataset['_y_target']
                mse_to_gt = ((y_final_cat - y_target_cat) ** 2).mean().item()
                print(f"  Teacher+Jacobian MSE to GT: {mse_to_gt:.6f}")
        
        if verbose:
            print(f"\nGenerated {dataset['z'].shape[0]} trajectory points")
            print(f"  - State dim: {dataset['z'].shape[1]}")
            print(f"  - Velocity range: [{dataset['v'].min():.4f}, {dataset['v'].max():.4f}]")
        
        # 保存数据集
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(dataset, save_path)
            if verbose:
                print(f"Dataset saved to {save_path}")
        
        return dataset


class ReflowDataset(torch.utils.data.Dataset):
    """
    Reflow 训练数据集
    
    存储矫正后的轨迹数据：(z, x, t, v_target)
    """
    
    def __init__(
        self, 
        data_dict: Optional[Dict] = None, 
        data_path: Optional[str] = None
    ):
        if data_dict is not None:
            self.data = data_dict
        elif data_path is not None:
            self.data = torch.load(data_path)
        else:
            raise ValueError("Must provide either data_dict or data_path")
        
        self.length = self.data['z'].shape[0]
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.data['z'][idx],
            self.data['x'][idx],
            self.data['t'][idx],
            self.data['v'][idx],
        )


def create_reflow_dataloader(
    dataset: ReflowDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """创建 Reflow 数据加载器"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


class ReflowTrainer:
    """
    Reflow 训练器
    
    使用混合损失训练 Student 模型：
    Loss = Loss_FM_linear + λ * Loss_FM_corrected
    
    Args:
        student_model: Student 流模型
        original_data: 原始训练数据 (x, y)，用于线性插值损失
        reflow_data: Reflow 轨迹数据，用于矫正轨迹损失
        lambda_reflow: Reflow 损失权重
        device: 计算设备
    """
    
    def __init__(
        self,
        student_model,
        anchor_generator,
        reflow_dataset: ReflowDataset,
        lambda_reflow: float = 1.0,
        device: str = 'cuda',
    ):
        self.student_model = student_model
        self.anchor_generator = anchor_generator
        self.reflow_dataset = reflow_dataset
        self.lambda_reflow = lambda_reflow
        self.device = device
        
        self.criterion = nn.L1Loss()
    
    def compute_linear_loss(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        z_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算原始线性插值的 Flow Matching 损失
        （保持学习整体分布的能力）
        """
        # 线性插值
        yt = t_batch * y_batch + (1 - t_batch) * z_batch
        vec_target = y_batch - z_batch
        
        # 预测速度
        vec_pred = self.student_model.model(x_batch, yt, t_batch)
        
        return self.criterion(vec_pred, vec_target)
    
    def compute_reflow_loss(
        self,
        z_batch: torch.Tensor,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
        v_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Reflow 轨迹的损失
        （学习矫正能力）
        """
        # 预测速度
        v_pred = self.student_model.model(x_batch, z_batch, t_batch)
        
        return self.criterion(v_pred, v_target)
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reflow_batch: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        单步训练
        
        Returns:
            loss: 总损失
            loss_dict: 各分项损失
        """
        batch_size = x_batch.shape[0]
        
        # 生成锚点
        with torch.no_grad():
            z_batch = self.anchor_generator(x_batch, use_mean=True)
        
        # 随机时间步
        t_batch = torch.rand(batch_size, 1, device=self.device)
        
        # 1. 线性插值损失
        loss_linear = self.compute_linear_loss(x_batch, y_batch, z_batch, t_batch)
        
        # 2. Reflow 轨迹损失
        z_reflow, x_reflow, t_reflow, v_reflow = reflow_batch
        z_reflow = z_reflow.to(self.device)
        x_reflow = x_reflow.to(self.device)
        t_reflow = t_reflow.to(self.device)
        v_reflow = v_reflow.to(self.device)
        
        loss_reflow = self.compute_reflow_loss(z_reflow, x_reflow, t_reflow, v_reflow)
        
        # 混合损失
        loss = loss_linear + self.lambda_reflow * loss_reflow
        
        loss_dict = {
            'linear': loss_linear.item(),
            'reflow': loss_reflow.item(),
            'total': loss.item(),
        }
        
        return loss, loss_dict


# ============================================================================
# 配置默认值
# ============================================================================

def get_default_reflow_config() -> Dict:
    """获取 Reflow 训练的默认配置"""
    return {
        'enabled': True,
        
        # === 轨迹生成参数 ===
        'correction_interval': 5,     # 每 5 步进行一次 Jacobian 修正
        'lambda_cor': 1.5,            # 法向修正增益
        'save_interval': 10,          # 每 10 步保存一个轨迹点
        'start_correction_t': 0.3,    # 从 t=0.3 开始应用修正
        'trajectory_epochs': 1,       # 生成轨迹时遍历数据集的次数
        
        # === 训练参数 ===
        'lambda_reflow': 1.0,         # Reflow 损失权重
        'num_epochs': 5000,           # Student 训练轮数
        'learning_rate': 1e-3,
        'weight_decay': 1e-6,
        
        # === 数据路径 ===
        'trajectory_data_path': 'data/reflow_trajectories.pt',
        'regenerate_trajectories': False,  # 是否重新生成轨迹
    }

