"""
Riemannian Flow Matching (RFM) 训练工具 - 增强版

核心改进：
从"中性稳定"变为"渐进稳定"系统。

训练目标：
v_label = P_tan @ (y_1 - y_0) - Clip(λ * F^+ @ f(y_t), -C, C)
         ├── 切向：去终点 ──┘  └── 法向：回流形（带Clip）──┘

关键特性：
1. 切向投影：保证模型"贴着流形走"
2. 法向修正：让误差指数衰减，主动拉回流形
3. 扰动训练：在插值点加噪声，让模型学会修正偏离
4. Clip截断：防止法向修正数值爆炸

参考：Baumgarte Stabilization / Feedback Linearization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple

# 导入 Jacobian 计算函数
from post_processing import compute_drift_correction_batch


class RFMTrainingHelper:
    """
    Riemannian Flow Matching 训练辅助类（增强版）
    
    管理切空间投影和法向修正的计算，支持：
    - 冻结雅可比策略减少计算开销
    - 扰动训练模拟漂移
    - Clip截断防止数值爆炸
    
    Args:
        sys_data: PowerSystemData 对象，用于计算 Jacobian (替代 env)
        freeze_interval: 冻结间隔，每隔多少步更新一次 P_tan
        soft_weight: 软约束权重，0=纯直连线，1=纯投影+修正
        lambda_cor: 法向修正增益（建议 5.0-10.0）
        add_perturbation: 是否在训练时添加扰动
        perturbation_scale: 扰动幅度（相对于状态范围）
        max_correction_norm: 法向修正的最大范数（Clip上限）
        device: 计算设备
    """
    
    def __init__(
        self, 
        sys_data, 
        freeze_interval: int = 10, 
        soft_weight: float = 1.0,
        lambda_cor: float = 5.0,
        add_perturbation: bool = True,
        perturbation_scale: float = 0.05,
        max_correction_norm: float = 10.0,
        device: str = 'cuda'
    ):
        self.sys_data = sys_data
        self.freeze_interval = freeze_interval
        self.soft_weight = soft_weight
        self.lambda_cor = lambda_cor
        self.add_perturbation = add_perturbation
        self.perturbation_scale = perturbation_scale
        self.max_correction_norm = max_correction_norm
        self.device = device
        
        # 缓存
        self._P_tan_cache: Optional[torch.Tensor] = None
        self._correction_cache: Optional[torch.Tensor] = None
        self._cache_step: int = -1
        self._cache_batch_size: int = 0
        
        # 统计信息
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'jacobian_computations': 0,
            'clip_events': 0,  # 被 Clip 截断的次数
        }
    
    def get_P_tan_and_correction(
        self, 
        yt: torch.Tensor, 
        x: torch.Tensor, 
        step_idx: int,
        force_compute: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取切空间投影矩阵 P_tan 和法向修正向量 correction
        
        使用冻结策略：每 freeze_interval 步更新一次，
        中间步骤复用缓存。
        
        Args:
            yt: 当前状态 (batch_size, output_dim)
            x: 条件输入 (batch_size, input_dim)
            step_idx: 当前训练步数（用于决定是否更新）
            force_compute: 强制重新计算（忽略缓存）
        
        Returns:
            P_tan: 切空间投影矩阵 (batch_size, output_dim, output_dim)
            correction: 法向修正向量 (batch_size, output_dim)
        """
        self.stats['total_calls'] += 1
        batch_size = yt.shape[0]
        
        # 判断是否需要重新计算
        should_compute = (
            force_compute or
            self._P_tan_cache is None or
            step_idx % self.freeze_interval == 0 or
            batch_size != self._cache_batch_size
        )
        
        if should_compute:
            # 计算 P_tan 和 correction
            P_tan, correction = compute_drift_correction_batch(
                yt, x, self.sys_data, lambda_cor=self.lambda_cor
            )
            
            # 更新缓存
            self._P_tan_cache = P_tan.detach()
            self._correction_cache = correction.detach()
            self._cache_step = step_idx
            self._cache_batch_size = batch_size
            self.stats['jacobian_computations'] += 1
        else:
            # 使用缓存
            P_tan = self._P_tan_cache
            correction = self._correction_cache
            self.stats['cache_hits'] += 1
        
        return P_tan, correction
    
    def clip_correction(self, correction: torch.Tensor) -> torch.Tensor:
        """
        对法向修正向量进行 Clip 截断，防止数值爆炸
        
        使用向量范数 Clip：如果 ||correction|| > max_norm，
        则缩放为 correction * (max_norm / ||correction||)
        
        Args:
            correction: 原始法向修正向量 (batch_size, output_dim)
        
        Returns:
            clipped_correction: 截断后的修正向量
        """
        # 计算每个样本的范数
        norms = correction.norm(dim=1, keepdim=True)  # (B, 1)
        
        # 找出需要截断的样本
        clip_mask = norms > self.max_correction_norm
        
        if clip_mask.any():
            # 计算缩放因子
            scale = torch.where(
                clip_mask,
                self.max_correction_norm / (norms + 1e-8),
                torch.ones_like(norms)
            )
            correction = correction * scale
            self.stats['clip_events'] += clip_mask.sum().item()
        
        return correction
    
    def compute_rfm_target(
        self,
        y_batch: torch.Tensor,
        z_batch: torch.Tensor,
        t_batch: torch.Tensor,
        x_batch: torch.Tensor,
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 RFM 训练所需的插值点和目标向量（增强版）
        
        核心公式：
        v_target = P_tan @ (y - z) + Clip(correction, -C, C)
                 = 切向（去终点）  +  法向（回流形）
        
        Args:
            y_batch: 目标解 (batch_size, output_dim)
            z_batch: 起始点/锚点 (batch_size, output_dim)
            t_batch: 时间步 (batch_size, 1)
            x_batch: 条件输入 (batch_size, input_dim)
            step_idx: 当前训练步数
        
        Returns:
            yt: 插值点（可能加了扰动）(batch_size, output_dim)
            vec_target: 组合目标向量 (batch_size, output_dim)
            P_tan: 切空间投影矩阵 (batch_size, output_dim, output_dim)
        """
        # 1. 计算插值点 yt = t * y + (1-t) * z
        yt = t_batch * y_batch + (1 - t_batch) * z_batch
        
        # 2. 添加扰动模拟漂移（让模型学会修正偏离的状态）
        if self.add_perturbation and self.training_mode:
            noise = torch.randn_like(yt) * self.perturbation_scale
            yt_perturbed = yt + noise
        else:
            yt_perturbed = yt
        
        # 3. 获取 P_tan 和 correction（基于扰动后的位置）
        P_tan, correction = self.get_P_tan_and_correction(
            yt_perturbed, x_batch, step_idx
        )
        
        # 4. Clip correction 防止数值爆炸
        correction_clipped = self.clip_correction(correction)
        
        # 5. 计算切向分量：P_tan @ (y - z)
        raw_vec = y_batch - z_batch
        raw_vec_expanded = raw_vec.unsqueeze(-1)  # (B, D, 1)
        vec_tangent = torch.bmm(P_tan, raw_vec_expanded).squeeze(-1)  # (B, D)
        
        # 6. 组合目标向量
        # v_target = 切向（去终点）+ 法向（回流形）
        if self.soft_weight < 1.0:
            # 软约束：混合原始目标和投影+修正
            vec_target = (1 - self.soft_weight) * raw_vec + \
                         self.soft_weight * (vec_tangent + correction_clipped)
        else:
            vec_target = vec_tangent + correction_clipped
        
        return yt_perturbed, vec_target, P_tan
    
    @property
    def training_mode(self) -> bool:
        """检查是否在训练模式（用于控制扰动）"""
        return True  # 默认为训练模式
    
    def reset_cache(self):
        """重置缓存"""
        self._P_tan_cache = None
        self._correction_cache = None
        self._cache_step = -1
        self._cache_batch_size = 0
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calls']
        else:
            stats['cache_hit_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'jacobian_computations': 0,
            'clip_events': 0,
        }


def compute_P_tan_batch(z: torch.Tensor, x_input: torch.Tensor, sys_data) -> torch.Tensor:
    """
    批量计算切空间投影矩阵 P_tan
    
    这是一个便捷函数，直接返回 P_tan 而不需要 correction。
    
    Args:
        z: 当前状态 (batch_size, output_dim)
        x_input: 条件输入 (batch_size, input_dim)
        sys_data: PowerSystemData 对象 (替代 env)
    
    Returns:
        P_tan: 切空间投影矩阵 (batch_size, output_dim, output_dim)
    """
    P_tan, _ = compute_drift_correction_batch(z, x_input, sys_data, lambda_cor=0)
    return P_tan


class RFMLoss(nn.Module):
    """
    Riemannian Flow Matching 损失函数
    
    支持多种损失组件：
    - MSE 损失：预测速度与目标的均方误差
    - 方向损失：余弦相似度损失，确保方向正确
    - 法向惩罚：惩罚速度在法向空间的分量（可选）
    
    Args:
        mse_weight: MSE 损失权重
        direction_weight: 方向损失权重
        normal_weight: 法向惩罚权重（需要提供 P_tan）
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        direction_weight: float = 0.0,
        normal_weight: float = 0.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.normal_weight = normal_weight
    
    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        P_tan: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 RFM 损失
        
        Args:
            v_pred: 预测速度 (batch_size, output_dim)
            v_target: 目标速度 (batch_size, output_dim)
            P_tan: 切空间投影矩阵（用于法向惩罚）
        
        Returns:
            loss: 总损失
            loss_dict: 各分项损失的字典
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. MSE 损失
        mse_loss = torch.nn.functional.mse_loss(v_pred, v_target)
        loss_dict['mse'] = mse_loss.item()
        total_loss = total_loss + self.mse_weight * mse_loss
        
        # 2. 方向损失（余弦相似度）
        if self.direction_weight > 0:
            target_norm = v_target.norm(dim=1)
            valid_mask = target_norm > 1e-6
            
            if valid_mask.sum() > 0:
                cos_sim = torch.nn.functional.cosine_similarity(
                    v_pred[valid_mask], v_target[valid_mask], dim=1
                )
                dir_loss = (1 - cos_sim).mean()
            else:
                dir_loss = torch.tensor(0.0, device=v_pred.device)
            
            loss_dict['direction'] = dir_loss.item()
            total_loss = total_loss + self.direction_weight * dir_loss
        
        # 3. 法向惩罚
        if self.normal_weight > 0 and P_tan is not None:
            v_pred_expanded = v_pred.unsqueeze(-1)
            v_tangent = torch.bmm(P_tan, v_pred_expanded).squeeze(-1)
            v_normal = v_pred - v_tangent
            
            normal_loss = v_normal.pow(2).mean()
            loss_dict['normal'] = normal_loss.item()
            total_loss = total_loss + self.normal_weight * normal_loss
        
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict


# ============================================================================
# 配置默认值
# ============================================================================

def get_default_rfm_config() -> Dict:
    """获取 RFM 训练的默认配置（增强版）"""
    return {
        'enabled': True,              # 是否启用 RFM 训练
        'freeze_interval': 10,        # 每 10 步更新一次 P_tan
        'soft_weight': 1.0,           # 软约束权重（1.0=纯投影+修正）
        
        # === 新增：法向修正参数 ===
        'lambda_cor': 5.0,            # 法向修正增益（建议 5.0-10.0）
        'add_perturbation': True,     # 训练时添加扰动模拟漂移
        'perturbation_scale': 0.05,   # 扰动幅度
        'max_correction_norm': 10.0,  # Clip 上限，防止数值爆炸
        
        # === 损失函数参数 ===
        'mse_weight': 1.0,            # MSE 损失权重
        'direction_weight': 0.0,      # 方向损失权重（可选）
        'normal_weight': 0.0,         # 法向惩罚权重（可选）
    }
