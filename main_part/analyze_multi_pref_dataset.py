#!/usr/bin/env python
# coding: utf-8
"""
分析多目标数据集的大小
用于训练流模型的多偏好数据集统计分析

Author: Auto-generated
Date: 2025
"""

import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from data_loader import load_multi_preference_dataset


def analyze_dataset_size():
    """
    分析多目标数据集的大小
    
    显示：
    - 总样本数
    - 训练集样本数
    - 验证集（测试集）样本数
    - 偏好数量
    - 每个偏好的数据量
    - 数据维度信息
    """
    print("=" * 80)
    print("多目标数据集大小分析")
    print("=" * 80)
    
    # 加载配置
    config = get_config()
    
    # 加载多偏好数据集
    print("\n正在加载数据集...")
    try:
        multi_pref_data, sys_data = load_multi_preference_dataset(config)
    except Exception as e:
        print(f"\n错误：无法加载数据集: {e}")
        print("\n请确保已运行 build_fully_covered_dataset.py 生成数据集")
        return
    
    print("\n" + "=" * 80)
    print("数据集大小统计")
    print("=" * 80)
    
    # 提取基本信息
    n_samples = multi_pref_data['n_samples']  # 总样本数
    n_train = multi_pref_data['n_train']      # 训练集样本数
    n_val = multi_pref_data['n_val']          # 验证集（测试集）样本数
    n_preferences = multi_pref_data['n_preferences']  # 偏好数量
    
    # 数据维度
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    
    # 偏好值列表
    lambda_carbon_values = multi_pref_data['lambda_carbon_values']
    
    # 显示基本信息
    print(f"\n【总体信息】")
    print(f"  总样本数 (n_samples): {n_samples:,}")
    print(f"  偏好数量 (n_preferences): {n_preferences}")
    print(f"  输入维度 (input_dim): {input_dim}")
    print(f"  输出维度 (output_dim): {output_dim}")
    
    # 显示训练/验证集分割
    print(f"\n【训练/验证集分割】")
    print(f"  训练集样本数 (n_train): {n_train:,} ({n_train/n_samples*100:.2f}%)")
    print(f"  验证集样本数 (n_val): {n_val:,} ({n_val/n_samples*100:.2f}%)")
    
    # 检查每个偏好的数据
    y_train_by_pref = multi_pref_data['y_train_by_pref']
    y_val_by_pref = multi_pref_data.get('y_val_by_pref', {})
    
    print(f"\n【各偏好数据详情】")
    print(f"  训练集偏好数: {len(y_train_by_pref)}")
    print(f"  验证集偏好数: {len(y_val_by_pref)}")
    
    # 显示每个偏好的数据形状
    if len(lambda_carbon_values) > 0:
        print(f"\n  前5个偏好的数据形状示例:")
        for i, lc in enumerate(lambda_carbon_values[:5]):
            if lc in y_train_by_pref:
                train_shape = y_train_by_pref[lc].shape
                val_shape = y_val_by_pref.get(lc, torch.empty(0)).shape if lc in y_val_by_pref else (0,)
                print(f"    λ_carbon = {lc:6.2f}: 训练集 {train_shape}, 验证集 {val_shape}")
        
        if len(lambda_carbon_values) > 5:
            print(f"    ... (共 {len(lambda_carbon_values)} 个偏好)")
    
    # 计算总数据量（考虑所有偏好）
    print(f"\n【总数据量统计（考虑所有偏好）】")
    total_train_samples = n_train * n_preferences
    total_val_samples = n_val * n_preferences
    total_samples_all_prefs = n_samples * n_preferences
    
    print(f"  训练集总样本数（所有偏好）: {total_train_samples:,}")
    print(f"     = {n_train:,} 个场景 × {n_preferences} 个偏好")
    print(f"  验证集总样本数（所有偏好）: {total_val_samples:,}")
    print(f"     = {n_val:,} 个场景 × {n_preferences} 个偏好")
    print(f"  总样本数（所有偏好）: {total_samples_all_prefs:,}")
    print(f"     = {n_samples:,} 个场景 × {n_preferences} 个偏好")
    
    # 数据存储大小估算
    if len(y_train_by_pref) > 0:
        sample_lc = list(y_train_by_pref.keys())[0]
        sample_tensor = y_train_by_pref[sample_lc]
        bytes_per_sample = sample_tensor.element_size() * sample_tensor.numel()
        
        print(f"\n【存储空间估算】")
        print(f"  单个样本大小（输出）: {bytes_per_sample / 1024:.2f} KB")
        print(f"  训练集存储大小（所有偏好）: {total_train_samples * bytes_per_sample / (1024**3):.2f} GB")
        print(f"  验证集存储大小（所有偏好）: {total_val_samples * bytes_per_sample / (1024**3):.2f} GB")
        print(f"  总存储大小（所有偏好）: {total_samples_all_prefs * bytes_per_sample / (1024**3):.2f} GB")
    
    # 偏好值范围
    print(f"\n【偏好值范围】")
    print(f"  λ_carbon 最小值: {min(lambda_carbon_values):.2f}")
    print(f"  λ_carbon 最大值: {max(lambda_carbon_values):.2f}")
    print(f"  λ_carbon 数量: {len(lambda_carbon_values)}")
    
    # 验证数据完整性
    print(f"\n【数据完整性检查】")
    all_prefs_have_train = all(lc in y_train_by_pref for lc in lambda_carbon_values)
    all_prefs_have_val = all(lc in y_val_by_pref for lc in lambda_carbon_values) if len(y_val_by_pref) > 0 else False
    
    print(f"  所有偏好都有训练数据: {'✓' if all_prefs_have_train else '✗'}")
    print(f"  所有偏好都有验证数据: {'✓' if all_prefs_have_val else '✗'}")
    
    # 检查每个偏好的训练/验证数据形状是否一致
    if len(y_train_by_pref) > 0:
        sample_lc = list(y_train_by_pref.keys())[0]
        expected_train_shape = (n_train, output_dim)
        expected_val_shape = (n_val, output_dim)
        
        train_shapes_consistent = all(
            y_train_by_pref[lc].shape == expected_train_shape 
            for lc in y_train_by_pref.keys()
        )
        val_shapes_consistent = all(
            y_val_by_pref[lc].shape == expected_val_shape 
            for lc in y_val_by_pref.keys()
        ) if len(y_val_by_pref) > 0 else True
        
        print(f"  训练数据形状一致性: {'✓' if train_shapes_consistent else '✗'}")
        print(f"  验证数据形状一致性: {'✓' if val_shapes_consistent else '✗'}")
        
        if train_shapes_consistent:
            print(f"    期望训练数据形状: {expected_train_shape}")
        if val_shapes_consistent and len(y_val_by_pref) > 0:
            print(f"    期望验证数据形状: {expected_val_shape}")
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)
    
    # 返回统计信息字典
    return {
        'n_samples': n_samples,
        'n_train': n_train,
        'n_val': n_val,
        'n_preferences': n_preferences,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'total_train_samples': total_train_samples,
        'total_val_samples': total_val_samples,
        'lambda_carbon_values': lambda_carbon_values,
    }


if __name__ == "__main__":
    stats = analyze_dataset_size()
    
    # 如果需要，可以保存统计结果
    if stats:
        print(f"\n统计结果已计算完成")
        print(f"可以使用返回值进行进一步分析")

