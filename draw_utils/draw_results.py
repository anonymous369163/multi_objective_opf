"""
画图辅助代码
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def ensure_pictures_dir():
    """确保pictures文件夹存在，如果不存在则创建"""
    if not os.path.exists('pictures'):
        os.makedirs('pictures')
        print("Created 'pictures' directory for saving plots.")


def plot_performance_comparison(results, colors, markers, method_names, 
                                enable_supervised, enable_actorflow, enable_runopp, test_ground_truth):
    """绘制性能对比图表（包含6个子图）
    
    该函数生成一个综合性能对比图，包含以下6个子图：
    1. 奖励变化对比 - 比较各方法在每个时间步的奖励值
    2. PQVI总违反量对比 - 比较各方法的总约束违反情况（P+Q+V+I）
    3. 经济成本对比 - 比较各方法的发电经济成本
    4. 碳排放成本对比 - 比较各方法的碳排放成本
    5. 总求解时间对比 - 柱状图显示各方法的总计算时间
    6. 平均每步求解时间对比 - 柱状图显示各方法的单步平均计算时间
    
    Args:
        results: 各方法的测试结果字典
        colors: 各方法的颜色映射
        markers: 各方法的标记样式映射
        method_names: 各方法的显示名称映射
        enable_supervised: 是否启用监督学习模型
        enable_actorflow: 是否启用流模型
        enable_runopp: 是否启用最优潮流求解器
        test_ground_truth: 是否测试真实标签
    """
    import matplotlib.pyplot as plt
    
    # 设置中文显示和样式
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')

    plt.figure(figsize=(20, 12))

    # 动态生成图例标题
    enabled_methods = [method_names[key] for key in results.keys()]
    title_suffix = " vs ".join(enabled_methods) + " Performance Comparison"

    # 1. 奖励对比图 - 展示各方法在每个时间步的奖励变化趋势
    plt.subplot(2, 3, 1)
    for key, result in results.items():
        plt.plot(range(len(result['step_rewards'])), result['step_rewards'], 
                 marker=markers[key], markersize=3, linewidth=2, 
                 label=method_names[key], color=colors[key])
    plt.title('Reward Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. 总PQVI违反对比图 - P+Q+V+I的总和，衡量整体约束满足情况
    plt.subplot(2, 3, 2)
    for key, result in results.items():
        total_pqvi = np.array(result['p_violation']) + np.array(result['q_violation']) + \
                     np.array(result['v_violation']) + np.array(result['i_violation'])
        plt.plot(range(len(total_pqvi)), total_pqvi, 
                 marker=markers[key], markersize=3, linewidth=2, 
                 label=method_names[key], color=colors[key])
    plt.title('Total PQVI Violation Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Violation Degree', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. 经济成本对比图 - 发电经济成本，越低越好
    plt.subplot(2, 3, 3)
    for key, result in results.items():
        plt.plot(range(len(result['gen_cost'])), result['gen_cost'], 
                 marker=markers[key], markersize=3, linewidth=2, 
                 label=method_names[key], color=colors[key])
    plt.title('Economic Cost Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cost ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. 碳排放成本对比图 - 碳排放成本，反映环保性能
    plt.subplot(2, 3, 4)
    for key, result in results.items():
        plt.plot(range(len(result['carbon_cost'])), result['carbon_cost'], 
                 marker=markers[key], markersize=3, linewidth=2, 
                 label=method_names[key], color=colors[key])
    plt.title('Carbon Emission Cost Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Carbon Cost ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 5. 总求解时间对比柱状图 - 展示各方法的计算效率（总时间）
    plt.subplot(2, 3, 5)
    method_list = list(results.keys())
    times = [results[key]['total_time'] for key in method_list]
    colors_list = [colors[key] for key in method_list]
    labels_list = [method_names[key] for key in method_list]
    
    bars = plt.bar(range(len(method_list)), times, color=colors_list, alpha=0.7)
    plt.title('Total Solution Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(range(len(method_list)), labels_list, rotation=15, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 在柱状图上添加数值标签
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s',
                ha='center', va='bottom', fontsize=10)

    # 6. 平均每步求解时间对比柱状图 - 展示各方法的单步计算效率
    plt.subplot(2, 3, 6)
    avg_times = [results[key]['avg_time_per_step'] for key in method_list]
    
    bars = plt.bar(range(len(method_list)), avg_times, color=colors_list, alpha=0.7)
    plt.title('Average Time per Step Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(range(len(method_list)), labels_list, rotation=15, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 在柱状图上添加数值标签
    for i, (bar, time_val) in enumerate(zip(bars, avg_times)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s',
                ha='center', va='bottom', fontsize=10)

    # 添加总标题
    plt.suptitle(title_suffix, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 确保pictures文件夹存在
    ensure_pictures_dir()
    
    # 根据启用的方法生成文件名
    filename_parts = []
    if enable_supervised:
        filename_parts.append("supervised")
    if enable_actorflow:
        filename_parts.append("actorflow")
    if enable_runopp:
        filename_parts.append("runopp")
    if test_ground_truth:
        filename_parts.append("ground_truth")
    
    filename = os.path.join('pictures', "_vs_".join(filename_parts) + "_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPerformance comparison chart saved as: {filename}")
    plt.show()


def print_performance_statistics(results, method_names):
    """打印各方法的性能统计信息
    
    该函数按照以下类别打印详细的统计信息：
    1. 奖励指标 - 平均奖励和总奖励
    2. 约束违反指标 - P/Q/V/I各项违反量及总违反量
    3. 经济成本指标 - 平均成本和总成本
    4. 碳排放成本指标 - 平均碳成本和总碳成本
    5. 求解时间指标 - 总时间和平均每步时间
    
    Args:
        results: 各方法的测试结果字典
        method_names: 各方法的显示名称映射
    """
    print("\n" + "="*60)
    print("Performance Statistics Comparison")
    print("="*60)
    
    # 1. 打印奖励指标
    print("\n[Reward Metrics]")
    for key, result in results.items():
        method_name = method_names[key]
        print(f"  {method_name}: Average={np.mean(result['step_rewards']):.4f}, Total={np.sum(result['step_rewards']):.4f}")
    
    # 2. 打印约束违反指标（包含P/Q/V/I各项）
    print("\n[Constraint Violation Metrics]")
    for key, result in results.items():
        method_name = method_names[key]
        total_pqvi = (np.mean(result['p_violation']) + np.mean(result['q_violation']) + 
                      np.mean(result['v_violation']) + np.mean(result['i_violation']))
        print(f"  {method_name}: Average PQVI Violation={total_pqvi:.4f}")
        print(f"    - P Violation: {np.mean(result['p_violation']):.4f}")
        print(f"    - Q Violation: {np.mean(result['q_violation']):.4f}")
        print(f"    - V Violation: {np.mean(result['v_violation']):.4f}")
        print(f"    - I Violation: {np.mean(result['i_violation']):.4f}")
    
    # 3. 打印经济成本指标
    print("\n[Economic Cost Metrics]")
    for key, result in results.items():
        method_name = method_names[key]
        print(f"  {method_name}: Average={np.mean(result['gen_cost']):.4f}, Total={np.sum(result['gen_cost']):.4f}")
    
    # 4. 打印碳排放成本指标
    print("\n[Carbon Emission Cost Metrics]")
    for key, result in results.items():
        method_name = method_names[key]
        print(f"  {method_name}: Average={np.mean(result['carbon_cost']):.4f}, Total={np.sum(result['carbon_cost']):.4f}")
    
    # 5. 打印求解时间指标
    print("\n[Solution Time Metrics]")
    for key, result in results.items():
        method_name = method_names[key]
        print(f"  {method_name}: Total Time={result['total_time']:.4f}s, Average per Step={result['avg_time_per_step']:.4f}s")
    
    print("\n" + "="*60)



def analyze_and_visualize_prediction_errors(results, method_names, ground_truth_vm_va):
    """分析并可视化模型预测与真实标签的差异
    
    该函数执行以下分析：
    1. 收集所有模型的预测Vm和Va
    2. 计算与真实标签的MSE/MAE/Max Error（总体、Vm、Va）
    3. 比较rectified和simple模型的差异（如果两者都存在）
    4. 分析预测误差与约束违反的相关性
    5. 生成6个可视化分析图
    6. 保存分析结果到JSON文件
    
    Args:
        results: 各方法的测试结果字典
        method_names: 各方法的显示名称映射
        ground_truth_vm_va: 真实标签的Vm和Va（numpy数组）
    """
    import matplotlib.pyplot as plt
    import json
    
    print("\n" + "="*60)
    print("Analysis of Model Prediction vs Ground Truth")
    print("="*60)
    
    # ========== 步骤1: 收集所有模型的预测Vm和Va ==========
    model_vm_va = {}
    for key, result in results.items():
        if key != 'ground_truth' and 'Vm' in result and 'Va' in result:
            if result['Vm'] is not None and result['Va'] is not None:
                vm = result['Vm']
                va = result['Va']
                
                # 处理可能的多采样维度 (num_steps, sample_num, num_buses) -> (num_steps, num_buses)
                if vm.ndim == 3:
                    # 取第一个采样点或者平均
                    vm = vm[:, 0, :]  # 取第一个采样点
                if va.ndim == 3:
                    va = va[:, 0, :]
                
                # 合并Vm和Va为[Vm, Va]格式，与ground_truth_vm_va一致
                model_vm_va[key] = np.concatenate([vm, va], axis=1)
    
    if not model_vm_va:
        print("\n  Warning: No model predictions collected, unable to perform difference analysis")
        return
    
    # 提取真实标签的Vm和Va
    vm_dim = ground_truth_vm_va.shape[1] // 2  # 电压幅值的维度
    ground_truth_vm = ground_truth_vm_va[:, :vm_dim]
    ground_truth_va = ground_truth_vm_va[:, vm_dim:]
    
    print(f"\nGround Truth Shape: {ground_truth_vm_va.shape}")
    print(f"  - Vm Dimension: {ground_truth_vm.shape}")
    print(f"  - Va Dimension: {ground_truth_va.shape}")
    
    # ========== 步骤2: 计算与真实标签的差异（MSE/MAE/Max Error） ==========
    print("\n[MSE/MAE Difference from Ground Truth]")
    
    for model_name, vm_va_pred in model_vm_va.items():
        # 提取预测的Vm和Va
        pred_vm = vm_va_pred[:, :vm_dim]
        pred_va = vm_va_pred[:, vm_dim:]
        
        # 计算MSE
        mse_total = np.mean((vm_va_pred - ground_truth_vm_va)**2)
        mse_vm = np.mean((pred_vm - ground_truth_vm)**2)
        mse_va = np.mean((pred_va - ground_truth_va)**2)
        
        # 计算MAE
        mae_total = np.mean(np.abs(vm_va_pred - ground_truth_vm_va))
        mae_vm = np.mean(np.abs(pred_vm - ground_truth_vm))
        mae_va = np.mean(np.abs(pred_va - ground_truth_va))
        
        # 计算最大误差
        max_error_total = np.max(np.abs(vm_va_pred - ground_truth_vm_va))
        max_error_vm = np.max(np.abs(pred_vm - ground_truth_vm))
        max_error_va = np.max(np.abs(pred_va - ground_truth_va))
        
        print(f"\n  {method_names[model_name]}:")
        print(f"    Overall:")
        print(f"      - MSE: {mse_total:.6f}")
        print(f"      - MAE: {mae_total:.6f}")
        print(f"      - Max Error: {max_error_total:.6f}")
        print(f"    Voltage Magnitude (Vm):")
        print(f"      - MSE: {mse_vm:.6f}")
        print(f"      - MAE: {mae_vm:.6f}")
        print(f"      - Max Error: {max_error_vm:.6f}")
        print(f"    Voltage Angle (Va):")
        print(f"      - MSE: {mse_va:.6f}")
        print(f"      - MAE: {mae_va:.6f}")
        print(f"      - Max Error: {max_error_va:.6f}")
    
    # ========== 步骤3: 比较rectified和simple模型的差异 ==========
    if 'rectified' in model_vm_va and 'simple' in model_vm_va:
        print("\n[Rectified vs Simple Model Difference Analysis]")
        rectified_vm_va = model_vm_va['rectified']
        simple_vm_va = model_vm_va['simple']
        
        # 计算两个模型预测的差异
        diff_vm_va = rectified_vm_va - simple_vm_va
        mse_diff = np.mean(diff_vm_va**2)
        mae_diff = np.mean(np.abs(diff_vm_va))
        
        # 计算与真实标签的相对差异
        rectified_error = rectified_vm_va - ground_truth_vm_va
        simple_error = simple_vm_va - ground_truth_vm_va
        
        # 统计哪个模型更接近真实标签（逐元素比较）
        rectified_closer = np.sum(np.abs(rectified_error) < np.abs(simple_error))
        simple_closer = np.sum(np.abs(simple_error) < np.abs(rectified_error))
        total_elements = rectified_error.size
          
        print(f"  Prediction Difference between Two Models:")
        print(f"    - MSE: {mse_diff:.6f}")
        print(f"    - MAE: {mae_diff:.6f}")
        print(f"\n  Which Model is Closer to Ground Truth (Element-wise):")
        print(f"    - Rectified Closer: {rectified_closer}/{total_elements} ({100*rectified_closer/total_elements:.2f}%)")
        print(f"    - Simple Closer: {simple_closer}/{total_elements} ({100*simple_closer/total_elements:.2f}%)")
        
        # 分析哪些场景下rectified表现更好（场景级别比较）
        scene_wise_rectified_error = np.mean(np.abs(rectified_error), axis=1)
        scene_wise_simple_error = np.mean(np.abs(simple_error), axis=1)
        
        rectified_better_scenes = np.sum(scene_wise_rectified_error < scene_wise_simple_error)
        simple_better_scenes = np.sum(scene_wise_simple_error < scene_wise_rectified_error)
        
        print(f"\n  Scenario-Level Comparison:")
        print(f"    - Rectified Better Scenarios: {rectified_better_scenes}/{len(scene_wise_rectified_error)} ({100*rectified_better_scenes/len(scene_wise_rectified_error):.2f}%)")
        print(f"    - Simple Better Scenarios: {simple_better_scenes}/{len(scene_wise_simple_error)} ({100*simple_better_scenes/len(scene_wise_simple_error):.2f}%)")
        
        # ========== 步骤4: 分析预测误差与约束违反的相关性 ==========
        print(f"\n[Correlation Analysis: Prediction Error vs Constraint Violation]")
        
        for model_name in ['rectified', 'simple']:
            if model_name in model_vm_va and model_name in results:
                vm_va_pred = model_vm_va[model_name]
                result = results[model_name]
                
                # 计算每个场景的预测误差
                scene_errors = np.mean(np.abs(vm_va_pred - ground_truth_vm_va), axis=1)
                
                # 计算每个场景的总约束违反
                scene_violations = (np.array(result['p_violation']) + 
                                  np.array(result['q_violation']) + 
                                  np.array(result['v_violation']) + 
                                  np.array(result['i_violation']))
                
                # 计算相关系数
                correlation = np.corrcoef(scene_errors, scene_violations)[0, 1]
                
                print(f"\n  {method_names[model_name]}:")
                print(f"    - Correlation Coefficient: {correlation:.4f}")
                print(f"    - Average Prediction Error: {np.mean(scene_errors):.6f}")
                print(f"    - Average Constraint Violation: {np.mean(scene_violations):.6f}")
    
    # ========== 步骤5: 生成可视化分析图（6个子图） ==========
    print("\n[Generating Visualization Charts]")
    _plot_error_analysis_figures(model_vm_va, method_names, ground_truth_vm_va, 
                                 ground_truth_vm, ground_truth_va, vm_dim, results)
    
    # ========== 步骤6: 生成节点级别（Bus-Level）误差分析图 ==========
    print("\n[Generating Bus-Level Error Analysis]")
    plot_bus_level_error_analysis(model_vm_va, method_names, ground_truth_vm_va,
                                   ground_truth_vm, ground_truth_va, vm_dim, results)
    
    # ========== 步骤7: 生成多模型对比的节点级别误差分析图 ==========
    if len(model_vm_va) > 1:
        plot_comparative_bus_level_analysis(model_vm_va, method_names, ground_truth_vm_va,
                                           ground_truth_vm, ground_truth_va, vm_dim)
    
    # ========== 步骤8: 保存分析结果到JSON文件 ==========
    _save_analysis_to_json(model_vm_va, method_names, ground_truth_vm_va, results)


def _plot_error_analysis_figures(model_vm_va, method_names, ground_truth_vm_va, 
                                 ground_truth_vm, ground_truth_va, vm_dim, results):
    """绘制误差分析可视化图表（6个子图）
    
    该函数生成以下6个分析图：
    1. 预测误差分布对比（直方图） - 展示各模型预测误差的分布情况
    2. Vm误差对比（折线图） - 展示各模型在每个场景的电压幅值误差
    3. Va误差对比（折线图） - 展示各模型在每个场景的电压相角误差
    4. Rectified模型：预测误差 vs 约束违反（散点图） - 分析误差与违反的关系
    5. Simple模型：预测误差 vs 约束违反（散点图） - 分析误差与违反的关系
    6. 模型性能对比（柱状图） - 标准化显示MSE和各项违反量
    
    Args:
        model_vm_va: 各模型预测的Vm和Va字典
        method_names: 各方法的显示名称映射
        ground_truth_vm_va: 真实标签的Vm和Va
        ground_truth_vm: 真实标签的Vm
        ground_truth_va: 真实标签的Va
        vm_dim: Vm的维度
        results: 各方法的测试结果字典
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20, 10))
    
    # 1. 预测误差分布对比（直方图）- 展示误差的统计分布
    plt.subplot(2, 3, 1)
    for model_name, vm_va_pred in model_vm_va.items():
        errors = np.mean(np.abs(vm_va_pred - ground_truth_vm_va), axis=1)
        plt.hist(errors, bins=30, alpha=0.5, label=method_names[model_name])
    plt.xlabel('Mean Absolute Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Vm误差对比（折线图）- 按场景展示电压幅值预测误差
    plt.subplot(2, 3, 2)
    for model_name, vm_va_pred in model_vm_va.items():
        pred_vm = vm_va_pred[:, :vm_dim]
        vm_errors = np.mean(np.abs(pred_vm - ground_truth_vm), axis=1)
        plt.plot(vm_errors, label=method_names[model_name], alpha=0.7, linewidth=1.5)
    plt.xlabel('Scenario Index', fontsize=12)
    plt.ylabel('Mean Absolute Error (Vm)', fontsize=12)
    plt.title('Voltage Magnitude Error', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Va误差对比（折线图）- 按场景展示电压相角预测误差
    plt.subplot(2, 3, 3)
    for model_name, vm_va_pred in model_vm_va.items():
        pred_va = vm_va_pred[:, vm_dim:]
        va_errors = np.mean(np.abs(pred_va - ground_truth_va), axis=1)
        plt.plot(va_errors, label=method_names[model_name], alpha=0.7, linewidth=1.5)
    plt.xlabel('Scenario Index', fontsize=12)
    plt.ylabel('Mean Absolute Error (Va)', fontsize=12)
    plt.title('Voltage Angle Error', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Rectified模型：预测误差 vs 约束违反（散点图）- 分析两者的相关性
    if 'rectified' in model_vm_va and 'rectified' in results:
        plt.subplot(2, 3, 4)
        vm_va_pred = model_vm_va['rectified']
        result = results['rectified']
        scene_errors = np.mean(np.abs(vm_va_pred - ground_truth_vm_va), axis=1)
        scene_violations = (np.array(result['p_violation']) + 
                          np.array(result['q_violation']) + 
                          np.array(result['v_violation']) + 
                          np.array(result['i_violation']))
        plt.scatter(scene_errors, scene_violations, alpha=0.5, s=20)
        plt.xlabel('Prediction Error (MAE)', fontsize=12)
        plt.ylabel('Total Constraint Violation', fontsize=12)
        plt.title('Rectified: Error vs Violation', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 5. Simple模型：预测误差 vs 约束违反（散点图）- 分析两者的相关性
    if 'simple' in model_vm_va and 'simple' in results:
        plt.subplot(2, 3, 5)
        vm_va_pred = model_vm_va['simple']
        result = results['simple']
        scene_errors = np.mean(np.abs(vm_va_pred - ground_truth_vm_va), axis=1)
        scene_violations = (np.array(result['p_violation']) + 
                          np.array(result['q_violation']) + 
                          np.array(result['v_violation']) + 
                          np.array(result['i_violation']))
        plt.scatter(scene_errors, scene_violations, alpha=0.5, s=20)
        plt.xlabel('Prediction Error (MAE)', fontsize=12)
        plt.ylabel('Total Constraint Violation', fontsize=12)
        plt.title('Simple: Error vs Violation', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 6. 模型性能对比（柱状图）- 标准化显示MSE和各项违反量
    if 'rectified' in model_vm_va and 'simple' in model_vm_va:
        plt.subplot(2, 3, 6)
        
        # 准备数据：MSE和各项违反量
        metrics_names = ['MSE', 'Avg P Violation', 'Avg Q Violation', 
                       'Avg V Violation', 'Avg I Violation']
        
        rectified_metrics = [
            np.mean((model_vm_va['rectified'] - ground_truth_vm_va)**2),
            np.mean(results['rectified']['p_violation']),
            np.mean(results['rectified']['q_violation']),
            np.mean(results['rectified']['v_violation']),
            np.mean(results['rectified']['i_violation'])
        ]
        
        simple_metrics = [
            np.mean((model_vm_va['simple'] - ground_truth_vm_va)**2),
            np.mean(results['simple']['p_violation']),
            np.mean(results['simple']['q_violation']),
            np.mean(results['simple']['v_violation']),
            np.mean(results['simple']['i_violation'])
        ]
        
        # 标准化到0-1范围（取最大值作为基准）
        max_vals = [max(rectified_metrics[i], simple_metrics[i]) for i in range(len(metrics_names))]
        rectified_norm = [rectified_metrics[i]/max_vals[i] if max_vals[i] > 0 else 0 for i in range(len(metrics_names))]
        simple_norm = [simple_metrics[i]/max_vals[i] if max_vals[i] > 0 else 0 for i in range(len(metrics_names))]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        plt.bar(x - width/2, rectified_norm, width, label='Rectified', alpha=0.8)
        plt.bar(x + width/2, simple_norm, width, label='Simple', alpha=0.8)
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Normalized Value', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, metrics_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 确保pictures文件夹存在
    ensure_pictures_dir()
    analysis_filename = os.path.join('pictures', 'action_difference_analysis.png')
    plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
    print(f"  Analysis chart saved as: {analysis_filename}")
    plt.show()


def _save_analysis_to_json(model_vm_va, method_names, ground_truth_vm_va, results):
    
    """保存分析结果到JSON文件
    
    该函数将以下分析结果保存为JSON格式：
    1. 各模型的误差统计（MSE、MAE、Max Error）
    2. 各模型的约束违反统计（P/Q/V/I各项及总和）
    3. Rectified和Simple模型的对比结果（如果两者都存在）
    
    Args:
        model_vm_va: 各模型预测的Vm和Va字典
        method_names: 各方法的显示名称映射
        ground_truth_vm_va: 真实标签的Vm和Va
        results: 各方法的测试结果字典
    """
    import json
    
    analysis_results = {
        'models': {},
        'comparison': {}
    }
    
    # 保存每个模型的误差统计
    for model_name, vm_va_pred in model_vm_va.items():
        mse_total = float(np.mean((vm_va_pred - ground_truth_vm_va)**2))
        mae_total = float(np.mean(np.abs(vm_va_pred - ground_truth_vm_va)))
        max_error_total = float(np.max(np.abs(vm_va_pred - ground_truth_vm_va)))
        
        analysis_results['models'][model_name] = {
            'mse': mse_total,
            'mae': mae_total,
            'max_error': max_error_total,
            'avg_p_violation': float(np.mean(results[model_name]['p_violation'])),
            'avg_q_violation': float(np.mean(results[model_name]['q_violation'])),
            'avg_v_violation': float(np.mean(results[model_name]['v_violation'])),
            'avg_i_violation': float(np.mean(results[model_name]['i_violation'])),
            'avg_total_violation': float(np.mean(
                np.array(results[model_name]['p_violation']) + 
                np.array(results[model_name]['q_violation']) + 
                np.array(results[model_name]['v_violation']) + 
                np.array(results[model_name]['i_violation'])
            ))
        }
    
    # 保存模型间比较结果（如果rectified和simple都存在）
    if 'rectified' in model_vm_va and 'simple' in model_vm_va:
        rectified_error = model_vm_va['rectified'] - ground_truth_vm_va
        simple_error = model_vm_va['simple'] - ground_truth_vm_va
        
        scene_wise_rectified_error = np.mean(np.abs(rectified_error), axis=1)
        scene_wise_simple_error = np.mean(np.abs(simple_error), axis=1)
        
        rectified_better_scenes = int(np.sum(scene_wise_rectified_error < scene_wise_simple_error))
        simple_better_scenes = int(np.sum(scene_wise_simple_error < scene_wise_rectified_error))
        
        analysis_results['comparison'] = {
            'rectified_better_scenes': rectified_better_scenes,
            'simple_better_scenes': simple_better_scenes,
            'total_scenes': len(scene_wise_rectified_error),
            'rectified_better_percentage': float(100*rectified_better_scenes/len(scene_wise_rectified_error)),
            'simple_better_percentage': float(100*simple_better_scenes/len(scene_wise_simple_error))
        }
    
    analysis_json_file = 'action_difference_analysis_results.json'
    with open(analysis_json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4, ensure_ascii=False)
    print(f"\n  Analysis results saved as: {analysis_json_file}")


def plot_bus_level_error_analysis(model_vm_va, method_names, ground_truth_vm_va,
                                   ground_truth_vm, ground_truth_va, vm_dim, results, top_k=20):
    """绘制节点级别（Bus-Level）的误差分析图
    
    该函数生成详细的节点级别误差分析，帮助定位哪些节点的预测误差较大。
    包含以下分析图：
    1. 每个bus的平均Vm误差（柱状图） - 识别电压幅值误差最大的节点
    2. 每个bus的平均Va误差（柱状图） - 识别电压相角误差最大的节点
    3. Bus x Scenario 热力图（Vm误差） - 展示每个节点在不同场景下的电压幅值误差
    4. Bus x Scenario 热力图（Va误差） - 展示每个节点在不同场景下的电压相角误差
    5. Top-K误差最大的bus的Vm误差分布（箱线图） - 详细分析问题节点
    6. Top-K误差最大的bus的Va误差分布（箱线图） - 详细分析问题节点
    
    Args:
        model_vm_va: 各模型预测的Vm和Va字典
        method_names: 各方法的显示名称映射
        ground_truth_vm_va: 真实标签的Vm和Va
        ground_truth_vm: 真实标签的Vm
        ground_truth_va: 真实标签的Va
        vm_dim: Vm的维度（即bus的数量）
        results: 各方法的测试结果字典
        top_k: 展示误差最大的前K个节点（默认20）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 为每个模型生成节点级别误差分析图
    for model_name, vm_va_pred in model_vm_va.items():
        print(f"\n  Generating bus-level analysis for {method_names[model_name]}...")
        
        # 提取预测的Vm和Va
        pred_vm = vm_va_pred[:, :vm_dim]
        pred_va = vm_va_pred[:, vm_dim:]
        
        # 计算每个bus的误差 (shape: num_scenarios x num_buses)
        vm_errors = np.abs(pred_vm - ground_truth_vm)
        va_errors = np.abs(pred_va - ground_truth_va)
        
        # 计算每个bus的平均误差 (shape: num_buses)
        bus_vm_mean_errors = np.mean(vm_errors, axis=0)
        bus_va_mean_errors = np.mean(va_errors, axis=0)
        
        # 找出误差最大的top-k个bus
        top_k_vm_buses = np.argsort(bus_vm_mean_errors)[-top_k:][::-1]
        top_k_va_buses = np.argsort(bus_va_mean_errors)[-top_k:][::-1]
        
        # ========== 生成6个子图 ==========
        fig = plt.figure(figsize=(24, 14))
        
        # 1. 每个bus的平均Vm误差（柱状图） - 展示哪些bus误差最大
        ax1 = plt.subplot(2, 3, 1)
        buses = np.arange(vm_dim)
        colors_vm = ['red' if i in top_k_vm_buses[:5] else 'steelblue' for i in buses]
        ax1.bar(buses, bus_vm_mean_errors, color=colors_vm, alpha=0.7, width=0.8)
        ax1.set_xlabel('Bus Index', fontsize=12)
        ax1.set_ylabel('Mean Absolute Error (Vm)', fontsize=12)
        ax1.set_title(f'{method_names[model_name]}: Vm Error per Bus', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        # 标注top 5误差最大的bus
        for i, bus_idx in enumerate(top_k_vm_buses[:5]):
            ax1.text(bus_idx, bus_vm_mean_errors[bus_idx], f'Bus {bus_idx}', 
                    ha='center', va='bottom', fontsize=8, color='red')
        
        # 2. 每个bus的平均Va误差（柱状图） - 展示哪些bus误差最大
        ax2 = plt.subplot(2, 3, 2)
        colors_va = ['red' if i in top_k_va_buses[:5] else 'steelblue' for i in buses]
        ax2.bar(buses, bus_va_mean_errors, color=colors_va, alpha=0.7, width=0.8)
        ax2.set_xlabel('Bus Index', fontsize=12)
        ax2.set_ylabel('Mean Absolute Error (Va)', fontsize=12)
        ax2.set_title(f'{method_names[model_name]}: Va Error per Bus', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        # 标注top 5误差最大的bus
        for i, bus_idx in enumerate(top_k_va_buses[:5]):
            ax2.text(bus_idx, bus_va_mean_errors[bus_idx], f'Bus {bus_idx}', 
                    ha='center', va='bottom', fontsize=8, color='red')
        
        # 3. Bus x Scenario 热力图（Vm误差） - 展示时空分布
        ax3 = plt.subplot(2, 3, 3)
        im1 = ax3.imshow(vm_errors.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax3.set_xlabel('Scenario Index', fontsize=12)
        ax3.set_ylabel('Bus Index', fontsize=12)
        ax3.set_title(f'{method_names[model_name]}: Vm Error Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax3, label='Absolute Error')
        
        # 4. Bus x Scenario 热力图（Va误差） - 展示时空分布
        ax4 = plt.subplot(2, 3, 4)
        im2 = ax4.imshow(va_errors.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax4.set_xlabel('Scenario Index', fontsize=12)
        ax4.set_ylabel('Bus Index', fontsize=12)
        ax4.set_title(f'{method_names[model_name]}: Va Error Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax4, label='Absolute Error')
        
        # 5. Top-K误差最大的bus的Vm误差分布（箱线图） - 详细展示问题节点
        ax5 = plt.subplot(2, 3, 5)
        vm_error_data = [vm_errors[:, bus_idx] for bus_idx in top_k_vm_buses[:10]]
        bp1 = ax5.boxplot(vm_error_data, labels=[f'B{bus_idx}' for bus_idx in top_k_vm_buses[:10]], 
                          patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax5.set_xlabel('Top-10 Buses with Largest Vm Error', fontsize=12)
        ax5.set_ylabel('Absolute Error Distribution (Vm)', fontsize=12)
        ax5.set_title(f'{method_names[model_name]}: Vm Error Distribution (Top Buses)', 
                     fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 6. Top-K误差最大的bus的Va误差分布（箱线图） - 详细展示问题节点
        ax6 = plt.subplot(2, 3, 6)
        va_error_data = [va_errors[:, bus_idx] for bus_idx in top_k_va_buses[:10]]
        bp2 = ax6.boxplot(va_error_data, labels=[f'B{bus_idx}' for bus_idx in top_k_va_buses[:10]], 
                          patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
        ax6.set_xlabel('Top-10 Buses with Largest Va Error', fontsize=12)
        ax6.set_ylabel('Absolute Error Distribution (Va)', fontsize=12)
        ax6.set_title(f'{method_names[model_name]}: Va Error Distribution (Top Buses)', 
                     fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 确保pictures文件夹存在
        ensure_pictures_dir()
        
        # 保存图表
        bus_level_filename = os.path.join('pictures', f'bus_level_error_analysis_{model_name}.png')
        plt.savefig(bus_level_filename, dpi=300, bbox_inches='tight')
        print(f"    Bus-level analysis saved as: {bus_level_filename}")
        plt.show()
        
        # ========== 打印统计信息：Top-K误差最大的bus ==========
        print(f"\n  [{method_names[model_name]}] Top-{min(10, top_k)} Buses with Largest Vm Error:")
        for i, bus_idx in enumerate(top_k_vm_buses[:10]):
            mean_err = bus_vm_mean_errors[bus_idx]
            max_err = np.max(vm_errors[:, bus_idx])
            std_err = np.std(vm_errors[:, bus_idx])
            print(f"    {i+1}. Bus {bus_idx}: Mean={mean_err:.6f}, Max={max_err:.6f}, Std={std_err:.6f}")
        
        print(f"\n  [{method_names[model_name]}] Top-{min(10, top_k)} Buses with Largest Va Error:")
        for i, bus_idx in enumerate(top_k_va_buses[:10]):
            mean_err = bus_va_mean_errors[bus_idx]
            max_err = np.max(va_errors[:, bus_idx])
            std_err = np.std(va_errors[:, bus_idx])
            print(f"    {i+1}. Bus {bus_idx}: Mean={mean_err:.6f}, Max={max_err:.6f}, Std={std_err:.6f}")
        
        # ========== 分析误差最大的bus与约束违反的关系 ==========
        if model_name in results:
            print(f"\n  [{method_names[model_name]}] Analyzing Relationship between Bus Errors and Constraint Violations:")
            
            # 计算每个场景的总约束违反
            scene_violations = (np.array(results[model_name]['p_violation']) + 
                              np.array(results[model_name]['q_violation']) + 
                              np.array(results[model_name]['v_violation']) + 
                              np.array(results[model_name]['i_violation']))
            
            # 分析top bus误差与约束违反的相关性
            for bus_type, top_buses, errors in [('Vm', top_k_vm_buses[:5], vm_errors),
                                                 ('Va', top_k_va_buses[:5], va_errors)]:
                print(f"\n    Top-5 {bus_type} Error Buses Correlation with Violations:")
                for bus_idx in top_buses:
                    bus_errors = errors[:, bus_idx]
                    if len(bus_errors) == len(scene_violations):
                        correlation = np.corrcoef(bus_errors, scene_violations)[0, 1]
                        print(f"      Bus {bus_idx}: Correlation = {correlation:.4f}")
                    else:
                        print(f"      Bus {bus_idx}: Length mismatch, cannot compute correlation")


def plot_comparative_bus_level_analysis(model_vm_va, method_names, ground_truth_vm_va,
                                         ground_truth_vm, ground_truth_va, vm_dim, top_k=15):
    """绘制多模型对比的节点级别误差分析图
    
    该函数在一张图中对比多个模型在各个节点上的预测误差。
    适用于有多个模型需要对比时使用。
    
    包含以下对比图：
    1. 所有模型的Vm误差对比（每个bus的误差） - 折线图
    2. 所有模型的Va误差对比（每个bus的误差） - 折线图
    3. Top-K误差最大bus的模型对比（Vm） - 分组柱状图
    4. Top-K误差最大bus的模型对比（Va） - 分组柱状图
    
    Args:
        model_vm_va: 各模型预测的Vm和Va字典
        method_names: 各方法的显示名称映射
        ground_truth_vm_va: 真实标签的Vm和Va
        ground_truth_vm: 真实标签的Vm
        ground_truth_va: 真实标签的Va
        vm_dim: Vm的维度（即bus的数量）
        top_k: 展示误差最大的前K个节点（默认15）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n[Generating Comparative Bus-Level Analysis for All Models]")
    
    # 计算所有模型的bus级别误差
    all_models_vm_errors = {}
    all_models_va_errors = {}
    
    for model_name, vm_va_pred in model_vm_va.items():
        pred_vm = vm_va_pred[:, :vm_dim]
        pred_va = vm_va_pred[:, vm_dim:]
        
        vm_errors = np.abs(pred_vm - ground_truth_vm)
        va_errors = np.abs(pred_va - ground_truth_va)
        
        # 每个bus的平均误差
        all_models_vm_errors[model_name] = np.mean(vm_errors, axis=0)
        all_models_va_errors[model_name] = np.mean(va_errors, axis=0)
    
    # 找出所有模型中误差最大的top-k个bus（按平均误差排序）
    avg_vm_errors_all = np.mean([errors for errors in all_models_vm_errors.values()], axis=0)
    avg_va_errors_all = np.mean([errors for errors in all_models_va_errors.values()], axis=0)
    
    top_k_vm_buses = np.argsort(avg_vm_errors_all)[-top_k:][::-1]
    top_k_va_buses = np.argsort(avg_va_errors_all)[-top_k:][::-1]
    
    # ========== 生成4个对比子图 ==========
    fig = plt.figure(figsize=(24, 12))
    
    # 1. 所有模型的Vm误差对比（折线图） - 展示各模型在每个bus上的误差差异
    ax1 = plt.subplot(2, 2, 1)
    buses = np.arange(vm_dim)
    for model_name, bus_errors in all_models_vm_errors.items():
        ax1.plot(buses, bus_errors, label=method_names[model_name], 
                alpha=0.7, linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Bus Index', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (Vm)', fontsize=12)
    ax1.set_title('All Models: Vm Error Comparison per Bus', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 所有模型的Va误差对比（折线图） - 展示各模型在每个bus上的误差差异
    ax2 = plt.subplot(2, 2, 2)
    for model_name, bus_errors in all_models_va_errors.items():
        ax2.plot(buses, bus_errors, label=method_names[model_name], 
                alpha=0.7, linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Bus Index', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (Va)', fontsize=12)
    ax2.set_title('All Models: Va Error Comparison per Bus', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top-K误差最大bus的模型对比（Vm） - 分组柱状图
    ax3 = plt.subplot(2, 2, 3)
    x = np.arange(len(top_k_vm_buses))
    width = 0.8 / len(model_vm_va)
    
    for i, (model_name, bus_errors) in enumerate(all_models_vm_errors.items()):
        offset = (i - len(model_vm_va)/2) * width + width/2
        ax3.bar(x + offset, bus_errors[top_k_vm_buses], width, 
               label=method_names[model_name], alpha=0.8)
    
    ax3.set_xlabel('Bus Index', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error (Vm)', fontsize=12)
    ax3.set_title(f'Top-{top_k} Buses with Largest Vm Error (Model Comparison)', 
                 fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'B{bus_idx}' for bus_idx in top_k_vm_buses], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Top-K误差最大bus的模型对比（Va） - 分组柱状图
    ax4 = plt.subplot(2, 2, 4)
    x = np.arange(len(top_k_va_buses))
    
    for i, (model_name, bus_errors) in enumerate(all_models_va_errors.items()):
        offset = (i - len(model_vm_va)/2) * width + width/2
        ax4.bar(x + offset, bus_errors[top_k_va_buses], width, 
               label=method_names[model_name], alpha=0.8)
    
    ax4.set_xlabel('Bus Index', fontsize=12)
    ax4.set_ylabel('Mean Absolute Error (Va)', fontsize=12)
    ax4.set_title(f'Top-{top_k} Buses with Largest Va Error (Model Comparison)', 
                 fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'B{bus_idx}' for bus_idx in top_k_va_buses], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 确保pictures文件夹存在
    ensure_pictures_dir()
    
    # 保存图表
    comparative_filename = os.path.join('pictures', 'comparative_bus_level_error_analysis.png')
    plt.savefig(comparative_filename, dpi=300, bbox_inches='tight')
    print(f"  Comparative bus-level analysis saved as: {comparative_filename}")
    plt.show()
    
    # ========== 打印对比统计信息 ==========
    print("\n[Comparative Statistics: Top Problematic Buses]")
    print(f"\nTop-{min(10, top_k)} Buses with Largest Average Vm Error (across all models):")
    for i, bus_idx in enumerate(top_k_vm_buses[:10]):
        print(f"  {i+1}. Bus {bus_idx}:")
        for model_name, bus_errors in all_models_vm_errors.items():
            print(f"      {method_names[model_name]}: {bus_errors[bus_idx]:.6f}")
    
    print(f"\nTop-{min(10, top_k)} Buses with Largest Average Va Error (across all models):")
    for i, bus_idx in enumerate(top_k_va_buses[:10]):
        print(f"  {i+1}. Bus {bus_idx}:")
        for model_name, bus_errors in all_models_va_errors.items():
            print(f"      {method_names[model_name]}: {bus_errors[bus_idx]:.6f}")