"""
测试模型 
author: peng yue
date: 2025-09-16
todo: need to show the gap between the trained model and baselines
"""

# 预训练模型
from env import PowerGridEnv
from power_grid_agent import PowerGridAgent
import torch
import numpy as np
from rl_utils import get_runopp_action

# 导入共享的训练配置（用于保持测试和训练配置一致）
import sys
sys.path.append('flow_model')
from flow_model.train_main import get_unified_config 
from draw_utils.draw_results import analyze_and_visualize_prediction_errors, _plot_error_analysis_figures, plot_performance_comparison, print_performance_statistics


# 检查发电机功率是否超出限制
def compare_constraint_calculations(env, agent, obs, predicted_Vm, predicted_Va):
    """对比env和actor两种约束计算方式的差异
    
    Args:
        env: PowerGridEnv实例（包含runpp计算的约束违反信息）
        agent: PowerGridAgent实例（包含actor模型）
        obs: 当前观测值
        predicted_Vm: 预测的电压幅值，而且是归一化后的值
        predicted_Va: 预测的电压相角，而且是归一化后的值
    """
    import pandas as pd
    
    device = next(agent.actor.parameters()).device
    
    # ===== 1. 从env获取runpp计算的约束违反 =====
    env_violations = env.constraint_violations
    env_p_viol = env_violations['p_violation']
    env_q_viol = env_violations['q_violation']
    env_v_viol = env_violations['v_violation']
    env_i_viol = env_violations['i_violation']
    env_total = env_violations['total_penalty']
    
    # ===== 2. 从actor获取手动计算的约束违反 =====
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
    
    # 确保predicted_Vm和predicted_Va是正确的格式
    if not isinstance(predicted_Vm, torch.Tensor):
        predicted_Vm = torch.tensor(predicted_Vm, dtype=torch.float32).to(device)
    if not isinstance(predicted_Va, torch.Tensor):
        predicted_Va = torch.tensor(predicted_Va, dtype=torch.float32).to(device)
    
    # 添加batch维度（如果需要）
    if predicted_Vm.dim() == 1:
        predicted_Vm = predicted_Vm.unsqueeze(0)
    if predicted_Va.dim() == 1:
        predicted_Va = predicted_Va.unsqueeze(0)
    
    # 计算约束损失并获取详细信息
    actor_constraint, actor_details = agent.actor.compute_constraint_loss(
        predicted_Vm, predicted_Va, obs_tensor, env, 
        reduction='mean', return_details=True, debug_mode=False   # debug_model只是用来决定是否打印约束违反情况
    )
    
    # 提取actor的约束违反（注意：actor中的约束已经乘以了100）
    actor_p_max_viol = actor_details['g1_pmax'] # 有功功率上限违反
    actor_p_min_viol = actor_details['g2_pmin']  # 有功功率下限违反
    actor_q_max_viol = actor_details['g5_qmax']  # 无功功率上限违反
    actor_q_min_viol = actor_details['g6_qmin']  # 无功功率下限违反
    actor_sf_viol = actor_details['g9_sf']      # 支路首端功率违反
    actor_st_viol = actor_details['g10_st']    # 支路末端功率违反
    
    actor_p_total = actor_p_max_viol + actor_p_min_viol
    actor_q_total = actor_q_max_viol + actor_q_min_viol
    actor_s_total = actor_sf_viol + actor_st_viol
    
    # ===== 3. 详细对比分析 =====
    print("\n" + "="*80)
    print(f"第 {env.current_step} 步约束违反对比分析")
    print("="*80)
    
    # 3.1 有功功率约束对比
    print("\n【有功功率约束对比】")
    print(f"  Env (runpp):  总违反 = {env_p_viol:.6f} MW")
    print(f"  Actor (手动): 总违反 = {actor_p_total:.6f} (已从p.u.转换为MW)")
    print(f"                - 上限违反 = {actor_p_max_viol:.6f}")
    print(f"                - 下限违反 = {actor_p_min_viol:.6f}")
    print(f"  差异: {abs(env_p_viol - actor_p_total):.6f} ({abs(env_p_viol - actor_p_total)/max(env_p_viol, 1e-6)*100:.2f}%)")
    print(f"  注意: Actor中标准化违反量已乘以100(基准功率)转换为实际单位")
    
    # 3.2 无功功率约束对比
    print("\n【无功功率约束对比】")
    print(f"  Env (runpp):  总违反 = {env_q_viol:.6f} Mvar")
    print(f"  Actor (手动): 总违反 = {actor_q_total:.6f} (已从p.u.转换为Mvar)")
    print(f"                - 上限违反 = {actor_q_max_viol:.6f}")
    print(f"                - 下限违反 = {actor_q_min_viol:.6f}")
    print(f"  差异: {abs(env_q_viol - actor_q_total):.6f} ({abs(env_q_viol - actor_q_total)/max(env_q_viol, 1e-6)*100:.2f}%)")
    print(f"  注意: Actor中标准化违反量已乘以100(基准功率)转换为实际单位")
    
    # 3.3 支路约束对比（env使用电流，actor使用功率）
    print("\n【支路约束对比】")
    print(f"  Env (runpp):  电流违反 = {env_i_viol:.6f} kA")
    print(f"  Actor (手动): 支路功率违反 = {actor_s_total:.6f} (已从p.u.转换为MVA)")
    print(f"                - 首端违反 = {actor_sf_viol:.6f}")
    print(f"                - 末端违反 = {actor_st_viol:.6f}")
    print(f"  注意: 两者约束类型不同（电流 vs 功率），无法直接比较")
    
    # 3.4 电压约束对比（env有，actor没有）
    print("\n【电压约束对比】")
    print(f"  Env (runpp):  电压违反 = {env_v_viol:.6f} p.u.")
    print(f"  Actor (手动): 不检查电压约束")
    
    # 3.5 总体对比
    print("\n【总体约束损失对比】")
    actor_total_comparable = actor_p_total + actor_q_total  # 只比较可比较的部分
    env_total_comparable = env_p_viol + env_q_viol
    print(f"  Env (runpp):  有功+无功违反 = {env_total_comparable:.6f} MW+Mvar")
    print(f"  Actor (手动): 有功+无功违反 = {actor_total_comparable:.6f} MW+Mvar")
    print(f"  差异: {abs(env_total_comparable - actor_total_comparable):.6f} ({abs(env_total_comparable - actor_total_comparable)/max(env_total_comparable, 1e-6)*100:.2f}%)")
    
    print(f"\n  Env 总惩罚项:  {env_total:.6f} (包含所有约束的加权和)")
    print(f"  Actor 约束损失: {actor_constraint.item():.6f} (只包含P/Q/支路约束)")
    
    # ===== 4. 详细分析差异来源 =====
    print("\n" + "="*80)
    print("差异来源分析")
    print("="*80)
    
    # 4.1 对比发电机有功功率
    print("\n【发电机有功功率详细对比】")
    analyze_power_differences(env, agent, obs_tensor, predicted_Vm, predicted_Va, device)
    
    # 4.2 总结主要差异来源
    print("\n【主要差异来源总结】")
    print("1. 潮流计算方法:")
    print("   - Env: 使用pandapower的Newton-Raphson方法（runpp）")
    print("   - Actor: 使用自定义的潮流计算方法（基于导纳矩阵）")
    print("\n2. 约束检查项:")
    print("   - Env: 检查发电机P/Q、电压、支路电流")
    print("   - Actor: 检查发电机P/Q、支路功率（不检查电压）")
    print("\n3. 权重和缩放:")
    print("   - Env: w1=w2=w3=1.0, w4=100.0")
    print("   - Actor: 所有约束乘以100")
    print("\n4. 潮流计算精度差异可能导致发电机功率略有不同")
    
    print("="*80 + "\n")


def analyze_power_differences(env, agent, obs_tensor, predicted_Vm, predicted_Va, device):
    """详细分析发电机功率的差异
    
    对比env中runpp计算的发电机功率和actor中手动计算的发电机功率
    """
    import pandas as pd
    
    # 从actor计算潮流
    P_actor, Q_actor, Sf_actor, St_actor = agent.actor.pf(predicted_Vm, predicted_Va)
    
    # 获取负荷数据
    Pd = obs_tensor.T[:env.num_pd]
    Qd = obs_tensor.T[env.num_pd : env.num_pd + env.num_qd]
    
    # 计算发电机功率（加上负荷）
    Pg_actor = P_actor.T.clone()
    Qg_actor = Q_actor.T.clone()
    
    pd_bus_idx = torch.from_numpy(env.pd_bus_idx).long().to(device)
    qd_bus_idx = torch.from_numpy(env.qd_bus_idx).long().to(device)
    gen_bus_idx = torch.from_numpy(env.gen_bus_idx).long().to(device)
    
    Pg_actor.index_add_(0, pd_bus_idx, Pd)
    Qg_actor.index_add_(0, qd_bus_idx, Qd)
    
    # 提取发电机节点的功率
    Pg_actor_gen = Pg_actor[gen_bus_idx].squeeze().detach().cpu().numpy()
    Qg_actor_gen = Qg_actor[gen_bus_idx].squeeze().detach().cpu().numpy()
    
    # 将标准化的功率值转换为实际单位（p.u. -> MW/Mvar）
    # 因为predicted_Vm和predicted_Va是标准化后的值，计算出的功率也是标准化的
    Pg_actor_gen = Pg_actor_gen * 100  # 转换为MW
    Qg_actor_gen = Qg_actor_gen * 100  # 转换为Mvar
    
    # 从env获取runpp计算的发电机功率
    # 需要从net中提取发电机功率和外部电网功率
    pg_env = []
    qg_env = []
    pg_limits = []
    qg_limits = []
    
    # 收集所有发电机功率
    for i, gen in enumerate(env.net.gen.itertuples()):
        pg_env.append(gen.p_mw)
        qg_env.append(env.net.res_gen.q_mvar.iloc[i])
        pg_limits.append((gen.min_p_mw, gen.max_p_mw))
        qg_limits.append((gen.min_q_mvar, gen.max_q_mvar))
    
    # 收集外部电网功率（如果有）
    if hasattr(env.net, 'res_ext_grid') and len(env.net.ext_grid) > 0:
        for ext_grid in env.net.res_ext_grid.itertuples():
            pg_env.append(ext_grid.p_mw)
            qg_env.append(ext_grid.q_mvar)
            # 外部电网通常没有严格限制
            pg_limits.append((-999, 999))
            qg_limits.append((-999, 999))
    
    pg_env = np.array(pg_env)
    qg_env = np.array(qg_env)
    
    # 确保维度匹配
    min_len = min(len(pg_env), len(Pg_actor_gen))
    
    # 创建对比表格
    comparison_data = []
    for i in range(min_len):
        p_diff = Pg_actor_gen[i] - pg_env[i]
        q_diff = Qg_actor_gen[i] - qg_env[i]
        
        comparison_data.append({
            '发电机': i,
            'P_env(MW)': f"{pg_env[i]:.4f}",
            'P_actor(MW)': f"{Pg_actor_gen[i]:.4f}",
            'P_diff(MW)': f"{p_diff:.4f}",
            'P_limit': f"[{pg_limits[i][0]:.1f}, {pg_limits[i][1]:.1f}]",
            'Q_env(Mvar)': f"{qg_env[i]:.4f}",
            'Q_actor(Mvar)': f"{Qg_actor_gen[i]:.4f}",
            'Q_diff(Mvar)': f"{q_diff:.4f}",
            'Q_limit': f"[{qg_limits[i][0]:.1f}, {qg_limits[i][1]:.1f}]"
        })
    
    # 打印表格（只显示差异较大的前10个）
    df = pd.DataFrame(comparison_data)
    df['P_diff_abs'] = df['P_diff(MW)'].astype(float).abs()
    df['Q_diff_abs'] = df['Q_diff(Mvar)'].astype(float).abs()
    df_sorted = df.sort_values(by='P_diff_abs', ascending=False)
    
    print("\n发电机功率对比（按有功功率差异排序，显示前10个）:")
    print(df_sorted.head(10).drop(columns=['P_diff_abs', 'Q_diff_abs']).to_string(index=False))
    
    # 统计信息
    p_diffs = df['P_diff(MW)'].astype(float).values
    q_diffs = df['Q_diff(Mvar)'].astype(float).values
    
    print(f"\n统计信息:")
    print(f"  有功功率差异: 平均={np.mean(np.abs(p_diffs)):.4f} MW, 最大={np.max(np.abs(p_diffs)):.4f} MW")
    print(f"  无功功率差异: 平均={np.mean(np.abs(q_diffs)):.4f} Mvar, 最大={np.max(np.abs(q_diffs)):.4f} Mvar")
    print(f"  有功功率总差异: {np.sum(np.abs(p_diffs)):.4f} MW")
    print(f"  无功功率总差异: {np.sum(np.abs(q_diffs)):.4f} Mvar")
    print(f"  注意: Actor功率已从标准化单位(p.u.)乘以100转换为实际单位(MW/Mvar)")


def run_single_model_test(agent, env, model_name, use_runopp=False, debug_mode=False, 
                          return_actions=False, ground_truth_vm_va=None, ground_truth_action=None):
    """运行单个模型的测试，返回性能指标
    
    Args:
        agent: 智能体
        env: 环境
        model_name: 模型名称
        use_runopp: 是否使用runopp求解器
        debug_mode: 是否开启调试模式
        return_actions: 是否返回预测的actions
        ground_truth_actions: 真实标签actions（如果提供，则直接使用）
    
    Returns:
        dict: 性能指标，如果return_actions=True，还包含'actions'字段
    """
    import time
    
    print(f"开始测试{model_name}...")
    
    # 重置环境
    obs = env.reset()
    
    # 记录指标
    step_rewards = []
    p_violation = []
    q_violation = []
    v_violation = []
    i_violation = []
    gen_cost = []
    carbon_cost = []
    
    # 记录actions（如果需要）
    actions_list = [] if return_actions else None
    predicted_Vm_list = [] if return_actions else None
    predicted_Va_list = [] if return_actions else None
    # 添加诊断信息
    convergence_failures = 0
    
    # 记录求解时间
    start_time = time.time()
    
    while not env.done:
        if env.current_step == env.num_timesteps-1:
            env.done = True  
        
        # 决定action来源
        if ground_truth_vm_va is not None:
            # 使用提供的真实标签actions
            predicted_Vm = ground_truth_vm_va[env.current_step, :env.num_buses]
            predicted_Va = ground_truth_vm_va[env.current_step, env.num_buses:]
            # predicted_Vm, predicted_Va 可能是numpy数组，需要转为torch Tensor以兼容agent.actor.test的实现 
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32).to(device)    
            if not isinstance(predicted_Vm, torch.Tensor):
                predicted_Vm = torch.tensor(predicted_Vm, dtype=torch.float32).to(device)
            if not isinstance(predicted_Va, torch.Tensor):
                predicted_Va = torch.tensor(predicted_Va, dtype=torch.float32).to(device) 
            action = agent.actor.test(obs, predicted_Vm, predicted_Va)  # 直接调用模型里的方法根据电压和相角计算得到最后的action
            action = action.cpu().numpy().squeeze() # 单位是[电压（p.u.），功率（p.u.）]
        elif ground_truth_action is not None:
            # 使用提供的真实标签actions
            action = ground_truth_action[env.current_step]
        elif use_runopp:
            # 使用pandapower的最优潮流求解器
            action = get_runopp_action(env)
        else:
            # 智能体决策 
            if hasattr(agent, 'actor') and hasattr(agent.actor, 'add_carbon_tax') and agent.actor.add_carbon_tax:
                agent.actor.carbon_tax = env.carbon_tax  # 从环境获取当前场景的碳税率

            action, predicted_Vm, predicted_Va = agent.act(obs, out_ma=True) 
        
        # 环境执行动作
        obs, reward, done, _ = env.step(action)

        # debug 模式 - 对比两种约束计算方式
        if debug_mode:
            compare_constraint_calculations(env, agent, obs, predicted_Vm, predicted_Va)
        
        # 检测潮流不收敛
        if not env.converged:
            convergence_failures += 1
            print(f"\n 警告：第 {env.current_step} 步潮流计算不收敛！")
            
            # 强制继续测试而不是提前终止（除非已经是最后一步）
            if env.current_step < env.num_timesteps - 1:
                env.done = False
                done = False
                print(f"   潮流不收敛，但继续测试（使用默认值）")
            else:
                print(f"   已到达最后一步，测试结束")
        
        # 记录性能指标
        step_rewards.append(reward if reward is not None else 0) 
        if return_actions: # 记录action（如果需要）
            actions_list.append(action.copy() if isinstance(action, np.ndarray) else action)
            predicted_Vm_list.append(predicted_Vm.copy() if isinstance(predicted_Vm, np.ndarray) else predicted_Vm)
            predicted_Va_list.append(predicted_Va.copy() if isinstance(predicted_Va, np.ndarray) else predicted_Va)
        
        # 对于不收敛的情况，记录默认值（惩罚值）
        if env.run_pp:
            if env.converged:
                # 正常收敛，记录实际值
                p_violation.append(env.constraint_violations['p_violation'])
                q_violation.append(env.constraint_violations['q_violation'])
                v_violation.append(env.constraint_violations['v_violation'])
                i_violation.append(env.constraint_violations['i_violation'])
                gen_cost.append(env.constraint_violations['total_costs'])
                carbon_cost.append(env.constraint_violations['carbon_costs'])
            else:
                # 不收敛，记录惩罚值（使用较大的值表示不收敛）
                p_violation.append(1e6)
                q_violation.append(1e6)
                v_violation.append(1e6)
                i_violation.append(1e6)
                gen_cost.append(1e6)
                carbon_cost.append(1e6)
    
    # 计算总求解时间
    total_time = time.time() - start_time
    
    # 打印诊断摘要
    print(f"\n{model_name}测试完成")
    print(f" 测试摘要：")
    print(f"   - 完成步数: {env.current_step} / {env.num_timesteps}")
    print(f"   - 潮流不收敛次数: {convergence_failures}")
    print(f"   - 总求解时间: {total_time:.4f}秒")
    print(f"   - 平均每步时间: {total_time/len(step_rewards):.4f}秒")
    if convergence_failures > 0:
        print(f"   警告：存在 {convergence_failures} 次潮流不收敛！")
        if env.current_step < env.num_timesteps:
            print(f"   测试提前终止于第 {env.current_step} 步")
    else:
        print(f"  所有步骤潮流计算均收敛")
    
    results = {
        'step_rewards': step_rewards,
        'p_violation': p_violation,
        'q_violation': q_violation,
        'v_violation': v_violation,
        'i_violation': i_violation,
        'gen_cost': gen_cost,
        'carbon_cost': carbon_cost,
        'total_time': total_time,
        'avg_time_per_step': total_time/len(step_rewards) if len(step_rewards) > 0 else 0
    }
    
    # 如果需要返回actions
    if return_actions:
        results['actions'] = np.array(actions_list) if actions_list else None

        # 检查predicted_Vm_list各元素是否为tensor，并且是否在GPU上，如果是则转为cpu和numpy，保证兼容性
        if predicted_Vm_list:
            vm_arr = []
            for item in predicted_Vm_list:
                # 如果是tensor且在cuda上，先转到cpu
                if isinstance(item, torch.Tensor):
                    item = item.detach()
                    if item.device.type == 'cuda':
                        item = item.cpu()
                    item = item.numpy()
                # 确保形状为(num_buses,)而不是(num_buses, 1)
                item = np.squeeze(item)
                vm_arr.append(item)
            results['Vm'] = np.array(vm_arr)
        else:
            results['Vm'] = None

        if predicted_Va_list:
            va_arr = []
            for item in predicted_Va_list:
                if isinstance(item, torch.Tensor):
                    item = item.detach()
                    if item.device.type == 'cuda':
                        item = item.cpu()
                    item = item.numpy() 
                item = np.squeeze(item)
                va_arr.append(item)
            results['Va'] = np.array(va_arr)
        else:
            results['Va'] = None
    
    return results


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from flow_model.models.actor import PowerSystemConfig
    import copy

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 注意: 导入 train_main 时会 os.chdir 到 flow_model 目录，所以路径需要回退一级
    ps_config = PowerSystemConfig(device=device, case_file_path='../saved_data/pglib_opf_case118.mat') 

    # ==================== 加载训练配置 ====================
    # 使用与训练时相同的配置
    config = get_unified_config(debug_mode=False)
    args = config.copy()  # 直接使用统一配置
    
    # 根据模型类型设置latent_dim
    model_list = config['model_list']
    args['latent_dim'] = config['latent_dim_vae'] if model_list[0] == 'vae' else config['latent_dim_flow']

    # ==================== 配置选择机制 ====================
    # 选择要对比的方法 (True表示启用该方法)
    ENABLE_SUPERVISED = False      # 监督学习模型 
    ENABLE_ACTORFLOW = True       # 流模型（ActorFlow）
    ENABLE_RUNOPP = False         # 最优潮流求解器
    TEST_GROUND_TRUTH_ACTIONS = True  # True: 测试真实标签; False: 不测试
    
    # ==================== 场景加载配置 ====================
    # 是否从训练数据集加载测试场景（用于公平对比）
    USE_DATASET_SCENARIOS = True  # True: 从数据集加载场景; False: 使用环境生成的场景
    
    # ==================== 真实标签验证配置 ====================
    # 是否测试真实标签actions的有效性
    # 是否收集并分析模型预测actions与真实标签的差异
    ANALYZE_ACTION_DIFFERENCES = True  # True: 分析差异; False: 不分析
    
    print("=== 模型对比测试配置 ===")
    print(f"监督学习模型: {'启用' if ENABLE_SUPERVISED else '禁用'}") 
    print(f"流模型(ActorFlow): {'启用' if ENABLE_ACTORFLOW else '禁用'}")
    print(f"最优潮流求解器: {'启用' if ENABLE_RUNOPP else '禁用'}")
    print(f"使用数据集场景: {'是' if USE_DATASET_SCENARIOS else '否'}")
    print(f"测试真实标签: {'是' if TEST_GROUND_TRUTH_ACTIONS else '否'}")
    print(f"分析动作差异: {'是' if ANALYZE_ACTION_DIFFERENCES else '否'}")
    print("=" * 50)

    # 创建保存模型和图片的文件夹 
    os.makedirs('training_plots', exist_ok=True)

    case_name = "case118"
    if case_name != "case118":
        ps_config = None
    num_timesteps = 10   # 288
    random_load = True
    run_pp = True 
    # ==================== 加载测试场景（如果启用）====================
    external_load_scenarios = None
    external_carbon_tax_scenarios = None
    ground_truth_vm_va = None
    data_v2 = None
    
    if USE_DATASET_SCENARIOS:
        print("\n=== 从数据集加载测试场景 ===")
        from flow_model.load_opf_data_v2 import OPF_Flow_Dataset_V2
        
        # 使用与训练相同的数据路径和配置
        data_path = config['data_path']
        add_carbon_tax = config['add_carbon_tax']
        single_target = True if 'prefernces' not in data_path else False   
        args['single_target'] = single_target
        # 加载数据集（与训练时相同的分割方式）
        data_v2 = OPF_Flow_Dataset_V2(
            data_path, 
            device=device, 
            test_ratio=0.2, 
            random_seed=42, 
            add_carbon_tax=add_carbon_tax,
            single_target=single_target
        )
        # 提取测试集的负荷和碳税  
        external_load_scenarios = data_v2.test_inputs * 100  # 转换回MW   
        if len(external_load_scenarios) > num_timesteps:
            external_load_scenarios = external_load_scenarios[:num_timesteps]
        
        # 碳税在preferences_test中
        if not single_target:
            external_carbon_tax_scenarios = data_v2.preferences_test.flatten()
            if len(external_carbon_tax_scenarios) > num_timesteps:
                external_carbon_tax_scenarios = external_carbon_tax_scenarios[:num_timesteps]
        else:
            external_carbon_tax_scenarios = None
        
        # 提取真实标签actions（测试集的targets） 
        ground_truth_vm_va_full = data_v2.y_test.cpu().numpy()  # 转换为numpy数组
        ground_truth_action = data_v2.actions_test if not single_target else None
        
        # 只使用前num_timesteps个样本（与环境步数匹配）
        if len(ground_truth_vm_va_full) >= num_timesteps:
            ground_truth_vm_va = ground_truth_vm_va_full[:num_timesteps] 
            if not single_target:
                ground_truth_action = ground_truth_action[:num_timesteps]
        
        print(f"  - 加载了 {len(external_load_scenarios)} 个测试场景") 
        print("=" * 50)

    # 创建正式环境
    env = PowerGridEnv(num_timesteps=num_timesteps, case_name=case_name, 
                     random_load=random_load if not USE_DATASET_SCENARIOS else False, 
                     run_pp=run_pp,
                     consider_renewable_generation=False,
                     device=device,
                     PowerSystemConfig=ps_config,
                     external_load_scenarios=external_load_scenarios,
                     external_carbon_tax_scenarios=external_carbon_tax_scenarios)

    env_cp = copy.deepcopy(env)
    args['env'] = env
    # 模型路径  
    model_save_path_ss = "models/based_version/pre_train_model.pth"   # ieee 118的监督学习模型 deep-opf-v 

    # 存储所有结果和配置信息
    results = {}
    colors = {}
    markers = {}
    method_names = {}

    # 设置碳税（仅在不使用数据集场景时生效）
    if not USE_DATASET_SCENARIOS:
        carbon_tax_for_test = 0.0   
        env.carbon_tax = carbon_tax_for_test
        print('please pay attention to the carbon_tax_for_test')
        print(f"carbon_tax_for_test: {carbon_tax_for_test}")
    else:
        print('使用数据集场景时，碳税由场景数据决定，不需要手动设置') 

    # 测试监督学习模型
    if ENABLE_SUPERVISED: 
        agent_supervised = PowerGridAgent(env, sigma=0.0, updated_v=True, device=device)
        if 'pth' in model_save_path_ss:
            agent_supervised.actor.load_state_dict(torch.load(model_save_path_ss,  map_location=torch.device("cpu")))
            agent_supervised.target_actor.load_state_dict(torch.load(model_save_path_ss,  map_location=torch.device("cpu")))
        else:
            agent_supervised.actor = torch.load(model_save_path_ss, weights_only=False) 
        results['supervised'] = run_single_model_test(agent_supervised, env, "监督学习模型", debug_mode=False)
        colors['supervised'] = '#1f77b4'
        markers['supervised'] = 'o'
        method_names['supervised'] = '监督学习模型' 
    else:
        agent_supervised = None

    # 测试真实标签actions（如果启用）
    if TEST_GROUND_TRUTH_ACTIONS and USE_DATASET_SCENARIOS and ground_truth_vm_va is not None:
        print("\n=== 开始测试真实标签Actions ===")
        try:
            agent_truth = PowerGridAgent(env, sigma=0.0, updated_v=True, device=device) if agent_supervised is None else agent_supervised
            results['ground_truth'] = run_single_model_test(
                agent_truth, 
                env, 
                "真实标签Actions",
                ground_truth_vm_va=ground_truth_vm_va,
                ground_truth_action=ground_truth_action
            )
            colors['ground_truth'] = '#17becf'    # 青色
            markers['ground_truth'] = 'P'         # 加号形状
            method_names['ground_truth'] = 'Ground Truth'
            print("真实标签测试完成")
        except Exception as e:
            print(f"真实标签测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试流模型（ActorFlow）
    if ENABLE_ACTORFLOW:                                      # main part for testing the flow model
        print("\n=== 开始测试流模型(ActorFlow) ===")
        
        # 流模型配置 - 使用最新训练的模型 (h512_l5, 2025-12-05)
        # 注意: train_main.py 中有 os.chdir 切换到 flow_model 目录，所以路径不需要 flow_model/ 前缀
        flow_model_configs = { 
            'simple': 'models/h512_l5_b512_lr0.001_wd1e-06_wc0.1_cg[True]_ctF_tmst/simple_mlp_separate_training_add_carbon_tax_False_20251205_153351_best.pth',
            'vae': 'models/h512_l5_b512_lr0.001_wd1e-06_wc0.1_cg[True]_ctF_tmst/vae_mlp_separate_training_add_carbon_tax_False_20251205_153826_best.pth',
            'rectified': 'models/h512_l5_b512_lr0.001_wd1e-06_wc0.1_cg[True]_ctF_tmst/rectified_mlp_separate_training_add_carbon_tax_False_20251205_154927_best.pth'
        }

        debug_mode = False

        # 选择要使用的流模型类型
        for selected_flow_type in flow_model_configs.keys(): 
            selected_flow_path = flow_model_configs[selected_flow_type]
            
            print(f"流模型类型: {selected_flow_type}")
            print(f"模型路径: {selected_flow_path}")

            # 这里需要根据模型修改arg
            if selected_flow_type == 'vae':
                args['latent_dim'] = args['latent_dim_vae']  
            else:
                args['latent_dim'] = args['latent_dim_flow'] 
            
            # 创建使用流模型的 agent 
            agent_actorflow = PowerGridAgent(
                env=env,
                sigma=0.0, 
                device=device,
                use_flow_model=True,  # 启用流模型
                flow_model_type=selected_flow_type,
                flow_model_path=selected_flow_path,
                flow_args=args
            )

            print("流模型加载成功，开始测试...")             
            # 运行测试（如果需要分析差异，则返回actions）
            results[selected_flow_type] = run_single_model_test(
                agent_actorflow, 
                env, 
                f"流模型({selected_flow_type})", 
                debug_mode=debug_mode,
                return_actions=ANALYZE_ACTION_DIFFERENCES
            )
            
            # 根据不同的 selected_flow_type 设置不同的颜色和形状
            if selected_flow_type == 'rectified':
                colors[selected_flow_type] = '#ff7f0e'    # 橙色
                markers[selected_flow_type] = 's'         # 方形 
            elif selected_flow_type == 'simple':
                colors[selected_flow_type] = '#2ca02c'    # 绿色
                markers[selected_flow_type] = 'D'         # 菱形
            elif selected_flow_type == 'vae':
                colors[selected_flow_type] = '#9467bd'    # 紫色
                markers[selected_flow_type] = '*'         # 星形
            else:
                colors[selected_flow_type] = '#8c564b'    # 其他
                markers[selected_flow_type] = 'x'         # x 形状

            method_names[selected_flow_type] = f'Flow Model ({selected_flow_type})'
            
            print(f"流模型({selected_flow_type})测试完成") 
    
    # 测试runopp基准
    if ENABLE_RUNOPP:
        try:
            results['runopp'] = run_single_model_test(None, env_cp, "最优潮流求解器(runopp)", use_runopp=True)
            colors['runopp'] = '#2ca02c'
            markers['runopp'] = '^'
            method_names['runopp'] = '最优潮流求解器'
        except Exception as e:
            print(f"最优潮流求解器运行失败: {e}")
            ENABLE_RUNOPP = False

    # 检查是否有有效的结果
    if not results:
        print("错误：没有成功运行任何方法！") 
        exit(1)

    print(f"\n成功运行了 {len(results)} 种方法: {list(method_names.values())}")

    # ==================== 绘制性能对比图表 ====================
    plot_performance_comparison(results, colors, markers, method_names, 
                                ENABLE_SUPERVISED, ENABLE_ACTORFLOW, ENABLE_RUNOPP, TEST_GROUND_TRUTH_ACTIONS)
    
    # ==================== 打印性能统计信息 ====================
    print_performance_statistics(results, method_names)
    
    # ==================== 分析模型预测与真实标签的差异 ====================
    if ANALYZE_ACTION_DIFFERENCES and ground_truth_vm_va is not None:
        analyze_and_visualize_prediction_errors(results, method_names, ground_truth_vm_va)
    
    print("\n" + "="*60)