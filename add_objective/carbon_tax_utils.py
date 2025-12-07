"""
碳税相关的工具函数

用于修改 pandapower 网络的目标函数，将碳成本纳入优化
"""

def update_cost_coefficients_with_carbon_tax(env, carbon_tax=None):
    """
    根据碳税率更新 pandapower 网络的成本系数
    
    这个函数修改 net.poly_cost 中的 cp1_eur_per_mw（线性成本系数），
    将碳成本 (τ·GCI) 加到原始成本系数上：
        cp1_new = cp1_original + carbon_tax × GCI
    
    目标函数变为：min [Σ(c2·P² + c1·P) + Σ(τ·GCI·P)]
    
    Args:
        env: PowerGridEnv 实例
        carbon_tax: 碳税率 ($/tCO2)
                   - 如果为 None，使用 env.carbon_tax
                   - 如果 > 0，添加碳成本
                   - 如果 = 0，恢复原始成本系数
    
    Returns:
        dict: 包含修改信息的字典
            - 'backup_created': 是否创建了备份
            - 'carbon_tax': 使用的碳税率
            - 'modified_gens': 修改的发电机数量
            - 'modified_ext_grids': 修改的外接电源数量
            - 'action': 执行的操作 ('add_carbon_cost', 'restore_original', 'no_change')
    
    Example:
        >>> from env import PowerGridEnv
        >>> env = PowerGridEnv(case_name="case118", carbon_tax=20.0)
        >>> 
        >>> # 添加碳成本
        >>> info = update_cost_coefficients_with_carbon_tax(env)
        >>> print(f"修改了 {info['modified_gens']} 台发电机")
        >>> 
        >>> # 更改碳税率
        >>> info = update_cost_coefficients_with_carbon_tax(env, carbon_tax=30.0)
        >>> 
        >>> # 恢复原始成本
        >>> info = update_cost_coefficients_with_carbon_tax(env, carbon_tax=0.0)
    """
    
    # 确定使用的碳税率
    if carbon_tax is None:
        carbon_tax = env.carbon_tax
    
    # 初始化返回信息
    info = {
        'backup_created': False,
        'carbon_tax': carbon_tax,
        'modified_gens': 0,
        'modified_ext_grids': 0,
        'action': 'no_change'
    }
    
    # 步骤1: 备份原始成本系数（如果还没有备份）
    if 'cp1_original' not in env.net.poly_cost.columns:
        env.net.poly_cost['cp1_original'] = env.net.poly_cost['cp1_eur_per_mw'].copy()
        info['backup_created'] = True
    
    # 步骤2: 根据碳税率决定操作
    if carbon_tax > 0 and 'GCI' in env.net.gen.columns:
        # 情况1: 添加碳成本到目标函数
        info['action'] = 'add_carbon_cost'
        
        # 修改发电机的成本系数
        for gen_idx in env.net.gen.index:
            gci = env.net.gen.loc[gen_idx, 'GCI']
            
            # 查找对应的成本数据
            cost_mask = (env.net.poly_cost['et'] == 'gen') & \
                       (env.net.poly_cost['element'] == gen_idx)
            
            if cost_mask.any():
                # 获取原始成本系数
                cp1_original = env.net.poly_cost.loc[cost_mask, 'cp1_original'].values[0]
                
                # 计算新的成本系数：原始成本 + 碳成本
                cp1_new = cp1_original + carbon_tax * gci
                
                # 更新成本系数
                env.net.poly_cost.loc[cost_mask, 'cp1_eur_per_mw'] = cp1_new
                info['modified_gens'] += 1
        
        # 修改外接电源（ext_grid）的成本系数
        # 仅当 ext_grid 代表内部平衡发电机时才计算碳成本
        if not env.ext_grid_is_external_market and 'GCI' in env.net.ext_grid.columns:
            for ext_idx in env.net.ext_grid.index:
                gci_ext = env.net.ext_grid.loc[ext_idx, 'GCI']
                
                # 仅当 GCI > 0 时才修改成本
                if gci_ext > 0:
                    cost_mask = (env.net.poly_cost['et'] == 'ext_grid') & \
                               (env.net.poly_cost['element'] == ext_idx)
                    
                    if cost_mask.any():
                        cp1_original = env.net.poly_cost.loc[cost_mask, 'cp1_original'].values[0]
                        cp1_new = cp1_original + carbon_tax * gci_ext
                        env.net.poly_cost.loc[cost_mask, 'cp1_eur_per_mw'] = cp1_new
                        info['modified_ext_grids'] += 1
    
    elif carbon_tax == 0 and 'cp1_original' in env.net.poly_cost.columns:
        # 情况2: 碳税为 0，恢复原始成本系数
        info['action'] = 'restore_original'
        env.net.poly_cost['cp1_eur_per_mw'] = env.net.poly_cost['cp1_original'].copy()
    
    return info


def print_cost_modification_summary(info, verbose=True):
    """
    打印成本系数修改的摘要信息
    
    Args:
        info: update_cost_coefficients_with_carbon_tax() 返回的信息字典
        verbose: 是否打印详细信息
    """
    if not verbose:
        return
    
    carbon_tax = info['carbon_tax']
    action = info['action']
    
    if action == 'add_carbon_cost':
        print(f"[INFO] 添加碳成本到目标函数 (碳税率: ${carbon_tax:.2f}/tCO2)")
        print(f"  - 修改了 {info['modified_gens']} 台发电机的成本系数")
        if info['modified_ext_grids'] > 0:
            print(f"  - 修改了 {info['modified_ext_grids']} 个外接电源的成本系数")
    elif action == 'restore_original':
        print(f"[INFO] 恢复原始成本系数 (碳税率: $0/tCO2)")
    else:
        print(f"[INFO] 成本系数未修改 (碳税率: ${carbon_tax:.2f}/tCO2)")
    
    if info['backup_created']:
        print(f"  - 已创建原始成本系数的备份")


def reset_cost_coefficients(env):
    """
    重置成本系数到原始值（如果有备份）
    
    Args:
        env: PowerGridEnv 实例
    
    Returns:
        bool: 是否成功重置
    """
    if 'cp1_original' in env.net.poly_cost.columns:
        env.net.poly_cost['cp1_eur_per_mw'] = env.net.poly_cost['cp1_original'].copy()
        return True
    return False


def get_current_carbon_cost_coefficients(env):
    """
    获取当前的碳成本系数（用于调试和验证）
    
    Args:
        env: PowerGridEnv 实例
    
    Returns:
        dict: 包含发电机和外接电源的碳成本系数信息
    """
    result = {
        'generators': [],
        'ext_grids': [],
        'carbon_tax': env.carbon_tax
    }
    
    # 检查是否有备份
    if 'cp1_original' not in env.net.poly_cost.columns:
        result['error'] = 'No backup found (cp1_original not in poly_cost)'
        return result
    
    # 获取发电机的信息
    if 'GCI' in env.net.gen.columns:
        for gen_idx in env.net.gen.index:
            gci = env.net.gen.loc[gen_idx, 'GCI']
            bus = env.net.gen.loc[gen_idx, 'bus']
            
            cost_mask = (env.net.poly_cost['et'] == 'gen') & \
                       (env.net.poly_cost['element'] == gen_idx)
            
            if cost_mask.any():
                cp1_original = env.net.poly_cost.loc[cost_mask, 'cp1_original'].values[0]
                cp1_current = env.net.poly_cost.loc[cost_mask, 'cp1_eur_per_mw'].values[0]
                carbon_coefficient = cp1_current - cp1_original
                
                result['generators'].append({
                    'gen_idx': gen_idx,
                    'bus': bus,
                    'GCI': gci,
                    'cp1_original': cp1_original,
                    'cp1_current': cp1_current,
                    'carbon_coefficient': carbon_coefficient
                })
    
    # 获取外接电源的信息
    if 'GCI' in env.net.ext_grid.columns:
        for ext_idx in env.net.ext_grid.index:
            gci = env.net.ext_grid.loc[ext_idx, 'GCI']
            bus = env.net.ext_grid.loc[ext_idx, 'bus']
            
            cost_mask = (env.net.poly_cost['et'] == 'ext_grid') & \
                       (env.net.poly_cost['element'] == ext_idx)
            
            if cost_mask.any():
                cp1_original = env.net.poly_cost.loc[cost_mask, 'cp1_original'].values[0]
                cp1_current = env.net.poly_cost.loc[cost_mask, 'cp1_eur_per_mw'].values[0]
                carbon_coefficient = cp1_current - cp1_original
                
                result['ext_grids'].append({
                    'ext_idx': ext_idx,
                    'bus': bus,
                    'GCI': gci,
                    'cp1_original': cp1_original,
                    'cp1_current': cp1_current,
                    'carbon_coefficient': carbon_coefficient
                })
    
    return result

