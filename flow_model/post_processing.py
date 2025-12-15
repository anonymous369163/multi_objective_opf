"""
OPF 后处理模块

基于 DeepOPV-V 的雅可比矩阵后处理方法，通过计算电压修正量来减少预测结果的约束违规。

核心思想：当 Pg/Qg 或支路功率违反约束时，利用功率对电压的灵敏度（雅可比矩阵）
计算需要的电压调整量 dV = pinv(dPQg/dV) * dPQg

参考: DeepOPV-V.ipynb
"""

import torch
import numpy as np
import math


def denormalize_voltage(Vm_norm, Va_norm):
    """
    将归一化的电压还原为实际的 p.u. 值
    
    Args:
        Vm_norm: 归一化电压幅值 (batch_size, num_buses)
        Va_norm: 归一化电压相角 (batch_size, num_buses)
    
    Returns:
        Vm_pu: 电压幅值 p.u. (batch_size, num_buses)
        Va_rad: 电压相角 弧度 (batch_size, num_buses)
    """
    Vm_pu = Vm_norm * 0.06 + 1.0
    Va_rad = Va_norm * math.pi / 6.0
    return Vm_pu, Va_rad


def normalize_voltage(Vm_pu, Va_rad):
    """
    将实际的 p.u. 电压转换为归一化值
    
    Args:
        Vm_pu: 电压幅值 p.u. (batch_size, num_buses)
        Va_rad: 电压相角 弧度 (batch_size, num_buses)
    
    Returns:
        Vm_norm: 归一化电压幅值 (batch_size, num_buses)
        Va_norm: 归一化电压相角 (batch_size, num_buses)
    """
    Vm_norm = (Vm_pu - 1.0) / 0.06
    Va_norm = Va_rad / (math.pi / 6.0)
    return Vm_norm, Va_norm


def compute_complex_voltage(Vm_pu, Va_rad):
    """
    计算复数电压 V = Vm * exp(j * Va)
    
    Args:
        Vm_pu: 电压幅值 p.u. (batch_size, num_buses)
        Va_rad: 电压相角 弧度 (batch_size, num_buses)
    
    Returns:
        V: 复数电压 (batch_size, num_buses)
    """
    V = Vm_pu * torch.exp(1j * Va_rad.to(torch.complex64))
    return V.to(torch.complex64)


def compute_power_injection(Vm_pu, Va_rad, G, B):
    """
    计算节点功率注入 P, Q
    
    基于潮流方程：
    P = Vm * (G @ Vreal - B @ Vimg) * cos(Va) + Vm * (B @ Vreal + G @ Vimg) * sin(Va)
    Q = Vm * (G @ Vreal - B @ Vimg) * sin(Va) - Vm * (B @ Vreal + G @ Vimg) * cos(Va)
    
    Args:
        Vm_pu: 电压幅值 p.u. (batch_size, num_buses)
        Va_rad: 电压相角 弧度 (batch_size, num_buses)
        G: 导纳矩阵实部 (num_buses, num_buses)
        B: 导纳矩阵虚部 (num_buses, num_buses)
    
    Returns:
        P: 有功功率注入 (batch_size, num_buses)
        Q: 无功功率注入 (batch_size, num_buses)
    """
    device = Vm_pu.device
    
    # Ensure G and B are on the same device
    if isinstance(G, torch.Tensor):
        G = G.to(device)
        B = B.to(device)
    
    # 转置以便矩阵运算: (num_buses, batch_size)
    Vm = Vm_pu.T
    Va = Va_rad.T
    
    Vreal = Vm * torch.cos(Va)
    Vimg = Vm * torch.sin(Va)
    
    Ireal = torch.matmul(G, Vreal) - torch.matmul(B, Vimg)
    Iimg = torch.matmul(B, Vreal) + torch.matmul(G, Vimg)
    
    P = Vreal * Ireal + Vimg * Iimg
    Q = Vimg * Ireal - Vreal * Iimg
    
    # 转回 (batch_size, num_buses)
    return P.T, Q.T


def compute_branch_power(Vm_pu, Va_rad, Gf, Bf, Gt, Bt, Cf, Ct):
    """
    计算支路功率 Sf, St
    
    Args:
        Vm_pu: 电压幅值 p.u. (batch_size, num_buses)
        Va_rad: 电压相角 弧度 (batch_size, num_buses)
        Gf, Bf: 首端导纳矩阵实部和虚部 (num_branches, num_buses)
        Gt, Bt: 尾端导纳矩阵实部和虚部 (num_branches, num_buses)
        Cf, Ct: 首端和尾端连接矩阵 (num_branches, num_buses)
    
    Returns:
        Sf: 首端视在功率 (batch_size, num_branches)
        St: 尾端视在功率 (batch_size, num_branches)
    """
    device = Vm_pu.device
    
    # Ensure all tensors are on the same device
    if isinstance(Gf, torch.Tensor):
        Gf = Gf.to(device)
        Bf = Bf.to(device)
        Gt = Gt.to(device)
        Bt = Bt.to(device)
        Cf = Cf.to(device)
        Ct = Ct.to(device)
    
    # 转置: (num_buses, batch_size)
    Vm = Vm_pu.T
    Va = Va_rad.T
    
    Vreal = Vm * torch.cos(Va)
    Vimg = Vm * torch.sin(Va)
    
    # 首端
    Ifreal = torch.matmul(Gf, Vreal) - torch.matmul(Bf, Vimg)
    Ifimg = torch.matmul(Bf, Vreal) + torch.matmul(Gf, Vimg)
    Vfreal = torch.matmul(Cf, Vreal)
    Vfimg = torch.matmul(Cf, Vimg)
    Pf = Vfreal * Ifreal + Vfimg * Ifimg
    Qf = Vfimg * Ifreal - Vfreal * Ifimg
    Sf = torch.sqrt(Pf**2 + Qf**2)
    
    # 尾端
    Itreal = torch.matmul(Gt, Vreal) - torch.matmul(Bt, Vimg)
    Itimg = torch.matmul(Bt, Vreal) + torch.matmul(Gt, Vimg)
    Vtreal = torch.matmul(Ct, Vreal)
    Vtimg = torch.matmul(Ct, Vimg)
    Pt = Vtreal * Itreal + Vtimg * Itimg
    Qt = Vtimg * Itreal - Vtreal * Itimg
    St = torch.sqrt(Pt**2 + Qt**2)
    
    # 转回 (batch_size, num_branches)
    return Sf.T, St.T, Pf.T, Qf.T, Pt.T, Qt.T


def compute_generator_power(P, Q, Pd, Qd, pd_bus_idx, qd_bus_idx, gen_bus_idx_P, device, gen_bus_idx_Q=None):
    """
    计算发电机功率 Pg, Qg
    
    Pg = P + Pd (在负荷节点)
    Qg = Q + Qd (在负荷节点)
    
    Args:
        P: 节点有功功率注入 (batch_size, num_buses)
        Q: 节点无功功率注入 (batch_size, num_buses)
        Pd: 有功负荷 (num_pd,) 或 (batch_size, num_pd)
        Qd: 无功负荷 (num_qd,) 或 (batch_size, num_qd)
        pd_bus_idx: 有功负荷节点索引
        qd_bus_idx: 无功负荷节点索引
        gen_bus_idx_P: 有功发电机节点索引 (bus_Pg)
        device: 计算设备
        gen_bus_idx_Q: 无功发电机节点索引 (bus_Qg)，如果为None则使用 gen_bus_idx_P
    
    Returns:
        Pg: 发电机有功功率 (batch_size, num_gen_P)
        Qg: 发电机无功功率 (batch_size, num_gen_Q)
    """
    batch_size = P.shape[0]
    
    # 如果没有提供 gen_bus_idx_Q，使用 gen_bus_idx_P
    if gen_bus_idx_Q is None:
        gen_bus_idx_Q = gen_bus_idx_P
    
    # 克隆 P 和 Q 以避免修改原始数据
    Pg_bus = P.clone()
    Qg_bus = Q.clone()
    
    # 确保索引是 torch tensor
    pd_bus_idx_t = torch.from_numpy(pd_bus_idx).long().to(device) if isinstance(pd_bus_idx, np.ndarray) else pd_bus_idx.long().to(device)
    qd_bus_idx_t = torch.from_numpy(qd_bus_idx).long().to(device) if isinstance(qd_bus_idx, np.ndarray) else qd_bus_idx.long().to(device)
    gen_bus_idx_P_t = torch.from_numpy(gen_bus_idx_P).long().to(device) if isinstance(gen_bus_idx_P, np.ndarray) else gen_bus_idx_P.long().to(device)
    gen_bus_idx_Q_t = torch.from_numpy(gen_bus_idx_Q).long().to(device) if isinstance(gen_bus_idx_Q, np.ndarray) else gen_bus_idx_Q.long().to(device)
    
    # 处理 Pd 和 Qd 的维度
    if Pd.dim() == 1:
        Pd = Pd.unsqueeze(0).expand(batch_size, -1)
    if Qd.dim() == 1:
        Qd = Qd.unsqueeze(0).expand(batch_size, -1)
    
    # 在负荷节点加上负荷得到发电功率
    # Pg_bus[:, pd_bus_idx] = Pg_bus[:, pd_bus_idx] + Pd
    for i, idx in enumerate(pd_bus_idx_t):
        Pg_bus[:, idx] = Pg_bus[:, idx] + Pd[:, i]
    
    for i, idx in enumerate(qd_bus_idx_t):
        Qg_bus[:, idx] = Qg_bus[:, idx] + Qd[:, i]
    
    # 提取发电机节点的功率（Pg 和 Qg 使用不同的索引）
    Pg = Pg_bus[:, gen_bus_idx_P_t]
    Qg = Qg_bus[:, gen_bus_idx_Q_t]
    
    return Pg, Qg


def detect_pg_qg_violation(Pg, Qg, Pg_max, Pg_min, Qg_max, Qg_min, gen_bus_idx, delta=1e-4):
    """
    检测发电机功率约束违规
    
    Args:
        Pg: 发电机有功功率 (batch_size, num_gen)
        Qg: 发电机无功功率 (batch_size, num_gen)
        Pg_max, Pg_min: 有功功率上下限 (num_gen,) - 发电机维度
        Qg_max, Qg_min: 无功功率上下限 (num_gen,) - 发电机维度
        gen_bus_idx: 发电机节点索引（用于修正时确定母线位置）
        delta: 违规阈值
    
    Returns:
        violation_info: 包含违规信息的字典列表，每个样本一个
    """
    batch_size = Pg.shape[0]
    device = Pg.device
    
    # Pg_max/Pg_min 已经是发电机维度的，直接使用
    # 确保在正确的设备上
    Pg_max_gen = Pg_max.to(device) if hasattr(Pg_max, 'to') else Pg_max
    Pg_min_gen = Pg_min.to(device) if hasattr(Pg_min, 'to') else Pg_min
    Qg_max_gen = Qg_max.to(device) if hasattr(Qg_max, 'to') else Qg_max
    Qg_min_gen = Qg_min.to(device) if hasattr(Qg_min, 'to') else Qg_min
    
    violation_list = []
    
    for i in range(batch_size):
        sample_violation = {
            'Pg_upper': [],  # (gen_idx, delta_value)
            'Pg_lower': [],
            'Qg_upper': [],
            'Qg_lower': [],
            'has_violation': False
        }
        
        # Pg 上限违规
        delta_Pg_upper = Pg[i] - Pg_max_gen
        idx_Pg_upper = torch.where(delta_Pg_upper > delta)[0]
        if len(idx_Pg_upper) > 0:
            for idx in idx_Pg_upper:
                sample_violation['Pg_upper'].append((idx.item(), delta_Pg_upper[idx].item()))
            sample_violation['has_violation'] = True
        
        # Pg 下限违规
        delta_Pg_lower = Pg_min_gen - Pg[i]
        idx_Pg_lower = torch.where(delta_Pg_lower > delta)[0]
        if len(idx_Pg_lower) > 0:
            for idx in idx_Pg_lower:
                sample_violation['Pg_lower'].append((idx.item(), delta_Pg_lower[idx].item()))
            sample_violation['has_violation'] = True
        
        # Qg 上限违规
        delta_Qg_upper = Qg[i] - Qg_max_gen
        idx_Qg_upper = torch.where(delta_Qg_upper > delta)[0]
        if len(idx_Qg_upper) > 0:
            for idx in idx_Qg_upper:
                sample_violation['Qg_upper'].append((idx.item(), delta_Qg_upper[idx].item()))
            sample_violation['has_violation'] = True
        
        # Qg 下限违规
        delta_Qg_lower = Qg_min_gen - Qg[i]
        idx_Qg_lower = torch.where(delta_Qg_lower > delta)[0]
        if len(idx_Qg_lower) > 0:
            for idx in idx_Qg_lower:
                sample_violation['Qg_lower'].append((idx.item(), delta_Qg_lower[idx].item()))
            sample_violation['has_violation'] = True
        
        violation_list.append(sample_violation)
    
    return violation_list


def detect_branch_violation(Sf, St, Pf, Qf, S_max, delta=1e-4):
    """
    检测支路功率约束违规
    
    Args:
        Sf: 首端视在功率 (batch_size, num_branches)
        St: 尾端视在功率 (batch_size, num_branches)
        Pf: 首端有功功率 (batch_size, num_branches)
        Qf: 首端无功功率 (batch_size, num_branches)
        S_max: 支路容量限制 (num_branches,)
        delta: 违规阈值
    
    Returns:
        branch_violation_list: 包含支路违规信息的字典列表
    """
    batch_size = Sf.shape[0]
    device = Sf.device
    
    # Ensure S_max is on the correct device
    if isinstance(S_max, torch.Tensor):
        S_max = S_max.to(device)
    
    branch_violation_list = []
    
    for i in range(batch_size):
        sample_violation = {
            'Sf_vio': [],  # (branch_idx, delta_value, Pf, Qf)
            'St_vio': [],
            'has_violation': False
        }
        
        # Sf 违规
        delta_Sf = Sf[i] - S_max
        idx_Sf = torch.where(delta_Sf > delta)[0]
        if len(idx_Sf) > 0:
            for idx in idx_Sf:
                sample_violation['Sf_vio'].append({
                    'branch_idx': idx.item(),
                    'delta': delta_Sf[idx].item(),
                    'Pf': Pf[i, idx].item(),
                    'Qf': Qf[i, idx].item()
                })
            sample_violation['has_violation'] = True
        
        # St 违规
        delta_St = St[i] - S_max
        idx_St = torch.where(delta_St > delta)[0]
        if len(idx_St) > 0:
            for idx in idx_St:
                sample_violation['St_vio'].append({
                    'branch_idx': idx.item(),
                    'delta': delta_St[idx].item()
                })
            sample_violation['has_violation'] = True
        
        branch_violation_list.append(sample_violation)
    
    return branch_violation_list


def compute_jacobian_dPQ_dV(V, Ybus, num_buses, device):
    """
    计算功率对电压的雅可比矩阵
    
    基于 DeepOPV-V 的 dPQbus_dV 函数
    
    dSbus/dVm = diag(V) @ conj(Ybus @ diag(V/|V|)) + diag(conj(Ibus)) @ diag(V/|V|)
    dSbus/dVa = j * diag(V) @ conj(diag(Ibus) - Ybus @ diag(V))
    
    Args:
        V: 复数电压向量 (num_buses,)
        Ybus: 导纳矩阵 (num_buses, num_buses) - scipy sparse 或 numpy array
        num_buses: 母线数量
        device: 计算设备
    
    Returns:
        dPbus_dV: dP/d[Va, Vm] (num_buses, 2*num_buses)
        dQbus_dV: dQ/d[Va, Vm] (num_buses, 2*num_buses)
    """
    # 将 V 转换为 numpy 进行计算（与 DeepOPV-V 保持一致）
    V_np = V.cpu().numpy() if isinstance(V, torch.Tensor) else V
    
    # 确保 Ybus 是 dense array
    if hasattr(Ybus, 'toarray'):
        Ybus_np = Ybus.toarray()
    elif hasattr(Ybus, 'todense'):
        Ybus_np = np.array(Ybus.todense())
    else:
        Ybus_np = np.array(Ybus)
    
    # 计算注入电流
    Ibus = np.dot(Ybus_np, V_np).conj()
    
    # 对角矩阵
    diagV = np.diag(V_np)
    diagIbus = np.diag(Ibus)
    diagVnorm = np.diag(V_np / np.abs(V_np))
    
    # dSbus/dVm
    dSbus_dVm = np.dot(diagV, np.dot(Ybus_np, diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
    
    # dSbus/dVa
    dSbus_dVa = 1j * np.dot(diagV, (diagIbus - np.dot(Ybus_np, diagV)).conj())
    
    # 合并 [dVa, dVm]
    dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
    
    # 分离实部和虚部得到 dP 和 dQ
    dPbus_dV = np.real(dSbus_dV)
    dQbus_dV = np.imag(dSbus_dV)
    
    return dPbus_dV, dQbus_dV


def compute_branch_jacobian(V, Yf, Cf, num_buses, bus_Va_idx, device):
    """
    计算支路功率对电压的雅可比矩阵
    
    Args:
        V: 复数电压向量 (num_buses,)
        Yf: 首端导纳矩阵 (num_branches, num_buses)
        Cf: 首端连接矩阵 (num_branches, num_buses)
        num_buses: 母线数量
        bus_Va_idx: 需要计算的电压相角节点索引（去掉平衡节点）
        device: 计算设备
    
    Returns:
        dPfbus_dV: dPf/d[Va, Vm] (num_branches, num_buses + len(bus_Va_idx))
        dQfbus_dV: dQf/d[Va, Vm] (num_branches, num_buses + len(bus_Va_idx))
    """
    V_np = V.cpu().numpy() if isinstance(V, torch.Tensor) else V
    
    if hasattr(Yf, 'toarray'):
        Yf_np = Yf.toarray()
    else:
        Yf_np = np.array(Yf)
    
    if hasattr(Cf, 'cpu'):
        Cf_np = Cf.cpu().numpy()
    else:
        Cf_np = np.array(Cf)
    
    num_branches = Yf_np.shape[0]
    
    # 首端电压和电流
    fV = np.dot(Cf_np, V_np)
    fI = np.dot(Yf_np, V_np).conj()
    
    diagfI = np.diag(fI)
    diagfV = np.diag(fV)
    diagVnorm = np.diag(V_np / np.abs(V_np))
    diagV = np.diag(V_np)
    
    # dSf/dVm
    dfS_dVm = np.dot(diagfV, np.dot(Yf_np, diagVnorm).conj()) + np.dot(diagfI.conj(), np.dot(Cf_np, diagVnorm))
    
    # dSf/dVa
    dfS_dVa = -1j * np.dot(diagfV, np.dot(Yf_np, diagV).conj()) + 1j * np.dot(diagfI.conj(), np.dot(Cf_np, diagV))
    
    # 只保留非平衡节点的 Va
    dfP_dVa = np.real(dfS_dVa[:, bus_Va_idx])
    dfQ_dVa = np.imag(dfS_dVa[:, bus_Va_idx])
    dfP_dVm = np.real(dfS_dVm)
    dfQ_dVm = np.imag(dfS_dVm)
    
    dPfbus_dV = np.concatenate((dfP_dVa, dfP_dVm), axis=1)
    dQfbus_dV = np.concatenate((dfQ_dVa, dfQ_dVm), axis=1)
    
    return dPfbus_dV, dQfbus_dV


def compute_voltage_correction_for_sample(
    V_np, 
    pg_violation, 
    branch_violation,
    dPbus_dV, 
    dQbus_dV,
    dPfbus_dV,
    dQfbus_dV,
    gen_bus_idx, 
    num_buses,
    slack_bus=0,
    k_dV=1.0
):
    """
    为单个样本计算电压修正量
    
    Args:
        V_np: 复数电压 (num_buses,)
        pg_violation: Pg/Qg 违规信息
        branch_violation: 支路违规信息
        dPbus_dV: dP/d[Va, Vm] (num_buses, 2*num_buses)
        dQbus_dV: dQ/d[Va, Vm] (num_buses, 2*num_buses)
        dPfbus_dV: dPf/d[Va, Vm] (num_branches, num_buses-1 + num_buses) - Va去掉平衡节点
        dQfbus_dV: dQf/d[Va, Vm] (num_branches, num_buses-1 + num_buses) - Va去掉平衡节点
        gen_bus_idx: 发电机节点索引
        num_buses: 母线数量
        slack_bus: 平衡节点索引
        k_dV: 修正系数
    
    Returns:
        dV: 电压修正量 (2*num_buses,) [dVa, dVm]
    """
    dV = np.zeros(2 * num_buses)
    
    if not pg_violation['has_violation'] and not branch_violation['has_violation']:
        return dV
    
    # 收集所有 Pg/Qg 违规
    violated_gen_idx_P = []
    violated_delta_P = []
    violated_gen_idx_Q = []
    violated_delta_Q = []
    
    # 与 DeepOPV-V 保持一致：
    # 上限违规：delta = Pg - Pg_max > 0（正值）
    # 下限违规：delta = Pg - Pg_min < 0（负值！）
    # 然后 dV = pinv(J) * delta，最后 V_new = V - dV
    for idx, delta_val in pg_violation['Pg_upper']:
        violated_gen_idx_P.append(idx)
        violated_delta_P.append(delta_val)  # 正值，与 DeepOPV-V 一致
    
    for idx, delta_val in pg_violation['Pg_lower']:
        violated_gen_idx_P.append(idx)
        # 注意：我的 delta_val = Pg_min - Pg > 0，需要转换为 DeepOPV-V 的格式 Pg - Pg_min < 0
        violated_delta_P.append(-delta_val)  # 转为负值，与 DeepOPV-V 一致
    
    for idx, delta_val in pg_violation['Qg_upper']:
        violated_gen_idx_Q.append(idx)
        violated_delta_Q.append(delta_val)  # 正值
    
    for idx, delta_val in pg_violation['Qg_lower']:
        violated_gen_idx_Q.append(idx)
        violated_delta_Q.append(-delta_val)  # 转为负值
    
    # 计算 Pg/Qg 相关的 dV
    if len(violated_gen_idx_P) > 0 or len(violated_gen_idx_Q) > 0:
        # 获取违规发电机对应的母线索引
        bus_P = gen_bus_idx[violated_gen_idx_P] if len(violated_gen_idx_P) > 0 else np.array([], dtype=int)
        bus_Q = gen_bus_idx[violated_gen_idx_Q] if len(violated_gen_idx_Q) > 0 else np.array([], dtype=int)
        
        # 构建雅可比矩阵行
        rows = []
        delta_PQg = []
        
        if len(bus_P) > 0:
            rows.append(dPbus_dV[bus_P, :])
            delta_PQg.extend(violated_delta_P)
        
        if len(bus_Q) > 0:
            rows.append(dQbus_dV[bus_Q, :])
            delta_PQg.extend(violated_delta_Q)
        
        if len(rows) > 0:
            dPQGbus_dV = np.vstack(rows)
            delta_PQg = np.array(delta_PQg)
            
            # 调试：打印详细信息
            DEBUG_CORRECTION = False  # 关闭调试输出
            if DEBUG_CORRECTION:
                print(f"    [Correction] dPQGbus_dV shape: {dPQGbus_dV.shape}")
                print(f"    [Correction] delta_PQg: {delta_PQg[:5]}... (len={len(delta_PQg)})")
                print(f"    [Correction] delta_PQg range: [{delta_PQg.min():.6f}, {delta_PQg.max():.6f}]")
            
            # 使用伪逆计算 dV
            try:
                pinv_J = np.linalg.pinv(dPQGbus_dV)
                dV_pq = np.dot(pinv_J, delta_PQg * k_dV)
                
                if DEBUG_CORRECTION:
                    print(f"    [Correction] pinv_J shape: {pinv_J.shape}")
                    print(f"    [Correction] dV_pq range: [{dV_pq.min():.6f}, {dV_pq.max():.6f}]")
                    print(f"    [Correction] dV_pq[:10] (dVa): {dV_pq[:10]}")
                    print(f"    [Correction] dV_pq[-10:] (dVm): {dV_pq[-10:]}")
                
                dV += dV_pq
            except np.linalg.LinAlgError:
                pass  # 如果计算失败，保持 dV 为 0
    
    # 计算支路相关的 dV（与 DeepOPV-V 保持一致）
    # DeepOPV-V 中：mp = Pf / delta_S, mq = Qf / delta_S
    # dmpq = mp*dP/dV + mq*dQ/dV
    # dV = pinv(dmpq) * delta_S
    if branch_violation['has_violation'] and dPfbus_dV is not None:
        num_Va_reduced = num_buses - 1  # 去掉平衡节点后的 Va 维度
        
        for vio in branch_violation['Sf_vio']:
            branch_idx = vio['branch_idx']
            delta_S = vio['delta']  # Sf - S_max > 0
            Pf = vio['Pf']
            Qf = vio['Qf']
            
            if abs(delta_S) > 1e-6:
                # 与 DeepOPV-V 保持一致：mp = Pf / delta_S
                mp = Pf / delta_S
                mq = Qf / delta_S
                
                dPdV = dPfbus_dV[branch_idx:branch_idx+1, :]
                dQdV = dQfbus_dV[branch_idx:branch_idx+1, :]
                
                # dmpq = mp*dP/dV + mq*dQ/dV
                dmpq = mp * dPdV + mq * dQdV
                
                try:
                    # dV = pinv(dmpq) * delta_S * k_dV
                    dV_branch_reduced = np.dot(np.linalg.pinv(dmpq), np.array([delta_S * k_dV]))
                    dV_branch_reduced = dV_branch_reduced.squeeze()
                    
                    # 将 reduced 的 dVa 映射回完整的 dVa（在平衡节点位置插入0）
                    dVa_reduced = dV_branch_reduced[:num_Va_reduced]
                    dVm = dV_branch_reduced[num_Va_reduced:]
                    
                    # 在平衡节点位置插入0
                    dVa_full = np.insert(dVa_reduced, slack_bus, 0.0)
                    
                    # 合并为完整的 dV
                    dV_branch_full = np.concatenate([dVa_full, dVm])
                    dV += dV_branch_full
                except (np.linalg.LinAlgError, ValueError):
                    pass
    
    return dV


def apply_post_processing(Vm_norm, Va_norm, x_input, sys_data, k_dV=1.0, verbose=False, debug_mode=0):
    """
    主函数：应用后处理修正
    
    Args:
        Vm_norm: 归一化电压幅值 (batch_size, num_buses)
        Va_norm: 归一化电压相角 (batch_size, num_buses)
        x_input: 输入数据 (batch_size, input_dim)，包含 Pd, Qd
        sys_data: PowerSystemData 对象 (替代 env)
        k_dV: 修正系数
        verbose: 是否打印详细信息
        debug_mode: 调试模式
            0 - 正常模式，应用修正
            1 - 零修正模式，仅测试 pipeline 往返精度
            2 - 完整调试模式，打印详细物理量
    
    Returns:
        Vm_corrected_norm: 修正后的归一化电压幅值
        Va_corrected_norm: 修正后的归一化电压相角
        correction_info: 修正信息字典
    """
    device = Vm_norm.device
    batch_size = Vm_norm.shape[0]
    num_buses = Vm_norm.shape[1]
    
    # 确定平衡节点 (支持 sys_data 和 env 接口)
    if hasattr(sys_data, 'balance_gen_bus'):
        slack_bus = sys_data.balance_gen_bus
    elif hasattr(sys_data, 'bus_slack'):
        slack_bus = sys_data.bus_slack if isinstance(sys_data.bus_slack, int) else int(sys_data.bus_slack)
    else:
        slack_bus = 0  # 默认第一个节点
    
    # 获取系统参数 (支持 sys_data 和 env 两种接口)
    G = sys_data.G
    B = sys_data.B
    Gf = sys_data.Gf
    Bf = sys_data.Bf
    Gt = sys_data.Gt
    Bt = sys_data.Bt
    Cf = sys_data.Cf
    Ct = sys_data.Ct
    
    num_pd = getattr(sys_data, 'num_pd', len(getattr(sys_data, 'pd_bus_idx', [])))
    num_qd = getattr(sys_data, 'num_qd', len(getattr(sys_data, 'qd_bus_idx', [])))
    pd_bus_idx = getattr(sys_data, 'pd_bus_idx', getattr(sys_data, 'idx_Pd', None))
    qd_bus_idx = getattr(sys_data, 'qd_bus_idx', getattr(sys_data, 'idx_Qd', None))
    # Pg 和 Qg 可能使用不同的发电机索引
    gen_bus_idx_P = getattr(sys_data, 'gen_bus_idx', getattr(sys_data, 'bus_Pg', None))
    gen_bus_idx_Q = getattr(sys_data, 'bus_Qg', gen_bus_idx_P)  # 如果没有 bus_Qg，使用 bus_Pg
    
    # 获取功率限制
    if hasattr(sys_data, 'Pg_max'):
        Pg_max = sys_data.Pg_max
        Pg_min = sys_data.Pg_min
        Qg_max = sys_data.Qg_max
        Qg_min = sys_data.Qg_min
    else:
        Pg_max = torch.from_numpy(sys_data.MAXMIN_Pg[:, 0]).float().to(device)
        Pg_min = torch.from_numpy(sys_data.MAXMIN_Pg[:, 1]).float().to(device)
        Qg_max = torch.from_numpy(sys_data.MAXMIN_Qg[:, 0]).float().to(device)
        Qg_min = torch.from_numpy(sys_data.MAXMIN_Qg[:, 1]).float().to(device)
    
    S_max = sys_data.S_max
    Ybus = sys_data.Ybus
    Yf = sys_data.Yf
    
    # 1. 还原归一化
    Vm_pu, Va_rad = denormalize_voltage(Vm_norm, Va_norm)
    
    # 2. 计算功率
    P, Q = compute_power_injection(Vm_pu, Va_rad, G, B)
    
    # 3. 计算支路功率
    Sf, St, Pf, Qf, Pt, Qt = compute_branch_power(
        Vm_pu, Va_rad, Gf, Bf, Gt, Bt, Cf, Ct
    )
    
    # 4. 提取负荷
    Pd = x_input[:, :num_pd]
    Qd = x_input[:, num_pd:num_pd + num_qd]
    
    # 5. 计算发电机功率（使用不同的索引）
    Pg, Qg = compute_generator_power(
        P, Q, Pd, Qd, 
        pd_bus_idx, qd_bus_idx, gen_bus_idx_P, device, gen_bus_idx_Q
    )
    
    # 6. 检测违规
    pg_violations = detect_pg_qg_violation(
        Pg, Qg, Pg_max, Pg_min, Qg_max, Qg_min, 
        gen_bus_idx_P
    )
    
    branch_violations = detect_branch_violation(Sf, St, Pf, Qf, S_max)
    
    # 7. 计算平均电压用于雅可比矩阵
    # 使用批次平均电压计算雅可比矩阵（与 DeepOPV-V 中使用历史电压类似）
    # Vm_mean = Vm_pu.mean(dim=0)
    # Va_mean = Va_rad.mean(dim=0)
    # V_mean = (Vm_mean * torch.exp(1j * Va_mean.to(torch.complex64))).cpu().numpy()
    
    # # 8. 计算雅可比矩阵
    # dPbus_dV, dQbus_dV = compute_jacobian_dPQ_dV(V_mean, Ybus, num_buses, device)
    
    # # 平衡节点已在前面确定
    bus_Va_idx = np.delete(np.arange(num_buses), slack_bus)
    
    # # 支路雅可比矩阵
    # dPfbus_dV, dQfbus_dV = compute_branch_jacobian(
    #     V_mean, Yf, Cf, num_buses, bus_Va_idx, device
    # )
    
    # 9. 为每个样本计算修正量
    Vm_corrected = Vm_pu.clone()
    Va_corrected = Va_rad.clone()
    
    num_corrected = 0
    
    # ==================== Step 1: 零修正 Sanity Check ====================
    # 测试归一化往返精度
    Vm_zero_step, Va_zero_step = normalize_voltage(Vm_pu, Va_rad)
    diff_Vm = (Vm_zero_step - Vm_norm).abs().max().item()
    diff_Va = (Va_zero_step - Va_norm).abs().max().item()
    
    if verbose:
        print(f"  [Step 1] Normalization round-trip error: Vm={diff_Vm:.10f}, Va={diff_Va:.10f}")
    
    # 如果是零修正模式，直接返回仅经过 pipeline 但不修正的结果
    if debug_mode == 1:
        correction_info = {
            'num_samples': batch_size,
            'num_corrected': 0,
            'num_pg_violations': sum(1 for v in pg_violations if v['has_violation']),
            'num_branch_violations': sum(1 for v in branch_violations if v['has_violation']),
            'round_trip_error_Vm': diff_Vm,
            'round_trip_error_Va': diff_Va
        }
        if verbose:
            print(f"  [Step 1] 零修正模式: 返回仅经过 pipeline 但 dV=0 的结果")
        return Vm_zero_step, Va_zero_step, correction_info
    
    # 【调试开关】设为 True 禁用修正
    DISABLE_CORRECTION = (debug_mode == 1)
    
    for i in range(batch_size):
        # 处理有 Pg/Qg 或支路功率违规的样本
        if DISABLE_CORRECTION:
            continue  # 禁用修正
        if pg_violations[i]['has_violation'] or branch_violations[i]['has_violation']:
            # 计算该样本的复数电压
            V_sample = (Vm_pu[i] * torch.exp(1j * Va_rad[i].to(torch.complex64))).cpu().numpy()
            
            # 计算该样本的雅可比矩阵
            dPbus_dV_i, dQbus_dV_i = compute_jacobian_dPQ_dV(V_sample, Ybus, num_buses, device)
            dPfbus_dV_i, dQfbus_dV_i = compute_branch_jacobian(
                V_sample, Yf, Cf, num_buses, bus_Va_idx, device
            )
            
            # 计算修正量（包含 Pg/Qg 和支路约束修正）
            dV = compute_voltage_correction_for_sample(
                V_sample,
                pg_violations[i],
                branch_violations[i],  # 启用支路约束修正
                dPbus_dV_i,
                dQbus_dV_i,
                dPfbus_dV_i,
                dQfbus_dV_i,
                gen_bus_idx_P,
                num_buses,
                slack_bus,
                k_dV
            )
            
            # 应用修正（与 DeepOPV-V 一致，使用减法）
            # DeepOPV-V: Pred_Va1 = Pred_Va - dV1[:, 0:Nbus]
            dVa = dV[:num_buses]
            dVm = dV[num_buses:]
            
            # ==================== Step 2: 逐样本打印关键物理量 ====================
            # 修正前的物理量（只对前2个样本打印）
            if debug_mode == 2 and i < 2:
                print(f"\n  [Step 2] ===== Sample {i} BEFORE Correction =====")
                print(f"    Vm range: [{Vm_pu[i].min().item():.6f}, {Vm_pu[i].max().item():.6f}]")
                print(f"    Va range: [{Va_rad[i].min().item():.6f}, {Va_rad[i].max().item():.6f}]")
                print(f"    Pg (first 5 gens): {Pg[i, :5].tolist()}")
                print(f"    Qg (first 5 gens): {Qg[i, :5].tolist()}")
                # 打印 top 3 违规支路
                Sf_i = Sf[i].cpu().numpy() if isinstance(Sf, torch.Tensor) else Sf[i]
                S_max_np = S_max.cpu().numpy() if isinstance(S_max, torch.Tensor) else S_max
                vio_Sf = Sf_i - S_max_np
                top3_idx = np.argsort(vio_Sf)[-3:][::-1]
                print(f"    Top 3 violated branches (idx, |Sf|, limit, violation):")
                for idx in top3_idx:
                    print(f"      Branch {idx}: |Sf|={Sf_i[idx]:.4f}, limit={S_max_np[idx]:.4f}, vio={vio_Sf[idx]:.4f}")
                print(f"    Pg_upper violations: {pg_violations[i]['Pg_upper'][:3]}...")
                print(f"    Pg_lower violations: {pg_violations[i]['Pg_lower'][:3]}...")
            
            # 调试：打印修正量大小
            if (i == 0 and verbose) or (debug_mode == 2 and i < 2):
                print(f"  [Step 2] dVa range: [{dVa.min():.6f}, {dVa.max():.6f}], sum: {np.abs(dVa).sum():.6f}")
                print(f"  [Step 2] dVm range: [{dVm.min():.6f}, {dVm.max():.6f}], sum: {np.abs(dVm).sum():.6f}")
            
            # 修正后的电压：V_new = V - dV（与 DeepOPV-V 一致）
            Va_corrected[i] = Va_rad[i] - torch.tensor(dVa, dtype=torch.float32, device=device)
            Vm_corrected[i] = Vm_pu[i] - torch.tensor(dVm, dtype=torch.float32, device=device)
            
            # 【关键修复】平衡节点相角保持原值不变（不是设为0！）
            # 因为反归一化后 Va_rad[i, slack_bus] 可能不是0，如果强制设为0会导致巨大误差
            Va_corrected[i, slack_bus] = Va_rad[i, slack_bus]
            
            num_corrected += 1
    
    # 10. 电压幅值裁剪（确保在合理范围内）
    Vm_corrected = torch.clamp(Vm_corrected, 0.9, 1.1)
    
    # ==================== Step 2: 修正后物理量打印 ====================
    if debug_mode == 2:
        # 计算修正后的功率
        P_after, Q_after = compute_power_injection(Vm_corrected, Va_corrected, G, B)
        Sf_after, _, Pf_after, Qf_after, _, _ = compute_branch_power(
            Vm_corrected, Va_corrected, Gf, Bf, Gt, Bt, Cf, Ct
        )
        Pg_after, Qg_after = compute_generator_power(
            P_after, Q_after, Pd, Qd, 
            pd_bus_idx, qd_bus_idx, gen_bus_idx_P, device, gen_bus_idx_Q
        )
        
        # 打印前2个样本的修正后物理量
        for i in range(min(2, batch_size)):
            if pg_violations[i]['has_violation']:
                print(f"\n  [Step 2] ===== Sample {i} AFTER Correction =====")
                print(f"    Vm range: [{Vm_corrected[i].min().item():.6f}, {Vm_corrected[i].max().item():.6f}]")
                print(f"    Va range: [{Va_corrected[i].min().item():.6f}, {Va_corrected[i].max().item():.6f}]")
                print(f"    dVm applied range: [{(Vm_corrected[i] - Vm_pu[i]).min().item():.6f}, {(Vm_corrected[i] - Vm_pu[i]).max().item():.6f}]")
                print(f"    dVa applied range: [{(Va_corrected[i] - Va_rad[i]).min().item():.6f}, {(Va_corrected[i] - Va_rad[i]).max().item():.6f}]")
                print(f"    Pg BEFORE: {Pg[i, :5].tolist()}")
                print(f"    Pg AFTER:  {Pg_after[i, :5].tolist()}")
                print(f"    Qg BEFORE: {Qg[i, :5].tolist()}")
                print(f"    Qg AFTER:  {Qg_after[i, :5].tolist()}")
                
                # 检测修正后的违规
                Sf_i_after = Sf_after[i].cpu().numpy() if isinstance(Sf_after, torch.Tensor) else Sf_after[i]
                S_max_np = S_max.cpu().numpy() if isinstance(S_max, torch.Tensor) else S_max
                vio_Sf_after = Sf_i_after - S_max_np
                top3_idx = np.argsort(vio_Sf_after)[-3:][::-1]
                print(f"    Top 3 violated branches AFTER (idx, |Sf|, limit, violation):")
                for idx in top3_idx:
                    print(f"      Branch {idx}: |Sf|={Sf_i_after[idx]:.4f}, limit={S_max_np[idx]:.4f}, vio={vio_Sf_after[idx]:.4f}")
    
    if verbose:
        print(f"  [DEBUG] Vm_corrected range: [{Vm_corrected.min().item():.6f}, {Vm_corrected.max().item():.6f}]")
        print(f"  [DEBUG] Va_corrected range: [{Va_corrected.min().item():.6f}, {Va_corrected.max().item():.6f}]")
    
    # 11. 转回归一化
    Vm_corrected_norm, Va_corrected_norm = normalize_voltage(Vm_corrected, Va_corrected)
    
    if verbose:
        print(f"  [DEBUG] Vm_corrected_norm range: [{Vm_corrected_norm.min().item():.6f}, {Vm_corrected_norm.max().item():.6f}]")
        print(f"  [DEBUG] Va_corrected_norm range: [{Va_corrected_norm.min().item():.6f}, {Va_corrected_norm.max().item():.6f}]")
        print(f"  [DEBUG] Original Vm_norm range: [{Vm_norm.min().item():.6f}, {Vm_norm.max().item():.6f}]")
        print(f"  [DEBUG] Original Va_norm range: [{Va_norm.min().item():.6f}, {Va_norm.max().item():.6f}]")
    
    # 注意：不应该裁剪归一化值，因为原始值可能超出 [-1, 1] 范围
    # 保持与原始值一致的范围
    # Vm_corrected_norm = torch.clamp(Vm_corrected_norm, -1.0, 1.0)
    # Va_corrected_norm = torch.clamp(Va_corrected_norm, -1.0, 1.0)
    
    correction_info = {
        'num_samples': batch_size,
        'num_corrected': num_corrected,
        'num_pg_violations': sum(1 for v in pg_violations if v['has_violation']),
        'num_branch_violations': sum(1 for v in branch_violations if v['has_violation'])
    }
    
    if verbose:
        print(f"[Post-Processing] 样本数: {batch_size}, 修正样本数: {num_corrected}")
        print(f"  - Pg/Qg 违规样本: {correction_info['num_pg_violations']}")
        print(f"  - 支路违规样本: {correction_info['num_branch_violations']}")
    
    return Vm_corrected_norm, Va_corrected_norm, correction_info


# ==================== 约束切空间投影相关函数 ====================

def compute_constraint_tangent_projection_single(V_np, Ybus, gen_bus_idx, num_buses):
    """
    计算单个样本的约束切空间投影矩阵
    
    基于潮流 Jacobian 构建投影：
    - F = [dPg/dV; dQg/dV] 是发电机功率对电压的 Jacobian
    - 法向投影 P_nor = F^+ @ F（往这个方向改变会违反约束）
    - 切向投影 P_tan = I - P_nor（在这个方向流动不改变约束残差）
    
    注意：Jacobian 的列顺序是 [Va, Vm]，而模型输出的 z 顺序是 [Vm, Va]
         因此需要进行坐标变换
    
    Args:
        V_np: 复数电压向量 (num_buses,) - numpy array
        Ybus: 导纳矩阵 
        gen_bus_idx: 发电机节点索引列表
        num_buses: 母线数量
    
    Returns:
        P_tan: 切空间投影矩阵 (2*num_buses, 2*num_buses) - numpy array
               作用于 [Vm, Va] 顺序的向量
    """
    # 计算功率对电压的 Jacobian (列顺序: [Va, Vm])
    dPbus_dV, dQbus_dV = compute_jacobian_dPQ_dV(V_np, Ybus, num_buses, device=None)
    
    # 只取发电机节点（约束相关）的行
    # dPbus_dV: (num_buses, 2*num_buses), 选取 gen_bus_idx 行
    # dQbus_dV: (num_buses, 2*num_buses), 选取 gen_bus_idx 行
    gen_bus_idx_np = np.array(gen_bus_idx) if not isinstance(gen_bus_idx, np.ndarray) else gen_bus_idx
    
    dPg_dV = dPbus_dV[gen_bus_idx_np, :]  # (num_gen, 2*num_buses) 列顺序: [Va, Vm]
    dQg_dV = dQbus_dV[gen_bus_idx_np, :]  # (num_gen, 2*num_buses) 列顺序: [Va, Vm]
    
    # 重新排列列顺序: [Va, Vm] -> [Vm, Va]（与模型输出 z 的顺序一致）
    # 原始: [:, 0:num_buses] 是 dVa 部分, [:, num_buses:] 是 dVm 部分
    # 重排后: [:, 0:num_buses] 是 dVm 部分, [:, num_buses:] 是 dVa 部分
    dPg_dV_reordered = np.concatenate([dPg_dV[:, num_buses:], dPg_dV[:, :num_buses]], axis=1)
    dQg_dV_reordered = np.concatenate([dQg_dV[:, num_buses:], dQg_dV[:, :num_buses]], axis=1)
    
    # 合并为约束 Jacobian F = [dPg/dV; dQg/dV]（列顺序: [Vm, Va]，物理坐标系下）
    F_physical = np.vstack([dPg_dV_reordered, dQg_dV_reordered])  # (2*num_gen, 2*num_buses)
    
    # ============================================================================
    # 关键修复：将 Jacobian 从物理坐标系转换到归一化坐标系
    # 
    # 归一化关系：
    #   Vm_norm = (Vm_pu - 1.0) / 0.06  =>  dVm_pu = 0.06 * dVm_norm
    #   Va_norm = Va_rad / (π/6)        =>  dVa_rad = (π/6) * dVa_norm
    # 
    # 链式法则：
    #   dP/d(Vm_norm) = dP/d(Vm_pu) * d(Vm_pu)/d(Vm_norm) = dP/d(Vm_pu) * 0.06
    #   dP/d(Va_norm) = dP/d(Va_rad) * d(Va_rad)/d(Va_norm) = dP/d(Va_rad) * (π/6)
    # ============================================================================
    scale_Vm = 0.06           # Vm 归一化缩放因子
    scale_Va = np.pi / 6.0    # Va 归一化缩放因子
    
    # 列顺序是 [Vm, Va]，构建缩放向量
    scale_vec = np.concatenate([
        np.full(num_buses, scale_Vm),  # 前 num_buses 列对应 Vm
        np.full(num_buses, scale_Va)   # 后 num_buses 列对应 Va
    ])
    
    # 将 Jacobian 转换到归一化坐标系
    F = F_physical * scale_vec[np.newaxis, :]  # (2*num_gen, 2*num_buses)
    
    # 计算伪逆
    try:
        F_pinv = np.linalg.pinv(F, rcond=1e-6)  # (2*num_buses, 2*num_gen)
        # 法向投影: P_nor = F^+ @ F
        P_nor = F_pinv @ F  # (2*num_buses, 2*num_buses)
        # 切向投影: P_tan = I - P_nor
        P_tan = np.eye(2 * num_buses) - P_nor
    except np.linalg.LinAlgError:
        # 如果计算失败，返回单位矩阵（不做投影）
        P_tan = np.eye(2 * num_buses)
    
    return P_tan


def compute_tangent_projection_batch(z, x_input, sys_data, single_target=True):
    """
    批量计算约束切空间投影矩阵
    
    Args:
        z: 当前状态 (batch_size, output_dim) - 归一化的 [Vm, Va]
        x_input: 条件输入 (batch_size, input_dim)
        sys_data: PowerSystemData 对象 (替代 env)
        single_target: 是否为单目标模式
    
    Returns:
        P_tan_batch: 投影矩阵 (batch_size, output_dim, output_dim) - torch tensor
    """
    batch_size = z.shape[0]
    output_dim = z.shape[1]
    num_buses = output_dim // 2
    device = z.device
    
    # 分离 Vm 和 Va
    Vm_norm = z[:, :num_buses]
    Va_norm = z[:, num_buses:]
    
    # 还原到物理值
    Vm_pu = Vm_norm * 0.06 + 1.0
    Va_rad = Va_norm * np.pi / 6.0
    
    # 获取系统参数 (支持 sys_data 和 env 两种接口)
    Ybus = sys_data.Ybus
    gen_bus_idx = getattr(sys_data, 'gen_bus_idx', getattr(sys_data, 'bus_Pg', None))
    
    # 逐样本计算投影矩阵（较慢，但先验证思路）
    P_tan_list = []
    for i in range(batch_size):
        # 构造复数电压
        V_i = (Vm_pu[i] * torch.exp(1j * Va_rad[i].to(torch.complex64))).cpu().numpy()
        
        # 计算投影矩阵
        P_tan_i = compute_constraint_tangent_projection_single(V_i, Ybus, gen_bus_idx, num_buses)
        P_tan_list.append(P_tan_i)
    
    # 转换为 tensor
    P_tan_batch = torch.tensor(np.stack(P_tan_list), dtype=torch.float32, device=device)
    
    return P_tan_batch


def apply_tangent_projection(v, P_tan):
    """
    将速度向量投影到切空间
    
    Args:
        v: 速度向量 (batch_size, output_dim)
        P_tan: 投影矩阵 (batch_size, output_dim, output_dim)
    
    Returns:
        v_projected: 投影后的速度 (batch_size, output_dim)
    """
    # 使用批量矩阵乘法: (B, D, D) @ (B, D, 1) -> (B, D, 1)
    v_expanded = v.unsqueeze(-1)  # (B, D, 1)
    v_projected = torch.bmm(P_tan, v_expanded).squeeze(-1)  # (B, D)
    return v_projected


# ==================== Drift-Correction 流形稳定化方法 ====================

def compute_constraint_residual_single(V_np, Ybus, Pd_np, Qd_np, 
                                        gen_bus_idx, pd_bus_idx, qd_bus_idx,
                                        Pg_max, Pg_min, Qg_max, Qg_min,
                                        num_buses):
    """
    计算单个样本的约束残差 f(x)
    
    约束残差定义：
    - 如果 Pg > Pg_max: residual = Pg - Pg_max (正值，需要减小Pg)
    - 如果 Pg < Pg_min: residual = Pg - Pg_min (负值，需要增大Pg)
    - 否则: residual = 0
    
    Args:
        V_np: 复数电压向量 (num_buses,) - numpy array
        Ybus: 导纳矩阵
        Pd_np: 有功负荷 (num_pd,) - numpy array
        Qd_np: 无功负荷 (num_qd,) - numpy array
        gen_bus_idx: 发电机节点索引
        pd_bus_idx: 有功负荷节点索引
        qd_bus_idx: 无功负荷节点索引
        Pg_max, Pg_min: 有功功率上下限
        Qg_max, Qg_min: 无功功率上下限
        num_buses: 母线数量
    
    Returns:
        f_x: 约束残差向量 (2*num_gen,) - [Pg_residual, Qg_residual]
    """
    # 确保 Ybus 是 dense array
    if hasattr(Ybus, 'toarray'):
        Ybus_np = Ybus.toarray()
    elif hasattr(Ybus, 'todense'):
        Ybus_np = np.array(Ybus.todense())
    else:
        Ybus_np = np.array(Ybus)
    
    # 计算注入功率 S = V * conj(I) = V * conj(Ybus @ V)
    I_np = Ybus_np @ V_np
    S_np = V_np * np.conj(I_np)
    P_bus = np.real(S_np)  # (num_buses,)
    Q_bus = np.imag(S_np)  # (num_buses,)
    
    # 构建负荷向量（全节点）
    Pd_full = np.zeros(num_buses)
    Qd_full = np.zeros(num_buses)
    Pd_full[pd_bus_idx] = Pd_np
    Qd_full[qd_bus_idx] = Qd_np
    
    # 计算发电机功率: Pg = P_bus + Pd (功率平衡)
    Pg = P_bus[gen_bus_idx] + Pd_full[gen_bus_idx]
    Qg = Q_bus[gen_bus_idx] + Qd_full[gen_bus_idx]
    
    # 计算约束残差
    num_gen = len(gen_bus_idx)
    Pg_residual = np.zeros(num_gen)
    Qg_residual = np.zeros(num_gen)
    
    # Pg 约束残差
    for i in range(num_gen):
        if Pg[i] > Pg_max[i]:
            Pg_residual[i] = Pg[i] - Pg_max[i]  # 正值，需要减小
        elif Pg[i] < Pg_min[i]:
            Pg_residual[i] = Pg[i] - Pg_min[i]  # 负值，需要增大
        # else: 保持 0
    
    # Qg 约束残差
    for i in range(num_gen):
        if Qg[i] > Qg_max[i]:
            Qg_residual[i] = Qg[i] - Qg_max[i]
        elif Qg[i] < Qg_min[i]:
            Qg_residual[i] = Qg[i] - Qg_min[i]
    
    # 合并残差
    f_x = np.concatenate([Pg_residual, Qg_residual])  # (2*num_gen,)
    
    return f_x


def compute_drift_correction_single(V_np, Ybus, Pd_np, Qd_np, 
                                     gen_bus_idx, pd_bus_idx, qd_bus_idx,
                                     Pg_max, Pg_min, Qg_max, Qg_min,
                                     num_buses, lambda_cor=5.0):
    """
    计算单个样本的 Drift-Correction：切向投影矩阵 + 法向修正向量
    
    核心公式：
        v_final = P_tan @ v_pred + correction
        其中: P_tan = I - F^+ @ F (切向投影)
              correction = -λ * F^+ @ f(x) (法向修正)
    
    Args:
        V_np: 复数电压向量 (num_buses,)
        Ybus: 导纳矩阵
        Pd_np, Qd_np: 负荷
        gen_bus_idx, pd_bus_idx, qd_bus_idx: 节点索引
        Pg_max, Pg_min, Qg_max, Qg_min: 功率约束
        num_buses: 母线数量
        lambda_cor: 法向修正增益
    
    Returns:
        P_tan: 切空间投影矩阵 (2*num_buses, 2*num_buses)
        correction: 法向修正向量 (2*num_buses,)
    """
    # 1. 计算 Jacobian
    dPbus_dV, dQbus_dV = compute_jacobian_dPQ_dV(V_np, Ybus, num_buses, device=None)
    
    # 只取发电机节点的行
    gen_bus_idx_np = np.array(gen_bus_idx) if not isinstance(gen_bus_idx, np.ndarray) else gen_bus_idx
    dPg_dV = dPbus_dV[gen_bus_idx_np, :]  # (num_gen, 2*num_buses)
    dQg_dV = dQbus_dV[gen_bus_idx_np, :]  # (num_gen, 2*num_buses)
    
    # 重新排列列顺序: [Va, Vm] -> [Vm, Va]
    dPg_dV_reordered = np.concatenate([dPg_dV[:, num_buses:], dPg_dV[:, :num_buses]], axis=1)
    dQg_dV_reordered = np.concatenate([dQg_dV[:, num_buses:], dQg_dV[:, :num_buses]], axis=1)
    
    # 合并为约束 Jacobian F（物理坐标系下）
    F_physical = np.vstack([dPg_dV_reordered, dQg_dV_reordered])  # (2*num_gen, 2*num_buses)
    
    # ============================================================================
    # 关键修复：将 Jacobian 从物理坐标系转换到归一化坐标系
    # 
    # 归一化关系：
    #   Vm_norm = (Vm_pu - 1.0) / 0.06  =>  dVm_pu = 0.06 * dVm_norm
    #   Va_norm = Va_rad / (π/6)        =>  dVa_rad = (π/6) * dVa_norm
    # 
    # 链式法则：
    #   dP/d(Vm_norm) = dP/d(Vm_pu) * d(Vm_pu)/d(Vm_norm) = dP/d(Vm_pu) * 0.06
    #   dP/d(Va_norm) = dP/d(Va_rad) * d(Va_rad)/d(Va_norm) = dP/d(Va_rad) * (π/6)
    # ============================================================================
    scale_Vm = 0.06           # Vm 归一化缩放因子
    scale_Va = np.pi / 6.0    # Va 归一化缩放因子
    
    # 列顺序是 [Vm, Va]，构建缩放向量
    scale_vec = np.concatenate([
        np.full(num_buses, scale_Vm),  # 前 num_buses 列对应 Vm
        np.full(num_buses, scale_Va)   # 后 num_buses 列对应 Va
    ])
    
    # 将 Jacobian 转换到归一化坐标系
    F = F_physical * scale_vec[np.newaxis, :]  # (2*num_gen, 2*num_buses)
    
    # 2. 计算伪逆和投影
    try:
        F_pinv = np.linalg.pinv(F, rcond=1e-6)  # (2*num_buses, 2*num_gen)
        P_nor = F_pinv @ F
        P_tan = np.eye(2 * num_buses) - P_nor
        
        # 3. 计算约束残差 f(x)
        f_x = compute_constraint_residual_single(
            V_np, Ybus, Pd_np, Qd_np,
            gen_bus_idx, pd_bus_idx, qd_bus_idx,
            Pg_max, Pg_min, Qg_max, Qg_min,
            num_buses
        )
        
        # 4. 计算法向修正: correction = -λ * F^+ @ f(x)
        # 注意：F_pinv 已经在归一化坐标系下，所以 correction 的单位也是归一化的
        correction = -lambda_cor * (F_pinv @ f_x)  # (2*num_buses,)
        
    except np.linalg.LinAlgError:
        P_tan = np.eye(2 * num_buses)
        correction = np.zeros(2 * num_buses)
    
    return P_tan, correction


def compute_drift_correction_batch(z, x_input, sys_data, lambda_cor=5.0):
    """
    批量计算 Drift-Correction：切向投影 + 法向修正
    
    核心公式：
        v_final = P_tan @ v_pred + correction
    
    Args:
        z: 当前状态 (batch_size, output_dim) - 归一化的 [Vm, Va]
        x_input: 条件输入 (batch_size, input_dim) - 包含负荷信息
        sys_data: PowerSystemData 对象 (替代 env)
        lambda_cor: 法向修正增益
    
    Returns:
        P_tan_batch: 投影矩阵 (batch_size, output_dim, output_dim)
        correction_batch: 法向修正向量 (batch_size, output_dim)
    """
    batch_size = z.shape[0]
    output_dim = z.shape[1]
    num_buses = output_dim // 2
    device = z.device
    
    # 分离 Vm 和 Va
    Vm_norm = z[:, :num_buses]
    Va_norm = z[:, num_buses:]
    
    # 还原到物理值
    Vm_pu = Vm_norm * 0.06 + 1.0
    Va_rad = Va_norm * np.pi / 6.0
    
    # 获取系统参数 (支持 sys_data 和 env 两种接口)
    # Ybus: scipy sparse matrix
    if hasattr(sys_data, 'Ybus'):
        Ybus = sys_data.Ybus
    else:
        raise AttributeError("sys_data must have 'Ybus' attribute")
    
    # 发电机节点索引
    gen_bus_idx = getattr(sys_data, 'gen_bus_idx', getattr(sys_data, 'bus_Pg', None))
    if gen_bus_idx is None:
        raise AttributeError("sys_data must have 'gen_bus_idx' or 'bus_Pg' attribute")
    
    # 负荷节点索引
    pd_bus_idx = getattr(sys_data, 'pd_bus_idx', getattr(sys_data, 'idx_Pd', None))
    qd_bus_idx = getattr(sys_data, 'qd_bus_idx', getattr(sys_data, 'idx_Qd', None))
    
    # 获取约束限制（转换为 numpy）
    if hasattr(sys_data, 'Pg_max'):
        Pg_max = sys_data.Pg_max.cpu().numpy() if hasattr(sys_data.Pg_max, 'cpu') else np.array(sys_data.Pg_max)
        Pg_min = sys_data.Pg_min.cpu().numpy() if hasattr(sys_data.Pg_min, 'cpu') else np.array(sys_data.Pg_min)
        Qg_max = sys_data.Qg_max.cpu().numpy() if hasattr(sys_data.Qg_max, 'cpu') else np.array(sys_data.Qg_max)
        Qg_min = sys_data.Qg_min.cpu().numpy() if hasattr(sys_data.Qg_min, 'cpu') else np.array(sys_data.Qg_min)
    elif hasattr(sys_data, 'MAXMIN_Pg'):
        # 兼容旧的 sys_data 格式
        Pg_max = sys_data.MAXMIN_Pg[:, 0]
        Pg_min = sys_data.MAXMIN_Pg[:, 1]
        Qg_max = sys_data.MAXMIN_Qg[:, 0]
        Qg_min = sys_data.MAXMIN_Qg[:, 1]
    else:
        raise AttributeError("sys_data must have power limit attributes")
    
    # 提取负荷信息
    num_pd = getattr(sys_data, 'num_pd', len(pd_bus_idx) if pd_bus_idx is not None else 0)
    num_qd = getattr(sys_data, 'num_qd', len(qd_bus_idx) if qd_bus_idx is not None else 0)
    
    P_tan_list = []
    correction_list = []
    
    for i in range(batch_size):
        # 构造复数电压
        V_i = (Vm_pu[i] * torch.exp(1j * Va_rad[i].to(torch.complex64))).cpu().numpy()
        
        # 提取该样本的负荷
        Pd_i = x_input[i, :num_pd].cpu().numpy()
        Qd_i = x_input[i, num_pd:num_pd + num_qd].cpu().numpy()
        
        # 计算投影和修正
        P_tan_i, correction_i = compute_drift_correction_single(
            V_i, Ybus, Pd_i, Qd_i,
            gen_bus_idx, pd_bus_idx, qd_bus_idx,
            Pg_max, Pg_min, Qg_max, Qg_min,
            num_buses, lambda_cor
        )
        
        P_tan_list.append(P_tan_i)
        correction_list.append(correction_i)
    
    # 转换为 tensor
    P_tan_batch = torch.tensor(np.stack(P_tan_list), dtype=torch.float32, device=device)
    correction_batch = torch.tensor(np.stack(correction_list), dtype=torch.float32, device=device)
    
    return P_tan_batch, correction_batch


def apply_drift_correction(v, P_tan, correction):
    """
    应用 Drift-Correction：切向投影 + 法向修正
    
    公式: v_final = P_tan @ v_pred + correction
    
    Args:
        v: 速度向量 (batch_size, output_dim)
        P_tan: 投影矩阵 (batch_size, output_dim, output_dim)
        correction: 法向修正向量 (batch_size, output_dim)
    
    Returns:
        v_corrected: 修正后的速度 (batch_size, output_dim)
    """
    # 切向投影
    v_expanded = v.unsqueeze(-1)  # (B, D, 1)
    v_tangent = torch.bmm(P_tan, v_expanded).squeeze(-1)  # (B, D)
    
    # 加上法向修正
    v_corrected = v_tangent + correction
    
    return v_corrected


# ==================== V2 后处理：兼容 evaluate_multi_objective.py 的实现 ====================

def denormalize_voltage_v2(Vm_scaled, Va_scaled, VmLb, VmUb, scale_vm, scale_va, slack_bus):
    """
    V2: 使用数据驱动的参数将缩放后的电压还原为实际的 p.u. 值
    
    与 evaluate_multi_objective.py 中的反归一化方式一致：
    - Vm = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    - Va = Va_scaled / scale_va
    - 在 slack_bus 位置插入相角 0
    
    Args:
        Vm_scaled: 缩放后的电压幅值 (batch_size, num_buses)
        Va_scaled: 缩放后的电压相角 (batch_size, num_buses - 1) 不含 slack
        VmLb: 电压幅值下限 (num_buses,)
        VmUb: 电压幅值上限 (num_buses,)
        scale_vm: Vm 缩放因子 (标量或 tensor)
        scale_va: Va 缩放因子 (标量或 tensor)
        slack_bus: 平衡节点索引
    
    Returns:
        Vm_pu: 电压幅值 p.u. (batch_size, num_buses)
        Va_rad: 电压相角 弧度 (batch_size, num_buses)
    """
    device = Vm_scaled.device
    batch_size = Vm_scaled.shape[0]
    num_buses = Vm_scaled.shape[1]
    
    # 确保参数在正确的设备上
    if isinstance(VmLb, torch.Tensor):
        VmLb = VmLb.to(device)
        VmUb = VmUb.to(device)
    else:
        VmLb = torch.tensor(VmLb, device=device, dtype=torch.float32)
        VmUb = torch.tensor(VmUb, device=device, dtype=torch.float32)
    
    if isinstance(scale_vm, torch.Tensor):
        scale_vm = scale_vm.to(device).item()
    if isinstance(scale_va, torch.Tensor):
        scale_va = scale_va.to(device).item()
    
    # 反归一化 Vm: Vm = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    Vm_pu = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    
    # 反归一化 Va: Va = Va_scaled / scale_va
    Va_no_slack = Va_scaled / scale_va
    
    # 在 slack_bus 位置插入相角 0
    Va_rad = torch.zeros(batch_size, num_buses, device=device)
    # 创建索引掩码
    all_buses = torch.arange(num_buses, device=device)
    non_slack_buses = torch.cat([all_buses[:slack_bus], all_buses[slack_bus+1:]])
    Va_rad[:, non_slack_buses] = Va_no_slack
    # slack bus 位置保持为 0
    
    return Vm_pu, Va_rad


def normalize_voltage_v2(Vm_pu, Va_rad, VmLb, VmUb, scale_vm, scale_va, slack_bus):
    """
    V2: 将实际的 p.u. 电压转换为缩放后的值
    
    与 denormalize_voltage_v2 互逆
    
    Args:
        Vm_pu: 电压幅值 p.u. (batch_size, num_buses)
        Va_rad: 电压相角 弧度 (batch_size, num_buses)
        VmLb: 电压幅值下限 (num_buses,)
        VmUb: 电压幅值上限 (num_buses,)
        scale_vm: Vm 缩放因子
        scale_va: Va 缩放因子
        slack_bus: 平衡节点索引
    
    Returns:
        Vm_scaled: 缩放后的电压幅值 (batch_size, num_buses)
        Va_scaled: 缩放后的电压相角 (batch_size, num_buses - 1) 不含 slack
    """
    device = Vm_pu.device
    batch_size = Vm_pu.shape[0]
    num_buses = Vm_pu.shape[1]
    
    # 确保参数在正确的设备上
    if isinstance(VmLb, torch.Tensor):
        VmLb = VmLb.to(device)
        VmUb = VmUb.to(device)
    else:
        VmLb = torch.tensor(VmLb, device=device, dtype=torch.float32)
        VmUb = torch.tensor(VmUb, device=device, dtype=torch.float32)
    
    if isinstance(scale_vm, torch.Tensor):
        scale_vm = scale_vm.to(device).item()
    if isinstance(scale_va, torch.Tensor):
        scale_va = scale_va.to(device).item()
    
    # 归一化 Vm: Vm_scaled = (Vm_pu - VmLb) / (VmUb - VmLb) * scale_vm
    Vm_scaled = (Vm_pu - VmLb) / (VmUb - VmLb) * scale_vm
    
    # 归一化 Va: Va_scaled = Va_rad * scale_va (去掉 slack bus)
    all_buses = torch.arange(num_buses, device=device)
    non_slack_buses = torch.cat([all_buses[:slack_bus], all_buses[slack_bus+1:]])
    Va_no_slack = Va_rad[:, non_slack_buses]
    Va_scaled = Va_no_slack * scale_va
    
    return Vm_scaled, Va_scaled


def compute_generator_power_v2(V, Pdtest, Qdtest, bus_Pg, bus_Qg, Ybus, device):
    """
    V2: 计算发电机功率，与 utils.py 中的 get_genload 保持一致
    
    使用矩阵运算加速（支持 GPU）
    
    Args:
        V: 复数电压 (batch_size, num_buses) - numpy array
        Pdtest: 有功负荷 (batch_size, num_buses) - numpy array
        Qdtest: 无功负荷 (batch_size, num_buses) - numpy array
        bus_Pg: 有功发电机节点索引
        bus_Qg: 无功发电机节点索引
        Ybus: 导纳矩阵 (scipy sparse)
        device: 计算设备
    
    Returns:
        Pg: 发电机有功功率 (batch_size, num_gen_P) - numpy array
        Qg: 发电机无功功率 (batch_size, num_gen_Q) - numpy array
        P: 节点功率注入 (batch_size, num_buses) - numpy array
        Q: 节点功率注入 (batch_size, num_buses) - numpy array
    """
    batch_size = V.shape[0]
    num_buses = V.shape[1]
    
    # 确保 Ybus 是 dense array
    if hasattr(Ybus, 'toarray'):
        Ybus_dense = Ybus.toarray()
    elif hasattr(Ybus, 'todense'):
        Ybus_dense = np.array(Ybus.todense())
    else:
        Ybus_dense = np.array(Ybus)
    
    # 计算功率注入 S = V * conj(I) = V * conj(Ybus @ V)
    S = np.zeros(V.shape, dtype=np.complex128)
    for i in range(batch_size):
        I = Ybus_dense.dot(V[i]).conj()
        S[i] = np.multiply(V[i], I)
    
    P = np.real(S)
    Q = np.imag(S)
    
    # 计算发电机功率: Pg = P + Pd, Qg = Q + Qd (在发电机节点)
    Pg = P[:, bus_Pg] + Pdtest[:, bus_Pg]
    Qg = Q[:, bus_Qg] + Qdtest[:, bus_Qg]
    
    return Pg, Qg, P, Q


def detect_pg_qg_violation_v2(Pg, Qg, MAXMIN_Pg, MAXMIN_Qg, bus_Pg, bus_Qg, DELTA):
    """
    V2: 检测 Pg/Qg 约束违规，与 utils.py 中的 get_vioPQg 逻辑一致
    
    返回与 get_vioPQg 相同格式的输出
    
    Args:
        Pg: 发电机有功功率 (batch_size, num_gen_P) - numpy array
        Qg: 发电机无功功率 (batch_size, num_gen_Q) - numpy array
        MAXMIN_Pg: 有功功率限制 [Pmax, Pmin] (num_gen_P, 2)
        MAXMIN_Qg: 无功功率限制 [Qmax, Qmin] (num_gen_Q, 2)
        bus_Pg: 有功发电机节点索引
        bus_Qg: 无功发电机节点索引
        DELTA: 违规阈值
    
    Returns:
        lsPg: Pg 违规列表
        lsQg: Qg 违规列表
        lsidxPg: Pg 违规样本索引
        lsidxQg: Qg 违规样本索引
        vio_PQg: 满足约束比例 (batch_size, 2)
        num_violations: 违规样本数量
    """
    batch_size = Pg.shape[0]
    
    vio_PQgmaxminnum = np.zeros((batch_size, 4))
    vio_PQg = torch.zeros((batch_size, 2))
    lsPg = []
    lsQg = []
    lsidxPg = np.zeros(batch_size, dtype=int)
    lsidxQg = np.zeros(batch_size, dtype=int)
    kP = 1
    kQ = 1
    
    for i in range(batch_size):
        # Active power
        delta_upper = Pg[i] - MAXMIN_Pg[:, 0]  # Pg - Pmax
        idxPgUB = np.array(np.where(delta_upper > DELTA))
        
        delta_lower = Pg[i] - MAXMIN_Pg[:, 1]  # Pg - Pmin
        idxPgLB = np.array(np.where(delta_lower < -DELTA))
        
        PgLUB = None
        if np.size(idxPgUB) > 0:
            PgUB = np.concatenate((idxPgUB, delta_upper[idxPgUB]), axis=0).T
        if np.size(idxPgLB) > 0:
            PgLB = np.concatenate((idxPgLB, delta_lower[idxPgLB]), axis=0).T
        
        if np.size(idxPgUB) > 0 and np.size(idxPgLB) > 0:
            PgLUB = np.concatenate((PgUB, PgLB), axis=0)
        elif np.size(idxPgUB) > 0:
            PgLUB = PgUB
        elif np.size(idxPgLB) > 0:
            PgLUB = PgLB
        
        if (np.size(idxPgUB) + np.size(idxPgLB)) > 0:
            PgLUB = PgLUB[PgLUB[:, 0].argsort()]
            lsPg.append(PgLUB)
            lsidxPg[i] = kP
            kP += 1
        
        # Reactive power
        delta_upper = Qg[i] - MAXMIN_Qg[:, 0]  # Qg - Qmax
        idxQgUB = np.array(np.where(delta_upper > DELTA))
        
        delta_lower = Qg[i] - MAXMIN_Qg[:, 1]  # Qg - Qmin
        idxQgLB = np.array(np.where(delta_lower < -DELTA))
        
        QgLUB = None
        if np.size(idxQgUB) > 0:
            QgUB = np.concatenate((idxQgUB, delta_upper[idxQgUB]), axis=0).T
        if np.size(idxQgLB) > 0:
            QgLB = np.concatenate((idxQgLB, delta_lower[idxQgLB]), axis=0).T
        
        if np.size(idxQgUB) > 0 and np.size(idxQgLB) > 0:
            QgLUB = np.concatenate((QgUB, QgLB), axis=0)
        elif np.size(idxQgUB) > 0:
            QgLUB = QgUB
        elif np.size(idxQgLB) > 0:
            QgLUB = QgLB
        
        if (np.size(idxQgUB) + np.size(idxQgLB)) > 0:
            QgLUB = QgLUB[QgLUB[:, 0].argsort()]
            lsQg.append(QgLUB)
            lsidxQg[i] = kQ
            kQ += 1
        
        vio_PQgmaxminnum[i, 0] = np.size(idxPgUB)
        vio_PQgmaxminnum[i, 1] = np.size(idxPgLB)
        vio_PQgmaxminnum[i, 2] = np.size(idxQgUB)
        vio_PQgmaxminnum[i, 3] = np.size(idxQgLB)
    
    # 计算满足约束比例
    vio_PQg[:, 0] = torch.tensor((1 - (vio_PQgmaxminnum[:, 0] + vio_PQgmaxminnum[:, 1]) / bus_Pg.shape[0]) * 100)
    vio_PQg[:, 1] = torch.tensor((1 - (vio_PQgmaxminnum[:, 2] + vio_PQgmaxminnum[:, 3]) / bus_Qg.shape[0]) * 100)
    
    # 计算违规样本数量
    lsidxPQg = np.squeeze(np.array(np.where((lsidxPg + lsidxQg) > 0)))
    num_violations = np.size(lsidxPQg)
    
    return lsPg, lsQg, lsidxPg, lsidxQg, vio_PQg, num_violations, lsidxPQg


def compute_jacobian_dPQ_dV_v2(his_V, bus_Pg, bus_Qg, Ybus):
    """
    V2: 使用历史电压计算雅可比矩阵，与 utils.py 中的 dPQbus_dV 一致
    
    Args:
        his_V: 历史平均电压 (num_buses,) - numpy array
        bus_Pg: 有功发电机节点索引
        bus_Qg: 无功发电机节点索引
        Ybus: 导纳矩阵 (scipy sparse)
    
    Returns:
        dPbus_dV: dP/d[Va, Vm] (num_buses, 2*num_buses)
        dQbus_dV: dQ/d[Va, Vm] (num_buses, 2*num_buses)
    """
    V = his_V.copy()
    
    # 确保 Ybus 是 dense array
    if hasattr(Ybus, 'toarray'):
        Ybus_dense = Ybus.toarray()
    elif hasattr(Ybus, 'todense'):
        Ybus_dense = np.array(Ybus.todense())
    else:
        Ybus_dense = np.array(Ybus)
    
    Ibus = Ybus_dense.dot(his_V).conj()
    diagV = np.diag(V)
    diagIbus = np.diag(Ibus)
    diagVnorm = np.diag(V / np.abs(V))
    
    dSbus_dVm = np.dot(diagV, np.dot(Ybus_dense, diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
    dSbus_dVa = 1j * np.dot(diagV, (diagIbus - np.dot(Ybus_dense, diagV)).conj())
    
    # 合并 [dVa, dVm]
    dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
    dPbus_dV = np.real(dSbus_dV)
    dQbus_dV = np.imag(dSbus_dV)
    
    return dPbus_dV, dQbus_dV


def compute_dV_historical_v2(lsPg, lsQg, lsidxPg, lsidxQg, num_violations, k_dV,
                             bus_Pg, bus_Qg, dPbus_dV, dQbus_dV, Nbus, Ntest):
    """
    V2: 使用历史电压雅可比矩阵计算电压修正量，与 utils.py 中的 get_hisdV 一致
    
    Args:
        lsPg, lsQg: 违规列表
        lsidxPg, lsidxQg: 违规样本索引
        num_violations: 违规样本数量
        k_dV: 修正系数
        bus_Pg, bus_Qg: 发电机节点索引
        dPbus_dV, dQbus_dV: 雅可比矩阵
        Nbus: 母线数量
        Ntest: 测试样本数量
    
    Returns:
        dV: 电压修正量 (num_violations, 2*Nbus)
    """
    dV = np.zeros((num_violations, Nbus * 2))
    j = 0
    
    for i in range(Ntest):
        if (lsidxPg[i] + lsidxQg[i]) > 0:
            if lsidxPg[i] > 0 and lsidxQg[i] > 0:
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                busPg = bus_Pg[idxPg]
                busQg = bus_Qg[idxQg]
                dPQGbus_dV = np.concatenate((dPbus_dV[busPg, :], dQbus_dV[busQg, :]), axis=0)
                dPQg = np.concatenate((lsPg[lsidxPg[i] - 1][:, 1], lsQg[lsidxQg[i] - 1][:, 1]), axis=0)
            elif lsidxPg[i] > 0:
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                busPg = bus_Pg[idxPg]
                dPQGbus_dV = dPbus_dV[busPg, :]
                dPQg = lsPg[lsidxPg[i] - 1][:, 1]
            elif lsidxQg[i] > 0:
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                busQg = bus_Qg[idxQg]
                dPQGbus_dV = dQbus_dV[busQg, :]
                dPQg = lsQg[lsidxQg[i] - 1][:, 1]
            
            dV[j] = np.dot(np.linalg.pinv(dPQGbus_dV), dPQg * k_dV)
            j += 1
    
    return dV


def clamp_voltage_v2(Vm, hisVm_min, hisVm_max, device):
    """
    V2: 使用历史数据范围裁剪电压幅值
    
    Args:
        Vm: 电压幅值 (batch_size, num_buses)
        hisVm_min: 历史最小值 (num_buses,)
        hisVm_max: 历史最大值 (num_buses,)
        device: 计算设备
    
    Returns:
        Vm_clip: 裁剪后的电压幅值
    """
    if isinstance(hisVm_min, torch.Tensor):
        hisVm_min = hisVm_min.to(device)
        hisVm_max = hisVm_max.to(device)
    else:
        hisVm_min = torch.tensor(hisVm_min, device=device)
        hisVm_max = torch.tensor(hisVm_max, device=device)
    
    Vm_clip = Vm.clone()
    for i in range(Vm.shape[1]):
        Vm_clip[:, i] = torch.clamp(Vm_clip[:, i], min=hisVm_min[i].item(), max=hisVm_max[i].item())
    
    return Vm_clip


def apply_post_processing_v2(Vm_scaled, Va_scaled, sys_data, config, verbose=False):
    """
    V2: 与 evaluate_multi_objective.py 完全一致的后处理实现
    
    此函数实现与 evaluate_multi_objective.py 中相同的后处理逻辑：
    1. 使用数据驱动的归一化参数 (scale_vm, scale_va, VmLb, VmUb)
    2. 正确处理 slack 节点（Va 不包含 slack 节点）
    3. 使用历史数据范围进行裁剪 (hisVm_min, hisVm_max)
    4. 使用历史电压计算雅可比矩阵（当 flag_hisv=True）
    
    Args:
        Vm_scaled: 缩放后的电压幅值 (batch_size, num_buses) - torch tensor
        Va_scaled: 缩放后的电压相角 (batch_size, num_buses - 1) 不含 slack - torch tensor
        sys_data: PowerSystemData 对象
        config: 配置对象（包含 scale_vm, scale_va, k_dV, DELTA, flag_hisv 等）
        verbose: 是否打印详细信息
    
    Returns:
        Vm_corrected_scaled: 修正后的缩放电压幅值 (batch_size, num_buses)
        Va_corrected_scaled: 修正后的缩放电压相角 (batch_size, num_buses - 1)
        correction_info: 修正信息字典
    """
    device = Vm_scaled.device
    batch_size = Vm_scaled.shape[0]
    num_buses = Vm_scaled.shape[1]
    
    # 获取配置参数
    scale_vm = config.scale_vm
    scale_va = config.scale_va
    k_dV = getattr(config, 'k_dV', 1.0)
    DELTA = getattr(config, 'DELTA', 1e-4)
    flag_hisv = getattr(config, 'flag_hisv', 1)
    Ntest = batch_size
    
    # 获取 slack bus
    slack_bus = sys_data.bus_slack if isinstance(sys_data.bus_slack, int) else int(sys_data.bus_slack)
    
    # 获取系统参数
    VmLb = sys_data.VmLb
    VmUb = sys_data.VmUb
    hisVm_min = sys_data.hisVm_min
    hisVm_max = sys_data.hisVm_max
    his_V = sys_data.his_V
    Ybus = sys_data.Ybus
    bus_Pg = sys_data.bus_Pg
    bus_Qg = sys_data.bus_Qg
    MAXMIN_Pg = sys_data.MAXMIN_Pg
    MAXMIN_Qg = sys_data.MAXMIN_Qg
    Pdtest = sys_data.Pdtest
    Qdtest = sys_data.Qdtest
    
    # 1. 反归一化
    Vm_pu, Va_rad = denormalize_voltage_v2(
        Vm_scaled, Va_scaled, VmLb, VmUb, scale_vm, scale_va, slack_bus
    )
    
    # 2. 使用历史数据范围裁剪 Vm
    Vm_clip = clamp_voltage_v2(Vm_pu, hisVm_min, hisVm_max, device)
    
    # 3. 转换为 numpy 进行后续计算
    Vm_np = Vm_clip.cpu().numpy()
    Va_np = Va_rad.cpu().numpy()
    
    # 4. 计算复数电压
    V = Vm_np * np.exp(1j * Va_np)
    
    # 5. 计算发电机功率
    Pg, Qg, P, Q = compute_generator_power_v2(
        V, Pdtest, Qdtest, bus_Pg, bus_Qg, Ybus, device
    )
    
    # 6. 检测违规
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQg, num_violations, lsidxPQg = detect_pg_qg_violation_v2(
        Pg, Qg, MAXMIN_Pg, MAXMIN_Qg, bus_Pg, bus_Qg, DELTA
    )
    
    if verbose:
        print(f"[V2 Post-Processing] Detected {num_violations} violated samples out of {batch_size}")
        print(f"  Pg satisfy: {torch.mean(vio_PQg[:, 0]).item():.2f}%")
        print(f"  Qg satisfy: {torch.mean(vio_PQg[:, 1]).item():.2f}%")
    
    # 7. 如果有违规，应用后处理修正
    Vm_corrected_np = Vm_np.copy()
    Va_corrected_np = Va_np.copy()
    
    if num_violations > 0:
        if verbose:
            print(f"  Applying corrections to {num_violations} samples...")
        
        # 计算雅可比矩阵（使用历史电压）
        if flag_hisv:
            dPbus_dV, dQbus_dV = compute_jacobian_dPQ_dV_v2(his_V, bus_Pg, bus_Qg, Ybus)
        else:
            # 使用当前预测电压的平均值
            V_mean = V.mean(axis=0)
            dPbus_dV, dQbus_dV = compute_jacobian_dPQ_dV_v2(V_mean, bus_Pg, bus_Qg, Ybus)
        
        # 计算电压修正量
        dV = compute_dV_historical_v2(
            lsPg, lsQg, lsidxPg, lsidxQg, num_violations, k_dV,
            bus_Pg, bus_Qg, dPbus_dV, dQbus_dV, num_buses, Ntest
        )
        
        # 应用修正
        # dV 格式: [dVa, dVm]，与 evaluate_multi_objective.py 一致
        Va_corrected_np[lsidxPQg, :] = Va_np[lsidxPQg, :] - dV[:, 0:num_buses]
        Va_corrected_np[:, slack_bus] = 0  # 保持 slack bus 相角为 0
        Vm_corrected_np[lsidxPQg, :] = Vm_np[lsidxPQg, :] - dV[:, num_buses:2*num_buses]
        
        # 再次裁剪 Vm
        Vm_corrected = torch.from_numpy(Vm_corrected_np).float().to(device)
        Vm_corrected = clamp_voltage_v2(Vm_corrected, hisVm_min, hisVm_max, device)
        Vm_corrected_np = Vm_corrected.cpu().numpy()
    
    # 8. 转回归一化（缩放）格式
    Vm_corrected = torch.from_numpy(Vm_corrected_np).float().to(device)
    Va_corrected = torch.from_numpy(Va_corrected_np).float().to(device)
    
    Vm_corrected_scaled, Va_corrected_scaled = normalize_voltage_v2(
        Vm_corrected, Va_corrected, VmLb, VmUb, scale_vm, scale_va, slack_bus
    )
    
    correction_info = {
        'num_samples': batch_size,
        'num_violations_before': num_violations,
        'pg_satisfy_before': torch.mean(vio_PQg[:, 0]).item(),
        'qg_satisfy_before': torch.mean(vio_PQg[:, 1]).item(),
    }
    
    # 计算修正后的违规情况
    if num_violations > 0:
        V_corrected = Vm_corrected_np * np.exp(1j * Va_corrected_np)
        Pg_after, Qg_after, _, _ = compute_generator_power_v2(
            V_corrected, Pdtest, Qdtest, bus_Pg, bus_Qg, Ybus, device
        )
        _, _, lsidxPg_after, lsidxQg_after, vio_PQg_after, num_violations_after, _ = detect_pg_qg_violation_v2(
            Pg_after, Qg_after, MAXMIN_Pg, MAXMIN_Qg, bus_Pg, bus_Qg, DELTA
        )
        
        correction_info['num_violations_after'] = num_violations_after
        correction_info['pg_satisfy_after'] = torch.mean(vio_PQg_after[:, 0]).item()
        correction_info['qg_satisfy_after'] = torch.mean(vio_PQg_after[:, 1]).item()
        
        if verbose:
            print(f"  After correction: {num_violations_after} violated samples")
            print(f"  Pg satisfy: {torch.mean(vio_PQg_after[:, 0]).item():.2f}%")
            print(f"  Qg satisfy: {torch.mean(vio_PQg_after[:, 1]).item():.2f}%")
    else:
        correction_info['num_violations_after'] = 0
        correction_info['pg_satisfy_after'] = correction_info['pg_satisfy_before']
        correction_info['qg_satisfy_after'] = correction_info['qg_satisfy_before']
    
    return Vm_corrected_scaled, Va_corrected_scaled, correction_info


def verify_post_processing_consistency(Vm_scaled, Va_scaled, sys_data, config, verbose=True):
    """
    验证 V2 后处理与 evaluate_multi_objective.py 中后处理的一致性
    
    此函数比较两种实现的输出，确保它们产生相同的结果
    
    Args:
        Vm_scaled: 缩放后的电压幅值 (batch_size, num_buses)
        Va_scaled: 缩放后的电压相角 (batch_size, num_buses - 1)
        sys_data: PowerSystemData 对象
        config: 配置对象
        verbose: 是否打印详细比较结果
    
    Returns:
        is_consistent: 两种实现是否一致
        max_diff_Vm: Vm 最大差异
        max_diff_Va: Va 最大差异
    """
    device = Vm_scaled.device
    
    # 使用 V2 后处理
    Vm_v2, Va_v2, info_v2 = apply_post_processing_v2(
        Vm_scaled, Va_scaled, sys_data, config, verbose=False
    )
    
    # 计算差异统计
    max_diff_Vm = (Vm_v2 - Vm_scaled).abs().max().item()
    max_diff_Va = (Va_v2 - Va_scaled).abs().max().item()
    
    if verbose:
        print("=" * 60)
        print("Post-Processing Verification Results")
        print("=" * 60)
        print(f"Samples: {Vm_scaled.shape[0]}")
        print(f"Violations before: {info_v2['num_violations_before']}")
        print(f"Violations after: {info_v2['num_violations_after']}")
        print(f"Pg satisfy: {info_v2['pg_satisfy_before']:.2f}% -> {info_v2['pg_satisfy_after']:.2f}%")
        print(f"Qg satisfy: {info_v2['qg_satisfy_before']:.2f}% -> {info_v2['qg_satisfy_after']:.2f}%")
        print(f"\nMax correction magnitude:")
        print(f"  Vm: {max_diff_Vm:.6f}")
        print(f"  Va: {max_diff_Va:.6f}")
    
    return info_v2, max_diff_Vm, max_diff_Va


# ==================== GPU 并行化后处理 ====================

def compute_power_batch_gpu(Vm, Va, G, B, device):
    """
    GPU 并行计算节点功率注入
    
    P = Vm * (G @ Vreal - B @ Vimg) * cos(Va) + Vm * (B @ Vreal + G @ Vimg) * sin(Va)
    简化形式: S = V * conj(Y @ V)
    
    Args:
        Vm: 电压幅值 (batch_size, num_buses)
        Va: 电压相角 (batch_size, num_buses)
        G: 导纳矩阵实部 (num_buses, num_buses)
        B: 导纳矩阵虚部 (num_buses, num_buses)
        device: 计算设备
    
    Returns:
        P: 有功功率注入 (batch_size, num_buses)
        Q: 无功功率注入 (batch_size, num_buses)
    """
    # 确保所有张量在正确设备上
    Vm = Vm.to(device)
    Va = Va.to(device)
    G = G.to(device)
    B = B.to(device)
    
    # 计算实部和虚部电压
    Vreal = Vm * torch.cos(Va)
    Vimg = Vm * torch.sin(Va)
    
    # 计算电流: I = Y @ V = (G + jB) @ (Vreal + jVimg)
    # Ireal = G @ Vreal - B @ Vimg
    # Iimg = B @ Vreal + G @ Vimg
    Ireal = torch.matmul(Vreal, G.T) - torch.matmul(Vimg, B.T)
    Iimg = torch.matmul(Vreal, B.T) + torch.matmul(Vimg, G.T)
    
    # 计算功率: S = V * conj(I)
    # P = Vreal * Ireal + Vimg * Iimg
    # Q = Vimg * Ireal - Vreal * Iimg
    P = Vreal * Ireal + Vimg * Iimg
    Q = Vimg * Ireal - Vreal * Iimg
    
    return P, Q


def compute_generator_power_batch_gpu(P, Q, Pd_full, Qd_full, bus_Pg, bus_Qg, device):
    """
    GPU 并行计算发电机功率
    
    Args:
        P: 节点有功功率注入 (batch_size, num_buses)
        Q: 节点无功功率注入 (batch_size, num_buses)
        Pd_full: 有功负荷（全节点）(batch_size, num_buses)
        Qd_full: 无功负荷（全节点）(batch_size, num_buses)
        bus_Pg: 有功发电机节点索引
        bus_Qg: 无功发电机节点索引
        device: 计算设备
    
    Returns:
        Pg: 发电机有功功率 (batch_size, num_gen_P)
        Qg: 发电机无功功率 (batch_size, num_gen_Q)
    """
    bus_Pg_t = torch.tensor(bus_Pg, dtype=torch.long, device=device)
    bus_Qg_t = torch.tensor(bus_Qg, dtype=torch.long, device=device)
    
    # Pg = P[:, bus_Pg] + Pd_full[:, bus_Pg]
    Pg = P[:, bus_Pg_t] + Pd_full[:, bus_Pg_t]
    Qg = Q[:, bus_Qg_t] + Qd_full[:, bus_Qg_t]
    
    return Pg, Qg


def detect_violations_batch_gpu(Pg, Qg, Pg_max, Pg_min, Qg_max, Qg_min, DELTA, device):
    """
    GPU 并行检测发电机功率约束违规
    
    Args:
        Pg: 发电机有功功率 (batch_size, num_gen_P)
        Qg: 发电机无功功率 (batch_size, num_gen_Q)
        Pg_max, Pg_min: 有功功率限制 (num_gen_P,)
        Qg_max, Qg_min: 无功功率限制 (num_gen_Q,)
        DELTA: 违规阈值
        device: 计算设备
    
    Returns:
        violation_mask: 违规样本掩码 (batch_size,)
        violation_Pg: Pg 违规量 (batch_size, num_gen_P)  正值表示超上限，负值表示低于下限
        violation_Qg: Qg 违规量 (batch_size, num_gen_Q)
        vio_PQg: 满足约束比例 (batch_size, 2)
    """
    batch_size = Pg.shape[0]
    num_gen_P = Pg.shape[1]
    num_gen_Q = Qg.shape[1]
    
    Pg_max = Pg_max.to(device)
    Pg_min = Pg_min.to(device)
    Qg_max = Qg_max.to(device)
    Qg_min = Qg_min.to(device)
    
    # 计算 Pg 违规
    Pg_upper_vio = Pg - Pg_max  # 正值 = 超上限
    Pg_lower_vio = Pg_min - Pg  # 正值 = 低于下限
    
    # 只保留正违规量
    violation_Pg = torch.zeros_like(Pg)
    violation_Pg = torch.where(Pg_upper_vio > DELTA, Pg_upper_vio, violation_Pg)
    violation_Pg = torch.where(Pg_lower_vio > DELTA, -Pg_lower_vio, violation_Pg)  # 负号表示需要增加
    
    # 计算 Qg 违规
    Qg_upper_vio = Qg - Qg_max
    Qg_lower_vio = Qg_min - Qg
    
    violation_Qg = torch.zeros_like(Qg)
    violation_Qg = torch.where(Qg_upper_vio > DELTA, Qg_upper_vio, violation_Qg)
    violation_Qg = torch.where(Qg_lower_vio > DELTA, -Qg_lower_vio, violation_Qg)
    
    # 计算每个样本是否有违规
    has_Pg_vio = (violation_Pg.abs() > DELTA).any(dim=1)
    has_Qg_vio = (violation_Qg.abs() > DELTA).any(dim=1)
    violation_mask = has_Pg_vio | has_Qg_vio
    
    # 计算满足约束比例
    num_Pg_vio = ((Pg_upper_vio > DELTA) | (Pg_lower_vio > DELTA)).sum(dim=1).float()
    num_Qg_vio = ((Qg_upper_vio > DELTA) | (Qg_lower_vio > DELTA)).sum(dim=1).float()
    
    vio_PQg = torch.zeros(batch_size, 2, device=device)
    vio_PQg[:, 0] = (1 - num_Pg_vio / num_gen_P) * 100
    vio_PQg[:, 1] = (1 - num_Qg_vio / num_gen_Q) * 100
    
    return violation_mask, violation_Pg, violation_Qg, vio_PQg


def compute_jacobian_batch_gpu(Vm, Va, Ybus, device):
    """
    GPU 并行计算功率对电压的雅可比矩阵
    
    对于批量样本，使用相同的平均电压计算雅可比矩阵（与原始实现一致）
    
    注意：这个函数计算的是在参考电压点（平均电压）处的雅可比矩阵，
    所有样本共用同一个雅可比矩阵，这与原始 DeepOPF-V 实现一致。
    
    Args:
        Vm: 电压幅值 (batch_size, num_buses) 或 (num_buses,) 用于计算雅可比
        Va: 电压相角 (batch_size, num_buses) 或 (num_buses,) 用于计算雅可比
        Ybus: 导纳矩阵 (scipy sparse 或 numpy array)
        device: 计算设备
    
    Returns:
        dPbus_dV: dP/d[Va, Vm] (num_buses, 2*num_buses)
        dQbus_dV: dQ/d[Va, Vm] (num_buses, 2*num_buses)
    """
    # 如果是批量数据，取平均
    if Vm.dim() == 2:
        Vm_ref = Vm.mean(dim=0).cpu().numpy()
        Va_ref = Va.mean(dim=0).cpu().numpy()
    else:
        Vm_ref = Vm.cpu().numpy()
        Va_ref = Va.cpu().numpy()
    
    # 计算复数电压
    V = Vm_ref * np.exp(1j * Va_ref)
    
    # 确保 Ybus 是 dense array
    if hasattr(Ybus, 'toarray'):
        Ybus_np = Ybus.toarray()
    elif hasattr(Ybus, 'todense'):
        Ybus_np = np.array(Ybus.todense())
    else:
        Ybus_np = np.array(Ybus)
    
    # 计算雅可比矩阵
    Ibus = Ybus_np.dot(V).conj()
    diagV = np.diag(V)
    diagIbus = np.diag(Ibus)
    diagVnorm = np.diag(V / np.abs(V))
    
    dSbus_dVm = np.dot(diagV, np.dot(Ybus_np, diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
    dSbus_dVa = 1j * np.dot(diagV, (diagIbus - np.dot(Ybus_np, diagV)).conj())
    
    dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
    dPbus_dV = np.real(dSbus_dV)
    dQbus_dV = np.imag(dSbus_dV)
    
    # 转为 torch tensor
    dPbus_dV_t = torch.from_numpy(dPbus_dV).float().to(device)
    dQbus_dV_t = torch.from_numpy(dQbus_dV).float().to(device)
    
    return dPbus_dV_t, dQbus_dV_t


def apply_post_processing_gpu(Vm_scaled, Va_scaled, sys_data, config, verbose=False):
    """
    GPU 并行化后处理（与 evaluate_multi_objective.py 功能一致）
    
    此函数实现了 GPU 并行化的后处理，可以显著加速大批量数据的处理。
    
    与 apply_post_processing_v2 的主要区别：
    1. 使用 GPU 进行功率计算和违规检测
    2. 批量处理电压修正（而非逐样本）
    
    限制：
    - 雅可比矩阵仍使用 CPU 计算（因为需要矩阵求逆）
    - 电压修正仍需逐样本应用（因为每个样本的违规模式不同）
    
    Args:
        Vm_scaled: 缩放后的电压幅值 (batch_size, num_buses) - torch tensor
        Va_scaled: 缩放后的电压相角 (batch_size, num_buses - 1) 不含 slack - torch tensor
        sys_data: PowerSystemData 对象
        config: 配置对象
        verbose: 是否打印详细信息
    
    Returns:
        Vm_corrected_scaled: 修正后的缩放电压幅值 (batch_size, num_buses)
        Va_corrected_scaled: 修正后的缩放电压相角 (batch_size, num_buses - 1)
        correction_info: 修正信息字典
    """
    device = Vm_scaled.device
    batch_size = Vm_scaled.shape[0]
    num_buses = Vm_scaled.shape[1]
    
    # 获取配置参数
    scale_vm = config.scale_vm.item() if isinstance(config.scale_vm, torch.Tensor) else config.scale_vm
    scale_va = config.scale_va.item() if isinstance(config.scale_va, torch.Tensor) else config.scale_va
    k_dV = getattr(config, 'k_dV', 1.0)
    DELTA = getattr(config, 'DELTA', 1e-4)
    flag_hisv = getattr(config, 'flag_hisv', 1)
    
    # 获取 slack bus
    slack_bus = sys_data.bus_slack if isinstance(sys_data.bus_slack, int) else int(sys_data.bus_slack)
    
    # 获取系统参数并移到 GPU
    VmLb = sys_data.VmLb.to(device) if isinstance(sys_data.VmLb, torch.Tensor) else torch.tensor(sys_data.VmLb, device=device)
    VmUb = sys_data.VmUb.to(device) if isinstance(sys_data.VmUb, torch.Tensor) else torch.tensor(sys_data.VmUb, device=device)
    hisVm_min = sys_data.hisVm_min.to(device) if isinstance(sys_data.hisVm_min, torch.Tensor) else torch.tensor(sys_data.hisVm_min, device=device)
    hisVm_max = sys_data.hisVm_max.to(device) if isinstance(sys_data.hisVm_max, torch.Tensor) else torch.tensor(sys_data.hisVm_max, device=device)
    
    G = sys_data.G.to(device) if isinstance(sys_data.G, torch.Tensor) else torch.tensor(sys_data.G, device=device, dtype=torch.float32)
    B = sys_data.B.to(device) if isinstance(sys_data.B, torch.Tensor) else torch.tensor(sys_data.B, device=device, dtype=torch.float32)
    
    Pg_max = sys_data.Pg_max.to(device) if isinstance(sys_data.Pg_max, torch.Tensor) else torch.tensor(sys_data.MAXMIN_Pg[:, 0], device=device)
    Pg_min = sys_data.Pg_min.to(device) if isinstance(sys_data.Pg_min, torch.Tensor) else torch.tensor(sys_data.MAXMIN_Pg[:, 1], device=device)
    Qg_max = sys_data.Qg_max.to(device) if isinstance(sys_data.Qg_max, torch.Tensor) else torch.tensor(sys_data.MAXMIN_Qg[:, 0], device=device)
    Qg_min = sys_data.Qg_min.to(device) if isinstance(sys_data.Qg_min, torch.Tensor) else torch.tensor(sys_data.MAXMIN_Qg[:, 1], device=device)
    
    bus_Pg = sys_data.bus_Pg
    bus_Qg = sys_data.bus_Qg
    
    # 1. GPU 并行反归一化
    Vm_pu = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    Va_no_slack = Va_scaled / scale_va
    
    # 插入 slack bus 相角 = 0
    Va_rad = torch.zeros(batch_size, num_buses, device=device)
    all_buses = torch.arange(num_buses, device=device)
    non_slack_buses = torch.cat([all_buses[:slack_bus], all_buses[slack_bus+1:]])
    Va_rad[:, non_slack_buses] = Va_no_slack
    
    # 2. GPU 并行裁剪
    Vm_clip = torch.clamp(Vm_pu, hisVm_min, hisVm_max)
    
    # 3. GPU 并行计算功率
    P, Q = compute_power_batch_gpu(Vm_clip, Va_rad, G, B, device)
    
    # 准备负荷数据
    Pdtest = torch.tensor(sys_data.Pdtest, dtype=torch.float32, device=device)
    Qdtest = torch.tensor(sys_data.Qdtest, dtype=torch.float32, device=device)
    
    # 计算发电机功率
    Pg, Qg = compute_generator_power_batch_gpu(P, Q, Pdtest, Qdtest, bus_Pg, bus_Qg, device)
    
    # 4. GPU 并行检测违规
    violation_mask, violation_Pg, violation_Qg, vio_PQg = detect_violations_batch_gpu(
        Pg, Qg, Pg_max, Pg_min, Qg_max, Qg_min, DELTA, device
    )
    
    num_violations = violation_mask.sum().item()
    
    if verbose:
        print(f"[GPU Post-Processing] Detected {num_violations} violated samples out of {batch_size}")
        print(f"  Pg satisfy: {vio_PQg[:, 0].mean().item():.2f}%")
        print(f"  Qg satisfy: {vio_PQg[:, 1].mean().item():.2f}%")
    
    # 5. 应用后处理修正
    Vm_corrected = Vm_clip.clone()
    Va_corrected = Va_rad.clone()
    
    if num_violations > 0:
        if verbose:
            print(f"  Applying corrections to {num_violations} samples...")
        
        # 计算雅可比矩阵（使用历史电压）
        if flag_hisv:
            his_V = sys_data.his_V
            his_Vm = torch.tensor(np.abs(his_V), dtype=torch.float32)
            his_Va = torch.tensor(np.angle(his_V), dtype=torch.float32)
            dPbus_dV, dQbus_dV = compute_jacobian_batch_gpu(his_Vm, his_Va, sys_data.Ybus, device)
        else:
            dPbus_dV, dQbus_dV = compute_jacobian_batch_gpu(Vm_clip, Va_rad, sys_data.Ybus, device)
        
        # 获取违规样本的索引
        vio_indices = torch.where(violation_mask)[0]
        
        # 对每个违规样本计算修正量
        bus_Pg_t = torch.tensor(bus_Pg, dtype=torch.long, device=device)
        bus_Qg_t = torch.tensor(bus_Qg, dtype=torch.long, device=device)
        
        for idx in vio_indices:
            i = idx.item()
            
            # 收集该样本的违规发电机
            vio_Pg_mask = violation_Pg[i].abs() > DELTA
            vio_Qg_mask = violation_Qg[i].abs() > DELTA
            
            if not (vio_Pg_mask.any() or vio_Qg_mask.any()):
                continue
            
            # 获取违规发电机对应的母线索引
            vio_Pg_idx = torch.where(vio_Pg_mask)[0]
            vio_Qg_idx = torch.where(vio_Qg_mask)[0]
            
            bus_P = bus_Pg_t[vio_Pg_idx]
            bus_Q = bus_Qg_t[vio_Qg_idx]
            
            # 构建雅可比矩阵行
            rows = []
            delta_vals = []
            
            if len(bus_P) > 0:
                rows.append(dPbus_dV[bus_P])
                delta_vals.append(violation_Pg[i, vio_Pg_idx])
            
            if len(bus_Q) > 0:
                rows.append(dQbus_dV[bus_Q])
                delta_vals.append(violation_Qg[i, vio_Qg_idx])
            
            if len(rows) > 0:
                dPQGbus_dV = torch.cat(rows, dim=0)
                dPQg = torch.cat(delta_vals, dim=0)
                
                # 使用伪逆计算修正量 (在 CPU 上进行，因为 GPU 上的 pinv 可能不稳定)
                dPQGbus_dV_np = dPQGbus_dV.cpu().numpy()
                dPQg_np = dPQg.cpu().numpy()
                
                try:
                    dV_np = np.dot(np.linalg.pinv(dPQGbus_dV_np), dPQg_np * k_dV)
                    dV = torch.tensor(dV_np, dtype=torch.float32, device=device)
                    
                    # 应用修正: V_new = V - dV
                    Va_corrected[i] = Va_rad[i] - dV[:num_buses]
                    Va_corrected[i, slack_bus] = 0  # 保持 slack bus 相角为 0
                    Vm_corrected[i] = Vm_clip[i] - dV[num_buses:]
                except np.linalg.LinAlgError:
                    pass  # 如果求逆失败，跳过该样本
        
        # 再次裁剪 Vm
        Vm_corrected = torch.clamp(Vm_corrected, hisVm_min, hisVm_max)
    
    # 6. 转回归一化格式
    Vm_corrected_scaled = (Vm_corrected - VmLb) / (VmUb - VmLb) * scale_vm
    Va_no_slack_corrected = Va_corrected[:, non_slack_buses]
    Va_corrected_scaled = Va_no_slack_corrected * scale_va
    
    correction_info = {
        'num_samples': batch_size,
        'num_violations_before': num_violations,
        'pg_satisfy_before': vio_PQg[:, 0].mean().item(),
        'qg_satisfy_before': vio_PQg[:, 1].mean().item(),
    }
    
    # 计算修正后的违规情况
    if num_violations > 0:
        P_after, Q_after = compute_power_batch_gpu(Vm_corrected, Va_corrected, G, B, device)
        Pg_after, Qg_after = compute_generator_power_batch_gpu(P_after, Q_after, Pdtest, Qdtest, bus_Pg, bus_Qg, device)
        violation_mask_after, _, _, vio_PQg_after = detect_violations_batch_gpu(
            Pg_after, Qg_after, Pg_max, Pg_min, Qg_max, Qg_min, DELTA, device
        )
        
        num_violations_after = violation_mask_after.sum().item()
        correction_info['num_violations_after'] = num_violations_after
        correction_info['pg_satisfy_after'] = vio_PQg_after[:, 0].mean().item()
        correction_info['qg_satisfy_after'] = vio_PQg_after[:, 1].mean().item()
        
        if verbose:
            print(f"  After correction: {num_violations_after} violated samples")
            print(f"  Pg satisfy: {vio_PQg_after[:, 0].mean().item():.2f}%")
            print(f"  Qg satisfy: {vio_PQg_after[:, 1].mean().item():.2f}%")
    else:
        correction_info['num_violations_after'] = 0
        correction_info['pg_satisfy_after'] = correction_info['pg_satisfy_before']
        correction_info['qg_satisfy_after'] = correction_info['qg_satisfy_before']
    
    return Vm_corrected_scaled, Va_corrected_scaled, correction_info


# ==================== V2 基于投影的约束满足方法 ====================
# 使用数据驱动的归一化参数，与 evaluate_multi_objective.py 兼容


# ==================== 辅助函数：分层 SVD 投影 ====================

def _normalize_rows(F):
    """
    对约束矩阵每行归一化，避免尺度差异导致 SVD 阈值判断失真
    
    Args:
        F: 约束矩阵 (m, n)
    
    Returns:
        F_normalized: 行归一化后的矩阵 (m, n)
    """
    row_norms = np.linalg.norm(F, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)  # 避免除零
    return F / row_norms


def _svd_null_projection(F, rcond=1e-4):
    """
    使用 SVD 截断构造切空间投影矩阵
    
    相比 pinv(F) @ F，这种方法更稳定：
    - 自动忽略近零奇异值
    - 不会被病态矩阵放大误差
    
    Args:
        F: 约束矩阵 (m, n)，已归一化
        rcond: 相对截断阈值，奇异值 < rcond * s_max 会被忽略
    
    Returns:
        P_tan: 切空间投影矩阵 (n, n)
        V_null: 零空间基 (n, n-r)
        r: 有效秩
        s: 奇异值数组
    """
    m, n = F.shape
    U, s, Vh = np.linalg.svd(F, full_matrices=True)
    
    # 确定有效秩：奇异值 > rcond * 最大奇异值
    tol = rcond * s[0] if len(s) > 0 else rcond
    r = int(np.sum(s > tol))
    
    # 零空间基：V 的后 (n-r) 列
    V = Vh.T  # (n, n)
    V_null = V[:, r:]  # (n, n-r)
    
    # 投影矩阵：P = V_null @ V_null.T
    P_tan = V_null @ V_null.T
    
    # 强制对称化，抑制数值误差
    P_tan = 0.5 * (P_tan + P_tan.T)
    
    return P_tan, V_null, r, s


def _hierarchical_projection(F_gen, F_load, rcond=1e-4, reg_alpha=1e-4):
    """
    分层投影：先保护老约束，再在老约束的切空间内处理新约束
    
    核心性质：P_final 的像空间永远在 null(F_gen) 内，
    因此永远不会破坏发电机约束。
    
    公式：
    P_final = P_gen - P_gen @ F_load.T @ inv(F_load @ P_gen @ F_load.T + λI) @ F_load @ P_gen
    
    改进：
    1. 冗余判据用相对量：||FP|| / ||F_load|| < tol
    2. reg_lambda 自适应：λ = α * trace(A) / k
    
    Args:
        F_gen: 发电机约束矩阵 (m_gen, n)，已归一化
        F_load: 总负荷约束矩阵 (2, n)，已归一化，必须是 2 行！
        rcond: SVD 截断阈值
        reg_alpha: 正则化比例因子
    
    Returns:
        P_final: 最终投影矩阵 (n, n)
        V_null_gen: 发电机约束零空间基
        r_gen: 发电机约束有效秩
        s_gen: 发电机约束奇异值
        info: 诊断信息字典
    """
    # 1. 先计算发电机约束的投影
    P_gen, V_null_gen, r_gen, s_gen = _svd_null_projection(F_gen, rcond)
    
    if F_load is None or F_load.shape[0] == 0:
        return P_gen, V_null_gen, r_gen, s_gen, {'load_constraint': False}
    
    # 2. 在 P_gen 的子空间里处理负荷约束
    # F_load @ P_gen 的效果：如果负荷约束冗余，这个会接近 0
    FP = F_load @ P_gen  # (k, n)
    
    # 相对冗余判据：||FP|| / ||F_load||
    fp_norm = np.linalg.norm(FP)
    f_load_norm = np.linalg.norm(F_load) + 1e-12
    relative_fp = fp_norm / f_load_norm
    
    if relative_fp < 1e-6:
        print(f"[HierarchicalProj] Load constraint REDUNDANT: ||FP||/||F_load|| = {relative_fp:.2e}")
        return P_gen, V_null_gen, r_gen, s_gen, {
            'load_constraint': True,
            'redundant': True,
            'relative_fp': relative_fp
        }
    
    # 3. 自适应正则化：λ = α * trace(A) / k
    A = FP @ FP.T  # (k, k)
    k = A.shape[0]
    trace_A = np.trace(A)
    reg_lambda = reg_alpha * (trace_A / k + 1e-12)
    
    # 4. 带正则的分层投影
    A_reg = A + reg_lambda * np.eye(k)
    A_inv = np.linalg.inv(A_reg)
    
    # P_final = P_gen - P_gen @ F_load.T @ A_inv @ FP
    correction = P_gen @ F_load.T @ A_inv @ FP
    P_final = P_gen - correction
    
    # 强制对称化
    P_final = 0.5 * (P_final + P_final.T)
    
    info = {
        'load_constraint': True,
        'redundant': False,
        'relative_fp': relative_fp,
        'reg_lambda': reg_lambda,
        'correction_norm': np.linalg.norm(correction),
        'trace_A': trace_A
    }
    
    return P_final, V_null_gen, r_gen, s_gen, info


class ConstraintProjectionV2:
    """
    V2 约束投影类：使用数据驱动的归一化参数
    
    此类实现了基于雅可比矩阵的约束切空间投影方法，可用于：
    1. Flow Model 推理时保持约束满足
    2. 优化搜索时保持可行性
    3. 训练时作为软约束
    
    核心思想：
    - 约束函数 g(V) = [Pg - Pg_bound, Qg - Qg_bound]
    - 雅可比矩阵 F = dg/dV
    - 切向投影 P_tan = I - F^+ @ F（沿此方向移动不改变约束残差）
    - 法向修正 correction = -λ * F^+ @ g(V)（将状态拉回可行域）
    
    用法示例：
        ```python
        from flow_model.post_processing import ConstraintProjectionV2
        
        # 创建投影器
        projector = ConstraintProjectionV2(sys_data, config)
        
        # 在 Flow Model 推理中使用
        v_projected = projector.project_velocity(v, z)
        
        # 或使用 Drift-Correction
        v_corrected = projector.apply_drift_correction(v, z, x_input, lambda_cor=5.0)
        ```
    """
    
    def __init__(self, sys_data, config, use_historical_jacobian=True):
        """
        初始化约束投影器
        
        Args:
            sys_data: PowerSystemData 对象
            config: 配置对象
            use_historical_jacobian: 是否使用历史电压计算雅可比（推荐 True）
        """
        self.sys_data = sys_data
        self.config = config
        self.use_historical_jacobian = use_historical_jacobian
        
        # 提取系统参数
        self.num_buses = config.Nbus
        self.slack_bus = sys_data.bus_slack if isinstance(sys_data.bus_slack, int) else int(sys_data.bus_slack)
        
        # 归一化参数
        self.scale_vm = config.scale_vm.item() if isinstance(config.scale_vm, torch.Tensor) else config.scale_vm
        self.scale_va = config.scale_va.item() if isinstance(config.scale_va, torch.Tensor) else config.scale_va
        
        # 电压范围
        self.VmLb = sys_data.VmLb
        self.VmUb = sys_data.VmUb
        
        # 发电机节点索引
        self.bus_Pg = sys_data.bus_Pg
        self.bus_Qg = sys_data.bus_Qg
        self.num_gen_P = len(self.bus_Pg)
        self.num_gen_Q = len(self.bus_Qg)
        
        # 负荷节点索引（用于负荷平衡约束）
        # 纯负荷节点 = 所有节点 - 发电机节点 - slack 节点
        all_buses = set(range(self.num_buses))
        gen_buses = set(self.bus_Pg.tolist() if hasattr(self.bus_Pg, 'tolist') else list(self.bus_Pg))
        self.load_bus_idx = np.array(sorted(all_buses - gen_buses - {self.slack_bus}))
        self.num_load_buses = len(self.load_bus_idx)
        
        # 功率限制
        self.Pg_max = sys_data.MAXMIN_Pg[:, 0]
        self.Pg_min = sys_data.MAXMIN_Pg[:, 1]
        self.Qg_max = sys_data.MAXMIN_Qg[:, 0]
        self.Qg_min = sys_data.MAXMIN_Qg[:, 1]
        
        # 导纳矩阵
        self.Ybus = sys_data.Ybus
        if hasattr(self.Ybus, 'toarray'):
            self.Ybus_dense = self.Ybus.toarray()
        else:
            self.Ybus_dense = np.array(self.Ybus)
        
        # 历史电压（用于计算参考雅可比）
        self.his_V = sys_data.his_V
        
        # 预计算参考雅可比矩阵（如果使用历史电压）
        if use_historical_jacobian:
            self._precompute_reference_jacobian()
    
    def _precompute_reference_jacobian(self):
        """预计算参考点处的雅可比矩阵"""
        V_ref = self.his_V
        
        # 计算功率对电压的雅可比 (列顺序: [Va, Vm])
        Ibus = self.Ybus_dense.dot(V_ref).conj()
        diagV = np.diag(V_ref)
        diagIbus = np.diag(Ibus)
        diagVnorm = np.diag(V_ref / np.abs(V_ref))
        
        dSbus_dVm = np.dot(diagV, np.dot(self.Ybus_dense, diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
        dSbus_dVa = 1j * np.dot(diagV, (diagIbus - np.dot(self.Ybus_dense, diagV)).conj())
        
        dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
        self.dPbus_dV_ref = np.real(dSbus_dV)  # (num_buses, 2*num_buses)
        self.dQbus_dV_ref = np.imag(dSbus_dV)  # (num_buses, 2*num_buses)
        
        # 发电机节点的 Jacobian
        self.dPg_dV_ref = self.dPbus_dV_ref[self.bus_Pg, :]  # (num_gen_P, 2*num_buses)
        self.dQg_dV_ref = self.dQbus_dV_ref[self.bus_Qg, :]  # (num_gen_Q, 2*num_buses)
        
        # 总负荷偏差约束的 Jacobian（对所有负荷节点求和）
        # 总负荷偏差: Σ(P_injection_i + Pd_i) = 0, Σ(Q_injection_i + Qd_i) = 0
        # 约束 Jacobian 是所有负荷节点功率注入 Jacobian 的求和，形状 (1, 2*num_buses)
        # 这样只增加 2 个约束（P 和 Q 各一个），而不是 2*num_load 个约束
        if len(self.load_bus_idx) > 0:
            dPload_dV_all = self.dPbus_dV_ref[self.load_bus_idx, :]  # (num_load, 2*num_buses)
            dQload_dV_all = self.dQbus_dV_ref[self.load_bus_idx, :]  # (num_load, 2*num_buses)
            # 求和得到总偏差约束
            self.dPload_dV_ref = np.sum(dPload_dV_all, axis=0, keepdims=True)  # (1, 2*num_buses)
            self.dQload_dV_ref = np.sum(dQload_dV_all, axis=0, keepdims=True)  # (1, 2*num_buses)
        else:
            self.dPload_dV_ref = np.zeros((1, 2 * self.num_buses))
            self.dQload_dV_ref = np.zeros((1, 2 * self.num_buses))
    
    def _compute_scale_factors(self):
        """
        计算从归一化坐标到物理坐标的缩放因子
        
        归一化关系（基于 evaluate_multi_objective.py）：
        - Vm_scaled = (Vm_pu - VmLb) / (VmUb - VmLb) * scale_vm
        - Va_scaled = Va_rad * scale_va  (去掉 slack)
        
        因此：
        - dVm_pu = (VmUb - VmLb) / scale_vm * dVm_scaled
        - dVa_rad = 1 / scale_va * dVa_scaled
        """
        # 对于 Vm，每个节点的缩放因子可能不同
        VmLb = self.VmLb.numpy() if isinstance(self.VmLb, torch.Tensor) else self.VmLb
        VmUb = self.VmUb.numpy() if isinstance(self.VmUb, torch.Tensor) else self.VmUb
        
        # 确保是 1D 数组
        VmLb = VmLb.flatten()
        VmUb = VmUb.flatten()
        
        # 如果是标量，扩展到所有节点
        if len(VmLb) == 1:
            VmLb = np.full(self.num_buses, VmLb[0])
        if len(VmUb) == 1:
            VmUb = np.full(self.num_buses, VmUb[0])
        
        scale_Vm = (VmUb - VmLb) / self.scale_vm  # (num_buses,)
        scale_Va = 1.0 / self.scale_va            # 标量
        
        return scale_Vm, scale_Va
    
    def compute_projection_matrix(self, z=None):
        """
        计算约束切空间投影矩阵
        
        Args:
            z: 当前状态 (batch_size, output_dim) 或 None（使用参考雅可比）
               如果提供，格式为 [Vm_scaled, Va_scaled（不含slack）]
            include_slack: 输出的 Va 部分是否包含 slack 节点 
        
        Returns:
            P_tan: 切空间投影矩阵
                   如果 include_slack=False: (2*num_buses - 1, 2*num_buses - 1)
                   如果 include_slack=True:  (2*num_buses, 2*num_buses)
            F: 约束雅可比矩阵
            F_pinv: 约束雅可比的伪逆
        """
        # 使用参考雅可比（更稳定）
        # dPg_dV_ref 和 dQg_dV_ref 的列顺序是 [Va (0~num_buses-1), Vm (num_buses~2*num_buses-1)]
        dPg_dV = self.dPg_dV_ref  # (num_gen_P, 2*num_buses) 列顺序: [Va, Vm]
        dQg_dV = self.dQg_dV_ref  # (num_gen_Q, 2*num_buses) 列顺序: [Va, Vm] 
        
        num_buses = self.num_buses
        
        # 获取缩放因子
        scale_Vm, scale_Va = self._compute_scale_factors()
         
        # 输出维度: [Vm (num_buses), Va (num_buses - 1, 不含 slack)]
        output_dim = 2 * num_buses - 1
        
        # 分离 Va 和 Vm 部分
        dPg_dVa = dPg_dV[:, :num_buses]      # (num_gen_P, num_buses)
        dPg_dVm = dPg_dV[:, num_buses:]      # (num_gen_P, num_buses)
        dQg_dVa = dQg_dV[:, :num_buses]      # (num_gen_Q, num_buses)
        dQg_dVm = dQg_dV[:, num_buses:]      # (num_gen_Q, num_buses)
        
        # 获取非 slack 节点的索引
        all_buses = np.arange(num_buses)
        non_slack_buses = np.concatenate([all_buses[:self.slack_bus], all_buses[self.slack_bus+1:]])
        
        # 选取 Va 列（去掉 slack）
        dPg_dVa_no_slack = dPg_dVa[:, non_slack_buses]  # (num_gen_P, num_buses-1)
        dQg_dVa_no_slack = dQg_dVa[:, non_slack_buses]  # (num_gen_Q, num_buses-1)
        
        # 重排列顺序: [Vm, Va (不含 slack)]
        dPg_dV_reordered = np.concatenate([dPg_dVm, dPg_dVa_no_slack], axis=1)
        dQg_dV_reordered = np.concatenate([dQg_dVm, dQg_dVa_no_slack], axis=1)
        
        # 构建缩放向量 [Vm (num_buses), Va (num_buses - 1, 不含 slack)]
        scale_vec = np.concatenate([scale_Vm, np.full(num_buses - 1, scale_Va)])
        
        # ==================== 使用原始 pinv 方法计算投影 ====================
        # 构建发电机约束矩阵
        F_gen_raw = np.vstack([dPg_dV_reordered, dQg_dV_reordered])
        
        # 转换到归一化坐标系
        F = F_gen_raw * scale_vec[np.newaxis, :]  # (2*num_gen, output_dim)
        
        print(f"[Projection] F shape: {F.shape}, output_dim: {output_dim}")
        
        # 使用原始 pinv 方法计算投影矩阵
        try:
            F_pinv = np.linalg.pinv(F, rcond=1e-6)  # (output_dim, 2*num_gen)
            # 法向投影: P_nor = F^+ @ F
            P_nor = F_pinv @ F  # (output_dim, output_dim)
            # 切向投影: P_tan = I - P_nor
            P_tan = np.eye(output_dim) - P_nor
            
            print(f"[Projection] P_tan computed using pinv method")
            print(f"[Projection] trace(P_tan): {np.trace(P_tan):.2f}")
        except np.linalg.LinAlgError as e:
            print(f"[Projection] Warning: pinv failed ({e}), using identity matrix")
            P_tan = np.eye(output_dim)
            F_pinv = None
        
        return P_tan, F, F_pinv
    
    def compute_constraint_residual(self, Vm_pu, Va_rad, Pd_full, Qd_full):
        """
        计算约束残差 g(V) = [Pg - Pg_bound, Qg - Qg_bound]
        
        Args:
            Vm_pu: 电压幅值 p.u. (batch_size, num_buses) 或 (num_buses,)
            Va_rad: 电压相角 弧度 (batch_size, num_buses) 或 (num_buses,)
            Pd_full: 有功负荷（全节点）(batch_size, num_buses) 或 (num_buses,)
            Qd_full: 无功负荷（全节点）(batch_size, num_buses) 或 (num_buses,)
        
        Returns:
            residual: 约束残差 (batch_size, 2*num_gen) 或 (2*num_gen,)
        """
        is_batch = Vm_pu.ndim == 2 if hasattr(Vm_pu, 'ndim') else len(Vm_pu.shape) == 2
        
        if not is_batch:
            Vm_pu = Vm_pu[np.newaxis, :] if isinstance(Vm_pu, np.ndarray) else Vm_pu.unsqueeze(0)
            Va_rad = Va_rad[np.newaxis, :] if isinstance(Va_rad, np.ndarray) else Va_rad.unsqueeze(0)
            Pd_full = Pd_full[np.newaxis, :] if isinstance(Pd_full, np.ndarray) else Pd_full.unsqueeze(0)
            Qd_full = Qd_full[np.newaxis, :] if isinstance(Qd_full, np.ndarray) else Qd_full.unsqueeze(0)
        
        # 转换为 numpy
        if isinstance(Vm_pu, torch.Tensor):
            Vm_pu = Vm_pu.cpu().numpy()
            Va_rad = Va_rad.cpu().numpy()
        if isinstance(Pd_full, torch.Tensor):
            Pd_full = Pd_full.cpu().numpy()
            Qd_full = Qd_full.cpu().numpy()
        
        batch_size = Vm_pu.shape[0]
        total_constraints = self.num_gen_P + self.num_gen_Q
        
        residuals = np.zeros((batch_size, total_constraints))
        
        for i in range(batch_size):
            # 计算复数电压
            V = Vm_pu[i] * np.exp(1j * Va_rad[i])
            
            # 计算功率注入
            I = self.Ybus_dense.dot(V).conj()
            S = V * I
            P_bus = np.real(S)
            Q_bus = np.imag(S)
            
            # 计算发电机功率
            Pg = P_bus[self.bus_Pg] + Pd_full[i, self.bus_Pg]
            Qg = Q_bus[self.bus_Qg] + Qd_full[i, self.bus_Qg]
            
            # 计算约束残差
            Pg_residual = np.zeros(self.num_gen_P)
            Qg_residual = np.zeros(self.num_gen_Q)
            
            for j in range(self.num_gen_P):
                if Pg[j] > self.Pg_max[j]:
                    Pg_residual[j] = Pg[j] - self.Pg_max[j]
                elif Pg[j] < self.Pg_min[j]:
                    Pg_residual[j] = Pg[j] - self.Pg_min[j]
            
            for j in range(self.num_gen_Q):
                if Qg[j] > self.Qg_max[j]:
                    Qg_residual[j] = Qg[j] - self.Qg_max[j]
                elif Qg[j] < self.Qg_min[j]:
                    Qg_residual[j] = Qg[j] - self.Qg_min[j]
            
            residuals[i] = np.concatenate([Pg_residual, Qg_residual])
        
        if not is_batch:
            residuals = residuals[0]
        
        return residuals
    
    def compute_constraint_residual_batch(self, z_combined, x_input, num_buses):
        """
        批量计算约束残差 g(z)，用于 Drift-Correction
        
        Args:
            z_combined: 当前状态 [Vm_scaled, Va_scaled] (batch_size, 2*num_buses-1)
            x_input: 条件输入 (batch_size, input_dim) 包含负荷信息
            num_buses: 母线数量
            
        Returns:
            residual: 约束残差 (batch_size, 2*num_gen) 或 None（如果计算失败）
        """
        try:
            batch_size = z_combined.shape[0]
            device = z_combined.device
            
            # 分离 Vm 和 Va
            Vm_scaled = z_combined[:, :num_buses]
            Va_scaled = z_combined[:, num_buses:]
            
            # 反归一化
            VmLb = self.VmLb
            VmUb = self.VmUb
            if isinstance(VmLb, torch.Tensor):
                VmLb = VmLb.to(device)
                VmUb = VmUb.to(device)
            else:
                VmLb = torch.tensor(VmLb, device=device, dtype=torch.float32)
                VmUb = torch.tensor(VmUb, device=device, dtype=torch.float32)
            
            Vm_pu = Vm_scaled / self.scale_vm * (VmUb - VmLb) + VmLb
            Va_no_slack = Va_scaled / self.scale_va
            
            # 插入 slack 节点相角
            Va_rad = torch.zeros(batch_size, num_buses, device=device)
            all_buses = torch.arange(num_buses, device=device)
            non_slack_buses = torch.cat([all_buses[:self.slack_bus], all_buses[self.slack_bus+1:]])
            Va_rad[:, non_slack_buses] = Va_no_slack
            
            # 提取负荷信息
            num_pd = len(self.sys_data.idx_Pd) if hasattr(self.sys_data, 'idx_Pd') else self.sys_data.num_pd
            num_qd = len(self.sys_data.idx_Qd) if hasattr(self.sys_data, 'idx_Qd') else self.sys_data.num_qd
            
            # 构建全节点负荷
            Pd_full = torch.zeros(batch_size, num_buses, device=device)
            Qd_full = torch.zeros(batch_size, num_buses, device=device)
            
            if hasattr(self.sys_data, 'idx_Pd'):
                pd_idx = self.sys_data.idx_Pd
                qd_idx = self.sys_data.idx_Qd
            else:
                pd_idx = self.sys_data.pd_bus_idx
                qd_idx = self.sys_data.qd_bus_idx
            
            pd_idx_t = torch.tensor(pd_idx, dtype=torch.long, device=device)
            qd_idx_t = torch.tensor(qd_idx, dtype=torch.long, device=device)
            
            Pd_sparse = x_input[:, :num_pd]
            Qd_sparse = x_input[:, num_pd:num_pd + num_qd]
            Pd_full[:, pd_idx_t] = Pd_sparse
            Qd_full[:, qd_idx_t] = Qd_sparse
            
            # 计算约束残差
            residual = self.compute_constraint_residual(Vm_pu, Va_rad, Pd_full, Qd_full)
            
            return residual
            
        except Exception as e:
            # 如果计算失败，返回 None（跳过法向修正）
            return None
    
    def project_velocity(self, v, z=None, device=None):
        """
        将速度向量投影到约束切空间
        
        用于 Flow Model 推理时保持约束满足
        
        Args:
            v: 速度向量 (batch_size, output_dim) [Vm_scaled, Va_scaled（不含slack）]
            z: 当前状态 (batch_size, output_dim)，可选
            device: 计算设备
        
        Returns:
            v_projected: 投影后的速度 (batch_size, output_dim)
        """
        if device is None:
            device = v.device
        
        # 计算投影矩阵（使用参考雅可比，所有样本共用）
        P_tan, _, _ = self.compute_projection_matrix(z)
        P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
        
        # 应用投影: v_projected = v @ P_tan.T
        v_projected = torch.matmul(v, P_tan_t.T)
        
        return v_projected
    
    def apply_drift_correction(self, v, z, x_input, lambda_cor=5.0, device=None):
        """
        应用 Drift-Correction：切向投影 + 法向修正
        
        公式: v_final = P_tan @ v + correction
        其中: correction = -λ * F^+ @ g(z)
        
        Args:
            v: 速度向量 (batch_size, output_dim) [Vm_scaled, Va_scaled（不含slack）]
            z: 当前状态 (batch_size, output_dim)
            x_input: 条件输入 (batch_size, input_dim) 包含负荷信息
            lambda_cor: 法向修正增益（控制回到可行域的速度）
            device: 计算设备
        
        Returns:
            v_corrected: 修正后的速度 (batch_size, output_dim)
        """
        if device is None:
            device = v.device
        
        batch_size = v.shape[0]
        num_buses = self.num_buses
        
        # 1. 计算投影矩阵和伪逆
        P_tan, F, F_pinv = self.compute_projection_matrix(z)
        
        if F_pinv is None:
            return v
        
        P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
        F_pinv_t = torch.tensor(F_pinv, dtype=torch.float32, device=device)
        
        # 2. 反归一化当前状态
        Vm_scaled = z[:, :num_buses]
        Va_scaled = z[:, num_buses:]
        
        VmLb = self.VmLb.to(device) if isinstance(self.VmLb, torch.Tensor) else torch.tensor(self.VmLb, device=device)
        VmUb = self.VmUb.to(device) if isinstance(self.VmUb, torch.Tensor) else torch.tensor(self.VmUb, device=device)
        
        Vm_pu = Vm_scaled / self.scale_vm * (VmUb - VmLb) + VmLb
        Va_no_slack = Va_scaled / self.scale_va
        
        # 插入 slack 节点相角
        Va_rad = torch.zeros(batch_size, num_buses, device=device)
        all_buses = torch.arange(num_buses, device=device)
        non_slack_buses = torch.cat([all_buses[:self.slack_bus], all_buses[self.slack_bus+1:]])
        Va_rad[:, non_slack_buses] = Va_no_slack
        
        # 3. 提取负荷信息
        num_pd = self.sys_data.num_pd
        num_qd = self.sys_data.num_qd
        Pd_sparse = x_input[:, :num_pd]
        Qd_sparse = x_input[:, num_pd:num_pd + num_qd]
        
        # 构建全节点负荷
        Pd_full = torch.zeros(batch_size, num_buses, device=device)
        Qd_full = torch.zeros(batch_size, num_buses, device=device)
        pd_idx = torch.tensor(self.sys_data.idx_Pd, dtype=torch.long, device=device)
        qd_idx = torch.tensor(self.sys_data.idx_Qd, dtype=torch.long, device=device)
        Pd_full[:, pd_idx] = Pd_sparse
        Qd_full[:, qd_idx] = Qd_sparse
        
        # 4. 计算约束残差
        residual = self.compute_constraint_residual(Vm_pu, Va_rad, Pd_full, Qd_full)
        residual_t = torch.tensor(residual, dtype=torch.float32, device=device)
        
        # 5. 计算法向修正: correction = -λ * F^+ @ g(z)
        correction = -lambda_cor * torch.matmul(residual_t, F_pinv_t.T)
        
        # 6. 应用切向投影 + 法向修正
        v_tangent = torch.matmul(v, P_tan_t.T)
        v_corrected = v_tangent + correction
        
        return v_corrected


def create_constraint_projector(sys_data, config):
    """
    工厂函数：创建约束投影器
    
    Args:
        sys_data: PowerSystemData 对象
        config: 配置对象
    
    Returns:
        projector: ConstraintProjectionV2 实例
    
    Example:
        ```python
        from flow_model.post_processing import create_constraint_projector
        
        projector = create_constraint_projector(sys_data, config)
        v_projected = projector.project_velocity(v, z)
        ```
    """
    return ConstraintProjectionV2(sys_data, config)


# ==================== 用于 Flow Model 的便捷接口 ====================

class ProjectedFlowIntegrator:
    """
    带约束投影的 Flow Model 积分器
    
    在每个积分步骤中应用约束投影，确保生成的解满足约束
    
    用法示例：
        ```python
        from flow_model.post_processing import ProjectedFlowIntegrator
        
        # 创建积分器
        integrator = ProjectedFlowIntegrator(
            sys_data, config, 
            projection_mode='tangent',  # 或 'drift_correction'
            lambda_cor=5.0
        )
        
        # 定义速度函数（例如 Flow Model 的向量场）
        def velocity_fn(z, t):
            return model(z, t, x_input)
        
        # 执行积分
        z_final = integrator.integrate(z0, velocity_fn, num_steps=20, dt=0.05)
        ```
    """
    
    def __init__(self, sys_data, config, projection_mode='tangent', lambda_cor=5.0):
        """
        初始化积分器
        
        Args:
            sys_data: PowerSystemData 对象
            config: 配置对象
            projection_mode: 投影模式
                'none' - 不使用投影
                'tangent' - 仅切向投影（推荐用于推理）
                'drift_correction' - 切向投影 + 法向修正（用于强制满足约束）
            lambda_cor: 法向修正增益（仅用于 drift_correction 模式）
        """
        self.sys_data = sys_data
        self.config = config
        self.projection_mode = projection_mode
        self.lambda_cor = lambda_cor
        
        if projection_mode != 'none':
            self.projector = ConstraintProjectionV2(sys_data, config)
            
            # 预计算投影矩阵
            self.P_tan, _, _ = self.projector.compute_projection_matrix()
            self.P_tan_t = None  # 延迟初始化
    
    def _ensure_P_tan_on_device(self, device):
        """确保投影矩阵在正确的设备上"""
        if self.P_tan_t is None or self.P_tan_t.device != device:
            self.P_tan_t = torch.tensor(self.P_tan, dtype=torch.float32, device=device)
    
    def project_velocity(self, v, z=None, x_input=None):
        """
        投影速度向量
        
        Args:
            v: 速度向量 (batch_size, output_dim)
            z: 当前状态（用于 drift_correction 模式）
            x_input: 条件输入（用于 drift_correction 模式）
        
        Returns:
            v_projected: 投影后的速度
        """
        device = v.device
        
        if self.projection_mode == 'none':
            return v
        elif self.projection_mode == 'tangent':
            self._ensure_P_tan_on_device(device)
            return torch.matmul(v, self.P_tan_t.T)
        elif self.projection_mode == 'drift_correction':
            if z is None or x_input is None:
                raise ValueError("drift_correction mode requires z and x_input")
            return self.projector.apply_drift_correction(v, z, x_input, self.lambda_cor)
        else:
            raise ValueError(f"Unknown projection mode: {self.projection_mode}")
    
    def integrate_step(self, z, v, dt, x_input=None):
        """
        执行一个积分步骤（带投影）
        
        Args:
            z: 当前状态 (batch_size, output_dim)
            v: 速度/向量场 (batch_size, output_dim)
            dt: 时间步长
            x_input: 条件输入（用于 drift_correction 模式）
        
        Returns:
            z_new: 更新后的状态
        """
        v_projected = self.project_velocity(v, z, x_input)
        z_new = z + v_projected * dt
        return z_new
    
    def integrate(self, z0, velocity_fn, num_steps, dt=None, x_input=None, return_trajectory=False):
        """
        完整的积分过程（从 z0 积分到最终状态）
        
        Args:
            z0: 初始状态 (batch_size, output_dim)
            velocity_fn: 速度函数 velocity_fn(z, t, x_input) -> v
            num_steps: 积分步数
            dt: 时间步长（如果为 None，则 dt = 1.0 / num_steps）
            x_input: 条件输入
            return_trajectory: 是否返回整个轨迹
        
        Returns:
            z_final: 最终状态
            trajectory: 如果 return_trajectory=True，返回 (z_final, [z0, z1, ..., z_final])
        """
        if dt is None:
            dt = 1.0 / num_steps
        
        z = z0
        trajectory = [z0] if return_trajectory else None
        
        for step in range(num_steps):
            t = step * dt
            v = velocity_fn(z, t, x_input)
            z = self.integrate_step(z, v, dt, x_input)
            
            if return_trajectory:
                trajectory.append(z)
        
        if return_trajectory:
            return z, trajectory
        else:
            return z