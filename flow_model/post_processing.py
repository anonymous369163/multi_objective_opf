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


def compute_generator_power(P, Q, Pd, Qd, pd_bus_idx, qd_bus_idx, gen_bus_idx, device):
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
        gen_bus_idx: 发电机节点索引
        device: 计算设备
    
    Returns:
        Pg: 发电机有功功率 (batch_size, num_gen)
        Qg: 发电机无功功率 (batch_size, num_gen)
    """
    batch_size = P.shape[0]
    
    # 克隆 P 和 Q 以避免修改原始数据
    Pg_bus = P.clone()
    Qg_bus = Q.clone()
    
    # 确保索引是 torch tensor
    pd_bus_idx_t = torch.from_numpy(pd_bus_idx).long().to(device) if isinstance(pd_bus_idx, np.ndarray) else pd_bus_idx.long().to(device)
    qd_bus_idx_t = torch.from_numpy(qd_bus_idx).long().to(device) if isinstance(qd_bus_idx, np.ndarray) else qd_bus_idx.long().to(device)
    gen_bus_idx_t = torch.from_numpy(gen_bus_idx).long().to(device) if isinstance(gen_bus_idx, np.ndarray) else gen_bus_idx.long().to(device)
    
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
    
    # 提取发电机节点的功率
    Pg = Pg_bus[:, gen_bus_idx_t]
    Qg = Qg_bus[:, gen_bus_idx_t]
    
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


def apply_post_processing(Vm_norm, Va_norm, x_input, env, k_dV=1.0, verbose=False, debug_mode=0):
    """
    主函数：应用后处理修正
    
    Args:
        Vm_norm: 归一化电压幅值 (batch_size, num_buses)
        Va_norm: 归一化电压相角 (batch_size, num_buses)
        x_input: 输入数据 (batch_size, input_dim)，包含 Pd, Qd
        env: 电网环境对象
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
    
    # 确定平衡节点
    if hasattr(env, 'balance_gen_bus'):
        slack_bus = env.balance_gen_bus
    else:
        slack_bus = 0  # 默认第一个节点
    
    # 1. 还原归一化
    Vm_pu, Va_rad = denormalize_voltage(Vm_norm, Va_norm)
    
    # 2. 计算功率
    P, Q = compute_power_injection(Vm_pu, Va_rad, env.G, env.B)
    
    # 3. 计算支路功率
    Sf, St, Pf, Qf, Pt, Qt = compute_branch_power(
        Vm_pu, Va_rad, env.Gf, env.Bf, env.Gt, env.Bt, env.Cf, env.Ct
    )
    
    # 4. 提取负荷
    Pd = x_input[:, :env.num_pd]
    Qd = x_input[:, env.num_pd:env.num_pd + env.num_qd]
    
    # 5. 计算发电机功率
    Pg, Qg = compute_generator_power(
        P, Q, Pd, Qd, 
        env.pd_bus_idx, env.qd_bus_idx, env.gen_bus_idx, device
    )
    
    # 6. 检测违规
    pg_violations = detect_pg_qg_violation(
        Pg, Qg, env.Pg_max, env.Pg_min, env.Qg_max, env.Qg_min, 
        env.gen_bus_idx
    )
    
    branch_violations = detect_branch_violation(Sf, St, Pf, Qf, env.S_max)
    
    # 7. 计算平均电压用于雅可比矩阵
    # 使用批次平均电压计算雅可比矩阵（与 DeepOPV-V 中使用历史电压类似）
    # Vm_mean = Vm_pu.mean(dim=0)
    # Va_mean = Va_rad.mean(dim=0)
    # V_mean = (Vm_mean * torch.exp(1j * Va_mean.to(torch.complex64))).cpu().numpy()
    
    # # 8. 计算雅可比矩阵
    # dPbus_dV, dQbus_dV = compute_jacobian_dPQ_dV(V_mean, env.Ybus, num_buses, device)
    
    # # 平衡节点已在前面确定
    bus_Va_idx = np.delete(np.arange(num_buses), slack_bus)
    
    # # 支路雅可比矩阵
    # dPfbus_dV, dQfbus_dV = compute_branch_jacobian(
    #     V_mean, env.Yf, env.Cf, num_buses, bus_Va_idx, device
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
            dPbus_dV_i, dQbus_dV_i = compute_jacobian_dPQ_dV(V_sample, env.Ybus, num_buses, device)
            dPfbus_dV_i, dQfbus_dV_i = compute_branch_jacobian(
                V_sample, env.Yf, env.Cf, num_buses, bus_Va_idx, device
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
                env.gen_bus_idx,
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
                S_max_np = env.S_max.cpu().numpy() if isinstance(env.S_max, torch.Tensor) else env.S_max
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
        P_after, Q_after = compute_power_injection(Vm_corrected, Va_corrected, env.G, env.B)
        Sf_after, _, Pf_after, Qf_after, _, _ = compute_branch_power(
            Vm_corrected, Va_corrected, env.Gf, env.Bf, env.Gt, env.Bt, env.Cf, env.Ct
        )
        Pg_after, Qg_after = compute_generator_power(
            P_after, Q_after, Pd, Qd, 
            env.pd_bus_idx, env.qd_bus_idx, env.gen_bus_idx, device
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
                S_max_np = env.S_max.cpu().numpy() if isinstance(env.S_max, torch.Tensor) else env.S_max
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


def compute_tangent_projection_batch(z, x_input, env, single_target=True):
    """
    批量计算约束切空间投影矩阵
    
    Args:
        z: 当前状态 (batch_size, output_dim) - 归一化的 [Vm, Va]
        x_input: 条件输入 (batch_size, input_dim)
        env: 电网环境对象
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
    
    # 获取环境参数
    Ybus = env.Ybus
    gen_bus_idx = env.gen_bus_idx
    
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


def compute_drift_correction_batch(z, x_input, env, lambda_cor=5.0):
    """
    批量计算 Drift-Correction：切向投影 + 法向修正
    
    核心公式：
        v_final = P_tan @ v_pred + correction
    
    Args:
        z: 当前状态 (batch_size, output_dim) - 归一化的 [Vm, Va]
        x_input: 条件输入 (batch_size, input_dim) - 包含负荷信息
        env: 电网环境对象
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
    
    # 获取环境参数
    Ybus = env.Ybus
    gen_bus_idx = env.gen_bus_idx
    pd_bus_idx = env.pd_bus_idx
    qd_bus_idx = env.qd_bus_idx
    
    # 获取约束限制（转换为 numpy）
    Pg_max = env.Pg_max.cpu().numpy() if hasattr(env.Pg_max, 'cpu') else np.array(env.Pg_max)
    Pg_min = env.Pg_min.cpu().numpy() if hasattr(env.Pg_min, 'cpu') else np.array(env.Pg_min)
    Qg_max = env.Qg_max.cpu().numpy() if hasattr(env.Qg_max, 'cpu') else np.array(env.Qg_max)
    Qg_min = env.Qg_min.cpu().numpy() if hasattr(env.Qg_min, 'cpu') else np.array(env.Qg_min)
    
    # 提取负荷信息
    num_pd = env.num_pd
    num_qd = env.num_qd
    
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