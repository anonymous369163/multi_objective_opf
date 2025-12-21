import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm 
import math
import numpy as np
torch.set_default_dtype(torch.float32)




class Actor(nn.Module):
    def __init__(self, input_dim, env=None, output_dim=118, norm=False):
        """
        初始化Actor网络
        
        Args:
            input_dim: 输入维度
            env: PowerGridEnv环境实例，包含电力系统相关参数
            norm: 是否使用层归一化
        """
        super(Actor, self).__init__()
        
        self.output_dim = output_dim
        # 从env中提取电力系统参数
        if env is not None:
            self.G = env.G
            self.B = env.B
            self.Gf = env.Gf
            self.Bf = env.Bf
            self.Cf = env.Cf
            self.Gt = env.Gt
            self.Bt = env.Bt
            self.Ct = env.Ct
            self.Ybus = env.Ybus
            
            # 负荷参数
            self.num_pd = env.num_pd
            self.num_qd = env.num_qd
            self.pd_bus_idx = env.pd_bus_idx
            self.qd_bus_idx = env.qd_bus_idx
            
            # 发电机参数
            self.Pg_bus_idx = env.Pg_bus_idx
            self.gen_bus_idx = env.gen_bus_idx
            
            # 约束参数
            self.ramp = env.ramp
            self.Pg_max = env.Pg_max
            self.Pg_min = env.Pg_min
            self.Qg_max = env.Qg_max
            self.Qg_min = env.Qg_min
            self.S_max = env.S_max
            
            # 并联补偿器(shunt)参数
            self.has_shunt = hasattr(env, 'has_shunt') and env.has_shunt
            if self.has_shunt:
                self.shunt_bus_idx = torch.from_numpy(env.shunt_bus_idx).long().to(self.G.device)
                self.shunt_q_base = torch.from_numpy(env.shunt_q_mvar_base / 100).float().to(self.G.device)  # 转换为p.u.
                self.shunt_p_base = torch.from_numpy(env.shunt_p_mw_base / 100).float().to(self.G.device)  # 转换为p.u.
        
        # 网络层定义
        self.vm1 = nn.Linear(input_dim, 512)
        self.vm2 = nn.Linear(512, 256)
        self.vm3 = nn.Linear(256, 128)
        self.vm4 = nn.Linear(128, self.output_dim)
        self.va1 = nn.Linear(input_dim, 512)
        self.va2 = nn.Linear(512, 256)
        self.va3 = nn.Linear(256, 128)
        self.va4 = nn.Linear(128, self.output_dim)

        if norm:
            self.layer_norm(self.vm1)
            self.layer_norm(self.vm2)
            self.layer_norm(self.vm3)
            self.layer_norm(self.vm4)
            self.layer_norm(self.va1)
            self.layer_norm(self.va2)
            self.layer_norm(self.va3)
            self.layer_norm(self.va4)
    # init parameter of network
    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)       # 常用正交初始化
        torch.nn.init.constant_(layer.bias, bias_const)
    # forward
    def forward(self, state):
        vm = torch.relu(self.vm1(state))
        vm = torch.relu(self.vm2(vm))
        vm = torch.relu(self.vm3(vm))
        vm = torch.tanh(self.vm4(vm))

        va = torch.relu(self.va1(state))
        va = torch.relu(self.va2(va))
        va = torch.relu(self.va3(va))
        va = torch.tanh(self.va4(va))
        
        return vm, va

    def act(self, state, out_ma=False):
        # out_ma 用于判断是否输出电压Vm和相角Va，如果为True，则输出电压Vm和相角Va，否则输出动作action
        Vm_o, Va_o = self.forward(state)
        Vm = torch.t(Vm_o)
        Va = torch.t(Va_o)
        Vm = Vm * 0.06 + 1
        Va = Va * math.pi / 6
        Vreal = Vm * torch.cos(Va)
        Vimg = Vm * torch.sin(Va)
        Ireal = torch.matmul(self.G, Vreal) - torch.matmul(self.B, Vimg)
        Iimg = torch.matmul(self.B, Vreal) + torch.matmul(self.G, Vimg)
        P = Vreal * Ireal + Vimg * Iimg
        Pd = state.T[: self.num_pd]
        Pg = P
        Pg[self.pd_bus_idx] = Pg[self.pd_bus_idx] + Pd
        Pg = Pg[self.Pg_bus_idx].T
        Vg = Vm[self.gen_bus_idx].T
        action = torch.cat((Vg, Pg), dim=1)  # 组合后单位是[电压（p.u.），功率（p.u.）]
        if not out_ma:
            return action
        else:
            return action, Vm_o, Va_o

    def test(self, state, Vm=None, Va=None):   
        if state.dim() == 1:
            state = state.unsqueeze(0) 
        if Vm is None or Va is None:
            Vm, Va = self.forward(state)
        else:
            # 确保Vm和Va都是二维张量
            if Vm.dim() == 1:
                Vm = Vm.unsqueeze(0)
            if Va.dim() == 1:
                Va = Va.unsqueeze(0)
        Vm = torch.t(Vm)
        Va = torch.t(Va)
        Vm = Vm * 0.06 + 1
        Va = Va * math.pi / 6
        Vreal = Vm * torch.cos(Va)
        Vimg = Vm * torch.sin(Va)
        Ireal = torch.matmul(self.G, Vreal) - torch.matmul(self.B, Vimg)
        Iimg = torch.matmul(self.B, Vreal) + torch.matmul(self.G, Vimg)
        P = Vreal * Ireal + Vimg * Iimg
        Pd = state.T[: self.num_pd]
        Pg = P
        Pd = Pd if isinstance(Pd, torch.Tensor) else torch.tensor(Pd, dtype=torch.float32).to(P.device)
        Pg[self.pd_bus_idx] = Pg[self.pd_bus_idx] + Pd
        Pg = Pg[self.Pg_bus_idx].T * 100
        Vg = Vm[self.gen_bus_idx].T
        action = torch.cat((Vg, Pg), dim=1)  # 组合后单位是[电压（p.u.），功率（p.u.）]
        return action

    # calc power flow
    def pf(self, Vm, Va):
        Vm = torch.t(Vm)
        Va = torch.t(Va)
        Vm = Vm * 0.06 + 1   # 还原会原来的值（以pu为单位，实际值为pu*0.06+1）
        Va = Va * math.pi / 6
        Vreal = Vm * torch.cos(Va)
        Vimg = Vm * torch.sin(Va)
        Ireal = torch.matmul(self.G, Vreal) - torch.matmul(self.B, Vimg)
        Iimg = torch.matmul(self.B, Vreal) + torch.matmul(self.G, Vimg)
        P = Vreal * Ireal + Vimg * Iimg
        Q = - Vreal * Iimg + Vimg * Ireal

        Ifreal = torch.matmul(self.Gf, Vreal) - torch.matmul(self.Bf, Vimg)
        Ifimg = torch.matmul(self.Bf, Vreal) + torch.matmul(self.Gf, Vimg)
        Vfreal = torch.matmul(self.Cf, Vreal)
        Vfimg = torch.matmul(self.Cf, Vimg)
        Pf = Vfreal * Ifreal + Vfimg * Ifimg
        Qf = - Vfreal * Ifimg + Vfimg * Ifreal
        Sf = torch.sqrt(torch.square(Pf) + torch.square(Qf))

        Itreal = torch.matmul(self.Gt, Vreal) - torch.matmul(self.Bt, Vimg)
        Itimg = torch.matmul(self.Bt, Vreal) + torch.matmul(self.Gt, Vimg)
        Vtreal = torch.matmul(self.Ct, Vreal)
        Vtimg = torch.matmul(self.Ct, Vimg)
        Pt = Vtreal * Itreal + Vtimg * Itimg
        Qt = - Vtreal * Itimg + Vtimg * Itreal
        St = torch.sqrt(torch.square(Pt) + torch.square(Qt))
        return P.T, Q.T, Sf.T, St.T

    def pf_deepopf(self, Vm_scaled, Va_scaled, scale_vm=10.0, scale_va=10.0, VmLb=None, VmUb=None):
        """
        DeepOPF 风格的潮流计算
        
        与 DeepOPV-V.ipynb 的标准化方式一致:
        - Vm: 从 [0, scale_vm] 还原到真实 p.u. 值
        - Va: 从缩放值还原到弧度
        
        Args:
            Vm_scaled: 缩放后的电压幅值 (batch_size, n_bus), 范围 [0, scale_vm]
            Va_scaled: 缩放后的电压相角 (batch_size, n_bus), 包含 slack bus (相角=0)
            scale_vm: Vm 缩放因子 (默认 10)
            scale_va: Va 缩放因子 (默认 10)
            VmLb: Vm 下界 tensor (n_bus,)
            VmUb: Vm 上界 tensor (n_bus,)
        
        Returns:
            P, Q, Sf, St: 与 pf 函数相同的输出
        """
        # 如果没有提供边界，使用典型值
        if VmLb is None:
            VmLb = torch.ones(Vm_scaled.shape[1], device=Vm_scaled.device) * 0.94
        if VmUb is None:
            VmUb = torch.ones(Vm_scaled.shape[1], device=Vm_scaled.device) * 1.06
        
        # 确保边界在正确的设备上
        VmLb = VmLb.to(Vm_scaled.device)
        VmUb = VmUb.to(Vm_scaled.device)
        
        # 反归一化 Vm: (Vm_scaled / scale_vm) * (VmUb - VmLb) + VmLb
        Vm_normalized = Vm_scaled / scale_vm  # [0, 1]
        Vm = Vm_normalized * (VmUb - VmLb) + VmLb  # 真实 p.u. 值
        
        # 反归一化 Va: Va_scaled / scale_va
        Va = Va_scaled / scale_va  # 弧度
        
        # 转置为 (n_bus, batch_size)
        Vm = torch.t(Vm)
        Va = torch.t(Va)
        
        # 计算复电压
        Vreal = Vm * torch.cos(Va)
        Vimg = Vm * torch.sin(Va)
        
        # 计算节点注入功率
        Ireal = torch.matmul(self.G, Vreal) - torch.matmul(self.B, Vimg)
        Iimg = torch.matmul(self.B, Vreal) + torch.matmul(self.G, Vimg)
        P = Vreal * Ireal + Vimg * Iimg
        Q = - Vreal * Iimg + Vimg * Ireal

        # 计算支路潮流 (from)
        Ifreal = torch.matmul(self.Gf, Vreal) - torch.matmul(self.Bf, Vimg)
        Ifimg = torch.matmul(self.Bf, Vreal) + torch.matmul(self.Gf, Vimg)
        Vfreal = torch.matmul(self.Cf, Vreal)
        Vfimg = torch.matmul(self.Cf, Vimg)
        Pf = Vfreal * Ifreal + Vfimg * Ifimg
        Qf = - Vfreal * Ifimg + Vfimg * Ifreal
        Sf = torch.sqrt(torch.square(Pf) + torch.square(Qf))

        # 计算支路潮流 (to)
        Itreal = torch.matmul(self.Gt, Vreal) - torch.matmul(self.Bt, Vimg)
        Itimg = torch.matmul(self.Bt, Vreal) + torch.matmul(self.Gt, Vimg)
        Vtreal = torch.matmul(self.Ct, Vreal)
        Vtimg = torch.matmul(self.Ct, Vimg)
        Pt = Vtreal * Itreal + Vtimg * Itimg
        Qt = - Vtreal * Itimg + Vtimg * Itreal
        St = torch.sqrt(torch.square(Pt) + torch.square(Qt))
        
        return P.T, Q.T, Sf.T, St.T

    def compute_constraint_loss(self, Vm, Va, batch_inputs, env, reduction='none', return_details=False, debug_mode=False, 
                                 use_deepopf_norm=False, deepopf_params=None):
        """
        根据模型输出的电压和相角计算约束违反损失
        
        Args:
            Vm: 电压幅值输出 (batch_size, num_buses)
            Va: 电压相角输出 (batch_size, num_buses)
            batch_inputs: 输入数据 (batch_size, input_dim)，包含 Pd, Qd, Pg_
            env: 环境对象，提供约束参数
            return_details: 是否返回详细的约束违反信息
            reduction: 'mean' 返回所有样本的平均损失（标量），'none' 返回每个样本的损失（向量）
            use_deepopf_norm: 是否使用 DeepOPF 标准化方式
            deepopf_params: DeepOPF 参数字典，包含 scale_vm, scale_va, VmLb, VmUb
            
        Returns:
            如果 return_details=False: 返回 loss_cons (标量或向量，取决于reduction)
            如果 return_details=True: 返回 (loss_cons, details_dict)
        """
        # 计算潮流
        if use_deepopf_norm and deepopf_params is not None:
            # 使用 DeepOPF 风格的标准化
            P, Q, Sf, St = self.pf_deepopf(
                Vm, Va, 
                scale_vm=deepopf_params.get('scale_vm', 10.0),
                scale_va=deepopf_params.get('scale_va', 10.0),
                VmLb=deepopf_params.get('VmLb'),
                VmUb=deepopf_params.get('VmUb')
            )
        else:
            # 使用原有的标准化方式
            P, Q, Sf, St = self.pf(Vm, Va) 
        
        # 从输入中提取负荷和上一时刻发电
        Pd = batch_inputs.T[: env.num_pd]
        Qd = batch_inputs.T[env.num_pd : env.num_pd + env.num_qd] 
        
        # 获取各节点的功率（这是母线注入功率，包括发电和负荷）
        Pg = P.T  # (num_buses, batch_size)
        Qg = Q.T 

        # 获取节点索引（转换为torch tensor以确保兼容性）
        pd_bus_idx = torch.from_numpy(env.pd_bus_idx).long().to(Pg.device)
        qd_bus_idx = torch.from_numpy(env.qd_bus_idx).long().to(Qg.device)
        Pg_bus_idx = torch.from_numpy(env.Pg_bus_idx).long().to(Pg.device)
        gen_bus_idx = torch.from_numpy(env.gen_bus_idx).long().to(Pg.device)

        # 在负荷节点加上负荷（使用index_add避免重复索引问题）
        # 注意：如果有多个负荷在同一母线，需要累加
        
        Pg.index_add_(0, pd_bus_idx, Pd)
        Qg.index_add_(0, qd_bus_idx, Qd)

        # 计算各项约束违反
        p_max = torch.clamp(Pg[gen_bus_idx].T - env.Pg_max, 0)
        p_min = torch.clamp(env.Pg_min - Pg[gen_bus_idx].T, 0)
        # pg_up = torch.clamp(Pg[Pg_bus_idx].T - Pg_up, 0)
        # pg_down = torch.clamp(Pg_down - Pg[Pg_bus_idx].T, 0)
        q_max = torch.clamp(Qg[gen_bus_idx].T - env.Qg_max, 0)
        q_min = torch.clamp(env.Qg_min - Qg[gen_bus_idx].T, 0)
        sf = torch.clamp(Sf - env.S_max, 0)
        st = torch.clamp(St - env.S_max, 0)
        
        # 计算每个样本的约束违反总和 (单位: p.u., 与 DeepOPF 对齐，不乘放大因子)
        g1 = torch.sum(p_max, dim=1)  # 发电机有功功率最大约束违反
        g2 = torch.sum(p_min, dim=1)  # 发电机有功功率最小约束违反
        # g3 = torch.sum(pg_up, dim=1)
        # g4 = torch.sum(pg_down, dim=1)
        g5 = torch.sum(q_max, dim=1)  # 发电机无功功率最大约束违反
        g6 = torch.sum(q_min, dim=1)  # 发电机无功功率最小约束违反
        g9 = torch.sum(sf, dim=1)     # 支路潮流约束违反 (from)
        g10 = torch.sum(st, dim=1)    # 支路潮流约束违反 (to)  
        
        # 计算总约束损失
        # loss_cons = torch.mean(g1 + g2 + g3 + g4 + g5 + g6 + g9 + g10)
        constraint_per_sample = g1 + g2 + g5 + g6 + g9 + g10
        if reduction == 'mean':
            loss_cons = torch.mean(constraint_per_sample)
        elif reduction == 'none':
            loss_cons = constraint_per_sample
        elif reduction == 'individual':
            loss_cons = [g1, g2, g5, g6, g9, g10], torch.mean(constraint_per_sample)
        else:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'none' or 'individual'.")
        if debug_mode:
            # 打印所有约束违反项，只显示小数点后三位
            print(f"g1_pmax: {g1.cpu().numpy().round(3)}")
            print(f"g2_pmin: {g2.cpu().numpy().round(3)}")
            # print(f"g3_ramp_up: {g3.cpu().numpy().round(3)}")
            # print(f"g4_ramp_down: {g4.cpu().numpy().round(3)}")
            print(f"g5_qmax: {g5.cpu().numpy().round(3)}")
            print(f"g6_qmin: {g6.cpu().numpy().round(3)}")
            print(f"g9_sf: {g9.cpu().numpy().round(3)}")
            print(f"g10_st: {g10.cpu().numpy().round(3)}")
            
        
        if return_details:
            # 返回详细的约束违反信息
            details = {
                'g1_pmax': torch.mean(g1).item(),
                'g2_pmin': torch.mean(g2).item(),
                # 'g3_ramp_up': torch.mean(g3).item(),
                # 'g4_ramp_down': torch.mean(g4).item(),
                'g5_qmax': torch.mean(g5).item(),
                'g6_qmin': torch.mean(g6).item(),
                'g9_sf': torch.mean(g9).item(),
                'g10_st': torch.mean(g10).item(),
                # 加入Pg，Qg, P 和 Q的输出
                # 'Pg': Pg[gen_bus_idx].detach().cpu().numpy().T,
                # 'Qg': Qg[gen_bus_idx].detach().cpu().numpy().T,
                # 'P': P_original.detach().cpu().numpy(),  # 使用原始的P，未加Pd
                # 'Q': Q_original.detach().cpu().numpy(),  # 使用原始的Q，未加Qd
            }
            return loss_cons, details
        else:
            return loss_cons

    def compute_deepopf_metrics(self, Vm, Va, batch_inputs, env, use_deepopf_norm=False, deepopf_params=None, delta=1e-4):
        """
        计算 DeepOPF 风格的评估指标
        
        与 DeepOPV-V.ipynb 中的 get_vioPQg 函数对齐:
        - 约束满足率百分比
        - 违反的发电机数量
        - 平均/最大违反量
        
        Args:
            Vm: 电压幅值输出 (batch_size, num_buses)
            Va: 电压相角输出 (batch_size, num_buses)
            batch_inputs: 输入数据 (batch_size, input_dim)
            env: 环境对象
            use_deepopf_norm: 是否使用 DeepOPF 标准化
            deepopf_params: DeepOPF 参数
            delta: 判断违反的阈值 (默认 1e-4，与 DeepOPF 一致)
        
        Returns:
            metrics: 字典，包含各类评估指标
        """
        # 计算潮流
        if use_deepopf_norm and deepopf_params is not None:
            P, Q, Sf, St = self.pf_deepopf(
                Vm, Va,
                scale_vm=deepopf_params.get('scale_vm', 10.0),
                scale_va=deepopf_params.get('scale_va', 10.0),
                VmLb=deepopf_params.get('VmLb'),
                VmUb=deepopf_params.get('VmUb')
            )
        else:
            P, Q, Sf, St = self.pf(Vm, Va)
        
        # 从输入中提取负荷
        Pd = batch_inputs.T[: env.num_pd]
        Qd = batch_inputs.T[env.num_pd : env.num_pd + env.num_qd]
        
        # 获取发电功率
        Pg = P.T.clone()
        Qg = Q.T.clone()
        
        # 获取节点索引
        pd_bus_idx = torch.from_numpy(env.pd_bus_idx).long().to(Pg.device)
        qd_bus_idx = torch.from_numpy(env.qd_bus_idx).long().to(Qg.device)
        gen_bus_idx = torch.from_numpy(env.gen_bus_idx).long().to(Pg.device)
        
        # 加上负荷
        Pg.index_add_(0, pd_bus_idx, Pd)
        Qg.index_add_(0, qd_bus_idx, Qd)
        
        # 获取发电机节点的功率
        Pg_gen = Pg[gen_bus_idx].T  # (batch_size, num_gen)
        Qg_gen = Qg[gen_bus_idx].T
        
        batch_size = Pg_gen.shape[0]
        num_gen = Pg_gen.shape[1]
        num_branch = Sf.shape[1]
        
        # ============ 计算约束违反 ============
        # Pg 约束
        pg_max_vio = torch.clamp(Pg_gen - env.Pg_max, min=0)  # 超出上限
        pg_min_vio = torch.clamp(env.Pg_min - Pg_gen, min=0)  # 低于下限
        
        # Qg 约束
        qg_max_vio = torch.clamp(Qg_gen - env.Qg_max, min=0)
        qg_min_vio = torch.clamp(env.Qg_min - Qg_gen, min=0)
        
        # 支路潮流约束
        sf_vio = torch.clamp(Sf - env.S_max, min=0)
        st_vio = torch.clamp(St - env.S_max, min=0)
        
        # ============ DeepOPF 风格的统计 ============
        # 违反数量统计 (使用阈值 delta)
        pg_max_vio_count = (pg_max_vio > delta).sum(dim=1).float()  # 每个样本违反 Pg_max 的发电机数
        pg_min_vio_count = (pg_min_vio > delta).sum(dim=1).float()
        qg_max_vio_count = (qg_max_vio > delta).sum(dim=1).float()
        qg_min_vio_count = (qg_min_vio > delta).sum(dim=1).float()
        sf_vio_count = (sf_vio > delta).sum(dim=1).float()
        st_vio_count = (st_vio > delta).sum(dim=1).float()
        
        # 满足率百分比 (与 DeepOPF 一致: (1 - 违反数/总数) * 100)
        pg_satisfy_rate = (1 - (pg_max_vio_count + pg_min_vio_count) / num_gen) * 100
        qg_satisfy_rate = (1 - (qg_max_vio_count + qg_min_vio_count) / num_gen) * 100
        branch_satisfy_rate = (1 - (sf_vio_count + st_vio_count) / (num_branch * 2)) * 100
        
        # 违反量统计 (p.u.)
        total_pg_vio = torch.sum(pg_max_vio + pg_min_vio, dim=1)
        total_qg_vio = torch.sum(qg_max_vio + qg_min_vio, dim=1)
        total_sf_vio = torch.sum(sf_vio + st_vio, dim=1)
        total_constraint_vio = total_pg_vio + total_qg_vio + total_sf_vio
        
        # 构建结果字典
        metrics = {
            # 总违反量 (p.u.)
            'total_violation_pu': total_constraint_vio.mean().item(),
            'pg_violation_pu': total_pg_vio.mean().item(),
            'qg_violation_pu': total_qg_vio.mean().item(),
            'branch_violation_pu': total_sf_vio.mean().item(),
            
            # 满足率百分比 (%)
            'pg_satisfy_rate': pg_satisfy_rate.mean().item(),
            'qg_satisfy_rate': qg_satisfy_rate.mean().item(),
            'branch_satisfy_rate': branch_satisfy_rate.mean().item(),
            
            # 平均违反的发电机数量
            'avg_pg_max_vio_count': pg_max_vio_count.mean().item(),
            'avg_pg_min_vio_count': pg_min_vio_count.mean().item(),
            'avg_qg_max_vio_count': qg_max_vio_count.mean().item(),
            'avg_qg_min_vio_count': qg_min_vio_count.mean().item(),
            
            # 最大违反量 (p.u.)
            'max_pg_violation': pg_max_vio.max().item() if pg_max_vio.max() > 0 else 0,
            'max_qg_violation': qg_max_vio.max().item() if qg_max_vio.max() > 0 else 0,
            'max_sf_violation': sf_vio.max().item() if sf_vio.max() > 0 else 0,
            
            # 总发电机数和支路数
            'num_generators': num_gen,
            'num_branches': num_branch,
        }
        
        return metrics

    def compute_economic_cost(self, Vm, Va, batch_inputs, env, reduction='mean'):
        """
        根据模型输出的电压和相角计算经济成本
        与 env._calculate_reward 中的成本计算方式一致
        
        Args:
            Vm: 电压幅值输出 (batch_size, num_buses)
            Va: 电压相角输出 (batch_size, num_buses)
            batch_inputs: 输入数据 (batch_size, input_dim)，包含 Pd, Qd
            env: 环境对象，提供成本系数
            reduction: 'mean' 返回平均成本，'none' 返回每个样本的成本
            
        Returns:
            economic_cost: 经济成本（$/h）
        """
        import math
        
        # 计算潮流
        P, Q, Sf, St = self.pf(Vm, Va)
        
        # 从输入中提取负荷
        Pd = batch_inputs.T[: env.num_pd]
        
        # 获取各节点的功率
        Pg = P.T  # (num_buses, batch_size)
        
        # 获取节点索引
        pd_bus_idx = torch.from_numpy(env.pd_bus_idx).long().to(Pg.device)
        gen_bus_idx = torch.from_numpy(env.gen_bus_idx).long().to(Pg.device)
        
        # 在负荷节点加上负荷（得到发电机功率）
        Pg.index_add_(0, pd_bus_idx, Pd)
        
        # 获取发电机节点的功率 (num_gen, batch_size)
        Pg_gen = Pg[gen_bus_idx]  # p.u.
        
        # 转换为 MW（乘以100，因为基准功率是100MVA）
        Pg_gen_mw = Pg_gen * 100  # MW
        
        # 获取成本系数
        # 从 env.net.poly_cost 获取发电机成本系数
        batch_size = Vm.shape[0]
        total_costs = torch.zeros(batch_size, device=Vm.device)
        
        # 1. 计算发电机成本
        if hasattr(env.net, 'poly_cost') and len(env.net.poly_cost) > 0:
            pc = env.net.poly_cost
            pc_gen = pc[pc.et == 'gen']
            
            for i in range(len(env.net.gen)):
                gen_cost_data = pc_gen[pc_gen.element == i]
                if len(gen_cost_data) > 0:
                    cp2 = gen_cost_data.cp2_eur_per_mw2.values[0]
                    cp1 = gen_cost_data.cp1_eur_per_mw.values[0]
                    
                    # 获取该发电机的功率
                    p_g = Pg_gen_mw[i]  # (batch_size,)
                    
                    # 计算成本: cp2 * p^2 + cp1 * p
                    gen_cost = cp2 * p_g**2 + cp1 * p_g
                    total_costs += gen_cost
        
        # 2. 计算外接电源（平衡机）成本
        # 平衡机功率需要从功率平衡计算得到
        # P_ext = 总负荷 + 损耗 - 其他发电机出力
        # 这里简化处理：从节点69（case118的平衡节点）获取功率
        if hasattr(env.net, 'ext_grid') and len(env.net.ext_grid) > 0:
            ext_bus_idx = env.net.ext_grid.bus.values
            for i, bus_idx in enumerate(ext_bus_idx):
                # 获取平衡机功率
                P_ext = Pg[bus_idx] * 100  # MW, (batch_size,)
                
                # 获取成本系数
                pc_ext = env.net.poly_cost[env.net.poly_cost.et == 'ext_grid']
                ext_cost_data = pc_ext[pc_ext.element == i]
                if len(ext_cost_data) > 0:
                    cp2 = ext_cost_data.cp2_eur_per_mw2.values[0]
                    cp1 = ext_cost_data.cp1_eur_per_mw.values[0]
                    
                    ext_cost = cp2 * P_ext**2 + cp1 * P_ext
                    total_costs += ext_cost
        
        if reduction == 'mean':
            return torch.mean(total_costs)
        else:
            return total_costs

    @staticmethod
    def phi(a):
        a = torch.clamp(a, 0)
        return torch.square(a)




"""
Naive model Class
"""
class Simple_NN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, output_act, pred_type):
        super(Simple_NN, self).__init__()
        if network == 'mlp':
            self.net = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act) 
        elif network == 'att':
            self.net = ATT(input_dim, 1, hidden_dim, num_layers, output_act, pred_type=pred_type)
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def loss(self, y_pred, y_target):
        return self.criterion(y_pred, y_target)


class GMM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, num_cluster, output_act, pred_type):
        super(GMM, self).__init__()
        self.num_cluster = num_cluster
        self.output_dim = output_dim
        if network == 'mlp':
            self.Predictor = MLP(input_dim, output_dim * num_cluster, hidden_dim, num_layers, output_act)
            self.Classifier = MLP(input_dim, num_cluster, hidden_dim, num_layers, 'gumbel', output_dim)
        elif network == 'att':
            self.Predictor = ATT(input_dim, num_cluster, hidden_dim, num_layers, output_act, pred_type=pred_type)
            self.Classifier = ATT(input_dim, num_cluster, hidden_dim, num_layers, 'gumbel', 1, pred_type=pred_type)
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()

    def forward(self, x):
        y_pred = self.Predictor(x).view(x.shape[0], -1, self.num_cluster)
        return y_pred

    def loss(self, x, y_pred, y_target):
        c_pred = self.Classifier(x, y_target).view(x.shape[0], -1, self.num_cluster)
        y_pred = (c_pred * y_pred).sum(-1)
        loss = self.criterion(y_pred, y_target)
        return loss
    
    def hindsight_loss(self, x, y_pred, y_target):
        loss = (y_pred - y_target.unsqueeze(-1))**2
        loss = loss.mean(dim=1)
        loss = loss.min(dim=1)[0]
        return loss.mean()


"""
VAE Encoder Class with Dual Input (x and y_target)
"""
import torch
import torch.nn as nn

class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, hidden_dim, num_layers, act='relu'):
        super().__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 'tanh': nn.Tanh()}
        act_fn = act_list[act]

        # x-branch
        self.x_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim)
        )

        # y-branch
        self.y_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim)
        )

        # FiLM 调制：用 y 特征调制 x 特征（也可反过来）
        self.gamma = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        self.beta  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))

        # 深层融合：Residual MLP
        blocks = []
        for _ in range(max(1, num_layers)):
            blocks += [
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.LayerNorm(hidden_dim)
            ]
        self.fusion_net = nn.Sequential(*blocks)

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y_target):
        x_feat = self.x_encoder(x)           # [B,H]
        y_feat = self.y_encoder(y_target)    # [B,H]

        # FiLM: 让 y 决定在条件 x 下需要的残差信息
        gamma = self.gamma(y_feat)           # [B,H]
        beta  = self.beta(y_feat)            # [B,H]
        fused = gamma * x_feat + beta        # [B,H]

        # 残差堆叠（小技巧：加一个 skip 更稳）
        deep = self.fusion_net(fused)
        deep = deep + fused

        mean   = self.mean_layer(deep)
        logvar = self.logvar_layer(deep)     # 训练时可 clamp：logvar = logvar.clamp(-20, 20)
        return mean, logvar
    
    def encode_from_condition(self, x):
        """
        只基于条件x进行编码，用于推理时（无需y_target）
        
        这样做的好处：
        1. 推理时可以利用encoder学到的条件信息
        2. 避免训练-推理不一致
        3. 生成的分布是 q(z|x) 而非简单的 p(z)=N(0,I)
        
        Args:
            x: [batch_size, input_dim] 条件输入（负荷、碳税等）
            
        Returns:
            mean: [batch_size, latent_dim] 条件分布的均值
            logvar: [batch_size, latent_dim] 条件分布的对数方差
        """
        x_feat = self.x_encoder(x)           # [B,H]
        # 不使用y_target，直接通过fusion_net处理x特征
        deep = self.fusion_net(x_feat)
        deep = deep + x_feat  # skip connection
        
        mean   = self.mean_layer(deep)
        logvar = self.logvar_layer(deep)
        return mean, logvar


"""
Generative Adversarial model Class
"""
class VAE(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type, use_cvae=True):
        super(VAE, self).__init__() 
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_cvae = use_cvae  # 是否使用条件VAE（Encoder同时看x和y）
        if network == 'mlp':
            if use_cvae:
                # 使用VAE_Encoder，能够同时处理x和y_target（标准CVAE）
                self.Encoder = VAE_Encoder(input_dim, output_dim, latent_dim, hidden_dim, num_layers, act='relu')
            else:
                # 仅基于条件x预测潜在分布（退化为条件生成模型）
                self.Encoder = MLP(input_dim, latent_dim*2, hidden_dim, num_layers, None)
            self.Decoder = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
        elif network == 'att':
            NotImplementedError
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()          

    def forward(self, x, z=None, use_mean=False):
        """
        VAE的前向传播（推理）
        
        Args:
            x: [batch_size, input_dim] 条件输入
            z: [batch_size, latent_dim] 可选的潜在向量
            use_mean: 是否使用均值而非采样（用于确定性推理）
            
        Returns:
            y_pred: [batch_size, output_dim] 预测输出
        """
        if z is None:
            # z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
            # 使用encoder基于条件x预测潜在分布
            if hasattr(self.Encoder, 'encode_from_condition'):
                mean, logvar = self.Encoder.encode_from_condition(x)
            else:
                para = self.Encoder(x)
                mean, logvar = torch.chunk(para, 2, dim=-1)
            
            if use_mean:
                # 确定性推理：直接使用均值
                z = mean
            else:
                # 随机推理：从预测的分布中采样
                z = self.reparameterize(mean, logvar)
        else:
            z = z.to(x.device)
        
        y_pred = self.Decoder(x, z)
        return y_pred

    def reparameterize(self, mean, logvar):
        z = torch.randn_like(mean).to(mean.device)
        return mean + z * torch.exp(0.5 * logvar)

    def encoder_decode(self, x, y_target=None):
        """
        编码-解码过程，将x和y_target编码到潜在空间，然后从潜在空间解码
        
        Args:
            x: [batch_size, input_dim] 条件输入
            y_target: [batch_size, output_dim] 目标值（电压和相角）
                      如果使用CVAE模式(use_cvae=True)，必须提供此参数
            
        Returns:
            y_recon: [batch_size, output_dim] 重构的目标值
            mean: [batch_size, latent_dim] 潜在分布的均值
            logvar: [batch_size, latent_dim] 潜在分布的对数方差
        """
        if self.use_cvae:
            # 标准CVAE：Encoder同时看x和y_target
            if y_target is None:
                raise ValueError("CVAE mode requires y_target for encoder_decode")
            mean, logvar = self.Encoder(x, y_target)
        else:
            # 条件生成模型：仅基于x预测潜在分布
            para = self.Encoder(x)
            mean, logvar = torch.chunk(para, 2, dim=-1)
        
        z = self.reparameterize(mean, logvar)
        y_recon = self.Decoder(x, z)
        return y_recon, mean, logvar

    def loss(self, y_recon, y_target, mean, logvar, beta=1.0):
        """
        VAE损失函数 = 重建损失 + beta * KL散度
        
        Args:
            y_recon: 重建的输出 [batch_size, output_dim]
            y_target: 目标输出 [batch_size, output_dim]
            mean: 潜在分布均值 [batch_size, latent_dim]
            logvar: 潜在分布对数方差 [batch_size, latent_dim]
            beta: KL散度权重，默认为1.0 (beta-VAE)
        """
        # 重建损失 (MSE)
        recon_loss = self.criterion(y_recon, y_target)
        
        # KL散度：先对latent_dim求和，再对batch取平均
        # KL(q(z|x,y) || p(z)) = -0.5 * sum(1 + log(var) - mean^2 - var)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        kl_div = torch.mean(kl_div)
        
        # 总损失 = 重建损失 + beta * KL散度
        return recon_loss + beta * kl_div
 
class GAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = MLP(input_dim, 1, hidden_dim, num_layers, 'sigmoid', output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, num_layers, 'sigmoid', latent_dim=1, agg=True)
        else:
            NotImplementedError
        self.criterion = nn.BCELoss()

    def forward(self, x, z):
        y_pred = self.Generator(x, z)
        return y_pred

    def loss_g(self, x, y_pred):
        valid = torch.ones([x.shape[0], 1]).to(x.device)
        return self.criterion(self.Discriminator(x, y_pred), valid)

    def loss_d(self, x, y_target, y_pred):
        valid = torch.ones([x.shape[0], 1]).to(x.device)
        fake = torch.zeros([x.shape[0], 1]).to(x.device)
        d_loss = (self.criterion(self.Discriminator(x, y_target), valid) +
                  self.criterion(self.Discriminator(x, y_pred.detach()), fake)) / 2
        return d_loss


class WGAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = Lip_MLP(input_dim, 1, hidden_dim, 1, None, output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, 1, None, latent_dim=1, agg=True,
                                     pred_type=pred_type)
        self.lambda_gp = 0.1

    def forward(self, x, z=None):
        if z is None:
            z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
        else:
            z = z.to(x.device)
        y_pred = self.Generator(x, z)
        return y_pred

    def loss_g(self, x, y_pred):
        return -torch.mean(self.Discriminator(x, y_pred))

    def loss_d(self, x, y_target, y_pred):
        w_dis_dual = torch.mean(self.Discriminator(x, y_pred.detach())) \
                     - torch.mean(self.Discriminator(x, y_target))
        return w_dis_dual 

    def loss_d_gp(self, x, y_target, y_pred):
        return self.loss_d(x, y_target, y_pred) + self.lambda_gp * self.gradient_penalty(x, y_target, y_pred)

    def gradient_penalty(self, x, y_target, y_pred):
        batch_size = x.shape[0]
        epsilon = torch.rand(batch_size, 1).expand_as(y_target).to(x.device)
        interpolated = epsilon * y_target + (1 - epsilon) * y_pred
        interpolated.requires_grad_(True)
        d_interpolated = self.Discriminator(x, interpolated)
        grad_outputs = torch.ones(d_interpolated.shape).to(x.device)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp


class DWGAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = Lip_MLP(input_dim, 1, hidden_dim, 1, 'abs', output_dim)
            # self.Discriminator = MLP(input_dim, 1,  hidden_dim, num_layers, output_act, output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, 1, None, latent_dim=1, agg=True,
                                     pred_type=pred_type)
        self.lambda_gp = 0.1

    def forward(self, x):
        z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
        v_pred = self.Generator(x, z)
        return z + v_pred

    def loss_g(self, x, y_pred):
        return torch.mean(self.Discriminator(x, y_pred))

    def loss_d(self, x, y_target, y_pred):
        w_dis_dual = torch.mean(self.Discriminator(x, y_pred.detach())) \
                     - torch.mean(self.Discriminator(x, y_target))
        return -w_dis_dual + 50 * (torch.mean(self.Discriminator(x, y_target))) ** 2

    def loss_d_gp(self, x, y_target, y_pred):
        return self.loss_d(x, y_target, y_pred) + self.lambda_gp * self.gradient_penalty(x, y_target, y_pred)

    def gradient_penalty(self, x, y_target, y_pred):
        batch_size = x.shape[0]
        epsilon = torch.rand(batch_size, 1).expand_as(y_target).to(x.device)
        interpolated = epsilon * y_target + (1 - epsilon) * y_pred
        interpolated.requires_grad_(True)
        d_interpolated = self.Discriminator(x, interpolated)
        grad_outputs = torch.ones(d_interpolated.shape).to(x.device)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp

"""
Generative Diffusion model Class
"""
class DM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(DM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.con_dim = input_dim
        self.time_step = time_step
        self.output_dim = output_dim
        beta_max = 0.02
        beta_min = 1e-4
        
        # Register schedule parameters as buffers so they move to GPU with model.to(device)
        # This eliminates CPU-GPU synchronization during inference
        betas = sigmoid_beta_schedule(self.time_step, beta_min, beta_max)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        
        self.criterion = nn.L1Loss()
        self.normalize = output_norm

    def predict_noise(self, x, y, t, noise):
        y_t = self.diffusion_forward(y, t, noise)
        noise_pred = self.model(x, y_t, t)
        return noise_pred

    def diffusion_forward(self, y, t, noise):
        """Optimized diffusion forward - no CPU-GPU sync needed."""
        if self.normalize:
            y = y * 2 - 1
        # Optimized: direct GPU indexing without CPU sync
        t_index = (t * self.time_step).long()
        alphas_1 = self.sqrt_alphas_cumprod[t_index.squeeze(-1)].unsqueeze(-1)
        alphas_2 = self.sqrt_one_minus_alphas_cumprod[t_index.squeeze(-1)].unsqueeze(-1)
        return (alphas_1 * y + alphas_2 * noise)

    def diffusion_backward(self, x, z, inf_step=100, eta=0.5):
        """
        Optimized diffusion backward sampling (DDPM/DDIM).
        All schedule parameters are registered as buffers and move to GPU with model,
        eliminating CPU-GPU synchronization overhead.
        """
        device = x.device
        
        if inf_step == self.time_step:
            """DDPM - Full step sampling"""
            for t in reversed(range(0, self.time_step)):
                noise = torch.randn_like(z)
                t_tensor = torch.ones(z.shape[0], 1, device=device) * t / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                # All schedule params are already on the same device as the model (registered as buffers)
                z = self.sqrt_recip_alphas[t] * (z - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) \
                    + torch.sqrt(self.posterior_variance[t]) * noise
        else: 
            """DDIM - Accelerated sampling with skip steps"""
            sample_time_step = torch.linspace(self.time_step - 1, 0, inf_step + 1, device=device).long()
            for i in range(1, inf_step + 1):
                t = sample_time_step[i - 1]
                prev_t = sample_time_step[i]
                noise = torch.randn_like(z)
                t_tensor = torch.ones(z.shape[0], 1, device=device) * t.float() / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                # Direct GPU indexing - no CPU-GPU sync needed (buffers are on same device)
                y_0 = (z - self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) / self.sqrt_alphas_cumprod[t]
                var = eta * self.posterior_variance[t]
                z = self.sqrt_alphas_cumprod[prev_t] * y_0 \
                    + torch.sqrt(torch.clamp(1 - self.alphas_cumprod[prev_t] - var, 0, 1)) * pred_noise \
                    + torch.sqrt(var) * noise
        
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

    def loss(self, noise_pred, noise):
        return self.criterion(noise_pred, noise)

    def predict_noise_with_anchor(self, x, y, t, noise, vae_anchor):
        """
        Predict noise with VAE anchor support
        
        This modifies the diffusion forward process to incorporate VAE prediction:
        Instead of diffusing from pure noise, we diffuse from VAE prediction.
        The residual (y - vae_anchor) represents what VAE missed.
        
        Args:
            x: condition input (batch_size, input_dim)
            y: target output (batch_size, output_dim)
            t: time step (batch_size, 1)
            noise: random noise (batch_size, output_dim)
            vae_anchor: VAE prediction (batch_size, output_dim)
            
        Returns:
            noise_pred: predicted noise
        """
        y_t = self.diffusion_forward_with_anchor(y, t, noise, vae_anchor)
        noise_pred = self.model(x, y_t, t)
        return noise_pred
    
    def diffusion_forward_with_anchor(self, y, t, noise, vae_anchor):
        """
        Modified diffusion forward with VAE anchor (optimized - no CPU-GPU sync)
        
        Standard diffusion: y_t = sqrt(alpha_t) * y + sqrt(1-alpha_t) * noise
        
        With VAE anchor: Instead of pure noise, we use (noise + residual) where
        residual = y - vae_anchor. This helps the diffusion model learn to 
        refine the VAE prediction.
        
        Args:
            y: target output (batch_size, output_dim)
            t: time step (batch_size, 1)
            noise: random noise (batch_size, output_dim)
            vae_anchor: VAE prediction (batch_size, output_dim)
            
        Returns:
            y_t: noisy sample at time t
        """
        if self.normalize:
            y = y * 2 - 1
            vae_anchor = vae_anchor * 2 - 1
        
        # Optimized: use gather for efficient GPU indexing without CPU sync
        t_index = (t * self.time_step).long()
        # Expand indices for gathering: (batch_size, 1) -> index into (time_step,) buffer
        alphas_1 = self.sqrt_alphas_cumprod[t_index.squeeze(-1)].unsqueeze(-1)
        alphas_2 = self.sqrt_one_minus_alphas_cumprod[t_index.squeeze(-1)].unsqueeze(-1)
        
        # Modified: blend between vae_anchor and y based on time, with noise
        residual = y - vae_anchor
        modified_noise = noise + 0.5 * residual  # Blend noise with VAE residual
        
        return (alphas_1 * y + alphas_2 * modified_noise)
    
    def diffusion_backward_with_anchor(self, x, z, vae_anchor, inf_step=100, eta=0.5, anchor_strength=0.3):
        """
        Optimized diffusion backward (sampling) starting from VAE anchor.
        All schedule parameters are registered as buffers, eliminating CPU-GPU sync overhead.
        
        Args:
            x: condition input (batch_size, input_dim)
            z: random noise (batch_size, output_dim)
            vae_anchor: VAE prediction (batch_size, output_dim)
            inf_step: number of inference steps
            eta: DDIM eta parameter
            anchor_strength: how much to weight the VAE anchor (0.0-1.0)
                           
        Returns:
            y: generated sample
        """
        device = x.device
        
        if self.normalize:
            vae_anchor_normalized = vae_anchor * 2 - 1
        else:
            vae_anchor_normalized = vae_anchor
        
        # Initialize z as blend of noise and VAE anchor
        z_start = z + anchor_strength * vae_anchor_normalized
        
        if inf_step == self.time_step:
            """DDPM with VAE anchor - Full step sampling"""
            z = z_start
            for t in reversed(range(0, self.time_step)):
                noise = torch.randn_like(z)
                t_tensor = torch.ones(z.shape[0], 1, device=device) * t / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                # Direct GPU indexing - buffers are already on same device as model
                z = self.sqrt_recip_alphas[t] * (z - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) \
                    + torch.sqrt(self.posterior_variance[t]) * noise
        else: 
            """DDIM with VAE anchor - Accelerated sampling"""
            z = z_start
            sample_time_step = torch.linspace(self.time_step - 1, 0, inf_step + 1, device=device).long()
            for i in range(1, inf_step + 1):
                t = sample_time_step[i - 1]
                prev_t = sample_time_step[i]
                noise = torch.randn_like(z)
                t_tensor = torch.ones(z.shape[0], 1, device=device) * t.float() / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                # Direct GPU indexing - no CPU-GPU sync needed (buffers are on same device)
                y_0 = (z - self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) / self.sqrt_alphas_cumprod[t]
                
                # Optional: blend with VAE anchor during denoising for guidance
                # y_0 = (1 - 0.1) * y_0 + 0.1 * vae_anchor_normalized
                
                var = eta * self.posterior_variance[t]
                z = self.sqrt_alphas_cumprod[prev_t] * y_0 \
                    + torch.sqrt(torch.clamp(1 - self.alphas_cumprod[prev_t] - var, 0, 1)) * pred_noise \
                    + torch.sqrt(var) * noise
        
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

class FM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(FM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        elif network == 'carbon_tax_aware_mlp':
            # 使用我们的条件MLP网络，专门处理[负荷, 碳税, 锚点]的条件输入
            self.model = CarbonTaxAwareMLP(input_dim, output_dim, hidden_dim, num_layers, None, 
                                          latent_dim=output_dim, carbon_tax_dim=1, anchor_dim=output_dim) 
        elif network == 'sdp_lip':
            self.model = SDP_MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)    # include SDPBasedLipschitzLinearLayer in mlp
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.min_sd = 0.01
        self.criterion = nn.L1Loss()
        self.normalize = output_norm

    def flow_forward(self, y, t, z, vec_type='gaussian', P_tan=None):
        """
        Flow 前向过程：计算插值点 y_t 和目标速度 vec
        
        Args:
            y: 目标解 (batch_size, output_dim)
            t: 时间步 (batch_size, 1)
            z: 起始点/锚点 (batch_size, output_dim)
            vec_type: 速度场类型
                - 'gaussian': 高斯噪声插值
                - 'conditional': 条件噪声插值
                - 'rectified': 直线插值（默认 Rectified Flow）
                - 'interpolation': 简单线性插值
                - 'riemannian': Riemannian Flow Matching（投影到切空间）
            P_tan: 切空间投影矩阵 (batch_size, output_dim, output_dim)
                   仅在 vec_type='riemannian' 时使用
        
        Returns:
            yt: 插值点 (batch_size, output_dim)
            vec: 目标速度 (batch_size, output_dim)
        """
        if self.normalize:
            y = 2 * y - 1  # [0,1] normalize to [-1,1]
        if vec_type == 'gaussian':
            """
            t = 0:  N(0, 1)
            t = 1:  N(y, sd)
            Linear interpolation for mu and sigma
            """
            mu = y * t
            sigma = (self.min_sd) * t + 1 * (1 - t)
            noise = torch.randn_like(y).to(y.device)
            yt = mu + noise * sigma
            vec = (y - (1 - self.min_sd) * yt) / (1 - (1 - self.min_sd) * t)
        elif vec_type == 'conditional':
            mu = t * y + (1 - t) * z
            sigma = ((self.min_sd * t) ** 2 + 2 * self.min_sd * t * (1 - t)) ** 0.5
            noise = torch.randn_like(y).to(y.device)
            yt = mu + noise * sigma
            vec = (y - (1 - self.min_sd) * yt) / (1 - (1 - self.min_sd) * t)
        elif vec_type == 'rectified':
            """
            t = 0:  N(0, 1)
            t = 1:  N(y, sd)
            Linear interpolation for z and y
            """
            yt = t * y + (1 - t) * z
            vec = y-z
        elif vec_type == 'interpolation':
            """
            t = 0:  x
            t = 1:  N(0,1)
            Linear interpolation for z and y
            """
            # yt = (1 - t) * y + t * z
            yt = t * y + (1 - t) * z
            vec = None
            # return torch.cos(torch.pi/2*t) * y + torch.sin(torch.pi/2*t) * z
            # return (torch.cos(torch.pi*t) + 1)/2 * y + (torch.cos(-torch.pi*t) +1)/2  * z
        elif vec_type == 'riemannian':
            """
            Riemannian Flow Matching (RFM)
            
            与 rectified 类似，但目标向量被投影到切空间：
            - yt = t * y + (1-t) * z （插值点）
            - raw_vec = y - z （直连向量）
            - vec = P_tan @ raw_vec （投影到切空间）
            
            这样模型学到的速度场天然"贴着可行流形走"，
            推理时漂移大幅减小，可以减少 Jacobian 修正频率。
            """
            yt = t * y + (1 - t) * z
            raw_vec = y - z
            
            if P_tan is not None:
                # 投影到切空间: vec = P_tan @ raw_vec
                raw_vec_expanded = raw_vec.unsqueeze(-1)  # (B, D, 1)
                vec = torch.bmm(P_tan, raw_vec_expanded).squeeze(-1)  # (B, D)
            else:
                # 如果没有提供 P_tan，回退到 rectified 模式
                vec = raw_vec
        else:
            raise NotImplementedError(f"Unknown vec_type: {vec_type}")
        return yt, vec

    def flow_backward(self, x, z, step=0.01, method='Euler', direction='forward', 
                     objective_fn=None, guidance_config=None, 
                     evolutionary_config=None, projection_config=None):
        """
        带梯度引导、演化算法增强和约束切空间投影的流模型反向采样
        
        Args:
            x: 条件输入
            z: 初始状态
            step: 步长
            method: ODE求解方法
            direction: 流动方向 ('forward' 或 'backward')
            objective_fn: 目标函数，用于计算引导梯度和投影约束
            guidance_config: 引导配置字典
            evolutionary_config: 演化算法配置字典（可选）
            projection_config: 约束切空间投影配置字典（可选），包含:
                - 'enabled': 是否启用投影
                - 'start_time': 开始投影的时间阈值
                - 'env': 电网环境对象
                - 'single_target': 是否为单目标模式
        
        Returns:
            最终采样结果，以及约束违反值（如果提供了objective_fn）
        """
        step = step if direction == 'forward' else -step
        t = 0 if direction == 'forward' else 1
        while (direction == 'forward' and t < 1) or (direction == 'backward' and t > 0): 
            z += ode_step(self.model, x, z, t, step, method, 
                        objective_fn=objective_fn, guidance_config=guidance_config, 
                        evolutionary_config=evolutionary_config, projection_config=projection_config)
            t += step
        
        # # 计算最终状态的约束违反值
        if objective_fn is not None:
            output_dim = z.shape[1]
            final_z = (z + 1) / 2 if self.normalize else z
            
            # 调用objective_fn计算平均损失（用于guidance）
            # 修复: 处理 guidance_config 为 None 的情况
            single_target = guidance_config.get('single_target', False) if guidance_config is not None else True
            x_real = x[:, :-1] if not single_target else x
            loss_value = objective_fn(
                final_z[:, :output_dim//2], 
                final_z[:, output_dim//2:], 
                x_real,
                'none'
            )

            constraint_violation = loss_value
                
            if self.normalize:
                return (z + 1) / 2, constraint_violation
            else:
                return z, constraint_violation 
        else: 
            if self.normalize:
                return (z + 1) / 2, None
            else:
                return z, None

    def predict_vec(self, x, yt, t):
        vec_pred = self.model(x, yt, t)
        # x_0 = self.model(x, yt, t)
        # vec_pred = (x_0 - yt)/(1-t+1e-5)
        return vec_pred

    def loss(self, y, z, vec_pred, vec, vec_type='gaussian'):
        if vec_type in ['gaussian', 'rectified', 'conditional', 'riemannian']:
            # riemannian 与 rectified 使用相同的 L1 损失，
            # 区别在于 vec 是投影后的目标向量
            return self.criterion(vec_pred, vec)
        elif vec_type in ['interpolation']:
            loss = 1 / 2 * torch.sum(vec_pred ** 2, dim=1, keepdim=True) \
                   - torch.sum((y - z) * vec_pred, dim=1, keepdim=True)
            return loss.mean()
            # loss = 1/2 * torch.sum(vec **2, dim=-1, keepdim=True) - torch.sum((-torch.pi/2*torch.sin(torch.pi/2*t) * y +  torch.pi/2*torch.cos(torch.pi/2*t) * z) * vec, dim=-1, keepdim=True)
            # loss = 1/2 * torch.sum(vec **2, dim=-1, keepdim=True) - torch.sum((-torch.pi*torch.sin(torch.pi*t)*y +    torch.pi*torch.sin(-torch.pi*t) * z) * vec, dim=-1, keepdim=True)
        else:
            raise NotImplementedError(f"Unknown vec_type: {vec_type}")

class AM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(AM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, 1, hidden_dim, num_layers, None, output_dim, 'silu')
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.min_sd = 0.01
        self.criterion = nn.MSELoss()
        self.normalize = output_norm
    
    def loss(self, x, y, z, t):
        s0 = self.model(x, y, torch.zeros_like(t))
        s1 = self.model(x, y, torch.ones_like(t))
        yt = t*y + (1-t)*z 
        yt.requires_grad = True
        st = self.model(x, yt, t)
        vec = torch.autograd.grad(st, yt, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        loss = self.criterion(vec, y-z)
        # loss =  1 / 2 * torch.sum(vec ** 2, dim=1, keepdim=True) \
        #            - torch.sum((y - z) * vec, dim=1, keepdim=True)
        # t.requires_grad = True
        # st = self.model(x, yt, t)
        # dsdt = torch.autograd.grad(st, t, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        # dsdy = torch.autograd.grad(st, yt, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        # loss = s0 - s1 + 0.5 * torch.sum(dsdy**2, dim=1, keepdim=True) + dsdt
        return loss.mean()

    def flow_backward(self, x, step=0.01, method='Euler', direction='forward'):
        z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device)
        step = step if direction == 'forward' else -step
        t = 0 if direction == 'forward' else 1
        while (direction == 'forward' and t < 1) or (direction == 'backward' and t > 0):
            z.requires_grad = True
            st = self.model(x, z, torch.ones(size=[z.shape[0],1]).to(x.device)*t)
            dsdz = torch.autograd.grad(st, z, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
            z.requires_grad = False
            z += dsdz.detach() * step
            t += step
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

class CM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(CM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
            # self.target_model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
            # self.target_model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.criterion = nn.MSELoss()
        self.normalize = output_norm
        self.std = 80
        self.eps = 0.002
    
    def flow_forward(self, y, z, t):
        yt = y + z * t * self.std
        # yt = (1-t) * y + z * t 
        return yt

    def predict(self, x, yt, t, model):
        return self.c_skip_t(t) * yt + self.c_out_t(t) * model(x, yt, t)

    def sampling(self, x, inf_step):
        z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) * self.std
        t = torch.ones(size=[x.shape[0],1]).to(x.device)
        y0 =  self.predict(x, z, t, self.target_model)
        return y0

    def loss(self, x, y, z, t1, t2, data, vec):
        # y = data.scaling_v(y)
        yt1 = self.flow_forward(y, z, t1)
        y01 = self.predict(x, yt1, t1, self.model)
        # y01_scale = data.scaling_v(y01)
        # x_scale = data.scaling_load(x)
        # pg, qg, vm, va = data.power_flow_v(x_scale, y01_scale)
        # pel = torch.abs(data.eq_resid(x_scale, torch.cat([pg,qg, vm,va],dim=1)))
        with torch.no_grad():
            yt2 = self.flow_forward(y, z, t2)
            y02 = self.predict(x, yt2, t2, self.target_model)
            # y02_scale = data.scaling_v(y02)
        return (self.criterion(y01, y02)).mean() #+ 0.0001*pel.mean()

    def c_skip_t(self, t):
        t = t * self.std
        return 0.25 / (t.pow(2) + 0.25)
    
    def c_out_t(self, t):
        t = t * self.std
        return 0.25 * t / ((t + self.eps).pow(2)).pow(0.5)

    def kerras_boundaries(self, sigma, eps, N, T):
        # This will be used to generate the boundaries for the time discretization
        return torch.tensor(
            [
                (eps ** (1 / sigma) + i / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma)))
                ** sigma
                for i in range(N)
            ]
        )

class CD(CM):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super().__init__(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type)
        self.criterion = nn.L1Loss()

    def loss(self, x, y, z, t1, step, forward_step, data, vec_model):
        # print(self.c_out_t(torch.zeros(1)), self.c_skip_t(torch.zeros(1)))
        yt1 = (1-t1) * z + t1 * y
        y01 = self.predict(x, yt1, t1, self.model)
        # y01_scale = data.scaling_v(y01)
        # x_scale = data.scaling_load(x)
        # pg, qg, vm, va = data.power_flow_v(x_scale, y01_scale)
        # pel = torch.abs(data.eq_resid(x_scale, torch.cat([pg,qg, vm,va],dim=1)))
        with torch.no_grad():
            v_pred_1 = vec_model.predict_vec(x, yt1, t1) * step
            yt2 = yt1 + v_pred_1
            t2 = t1 + step
            for _ in range(forward_step):
                v_pred_2 = vec_model.predict_vec(x, yt2, t2) * step
                yt2 = yt2 + v_pred_2
                t2 = t2 + step
            y02 = self.predict(x, yt2, t2, self.model)

            # v_pred_1 = vec_model.predict_vec(x, yt2, t2) * step
            # yt2 = yt1 + (v_pred_0 + v_pred_1)/2 
            # v_pred_0 = vec_model.predict_vec(x, yt1, t1) * step
            # v_pred_1 = vec_model.predict_vec(x, yt1 + v_pred_0 * 0.5, t1 + step * 0.5) * step
            # v_pred_2 = vec_model.predict_vec(x, yt1 + v_pred_1 * 0.5, t1 + step * 0.5) * step
            # v_pred_3 = vec_model.predict_vec(x, yt1 + v_pred_2, t1 + step) * step
            # yt2 = yt1 + (v_pred_0 + 2 * v_pred_1 + 2 * v_pred_2 + v_pred_3) / 6
        return (self.criterion(y01, y02)).mean() #+ 0.001*pel.mean()
    
    def continnuous_loss(self, x, y, z, t, stpe, forward_step, data, vec_model):
        yt = (1-t) * z + t * y
        with torch.no_grad():
            vec = vec_model.predict_vec(x, yt, t)
        yt.requires_grad_(True)
        t.requires_grad_(True)
        y0 = self.predict(x, yt, t, self.model)
        dy = torch.autograd.grad(y0, yt, torch.ones_like(y0), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        dt = torch.autograd.grad(y0, t, torch.ones_like(y0), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        return torch.square(dy * vec + dt).mean()



    def predict(self, x, yt, t, model):
        return yt + (1-t) * model(x, yt, t)

    def sampling(self, x, inf_step):
        # z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
        # t = torch.zeros(size=[x.shape[0],1]).to(x.device)
        # y0 =  self.predict(x, z, t, self.target_model)
        y0 = 0
        for dt in torch.linspace(0,1,1):
            z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
            t = torch.ones(size=[x.shape[0],1]).to(x.device) * dt
            yt = (1-t) * z + t * y0
            y0 =  self.predict(x, yt, t, self.model)
        # z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
        # yt1 = (y0+z)/2
        # y0 =  self.predict(x, yt1, t+0.5, self.target_model)
        return y0

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def ode_step(model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, t: float, step: float, 
              method: str = 'Euler', objective_fn=None, guidance_config=None,
              evolutionary_config=None, projection_config=None):
    """
    改进的ODE步进函数，支持基于论文的梯度引导、演化算法增强和约束切空间投影
    
    Args:
        model: 流模型
        x: 条件输入 (batch_size, input_dim) - [负荷特征, 碳税, 锚点]
        z: 当前状态 (batch_size, output_dim) - 对应论文中的 x_t
        t: 当前时间步
        step: 步长
        method: ODE求解方法
        objective_fn: 目标函数 l(x_0; G)，接受 (Vm, Va, 负荷特征, reduction)，返回约束损失
        guidance_config: 引导配置字典，包含:
            - 'enabled': 是否启用引导
            - 'scale': 引导强度
            - 'perp_scale': 垂直方向引导强度
        evolutionary_config: 演化算法配置字典（参见adaptive_evolutionary_guidance）
        projection_config: 约束切空间投影配置字典，包含:
            - 'enabled': 是否启用投影（默认False）
            - 'start_time': 开始投影的时间阈值（默认0.5）
            - 'env': 电网环境对象（必需）
            - 'single_target': 是否为单目标模式（默认True）
            - 'verbose': 是否打印调试信息
    
    Returns:
        v_pred: 预测的速度 * 步长
    """
    model.eval()
    t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t

    def model_eval(z_eval, t_eval):
        return model(x, z_eval, t_eval)

    # 统一转换为首字母大写，避免大小写不匹配问题
    method = method.capitalize()

    # 步骤1: 计算基础ODE步进（无引导）
    with torch.no_grad():
        if method == 'Euler':
            v_pred = model_eval(z, t_tensor) * step
        else:
            v_pred_0 = model_eval(z, t_tensor) * step
            if method == 'Heun':
                v_pred_1 = model_eval(z + v_pred_0, t_tensor + step) * step
                v_pred = (v_pred_0 + v_pred_1) / 2
            elif method == 'Mid':
                v_pred = model_eval(z + v_pred_0 * 0.5, t_tensor + step * 0.5) * step
            elif method == 'Rk4':
                v_pred_1 = model_eval(z + v_pred_0 * 0.5, t_tensor + step * 0.5) * step
                v_pred_2 = model_eval(z + v_pred_1 * 0.5, t_tensor + step * 0.5) * step
                v_pred_3 = model_eval(z + v_pred_2, t_tensor + step) * step
                v_pred = (v_pred_0 + 2 * v_pred_1 + 2 * v_pred_2 + v_pred_3) / 6
            else:
                # 如果method不在支持列表中，回退到Euler方法
                v_pred = v_pred_0
                
    # ============================================================================
    # 步骤1: 应用梯度引导（如果启用）- 使用预计算的约束
    # ============================================================================
    if (guidance_config is not None) and guidance_config.get('enabled', False):
        guidance_scale = guidance_config.get('scale', 0.1)
        perp_scale = guidance_config.get('perp_scale', 0.001)
        should_guidance = (t + step >= guidance_config.get('start_time', 0.8))
        if should_guidance and objective_fn is not None:
            # 使用requires_grad计算梯度
            z_for_grad = z.detach().requires_grad_(True)
            output_dim = z_for_grad.shape[1]
            
            # 计算梯度（需要带梯度） 
            x_real = x[:, :-1] if not guidance_config.get('single_target', False) else x
            constraint_violations = objective_fn(z_for_grad[:, :output_dim//2], z_for_grad[:, output_dim//2:], x_real, 'none')
            
            # 为每个样本计算个性化梯度 
            grad_z = torch.autograd.grad(
                constraint_violations,
                z_for_grad,
                grad_outputs=torch.ones_like(constraint_violations),
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]  
            
            # guidance 拆成两部分：沿基础流方向和垂直基础流方向的分量
            v_base = v_pred  # 基础流（不含引导）
            base_norm = v_base.norm(dim=1, keepdim=True)
            safe_mask = (base_norm > 1e-8).float()
            g = (-grad_z) * step * guidance_scale  # 原始引导
            v_base_norm = v_base / (base_norm + 1e-8)
            g_parallel = (g * v_base_norm).sum(dim=1, keepdim=True) * v_base_norm * safe_mask
            g_perp = (g - g_parallel) * safe_mask
            v_guidance = g_parallel + perp_scale * g_perp 
    else:
        constraint_violations = None
        v_guidance = torch.zeros_like(v_pred)
    # ============================================================================
    # 步骤1.5: 应用演化算法引导（如果启用）- 使用预计算的约束
    # ============================================================================
    if evolutionary_config is not None and evolutionary_config.get('enabled', False):
        if evolutionary_config.get('start_time', 0.8) <= t + step:
            if constraint_violations is None: 
                x_real = x[:, :-1] if not evolutionary_config.get('single_target', False) else x
                constraint_violations = objective_fn(z[:, :z.shape[1]//2], z[:, z.shape[1]//2:], x_real, 'none')
            v_pred = adaptive_evolutionary_guidance(
                z=z,
                v_pred=v_pred,
                v_guidance=v_guidance,
                t=t,
                step=step,
                x_input=x_real,
                objective_fn=objective_fn,
                evo_config=evolutionary_config,
                constraint_violations=constraint_violations  # 传递预计算的约束
            )
    else:
        v_pred = v_pred + v_guidance
    
    # ============================================================================
    # 步骤2: 应用 Drift-Correction 流形稳定化（如果启用）
    # 核心公式: v_final = P_tan @ v_pred + correction
    #   - P_tan @ v_pred: 切向投影（保持流动方向）
    #   - correction = -λ * F^+ @ f(x): 法向修正（拉回可行域） 
    #     'jacobian/sparse_jacobian': 使用 Jacobian 计算修正（精确但慢，O(N³)） 
    # ============================================================================
    if projection_config is not None and projection_config.get('enabled', False):
        start_time = projection_config.get('start_time', 0.5)
        should_project = (t + step >= start_time)
        
        if should_project:
            env = projection_config.get('env', None)
            correction_mode = projection_config.get('mode', 'jacobian')  # 'jacobian' / 'sparse_jacobian'
            
            if env is not None:
                single_target = projection_config.get('single_target', True)
                x_real = x if single_target else x[:, :-1]
                
                if correction_mode == 'sparse_jacobian':
                    # ============================================================
                    # 稀疏 Jacobian 模式：只在关键时间点使用 Jacobian
                    # 速度介于 Jacobian 和无修正之间，精度接近 Jacobian
                    # ============================================================
                    from post_processing import compute_drift_correction_batch, apply_drift_correction
                    
                    # 只在特定时间点应用 Jacobian 修正（如每隔 5 步）
                    sparse_interval = projection_config.get('sparse_interval', 5)
                    current_step = int((t - start_time) / step) if step > 0 else 0
                    
                    if current_step % sparse_interval == 0:
                        lambda_cor = projection_config.get('lambda_cor', 1.5)
                        P_tan, correction = compute_drift_correction_batch(z, x_real, env, lambda_cor)
                        
                        v_pred_before = v_pred.clone()
                        v_pred = apply_drift_correction(v_pred, P_tan, correction)
                        
                        if projection_config.get('verbose', False):
                            correction_norm = correction.norm(dim=1).mean().item()
                            print(f"  [SparseJac] t={t:.2f}, step={current_step}, applied, norm={correction_norm:.4f}")
                    else:
                        if projection_config.get('verbose', False):
                            print(f"  [SparseJac] t={t:.2f}, step={current_step}, skipped")
                
                if correction_mode == 'jacobian' and env is not None:
                    # ============================================================
                    # 使用 Jacobian 计算修正（精确模式）
                    # ============================================================
                    from post_processing import compute_drift_correction_batch, apply_drift_correction
                    
                    lambda_cor = projection_config.get('lambda_cor', 1.5)
                    
                    # 计算 Drift-Correction: 切向投影矩阵 + 法向修正向量
                    P_tan, correction = compute_drift_correction_batch(z, x_real, env, lambda_cor)
                    
                    # 应用 Drift-Correction
                    v_pred_before = v_pred.clone()
                    v_pred = apply_drift_correction(v_pred, P_tan, correction)
                    
                    if projection_config.get('verbose', False):
                        v_norm_before = v_pred_before.norm(dim=1).mean().item()
                        v_norm_after = v_pred.norm(dim=1).mean().item()
                        correction_norm = correction.norm(dim=1).mean().item()
                        print(f"  [Jacobian] t={t:.2f}, v_norm: {v_norm_before:.4f} -> {v_norm_after:.4f}, correction_norm={correction_norm:.4f}")
    
    return v_pred

# ============================================================================
# 演化算法增强模块 (Evolutionary Algorithm Enhancement)
# ============================================================================


def differential_evolution_guidance(z, v_pred, v_guidance, x_input, objective_fn, 
                                   F=0.5, CR=0.7, strategy='best/1', 
                                   clip_norm=None, blend_temp=1.0,
                                   curr_viol=None, bounds=None):
    """
    差分进化 (Differential Evolution) 引导 - 完全向量化实现
    
    在batch维度上将多个z样本视为一个种群，通过DE的变异和交叉操作
    生成更好的演化方向，替代或修正原始的v_pred。
    
    Args:
        z: 当前状态 (batch_size, dim)
        v_pred: flow matching预测的速度 (batch_size, dim)
        x_input: 条件输入 (batch_size, input_dim)
        objective_fn: 约束计算函数
        F: 缩放因子，控制变异强度 [0.3-1.0]
        CR: 交叉概率 [0.5-0.9]
        strategy: DE变异策略 ('best/1', 'rand/1', 'current-to-best/1')
        constraint_violations: 预计算的约束违反值 (可选，用于避免重复计算)
    
    Returns:
        v_pred_enhanced: DE增强后的速度向量 (batch_size, dim)
    """
    if objective_fn is None:
        return v_pred

    B, D = z.shape
    device = z.device
    half_dim = D // 2 
    
    # 评估当前种群的约束违反（如果未提供）
    if curr_viol is None:  
        with torch.no_grad():
            curr_viol = objective_fn(
                z[:, :half_dim], z[:, half_dim:], 
                x_input,
                reduction='none'
            )  # (batch_size,)
    
    # 找到最优个体
    best_idx = torch.argmin(curr_viol)
    z_best = z[best_idx:best_idx+1]  # (1, dim)
    
    # ---------- 生成 r0, r1, r2 索引（避免 i、自身且彼此不同） ----------
    # 用循环移位的打乱表法，避免 O(B^2 logB) 的 argsort
    base = torch.arange(B, device=device)
    perm = torch.stack([torch.randperm(B, device=device) for _ in range(3)], dim=1)  # (B,3)
    # 确保不包含自身：如果撞到了 i，则循环移位修正
    for k in range(3):
        hit = perm[:, k].eq(base)
        if hit.any():
            perm[hit, k] = (perm[hit, k] + 1) % B
    # 确保三者彼此不同
    # 若出现重复，做一次简单修复（概率极低，多次修复可加 while）
    same01 = perm[:,0].eq(perm[:,1])
    perm[same01,1] = (perm[same01,1] + 1) % B
    same02 = perm[:,0].eq(perm[:,2])
    perm[same02,2] = (perm[same02,2] + 1) % B
    same12 = perm[:,1].eq(perm[:,2])
    perm[same12,2] = (perm[same12,2] + 1) % B

    r0, r1, r2 = perm[:,0], perm[:,1], perm[:,2]
    zr0, zr1, zr2 = z[r0], z[r1], z[r2]

    # ---------- 变异 ----------
    if strategy == 'rand/1':
        z_mut = zr0 + F * (zr1 - zr2)
    elif strategy == 'current-to-best/1':
        z_mut = z + F * (z_best - z) + F * (zr1 - zr2) + F * v_guidance * 0.1    # 变异的时候同时考虑随机性和梯度信息
    else:  # 'best/1'
        z_mut = z_best + F * (zr1 - zr2)
    
    # ============================================================================
    # 向量化交叉操作
    # ============================================================================
    # ---------- 交叉（trial 个体） ----------
    cross_mask = (torch.rand(B, D, device=device) < CR)
    no_x = ~cross_mask.any(dim=1)
    if no_x.any():
        cross_mask[no_x, torch.randint(0, D, (no_x.sum(),), device=device)] = True
    u = torch.where(cross_mask, z_mut, z) 
    
    # ---------- 边界修复（可选） ----------
    if bounds is not None:
        if callable(bounds):
            lower, upper = bounds(u)
        else:
            lower, upper = bounds
        if lower is not None:
            u = torch.maximum(u, torch.as_tensor(lower, device=device, dtype=z.dtype))
        if upper is not None:
            u = torch.minimum(u, torch.as_tensor(upper, device=device, dtype=z.dtype))

     # ---------- trial 评价 + 择优 ----------
    with torch.no_grad(): 
        u_viol = objective_fn(u[:, :half_dim], u[:, half_dim:], x_input, reduction='none').to(z.dtype)
        u_viol = torch.nan_to_num(u_viol, nan=1e6, posinf=1e6, neginf=1e6)

        improve = u_viol < curr_viol  # 可行性/惩罚更小视为改进
    
    # ---------- 形成 DE 建议速度（仅在改进时生效） ----------
    v_de = u - z
    # 归一化并按 v_pred 尺度缩放
    v_pred_norm = torch.linalg.norm(v_pred, dim=1, keepdim=True)  # (B,1)
    v_de_norm   = torch.linalg.norm(v_de,   dim=1, keepdim=True).clamp_min(1e-8)
    v_de_scaled = v_de / v_de_norm * (v_pred_norm + 1e-8)

    # 范数裁剪（可选）
    if clip_norm is not None:
        v_de_scaled = v_de_scaled * (clip_norm / torch.maximum(
            clip_norm * torch.ones_like(v_de_norm), v_de_norm))

        # ---------- 自适应融合权 g（违反越大，g 越大） ----------
    with torch.no_grad():
        # 软归一化：越大越接近 1
        g = (curr_viol / (curr_viol + 1.0)).unsqueeze(1)  # (B,1)
        if blend_temp is not None and blend_temp != 1.0:
            # 温度调整（让分布更平/更尖）
            g = torch.sigmoid(torch.logit(g.clamp(1e-6, 1-1e-6)) / blend_temp)

    # 只对“有改进”的样本启用 DE 速度，否则为 0
    v_take = torch.where(improve.unsqueeze(1), v_de_scaled, torch.zeros_like(v_de_scaled))

    # ---------- 最终速度 ----------
    v_out = (1 - g) * v_pred + g * v_take 
    # v_out = v_take
    return v_out


def sep_cma_es_guidance(z, v_pred, x_input, objective_fn, 
                       constraint_violations=None, cma_state=None,
                       blend_ratio=0.7, update_state=True):
    """
    对角CMA-ES (Separable CMA-ES) 引导 - 带记忆功能
    
    使用对角协方差矩阵的轻量级CMA-ES，避免完整协方差矩阵的O(d²)存储和O(d³)计算。
    通过CMAESState保存演化路径和协方差信息，实现跨时间步的记忆。
    
    Args:
        z: 当前状态 (batch_size, dim)
        v_pred: flow matching预测的速度 (batch_size, dim)
        x_input: 条件输入 (batch_size, input_dim)
        objective_fn: 约束计算函数
        constraint_violations: 预计算的约束违反值 (可选)
        cma_state: CMAESState对象，保存历史信息 (可选)
        blend_ratio: CMA方向和flow方向的混合比例
        update_state: 是否更新CMA状态
    
    Returns:
        v_pred_enhanced: CMA-ES增强后的速度向量 (batch_size, dim)
    """
    if objective_fn is None:
        return v_pred
    
    batch_size = z.shape[0]
    dim = z.shape[1]
    
    if batch_size < 4:
        return v_pred
    
    # 评估当前种群
    output_dim = dim
    half_dim = output_dim // 2
    vm = z[:, :half_dim]
    va = z[:, half_dim:]
 
    if constraint_violations is None:
        with torch.no_grad():
            constraint_violations = objective_fn(
                vm, va,
                x_input[:, :-(1 + output_dim)],
                reduction='none'
            )  # (batch_size,)
    
    # 排序：找到表现最好的一半
    sorted_indices = torch.argsort(constraint_violations)
    mu = batch_size // 2  # 选择前50%
    elite_indices = sorted_indices[:mu]
    
    # 计算加权重心（更好的个体权重更高）
    weights = torch.log(torch.tensor(mu + 0.5, device=z.device)) - \
              torch.log(torch.arange(1, mu + 1, device=z.device, dtype=torch.float32))
    weights = weights / weights.sum()  # 归一化
    
    # 计算加权平均（分布中心）
    z_elite = z[elite_indices]  # (mu, dim)
    
    # 如果有CMA状态，使用状态中的均值作为old_mean
    if cma_state is not None and cma_state.mean is not None:
        old_mean = cma_state.mean
    else:
        # 第一次调用，使用当前种群平均
        old_mean = z.mean(dim=0)
    
    # 计算新均值
    mean_z = (weights.unsqueeze(1) * z_elite).sum(dim=0)  # (dim,)
    
    # 更新CMA-ES状态（如果提供了状态对象且需要更新）
    if cma_state is not None and update_state:
        cma_state.update(z_elite, weights, old_mean)
        # 使用状态中的采样标准差
        std = cma_state.get_sampling_std()
    else:
        # 无状态模式：估计对角协方差（每个维度的方差）
        centered = z_elite - mean_z.unsqueeze(0)  # (mu, dim)
        variance = (weights.unsqueeze(1) * (centered ** 2)).sum(dim=0)  # (dim,)
        variance = torch.clamp(variance, min=1e-8)  # 防止退化
        std = torch.sqrt(variance)  # (dim,)
    
    # 为整个batch生成改进方向（向量化操作）
    direction_to_mean = (mean_z.unsqueeze(0) - z)  # (batch, dim)
    
    # 使用CMA-ES学到的标准差生成自适应噪声
    adaptive_noise = torch.randn_like(z) * std.unsqueeze(0)  # (batch, dim)
    
    # CMA方向 = 向均值移动 + 自适应探索噪声
    if cma_state is not None:
        # 有状态时，使用状态中的sigma
        v_cma = direction_to_mean + adaptive_noise * cma_state.sigma
    else:
        # 无状态时，使用固定的探索强度
        v_cma = direction_to_mean * 0.2 + adaptive_noise * 0.1
    
    # 与原始flow方向混合
    v_pred_enhanced = blend_ratio * v_cma + (1.0 - blend_ratio) * v_pred
    
    return v_pred_enhanced


def adaptive_evolutionary_guidance(z, v_pred, t, step, x_input, objective_fn, v_guidance,
                                   evo_config=None, constraint_violations=None):
    """
    演化算法引导 - 默认使用DE，可选CMA-ES
    
    DE（差分进化）更稳健，适合有约束的优化问题：
    - 基于种群中的最优个体
    - 不会过度偏离可行域
    - 对约束违反的容忍度更好
    
    CMA-ES（协方差矩阵适应）可选，适合约束较松的情况：
    - 带记忆功能，学习问题结构
    - 更强的探索能力
    - 但可能在强约束下偏离
    
    Args:
        z: 当前状态 (batch_size, dim)
        v_pred: flow matching预测的速度 (batch_size, dim)
        t: 当前时间步 [0, 1]
        step: 时间步长
        x_input: 条件输入
        objective_fn: 约束计算函数
        evo_config: 演化算法配置字典
        constraint_violations: 预计算的约束违反值 (可选，用于避免重复计算)
            {
                'enabled': bool,              # 是否启用
                'method': str,                # 'DE' (default) 或 'CMA-ES'
                'start_time': float,          # 开始应用的时间步 (default 0.9)
                
                # DE参数
                'de_F': float,                # 变异强度 (default 0.5)
                'de_CR': float,               # 交叉概率 (default 0.7)
                'de_strategy': str,           # 变异策略 (default 'best/1')
                
                # CMA-ES参数（仅当method='CMA-ES'时使用）
                'blend_ratio': float,         # CMA方向混合比例 (default 0.7)
                'c_sigma': float,             # 步长学习率 (default 0.3)
                'c_c': float,                 # 协方差路径学习率 (default 0.4)
                'c_cov': float,               # 协方差矩阵学习率 (default 0.6)
                'damps': float,               # 步长阻尼 (default 1.0)
                'cma_state': CMAESState,      # 自动创建
                
                'verbose': bool               # 是否打印详细信息 (default False)
            }
    
    Returns:
        v_pred_enhanced: 演化算法增强后的速度向量
    """
    
    # 获取配置参数
    method = evo_config.get('method', 'DE')  # 默认使用DE 
    de_F = evo_config.get('de_F', 0.5)
    de_CR = evo_config.get('de_CR', 0.7)
    de_strategy = evo_config.get('de_strategy', 'best/1')
    verbose = evo_config.get('verbose', False) 
    
    batch_size = z.shape[0] 
    
    if batch_size < 4:
        if verbose:
            print(f"  [Evo] Batch size {batch_size} too small, skipping")
        return v_pred
    
    # 计算平均约束违反（用于verbose输出和CMA-ES）
    if verbose:
        avg_violation = constraint_violations.mean().item()
    
    # 根据配置选择方法
    if method.upper() == 'DE':
        # 使用差分进化（默认，更稳健）
        v_enhanced = differential_evolution_guidance(
            z, v_pred, v_guidance, x_input, objective_fn,
            F=de_F, CR=de_CR, strategy=de_strategy,
            curr_viol=constraint_violations  # 传递预计算的约束
        )
        
        if verbose:
            best_violation = constraint_violations.min().item()
            worst_violation = constraint_violations.max().item()
            print(f"  [DE t={t:.3f}] Strategy: {de_strategy}, F: {de_F:.2f}, CR: {de_CR:.2f}, "
                  f"Avg violation: {avg_violation:.6f}, "
                  f"Best: {best_violation:.6f}, Worst: {worst_violation:.6f}")
    
    else:
        # 未知方法，返回原始v_pred
        if verbose:
            print(f"  [Evo] Unknown method '{method}', using original v_pred")
        v_enhanced = v_pred
    
    return v_enhanced


# ============================================================================
# 演化算法增强模块结束
# ============================================================================


class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, x):
        gumbel_noise = self.sample_gumbel(x.size())
        y = x + gumbel_noise.to(x.device)
        soft_sample = F.softmax(y / self.temperature, dim=-1)

        if self.hard:
            hard_sample = torch.zeros_like(soft_sample).scatter(-1, soft_sample.argmax(dim=-1, keepdim=True), 1.0)
            sample = hard_sample - soft_sample.detach() + soft_sample
        else:
            sample = soft_sample

        return sample

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(x)


class Time_emb(nn.Module):
    def __init__(self, emb_dim, time_steps, max_period):
        super(Time_emb, self).__init__()
        self.emb_dim = emb_dim
        self.time_steps = time_steps
        self.max_period = max_period

    def forward(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        t = t.view(-1) * self.time_steps
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.emb_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0, act='relu'):
        super(MLP, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)
        net = []
        for _ in range(num_layer):
            net.extend([nn.Linear(hidden_dim, hidden_dim), act])
            # net.append(ResBlock(hidden_dim, hidden_dim//4))
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
        if output_activation:
            self.out_act = act_list[output_activation]
        else:
            self.out_act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x) #self.emb(torch.cat([x,z],dim=1))#
            if t is not None:
                emb = emb + self.temb(t)
        y = self.net(emb) 
        return self.out_act(y)

class SDP_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0, act='relu'):
        super(SDP_MLP, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)
        net = []
        for _ in range(num_layer):
            net.extend([SDPBasedLipschitzLinearLayer(hidden_dim, hidden_dim), act])
            # net.extend([nn.Linear(hidden_dim, hidden_dim), act])
            # net.append(ResBlock(hidden_dim, hidden_dim//4))
        net.append(SDPBasedLipschitzLinearLayer(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
        if output_activation:
            self.out_act = act_list[output_activation]
        else:
            self.out_act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x) #self.emb(torch.cat([x,z],dim=1))#
            if t is not None:
                emb = emb + self.temb(t)
        y = self.net(emb) 
        return self.out_act(y)
 

class CarbonTaxAwareMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer,
                 output_activation, latent_dim=0, act='relu',
                 carbon_tax_dim=1, anchor_dim=None):
        """
        input_dim: x 的总维度 = 负荷特征 + 碳税率 (tau) + 锚点 (anchor)
        carbon_tax_dim: 碳税率占用的维度，一般=1
        anchor_dim: 锚点占用的维度，一般等于 output_dim (如果为None则自动设为output_dim)
        latent_dim: z 的维度（或者说 emb(z) 的输入维）
        """

        super(CarbonTaxAwareMLP, self).__init__()

        # 激活函数表
        act_list = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'softplus': nn.Softplus(), 
            'sigmoid':  nn.Sigmoid(),
            'softmax': nn.Softmax(dim=-1),
            'gumbel': GumbelSoftmax(hard=True),
            'abs': Abs()
        }
        act_fn = act_list[act]

        # ------- 1. 基本设置 -------
        self.total_input_dim = input_dim
        self.carbon_tax_dim = carbon_tax_dim
        self.anchor_dim = anchor_dim if anchor_dim is not None else output_dim
        self.feature_dim = input_dim - carbon_tax_dim  #  - self.anchor_dim  # 除碳税率和锚点外的部分

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # ------- 2. 针对 tau(碳税率) 的专门门控网络 -------
        # tau_gate: tau -> [hidden_dim] 缩放向量 # 作用：控制"碳税敏感度"，系统性地影响不同策略分支
        self.tau_gate = nn.Sequential(nn.Linear(self.carbon_tax_dim, hidden_dim), nn.Sigmoid())

        # ------- 3. 针对 anchor(锚点) 的专门编码网络 -------
        # anchor_encoder: anchor -> [hidden_dim] 嵌入向量  # 作用：捕获锚点信息，提供"起始状态"的上下文
        self.anchor_encoder = nn.Sequential(nn.Linear(self.anchor_dim, hidden_dim), act_fn)

        # ------- 4. 原本的 W(x) 和 B(x) / emb(z) 结构 -------
        # 注意：这里用的是除碳税率和锚点外的 x_feat (负荷等特征)
        if latent_dim > 0:
            # W 和 B 只依赖 x_feat (负荷等特征)
            self.w = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim)) 
            self.emb_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), act_fn)
        else:
            # 没有 latent，则直接把 x_feat 过 emb_x
            self.emb_x = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim), act_fn)

        # ------- 5. 时间embedding 保留 -------
        self.temb = nn.Sequential(
            Time_emb(hidden_dim, time_steps=1000, max_period=1000)
        )

        # ------- 6. 主干网络 (和原来一致) -------
        net = []
        for _ in range(num_layer):
            net.extend([
                nn.Linear(hidden_dim, hidden_dim),
                act_fn
            ])
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)

        # 输出激活
        if output_activation:
            self.out_act = act_list[output_activation]
        else:
            self.out_act = nn.Identity()


    def forward(self, x, z=None, t=None):
        """
        x: [B, total_input_dim] = [features..., tau, anchor]
        z: [B, latent_dim] or None
        t: [B, 1] or [B]
        """

        # --- 拆分出负荷特征、碳税率tau、锚点anchor ---
        # 假设结构为: [负荷特征 | 碳税率 | 锚点]
        x_feat = x[..., :self.feature_dim]                           # [B, feature_dim]
        tau    = x[..., self.feature_dim:self.feature_dim + self.carbon_tax_dim]  # [B, carbon_tax_dim]
        anchor = x[..., self.feature_dim + self.carbon_tax_dim:]     # [B, anchor_dim]

        # --- 分别编码三个关键因素 ---
        # 1. tau 门控向量，形状 [B, hidden_dim], 值域(0,1)
        tau_gate = self.tau_gate(tau)
        # 2. anchor 编码向量，形状 [B, hidden_dim]
        anchor_emb = self.anchor_encoder(anchor)

        # --- 走两条路：有 z (FiLM 风格) / 无 z (直接 encode x_feat) ---
        if z is None or self.latent_dim == 0:
            # 没有 z：就像原始分支的 if z is None
            emb = self.emb_x(x_feat)               # [B, hidden_dim]
        else:
            # 有 z：FiLM/调制
            # emb_z(z) -> [B, hidden_dim]
            # w(x_feat), b(x_feat) -> [B, hidden_dim]
            w_x  = self.w(x_feat)                  # [B, hidden_dim]
            b_x  = self.b(x_feat)                  # [B, hidden_dim]
            zemb = self.emb_z(z)                   # [B, hidden_dim]

            # emb = w(x)*emb(z) + b(x)
            emb = w_x * zemb + b_x                 # [B, hidden_dim]

        # --- 注入时间信息 ---
        if t is not None:
            emb = emb + self.temb(t)               # [B, hidden_dim]

        # --- 关键改动：融合三个因素 ---
        # 1. 先加入锚点信息（加法融合）
        emb = emb + anchor_emb                     # [B, hidden_dim]
        
        # 2. 再用碳税率进行门控（乘法调制）
        # 解释：当碳税率升高时，网络会系统性地偏向"高碳惩罚/低排放解"方向
        # tau_gate 是 [0,1]，可以理解成在每个隐藏通道上开/关不同策略分支
        emb = emb * tau_gate                       # [B, hidden_dim]

        # --- 后续主干 MLP + 输出激活 ---
        y = self.net(emb)                          # [B, output_dim]
        return self.out_act(y)



class Lip_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0):
        super(Lip_MLP, self).__init__()
        if latent_dim > 0:
            w = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
            b = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
            t = [Time_emb(hidden_dim, time_steps=1000, max_period=1000)]
            self.w = nn.Sequential(*w)
            self.b = nn.Sequential(*b)
            self.t = nn.Sequential(*t)
            net = []
        else:
            latent_dim = input_dim

        emb = [LinearNormalized(latent_dim, hidden_dim), nn.ReLU()]
        self.emb = nn.Sequential(*emb)
        net = []
        for _ in range(num_layer):
            net.extend([LinearNormalized(hidden_dim, hidden_dim), nn.ReLU()])
        net.append(LinearNormalized(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif output_activation == 'gumbel':
            self.act = GumbelSoftmax(hard=True)
        elif output_activation == 'abs':
            self.act = Abs()
        else:
            self.act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x)
            if t is not None:
                emb = emb + self.t(t)
        y = self.net(emb)
        return self.act(y)

    def project_weights(self):
        self.net.project_weights()


class ATT(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation=None, latent_dim=0, agg=False,
                 pred_type='node', act='relu'):
        super(ATT, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)

        net = []
        # for _ in range(num_layer):
            # net.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            # net.extend([MHA(hidden_dim, 64, hidden_dim // 64),
            #             ResBlock(hidden_dim, hidden_dim//4)])
        # net.append(nn.Linear(hidden_dim, output_dim))
        # self.net = nn.Sequential(*net)
        self.net = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=max(hidden_dim // 64, 1))
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        # self.mha  = MHA(hidden_dim, 64, hidden_dim // 64)

        self.agg = agg
        self.pred_type = pred_type

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif output_activation == 'gumbel':
            self.act = GumbelSoftmax(hard=True)
        elif output_activation == 'abs':
            self.act = Abs()
        else:
            self.act = nn.Identity()

    def forward(self, x, z=None, t=None):
        ### x: B * N * F
        ### z: B * N
        ### t: B * 1
        batch_size = x.shape[0]
        node_size = x.shape[1]
        if z is None:
            # print(x.shape, self.emb)
            emb = self.emb(x)
        else:
            z = z.view(batch_size, -1, 1)
            emb = self.w(x) * self.emb(z) + self.b(x)
            if t is not None:
                emb = emb + self.temb(t).view(batch_size, 1, -1)
        emb = emb.permute(1, 0, 2)
        emb = self.trans(emb)
        emb = emb.permute(1, 0, 2)
        y = self.net(emb)  # B * N * 1
        
        if self.agg:
            y = y.mean(1)
        else:
            if self.pred_type == 'node':
                y = y.view(x.shape[0], -1)  # B * N
            else:
                y = torch.matmul(y.view(batch_size, node_size, 1), y.view(batch_size, 1, node_size))  # B * N * N
                col, row = torch.triu_indices(node_size, node_size, 1)
                y = y[:, col, row]
        return self.act(y)


class MHA(nn.Module):
    def __init__(self, n_in, n_emb, n_head):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.key = nn.Linear(n_in, n_in)
        self.query = nn.Linear(n_in, n_in)
        self.value = nn.Linear(n_in, n_in)
        self.proj = nn.Linear(n_in, n_in)

    def forward(self, x):
        # x: B * node * n_in
        batch = x.shape[0]
        node = x.shape[1]
        ### softmax
        #### key: B H node emb
        #### que: B H emb node
        key = self.key(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,1,2)
        query = self.query(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,2,1)
        value = self.key(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,1,2)
        score = torch.matmul(key, query)/(self.n_emb**0.5) # x: B * H * node * node
        prob = torch.softmax(score, dim=-1) # B * H * node * node (prob)
        out = torch.matmul(prob, value) # B * H * Node * 64
        out = out.permute(0,2,3,1).contiguous() # B * N * F * H
        out = out.view(batch, -1, self.n_emb*self.n_head)
        return x + self.proj(out)


class SimpleResBlock(nn.Module):
    """简单的残差块，使用固定的ReLU激活函数"""
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid), 
                                 nn.ReLU(),
                                 nn.Linear(n_hid, n_in))
    def forward(self, x):
        return x + self.net(x)


class SDPBasedLipschitzLinearLayer(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6):
        super(SDPBasedLipschitzLinearLayer, self).__init__()

        self.activation = nn.ReLU(inplace=False)
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q)
        q = q_abs[None, :]
        q_inv = (1 / (q_abs + self.epsilon))[:, None]
        T = 2 / (torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1) + self.epsilon)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out


class LinearNormalized(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)
        self.linear = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)


class PartialLinearNormalized(nn.Module):
    def __init__(self, input_dim, output_dim, con_dim):
        super(PartialLinearNormalized, self).__init__()
        self.con_dim = con_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear_1 = spectral_norm(nn.Linear(input_dim - con_dim, output_dim))

    def forward(self, x):
        with torch.no_grad():
            weight_copy = self.linear_1.weight.data.clone()
            self.linear.weight.data[:, self.con_dim:] = weight_copy
        return self.linear(x)


class distance_estimator(nn.Module):
    def __init__(self, n_feats, n_hid_params, hidden_layers, n_projs=2, beta=0.5):
        super().__init__()
        self.hidden_layers = hidden_layers  # number of hidden layers in the network
        self.n_projs = n_projs  # number of projections to use for weights onto Steiffel manifold
        self.beta = beta  # scalar in (0,1) for stabilizing feed forward operations

        # Intialize initial, middle, and final layers
        self.fc_one = torch.nn.Linear(n_feats, n_hid_params, bias=True)
        self.fc_mid = nn.ModuleList(
            [torch.nn.Linear(n_hid_params, n_hid_params, bias=True) for i in range(self.hidden_layers)])
        self.fc_fin = torch.nn.Linear(n_hid_params, 1, bias=True)

        # Normalize weights (helps ensure stability with learning rate)
        self.fc_one.weight = nn.Parameter(self.fc_one.weight / torch.norm(self.fc_one.weight))
        for i in range(self.hidden_layers):
            self.fc_mid.weight = nn.Parameter(self.fc_mid[i].weight / torch.norm(self.fc_mid[i].weight))
        self.fc_fin.weight = nn.Parameter(self.fc_fin.weight / torch.norm(self.fc_fin.weight))

    def forward(self, u):
        u = self.fc_one(u).sort(1)[0]  # Apply first layer affine mapping
        for i in range(self.hidden_layers):  # Loop for each hidden layer
            u = u + self.beta * (self.fc_mid[i](u).sort(1)[0] - u)  # Convex combo of u and sort(W*u+b)
        u = self.fc_fin(u)  # Final layer is scalar (no need to sort)
        J = torch.abs(u)
        return J

    def project_weights(self):
        self.fc_one.weight.data = self.proj_Stiefel(self.fc_one.weight.data, self.n_projs)
        for i in range(self.hidden_layers):
            self.fc_mid[i].weight.data = self.proj_Stiefel(self.fc_mid[i].weight.data, self.n_projs)
        self.fc_fin.weight.data = self.proj_Stiefel(self.fc_fin.weight.data, self.n_projs)

    def proj_Stiefel(self, Ak, proj_iters):  # Project to closest orthonormal matrix
        n = Ak.shape[1]
        I = torch.eye(n)
        for k in range(proj_iters):
            Qk = I - Ak.permute(1, 0).matmul(Ak)
            Ak = Ak.matmul(I + 0.5 * Qk)
        return Ak


# ============================================================================
# DeepOPF 风格的双网络 MLP (与 DeepOPV-V.ipynb 对齐)
# ============================================================================

class DeepOPF_NetVm(nn.Module):
    """
    DeepOPF 风格的电压幅值预测网络
    与 DeepOPV-V.ipynb 中的 NetVm 结构一致
    """
    def __init__(self, input_dim, output_dim, hidden_units, khidden):
        """
        Args:
            input_dim: 输入维度 (负荷特征)
            output_dim: 输出维度 (母线数量, 即 Vm 的维度)
            hidden_units: 基础隐藏单元数
            khidden: 各层的隐藏单元倍数数组, 如 [8, 6, 4, 2]
        """
        super(DeepOPF_NetVm, self).__init__()
        self.num_layer = len(khidden)
        
        # 动态创建隐藏层
        self.fc1 = nn.Linear(input_dim, khidden[0] * hidden_units)
        if self.num_layer >= 2:
            self.fc2 = nn.Linear(khidden[0] * hidden_units, khidden[1] * hidden_units)
        if self.num_layer >= 3:
            self.fc3 = nn.Linear(khidden[1] * hidden_units, khidden[2] * hidden_units)
        if self.num_layer >= 4:
            self.fc4 = nn.Linear(khidden[2] * hidden_units, khidden[3] * hidden_units)
        if self.num_layer >= 5:
            self.fc5 = nn.Linear(khidden[3] * hidden_units, khidden[4] * hidden_units)
        if self.num_layer >= 6:
            self.fc6 = nn.Linear(khidden[4] * hidden_units, khidden[5] * hidden_units)
        
        # 最后两层
        self.fcbfend = nn.Linear(khidden[-1] * hidden_units, output_dim)
        self.fcend = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.num_layer >= 2:
            x = F.relu(self.fc2(x))
        if self.num_layer >= 3:
            x = F.relu(self.fc3(x))
        if self.num_layer >= 4:
            x = F.relu(self.fc4(x))
        if self.num_layer >= 5:
            x = F.relu(self.fc5(x))
        if self.num_layer >= 6:
            x = F.relu(self.fc6(x))
        
        x = F.relu(self.fcbfend(x))
        x_pred = self.fcend(x)  # 无输出激活函数
        return x_pred


class DeepOPF_NetVa(nn.Module):
    """
    DeepOPF 风格的电压相角预测网络
    与 DeepOPV-V.ipynb 中的 NetVa 结构一致
    """
    def __init__(self, input_dim, output_dim, hidden_units, khidden):
        """
        Args:
            input_dim: 输入维度 (负荷特征)
            output_dim: 输出维度 (母线数量-1, 不包括 slack bus)
            hidden_units: 基础隐藏单元数
            khidden: 各层的隐藏单元倍数数组, 如 [8, 6, 4, 2]
        """
        super(DeepOPF_NetVa, self).__init__()
        self.num_layer = len(khidden)
        
        # 动态创建隐藏层
        self.fc1 = nn.Linear(input_dim, khidden[0] * hidden_units)
        if self.num_layer >= 2:
            self.fc2 = nn.Linear(khidden[0] * hidden_units, khidden[1] * hidden_units)
        if self.num_layer >= 3:
            self.fc3 = nn.Linear(khidden[1] * hidden_units, khidden[2] * hidden_units)
        if self.num_layer >= 4:
            self.fc4 = nn.Linear(khidden[2] * hidden_units, khidden[3] * hidden_units)
        if self.num_layer >= 5:
            self.fc5 = nn.Linear(khidden[3] * hidden_units, khidden[4] * hidden_units)
        if self.num_layer >= 6:
            self.fc6 = nn.Linear(khidden[4] * hidden_units, khidden[5] * hidden_units)
        
        # 最后两层
        self.fcbfend = nn.Linear(khidden[-1] * hidden_units, output_dim)
        self.fcend = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.num_layer >= 2:
            x = F.relu(self.fc2(x))
        if self.num_layer >= 3:
            x = F.relu(self.fc3(x))
        if self.num_layer >= 4:
            x = F.relu(self.fc4(x))
        if self.num_layer >= 5:
            x = F.relu(self.fc5(x))
        if self.num_layer >= 6:
            x = F.relu(self.fc6(x))
        
        x = F.relu(self.fcbfend(x))
        x_pred = self.fcend(x)  # 无输出激活函数
        return x_pred


class DeepOPF_MLP(nn.Module):
    """
    DeepOPF 风格的双网络 MLP 模型
    
    与 DeepOPV-V.ipynb 完全一致：
    - 使用两个独立网络分别预测 Vm 和 Va
    - 无输出激活函数
    - 使用 scale_vm 和 scale_va 进行输出缩放
    - Vm 使用 min-max 归一化到 [0,1] 再乘以 scale
    - Va 直接乘以 scale
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layer=4, 
                 scale_vm=10.0, scale_va=10.0, slack_bus_idx=0,
                 VmLb=None, VmUb=None):
        """
        Args:
            input_dim: 输入维度 (负荷特征)
            output_dim: 总输出维度 (Vm + Va)
            hidden_dim: 基础隐藏单元数 (对应 DeepOPF 的 hidden_units)
            num_layer: 隐藏层数量
            scale_vm: Vm 输出缩放因子 (默认10)
            scale_va: Va 输出缩放因子 (默认10)
            slack_bus_idx: slack bus 索引
            VmLb: Vm 下界 (用于反归一化)
            VmUb: Vm 上界 (用于反归一化)
        """
        super(DeepOPF_MLP, self).__init__()
        
        self.output_dim = output_dim
        self.n_bus = output_dim // 2
        self.scale_vm = scale_vm
        self.scale_va = scale_va
        self.slack_bus_idx = slack_bus_idx
        
        # 保存 Vm 的上下界用于反归一化
        # 如果没有提供，使用典型的 IEEE 系统值
        if VmLb is None:
            VmLb = torch.ones(self.n_bus) * 0.94
        if VmUb is None:
            VmUb = torch.ones(self.n_bus) * 1.06
        self.register_buffer('VmLb', VmLb if isinstance(VmLb, torch.Tensor) else torch.tensor(VmLb, dtype=torch.float32))
        self.register_buffer('VmUb', VmUb if isinstance(VmUb, torch.Tensor) else torch.tensor(VmUb, dtype=torch.float32))
        
        # 根据层数创建 khidden 数组 (参考 DeepOPF: [8, 6, 4, 2])
        if num_layer == 4:
            khidden = [8, 6, 4, 2]
        elif num_layer == 3:
            khidden = [8, 4, 2]
        elif num_layer == 5:
            khidden = [8, 6, 5, 4, 2]
        else:
            # 默认线性递减
            khidden = [max(8 - i, 2) for i in range(num_layer)]
        
        # Vm 预测网络
        self.net_vm = DeepOPF_NetVm(input_dim, self.n_bus, hidden_dim, khidden)
        
        # Va 预测网络 (不预测 slack bus, 所以维度是 n_bus - 1)
        self.net_va = DeepOPF_NetVa(input_dim, self.n_bus - 1, hidden_dim, khidden)
        
        # 损失函数
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, input_dim)
        
        Returns:
            y: 拼接的输出 [Vm_scaled, Va_scaled] (batch_size, output_dim)
               Vm_scaled: 范围 [0, scale_vm]
               Va_scaled: 无范围限制
        """
        # 分别预测 Vm 和 Va
        vm_pred = self.net_vm(x)  # (batch_size, n_bus), 范围 [0, scale_vm]
        va_pred = self.net_va(x)  # (batch_size, n_bus - 1), 范围无限制
        
        # 在 slack bus 位置插入 0 (slack bus 的相角为 0)
        batch_size = x.shape[0]
        va_full = torch.zeros(batch_size, self.n_bus, device=x.device)
        # 填充非 slack bus 的相角
        if self.slack_bus_idx == 0:
            va_full[:, 1:] = va_pred
        elif self.slack_bus_idx == self.n_bus - 1:
            va_full[:, :-1] = va_pred
        else:
            va_full[:, :self.slack_bus_idx] = va_pred[:, :self.slack_bus_idx]
            va_full[:, self.slack_bus_idx + 1:] = va_pred[:, self.slack_bus_idx:]
        
        # 拼接输出
        y = torch.cat([vm_pred, va_full], dim=1)
        return y
    
    def loss(self, y_pred, y_target):
        """计算 MSE 损失"""
        return self.criterion(y_pred, y_target)
    
    def denormalize_vm(self, vm_scaled):
        """
        将缩放后的 Vm 反归一化到真实 p.u. 值
        
        公式: Vm_pu = (vm_scaled / scale_vm) * (VmUb - VmLb) + VmLb
        """
        vm_normalized = vm_scaled / self.scale_vm  # [0, 1]
        vm_pu = vm_normalized * (self.VmUb - self.VmLb) + self.VmLb
        return vm_pu
    
    def denormalize_va(self, va_scaled):
        """
        将缩放后的 Va 反归一化到真实弧度值
        
        公式: Va_rad = va_scaled / scale_va
        """
        return va_scaled / self.scale_va
    
    def denormalize(self, y):
        """
        反归一化整个输出
        
        Args:
            y: 拼接的缩放输出 [Vm_scaled, Va_scaled]
        
        Returns:
            Vm_pu: 真实 p.u. 值
            Va_rad: 真实弧度值
        """
        vm_scaled = y[:, :self.n_bus]
        va_scaled = y[:, self.n_bus:]
        
        vm_pu = self.denormalize_vm(vm_scaled)
        va_rad = self.denormalize_va(va_scaled)
        
        return vm_pu, va_rad

