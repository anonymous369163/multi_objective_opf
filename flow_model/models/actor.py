import copy
from sympy.logic import true
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import scipy.io as scio
import scipy.sparse as scsp
import pypower.api as pp
from pypower.idx_bus import BUS_TYPE, PD, QD, VMAX, VMIN, VM, VA, MU_VMIN
from pypower.idx_gen import PMAX, PMIN, QMAX, QMIN, RAMP_10, VG, PG, QG, GEN_BUS, MU_QMIN, MU_PMAX, MU_PMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, PF, PT, QF, QT, RATE_A, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX
import pandapower as ppow
import pandapower.converter as pc


class PowerSystemConfig:
    """电力系统配置类，封装所有电力系统相关的参数和矩阵"""
    
    def __init__(self, case_file_path, device=None):
        """
        初始化电力系统配置
        
        Args:
            case_file_path: 电力系统案例文件路径
        """
        self.device = device
        # 加载案例数据
        case118 = scio.loadmat(case_file_path)
        ppc = pp.case118()
        ppc['bus'] = case118['bus']
        ppc['gen'] = case118['gen']
        ppc['branch'] = case118['branch']
        ppc['gencost'] = case118['gencost']
        
        # 基本参数
        self.nb = np.shape(ppc['bus'])[0]  # 母线数量
        self.nl = np.shape(ppc['branch'])[0]  # 支路数量
        self.ng = np.shape(ppc['gen'])[0]  # 发电机数量
        
        ppc = copy.deepcopy(ppc)
        
        # 发电机相关参数
        self.gen_bus = ppc['gen'][:, GEN_BUS].astype(int) - 1    # 每个机组连接的母线号
        self.balance_gen_bus = np.where(ppc['bus'][:, BUS_TYPE]==3)[0][0]
        self.balance_gen = np.where(self.gen_bus==self.balance_gen_bus)[0][0]  # 平衡机组所在位置
        # 生成去掉平衡机组后的 self.gen_bus_no_balance
        self.gen_bus_no_balance = np.delete(self.gen_bus, self.balance_gen)
        
        # 负荷相关参数
        Pdb = ppc['bus'][:, PD]
        Qdb = ppc['bus'][:, QD]
        self.idx_Pd = np.squeeze(np.where(np.abs(Pdb)>0), axis=0)
        self.idx_Qd = np.squeeze(np.where(np.abs(Qdb)>0), axis=0)
        self.npd = np.shape(self.idx_Pd)[0]
        self.nqd = np.shape(self.idx_Qd)[0]
        
        # 发电机功率限制
        Pg_max = np.zeros(self.nb)
        Pg_min = np.zeros(self.nb)
        Qg_max = np.zeros(self.nb)
        Qg_min = np.zeros(self.nb)
        
        # 选出所有最大有功出力PMAX大于0的发电机的索引（即有效发电机的索引）
        self.Pg_gen = np.squeeze(np.where(ppc['gen'][:, PMAX]>0), axis=0)
        self.Pg_bus = self.gen_bus[self.Pg_gen]  # 表示可调节gen对应的bus是哪些
        self.npg = np.shape(self.Pg_gen)[0]
        
        Pg_max[self.gen_bus] = (ppc['gen'][:, PMAX]) / 100
        Pg_min[self.gen_bus] = (ppc['gen'][:, PMIN]) / 100
        Qg_max[self.gen_bus] = (ppc['gen'][:, QMAX] - 1) / 100
        Qg_min[self.gen_bus] = (ppc['gen'][:, QMIN] + 1) / 100
        
        # 转换为torch张量
        self.Pg_max = torch.from_numpy(Pg_max).float().to(self.device)
        self.Pg_min = torch.from_numpy(Pg_min).float().to(self.device)
        self.Qg_max = torch.from_numpy(Qg_max).float().to(self.device)
        self.Qg_min = torch.from_numpy(Qg_min).float().to(self.device)
        
        # 爬坡限制
        ramp = ppc['gen'][self.Pg_gen, PMAX] / 100 * 0.05
        self.ramp = torch.from_numpy(ramp).float().to(self.device)
        
        # 支路容量限制 - 这部分将被删除
        # S_max = (ppc['branch'][:, RATE_A] - 1.5) / 100
        # self.S_max = torch.from_numpy(S_max).float().to(self.device)
        # self.S_max = torch.square(self.S_max)
        
        # 成本系数
        c2 = ppc['gencost'][:, 4].copy()
        c1 = ppc['gencost'][:, 5].copy()
        self.c2 = torch.from_numpy(c2).float().to(self.device)
        self.c1 = torch.from_numpy(c1).float().to(self.device)
        
        # 导纳矩阵计算
        ppc = pp.ext2int(ppc)
        Ybus, Yf, Yt = pp.makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])
        self.ppc = ppc

        self.Ybus = Ybus
        self.Yf = Yf
        self.Yt = Yt
        
        # ⚠️ 更新支路数量（ext2int 后可能减少）
        self.nl = ppc['branch'].shape[0]
        
        # ⚠️ 关键：定义 S_max，使用和 Yf/Yt 相同的 ppc（ext2int 后的）
        rate_a = ppc['branch'][:, RATE_A]
        S_max = np.where(
            (rate_a == 0),
            1e10,  # 无限制
            rate_a / 100  # 转换为 p.u.
        )
        self.S_max = torch.from_numpy(S_max).float().to(self.device)
        self.S_max = torch.square(self.S_max) 

        # 节点导纳矩阵
        G = Ybus.real.toarray()
        B = Ybus.imag.toarray()
        self.G = torch.from_numpy(G).float().to(self.device)
        self.B = torch.from_numpy(B).float().to(self.device)
        
        # 支路导纳矩阵
        Gf = Yf.real.toarray()
        Bf = Yf.imag.toarray()
        self.Gf = torch.from_numpy(Gf).float().to(self.device)
        self.Bf = torch.from_numpy(Bf).float().to(self.device)
        
        Gt = Yt.real.toarray()
        Bt = Yt.imag.toarray()
        self.Gt = torch.from_numpy(Gt).float().to(self.device)
        self.Bt = torch.from_numpy(Bt).float().to(self.device)
        
        # 支路连接矩阵
        f = ppc['branch'][:, F_BUS]
        t = ppc['branch'][:, T_BUS]
        Cf = scsp.coo_matrix((np.ones(self.nl), (np.arange(self.nl), f)), shape=(self.nl, self.nb)).toarray()
        Ct = scsp.coo_matrix((np.ones(self.nl), (np.arange(self.nl), t)), shape=(self.nl, self.nb)).toarray()
        self.Cf = torch.from_numpy(Cf).float().to(self.device)
        self.Ct = torch.from_numpy(Ct).float().to(self.device)
    
    def create_pandapower_net(self):
        """
        将pypower的ppc数据转换成pandapower的net对象
        
        Returns:
            net: pandapower网络对象
        """
        # 使用pandapower的converter将ppc转换为net
        net = pc.from_ppc(copy.deepcopy(self.ppc), f_hz=60, validate=True)
        ppow.runpp(net, calculate_voltage_angles=True, init="flat", numba=True, enforce_q_lims=False) 
        return net
    
    def extract_gen_data_from_pandapower(self, net, use_results=True):
        """
        从pandapower网络中提取完整的发电机数据（包括平衡发电机）
        返回的数据顺序与PyPower的Pg_gen一致
        
        Args:
            net: pandapower网络对象
            use_results: 是否使用潮流计算结果（res_gen/res_ext_grid），
                        False则使用设定值（gen.p_mw）
            
        Returns:
            gen_p: 发电机有功功率数组 (MW)
            gen_q: 发电机无功功率数组 (MVar) 
            gen_p_max: 发电机有功功率上限 (MW)
            gen_p_min: 发电机有功功率下限 (MW)
        """
        if not hasattr(self, 'pg_to_pandapower_map'):
            raise RuntimeError("请先调用 create_pandapower_net() 创建映射关系")
        
        n_gens = len(self.pg_to_pandapower_map)
        gen_p = np.zeros(n_gens)
        gen_q = np.zeros(n_gens)
        gen_p_max = np.zeros(n_gens)
        gen_p_min = np.zeros(n_gens)
        
        # 根据映射关系提取数据
        for i, mapping in enumerate(self.pg_to_pandapower_map):
            if mapping['source'] == 'ext_grid':
                # 从ext_grid提取
                idx = mapping['index']
                if use_results and hasattr(net, 'res_ext_grid') and len(net.res_ext_grid) > 0:
                    gen_p[i] = net.res_ext_grid.iloc[idx]['p_mw']
                    gen_q[i] = net.res_ext_grid.iloc[idx]['q_mvar']
                else:
                    # 使用设定值或0
                    gen_p[i] = net.ext_grid.iloc[idx].get('p_mw', 0.0) if 'p_mw' in net.ext_grid.columns else 0.0
                    gen_q[i] = 0.0
                
                # ext_grid的上下限从原始ppc中获取
                ppc_row = mapping['ppc_row']
                gen_p_max[i] = self.ppc['gen'][ppc_row, PMAX]
                gen_p_min[i] = self.ppc['gen'][ppc_row, PMIN]
                
            else:  # 'gen'
                # 从net.gen提取
                idx = mapping['index']
                if use_results and hasattr(net, 'res_gen') and len(net.res_gen) > 0:
                    gen_p[i] = net.res_gen.loc[idx, 'p_mw']
                    gen_q[i] = net.res_gen.loc[idx, 'q_mvar']
                else:
                    gen_p[i] = net.gen.loc[idx, 'p_mw']
                    gen_q[i] = 0.0
                
                gen_p_max[i] = net.gen.loc[idx, 'max_p_mw']
                gen_p_min[i] = net.gen.loc[idx, 'min_p_mw']
        
        return gen_p, gen_q, gen_p_max, gen_p_min

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
        
        # region 修正Q：扣除shunt的无功功率
        # 注意：shunt的无功功率与电压平方成正比 Q_shunt = Q_base * V^2
        # Actor的pf计算出的Q包含了流向shunt的功率，需要扣除
        # if self.has_shunt:
        #     vm_pu = Vm * 0.06 + 1  # 还原电压到p.u.单位 (batch_size, num_buses)
        #     for i in range(len(self.shunt_bus_idx)):
        #         bus_idx = self.shunt_bus_idx[i]
        #         # 计算该shunt在当前电压下的实际无功功率
        #         vm_bus = vm_pu[:, bus_idx]  # (batch_size,)
        #         q_shunt_actual = self.shunt_q_base[i] * (vm_bus ** 2)  # (batch_size,) p.u.
        #         # 从Q中扣除shunt消耗的无功（注意：P也需要扣除，但case118中P_shunt都为0）
        #         Q[:, bus_idx] = Q[:, bus_idx] - q_shunt_actual
        #         # 如果有有功分量，也扣除（虽然case118中都是0）
        #         if self.shunt_p_base[i] != 0:
        #             p_shunt_actual = self.shunt_p_base[i] * (vm_bus ** 2)
        #             P[:, bus_idx] = P[:, bus_idx] - p_shunt_actual
        # endregion
        
        # 从输入中提取负荷和上一时刻发电
        Pd = batch_inputs.T[: env.num_pd]
        Qd = batch_inputs.T[env.num_pd : env.num_pd + env.num_qd]
        # Pg_ = batch_inputs.T[env.num_pd + env.num_qd :]

        # 计算爬坡约束的上下限
        # Pg_up = Pg_.T + env.ramp
        # Pg_down = Pg_.T - env.ramp

        # 保存修正后的P和Q用于返回详细信息
        # P_original = P.clone()
        # Q_original = Q.clone()
        
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


class ActorFlow(Actor):
    """
    ActorFlow类：继承Actor，使用流模型进行预测
    
    主要改进:
    - forward方法使用flow model进行预测
    - 使用load_flow_model加载预训练流模型
    - 使用model_forward完成模型推理
    """
    def __init__(self, input_dim, env=None, output_dim=118, norm=False, 
                 args=None, model_type='rectified', device='cuda'):
        """
        初始化ActorFlow网络
        
        Args:
            input_dim: 输入维度
            env: PowerGridEnv环境实例
            output_dim: 输出维度（电压和相角）
            norm: 是否使用层归一化
            args: 流模型的参数字典
            model_type: 流模型类型 (rectified, gaussian, etc.)
            device: 计算设备
        """
        # 调用父类初始化
        super(ActorFlow, self).__init__(input_dim, env, output_dim, norm)
        
        # 保存流模型相关参数
        self.args = args
        self.model_type = model_type
        self.device = device 
        
        # 一次性导入所需的工具函数 
        self.build_forward_function()
        
        self.objective_fn = lambda Vm, Va, x_input, reduction: self.compute_constraint_loss(Vm, Va, x_input, self.args['env'], reduction=reduction)
        
        # 梯度引导配置（训练时使用）
        # self.guidance_config = {
        #     'enabled': True,
        #     'scale': 0.1,        # 目前效果最好的情况是设置为0.1
        #     'perp_scale': 0.01,
        #     'start_time': 0.8
        # }  
        self.guidance_config = self.args['guidance_config']
        print(f"  梯度引导配置: {self.guidance_config}")
        # DE方法引导
        self.evolutionary_config = self.args['evolutionary_config']
        self.single_target = self.args['single_target']
        # self.set_evolutionary_config('moderate')
        # self.evolutionary_config['enabled'] = False
        print(f" 演化算法参数配置: {self.evolutionary_config}")
    
    def set_evolutionary_config(self, preset='moderate', **kwargs):
        """
        动态设置演化算法配置（方便实验对比）
        
        Args:
            preset: 预设方案 ('minimal', 'conservative', 'moderate', 'disabled')
            **kwargs: 覆盖预设的具体参数
        
        示例：
            # 使用预设
            actor.set_evolutionary_config('minimal')
            actor.set_evolutionary_config('conservative')
            actor.set_evolutionary_config('disabled')
            
            # 自定义微调
            actor.set_evolutionary_config('minimal', start_time=0.99, de_CR=0.1)
        """
        presets = {
            'minimal': {
                'enabled': True,
                'method': 'DE',
                'start_time': 0.8,
                'de_F': 0.2,
                'de_CR': 0.2,
                'de_strategy': 'best/1',
                'verbose': False
            },
            'conservative': {
                'enabled': True,
                'method': 'DE',
                'start_time': 0.95,
                'de_F': 0.3,
                'de_CR': 0.3,
                'de_strategy': 'best/1',
                'verbose': False
            },
            'moderate': {
                'enabled': True,
                'method': 'DE',
                'start_time': 0.80,
                'de_F': 0.4,
                'de_CR': 0.4,
                'de_strategy': 'current-to-best/1',
                'verbose': False
            },
            'disabled': {
                'enabled': False,
                'verbose': False
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")
        
        # 从预设开始
        self.evolutionary_config = presets[preset].copy()
        
        # 应用自定义覆盖
        self.evolutionary_config.update(kwargs)
        
        print(f"[OK] DE config updated: {preset}")
    
    def build_forward_function(self):
        """
        一次性设置路径并导入所需函数（仅在初始化时调用）
        避免每次forward调用时都重复导入
        """
        import sys
        import os
        
        # 将flow_model目录添加到路径（如果尚未添加）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # 导入所需函数并保存为实例属性
        from utiles_v2 import model_forward
        self._model_forward_fn = model_forward
    
    def load_model(self, model_path):
        """加载预训练的流模型"""
        self.flow_model = torch.load(model_path, weights_only=False)
        self.flow_model.eval()
        # 根据model_path里add_carbon_tax后面是否有True来选择是否是True还是False
        import re
        m = re.search(r'add_carbon_tax_([^_/\\]+)', model_path)
        if m is not None and m.group(1) == 'True':
            self.add_carbon_tax = True
        else:
            self.add_carbon_tax = False
        print(f"[OK] Flow model loaded: {model_path}")
    
    def forward(self, state, apply_post_process=True):
        """
        使用流模型进行前向传播
        
        Args:
            state: 输入状态 (batch_size, input_dim)
            apply_post_process: 是否应用后处理修正（默认True）
            
        Returns:
            vm: 电压幅值 (batch_size, output_dim)
            va: 电压相角 (batch_size, output_dim)
        """ 
        
        # 准备输入数据（根据需要可以添加噪声或其他处理） 
        if self.add_carbon_tax:
            carbon_tax_tensor = torch.tensor([self.carbon_tax], dtype=torch.float32).to(self.device).unsqueeze(0)
            x_test = torch.cat([state, carbon_tax_tensor], dim=1) 
        else:
            x_test = state
        
        # 获取 env 对象用于后处理
        env = self.args.get('env', None)
        
        # 使用已导入的model_forward函数进行预测
        y_pred = self._model_forward_fn(
            model=self.flow_model,
            model_type=self.model_type,
            x_test=x_test, 
            args=self.args,
            objective_fn=self.objective_fn,
            guidance_config=self.guidance_config, 
            evolutionary_config=self.evolutionary_config,  # 传递演化算法配置
            sample_num=128,  # 可以设置为1获取单个预测
            device=self.device,
            apply_post_process=apply_post_process,  # 传递后处理参数
            env=env,  # 传递环境对象
            max_iterations=5  # 后处理最大迭代次数
        )
        
        # 分割输出为vm和va
        # 假设y_pred的前半部分是vm，后半部分是va
        output_dim_half = self.output_dim
        vm = y_pred[:, :output_dim_half]
        va = y_pred[:, output_dim_half:]
        
        return vm, va


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)


    # forward
    def forward(self, state, action):
        q = torch.relu(self.fc1(torch.cat([state, action], 1)))
        q = torch.relu(self.fc2(q))
        q = self.fc3(q)
        
        return q





