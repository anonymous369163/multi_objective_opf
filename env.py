"""
构建一个简单的电力系统，并使用pandapower进行潮流计算

# 3-13 目前的环境的负荷是生成的，且不需要预测，然后环境更新部分，目前看起来都没啥问题，计算的约束违反和奖励函数，目前看起来也没啥问题
# 但是现在没加入这个相邻时间功率之间的约束，需要加入
# 3-16 目前环境里的负荷曲线是固定的，然后网络拓扑也是不变的，需要修改。
"""


# import gym
# from gym import spaces


import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import torch
from pypower.idx_brch import RATE_A

from add_objective.carbon_casefile import carbon_casefile, fuel_dict_generation

# GCI Lookup Tables - 在 env.py 中定义以避免依赖 carbon_casefile.py 的内部实现
# 数据来源: PGLib-CO2, US EPA, IPCC
# CO2 emissions only (tCO2/MWh)
FUEL_LOOKUP_CO2 = {
    "ANT": 0.9095,  # Anthracite Coal
    "BIT": 0.8204,  # Bituminous Coal
    "Oil": 0.7001,  # Heavy Oil
    "GAS": 0.5173,  # Natural Gas
    "CCGT": 0.3621,  # Gas Combined Cycle
    "ICE": 0.6030,  # Internal Combustion Engine
    "Thermal": 0.6874,  # Thermal Power (General)
    "NUC": 0.0,  # Nuclear Power
    "RE": 0.0,  # Renewable Energy
    "HYD": 0.0,  # Hydropower
    "N/A": 0.0  # Default case
}

# CO2 equivalent: CO2 + (CH4 × 21) + (N2O × 310) (tCO2e/MWh)
FUEL_LOOKUP_CO2E = {
    "ANT": 0.9143,  # Anthracite Coal
    "BIT": 0.8230,  # Bituminous Coal
    "Oil": 0.7018,  # Heavy Oil
    "GAS": 0.5177,  # Natural Gas
    "CCGT": 0.3625,  # Gas Combined Cycle
    "ICE": 0.6049,  # Internal Combustion Engine
    "Thermal": 0.6894,  # Thermal Power (General)
    "NUC": 0.0,  # Nuclear Power
    "RE": 0.0,  # Renewable Energy
    "HYD": 0.0,  # Hydropower
    "N/A": 0.0  # Default case
}

def get_gci_value(fuel_type, emissions_type="CO2"):
    """
    获取指定燃料类型的 GCI 值
    
    Args:
        fuel_type: 燃料类型 (如 "BIT", "GAS", "CCGT", "ANT")
        emissions_type: 排放类型 ("CO2" 或 "CO2e")
    
    Returns:
        float: GCI 值 (tCO2/MWh 或 tCO2e/MWh)
    """
    lookup_table = FUEL_LOOKUP_CO2E if emissions_type == "CO2e" else FUEL_LOOKUP_CO2
    return lookup_table.get(fuel_type, 0.0)

class BranchCurrentLayer(torch.nn.Module):
    """
    输入:
      vm_pu: (B, nb)  电压幅值（pu）
      va_deg: (B, nb) 电压相角（度）
    输出:
      Ibus: (B, nb)   母线注入电流（pu，复数）
      If:   (B, nl)   每条支路 from 端电流（pu，复数）
      It:   (B, nl)   每条支路 to   端电流（pu，复数）
      Vf:   (B, nl)   from 端电压（pu，复数）
      Vt:   (B, nl)   to   端电压（pu，复数）
    备注:
      - Y 矩阵在图中是常数（不求导），梯度会一直回传到 vm_pu 和 va_deg
      - 采用复数计算，torch 会自动处理 conj / abs 等的复导数（Wirtinger）
    """
    def __init__(self, Ybus_np, Yf_np, Yt_np, fb_np=None, tb_np=None, device="cpu", use_double=False):
        super().__init__()
        dtype_r = torch.float64 if use_double else torch.float32
        dtype_c = torch.complex128 if use_double else torch.complex64

        # 常数矩阵注册为 buffer（不参与优化，但随 model.to(device) 迁移）
        Gbus = torch.from_numpy(np.real(Ybus_np)).to(device=device, dtype=dtype_r)
        Bbus = torch.from_numpy(np.imag(Ybus_np)).to(device=device, dtype=dtype_r)
        Gf   = torch.from_numpy(np.real(Yf_np)).to(device=device, dtype=dtype_r)
        Bf   = torch.from_numpy(np.imag(Yf_np)).to(device=device, dtype=dtype_r)
        Gt   = torch.from_numpy(np.real(Yt_np)).to(device=device, dtype=dtype_r)
        Bt   = torch.from_numpy(np.imag(Yt_np)).to(device=device, dtype=dtype_r)

        self.register_buffer("Ybus", Gbus + 1j * Bbus)  # (nb, nb) 复数
        self.register_buffer("Yf",   Gf   + 1j * Bf)     # (nl, nb)
        self.register_buffer("Yt",   Gt   + 1j * Bt)     # (nl, nb)

        if fb_np is not None and tb_np is not None:
            self.register_buffer("fb", torch.from_numpy(fb_np).to(device=device))
            self.register_buffer("tb", torch.from_numpy(tb_np).to(device=device))

        self.dtype_c = dtype_c

    def forward(self, vm_pu, va_deg):
        """
        vm_pu, va_deg: (B, nb), 实数 tensor
        返回复数 tensor: Ibus, If, It, Vf, Vt
        """
        # 复电压 V = Vm * exp(j*Va)
        va_rad = va_deg * (torch.pi / 180.0)
        V = vm_pu.to(dtype=self.Ybus.real.dtype, device=self.Ybus.device)
        V = V * torch.exp(1j * va_rad.to(V.dtype))  # (B, nb), complex

        # 母线注入电流 Ibus = Ybus @ V^T → 再转回 (B, nb)
        # 更高效的形状计算： (B, nb) x (nb, nb) -> (B, nb)，即 V @ Ybus^T
        Ibus = torch.matmul(V, self.Ybus.T)  # (B, nb)

        # 分支电流 If = Yf @ V^T → (nl, B) → 转置; 等价于 V @ Yf^T
        If = torch.matmul(V, self.Yf.T)  # (B, nl)
        It = torch.matmul(V, self.Yt.T)  # (B, nl)

        # 取分支两端电压（用于计算支路功率/越限等）；索引用 gather
        # fb/tb 形状 (nl,)
        if self.fb is not None and self.tb is not None:
            Vf = V.index_select(dim=1, index=self.fb)  # (B, nl), from 端电压
            Vt = V.index_select(dim=1, index=self.tb)  # (B, nl), to   端电压
        else:
            Vf = None
            Vt = None

        return Ibus, If, It, Vf, Vt

import copy
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, PF, PT, QF, QT, RATE_A, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX
import scipy.sparse as scsp

class PowerGridEnv(gym.Env):
    """Custom Power Grid Environment that follows the OpenAI Gym interface"""
    
    def __init__(self, 
                num_timesteps=24*12*300, 
                case_name="case9", 
                random_load=False, 
                run_pp=True, 
                consider_renewable_generation=False, 
                device="cpu", 
                PowerSystemConfig=None,
                carbon_tax=0.0,
                ext_grid_is_external_market=False,
                external_load_scenarios=None,
                external_carbon_tax_scenarios=None):
        """
        初始化电力网络环境
        
        Args:
            num_timesteps: 时间步数
            case_name: 案例名称 ("case9" or "case118")
            random_load: 是否使用随机负荷
            run_pp: 是否运行潮流计算
            consider_renewable_generation: 是否考虑可再生能源
            device: 计算设备 ("cpu" or "cuda")
            PowerSystemConfig: 电力系统配置对象
            carbon_tax: 碳税率 ($/tCO2)，默认为0表示不考虑碳成本
            ext_grid_is_external_market: ext_grid是否代表外部市场购电
                - False (默认): ext_grid是系统内部平衡发电机，需要计算碳排放
                - True: ext_grid是外部市场购电，碳成本已包含在电价中，不重复计算
            external_load_scenarios: 外部提供的负荷场景序列，shape: (num_scenarios, num_load_buses)
                - 如果提供，则使用这些场景而不是生成负荷曲线
                - 每次reset()会按顺序或随机选择一个场景
            external_carbon_tax_scenarios: 外部提供的碳税场景序列，shape: (num_scenarios,)
                - 与external_load_scenarios配对使用
                - 每次reset()会使用对应的碳税值
        """
        super(PowerGridEnv, self).__init__()
        
        # Create a pandapower network
        self.case_name = case_name
        self.PowerSystemConfig = PowerSystemConfig
        
        # 碳税率设置 ($/tCO2)
        self.carbon_tax = carbon_tax  # 可以设置为 10, 20, 30 等不同的碳税率 
        
        # ext_grid 类型设置
        self.ext_grid_is_external_market = ext_grid_is_external_market
        
        # 外部场景设置
        self.external_load_scenarios = external_load_scenarios
        self.external_carbon_tax_scenarios = external_carbon_tax_scenarios
        self.use_external_scenarios = (external_load_scenarios is not None)
        self.current_scenario_idx = 0  # 当前使用的场景索引
        if self.PowerSystemConfig is not None:
            # 从PowerSystemConfig创建pandapower网络
            self.net = self.PowerSystemConfig.create_pandapower_net()
        else:
            self.net = self._create_network()   # init network

        self.num_timesteps = num_timesteps
        self.random_load = random_load
        self.device = device

        # 如果不是pretrain，则考虑可再生能源的出力
        self.consider_renewable_generation = consider_renewable_generation

        self.run_pp = run_pp

        pp.runpp(self.net) 
        
        # Define action and observation spaces  
        self.num_buses = len(self.net.bus)
        self.num_gen = len(self.net.gen)
        self.num_load = len(self.net.load)
        self.num_pg = np.sum(self.net.gen['max_p_mw']>0)
        self.num_pd = np.sum(self.net.load['p_mw']>0)
        self.num_qd = np.sum(self.net.load['q_mvar']>0)
 
        self.voltage_low = self.net.bus.min_vm_pu.values  # 电压下限
        self.voltage_high = self.net.bus.max_vm_pu.values  # 电压上限  验证发现和ppc里得到的VMIN和VMAX是一致的
        
        # 相角范围
        self.p_gen_low = self.net.gen.min_p_mw.values  # 验证发现p_gen_high 和 p_gen_low 与ppc里提取得到的Pg_max 和 Pg_min一致，只是缺少了slack对应的max 和 min的值。
        self.p_gen_high = self.net.gen.max_p_mw.values 

        # 先找到可调发电机的位置 
        self.Pg_idx = self.net.gen['max_p_mw']>0
        self.Pg_bus_idx = self.net.gen['bus'][self.Pg_idx].values  # 表示可调节发电机对应的bus索引

        # 负荷索引
        self.pd_idx = self.net.load['p_mw']>0
        self.qd_idx = self.net.load['q_mvar']>0
        self.pd_bus_idx = self.net.load['bus'][self.pd_idx].values
        self.qd_bus_idx = self.net.load['bus'][self.qd_idx].values  # 表示无功负荷对应的bus索引 且验证发现和ppc里得到的idx_pd是一致的
        
        # 生成负荷功率曲线
        if self.use_external_scenarios:
            # 使用外部提供的场景，为每个场景创建一个时间步的负荷曲线
            print(f"[INFO] 使用外部提供的 {len(self.external_load_scenarios)} 个负荷场景")
            
            # 场景数量可能与num_timesteps不同，记录实际场景数
            self.num_scenarios = np.min([len(self.external_load_scenarios), self.num_timesteps])
            
            # 初始化负荷曲线（每个场景作为一个时间步）
            self.load_profiles_p = np.zeros((self.num_load, self.num_scenarios), dtype=np.float32)
            self.load_profiles_q = np.zeros((self.num_load, self.num_scenarios), dtype=np.float32)
            
            # 将external_load_scenarios的数据填充到load_profiles中
            # external_load_scenarios的格式: (num_scenarios, num_pd + num_qd)
            # 前num_pd列是有功负荷（只包含p>0的节点），后num_qd列是无功负荷（只包含q>0的节点）
            for i in range(self.num_scenarios):
                current_scenario = self.external_load_scenarios[i]
                
                # 检查场景数据维度
                if len(current_scenario) == self.num_pd + self.num_qd:
                    # 场景数据包含P和Q
                    # 将有功负荷数据赋值给p>0的负荷节点
                    self.load_profiles_p[self.pd_idx, i] = current_scenario[:self.num_pd]
                    # 将无功负荷数据赋值给q>0的负荷节点
                    self.load_profiles_q[self.qd_idx, i] = current_scenario[self.num_pd:]
                else:
                    raise ValueError(f"外部场景数据维度错误: 期望{self.num_pd}或{self.num_pd + self.num_qd}，实际{len(current_scenario)}")
            
            print(f"  - 成功加载 {self.num_scenarios} 个场景到负荷曲线")
            print(f"  - load_profiles_p shape: {self.load_profiles_p.shape}")
            print(f"  - load_profiles_q shape: {self.load_profiles_q.shape}")
        else:
            # 生成随机或固定的负荷功率曲线   
            if self.random_load:
                self._generate_random_load_profiles() 
            else:
            # 随机生成负荷功率曲线
                self._generate_load_profiles()
            # self._generate_random_load_profiles() #  生成+-10%的负荷数据
            # 生成风电出力曲线（仅当case为case118时）
            if self.case_name == "case118" and self.consider_renewable_generation:
                self._generate_wind_profiles()

        self.gen_bus_idx = self.net.gen.bus.to_numpy(dtype=int)  # 表示所有发电机对应的bus索引

        # Example: observe bus voltages, line loadings, and generator outputs 
        obs_dim = self.num_pd + self.num_qd  #  + self.num_pg

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                          shape=(obs_dim,), dtype=np.float32)

        # 发电机电压和可调发电机有功功率的集合
        action_low = np.concatenate([self.voltage_low[self.gen_bus_idx], self.p_gen_low[self.Pg_idx]])
        action_high = np.concatenate([self.voltage_high[self.gen_bus_idx], self.p_gen_high[self.Pg_idx]])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self.Pg_max = self.net.gen['max_p_mw'].values / 100  # p.u.
        self.Pg_min = self.net.gen['min_p_mw'].values / 100  # p.u.
        self.Qg_max = (self.net.gen['max_q_mvar'].values) / 100  # p.u.
        self.Qg_min = (self.net.gen['min_q_mvar'].values) / 100  # p.u.
        self.Pg_max = torch.from_numpy(self.Pg_max).float().to(self.device)
        self.Pg_min = torch.from_numpy(self.Pg_min).float().to(self.device)
        self.Qg_max = torch.from_numpy(self.Qg_max).float().to(self.device)
        self.Qg_min = torch.from_numpy(self.Qg_min).float().to(self.device)
        
        # S_max需要平方，因为pf函数返回的Sf和St是平方值
        rate_a = self.net._ppc['branch'][:, RATE_A]
        # 检查 rate_a 是否为 0 (无限制)，为 0 则设为极大值，否则除以 100 转换为 p.u.
        S_max_linear = np.where(rate_a == 0, 1e10, rate_a / 100.0)  # p.u.
        self.S_max = torch.from_numpy(S_max_linear).float().to(self.device)  # p.u.

        # 将每个发电机的功率爬坡速率限制在其最大出力的1/20 
        self.ramp = (self.net.gen['max_p_mw'][self.Pg_idx] / 20) / 100  # p.u.
        self.ramp = torch.from_numpy(self.ramp.values).float().to(self.device)
        
        # 提取并联补偿器(shunt)信息
        # 注意：shunt的功率与电压平方成正比 Q_shunt = Q_base * V^2
        self.has_shunt = len(self.net.shunt) > 0
        if self.has_shunt:
            self.shunt_bus_idx = self.net.shunt.bus.values  # shunt所在母线
            self.shunt_q_mvar_base = self.net.shunt.q_mvar.values  # 基准无功功率(MW)，1.0 p.u.电压时
            self.shunt_p_mw_base = self.net.shunt.p_mw.values  # 基准有功功率(MW)
            print(f"[INFO] 系统中有 {len(self.shunt_bus_idx)} 个并联补偿器(shunt)")
        else:
            self.shunt_bus_idx = np.array([])
            self.shunt_q_mvar_base = np.array([])
            self.shunt_p_mw_base = np.array([])

        # 计算导纳矩阵
        Ybus, Yf, Yt = makeYbus(self.net["_ppc"]["baseMVA"], self.net["_ppc"]["bus"], self.net["_ppc"]["branch"])  # 验证发现和ppc里得到的Ybus是一致的

        self.Ybus = Ybus.toarray()  # 转换为numpy数组
        self.Yf = Yf.toarray()
        self.Yt = Yt.toarray()  
        internal = self.net._pd2ppc_lookups      # 外部索引到 ppc 行的映射
        line_indices = internal["branch"]["line"]
        line_start, line_end = line_indices
        self.line_rows = torch.arange(line_start, line_end, device=self.device) 

        # 保留导纳矩阵的一些量：
        self.G = torch.from_numpy(Ybus.real.toarray()).float().to(self.device)
        self.B = torch.from_numpy(Ybus.imag.toarray()).float().to(self.device)
        self.Gf = torch.from_numpy(Yf.real.toarray()).float().to(self.device)
        self.Bf = torch.from_numpy(Yf.imag.toarray()).float().to(self.device)
        self.Gt = torch.from_numpy(Yt.real.toarray()).float().to(self.device)
        self.Bt = torch.from_numpy(Yt.imag.toarray()).float().to(self.device)
        f = self.net._ppc['branch'][:, F_BUS]
        t = self.net._ppc['branch'][:, T_BUS]

        # 后面 这几行是为了便于后续计算首端和尾端的有功功率什么的
        nb = np.shape(self.net._ppc['bus'])[0]
        nl = np.shape(self.net._ppc['branch'])[0]
        Cf_np = scsp.coo_matrix((np.ones(nl), (np.arange(nl), f)), shape=(nl, nb)).toarray()
        Ct_np = scsp.coo_matrix((np.ones(nl), (np.arange(nl), t)), shape=(nl, nb)).toarray()
        self.Cf = torch.from_numpy(Cf_np).float().to(self.device)
        self.Ct = torch.from_numpy(Ct_np).float().to(self.device)

        # 加入二氧化碳排放量的考虑
        # Generate the fuel dictionary
        fuel_dict = fuel_dict_generation(self.net)

        # Configure fuel types based on generator capacity for IEEE 118 system
        # This configuration follows realistic power system practices:
        # - Large generators (>300 MW): Coal-fired baseload plants
        # - Medium generators (100-300 MW): Gas-fired load-following plants
        # - Small generators (<100 MW): Gas/Oil-fired peaking plants
        # - Zero capacity generators: Synchronous condensers or reserves (N/A)
        
        if self.case_name == "case118":
            # 自动配置燃料类型（基于实际发电容量）
            # 修复：只为 max_p_mw > 0 的发电机配置燃料类型
            print("\n[INFO] 自动配置 IEEE 118 发电机燃料类型...")
            
            # 只处理有发电能力的机组
            active_gens = self.net.gen[self.net.gen['max_p_mw'] > 0].sort_values('max_p_mw', ascending=False)
            
            # 统计各类型机组数量
            large_count = medium_count = small_count = 0
            large_buses = []
            medium_buses = []
            small_buses = []
            
            for idx, gen in active_gens.iterrows():
                bus = gen['bus']
                capacity = gen['max_p_mw']
                
                if capacity > 300:
                    # 大型基荷煤电机组 (>300 MW)
                    # 交替使用烟煤(BIT)和无烟煤(ANT)以增加多样性
                    fuel_type = "BIT" if large_count % 2 == 0 else "ANT"
                    fuel_dict[bus] = {"type": fuel_type, "emissions": "CO2"}
                    large_count += 1
                    large_buses.append(bus)
                    
                elif capacity > 100:
                    # 中型联合循环机组 (100-300 MW)
                    fuel_dict[bus] = {"type": "CCGT", "emissions": "CO2e"}
                    medium_count += 1
                    medium_buses.append(bus)
                    
                else:
                    # 小型燃气/燃油调峰机组 (<100 MW)
                    # 前几台使用燃气，最后一台使用燃油
                    if small_count < len(active_gens[active_gens['max_p_mw'] <= 100]) - 1:
                        fuel_type = "GAS"
                    else:
                        fuel_type = "Oil"
                    fuel_dict[bus] = {"type": fuel_type, "emissions": "CO2e"}
                    small_count += 1
                    small_buses.append(bus)
            
            print(f"  大型煤电机组 (>300 MW): {large_count} 台, 节点: {large_buses}")
            print(f"  中型CCGT机组 (100-300 MW): {medium_count} 台, 节点: {medium_buses}")
            print(f"  小型燃气/油机组 (<100 MW): {small_count} 台, 节点: {small_buses}")
            print(f"  总计: {large_count + medium_count + small_count} 台有效发电机")
            
        elif self.case_name == "case9":
            # Simple configuration for case9
            fuel_dict[1] = {"type": "GAS", "emissions": "CO2e"}
            fuel_dict[2] = {"type": "ANT", "emissions": "CO2"}
            fuel_dict[3] = {"type": "BIT", "emissions": "CO2"}

        # Create the carbon casefile
        carbon_casefile(self.net, fuel_dict)
        
        # 诊断：检查燃料配置是否正确
        if self.case_name == "case118":
            print("\n" + "="*80)
            print("【诊断】IEEE 118 发电机燃料配置检查")
            print("="*80)
            
            # 显示所有有GCI的发电机
            gens_with_gci = self.net.gen[self.net.gen['GCI'] > 0]
            print(f"\n配置了GCI的发电机数量: {len(gens_with_gci)}")
            print("\n节点  最大容量(MW)  GCI      状态")
            print("-" * 50)
            for idx, gen in gens_with_gci.iterrows():
                bus = gen['bus']
                max_p = gen['max_p_mw']
                gci = gen['GCI']
                status = "[OK]" if max_p > 0 else "[WARNING: max_p_mw=0]"
                print(f"{bus:3d}   {max_p:7.1f}       {gci:.4f}   {status}")
            
            # 检查是否有 max_p_mw=0 但配置了GCI的发电机
            invalid_config = gens_with_gci[gens_with_gci['max_p_mw'] == 0]
            if len(invalid_config) > 0:
                print(f"\n[WARNING] 发现 {len(invalid_config)} 台 max_p_mw=0 的发电机被配置了GCI！")
                print("这些发电机实际上无法发电，不应该配置燃料类型。")
                print(f"问题节点: {sorted(invalid_config['bus'].values.tolist())}")
            
            # 显示有发电能力但未配置GCI的发电机
            active_gens = self.net.gen[self.net.gen['max_p_mw'] > 0]
            unconfigured = active_gens[active_gens['GCI'] == 0]
            if len(unconfigured) > 0:
                print(f"\n[WARNING] 发现 {len(unconfigured)} 台有发电能力但未配置GCI的发电机！")
                print(f"未配置节点: {sorted(unconfigured['bus'].values.tolist())}")
            
            print("\n" + "="*80)
        
        # 为外接电源（ext_grid）设置 GCI 值
        # 注意：ext_grid 的性质取决于建模假设
        if len(self.net.ext_grid) > 0:
            if 'GCI' not in self.net.ext_grid.columns:
                self.net.ext_grid['GCI'] = 0.0
            
            # 根据 ext_grid 的性质设置 GCI
            if self.ext_grid_is_external_market:
                # 情况1：ext_grid 代表外部市场购电
                # 碳成本已包含在市场电价中，不需要单独计算碳排放
                # 设置 GCI = 0
                self.net.ext_grid['GCI'] = 0.0
                print(f"[INFO] ext_grid 设置为外部市场购电，GCI = 0 (碳成本已包含在电价中)")
            else:
                # 情况2：ext_grid 代表系统内部的平衡发电机
                # 需要像其他发电机一样计算碳排放
                
                # 方法1：使用系统平均 GCI
                if len(self.net.gen) > 0 and 'GCI' in self.net.gen.columns:
                    avg_gci = self.net.gen['GCI'].mean()
                    self.net.ext_grid['GCI'] = avg_gci
                
                # 方法2：或者根据 ext_grid 所在节点设置特定的燃料类型
                # 对于 case118，平衡机在节点 69（大型煤电机组）
                if self.case_name == "case118":
                    for i, ext_bus in enumerate(self.net.ext_grid['bus']):
                        if ext_bus in fuel_dict:
                            # 使用该节点对应的燃料类型
                            fuel_type = fuel_dict[ext_bus]["type"]
                            emissions = fuel_dict[ext_bus]["emissions"]
                            
                            # 使用导入的函数获取 GCI 值（避免重复定义字典）
                            gci_value = get_gci_value(fuel_type, emissions)
                            
                            self.net.ext_grid.loc[i, 'GCI'] = gci_value
                            print(f"[INFO] ext_grid 节点 {ext_bus} 设置为内部平衡发电机，燃料类型: {fuel_type}, GCI = {gci_value:.4f}")

        # example to calculate the carbon emissions
        # Calculate carbon emissions for each generator
        # self.net.res_gen["carbon emission"]=self.net.res_gen["p_mw"]*self.net.gen["GCI"]
        

    def draw_load_profiles(self):
        """绘制总负荷曲线和负荷比率曲线"""  
        
        # 设置matplotlib参数，避免网络连接问题
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 清除之前的图形
        plt.close('all') 
        
        num_loads = len(self.net.load)
        num_timesteps = len(self.load_profiles_p[0])
        
        # 创建时间轴标签(小时)
        hours = np.linspace(0, 24, num_timesteps)
        
        # 计算总有功功率
        total_active_power = np.sum(self.load_profiles_p, axis=0)
        # 计算比率（相对于平均值）
        ratio = total_active_power / np.mean(total_active_power)
        
        # 创建图形和子图
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 绘制总有功功率曲线
        ax1.plot(hours, total_active_power, 'b-', label='Active load')
        ax1.set_xlabel('Time (hour)')
        ax1.set_ylabel('Active load (MW)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xlim(0, 24)
        # 设置时间刻度为3小时间隔
        ax1.set_xticks(np.arange(0, 25, 3))
        ax1.set_xticklabels([f'{h:d}:00' for h in range(0, 25, 3)])
        ax1.grid(True)
        
        # 创建第二个Y轴
        ax2 = ax1.twinx()
        ax2.plot(hours, ratio, 'r-', label='Ratio')
        ax2.set_ylabel('Ratio', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('24-hour Load Profile')
        # plt.tight_layout()
        
        # 保存图形到文件
        plt.savefig('load_profile_total.png')
        
        # 另外绘制每个负荷节点的曲线
        plt.figure(figsize=(12, 6))
        for i in range(num_loads):
            plt.plot(hours, self.load_profiles_p[i], label=f'Load {i}')
        
        plt.xlabel('Time (hour)')
        plt.ylabel('Active Power (MW)')
        plt.title('24-hour Load Profiles for Each Node')
        plt.grid(True)
        plt.legend()
        # plt.tight_layout()
        
        # 保存第二个图形到文件
        plt.savefig('load_profiles_individual.png')
        
        # 关闭所有图形
        plt.close('all')

    def _generate_load_profiles(self):
        """
        生成基于真实负荷模式的24小时负荷曲线
        具有明显的早晚高峰特征
        """
        num_timesteps = 288  # 24小时 * 12个5分钟
        num_loads = len(self.net.load)
        # num_timesteps = self.num_timesteps
        
        # 基础负荷值
        base_p = self.net.load['p_mw'].values
        base_q = self.net.load['q_mvar'].values
        
        # 初始化负荷数据数组
        self.load_profiles_p = np.zeros((num_loads, self.num_timesteps), dtype=np.float32)
        self.load_profiles_q = np.zeros((num_loads, self.num_timesteps), dtype=np.float32)
        
        # 创建时间轴
        time_hours = np.linspace(0, 24, num_timesteps)
        
        # 设置随机种子以保证一致性
        np.random.seed(42)
        
        # 定义双峰负荷模式的系数曲线（基于图像中的模式）
        load_pattern = np.zeros(num_timesteps)
        
        # 为每个时间点定义负荷系数
        for i, hour in enumerate(time_hours):
            if hour < 3:  # 0:00-3:00 凌晨低谷
                load_pattern[i] = 0.80 + 0.01 * (3 - hour)
            elif hour < 5:  # 3:00-5:00 保持平稳低谷
                load_pattern[i] = 0.80
            elif hour < 9:  # 5:00-9:00 早高峰上升
                load_pattern[i] = 0.80 + 0.20 * ((hour - 5) / 4)
            elif hour < 12:  # 9:00-12:00 早高峰下降
                load_pattern[i] = 1.00 - 0.10 * ((hour - 9) / 3)
            elif hour < 15:  # 12:00-15:00 中午降至低谷
                load_pattern[i] = 0.90 - 0.10 * ((hour - 12) / 3)
            elif hour < 18:  # 15:00-18:00 晚高峰上升
                load_pattern[i] = 0.80 + 0.15 * ((hour - 15) / 3)
            elif hour < 21:  # 18:00-21:00 晚高峰下降
                load_pattern[i] = 0.95 - 0.10 * ((hour - 18) / 3)
            else:  # 21:00-24:00 逐渐回落到低谷
                load_pattern[i] = 0.85 - 0.05 * ((hour - 21) / 3)
        
        # 确保负荷系数在合理范围内
        load_pattern = np.clip(load_pattern, 0.75, 1.0)
        # load_pattern = np.clip(load_pattern, 0.9, 1.1)
        
        # 应用负荷模式到每个负荷节点
        for i in range(num_loads):
            # 添加节点特定的小幅随机性（只针对基础的288步）
            node_variations = 0.01 * np.random.randn(len(load_pattern))
            node_pattern = load_pattern + node_variations 

            # node_pattern = np.clip(node_pattern, 0.9, 1.1)
            
            # 生成有功功率曲线（基础288步）
            base_profile_p = base_p[i] * node_pattern
            
            # 生成无功功率曲线（基础288步）
            base_profile_q = base_q[i] * node_pattern
            
            # 如果需要更多时间步，则重复基础模式
            if self.num_timesteps > num_timesteps:
                self.load_profiles_p[i] = np.tile(base_profile_p, self.num_timesteps // num_timesteps)
                self.load_profiles_q[i] = np.tile(base_profile_q, self.num_timesteps // num_timesteps)
            else:
                self.load_profiles_p[i] = base_profile_p
                self.load_profiles_q[i] = base_profile_q
        
        # 确保负荷值为正
        self.load_profiles_p = np.maximum(self.load_profiles_p, 0.1 * base_p[:, np.newaxis])
        self.load_profiles_q = np.maximum(self.load_profiles_q, 0.1 * base_q[:, np.newaxis])
        
        print(f"self.load_profiles_p.shape: {self.load_profiles_p.shape}")

        self.draw_load_curve = False
        if self.draw_load_curve:  # debug 绘制负荷曲线
            self.draw_load_profiles()
    
    def _generate_wind_profiles(self):
        """
        生成风电场的出力曲线
        针对IEEE 118-bus system中的节点59、90和116
        基于真实风电场的出力特征图精确匹配
        """
        num_timesteps = 288  # 24小时 * 12个5分钟
        time_hours = np.linspace(0, 24, num_timesteps)
        
        # 初始化风电出力数组
        self.wind_profiles = {
            59: np.zeros(num_timesteps),
            90: np.zeros(num_timesteps),
            116: np.zeros(num_timesteps)
        }
        
        # 设置随机种子以保证一致性
        np.random.seed(42)
        
        # 基础曲线 - Bus 59 (黄线)
        base_59 = np.zeros(num_timesteps)
        for i, hour in enumerate(time_hours):
            if hour < 3:  # 0:00-3:00 初始较高，快速下降
                base_59[i] = 10 - 2 * hour
            elif hour < 12:  # 3:00-12:00 维持低水平
                base_59[i] = 4 
            elif hour < 15:  # 12:00-15:00 快速上升到20MW
                base_59[i] = 4 + (20-4) * (hour - 12) / 3
            elif hour < 17:  # 15:00-17:00 维持在20MW左右
                base_59[i] = 20
            elif hour < 19:  # 17:00-19:00 上升到27MW左右并震荡
                base_59[i] = 20 + (27-20) * (hour - 17) / 2
            elif hour < 21:  # 19:00-21:00 快速下降
                base_59[i] = 27 - 11.5 * (hour - 19)
            else:  # 21:00-24:00 继续下降至4MW左右
                base_59[i] = 4 + 0.5 * np.sin((hour - 21) * np.pi / 3)
        
        # 基础曲线 - Bus 90 (蓝线)
        base_90 = np.zeros(num_timesteps)
        for i, hour in enumerate(time_hours):
            if hour < 1:  # 0:00-1:00 
                base_90[i] = 5 - 3 * hour
            elif hour < 8:  # 1:00-8:00 保持低水平
                base_90[i] = 1.5
            elif hour < 19:  # 8:00-19:00 中等水平，有规律波动
                base_90[i] = 3.5
            else:  # 19:00-24:00 升至高峰
                base_90[i] = 3.5 + 10 * (1 - np.exp(-(hour - 19) / 2))
        
        # 基础曲线 - Bus 116 (绿线) - 修改为9:00后逐步上升到15MW左右
        base_116 = np.zeros(num_timesteps)
        for i, hour in enumerate(time_hours):
            if hour < 2:  # 0:00-2:00 
                base_116[i] = 3.5 - 1.5 * hour
            elif hour < 6:  # 2:00-6:00 
                base_116[i] = 1 + 0.8 * (hour - 2)
            elif hour < 9:  # 6:00-9:00 维持在低水平并有震荡
                base_116[i] = 4 + 0.5 * np.sin((hour - 6) * np.pi)
            elif hour < 21:  # 9:00-21:00 逐步上升到15MW左右
                # 从4MW缓慢上升到15MW，使用非线性曲线使上升速度后期变缓
                progress = (hour - 9) / 12  # 0-1之间的进度值
                curve_factor = np.sqrt(progress)  # 非线性因子，使曲线更平滑
                base_116[i] = 4 + (15 - 4) * curve_factor
            else:  # 21:00-24:00 保持在15MW左右
                base_116[i] = 15 + 0.8 * np.sin((hour - 21) * np.pi / 1.5)
        
        # 添加波动 - 使用更合适的波动模式
        # 生成不同频率的噪声分量，然后组合
        for bus, base in zip([59, 90, 116], [base_59, base_90, base_116]):
            # 初始化波动分量
            oscillation = np.zeros(num_timesteps)
            
            # 添加长周期波动 (约3-4小时)
            num_long_waves = 3
            amp_long = 1.2 if bus == 90 else 1.0  # Bus 90波动更明显
            for j in range(num_long_waves):
                freq = 0.15 + 0.05 * j  # 频率变化
                phase = np.random.rand() * 2 * np.pi
                oscillation += amp_long * np.sin(freq * time_hours + phase)
            
            # 添加中周期波动 (约1-2小时)
            num_mid_waves = 5
            amp_mid = 0.7 if bus == 90 else 0.5
            for j in range(num_mid_waves):
                freq = 0.3 + 0.1 * j
                phase = np.random.rand() * 2 * np.pi
                oscillation += amp_mid * np.sin(freq * time_hours + phase)
            
            # 添加高频小震荡（约15-30分钟周期）
            num_high_freq_waves = 8
            amp_high = 0.3 if bus == 90 else 0.2
            for j in range(num_high_freq_waves):
                freq = 1.0 + 0.3 * j
                phase = np.random.rand() * 2 * np.pi
                oscillation += amp_high * np.sin(freq * time_hours + phase)
                
            # 添加更高频小震荡 (每5-10分钟)
            num_very_high_freq = 5
            amp_very_high = 0.15 if bus == 90 else 0.1
            for j in range(num_very_high_freq):
                freq = 2.5 + 0.8 * j
                phase = np.random.rand() * 2 * np.pi
                oscillation += amp_very_high * np.sin(freq * time_hours + phase)
            
            # 按照时间段调整波动幅度
            oscillation_amplitude = np.ones_like(base)
            
            if bus == 59:  # Bus 59在高峰期波动较大
                for i, hour in enumerate(time_hours):
                    if hour < 12:
                        oscillation_amplitude[i] = 0.5  # 低水平期波动增加
                    elif hour < 15:
                        oscillation_amplitude[i] = 0.6  # 上升期波动适中
                    elif hour < 17:
                        oscillation_amplitude[i] = 0.8  # 20MW平台期波动适中
                    elif hour < 19:
                        oscillation_amplitude[i] = 1.5  # 17:00-19:00上升到27MW时波动最大
                    else:
                        oscillation_amplitude[i] = 0.5  # 下降期波动减小
            
            elif bus == 90:  # Bus 90在全天都有显著波动
                for i, hour in enumerate(time_hours):
                    if hour < 9:
                        oscillation_amplitude[i] = 0.6  # 低水平期波动增加
                    elif hour < 19:
                        oscillation_amplitude[i] = 1.2  # 中间期波动加大
                    else:
                        oscillation_amplitude[i] = 2.0  # 高峰期波动最大
            
            elif bus == 116:  # Bus 116的波动调整
                for i, hour in enumerate(time_hours):
                    if hour < 9:
                        oscillation_amplitude[i] = 0.8  # 初始阶段中等波动
                    elif hour < 15:
                        oscillation_amplitude[i] = 0.9  # 上升初期适中波动
                    elif hour < 21:
                        oscillation_amplitude[i] = 1.2  # 上升后期波动加大
                    else:
                        oscillation_amplitude[i] = 1.5  # 稳定在高位时波动更明显
            
            # 特殊处理 - Bus 90在早上有一段时间接近零
            if bus == 90:
                zero_mask = (time_hours >= 6) & (time_hours <= 8)
                base[zero_mask] = 0.2
                oscillation_amplitude[zero_mask] = 0.1
            
            # 特殊处理 - Bus 116在8-9小时处有一个显著下降
            if bus == 116:
                dip_mask = (time_hours >= 8) & (time_hours <= 9)
                base[dip_mask] = 3.0  # 调整为轻微下降，但不会太低
                oscillation_amplitude[dip_mask] = 0.5
            
            # 特殊处理 - Bus 59在17:00-19:00时段有更强的震荡
            if bus == 59:
                strong_osc_mask = (time_hours >= 17) & (time_hours <= 19)
                # 添加额外的震荡分量
                extra_osc = 1.2 * np.sin(6 * time_hours[strong_osc_mask] + np.random.rand() * np.pi)
                self.wind_profiles[bus][strong_osc_mask] = base[strong_osc_mask]
                # 应用基础波动
                self.wind_profiles[bus] = base + oscillation * oscillation_amplitude
                # 叠加额外震荡
                self.wind_profiles[bus][strong_osc_mask] += extra_osc
            else:
                # 将基础曲线和波动组合
                self.wind_profiles[bus] = base + oscillation * oscillation_amplitude
            
            # 确保出力非负
            self.wind_profiles[bus] = np.maximum(self.wind_profiles[bus], 0)
            
            # 添加小的随机抖动，使曲线更符合实际
            jitter = 0.2 * np.random.randn(num_timesteps)
            self.wind_profiles[bus] += jitter
            
            # 轻微平滑处理，只是去除过于尖锐的波动
            from scipy.ndimage import gaussian_filter1d
            self.wind_profiles[bus] = gaussian_filter1d(self.wind_profiles[bus], sigma=0.3)
        
        # 确保所有曲线都在0-30范围内
        for bus in [59, 90, 116]:
            self.wind_profiles[bus] = np.clip(self.wind_profiles[bus], 0, 30)

        
        # 将负荷曲线复制300次，生成更长的时间序列   
        if self.num_timesteps > num_timesteps:  # 如果时间步数大于num_timesteps，则将负荷曲线复制，满足num_timesteps长度要求
            for bus in [59, 90, 116]:
                self.wind_profiles[bus] = np.tile(self.wind_profiles[bus], (self.num_timesteps // num_timesteps)) 
        
        # 添加绘制风电出力曲线的功能
        self.draw_wind_curve = False
        if self.draw_wind_curve:
            self.draw_wind_profiles()
        
    def draw_wind_profiles(self):
        """绘制风电场的出力曲线，风格与原始图片一致""" 
        
        num_timesteps = len(self.wind_profiles[59])
        hours = np.linspace(0, 24, num_timesteps)
        
        # 设置图表样式以匹配原始图像
        plt.figure(figsize=(12, 8))
        
        # 使用与原图相似的颜色
        plt.plot(hours, self.wind_profiles[59], color='#FFA500', label='Bus 59', linewidth=1.5)
        plt.plot(hours, self.wind_profiles[90], color='#1E90FF', label='Bus 90', linewidth=1.5)
        plt.plot(hours, self.wind_profiles[116], color='#228B22', label='Bus 116', linewidth=1.5)
        
        # 设置轴标签和标题
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Wind generation (MW)', fontsize=20)
        plt.title('24-hour Wind Power Generation Profile', fontsize=16)
        
        # 设置网格
        plt.grid(True, linestyle='-', alpha=0.7)
        
        # 设置刻度
        plt.yticks(np.arange(0, 31, 5), fontsize=14)
        plt.xticks(np.arange(0, 25, 3), [f'{h:d}:00' for h in range(0, 25, 3)], fontsize=14)
        
        # 设置y轴范围，与原图一致
        plt.ylim(0, 30)
        plt.xlim(0, 24)
        
        # 添加图例
        plt.legend(fontsize=16, loc='upper left')
        
        # 添加边框
        ax = plt.gca()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        # plt.tight_layout()
        
        # 保存图形到文件
        plt.savefig('wind_profiles.png', dpi=300)
        plt.close()

    def _create_network(self):
        """Create a pandapower network for the environment"""
        # net = pp.create_empty_network()
        # ... create your network components ... 
        if self.case_name == "case5":
            net = pn.case5()
        elif self.case_name == "case9":
            net = pn.case9()
        elif self.case_name == "case30":
            # net = pn.case30()
            net = pn.case_ieee30()
        elif self.case_name == "case57":
            net = pn.case57()
        elif self.case_name == "case118":
            net = pn.case118()
        print(f"\n {self.case_name} created")
        return net

    def _get_observation(self):
        """Convert pandapower network state to observation vector
        
        注意：在pandapower中，平衡发电机被转换为ext_grid，
        因此net.gen不包含平衡发电机的数据。
        这里会提取完整的发电机功率（包括平衡机）。
        """
        # 提取相关信息作为观测
        load_p = self.net.load['p_mw'][self.pd_idx].values  # 负荷的有功功率
        load_q = self.net.load['q_mvar'][self.qd_idx].values  # 负荷的无功功率
        
        # 提取发电机有功功率（包括平衡机） 
        # gen_p = self.net.gen['p_mw'][self.Pg_idx].values
        # gen_balance = self.net.res_ext_grid['p_mw'].values
 
        # # 根据self.PowerSystemConfig中的平衡发电机索引，重建完整的发电机数组 
        # balance_gen_idx = self.PowerSystemConfig.balance_gen
        # total_gens = len(gen_p) + len(gen_balance)
        # all_gen_p = np.zeros(total_gens)
        # indexs = np.zeros(total_gens, dtype=int)
        # # 先将gen_p批量赋值到all_gen_p，跳过平衡发电机索引
        # mask = np.arange(total_gens) != balance_gen_idx
        # all_gen_p[mask] = gen_p
        # indexs[mask] = self.net.gen['bus'].values
        # all_gen_p[balance_gen_idx] = gen_balance[0] 
        # indexs[balance_gen_idx] = self.net.ext_grid['bus'].values
        # all_gen_p = pd.Series(all_gen_p, index=indexs)
        # P_gen_p = all_gen_p[self.PowerSystemConfig.Pg_bus] 
 
        # from pypower.idx_gen import PG
        # ppc_gen_p = self.PowerSystemConfig.ppc['gen'][self.PowerSystemConfig.ppc['gen'][:, PG]>0, PG] 
        # observation = np.concatenate([load_p, load_q, gen_p]) 
        observation = np.concatenate([load_p, load_q])
        # 将单位MW/MVar 转化为p.u.
        observation = observation / 100   # todo: 这块得需要思考下不同的case 会不会归一化不一样
        # 确保返回float32类型，与模型训练时的数据类型一致
        return observation.astype(np.float32)
    
    def update_load_profiles(self):
        """Update the load profiles for the next time step"""
        self.net.load['p_mw'] = self.load_profiles_p[:, self.current_step]
        self.net.load['q_mvar'] = self.load_profiles_q[:, self.current_step] 

        """更新风电出力曲线，并调整相应节点的负荷"""
        if self.case_name == "case118" and self.consider_renewable_generation:
            # 更新风电出力
            # print(f"更新风电出力，当前时间步为{self.current_step}")
            wind_buses = [59, 90, 116]
            for bus in wind_buses:
                wind_power = self.wind_profiles[bus][self.current_step]
                
                # 查找该节点是否有负荷
                load_indices = self.net.load[self.net.load.bus == bus].index
                if len(load_indices) > 0:
                    # 如果节点有负荷，则调整负荷值（减去风电出力）
                    load_idx = load_indices[0]
                    original_load = self.load_profiles_p[load_idx][self.current_step]
                    
                    # 负荷等于总负荷减去新能源出力
                    adjusted_load = max(0.1, original_load - wind_power)  # 确保负荷不小于0.1
                    self.net.load.at[load_idx, 'p_mw'] = adjusted_load
                    
                    # 按相同比例调整无功功率
                    # if original_load > 0:
                    #     ratio = adjusted_load / original_load
                    #     original_q = self.load_profiles_q[load_idx][self.current_step]
                    #     self.net.load.at[load_idx, 'q_mvar'] = original_q * ratio

    def _calculate_reward(self):
        """Calculate reward based on network state and physical constraints"""
        if not self.converged:
            return -100  # Large penalty for non-convergence
        
        # 1. 发电成本最小化 (假设二次函数形式的发电成本)
        gen_costs = 0
        
        # 1.1 计算发电机成本
        for i, gen in enumerate(self.net.gen.itertuples()):
            p_g = gen.p_mw
            # 使用pandapower案例中的多项式成本系数
            if hasattr(self.net, 'poly_cost') and len(self.net.poly_cost) > 0:
                # 查找对应发电机的成本系数
                gen_cost_data = self.net.poly_cost[(self.net.poly_cost.et == 'gen') & 
                                                  (self.net.poly_cost.element == i)]
                
                # 使用二次多项式成本: cp2*p^2 + cp1*p + cp0
                cp2 = gen_cost_data.cp2_eur_per_mw2.values[0] 
                cp1 = gen_cost_data.cp1_eur_per_mw.values[0] 
                cp0 = gen_cost_data.cp0_eur.values[0] 
                gen_cost = cp2 * p_g**2 + cp1 * p_g # + cp0

            else:
                # 默认成本系数
                raise ValueError(f"发电机 {i} 的成本系数是手动设置的")
            
            gen_costs += gen_cost
        
        # 1.2 计算外接电源成本
        ext_costs = 0
        pc = self.net.poly_cost
        pc_ext = pc[(pc.et == 'ext_grid')].copy()
        if hasattr(self.net, 'res_ext_grid') and len(self.net.ext_grid) > 0:
            for i, ext_grid in enumerate(self.net.res_ext_grid.itertuples()):
                # 从结果表中获取外接电源的实际功率
                p_ext = ext_grid.p_mw
                
                # 查找对应外接电源的成本系数
                if hasattr(self.net, 'poly_cost') and len(self.net.poly_cost) > 0:
                    ext_cost_data = pc_ext[pc_ext.element == ext_grid.Index]
                    # 使用二次多项式成本: cp2*p^2 + cp1*p + cp0
                    cp2 = ext_cost_data.cp2_eur_per_mw2.values[0] 
                    cp1 = ext_cost_data.cp1_eur_per_mw.values[0] 
                    cp0 = ext_cost_data.cp0_eur.values[0] 
                    ext_cost = cp2 * p_ext**2 + cp1 * p_ext #  + cp0

                else:
                    # 没有poly_cost表时的默认处理
                    raise ValueError(f"外接电源 {i} 的成本系数是手动设置的")
                
                ext_costs += ext_cost
        
        # 总发电成本 = 发电机成本 + 外接电源成本
        total_gen_costs = gen_costs + ext_costs
        
        # 1.3 计算碳成本 (Carbon-aware OPF)
        # 碳成本函数: c_g^C(P_g) = (τ · GCI_g) × P_g
        # 其中 τ 为碳税率 ($/tCO2), GCI_g 为发电机g的碳排放强度 (tCO2/MWh)
        carbon_costs = 0
        total_carbon_emission = 0  # 总碳排放量 (tCO2)
        
        if self.carbon_tax > 0:
            # 检查GCI列是否存在
            if 'GCI' in self.net.gen.columns:
                # 1.3.1 计算发电机的碳成本
                for i, gen in enumerate(self.net.gen.itertuples()):
                    p_g = gen.p_mw  # 发电功率 (MW)
                    gci = gen.GCI   # 碳排放强度 (tCO2/MWh)
                    
                    # 碳排放量 = GCI × P_g (tCO2)
                    carbon_emission = gci * p_g
                    total_carbon_emission += carbon_emission
                    
                    # 碳成本 = τ × 碳排放量 ($)
                    carbon_cost = self.carbon_tax * carbon_emission
                    carbon_costs += carbon_cost
                
                # 1.3.2 计算外接电源（平衡机）的碳成本
                if hasattr(self.net, 'res_ext_grid') and len(self.net.ext_grid) > 0:
                    for i, ext_grid in enumerate(self.net.res_ext_grid.itertuples()):
                        p_ext = ext_grid.p_mw  # 外接电源功率 (MW)
                        
                        # 如果ext_grid也有GCI信息，使用它；否则使用默认值
                        if 'GCI' in self.net.ext_grid.columns:
                            gci = self.net.ext_grid.GCI.iloc[ext_grid.Index]
                        else:
                            # 默认使用系统平均碳排放强度或设为0
                            gci = 0.0  # 可以根据需要设置默认值
                        
                        carbon_emission = gci * p_ext
                        total_carbon_emission += carbon_emission
                        
                        carbon_cost = self.carbon_tax * carbon_emission
                        carbon_costs += carbon_cost
        
        # 总成本 = 经济成本 + 碳成本
        total_costs = total_gen_costs + carbon_costs
        
        # 2，3. 发电机有功、无功功率限制违反惩罚
        p_violation = 0
        q_violation = 0
        for gen in self.net.gen.itertuples():
            p_min = gen.min_p_mw if hasattr(gen, 'min_p_mw') else 0
            p_max = gen.max_p_mw if hasattr(gen, 'max_p_mw') else float('inf')
            p_g = gen.p_mw
            p_violation += max(0, p_min - p_g) + max(0, p_g - p_max)

            # 直接从res_gen获取发电机的无功功率
            q_g = self.net.res_gen.q_mvar.iloc[gen.Index]
            q_min = gen.min_q_mvar if hasattr(gen, 'min_q_mvar') else 0
            q_max = gen.max_q_mvar if hasattr(gen, 'max_q_mvar') else float('inf')
            q_violation += max(0, q_min - q_g) + max(0, q_g - q_max)
        
        # 4. 所有节点电压限制违反惩罚
        v_violation = 0 
        for i, bus in enumerate(self.net.res_bus.itertuples()):
            v_pu = bus.vm_pu
            v_violation += max(0, self.voltage_low[i] - v_pu) + max(0, v_pu - self.voltage_high[i])
        
        # 5. 所有支路电流限制违反惩罚
        i_violation = 0
        if hasattr(self.net, 'res_line'):
            for line in self.net.res_line.itertuples():
                i_ka = abs(line.i_ka)  # 取电流绝对值
                # 从原始line数据中获取最大电流限制
                line_idx = line.Index
                i_max = self.net.line.max_i_ka.iloc[line_idx]
                i_violation += max(0, i_ka - i_max)
        
        # 加权惩罚项
        # w1,w2,w3 单位为 MW^-1, w4(αv)单位为 p.u.^-1
        w1, w2, w3 = 1.0, 1.0, 1.0  # 有功/无功/电压违反惩罚系数 (1/MW)
        w4 = 100.0  # 电流违反惩罚系数 αv (1/p.u.)
        total_penalty = (w1 * p_violation +  # MW * (1/MW) = 1
                        w2 * q_violation +   # Mvar * (1/MW) = 1  
                        w3 * v_violation +   # p.u. * (1/MW) = p.u./MW
                        w4 * i_violation)    # p.u. * (1/p.u.) = 1
        
        # 最终奖励 = 负的总成本（经济成本+碳成本） - 惩罚项
        # 注意：在碳感知OPF中，目标是最小化 (经济成本 + 碳成本)
        a = 10**-4
        reward = a * total_costs + total_penalty
        
        # 记录约束违反程度和碳排放信息
        constraint_violations = {
            'p_violation': p_violation,
            'q_violation': q_violation,
            'v_violation': v_violation,
            'i_violation': i_violation,
            'total_penalty': total_penalty,
            'economic_costs': total_gen_costs,      # 经济成本 ($)
            'carbon_costs': carbon_costs,           # 碳成本 ($)
            'total_costs': total_costs,             # 总成本 = 经济成本 + 碳成本 ($)
            'carbon_emission': total_carbon_emission,  # 总碳排放量 (tCO2)
            'carbon_tax_rate': self.carbon_tax      # 碳税率 ($/tCO2)
        }
        
        # 将约束违反信息保存到环境中，以便外部访问
        self.constraint_violations = constraint_violations
        return reward
 
    def reset(self, scenario_idx=None):
        """
        Reset the environment to an initial state
        
        Args:
            scenario_idx: 指定要使用的场景索引（仅在使用外部场景时有效）
                         如果为None，则按顺序使用下一个场景
        """ 
        # self.net = self._create_network()
        self.done = False
        
        # 初始化时间步
        self.current_step = 0
        
        # 设置初始负荷
        if self.use_external_scenarios:
            # 使用外部提供的场景
            if scenario_idx is not None:
                self.current_scenario_idx = scenario_idx
            else:
                # 按顺序使用下一个场景，循环使用
                self.current_scenario_idx = self.current_scenario_idx % self.num_scenarios
            
            # 从load_profiles中获取当前场景的负荷数据（已在初始化时填充）
            self.net.load['p_mw'] = self.load_profiles_p[:, self.current_scenario_idx]
            self.net.load['q_mvar'] = self.load_profiles_q[:, self.current_scenario_idx]
            
            # 更新碳税率（如果提供了外部碳税场景）
            if self.external_carbon_tax_scenarios is not None:
                self.carbon_tax = self.external_carbon_tax_scenarios[self.current_scenario_idx]
            
            # 准备下一次reset使用下一个场景
            self.current_scenario_idx = (self.current_scenario_idx + 1) % self.num_scenarios
            
        else:
            # 使用生成的负荷曲线
            self.net.load['p_mw'] = self.load_profiles_p[:, 0]   
            self.net.load['q_mvar'] = self.load_profiles_q[:, 0]

        self.constraint_violations = None

        # 如果是case118，更新负荷以考虑可再生能源的出力
        if 'case118' in self.case_name and self.consider_renewable_generation and not self.use_external_scenarios:
            # 检查是否有风电场节点
            wind_farm_buses = [59, 90, 116]
            for bus in wind_farm_buses:
                if bus in self.net.load.bus.values:
                    # 找到对应的负荷索引
                    load_idx = self.net.load[self.net.load.bus == bus].index[0]
                    # 获取当前时间步的风电出力
                    wind_power = self.wind_profiles[bus][0]  # 初始时间步为0
                    # 更新负荷值 = 原始负荷 - 风电出力
                    original_load = self.load_profiles_p[load_idx, 0]
                    # 确保负荷减去风电后不会变为负值
                    new_load = max(0.1, original_load - wind_power)
                    # 更新负荷值
                    self.net.load.at[load_idx, 'p_mw'] = new_load 
        
        # 设置初始发电机功率
        # 方法1：根据负荷总和按比例分配 
        total_load = np.sum(self.net.load['p_mw'].values)
        num_gens = len(self.net.gen)
        # 考虑一些损耗，总发电量略大于总负荷
        total_gen = total_load * 1.05 

        # 只对可调节发电机分配功率
        adjustable_gens = self.net.gen['max_p_mw'] > 0
        num_adjustable = adjustable_gens.sum()

        if num_adjustable > 0:
            # 对可调节发电机均匀分配
            self.net.gen.loc[adjustable_gens, 'p_mw'] = total_gen / num_adjustable
            # 不可调节发电机设为0
            self.net.gen.loc[~adjustable_gens, 'p_mw'] = 0
        else:
            # 如果没有可调节发电机,使用原始逻辑
            self.net.gen['p_mw'] = np.ones(num_gens) * (total_gen / num_gens)
        
        # 运行潮流以更新res_gen，确保observation中的gen_p是正确的 
        # pp.runpp(self.net) 
        # Get initial observation
        observation = self._get_observation()  # reset
        # print(f"初始观察: {observation}")
        return observation
    
    def step(self, action):  
        """Execute one time step within the environment"""  
        # Calculate reward
        # 鲁棒设计，如果action的维度是二维的，则给它降成一维度的
        if len(action.shape) == 2:
            action = action.squeeze()
        if self.run_pp:
            # update the active power and voltage of the generator nodes P_g, V_g
            v_g = action[:len(self.gen_bus_idx)]   # p.u.
            p_g = action[len(self.gen_bus_idx):]   # mw

            self.net.gen['vm_pu'] = v_g
            self.net.gen.loc[self.Pg_idx, 'p_mw'] = p_g
            self.net.gen.loc[~self.Pg_idx, 'p_mw'] = 0
            try:
                pp.runpp(self.net)   # run the power flow calculation by the newton-raphson method
                self.converged = True
            except:
                self.converged = False
                print("潮流计算不收敛") 

            reward = self._calculate_reward() 
        else:
            reward = None
            self.converged = True

        self.current_step += 1 # 更新时间步

        # Get new observation
        if self.current_step < self.num_timesteps: 
            self.update_load_profiles()             # update the load value of the current time step
            observation = self._get_observation()   # step 
        else:
            observation = self._get_observation()   # step 

        # Check if episode is done
        if not self.converged:
            self.done = True
        if self.current_step >= self.num_timesteps:
            self.done = True
        
        # Additional info
        info = {}

        return observation, reward, self.done, info
    
    # def step_pre(self, action):
    #     """Execute one time step within the environment"""  
    #     # update the active power and voltage of the generator nodes P_g, V_g
    #     v_g = action[:self.num_buses]
    #     p_g = action[self.num_buses:]
        
    #     # 修复：检查p_g的维度是否与发电机数量匹配
    #     if len(p_g) == len(self.net.gen):
    #         # 如果p_g的长度等于发电机数量，则直接按发电机索引分配
    #         for gen_idx, gen in enumerate(self.net.gen.itertuples()):
    #             bus_idx = gen.bus  # 获取发电机连接的母线索引
    #             gen_table_idx = gen.Index  # 获取发电机在gen表中的索引
                
    #             # 使用发电机索引而不是母线索引
    #             self.net.gen.at[gen_table_idx, 'p_mw'] = p_g[gen_idx]
    #             self.net.gen.at[gen_table_idx, 'vm_pu'] = v_g[bus_idx]
    #     else:
    #         # 原始逻辑：如果p_g的长度等于母线数量
    #         for gen in self.net.gen.itertuples():
    #             bus_idx = gen.bus  # 获取发电机连接的母线索引
    #             gen_idx = gen.Index  # 获取发电机在gen表中的索引
                
    #             # 直接使用母线索引从动作中获取对应的设定值
    #             if bus_idx < len(p_g):  # 添加边界检查
    #                 self.net.gen.at[gen_idx, 'p_mw'] = p_g[bus_idx]
    #                 self.net.gen.at[gen_idx, 'vm_pu'] = v_g[bus_idx]
        
    #     # Calculate reward 
    #     try:
    #         pp.runpp(self.net)   # run the power flow calculation by the newton-raphson method
    #         self.converged = True
    #     except:
    #         self.converged = False
    #         print("潮流计算不收敛") 

    #     reward = self._calculate_reward()
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            # Print key network statistics
            print(f"Bus Voltages: {self.net.res_bus.vm_pu.values}")
            print(f"Line Loadings: {self.net.res_line.loading_percent.values}")
        return

    def update_gen_constraints(self):
        if self.current_step != 0:
            # 向量化处理，避免for循环
            previous_p = self.net.res_gen.p_mw.values

            # 取原始上下限
            min_p_mw_o = self.min_p_mw_o.values
            max_p_mw_o = self.max_p_mw_o.values

            # 只对可调节发电机(max_p_mw > 0)应用斜坡约束
            threshold = self.ramp.detach().cpu().numpy() * 100   # mw

            adjustable_mask = self.Pg_idx.to_numpy()

            # 初始化新的上下限为原始值
            new_min_p = min_p_mw_o.copy()
            new_max_p = max_p_mw_o.copy()

            # 对可调节发电机应用斜坡约束
            if np.any(adjustable_mask):
                min_p_ramp = previous_p[adjustable_mask] - threshold
                max_p_ramp = previous_p[adjustable_mask] + threshold
                
                # 确保斜坡约束在原始约束范围内
                new_min_p[adjustable_mask] = np.maximum(min_p_ramp, min_p_mw_o[adjustable_mask])
                new_max_p[adjustable_mask] = np.minimum(max_p_ramp, max_p_mw_o[adjustable_mask])
                
                # 确保可行域有效(min <= max)
                new_min_p[adjustable_mask] = np.minimum(new_min_p[adjustable_mask], new_max_p[adjustable_mask])

            # 更新DataFrame
            self.net.gen['min_p_mw'] = new_min_p
            self.net.gen['max_p_mw'] = new_max_p
        else:
            self.min_p_mw_o = copy.deepcopy(self.net.gen.min_p_mw)
            self.max_p_mw_o = copy.deepcopy(self.net.gen.max_p_mw)  

    def _generate_random_load_profiles(self):
        """
        生成随机负荷曲线，服从默认负荷正负10%范围内的均匀分布
        
        参数:
        num_timesteps (int): 负荷曲线的时间步长，默认为288（24小时×12个5分钟）
        """
        num_loads = len(self.net.load)
        num_timesteps = self.num_timesteps
        
        # 基础负荷值
        base_p = self.net.load['p_mw'].values
        base_q = self.net.load['q_mvar'].values
        
        # 初始化负荷数据数组
        self.load_profiles_p = np.zeros((num_loads, num_timesteps), dtype=np.float32)
        self.load_profiles_q = np.zeros((num_loads, num_timesteps), dtype=np.float32)
        
        # 设置随机种子以保证一致性
        # np.random.seed(42)    # todo: 不随机了，也就是初始化的时候负荷是不固定的
        
        # 为每个负荷节点生成随机负荷曲线
        for i in range(num_loads):
            # 生成服从均匀分布的随机系数，范围为0.9到1.1（即默认负荷的正负10%）
            random_factors_p = np.random.uniform(0.9, 1.1, num_timesteps)
            random_factors_q = np.random.uniform(0.9, 1.1, num_timesteps)
            
            # 应用随机系数到基础负荷值
            self.load_profiles_p[i] = base_p[i] * random_factors_p
            self.load_profiles_q[i] = base_q[i] * random_factors_q
        
        # 计算并存储活跃负荷和比率数据用于可视化
        # self.active_load_total = np.sum(self.load_profiles_p, axis=0)
        # self.ratio_data = self.active_load_total / np.mean(self.active_load_total)

        # 对于调试目的，如果需要可以绘制负荷曲线
        self.draw_load_curve = False
        if self.draw_load_curve:
            self.draw_load_profiles()

    def verify_power_balance(self):
        """
        快速验证运行pp.runpp()后各节点的功率平衡情况
        显示每个节点的净负荷、发电机功率和外接电源功率
        """
        if not hasattr(self.net, 'res_bus') or self.net.res_bus.empty:
            print("错误：请先运行 pp.runpp(self.net) 进行潮流计算")
            return
        
        print("="*80)
        print("电网功率平衡验证报告")
        print("="*80)
        
        # 1. 总体功率平衡
        total_load_p = self.net.load.p_mw.sum() if hasattr(self.net, 'load') else 0
        total_load_q = self.net.load.q_mvar.sum() if hasattr(self.net, 'load') else 0
        total_gen_p = self.net.res_gen.p_mw.sum() if hasattr(self.net, 'res_gen') else 0
        total_gen_q = self.net.res_gen.q_mvar.sum() if hasattr(self.net, 'res_gen') else 0
        total_ext_p = self.net.res_ext_grid.p_mw.sum() if hasattr(self.net, 'res_ext_grid') else 0
        total_ext_q = self.net.res_ext_grid.q_mvar.sum() if hasattr(self.net, 'res_ext_grid') else 0
        
        print(f"\n【总体功率平衡】")
        print(f"总负荷：{total_load_p:.2f} MW (有功), {total_load_q:.2f} Mvar (无功)")
        print(f"总发电：{total_gen_p:.2f} MW (有功), {total_gen_q:.2f} Mvar (无功)")
        print(f"外接电源：{total_ext_p:.2f} MW (有功), {total_ext_q:.2f} Mvar (无功)")
        print(f"功率平衡 (发电+外接-负荷)：{(total_gen_p + total_ext_p - total_load_p):.4f} MW (有功)")
        print(f"功率平衡 (发电+外接-负荷)：{(total_gen_q + total_ext_q - total_load_q):.4f} Mvar (无功)")
        
        # 2. 逐节点功率分析
        print(f"\n【各节点功率详情】")
        print(f"{'节点':<4} {'类型':<12} {'电压(pu)':<10} {'相角(°)':<10} {'净负荷P(MW)':<12} {'净负荷Q(Mvar)':<12} {'发电P(MW)':<10} {'发电Q(Mvar)':<10} {'外接P(MW)':<10} {'外接Q(Mvar)':<10}")
        print("-" * 120)
        
        for bus_idx in range(len(self.net.bus)):
            # 节点基本信息
            vm_pu = self.net.res_bus.vm_pu.iloc[bus_idx]
            va_deg = self.net.res_bus.va_degree.iloc[bus_idx]
            
            # 负荷功率
            load_p = 0
            load_q = 0
            if hasattr(self.net, 'load'):
                load_at_bus = self.net.load[self.net.load.bus == bus_idx]
                if not load_at_bus.empty:
                    load_p = load_at_bus.p_mw.sum()
                    load_q = load_at_bus.q_mvar.sum()
            
            # 发电机功率
            gen_p = 0
            gen_q = 0
            if hasattr(self.net, 'gen') and hasattr(self.net, 'res_gen'):
                gen_at_bus = self.net.gen[self.net.gen.bus == bus_idx]
                if not gen_at_bus.empty:
                    for gen_idx in gen_at_bus.index:
                        # 在res_gen中找到对应的发电机结果
                        res_gen_idx = list(self.net.gen.index).index(gen_idx)
                        gen_p += self.net.res_gen.p_mw.iloc[res_gen_idx]
                        gen_q += self.net.res_gen.q_mvar.iloc[res_gen_idx]
            
            # 外接电源功率
            ext_p = 0
            ext_q = 0
            if hasattr(self.net, 'ext_grid') and hasattr(self.net, 'res_ext_grid'):
                ext_at_bus = self.net.ext_grid[self.net.ext_grid.bus == bus_idx]
                if not ext_at_bus.empty:
                    for ext_idx in ext_at_bus.index:
                        res_ext_idx = list(self.net.ext_grid.index).index(ext_idx)
                        ext_p += self.net.res_ext_grid.p_mw.iloc[res_ext_idx]
                        ext_q += self.net.res_ext_grid.q_mvar.iloc[res_ext_idx]
            
            # 判断节点类型
            node_type = "PQ节点"
            if hasattr(self.net, 'ext_grid') and bus_idx in self.net.ext_grid.bus.values:
                node_type = "平衡节点"
            elif hasattr(self.net, 'gen') and bus_idx in self.net.gen.bus.values:
                if load_p > 0:
                    node_type = "PV节点(发电+负荷)"
                else:
                    node_type = "PV节点(发电)"
            elif load_p == 0:
                node_type = "空节点"
            
            # 打印节点信息
            print(f"{bus_idx:<4} {node_type:<12} {vm_pu:<10.4f} {va_deg:<10.2f} {load_p:<12.2f} {load_q:<12.2f} {gen_p:<10.2f} {gen_q:<10.2f} {ext_p:<10.2f} {ext_q:<10.2f}")
        
        # 3. 约束检查
        print(f"\n【约束违反检查】")
        # 电压约束
        v_violations = 0
        for i, bus in enumerate(self.net.res_bus.itertuples()):
            v_pu = bus.vm_pu
            if v_pu < self.voltage_low[i] or v_pu > self.voltage_high[i]:
                v_violations += 1
                print(f"节点 {i} 电压违约：{v_pu:.4f} pu (限制：{self.voltage_low[i]:.2f} - {self.voltage_high[i]:.2f})")
        
        if v_violations == 0:
            print("[OK] 所有节点电压均在允许范围内")
        
        # 发电机约束
        if hasattr(self.net, 'gen') and hasattr(self.net, 'res_gen'):
            gen_violations = 0
            for i, gen in enumerate(self.net.gen.itertuples()):
                p_g = self.net.res_gen.p_mw.iloc[i]
                p_min = gen.min_p_mw if hasattr(gen, 'min_p_mw') else 0
                p_max = gen.max_p_mw if hasattr(gen, 'max_p_mw') else float('inf')
                
                if p_g < p_min or p_g > p_max:
                    gen_violations += 1
                    print(f"发电机 {i} (节点 {gen.bus}) 功率违约：{p_g:.2f} MW (限制：{p_min:.2f} - {p_max:.2f})")
            
            if gen_violations == 0:
                print("[OK] 所有发电机功率均在允许范围内")
        
        # 线路约束
        if hasattr(self.net, 'res_line'):
            line_violations = 0
            for line in self.net.res_line.itertuples():
                loading = line.loading_percent
                if loading > 100:
                    line_violations += 1
                    print(f"线路 {line.Index} 过载：{loading:.1f}%")
            
            if line_violations == 0:
                print("[OK] 所有线路负载率均在允许范围内")
        
        print("="*80)

    def quick_power_summary(self):
        """
        快速显示功率平衡摘要
        """
        if not hasattr(self.net, 'res_bus') or self.net.res_bus.empty:
            print("请先运行 pp.runpp(self.net)")
            return
        
        # 统计各类节点数量和功率
        total_load_p = self.net.load.p_mw.sum() if hasattr(self.net, 'load') else 0
        total_gen_p = self.net.res_gen.p_mw.sum() if hasattr(self.net, 'res_gen') else 0
        total_ext_p = self.net.res_ext_grid.p_mw.sum() if hasattr(self.net, 'res_ext_grid') else 0
        
        print(f"\n功率平衡快速摘要：")
        print(f"负荷总计：{total_load_p:.2f} MW")
        print(f"发电总计：{total_gen_p:.2f} MW")
        print(f"外接电源：{total_ext_p:.2f} MW")
        print(f"平衡差值：{(total_gen_p + total_ext_p - total_load_p):.4f} MW")
        print(f"潮流收敛：{'是' if self.converged else '否'}")



if __name__ == "__main__":
    # 创建环境实例
    case_name = "case118"   
    env = PowerGridEnv(case_name=case_name)
    
    # 重置环境
    obs = env.reset()
    print("初始观测值:", obs)
    
    # 创建一个简单的动作
    num_buses = len(env.net.bus)
    v_g = np.ones(num_buses)  # 所有节点电压设为1.0标幺值
    p_g = np.zeros(num_buses)  # 创建发电机出力矩阵
    
    # 对发电机节点设置出力
    for i, gen in enumerate(env.net.gen.itertuples()):
        bus_idx = gen.bus
        p_g[bus_idx] = 50  # 设置发电机出力为50MW
    action = np.concatenate([v_g, p_g])
    # 执行一步
    obs, reward, done, info = env.step(action)
    print("\n执行一步后:")
    print("观测值:", obs)
    print("奖励值:", reward)
    print("是否结束:", done)
    
    # 渲染环境状态
    print("\n环境状态:")
    env.render()


