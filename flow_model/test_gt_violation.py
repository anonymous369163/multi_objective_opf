"""测试GT数据的约束违反"""
import sys
import os
import torch
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import PowerGridEnv
from flow_model.load_opf_data_v2 import OPF_Flow_Dataset_V2
from flow_model.models.actor import PowerSystemConfig


def denormalize_voltage(Vm_norm, Va_norm):
    Vm_pu = Vm_norm * 0.06 + 1.0
    Va_rad = Va_norm * math.pi / 6.0
    return Vm_pu, Va_rad


def convert_to_env_action(Vm_norm, Va_norm, x_input, env):
    device = Vm_norm.device
    Vm_pu, Va_rad = denormalize_voltage(Vm_norm, Va_norm)
    Vm = Vm_pu.T
    Va = Va_rad.T
    Vreal = Vm * torch.cos(Va)
    Vimg = Vm * torch.sin(Va)
    G = env.G.to(device)
    B = env.B.to(device)
    Ireal = torch.matmul(G, Vreal) - torch.matmul(B, Vimg)
    Iimg = torch.matmul(B, Vreal) + torch.matmul(G, Vimg)
    P = Vreal * Ireal + Vimg * Iimg
    
    Pd = x_input.T[:env.num_pd].to(device)
    Pg = P.clone()
    pd_bus_idx = torch.from_numpy(env.pd_bus_idx).long().to(device)
    Pg[pd_bus_idx] = Pg[pd_bus_idx] + Pd
    
    Pg_bus_idx = torch.from_numpy(env.Pg_bus_idx).long().to(device)
    gen_bus_idx = torch.from_numpy(env.gen_bus_idx).long().to(device)
    Pg_gen = Pg[Pg_bus_idx].T * 100
    Vg = Vm[gen_bus_idx].T
    
    action = torch.cat([Vg, Pg_gen], dim=1)
    return action


def main():
    # 加载数据
    print("Loading data...")
    data = OPF_Flow_Dataset_V2(
        data_path='../saved_data/training_data_case118_40k.npz',
        device='cpu', test_ratio=0.2, add_carbon_tax=False, single_target=True
    )
    
    # 使用与数据生成时相同的配置 (pglib_opf_case118.mat)
    print("\nCreating PowerSystemConfig from pglib_opf_case118.mat...")
    ps_config = PowerSystemConfig(
        device='cpu', 
        case_file_path='../saved_data/pglib_opf_case118.mat'
    )
    
    print("Creating environment with PowerSystemConfig...")
    env = PowerGridEnv(
        num_timesteps=288, case_name='case118', 
        random_load=False, run_pp=True, device='cpu',
        PowerSystemConfig=ps_config  # 使用pglib的配置！
    )
    
    # 验证发电机限制是否正确
    print("\n[Verifying generator limits from pglib config...]")
    for bus in [25, 58, 99]:
        gen_mask = env.net.gen.bus == bus
        if gen_mask.any():
            g = env.net.gen[gen_mask].iloc[0]
            print(f"  Bus {bus}: max_p_mw = {g.max_p_mw:.1f} MW")
    
    # 测试多个GT样本
    test_indices = [0, 100, 500, 1000]
    
    for idx in test_indices:
        x = data.x_train[idx:idx+1]
        y_gt = data.y_train[idx:idx+1]
        
        print(f"\n{'='*60}")
        print(f"Testing GT sample {idx}")
        print(f"{'='*60}")
        
        # 设置负荷
        Pd = x[0, :env.num_pd].numpy() * 100
        Qd = x[0, env.num_pd:env.num_pd+env.num_qd].numpy() * 100
        
        env.reset()
        for i, bus_idx in enumerate(env.pd_bus_idx):
            env.net.load.at[env.net.load[env.net.load.bus == bus_idx].index[0], 'p_mw'] = Pd[i]
        for i, bus_idx in enumerate(env.qd_bus_idx):
            load_rows = env.net.load[env.net.load.bus == bus_idx]
            if len(load_rows) > 0:
                env.net.load.at[load_rows.index[0], 'q_mvar'] = Qd[i]
        
        print(f"  Total Pd: {Pd.sum():.2f} MW")
        print(f"  Total Qd: {Qd.sum():.2f} MVar")
        
        # 反归一化GT查看
        Vm_pu, Va_rad = denormalize_voltage(y_gt[:, :118], y_gt[:, 118:])
        print(f"  GT Vm range: {Vm_pu.min().item():.4f} to {Vm_pu.max().item():.4f}")
        print(f"  GT Va range: {Va_rad.min().item():.4f} to {Va_rad.max().item():.4f} rad")
        
        # 调试：检查转换过程
        Vm_norm = y_gt[:, :118]
        Va_norm = y_gt[:, 118:]
        Vm_pu, Va_rad = denormalize_voltage(Vm_norm, Va_norm)
        
        print(f"  Vm_pu shape: {Vm_pu.shape}")
        print(f"  gen_bus_idx: {env.gen_bus_idx[:5]}... (len={len(env.gen_bus_idx)})")
        
        # 手动提取Vg检查
        Vg_manual = Vm_pu[0, env.gen_bus_idx]
        print(f"  Vg (manual) range: {Vg_manual.min().item():.4f} to {Vg_manual.max().item():.4f}")
        
        # 转换为action
        action = convert_to_env_action(y_gt[:, :118], y_gt[:, 118:], x, env)
        num_gen = len(env.gen_bus_idx)
        print(f"  Action shape: {action.shape}")
        print(f"  Action Vg range (0:{num_gen}): {action[0, :num_gen].min().item():.4f} to {action[0, :num_gen].max().item():.4f}")
        print(f"  Action Pg range ({num_gen}:): {action[0, num_gen:].min().item():.2f} to {action[0, num_gen:].max().item():.2f} MW")
        print(f"  Total Pg: {action[0, num_gen:].sum().item():.2f} MW")
        
        # 调用env.step
        action_np = action.numpy().squeeze()
        obs, reward, done, info = env.step(action_np)
        
        if env.constraint_violations:
            cv = env.constraint_violations
            total = cv['p_violation'] + cv['q_violation'] + cv['v_violation'] + cv['i_violation']
            print(f"\n  Constraint violations:")
            print(f"    p_violation: {cv['p_violation']:.4f}")
            print(f"    q_violation: {cv['q_violation']:.4f}")
            print(f"    v_violation: {cv['v_violation']:.4f}")
            print(f"    i_violation: {cv['i_violation']:.4f}")
            print(f"    TOTAL: {total:.4f}")
            
            # 详细分析P violation
            print(f"\n  Detailed P violation analysis:")
            p_vio_count = 0
            for gen in env.net.gen.itertuples():
                p_min = gen.min_p_mw if hasattr(gen, 'min_p_mw') else 0
                p_max = gen.max_p_mw if hasattr(gen, 'max_p_mw') else float('inf')
                p_g = gen.p_mw
                vio = max(0, p_min - p_g) + max(0, p_g - p_max)
                if vio > 0.1:
                    p_vio_count += 1
                    print(f"    Gen {gen.Index} (bus {gen.bus}): p={p_g:.1f}, "
                          f"limit=[{p_min:.1f}, {p_max:.1f}], vio={vio:.2f}")
                    if p_vio_count >= 5:
                        print(f"    ... (showing first 5 violations)")
                        break
            
            # 检查ext_grid的功率
            if hasattr(env.net, 'res_ext_grid'):
                print(f"\n  Ext_grid power:")
                for i, ext in enumerate(env.net.res_ext_grid.itertuples()):
                    print(f"    Ext {i}: p={ext.p_mw:.2f} MW, q={ext.q_mvar:.2f} MVar")
        else:
            print("  Power flow FAILED!")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

