#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyPower OPF Solver - 使用 PYPOWER 求解最优潮流

使用方法:
  # 单目标优化 (从 .mat 文件加载负荷)
  python opf_by_pypower.py --nsamples 100

  # 多目标优化 (成本 + 碳排放)
  python opf_by_pypower.py --nsamples 100 --multi_objective --lambda_carbon 5

  # 自动生成负荷场景 (无需 .mat 文件)
  python opf_by_pypower.py --generate --nsamples 100

依赖: pypower (`pip install pypower`)
"""

import re
import numpy as np
from typing import Dict, Optional


# =============================================================================
# MATPOWER .m 文件解析
# =============================================================================

def load_case_from_m(case_m_path: str) -> dict:
    """从 MATPOWER .m 文件加载电网数据"""
    with open(case_m_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    
    # 解析 baseMVA
    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9eE\.\+\-]+)", txt)
    if not m:
        raise KeyError("Cannot find mpc.baseMVA in .m file")
    baseMVA = float(m.group(1))
    
    def extract_matrix(name):
        m = re.search(rf"mpc\.{re.escape(name)}\s*=\s*\[", txt)
        if not m:
            raise KeyError(f"Cannot find 'mpc.{name}' in .m file")
        start = m.end()
        end = txt.find("];", start)
        block = txt[start:end]
        
        # 清理注释和续行符
        lines = []
        for line in block.splitlines():
            line = line.split("%", 1)[0].replace("...", " ")
            if line.strip():
                lines.append(line)
        
        # 解析数值
        rows = [r.strip() for r in "\n".join(lines).split(";") if r.strip()]
        float_re = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
        data = [[float(x) for x in re.findall(float_re, r)] for r in rows]
        maxlen = max(len(row) for row in data)
        return np.array([row + [np.nan] * (maxlen - len(row)) for row in data])
    
    return {
        "version": "2",
        "baseMVA": baseMVA,
        "bus": extract_matrix("bus"),
        "gen": extract_matrix("gen"),
        "branch": extract_matrix("branch"),
        "gencost": extract_matrix("gencost"),
    }


# =============================================================================
# 数据加载/生成
# =============================================================================

def _transpose_matlab(arr: np.ndarray) -> np.ndarray:
    """转置 MATLAB 数组（列优先 -> 行优先）"""
    if arr.ndim <= 1:
        return arr
    axes = tuple(reversed(range(arr.ndim)))
    return np.transpose(arr, axes=axes)


def _h5_read_dataset(f, obj) -> np.ndarray:
    """
    从 HDF5 文件读取 MATLAB v7.3 数据集
    处理对象引用和转置问题
    """
    import h5py as _h5
    
    if isinstance(obj, _h5.Dataset):
        # 检查是否是对象引用
        try:
            is_ref_dtype = obj.dtype == _h5.special_dtype(ref=_h5.Reference)
        except (TypeError, AttributeError):
            is_ref_dtype = False
        
        if obj.dtype.kind == 'O' or is_ref_dtype:
            ref = obj[()]
            if np.isscalar(ref):
                return _h5_read_dataset(f, f[ref])
            elif isinstance(ref, np.ndarray) and ref.size > 0:
                if ref.ndim == 0:
                    return _h5_read_dataset(f, f[ref.item()])
                # 多维引用数组
                out = np.empty(ref.shape, dtype=object)
                it = np.nditer(ref, flags=['multi_index', 'refs_ok'])
                for x in it:
                    out[it.multi_index] = _h5_read_dataset(f, f[x.item()])
                return out
        
        # 普通数据集，读取并转置
        arr = np.array(obj)
        return _transpose_matlab(arr)
    elif isinstance(obj, _h5.Group):
        # 组对象，递归读取
        return {k: _h5_read_dataset(f, obj[k]) for k in obj.keys()}
    
    return obj


def _load_mat_file(data_mat_path: str) -> dict:
    """
    加载 MATLAB .mat 文件，支持 v7.3 (HDF5) 和旧版本格式
    
    Args:
        data_mat_path: .mat 文件路径
    
    Returns:
        包含 RPd, RQd, RVm, RVa, load_idx 等字段的字典
    """
    # 尝试使用 h5py 读取 v7.3 格式
    try:
        import h5py
        with h5py.File(data_mat_path, 'r') as f:
            keys = list(f.keys())
            mat = {}
            
            # 需要的变量名
            required_vars = ['RPd', 'RQd', 'RVm', 'RVa', 'load_idx']
            
            # 尝试直接从根目录读取
            for key in required_vars:
                if key in keys:
                    try:
                        mat[key] = _h5_read_dataset(f, f[key])
                    except Exception as e:
                        # 如果读取失败，继续尝试其他方法
                        pass
            
            # 如果成功读取到必需变量，返回
            if 'RPd' in mat and 'RQd' in mat and len(mat) >= 2:
                return mat
            
            # 如果读取不完整，尝试通过引用读取
            # 某些 MATLAB v7.3 文件使用引用系统
            for key in required_vars:
                if key not in mat and key in keys:
                    try:
                        obj = f[key]
                        # 尝试作为引用处理
                        mat[key] = _h5_read_dataset(f, obj)
                    except Exception:
                        pass
            
            if 'RPd' in mat and 'RQd' in mat:
                return mat
            
            # 如果仍未成功，抛出异常，让 scipy 尝试
            raise ValueError("无法从 HDF5 文件中读取所需变量")
            
    except (OSError, ImportError, ValueError):
        # 不是 v7.3 格式、h5py 不可用或读取失败，继续尝试 scipy
        pass
    except Exception:
        # 其他错误，可能是文件格式问题，尝试 scipy
        pass
    
    # 回退到 scipy.io.loadmat（旧版本格式）
    try:
        import scipy.io
        mat = scipy.io.loadmat(data_mat_path)
        return mat
    except Exception as e:
        raise RuntimeError(
            f"无法读取 .mat 文件 '{data_mat_path}'。\n"
            f"请确保文件是有效的 MATLAB 格式。\n"
            f"如果文件是 v7.3 格式，请安装 h5py: pip install h5py\n"
            f"原始错误: {str(e)}"
        )


def load_data(case_m_path: str, 
              data_mat_path: Optional[str] = None,
              n_samples: int = 100,
              delta: float = 0.1,
              seed: Optional[int] = None) -> Dict:
    """
    加载或生成 OPF 求解所需的负荷数据
    
    Args:
        case_m_path: MATPOWER .m 文件路径
        data_mat_path: 训练数据 .mat 文件路径。如果为 None，自动生成负荷场景。
        n_samples: 样本数量
        delta: 自动生成时的每节点负荷变化范围 (±10%)
        seed: 随机种子
    
    Returns:
        {
            "x_load_pu": [n_samples, 2*nbus] 负荷向量 (p.u.)
            "Vm_ref": [n_samples, nbus] 参考电压 (如果有)
            "Va_ref_deg": [n_samples, nbus] 参考相角 (如果有)
            "baseMVA": float
            "nbus": int
        }
    """
    ppc = load_case_from_m(case_m_path)
    baseMVA = float(ppc["baseMVA"])
    nbus = ppc["bus"].shape[0]
    
    # 模式1: 从 .mat 文件加载
    if data_mat_path is not None:
        mat = _load_mat_file(data_mat_path)
        
        RPd = mat['RPd']
        RQd = mat['RQd']
        load_idx = np.squeeze(mat['load_idx']).astype(int) - 1
        
        n_samples = min(n_samples, RPd.shape[0])
        
        # 展开到全部节点
        Pd = np.zeros((n_samples, nbus))
        Qd = np.zeros((n_samples, nbus))
        Pd[:, load_idx] = RPd[:n_samples]
        Qd[:, load_idx] = RQd[:n_samples]
        
        x_load_pu = np.concatenate([Pd / baseMVA, Qd / baseMVA], axis=1)
        
        return {
            "x_load_pu": x_load_pu,
            "Vm_ref": mat['RVm'][:n_samples],
            "Va_ref_deg": mat['RVa'][:n_samples],
            "baseMVA": baseMVA,
            "nbus": nbus,
        }
    
    # 模式2: 自动生成负荷场景
    Pd_base = ppc["bus"][:, 2] / baseMVA  # p.u.
    Qd_base = ppc["bus"][:, 3] / baseMVA
    
    rng = np.random.default_rng(seed)
    k = rng.uniform(1.0 - delta, 1.0 + delta, size=(n_samples, nbus))
    
    Pd = Pd_base * k
    Qd = Qd_base * k
    x_load_pu = np.concatenate([Pd, Qd], axis=1)
    
    return {
        "x_load_pu": x_load_pu,
        "Vm_ref": None,
        "Va_ref_deg": None,
        "baseMVA": baseMVA,
        "nbus": nbus,
    }


# =============================================================================
# PyPowerOPFSolver 类
# =============================================================================

class PyPowerOPFSolver:
    """
    使用 PYPOWER 求解 AC 最优潮流 (OPF)
    
    Example:
        solver = PyPowerOPFSolver("case300.m")
        result = solver.forward(x_load_pu)
        print(result["summary"]["total_cost"])
        print(result["bus"]["Vm"])
    """
    
    def __init__(self, case_m_path: str, 
                 use_multi_objective: bool = False, 
                 lambda_carbon: float = 0.0,
                 verbose: bool = False):
        """
        Args:
            case_m_path: MATPOWER .m 文件路径
            use_multi_objective: 是否启用多目标优化 (成本 + 碳排放)
            lambda_carbon: 碳排放权重。目标函数: cost + lambda_carbon * carbon_emission
            verbose: 是否打印初始化信息
        """
        from pypower.api import runopf
        from pypower.ppoption import ppoption
        
        self.runopf = runopf
        self.ppc_base = load_case_from_m(case_m_path)
        self.baseMVA = float(self.ppc_base["baseMVA"])
        self.nbus = self.ppc_base["bus"].shape[0]
        self.ngen = self.ppc_base["gen"].shape[0]
        
        # 找到 slack 节点
        self.slack_row = int(np.where(self.ppc_base["bus"][:, 1] == 3)[0][0])
        
        # OPF 选项
        self.ppopt = ppoption(VERBOSE=0, OUT_ALL=0, OPF_VIOLATION=1e-4, FEASTOL=1e-4)
        
        # 多目标设置
        self.use_multi_objective = use_multi_objective
        self.lambda_carbon = lambda_carbon
        self.gci_values = None
        self.idxPg = None
        self.gencost_original = None
        
        if use_multi_objective:
            # 计算 GCI (碳排放强度)
            self.gencost_original = self.ppc_base["gencost"].copy()
            gen = self.ppc_base["gen"]
            self.idxPg = np.where(gen[:, 8] > 0)[0]  # Pmax > 0
            self.gci_values = self._compute_gci()
            
            if verbose:
                print(f"[Multi-Objective] 目标: cost + {lambda_carbon} * carbon")
                print(f"  Active generators: {len(self.idxPg)}, "
                      f"GCI: [{self.gci_values.min():.3f}, {self.gci_values.max():.3f}] tCO2/MWh")
        
        if verbose:
            print(f"[PyPowerOPFSolver] nbus={self.nbus}, ngen={self.ngen}, slack={self.slack_row}")
    
    def _compute_gci(self) -> np.ndarray:
        """根据发电成本计算碳排放强度 (GCI)"""
        gencost = self.ppc_base["gencost"]
        col_c1 = 5 if gencost.shape[1] > 4 else 1
        c1 = gencost[:self.ngen, col_c1]
        
        # 按成本分位数分配燃料类型
        p25, p50, p75 = np.percentile(c1, [25, 50, 75])
        gci_lookup = {"coal": 0.85, "oil": 0.70, "gas": 0.52, "ccgt": 0.36}
        
        gci = np.zeros(self.ngen)
        for i in range(self.ngen):
            if c1[i] <= p25:
                gci[i] = gci_lookup["coal"]
            elif c1[i] <= p50:
                gci[i] = gci_lookup["oil"]
            elif c1[i] <= p75:
                gci[i] = gci_lookup["gas"]
            else:
                gci[i] = gci_lookup["ccgt"]
        
        return gci[self.idxPg]
    
    def forward(self, x_load_pu: np.ndarray, 
                lambda_carbon: Optional[float] = None,
                cost_perturb: Optional[float] = None,
                perturb_seed: Optional[int] = None) -> Dict:
        """
        求解 OPF
        
        Args:
            x_load_pu: 负荷向量 [2*nbus]，格式 [Pd_all, Qd_all] (p.u.)
            lambda_carbon: 碳排放权重 (可选，覆盖初始化值)
            cost_perturb: 成本扰动比例 (如 0.05 表示 ±5%)。用于生成 multi-valued mapping。
            perturb_seed: 扰动的随机种子
        
        Returns:
            {
                "success": bool,
                "bus": {"Vm": [...], "Va_rad": [...], ...},
                "gen": {"Pg_MW": [...], ...},
                "summary": {"total_cost": ..., "economic_cost": ..., "carbon_emission": ..., ...}
            }
        """
        x = np.asarray(x_load_pu).reshape(-1)
        lam_c = lambda_carbon if lambda_carbon is not None else self.lambda_carbon
        
        # 解析负荷
        if len(x) == 2 * self.nbus:
            Pd_pu, Qd_pu = x[:self.nbus], x[self.nbus:]
        else:
            raise ValueError(f"x_load_pu 长度必须为 2*nbus={2*self.nbus}, 实际为 {len(x)}")
        
        # 克隆基础案例
        ppc = {k: v.copy() if hasattr(v, "copy") else v for k, v in self.ppc_base.items()}
        
        # 设置负荷
        Pd_pu = np.maximum(Pd_pu, 0)
        Qd_pu = np.maximum(Qd_pu, 0)
        ppc["bus"][:, 2] = Pd_pu * self.baseMVA
        ppc["bus"][:, 3] = Qd_pu * self.baseMVA
        
        # 成本扰动 (用于生成 multi-valued mapping)
        gencost = ppc["gencost"].copy()
        if cost_perturb is not None and cost_perturb > 0:
            rng = np.random.default_rng(perturb_seed)
            col_c2 = 4 if gencost.shape[1] > 4 else 0
            col_c1 = 5 if gencost.shape[1] > 4 else 1
            # 对二次项和一次项系数添加扰动
            perturb_c2 = 1.0 + rng.uniform(-cost_perturb, cost_perturb, size=self.ngen)
            perturb_c1 = 1.0 + rng.uniform(-cost_perturb, cost_perturb, size=self.ngen)
            gencost[:self.ngen, col_c2] *= perturb_c2
            gencost[:self.ngen, col_c1] *= perturb_c1
        
        # 多目标: 修改 gencost
        if self.use_multi_objective and lam_c > 0 and self.gci_values is not None:
            col_c1 = 5 if gencost.shape[1] > 4 else 1
            for i, gen_idx in enumerate(self.idxPg):
                gencost[gen_idx, col_c1] += lam_c * self.gci_values[i]
        
        ppc["gencost"] = gencost
        
        # 运行 OPF
        result = self.runopf(ppc, self.ppopt)
        
        if not result.get("success", 0):
            return {"success": False, "error": "OPF did not converge"}
        
        # 解析结果
        bus, gen = result["bus"], result["gen"]
        
        # 计算经济成本
        if self.use_multi_objective and self.gencost_original is not None:
            gc = self.gencost_original
            col_c2, col_c1 = (4, 5) if gc.shape[1] > 4 else (0, 1)
            Pg = gen[self.idxPg, 1]
            economic_cost = float(np.sum(gc[self.idxPg, col_c2] * Pg**2 + gc[self.idxPg, col_c1] * Pg))
        else:
            economic_cost = float(result.get("f", 0))
        
        # 计算碳排放
        carbon = 0.0
        if self.use_multi_objective and self.gci_values is not None:
            Pg = np.maximum(gen[self.idxPg, 1], 0)
            carbon = float(np.sum(self.gci_values * Pg))
        
        return {
            "success": True,
            "bus": {
                "Vm": bus[:, 7],
                "Va_deg": bus[:, 8],
                "Va_rad": np.deg2rad(bus[:, 8]),
                "Pd_MW": bus[:, 2],
                "Qd_MVAr": bus[:, 3],
            },
            "gen": {
                "Pg_MW": gen[:, 1],
                "Qg_MVAr": gen[:, 2],
            },
            "summary": {
                "total_cost": economic_cost + lam_c * carbon,
                "economic_cost": economic_cost,
                "carbon_emission": carbon,
                "total_Pg_MW": float(np.sum(gen[:, 1])),
                "total_Pd_MW": float(np.sum(bus[:, 2])),
                "lambda_carbon": lam_c,
            },
        }


# =============================================================================
# 数据格式转换函数已移至 main_part/utils.py
# 使用 utils.convert_pypower_result_to_matlab_format() 进行格式转换
# =============================================================================


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="PyPower OPF Solver")
    parser.add_argument("--case_m", default="main_part/data/case300_ieee_modified.m")
    parser.add_argument("--data_mat", default="main_part/generate_data/XY_case300real_db.mat", help="训练数据 .mat 文件")
    parser.add_argument("--generate", action="store_true", help="自动生成负荷场景")
    parser.add_argument("--nsamples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--multi_objective", action="store_true")
    parser.add_argument("--lambda_carbon", type=float, default=0.0) 
    
    args = parser.parse_args()
    
    # 确定数据来源
    if args.data_mat is None and not args.generate:
        default_mat = "main_part/data/XY_case300real.mat"
        if os.path.exists(default_mat):
            args.data_mat = default_mat
        else:
            args.generate = True
    
    # 加载数据
    print("=" * 60)
    if args.generate:
        print("自动生成负荷场景...")
        data = load_data(args.case_m, None, args.nsamples, args.delta, args.seed)
    else:
        print(f"从 {args.data_mat} 加载数据...")
        data = load_data(args.case_m, args.data_mat, args.nsamples, seed=args.seed)
    print(f"样本数: {data['x_load_pu'].shape[0]}, 节点数: {data['nbus']}")
    
    # 多目标默认 lambda_carbon
    if args.multi_objective and args.lambda_carbon == 0:
        args.lambda_carbon = 1.0
    
    # 创建求解器
    solver = PyPowerOPFSolver(
        args.case_m,
        use_multi_objective=args.multi_objective,
        lambda_carbon=args.lambda_carbon,
        verbose=True
    )
    
    # 批量求解
    print(f"\n求解 {args.nsamples} 个样本...")
    print("-" * 60)
    
    results = {"success": 0, "cost": [], "carbon": [], "vm_mae": [], "va_mae": []}
    has_ref = data["Vm_ref"] is not None
    opf_results_list = []  # 保存所有 OPF 结果用于格式转换验证
    
    for i in range(min(args.nsamples, data["x_load_pu"].shape[0])):
        r = solver.forward(data["x_load_pu"][i])
        opf_results_list.append(r)
        
        if not r["success"]:
            continue
        
        results["success"] += 1
        results["cost"].append(r["summary"]["economic_cost"])
        results["carbon"].append(r["summary"]["carbon_emission"])
        
        if has_ref:
            results["vm_mae"].append(np.mean(np.abs(r["bus"]["Vm"] - data["Vm_ref"][i])))
            results["va_mae"].append(np.mean(np.abs(r["bus"]["Va_rad"] - np.deg2rad(data["Va_ref_deg"][i]))))
        
        # 打印前5个
        if i < 5:
            s = r["summary"]
            line = f"[{i+1}] 成本={s['economic_cost']:.0f}"
            if args.multi_objective:
                line += f", 碳排放={s['carbon_emission']:.0f} tCO2/h"
            if has_ref:
                line += f", Vm_MAE={results['vm_mae'][-1]:.4e}"
            print(line)
    
    # 统计
    print("-" * 60)
    n = results["success"]
    print(f"成功: {n}/{args.nsamples} ({100*n/args.nsamples:.0f}%)")
    if results["cost"]:
        print(f"经济成本: {np.mean(results['cost']):.0f} ± {np.std(results['cost']):.0f}")
    if results["carbon"]:
        print(f"碳排放: {np.mean(results['carbon']):.0f} ± {np.std(results['carbon']):.0f} tCO2/h")
    if results["vm_mae"]:
        print(f"Vm MAE: {np.mean(results['vm_mae']):.4e}")
        print(f"Va MAE: {np.mean(results['va_mae']):.4e} rad ({np.rad2deg(np.mean(results['va_mae'])):.3f} deg)") 

    print("=" * 60)


if __name__ == "__main__":
    main()
