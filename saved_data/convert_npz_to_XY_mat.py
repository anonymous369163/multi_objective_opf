import numpy as np
import scipy.io
from typing import Optional, Dict, Any


def convert_npz_to_XY_mat(
    x_npz_path: str,
    y_npz_path: str,
    out_mat_path: str,
    case_m_path: Optional[str] = None,
    slack_bus: Optional[int] = None,
    vm_lb: float = 0.94,
    vm_ub: float = 1.06,
    eps_nonzero_load: float = 1e-12,
    keep_only_success: bool = True,
    include_cost_as_rf: bool = False,
    include_id_sample: bool = True,
) -> Dict[str, Any]:
    """
    将 expand_training_data_multi_preference.py 产生的中间数据：
      - x_train.npz: x_load_pu [n, 2*nbus], nbus, baseMVA
      - y_train_lcXX.npz: solutions [n, 2*nbus-1] = [Va_noslack(rad), Vm(p.u.)]
    转为与 XY_case300real.mat 一致的字段格式：
      RPd [n, n_loads] MW
      RQd [n, n_loads] MVAr
      RVm [n, nbus] p.u.
      RVa [n, nbus] degrees
      load_idx [n_loads, 1] uint16, MATLAB 1-based
      VmLb [1,1], VmUb [1,1]

    强烈建议提供 case_m_path，以自动定位 slack bus（bus type==3）。
    """

    # ----------------------------
    # 1) load npz
    # ----------------------------
    x_data = np.load(x_npz_path)
    y_data = np.load(y_npz_path)

    x_load_pu = np.asarray(x_data["x_load_pu"])
    nbus = int(x_data["nbus"])
    baseMVA = float(x_data["baseMVA"])

    solutions = np.asarray(y_data["solutions"])
    success_mask = np.asarray(y_data["success_mask"]).astype(bool) if "success_mask" in y_data else None
    costs = np.asarray(y_data["costs"]) if "costs" in y_data else None

    if x_load_pu.ndim != 2 or x_load_pu.shape[1] != 2 * nbus:
        raise ValueError(f"x_load_pu shape expected [n, {2*nbus}], got {x_load_pu.shape}")

    if solutions.ndim != 2 or solutions.shape[1] != (2 * nbus - 1):
        raise ValueError(
            f"solutions shape expected [n, {2*nbus-1}] = [n, Va(nbus-1)+Vm(nbus)], got {solutions.shape}"
        )

    n_total = x_load_pu.shape[0]
    if solutions.shape[0] != n_total:
        raise ValueError(f"Row mismatch: x has {n_total} samples, y has {solutions.shape[0]} samples")

    # ----------------------------
    # 2) decide slack bus index
    # ----------------------------
    slack_row = None

    if slack_bus is not None:
        slack_row = int(slack_bus)
    elif case_m_path is not None:
        # 复用项目里的 load_case_from_m（最稳）
        # 要求你的环境里 opf_by_pypower.py 可 import
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))  # saved_data/
        project_root = os.path.dirname(current_dir)  # 项目根目录
        generate_data_dir = os.path.join(project_root, 'main_part', 'generate_data')
        if generate_data_dir not in sys.path:
            sys.path.insert(0, generate_data_dir)
        
        from opf_by_pypower import load_case_from_m

        ppc = load_case_from_m(case_m_path)
        bus = ppc["bus"]
        # bus[:,1] == 3 表示 reference/slack bus（与 PyPowerOPFSolver 一致）
        slack_bus_indices = np.where(bus[:, 1] == 3)[0]
        if len(slack_bus_indices) == 0:
            raise ValueError(f"No slack bus (bus type == 3) found in case file: {case_m_path}")
        elif len(slack_bus_indices) > 1:
            raise ValueError(f"Multiple slack buses found (indices: {slack_bus_indices}), expected exactly one")
        slack_row = int(slack_bus_indices[0])

        # 交叉校验（可选，但建议）
        nbus_case = int(bus.shape[0])
        if nbus_case != nbus:
            raise ValueError(f"nbus mismatch: npz nbus={nbus}, case file nbus={nbus_case}")
        baseMVA_case = float(ppc["baseMVA"])
        # 如果不一致，以 case 文件为准更安全
        baseMVA = baseMVA_case
    else:
        raise ValueError("No slack bus found")

    if not (0 <= slack_row < nbus):
        raise ValueError(f"Invalid slack_row={slack_row} for nbus={nbus}")

    # ----------------------------
    # 3) filter success samples
    # ----------------------------
    if keep_only_success and success_mask is not None:
        keep = success_mask
    else:
        keep = np.ones(n_total, dtype=bool)

    x_ok = x_load_pu[keep]
    sol_ok = solutions[keep]
    costs_ok = costs[keep] if costs is not None else None
    kept_indices = np.where(keep)[0]

    n = x_ok.shape[0]
    if n == 0:
        raise ValueError("No samples left after filtering success_mask.")

    # ----------------------------
    # 4) build load_idx (non-zero load buses) from x
    #    RPd/RQd only keep those buses; load_idx is MATLAB 1-based
    # ----------------------------
    Pd_pu = x_ok[:, :nbus]
    Qd_pu = x_ok[:, nbus:]

    # 某个 bus 只要在任意样本出现过非零负荷，就认为是 load bus
    load_mask = (np.max(np.abs(Pd_pu), axis=0) > eps_nonzero_load) | (np.max(np.abs(Qd_pu), axis=0) > eps_nonzero_load)
    load_idx0 = np.where(load_mask)[0]  # 0-based
    load_idx = (load_idx0 + 1).astype(np.uint16).reshape(-1, 1)  # MATLAB 1-based, [n_loads,1]

    # 转成 MW/MVAr（实际值），符合 XY_case300real 规范
    RPd = (Pd_pu[:, load_idx0] * baseMVA).astype(np.float64)
    RQd = (Qd_pu[:, load_idx0] * baseMVA).astype(np.float64)

    # ----------------------------
    # 5) reconstruct full RVm / RVa
    #    solutions = [Va_noslack(rad), Vm(p.u.)]
    # ----------------------------
    Va_noslack_rad = sol_ok[:, : (nbus - 1)]
    RVm = sol_ok[:, (nbus - 1) :].astype(np.float64)  # [n, nbus]

    # 把 Va_noslack 填回到全长 nbus（slack 位置补 0）
    RVa_rad = np.zeros((n, nbus), dtype=np.float64)
    mask_full = np.ones(nbus, dtype=bool)
    mask_full[slack_row] = False
    RVa_rad[:, mask_full] = Va_noslack_rad.astype(np.float64)
    RVa_rad[:, slack_row] = 0.0

    # rad -> degree
    RVa = (RVa_rad * (180.0 / np.pi)).astype(np.float64)

    # ----------------------------
    # 6) pack & save mat
    # ----------------------------
    mat_dict: Dict[str, Any] = {
        "RPd": RPd,
        "RQd": RQd,
        "RVm": RVm,
        "RVa": RVa,
        "load_idx": load_idx,
        "VmLb": np.array([[vm_lb]], dtype=np.float64),
        "VmUb": np.array([[vm_ub]], dtype=np.float64),
    }

    # 可选：保存 costs 到 Rf（原格式里是可选字段）
    if include_cost_as_rf and costs_ok is not None:
        mat_dict["Rf"] = costs_ok.astype(np.float64).reshape(1, -1)

    # 可选：保存保留下来的样本原始下标，方便追溯（不影响 loader）
    if include_id_sample:
        # 这里我用 0-based 原始样本 index（你也可以改成 +1）
        mat_dict["id_sample"] = kept_indices.astype(np.int32).reshape(1, -1)

    scipy.io.savemat(out_mat_path, mat_dict, do_compression=True)

    return {
        "out_mat_path": out_mat_path,
        "n_samples": int(n),
        "nbus": int(nbus),
        "baseMVA": float(baseMVA),
        "slack_row": int(slack_row),
        "n_loads": int(load_idx.shape[0]),
    }



if __name__ == "__main__":
    info = convert_npz_to_XY_mat(
        x_npz_path="saved_data/multi_preference_solutions/x_train.npz",
        y_npz_path="saved_data/multi_preference_solutions/y_train_lc0.00.npz",
        out_mat_path="main_part/data/XY_case118real_from_npz_lc0.00.mat",
        case_m_path="main_part/data/case118_ieee_modified.m",   # 强烈建议提供（用于 slack_row）
        include_cost_as_rf=True,                                # 可选
    )
    print(info)
