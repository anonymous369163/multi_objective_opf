#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the provided MATPOWER .m case (case300_ieee_modified.m) to build a PYPOWER/MATPOWER ppc,
then run OPF for load scenarios from your training set (ngt_data["x_train"]) and optionally
compare OPF voltages with labels (ngt_data["y_train"]). 

Run:
  python opf_case300_from_m_and_compare.py --case_m /path/to/case300_ieee_modified.m --nsamples 10

Assumes it is executed in the same repo environment where:
  - config.py (get_config)
  - data_loader.py (load_all_data, load_ngt_training_data)
exist (same assumption as your single_check.py).

Requirements:
- PYPOWER installed (`pip install pypower`) OR pandapower installed (fallback uses pandapower.pypower).
"""

import re
import numpy as np
from typing import Dict, Optional, Union, Tuple


# -----------------------
# 0) PyPowerOPFSolver Class
# -----------------------

class PyPowerOPFSolver:
    """
    A class for solving AC Optimal Power Flow (OPF) using PYPOWER/MATPOWER.
    
    This class loads a MATPOWER case file (.m format) and provides a forward method
    to solve OPF for given load scenarios, returning structured results including
    voltages, phase angles, active/reactive power, and generator outputs.
    
    Example:
        solver = PyPowerOPFSolver(case_m_path="data/case300_ieee_modified.m", ngt_data=ngt_data)
        result = solver.forward(x_load_pu)
        print(result["bus"]["Vm"])  # Voltage magnitudes
        print(result["gen"]["Pg_MW"])  # Generator active power
    """
    
    def __init__(self, case_m_path: str, ngt_data: Optional[Dict] = None, verbose: bool = False,
                 use_multi_objective: bool = False, lambda_cost: float = 0.9, lambda_carbon: float = 0.1,
                 carbon_scale: float = 1.0, sys_data=None, 
                 opf_violation: float = 1e-4, feastol: float = 1e-4):
        """
        Initialize the OPF solver by loading a MATPOWER case file.
        
        Args:
            case_m_path: Path to the MATPOWER .m case file
            ngt_data: Optional dictionary containing load mapping information 
                     (e.g., bus_Pd, bus_Qd keys for custom load bus indices)
            verbose: If True, print initialization information
            use_multi_objective: If True, enable multi-objective optimization (cost + carbon)
            lambda_cost: Weight for economic cost (default: 0.9)
            lambda_carbon: Weight for carbon emission (default: 0.1)
            carbon_scale: Scale factor for carbon emission (default: 30.0)
            sys_data: Optional PowerSystemData object for GCI calculation (required if use_multi_objective=True)
            opf_violation: OPF constraint violation tolerance (default: 1e-4, relaxed from 1e-6 for better convergence)
            feastol: Feasibility tolerance (default: 1e-4, relaxed from 1e-6 for better convergence)
        """
        self.case_m_path = case_m_path
        self.ngt_data = ngt_data or {}
        self.verbose = verbose
        
        # [MOD] Multi-objective parameters
        self.use_multi_objective = use_multi_objective
        self.lambda_cost = lambda_cost
        self.lambda_carbon = lambda_carbon
        self.carbon_scale = carbon_scale
        
        # Load case from .m file
        self.ppc_base = load_case_from_m(case_m_path)
        self.baseMVA = float(self.ppc_base["baseMVA"])
        self.nbus = int(self.ppc_base["bus"].shape[0])
        self.ngen = int(self.ppc_base["gen"].shape[0])
        self.nbranch = int(self.ppc_base["branch"].shape[0])
        
        # Infer slack bus
        self.slack_row = _infer_slack_bus_row(self.ppc_base)
        self.slack_bus_id = int(self.ppc_base["bus"][self.slack_row, 0])
        
        # [MOD] Calculate GCI values for carbon emission if multi-objective is enabled
        self.gci_values = None
        self.idxPg = None
        self.gencost_original = None  # Save original gencost for economic cost calculation
        if self.use_multi_objective:
            if sys_data is None:
                # Try to get sys_data from ngt_data or raise error
                raise ValueError("sys_data is required when use_multi_objective=True. "
                               "Please provide sys_data parameter or load it from config/data_loader.")
            
            from main_part.utils import get_gci_for_generators
            # Get GCI values for all generators
            gci_all = get_gci_for_generators(sys_data)
            
            # Find active generators (Pmax > 0)
            gen = self.ppc_base["gen"]
            self.idxPg = np.where(gen[:, 8] > 0)[0]  # Column 8 is Pmax
            self.gci_values = gci_all[self.idxPg] if len(self.idxPg) > 0 else np.array([])
            
            # Save original gencost for later economic cost calculation
            self.gencost_original = self.ppc_base["gencost"].copy()
            
            if self.verbose:
                print(f"[Multi-Objective] Enabled")
                print(f"  lambda_cost={self.lambda_cost}, lambda_carbon={self.lambda_carbon}")
                print(f"  carbon_scale={self.carbon_scale}")
                print(f"  Active generators: {len(self.idxPg)}, GCI range: [{self.gci_values.min():.4f}, {self.gci_values.max():.4f}]")
        
        # Import PYPOWER solver
        self.runopf, self.ppoption = _safe_import_pypower()
        # Use relaxed tolerance to improve convergence for edge cases
        # Default: OPF_VIOLATION=1e-6 is too strict for some load scenarios
        # Relaxed: OPF_VIOLATION=1e-4 significantly improves convergence while maintaining solution quality
        # Users can further adjust via opf_violation and feastol parameters
        self.ppopt = self.ppoption(
            VERBOSE=0, 
            OUT_ALL=0,
            OPF_VIOLATION=opf_violation,  # Constraint violation tolerance (default: 1e-4)
            FEASTOL=feastol                # Feasibility tolerance (default: 1e-4)
        )
        
        if self.verbose:
            print(f"[PyPowerOPFSolver] Initialized")
            print(f"  Case file: {case_m_path}")
            print(f"  baseMVA={self.baseMVA}, nbus={self.nbus}, ngen={self.ngen}, nbranch={self.nbranch}")
            print(f"  Slack bus: row={self.slack_row} (0-based), bus_id={self.slack_bus_id}")
    
    def forward(self, x_load_pu: Union[np.ndarray, list], 
                preference: Optional[Union[np.ndarray, Tuple[float, float]]] = None) -> Dict:
        """
        Solve OPF for a given load scenario.
        
        Args:
            x_load_pu: Load vector in per-unit (p.u.) format. Can be:
                      - 2*nbus: [Pd_all, Qd_all] for all buses
                      - 2*(nbus-1): [Pd_non_slack, Qd_non_slack] excluding slack bus
                      - Custom format if ngt_data provides bus_Pd/bus_Qd indices
            preference: Optional preference weights [lambda_cost, lambda_carbon] for multi-objective.
                       If None and use_multi_objective=True, uses default weights from __init__.
                       Can be a tuple/list of 2 floats or a numpy array of shape (2,).
        
        Returns:
            Dictionary containing structured OPF results:
            {
                "success": bool,  # Whether OPF converged
                "load_mode": str,  # Load mapping mode used
                "raw_result": dict,  # Original PYPOWER result dictionary
                
                "bus": {
                    "bus_id": np.ndarray,  # Bus IDs (external numbering)
                    "Vm": np.ndarray,  # Voltage magnitude (p.u.)
                    "Va_deg": np.ndarray,  # Voltage phase angle (degrees)
                    "Va_rad": np.ndarray,  # Voltage phase angle (radians)
                    "Pd_MW": np.ndarray,  # Active power demand (MW)
                    "Qd_MVAr": np.ndarray,  # Reactive power demand (MVAr)
                    "Pg_MW": np.ndarray,  # Active power generation at bus (MW)
                    "Qg_MVAr": np.ndarray,  # Reactive power generation at bus (MVAr)
                },
                
                "gen": {
                    "gen_id": np.ndarray,  # Generator indices (0-based)
                    "bus_id": np.ndarray,  # Bus IDs where generators are connected
                    "Pg_MW": np.ndarray,  # Active power generation (MW)
                    "Qg_MVAr": np.ndarray,  # Reactive power generation (MVAr)
                    "Pmax_MW": np.ndarray,  # Maximum active power limit (MW)
                    "Pmin_MW": np.ndarray,  # Minimum active power limit (MW)
                    "Qmax_MVAr": np.ndarray,  # Maximum reactive power limit (MVAr)
                    "Qmin_MVAr": np.ndarray,  # Minimum reactive power limit (MVAr)
                },
                
                "branch": {
                    "from_bus": np.ndarray,  # From bus IDs
                    "to_bus": np.ndarray,  # To bus IDs
                    "Pf_MW": np.ndarray,  # Active power flow from->to (MW)
                    "Qf_MVAr": np.ndarray,  # Reactive power flow from->to (MVAr)
                    "Pt_MW": np.ndarray,  # Active power flow to->from (MW)
                    "Qt_MVAr": np.ndarray,  # Reactive power flow to->from (MVAr)
                },
                
                "summary": {
                    "total_cost": float,  # Total generation cost
                    "total_Pg_MW": float,  # Total active power generation (MW)
                    "total_Pd_MW": float,  # Total active power demand (MW)
                    "total_Qg_MVAr": float,  # Total reactive power generation (MVAr)
                    "total_Qd_MVAr": float,  # Total reactive power demand (MVAr)
                }
            }
            
            If OPF fails, returns:
            {
                "success": False,
                "load_mode": str,
                "raw_result": dict,
                "error": str  # Error message
            }
        """
        # Convert input to numpy array
        x_load_pu = np.asarray(x_load_pu, dtype=float).reshape(-1)
        
        # [MOD] Parse preference weights for multi-objective
        if self.use_multi_objective:
            if preference is not None:
                pref_arr = np.asarray(preference, dtype=float).reshape(-1)
                if len(pref_arr) != 2:
                    raise ValueError(f"preference must have 2 elements [lambda_cost, lambda_carbon], got {len(pref_arr)}")
                lam_cost = float(pref_arr[0])
                lam_carbon = float(pref_arr[1])
            else:
                lam_cost = self.lambda_cost
                lam_carbon = self.lambda_carbon
        else:
            lam_cost = 1.0
            lam_carbon = 0.0
        
        # Map load vector to Pd/Qd arrays
        Pd_pu, Qd_pu, load_mode = _infer_load_mapping(
            x_load_pu, self.nbus, self.slack_row, self.ngt_data
        )
        
        # Clone base case
        ppc = {k: (v.copy() if hasattr(v, "copy") else v) 
               for k, v in self.ppc_base.items()}
        
        # [MOD] Modify gencost for multi-objective optimization
        # PYPOWER uses: cost = c2*Pg^2 + c1*Pg + c0 (coefficients are in MW units, not p.u.)
        # For multi-objective: total_cost = lambda_cost * (c2*Pg^2 + c1*Pg + c0) + lambda_carbon * (carbon_scale * GCI * Pg)
        # We scale all original coefficients by lambda_cost, then add carbon cost to c1
        if self.use_multi_objective and lam_carbon > 0 and self.gci_values is not None and len(self.gci_values) > 0:
            # Get original gencost
            gencost = ppc["gencost"].copy()
            
            # Determine gencost format: MATPOWER (7 columns) or simplified (2+ columns)
            is_matpower_format = gencost.shape[1] > 4
            
            if is_matpower_format:
                # MATPOWER format: [MODEL, STARTUP, SHUTDOWN, NCOST, c2, c1, c0]
                # Column indices: 0=MODEL, 1=STARTUP, 2=SHUTDOWN, 3=NCOST, 4=c2, 5=c1, 6=c0
                col_n = 3  # NCOST column
                col_c2, col_c1, col_c0 = 4, 5, 6
            else:
                # Simplified format: [c2, c1] or [c2, c1, c0, ...]
                col_n = None
                col_c2, col_c1 = 0, 1
                col_c0 = 2 if gencost.shape[1] > 2 else None
            
            # Modify cost coefficients for active generators
            if lam_cost > 0:
                carbon_cost_per_MW = lam_carbon * self.carbon_scale * self.gci_values
                
                for i, gen_idx in enumerate(self.idxPg):
                    if gen_idx >= gencost.shape[0]:
                        continue
                    
                    # Determine polynomial order
                    if is_matpower_format:
                        n = int(gencost[gen_idx, col_n]) if col_n is not None else 3
                    else:
                        # Simplified format: assume quadratic (c2, c1) or higher
                        n = min(gencost.shape[1], 3)
                    
                    # Scale original economic cost coefficients by lambda_cost
                    if n >= 3 and col_c2 is not None:  # Quadratic: c2*Pg^2 + c1*Pg + c0
                        gencost[gen_idx, col_c2] *= lam_cost
                        gencost[gen_idx, col_c1] *= lam_cost
                        if col_c0 is not None and gencost.shape[1] > col_c0:
                            gencost[gen_idx, col_c0] *= lam_cost
                    elif n >= 2:  # Linear: c1*Pg + c0
                        gencost[gen_idx, col_c1] *= lam_cost
                        if col_c0 is not None and gencost.shape[1] > col_c0:
                            gencost[gen_idx, col_c0] *= lam_cost
                    
                    # Add carbon cost to linear coefficient (c1)
                    # Note: gencost coefficients are in MW units, so no baseMVA scaling needed
                    gencost[gen_idx, col_c1] += carbon_cost_per_MW[i]
            
            ppc["gencost"] = gencost
        
        # Clamp negative loads to 0 (negative values represent net generation, 
        # which OPF solver cannot handle as load)
        # This is necessary because bus_Pd/bus_Qd are identified from first sample only,
        # but subsequent samples may have negative values at these buses
        Pd_pu = np.maximum(Pd_pu, 0.0)
        Qd_pu = np.maximum(Qd_pu, 0.0)
        
        # Set loads (convert from p.u. to MW/MVAr)
        ppc["bus"][:, 2] = Pd_pu * self.baseMVA  # Active power demand (MW)
        ppc["bus"][:, 3] = Qd_pu * self.baseMVA  # Reactive power demand (MVAr)
        
        # Run OPF
        raw_result = self.runopf(ppc, self.ppopt)
        
        # Check if OPF converged
        success = bool(raw_result.get("success", 0))
        
        if not success:
            return {
                "success": False,
                "load_mode": load_mode,
                "raw_result": raw_result,
                "error": f"OPF did not converge. Check load limits and generator constraints."
            }
        
        # Parse and structure the results
        return self._parse_opf_result(raw_result, load_mode, lam_cost, lam_carbon)
    
    def _parse_opf_result(self, raw_result: Dict, load_mode: str, 
                         lam_cost: float = 1.0, lam_carbon: float = 0.0) -> Dict:
        """
        Parse PYPOWER OPF result into a structured, human-readable format.
        
        Args:
            raw_result: Raw result dictionary from PYPOWER runopf
            load_mode: Load mapping mode string
            lam_cost: Weight for economic cost (used for multi-objective)
            lam_carbon: Weight for carbon emission (used for multi-objective)
        
        Returns:
            Structured result dictionary
        """
        bus = raw_result["bus"]
        gen = raw_result["gen"]
        branch = raw_result["branch"]
        
        # --- Bus-level results ---
        # MATPOWER bus matrix columns:
        # 0: bus_id, 1: type, 2: Pd, 3: Qd, 7: Vm, 8: Va, 9: baseKV
        bus_result = {
            "bus_id": bus[:, 0].astype(int),  # External bus numbering
            "Vm": bus[:, 7],  # Voltage magnitude (p.u.)
            "Va_deg": bus[:, 8],  # Voltage phase angle (degrees)
            "Va_rad": np.deg2rad(bus[:, 8]),  # Voltage phase angle (radians)
            "Pd_MW": bus[:, 2],  # Active power demand (MW)
            "Qd_MVAr": bus[:, 3],  # Reactive power demand (MVAr)
        }
        
        # Aggregate generation at each bus (sum generators connected to same bus)
        bus_Pg = np.zeros(self.nbus)
        bus_Qg = np.zeros(self.nbus)
        for i in range(gen.shape[0]):
            gen_bus_idx = int(gen[i, 0]) - 1  # Convert to 0-based internal index
            # Find bus row index
            bus_row = np.where(bus[:, 0] == gen[i, 0])[0]
            if len(bus_row) > 0:
                bus_Pg[bus_row[0]] += gen[i, 1]  # Pg (MW)
                bus_Qg[bus_row[0]] += gen[i, 2]  # Qg (MVAr)
        
        bus_result["Pg_MW"] = bus_Pg
        bus_result["Qg_MVAr"] = bus_Qg
        
        # --- Generator-level results ---
        # MATPOWER gen matrix columns:
        # 0: bus_id, 1: Pg, 2: Qg, 3: Qmax, 4: Qmin, 8: Pmax, 9: Pmin
        gen_result = {
            "gen_id": np.arange(self.ngen, dtype=int),  # 0-based generator index
            "bus_id": gen[:, 0].astype(int),  # Bus ID where generator is connected
            "Pg_MW": gen[:, 1],  # Active power generation (MW)
            "Qg_MVAr": gen[:, 2],  # Reactive power generation (MVAr)
            "Pmax_MW": gen[:, 8],  # Maximum active power limit (MW)
            "Pmin_MW": gen[:, 9],  # Minimum active power limit (MW)
            "Qmax_MVAr": gen[:, 3],  # Maximum reactive power limit (MVAr)
            "Qmin_MVAr": gen[:, 4],  # Minimum reactive power limit (MVAr)
        }
        
        # [MOD] Add carbon emission per generator if multi-objective is enabled
        if self.use_multi_objective and self.gci_values is not None and len(self.gci_values) > 0:
            # Initialize carbon array for all generators
            carbon_per_gen = np.zeros(self.ngen)
            # Calculate carbon for active generators only
            for i, gen_idx in enumerate(self.idxPg):
                if gen_idx < self.ngen:
                    Pg_clamped = max(gen[gen_idx, 1], 0)  # Only positive Pg (MW)
                    carbon_per_gen[gen_idx] = self.gci_values[i] * Pg_clamped  # tCO2/h
            gen_result["carbon_emission_tCO2h"] = carbon_per_gen
            # Add GCI values for reference
            gci_full = np.zeros(self.ngen)
            for i, gen_idx in enumerate(self.idxPg):
                if gen_idx < self.ngen:
                    gci_full[gen_idx] = self.gci_values[i]
            gen_result["GCI_tCO2MWh"] = gci_full
        
        # --- Branch-level results ---
        # MATPOWER branch matrix columns:
        # 0: from_bus, 1: to_bus, 13: Pf, 14: Qf, 15: Pt, 16: Qt
        branch_result = {
            "from_bus": branch[:, 0].astype(int),
            "to_bus": branch[:, 1].astype(int),
            "Pf_MW": branch[:, 13],  # Active power flow from->to (MW)
            "Qf_MVAr": branch[:, 14],  # Reactive power flow from->to (MVAr)
            "Pt_MW": branch[:, 15],  # Active power flow to->from (MW)
            "Qt_MVAr": branch[:, 16],  # Reactive power flow to->from (MVAr)
        }
        
        # --- Summary statistics ---
        total_cost_raw = float(raw_result.get("f", 0.0))  # Objective function value (may include carbon if multi-objective)
        total_Pg = float(np.sum(gen[:, 1]))
        total_Pd = float(np.sum(bus[:, 2]))
        total_Qg = float(np.sum(gen[:, 2]))
        total_Qd = float(np.sum(bus[:, 3]))
        
        # [MOD] Calculate economic cost and carbon emission separately
        # If multi-objective, we need to recalculate economic cost using original gencost coefficients
        if self.use_multi_objective and self.gencost_original is not None:
            # Recalculate economic cost using original gencost (before modification)
            from main_part.utils import get_Pgcost
            # Only get active generators' power (get_Pgcost expects Pg to have shape [n_samples, n_active_gens])
            Pg_active_MW = gen[self.idxPg, 1]  # Active generators' power in MW
            Pg_active_pu = Pg_active_MW / self.baseMVA  # Convert MW to p.u.
            economic_cost = float(get_Pgcost(Pg_active_pu.reshape(1, -1), self.idxPg, self.gencost_original, self.baseMVA)[0])
        else:
            economic_cost = total_cost_raw
        
        # [MOD] Calculate carbon emission (tCO2/h)
        carbon_emission = 0.0
        carbon_emission_scaled = 0.0
        if self.use_multi_objective and self.gci_values is not None and len(self.gci_values) > 0:
            # Carbon = Σ GCI_i × Pg_i (MW) for active generators
            # Reference: deepopf_ngt_loss.py line 573: carbon_per = torch.sum(Pg_clamped * gci_tensor.unsqueeze(0), dim=1)
            Pg_clamped = np.maximum(gen[self.idxPg, 1], 0)  # Only positive Pg (MW)
            carbon_emission = float(np.sum(self.gci_values * Pg_clamped))  # tCO2/h
            carbon_emission_scaled = carbon_emission * self.carbon_scale
        
        # [MOD] Calculate multi-objective weighted cost
        if self.use_multi_objective and lam_carbon > 0:
            total_cost_weighted = lam_cost * economic_cost + lam_carbon * carbon_emission_scaled
        else:
            total_cost_weighted = economic_cost
        
        summary = {
            "total_cost": total_cost_weighted,  # Weighted objective (cost + carbon)
            "economic_cost": economic_cost,  # Pure economic cost
            "carbon_emission": carbon_emission,  # Carbon emission (tCO2/h, unscaled)
            "carbon_emission_scaled": carbon_emission_scaled,  # Carbon emission scaled
            "total_Pg_MW": total_Pg,
            "total_Pd_MW": total_Pd,
            "total_Qg_MVAr": total_Qg,
            "total_Qd_MVAr": total_Qd,
            # [MOD] Multi-objective information
            "lambda_cost": lam_cost,
            "lambda_carbon": lam_carbon,
        }
        
        return {
            "success": True,
            "load_mode": load_mode,
            "raw_result": raw_result,
            "bus": bus_result,
            "gen": gen_result,
            "branch": branch_result,
            "summary": summary,
        }


# -----------------------
# 1) Robust MATPOWER .m parser (only for "mpc.xxx = [ ... ];" blocks)
# -----------------------

_FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def _extract_block(text: str, name: str) -> str:
    m = re.search(rf"mpc\.{re.escape(name)}\s*=\s*\[", text)
    if not m:
        raise KeyError(f"Cannot find 'mpc.{name} = [' block in .m file")
    start = m.end()
    end = text.find("];", start)
    if end < 0:
        raise ValueError(f"Cannot find closing '];' for mpc.{name} block")
    return text[start:end]


def _parse_matrix(block: str) -> np.ndarray:
    # strip comments and continuation
    lines = []
    for line in block.splitlines():
        line = line.split("%", 1)[0]
        line = line.replace("...", " ")
        if line.strip():
            lines.append(line)
    joined = "\n".join(lines)
    rows = [r.strip() for r in joined.split(";") if r.strip()]
    data = []
    maxlen = 0
    for r in rows:
        nums = [float(x) for x in re.findall(_FLOAT_RE, r)]
        data.append(nums)
        maxlen = max(maxlen, len(nums))
    arr = np.array([row + [np.nan] * (maxlen - len(row)) for row in data], dtype=float)
    return arr


def load_case_from_m(case_m_path: str) -> dict:
    with open(case_m_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9eE\.\+\-]+)", txt)
    if not m:
        raise KeyError("Cannot find mpc.baseMVA in .m file")
    baseMVA = float(m.group(1))

    bus = _parse_matrix(_extract_block(txt, "bus"))
    gen = _parse_matrix(_extract_block(txt, "gen"))
    branch = _parse_matrix(_extract_block(txt, "branch"))
    gencost = _parse_matrix(_extract_block(txt, "gencost"))

    ppc = {
        "version": "2",
        "baseMVA": baseMVA,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
    }
    return ppc


def get_base_load_info(case_m_path: str, verbose: bool = False) -> Dict:
    """
    Extract base load information from a MATPOWER case file.
    
    This function parses the case file and extracts comprehensive load and 
    generation capacity information useful for load scenario generation.
    
    Args:
        case_m_path: Path to the MATPOWER .m case file
        verbose: If True, print detailed information
    
    Returns:
        Dictionary containing:
        {
            # System parameters
            "baseMVA": float,           # System base MVA (typically 100)
            "nbus": int,                # Total number of buses
            "ngen": int,                # Total number of generators
            "nbranch": int,             # Total number of branches
            
            # Base load (MW/MVAr) - per bus
            "Pd_base_MW": np.ndarray,   # Base active power demand per bus (MW)
            "Qd_base_MVAr": np.ndarray, # Base reactive power demand per bus (MVAr)
            
            # Base load (p.u.) - per bus
            "Pd_base_pu": np.ndarray,   # Base active power demand per bus (p.u.)
            "Qd_base_pu": np.ndarray,   # Base reactive power demand per bus (p.u.)
            
            # Aggregate statistics
            "total_Pd_MW": float,       # Total system active load (MW)
            "total_Qd_MVAr": float,     # Total system reactive load (MVAr)
            "total_Pd_pu": float,       # Total system active load (p.u.)
            "total_Qd_pu": float,       # Total system reactive load (p.u.)
            
            # Generation capacity
            "total_Pmax_MW": float,     # Total maximum generation capacity (MW)
            "total_Pmin_MW": float,     # Total minimum generation (MW)
            "total_Qmax_MVAr": float,   # Total maximum reactive generation (MVAr)
            "total_Qmin_MVAr": float,   # Total minimum reactive generation (MVAr)
            "Pmax_per_gen_MW": np.ndarray,  # Pmax for each generator (MW)
            "Pmin_per_gen_MW": np.ndarray,  # Pmin for each generator (MW)
            
            # Bus indices
            "idx_load_bus": np.ndarray, # Indices of buses with Pd > 0 (0-based)
            "idx_gen_bus": np.ndarray,  # Indices of buses with generators (0-based)
            "idx_slack_bus": int,       # Index of slack/reference bus (0-based)
            "bus_ids": np.ndarray,      # External bus IDs
            
            # Load ratio information
            "base_load_ratio": float,   # Base load / Pmax (typically 0.6~0.8)
            
            # Raw ppc for advanced usage
            "ppc": dict,                # Original PYPOWER case dictionary
        }
    
    Example:
        >>> info = get_base_load_info("case300_ieee_modified.m")
        >>> print(f"Base load: {info['total_Pd_MW']:.2f} MW")
        >>> print(f"Load ratio: {info['base_load_ratio']*100:.1f}%")
        >>> # Generate load at 90% of base
        >>> Pd_scaled = info['Pd_base_pu'] * 0.9
    """
    # Load the case file
    ppc = load_case_from_m(case_m_path)
    
    baseMVA = float(ppc["baseMVA"])
    bus = ppc["bus"]
    gen = ppc["gen"]
    branch = ppc["branch"]
    
    nbus = int(bus.shape[0])
    ngen = int(gen.shape[0])
    nbranch = int(branch.shape[0])
    
    # --- Extract base load (MATPOWER bus columns: 2=Pd, 3=Qd) ---
    Pd_base_MW = bus[:, 2].astype(float)
    Qd_base_MVAr = bus[:, 3].astype(float)
    
    # Convert to per-unit
    Pd_base_pu = Pd_base_MW / baseMVA
    Qd_base_pu = Qd_base_MVAr / baseMVA
    
    # Aggregate totals
    total_Pd_MW = float(np.sum(Pd_base_MW))
    total_Qd_MVAr = float(np.sum(Qd_base_MVAr))
    total_Pd_pu = float(np.sum(Pd_base_pu))
    total_Qd_pu = float(np.sum(Qd_base_pu))
    
    # --- Extract generation capacity (MATPOWER gen columns: 8=Pmax, 9=Pmin, 3=Qmax, 4=Qmin) ---
    Pmax_per_gen_MW = gen[:, 8].astype(float)
    Pmin_per_gen_MW = gen[:, 9].astype(float)
    Qmax_per_gen_MVAr = gen[:, 3].astype(float)
    Qmin_per_gen_MVAr = gen[:, 4].astype(float)
    
    total_Pmax_MW = float(np.sum(Pmax_per_gen_MW))
    total_Pmin_MW = float(np.sum(Pmin_per_gen_MW))
    total_Qmax_MVAr = float(np.sum(Qmax_per_gen_MVAr))
    total_Qmin_MVAr = float(np.sum(Qmin_per_gen_MVAr))
    
    # --- Bus indices ---
    # Load buses: Pd > 0
    idx_load_bus = np.where(Pd_base_MW > 0)[0]
    
    # Generator buses (from gen matrix, column 0 is bus ID)
    gen_bus_ids = gen[:, 0].astype(int)
    bus_ids = bus[:, 0].astype(int)
    # Map generator bus IDs to row indices
    idx_gen_bus = np.array([np.where(bus_ids == gid)[0][0] for gid in gen_bus_ids 
                           if len(np.where(bus_ids == gid)[0]) > 0], dtype=int)
    idx_gen_bus = np.unique(idx_gen_bus)
    
    # Slack bus (type == 3)
    idx_slack_bus = int(np.where(bus[:, 1].astype(int) == 3)[0][0])
    
    # --- Load ratio ---
    base_load_ratio = total_Pd_MW / total_Pmax_MW if total_Pmax_MW > 0 else 0.0
    
    # --- Build result dictionary ---
    result = {
        # System parameters
        "baseMVA": baseMVA,
        "nbus": nbus,
        "ngen": ngen,
        "nbranch": nbranch,
        
        # Base load (MW/MVAr)
        "Pd_base_MW": Pd_base_MW,
        "Qd_base_MVAr": Qd_base_MVAr,
        
        # Base load (p.u.)
        "Pd_base_pu": Pd_base_pu,
        "Qd_base_pu": Qd_base_pu,
        
        # Aggregate statistics
        "total_Pd_MW": total_Pd_MW,
        "total_Qd_MVAr": total_Qd_MVAr,
        "total_Pd_pu": total_Pd_pu,
        "total_Qd_pu": total_Qd_pu,
        
        # Generation capacity
        "total_Pmax_MW": total_Pmax_MW,
        "total_Pmin_MW": total_Pmin_MW,
        "total_Qmax_MVAr": total_Qmax_MVAr,
        "total_Qmin_MVAr": total_Qmin_MVAr,
        "Pmax_per_gen_MW": Pmax_per_gen_MW,
        "Pmin_per_gen_MW": Pmin_per_gen_MW,
        
        # Bus indices
        "idx_load_bus": idx_load_bus,
        "idx_gen_bus": idx_gen_bus,
        "idx_slack_bus": idx_slack_bus,
        "bus_ids": bus_ids,
        
        # Load ratio
        "base_load_ratio": base_load_ratio,
        
        # Raw ppc
        "ppc": ppc,
    }
    
    if verbose:
        print("=" * 60)
        print("Base Load Information")
        print("=" * 60)
        print(f"System: baseMVA={baseMVA}, nbus={nbus}, ngen={ngen}, nbranch={nbranch}")
        print()
        print("--- Base Load ---")
        print(f"  Total Pd: {total_Pd_MW:.2f} MW = {total_Pd_pu:.4f} p.u.")
        print(f"  Total Qd: {total_Qd_MVAr:.2f} MVAr = {total_Qd_pu:.4f} p.u.")
        print(f"  Load buses (Pd>0): {len(idx_load_bus)}")
        print(f"  Pd range: [{Pd_base_MW.min():.2f}, {Pd_base_MW.max():.2f}] MW")
        print()
        print("--- Generation Capacity ---")
        print(f"  Total Pmax: {total_Pmax_MW:.2f} MW")
        print(f"  Total Pmin: {total_Pmin_MW:.2f} MW")
        print(f"  Total Qmax: {total_Qmax_MVAr:.2f} MVAr")
        print(f"  Total Qmin: {total_Qmin_MVAr:.2f} MVAr")
        print()
        print("--- Load Ratio ---")
        print(f"  Base load / Pmax: {base_load_ratio*100:.1f}%")
        print(f"  Slack bus: row={idx_slack_bus}, bus_id={bus_ids[idx_slack_bus]}")
        print("=" * 60)
    
    return result


def generate_scaled_load(base_info: Dict, 
                         delta: float = 0.1,
                         global_scale: float = 1.0,
                         seed: Optional[int] = None,
                         n_samples: int = 1) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                       Tuple[np.ndarray, np.ndarray]]:
    """
    Generate load scenarios using the classic per-bus random scaling method.
    
    Classic Method:
        For each load bus i, independently sample a scaling factor k_i ~ U(1-Δ, 1+Δ),
        then apply: Pd_i = Pd_base_i × k_i, Qd_i = Qd_base_i × k_i
        
        This keeps the power factor constant at each bus (Pd and Qd scale together).
    
    Args:
        base_info: Dictionary from get_base_load_info()
        delta: Per-bus variation range, default 0.1 means ±10%
               Each bus samples k_i ~ U(1-delta, 1+delta)
        global_scale: Optional global scaling factor applied after per-bus variation.
                     Useful for generating scenarios at different load levels.
                     Final load = Pd_base × k_i × global_scale
        seed: Random seed for reproducibility
        n_samples: Number of samples to generate (default: 1)
                  If n_samples=1, returns (Pd_pu, Qd_pu) each of shape (nbus,)
                  If n_samples>1, returns (Pd_pu, Qd_pu) each of shape (n_samples, nbus)
    
    Returns:
        Tuple of (Pd_pu, Qd_pu) arrays in per-unit.
        Shape is (nbus,) if n_samples=1, else (n_samples, nbus).
    
    Example:
        >>> base_info = get_base_load_info("case300.m")
        >>> 
        >>> # Single sample with ±10% per-bus variation
        >>> Pd, Qd = generate_scaled_load(base_info, delta=0.1)
        >>> 
        >>> # Single sample at 90% load level with ±10% variation
        >>> Pd, Qd = generate_scaled_load(base_info, delta=0.1, global_scale=0.9)
        >>> 
        >>> # Generate 100 samples with ±15% variation
        >>> Pd_batch, Qd_batch = generate_scaled_load(base_info, delta=0.15, n_samples=100, seed=42)
        >>> print(Pd_batch.shape)  # (100, 300)
    
    Note:
        - Power factor is preserved: Qd_i / Pd_i = Qd_base_i / Pd_base_i
        - Buses with zero base load remain zero
        - The same k_i is applied to both Pd and Qd at each bus
    """
    rng = np.random.default_rng(seed)
    
    # Get base load
    Pd_base_pu = base_info["Pd_base_pu"]  # shape: (nbus,)
    Qd_base_pu = base_info["Qd_base_pu"]  # shape: (nbus,)
    nbus = len(Pd_base_pu)
    
    if n_samples == 1:
        # Single sample: shape (nbus,)
        # Sample per-bus scaling factors k_i ~ U(1-delta, 1+delta)
        k = rng.uniform(1.0 - delta, 1.0 + delta, size=nbus)
        
        # Apply scaling: Pd_i = Pd_base_i × k_i × global_scale
        # Same k_i for both Pd and Qd to preserve power factor
        Pd_pu = Pd_base_pu * k * global_scale
        Qd_pu = Qd_base_pu * k * global_scale
        
        return Pd_pu, Qd_pu
    else:
        # Multiple samples: shape (n_samples, nbus)
        # Sample per-bus scaling factors for all samples at once
        k = rng.uniform(1.0 - delta, 1.0 + delta, size=(n_samples, nbus))
        
        # Apply scaling with broadcasting
        # Pd_base_pu: (nbus,) -> broadcast with k: (n_samples, nbus)
        Pd_pu = Pd_base_pu[np.newaxis, :] * k * global_scale  # (n_samples, nbus)
        Qd_pu = Qd_base_pu[np.newaxis, :] * k * global_scale  # (n_samples, nbus)
        
        return Pd_pu, Qd_pu


def generate_load_batch(base_info: Dict,
                        n_samples: int,
                        delta: float = 0.1,
                        global_scale_range: Optional[Tuple[float, float]] = None,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a batch of load scenarios with optional global scale variation.
    
    This is a convenience function that combines per-bus variation with 
    optional system-level load variation for generating diverse training data.
    
    Method:
        1. For each sample, optionally sample a global scale factor from global_scale_range
        2. For each bus, sample k_i ~ U(1-delta, 1+delta)
        3. Final load: Pd_i = Pd_base_i × k_i × global_scale
    
    Args:
        base_info: Dictionary from get_base_load_info()
        n_samples: Number of samples to generate
        delta: Per-bus variation range (default: 0.1 = ±10%)
        global_scale_range: Optional (min, max) for system-level load variation.
                           If None, global_scale=1.0 for all samples.
                           If provided, each sample gets a random global scale.
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (Pd_pu, Qd_pu, global_scales):
            - Pd_pu: shape (n_samples, nbus) - Active power in p.u.
            - Qd_pu: shape (n_samples, nbus) - Reactive power in p.u.
            - global_scales: shape (n_samples,) - Global scale factor used for each sample
    
    Example:
        >>> base_info = get_base_load_info("case300.m")
        >>> 
        >>> # Generate 1000 samples with ±10% per-bus, system load 80%-100%
        >>> Pd, Qd, scales = generate_load_batch(
        ...     base_info, n_samples=1000, delta=0.1,
        ...     global_scale_range=(0.8, 1.0), seed=42
        ... )
        >>> print(f"Total Pd range: [{Pd.sum(axis=1).min():.2f}, {Pd.sum(axis=1).max():.2f}] p.u.")
    """
    rng = np.random.default_rng(seed)
    
    Pd_base_pu = base_info["Pd_base_pu"]
    Qd_base_pu = base_info["Qd_base_pu"]
    nbus = len(Pd_base_pu)
    
    # Determine global scales
    if global_scale_range is not None:
        global_scales = rng.uniform(global_scale_range[0], global_scale_range[1], size=n_samples)
    else:
        global_scales = np.ones(n_samples)
    
    # Sample per-bus scaling factors: k ~ U(1-delta, 1+delta)
    k = rng.uniform(1.0 - delta, 1.0 + delta, size=(n_samples, nbus))
    
    # Apply both per-bus and global scaling
    # global_scales: (n_samples,) -> (n_samples, 1) for broadcasting
    Pd_pu = Pd_base_pu[np.newaxis, :] * k * global_scales[:, np.newaxis]
    Qd_pu = Qd_base_pu[np.newaxis, :] * k * global_scales[:, np.newaxis]
    
    return Pd_pu, Qd_pu, global_scales


# -----------------------
# 2) PYPOWER import (with pandapower fallback)
# -----------------------

def _safe_import_pypower():
    try:
        from pypower.api import runopf
        from pypower.ppoption import ppoption
        return runopf, ppoption
    except Exception:
        from pandapower.pypower.api import runopf
        from pandapower.pypower.ppoption import ppoption
        return runopf, ppoption


# -----------------------
# 3) Load mapping helpers
# -----------------------

def _infer_slack_bus_row(ppc: dict) -> int:
    # MATPOWER bus types: 3 = REF
    ref_rows = np.where(ppc["bus"][:, 1].astype(int) == 3)[0]
    if len(ref_rows) != 1:
        raise RuntimeError(f"Cannot infer slack (REF) bus uniquely, got {len(ref_rows)} REF buses")
    return int(ref_rows[0])


def _infer_load_mapping(x_row_pu: np.ndarray, nbus: int, slack_row: int, ngt_data: dict):
    """
    Map a single sample load vector x_row_pu to full-bus Pd/Qd arrays in p.u.
    Priority:
      A) x_dim == 2*nbus        -> [Pd_all, Qd_all]
      B) x_dim == 2*(nbus-1)    -> no slack -> insert slack at slack_row
      C) if ngt_data provides explicit Pd/Qd bus index lists (project-specific keys), use them
    """
    x_row_pu = np.asarray(x_row_pu, dtype=float).reshape(-1)
    dim = int(x_row_pu.shape[0])

    if dim == 2 * nbus:
        Pd = x_row_pu[:nbus]
        Qd = x_row_pu[nbus:]
        return Pd, Qd, "A:2*nbus"

    if dim == 2 * (nbus - 1):
        Pd_full = np.zeros(nbus, dtype=float)
        Qd_full = np.zeros(nbus, dtype=float)
        Pd_part = x_row_pu[: nbus - 1]
        Qd_part = x_row_pu[nbus - 1:]
        mask = np.ones(nbus, dtype=bool)
        mask[slack_row] = False
        Pd_full[mask] = Pd_part
        Qd_full[mask] = Qd_part
        return Pd_full, Qd_full, "B:2*(nbus-1)"

    # Project-specific: if you have idx lists inside ngt_data, add here
    for kP, kQ in [
        ("idx_bus_Pd", "idx_bus_Qd"),
        ("bus_Pd", "bus_Qd"),
        ("pd_buses", "qd_buses"),
    ]:
        if kP in ngt_data and kQ in ngt_data:
            busesP = np.array(ngt_data[kP]).astype(int).reshape(-1)
            busesQ = np.array(ngt_data[kQ]).astype(int).reshape(-1)
            if dim != (len(busesP) + len(busesQ)):
                continue
            Pd_full = np.zeros(nbus, dtype=float)
            Qd_full = np.zeros(nbus, dtype=float)
            Pd_full[busesP] = x_row_pu[: len(busesP)]
            Qd_full[busesQ] = x_row_pu[len(busesP):]
            return Pd_full, Qd_full, f"C:{kP}/{kQ}"

    raise ValueError(
        f"Cannot map x dim={dim} to Pd/Qd with nbus={nbus}. "
        f"Please edit _infer_load_mapping() for your project."
    )


def _infer_label_angle_unit(y_va_part: np.ndarray) -> str:
    # heuristic: <= ~3.5 => rad, else deg
    m = float(np.max(np.abs(y_va_part)))
    return "rad" if m <= 3.5 else "deg"


# -----------------------
# 4) Main: run OPF for training samples and compare with labels
# -----------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()    
    parser.add_argument("--case_m", type=str, default='main_part/data/case300_ieee_modified.m', help="Path to case300_ieee_modified.m")
    parser.add_argument("--nsamples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0) 
    args = parser.parse_args()

    # Your project loaders
    import sys
    import os
    # 将项目根目录导入 sys.path，确保主目录包可以被正确导入
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from main_part.train_unsupervised import get_unsupervised_config
    from main_part.data_loader import load_all_data, load_ngt_training_data

    config = get_unsupervised_config()
    sys_data, _, _ = load_all_data(config)
    ngt_data, sys_data = load_ngt_training_data(config, sys_data=sys_data)

    x_train = ngt_data["x_train"].detach().cpu().numpy()
    y_train = ngt_data["y_train"].detach().cpu().numpy() 
    # Use the new PyPowerOPFSolver class
    print("=" * 60)
    print("Using PyPowerOPFSolver class")
    print("=" * 60)
    
    # [MOD] Initialize solver with multi-objective support
    solver = PyPowerOPFSolver(
        case_m_path=args.case_m,
        ngt_data=ngt_data,
        verbose=True,
        use_multi_objective=False,  # Enable multi-objective optimization
        lambda_cost=1,
        lambda_carbon=0,
        carbon_scale=1.0,
        sys_data=sys_data  # Required for GCI calculation
    )
    
    # Sample indices
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(x_train.shape[0], size=min(args.nsamples, x_train.shape[0]), replace=False)
    
    mae_vm_list, mae_va_list = [], []
    
    for si, idx in enumerate(idxs, start=1):
        x = x_train[idx]
        y = y_train[idx]
        
        # Solve OPF using the class
        result = solver.forward(x)
        
        if not result["success"]:
            print(f"[{si}/{len(idxs)}] sample={idx} OPF FAILED (load_mode={result['load_mode']})")
            if "error" in result:
                print(f"  Error: {result['error']}")
            continue
        
        # Extract results
        Vm = result["bus"]["Vm"]
        Va_deg = result["bus"]["Va_deg"]
        Va_rad = result["bus"]["Va_rad"]
        
        # Get bus indices for comparison (same logic as before)
        nbus = solver.nbus
        slack_row = solver.slack_row
        ydim = int(y_train.shape[1])
        n_nonzib = int((ydim + 1) // 2)
        
        bus_pred0 = None
        for key in ["bus_Pnet_all", "bus_Pnet", "idx_bus_Pnet_all", "idx_bus_Pnet"]:
            if key in ngt_data:
                bus_pred0 = np.array(ngt_data[key]).astype(int).reshape(-1)
                break
        if bus_pred0 is None:
            bus_pred0 = np.arange(nbus, dtype=int)
        if bus_pred0.min() >= 1 and bus_pred0.max() <= nbus:
            bus_pred0 = bus_pred0 - 1
        if len(bus_pred0) != n_nonzib:
            n_use = min(len(bus_pred0), n_nonzib)
            bus_pred0 = bus_pred0[:n_use]
            n_nonzib = n_use
        
        bus_pred_noslack0 = bus_pred0[bus_pred0 != slack_row]
        
        # Extract OPF values in NGT bus subset
        Vm_sub = Vm[bus_pred0]
        Va_sub_noslack_deg = Va_deg[bus_pred_noslack0]
        Va_sub_noslack_rad = Va_rad[bus_pred_noslack0]
        
        # Split label y: [Va_nonZIB_noslack, Vm_nonZIB]
        y_va = y[: len(bus_pred_noslack0)]
        y_vm = y[len(bus_pred_noslack0): len(bus_pred_noslack0) + len(bus_pred0)]
        
        unit = _infer_label_angle_unit(y_va)
        Va_sub = Va_sub_noslack_rad if unit == "rad" else np.deg2rad(Va_sub_noslack_deg)
        
        e_vm = Vm_sub - y_vm
        e_va = Va_sub - y_va
        
        mae_vm = float(np.mean(np.abs(e_vm)))
        mae_va = float(np.mean(np.abs(e_va)))
        mae_vm_list.append(mae_vm)
        mae_va_list.append(mae_va)
        
        # Print detailed results
        print(f"\n[{si}/{len(idxs)}] sample={idx} success | load_mode={result['load_mode']} | Va_unit={unit}")
        print(f"  Vm_MAE={mae_vm:.4e}, Va_MAE={mae_va:.4e}")
        # [MOD] Print multi-objective results
        if solver.use_multi_objective:
            print(f"  Summary: Weighted cost={result['summary']['total_cost']:.2f}, "
                  f"Economic cost={result['summary']['economic_cost']:.2f}, "
                  f"Carbon={result['summary']['carbon_emission']:.4f} tCO2/h")
            print(f"  Preference: λ_cost={result['summary']['lambda_cost']:.2f}, "
                  f"λ_carbon={result['summary']['lambda_carbon']:.2f}")
        else:
            print(f"  Summary: Total cost={result['summary']['total_cost']:.2f}, "
                  f"Total Pg={result['summary']['total_Pg_MW']:.2f} MW, "
                  f"Total Pd={result['summary']['total_Pd_MW']:.2f} MW")
        print(f"  Generator outputs (first 5): {result['gen']['Pg_MW'][:5]}")
    
    if mae_vm_list:
        print("\n=== Summary over successful samples ===")
        print(f"Vm MAE mean={np.mean(mae_vm_list):.4e}, max={np.max(mae_vm_list):.4e}")
        print(f"Va MAE mean={np.mean(mae_va_list):.4e}, max={np.max(mae_va_list):.4e}")
    else:
        print("\nNo successful OPF runs. Check load mapping and generator limits.")
    return


if __name__ == "__main__":
    main()
