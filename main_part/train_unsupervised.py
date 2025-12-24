#!/usr/bin/env python
# coding: utf-8
# Training Script for DeepOPF-V
# Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.
# Author: Peng Yue
# Date: December 15th, 2025

import torch 
import torch.utils.data as Data
import numpy as np 
import time
import os
import sys 
import math

# Add parent directory to path for flow_model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from models import NetV
from data_loader import load_all_data, load_ngt_training_data 
from deepopf_ngt_loss import DeepOPFNGTLoss 
from unified_eval import post_process_like_evaluate_model, EvalContext, _ensure_1d_int, _as_numpy

 
# ============================================================================
# Helper functions for gradient descent training
# ============================================================================

def _compute_constraint_violation(loss_dict):
    """Compute total constraint violation from loss dictionary."""
    return (
        loss_dict.get('loss_Pgi_sum', 0.0) +
        loss_dict.get('loss_Qgi_sum', 0.0) +
        loss_dict.get('loss_Pdi_sum', 0.0) +
        loss_dict.get('loss_Qdi_sum', 0.0) +
        loss_dict.get('loss_Vi_sum', 0.0)
    )


def _compute_weighted_objective(loss_dict, lambda_cost, lambda_carbon):
    """Compute weighted objective function: lambda_cost * cost + lambda_carbon * carbon."""
    return (
        lambda_cost * loss_dict.get('loss_cost', 0.0) +
        lambda_carbon * loss_dict.get('loss_carbon', 0.0)
    )


def _clip_gradient(grad_V, grad_clip_norm):
    """Clip gradient per sample to specified norm."""
    if grad_clip_norm is None or grad_clip_norm <= 0:
        return grad_V
    grad_norm_per_sample = torch.norm(grad_V, dim=1, keepdim=True)
    clip_coef = grad_clip_norm / (grad_norm_per_sample + 1e-8)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    return grad_V * clip_coef

from models import create_model

def train_unsupervised_ngt_gradient_descent(
    config, sys_data, device=None, 
    num_iterations=50,
    learning_rate=1e-5,  # Default to much smaller learning rate
    tb_logger=None, 
    grad_clip_norm=1.0,  # Gradient clipping norm
    use_post_processing=True,  # Enable post-processing after each gradient update
):
    """
    Validation function: Direct gradient descent on V_anchor.
    
    This function tests whether we can improve V_anchor by:
    1. Starting from VAE-generated V_anchor (physical space)
    2. Computing loss gradient w.r.t. V_anchor (ONLY OBJECTIVE, NO CONSTRAINTS - only_obj=True) 
    4. Updating V_anchor directly using projected gradient
    5. Iterating to find better solutions (optimizing objective while maintaining constraints via post-processing)
    
    NOTE: This function uses only_obj=True, meaning it only optimizes the objective function
    (cost + carbon) and ignores constraint violations in the loss. Constraints are maintained
    through post-processing after each gradient update.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object (optional, will load if None)
        device: Device (optional, uses config.device if None) 
        num_iterations: Number of gradient descent iterations
        learning_rate: Learning rate for gradient updates (DEFAULT: 1e-5, recommended range: 1e-5 to 1e-6)
        tb_logger: TensorBoardLogger instance for logging (optional) 
        lambda_cor: Drift correction gain (default: 5.0)
        grad_clip_norm: Gradient clipping norm (default: 1.0) 
    """
    
    if device is None:
        device = config.device
    
    # ========================================================================
    # Initialization
    # ========================================================================
    # Load system data and compute BRANFT (needed for post-processing) 
    branch_np = sys_data.branch if isinstance(sys_data.branch, np.ndarray) else sys_data.branch.numpy()
    BRANFT = branch_np[:, 0:2] - 1  # Convert to 0-indexed
    # Load NGT training data
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)
    Vscale = ngt_data['Vscale'].to(device)
    Vbias = ngt_data['Vbias'].to(device)
    
    bus_slack = int(sys_data.bus_slack)
    
    # Load and freeze VAE models for anchor generation
    def _load_frozen_vae_or_raise():
        """Load and freeze VAE models for generating voltage anchors."""
        vae_vm_path = config.pretrain_model_path_vm
        vae_va_path = config.pretrain_model_path_va
        input_dim = ngt_data['input_dim']
        
        vae_vm = create_model('vae', input_dim, config.Nbus, config, is_vm=True).to(device)
        vae_va = create_model('vae', input_dim, config.Nbus - 1, config, is_vm=False).to(device)
        vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=True)
        vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=True)
        
        vae_vm.eval()
        vae_va.eval()
        for p in vae_vm.parameters():
            p.requires_grad = False
        for p in vae_va.parameters():
            p.requires_grad = False
        return vae_vm, vae_va
    
    vae_vm, vae_va = _load_frozen_vae_or_raise()
    
    # Setup loss function 
    loss_fn = DeepOPFNGTLoss(sys_data, config)
    loss_fn.cache_to_gpu(device) 
    
    # Create training DataLoader with indices
    class IndexedTensorDataset(Data.Dataset):
        """Dataset that returns data with indices for batch tracking."""
        def __init__(self, *tensors):
            assert all(t.size(0) == tensors[0].size(0) for t in tensors)
            self.tensors = tensors
        
        def __getitem__(self, index):
            return tuple(t[index] for t in self.tensors) + (index,)
        
        def __len__(self):
            return self.tensors[0].size(0)
    
    indexed_dataset = IndexedTensorDataset(ngt_data['x_train'], ngt_data['y_train'])
    training_loader = Data.DataLoader(indexed_dataset, batch_size=config.ngt_batch_size, shuffle=False)
    
    # Precompute initial V_anchor (physical space) for all training samples
    print("\n[Gradient Descent] Computing V_anchor from VAE...")
    V_anchor_all = _precompute_V_anchor_physical_all(
        config=config,
        sys_data=sys_data,
        ngt_data=ngt_data,
        device=device,
        vae_vm=vae_vm,
        vae_va=vae_va,
        Vscale=Vscale,
        Vbias=Vbias,
        bus_slack=bus_slack,
    )
    
    # Initialize loss history tracking
    loss_history = {
        'total': [],
        'cost': [],
        'carbon': [],
        'constraint_violation': [],
    }
    
    # ========================================================================
    # Gradient Descent Training Loop
    # ========================================================================
    
    start_time = time.time()
    
    for epoch in range(num_iterations):
        epoch_losses = []
        epoch_costs = []
        epoch_carbons = []
        epoch_constraint_violations = []
        epoch_weighted_objectives = []  # Track weighted objective function
        
        for batch_idx, (train_x, train_y, batch_indices) in enumerate(training_loader):
            train_x = train_x.to(device) 
            
            # Get current V_anchor for this batch and enable gradients
            V_anchor_batch = V_anchor_all[batch_indices].clone().requires_grad_(True) 
            
            # Compute loss and gradient 
            loss, loss_dict = loss_fn(V_anchor_batch, train_x, only_obj=False) 
            
            grad_V = torch.autograd.grad(
                outputs=loss,
                inputs=V_anchor_batch,
                create_graph=False,
                retain_graph=False,
            )[0]
            
            # Clip gradient to prevent large updates
            grad_V = _clip_gradient(grad_V, grad_clip_norm)
            
            # Update V_anchor using gradient descent with optional post-processing 
            with torch.no_grad():
                # Gradient descent update
                V_anchor_batch_new = V_anchor_batch - learning_rate * grad_V
                
                # Clamp to valid voltage range
                min_v = Vbias - 2 * Vscale
                max_v = Vbias + 2 * Vscale
                V_anchor_batch_new = torch.clamp(V_anchor_batch_new, min=min_v, max=max_v)
                
                # Apply post-processing if enabled
                if use_post_processing:
                    try:
                        V_anchor_batch_new = _apply_post_processing_to_batch(
                            V_anchor_batch_new.detach().cpu().numpy(),
                            train_x,
                            ngt_data,
                            sys_data,
                            config,
                            device,
                            BRANFT,
                            verbose=(epoch == 0 and batch_idx == 0)
                        )
                        # Ensure voltage remains in valid range after post-processing
                        V_anchor_batch_new = torch.clamp(V_anchor_batch_new, min=min_v, max=max_v)
                    except Exception as e:
                        if epoch == 0 and batch_idx == 0:
                            print(f"[Warning] Post-processing failed: {e}, using gradient-updated voltage") 
            
            # Update stored V_anchor
            V_anchor_all[batch_indices] = V_anchor_batch_new
            
            # Record metrics
            epoch_losses.append(loss.item())
            epoch_costs.append(loss_dict.get('loss_cost', 0.0))
            epoch_carbons.append(loss_dict.get('loss_carbon', 0.0))
            lambda_cost = config.ngt_lambda_cost
            lambda_carbon = 1 - lambda_cost
            epoch_weighted_objectives.append(_compute_weighted_objective(loss_dict, lambda_cost, lambda_carbon))
            epoch_constraint_violations.append(_compute_constraint_violation(loss_dict))
        
        # Compute average metrics for this epoch
        n_batches = len(epoch_losses)
        if n_batches > 0:
            avg_loss = sum(epoch_losses) / n_batches
            avg_cost = sum(epoch_costs) / n_batches
            avg_carbon = sum(epoch_carbons) / n_batches
            avg_constraint_violation = sum(epoch_constraint_violations) / n_batches
            avg_weighted_obj = sum(epoch_weighted_objectives) / n_batches
        else:
            avg_loss = avg_cost = avg_carbon = avg_constraint_violation = avg_weighted_obj = 0.0
        
        # Update loss history
        loss_history['total'].append(avg_loss)
        loss_history['cost'].append(avg_cost)
        loss_history['carbon'].append(avg_carbon)
        loss_history['constraint_violation'].append(avg_constraint_violation)
        
        # Print progress periodically
        print_interval = max(1, num_iterations // 500)
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Iteration {epoch+1}/{num_iterations}: "
                  f"loss={avg_loss:.4f}, cost={avg_cost:.2f}, carbon={avg_carbon:.4f}, "
                  f"weighted_obj={avg_weighted_obj:.2f}, constraint_vio={avg_constraint_violation:.4f}, "
                  f"lambda_cost={lambda_cost:.2f}, lambda_carbon={lambda_carbon:.2f}")
        
        # TensorBoard logging
        if tb_logger:
            tb_logger.log_scalar('gradient_descent/loss', avg_loss, epoch)
            tb_logger.log_scalar('gradient_descent/cost', avg_cost, epoch)
            tb_logger.log_scalar('gradient_descent/carbon', avg_carbon, epoch)
            tb_logger.log_scalar('gradient_descent/constraint_violation', avg_constraint_violation, epoch)
    
    elapsed_time = time.time() - start_time
    print(f"\n[Gradient Descent] Completed in {elapsed_time:.2f}s")   


def _precompute_V_anchor_physical_all(
    *,
    config, sys_data, ngt_data, device,
    vae_vm, vae_va,
    Vscale, Vbias,
    bus_slack,
):
    """
    Precompute V_anchor in physical space (NGT format) for all training samples.
    
    Returns:
        V_anchor_physical: [N, output_dim] in physical space
    """
    x_train = ngt_data['x_train'].to(device)
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    
    # VAE forward pass
    with torch.no_grad():
        Vm_scaled = vae_vm(x_train, use_mean=True)           # [N, Nbus] (scaled)
        Va_scaled_noslack = vae_va(x_train, use_mean=True)   # [N, Nbus-1] (scaled)
    
    scale_vm = config.scale_vm.item() if hasattr(config.scale_vm, 'item') else float(config.scale_vm)
    scale_va = config.scale_va.item() if hasattr(config.scale_va, 'item') else float(config.scale_va)
    
    VmLb = sys_data.VmLb
    VmUb = sys_data.VmUb
    if isinstance(VmLb, np.ndarray):
        VmLb = torch.from_numpy(VmLb).float().to(device)
        VmUb = torch.from_numpy(VmUb).float().to(device)
    elif isinstance(VmLb, torch.Tensor):
        VmLb = VmLb.to(device)
        VmUb = VmUb.to(device)
    
    Vm_anchor_full = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    Va_anchor_full_noslack = Va_scaled_noslack / scale_va
    
    N_samples = x_train.shape[0]
    Va_anchor_full = torch.zeros(N_samples, config.Nbus, device=device)
    Va_anchor_full[:, :bus_slack] = Va_anchor_full_noslack[:, :bus_slack]
    Va_anchor_full[:, bus_slack + 1:] = Va_anchor_full_noslack[:, bus_slack:]
    
    Vm_nonZIB = Vm_anchor_full[:, bus_Pnet_all]
    Va_nonZIB_noslack = Va_anchor_full[:, bus_Pnet_noslack_all]
    V_anchor_physical = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)
    
    return V_anchor_physical.detach()

 
# ------------------------ helpers ------------------------

def _ngt_to_full_voltage(V_ngt, ngt_data, sys_data, config, device):
    """
    Convert NGT format voltage to full voltage (Vm_full, Va_full).
    
    Args:
        V_ngt: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        device: Device
        
    Returns:
        Vm_full: [batch, Nbus] full voltage magnitude
        Va_full: [batch, Nbus] full voltage angle (with slack inserted)
    """
    batch_size = V_ngt.shape[0]
    bus_slack = int(sys_data.bus_slack)
    
    # Convert to numpy
    V_ngt_np = V_ngt.detach().cpu().numpy() if torch.is_tensor(V_ngt) else V_ngt
    
    # Insert slack bus Va (=0) to get full non-ZIB voltage
    xam_P = np.insert(V_ngt_np, ngt_data['idx_bus_Pnet_slack'][0], 0, axis=1)
    Va_len_with_slack = ngt_data['NPred_Va'] + 1
    Va_nonZIB = xam_P[:, :Va_len_with_slack]
    Vm_nonZIB = xam_P[:, Va_len_with_slack:Va_len_with_slack + ngt_data['NPred_Vm']]
    
    # Convert to complex and reconstruct full voltage
    Vx = Vm_nonZIB * np.exp(1j * Va_nonZIB)
    
    # Recover ZIB voltages if needed
    if ngt_data['NZIB'] > 0 and ngt_data.get('param_ZIMV') is not None:
        Vy = np.dot(ngt_data['param_ZIMV'], Vx.T).T
    else:
        Vy = None
    
    Ve = np.zeros((batch_size, config.Nbus))
    Vf = np.zeros((batch_size, config.Nbus))
    Ve[:, ngt_data['bus_Pnet_all']] = Vx.real
    Vf[:, ngt_data['bus_Pnet_all']] = Vx.imag
    if Vy is not None:
        Ve[:, ngt_data['bus_ZIB_all']] = Vy.real
        Vf[:, ngt_data['bus_ZIB_all']] = Vy.imag
    
    Vm_full = np.sqrt(Ve**2 + Vf**2)
    Va_full = np.arctan2(Vf, Ve)
    
    return Vm_full, Va_full


def _full_to_ngt_voltage(Vm_full, Va_full, ngt_data, sys_data, config):
    """
    Convert full voltage back to NGT format.
    
    Args:
        Vm_full: [batch, Nbus] full voltage magnitude
        Va_full: [batch, Nbus] full voltage angle (with slack)
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        
    Returns:
        V_ngt: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
    """
    batch_size = Vm_full.shape[0]
    bus_slack = int(sys_data.bus_slack)
    bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    
    # Extract non-ZIB voltages
    Vm_nonZIB = Vm_full[:, bus_Pnet_all]
    Va_nonZIB_noslack = Va_full[:, bus_Pnet_noslack_all]
    
    # Concatenate to NGT format: [Va_nonZIB_noslack, Vm_nonZIB]
    V_ngt = np.concatenate([Va_nonZIB_noslack, Vm_nonZIB], axis=1)
    
    return V_ngt


def _apply_post_processing_to_batch(
    V_ngt_batch, train_x_batch, ngt_data, sys_data, config, device, BRANFT, verbose=False
):
    """
    Apply post-processing to a batch of NGT format voltages.
    
    Args:
        V_ngt_batch: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
        train_x_batch: [batch, input_dim] load data
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        device: Device
        BRANFT: Branch from-to indices
        verbose: Whether to print warnings
        
    Returns:
        V_ngt_corrected: [batch, NPred_Va + NPred_Vm] corrected NGT format voltage
    """
    batch_size = V_ngt_batch.shape[0]
    
    # Convert NGT to full voltage
    Vm_full, Va_full = _ngt_to_full_voltage(V_ngt_batch, ngt_data, sys_data, config, device)
    
    # Build load arrays
    num_Pd = len(ngt_data['bus_Pd'])
    Pd_full = np.zeros((batch_size, config.Nbus))
    Qd_full = np.zeros((batch_size, config.Nbus))
    train_x_np = train_x_batch.detach().cpu().numpy() if torch.is_tensor(train_x_batch) else train_x_batch
    Pd_full[:, ngt_data['bus_Pd']] = train_x_np[:, :num_Pd]
    Qd_full[:, ngt_data['bus_Qd']] = train_x_np[:, num_Pd:]
    
    # Create a minimal EvalContext for post-processing
    try:
        # Create a minimal context-like structure
        class TempContext:
            def __init__(self):
                self.config = config
                self.sys_data = sys_data
                self.BRANFT = BRANFT
                self.device = device
                self.Nbus = config.Nbus
                self.Ntest = batch_size
                self.bus_slack = int(sys_data.bus_slack)
                self.baseMVA = float(sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else sys_data.baseMVA)
                self.branch = _as_numpy(sys_data.branch)
                self.Ybus = sys_data.Ybus
                self.Yf = sys_data.Yf
                self.Yt = sys_data.Yt
                self.bus_Pg = _ensure_1d_int(sys_data.bus_Pg)
                self.bus_Qg = _ensure_1d_int(sys_data.bus_Qg)
                self.MAXMIN_Pg = _as_numpy(ngt_data['MAXMIN_Pg'])
                self.MAXMIN_Qg = _as_numpy(ngt_data['MAXMIN_Qg'])
                self.idxPg = _ensure_1d_int(sys_data.idxPg)
                self.gencost = _as_numpy(sys_data.gencost)
                self.gencost_Pg = _as_numpy(ngt_data.get('gencost_Pg', None))
                self.his_V = _as_numpy(sys_data.his_V)
                self.hisVm_min = _as_numpy(sys_data.hisVm_min)
                self.hisVm_max = _as_numpy(sys_data.hisVm_max)
                self.bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
                self.bus_Pnet_noslack_all = self.bus_Pnet_all[self.bus_Pnet_all != self.bus_slack]
                self.bus_ZIB_all = _ensure_1d_int(ngt_data['bus_ZIB_all']) if 'bus_ZIB_all' in ngt_data else None
                self.param_ZIMV = ngt_data.get('param_ZIMV', None)
                self.VmLb = getattr(config, 'ngt_VmLb', None)
                self.VmUb = getattr(config, 'ngt_VmUb', None)
                self.DELTA = float(getattr(config, 'DELTA', 1e-4))
                self.k_dV = float(getattr(config, 'k_dV', 1.0))
                # [IMPROVEMENT] Use current voltage for Jacobian calculation instead of historical
                # This ensures more accurate linearization when voltage deviates from historical values
                self.flag_hisv = False  # Use current voltage, not historical
                self.Pdtest = Pd_full
                self.Qdtest = Qd_full
                # Store current voltage for Jacobian calculation
                self.current_V = Vm_full * np.exp(1j * Va_full)
        
        temp_ctx = TempContext()
        
        # Apply post-processing
        Vm_corrected, Va_corrected, _, dbg_info = post_process_like_evaluate_model(
            temp_ctx, Vm_full, Va_full
        )
        
        # [FIX] Only re-apply Kron reconstruction if post-processing used strict subspace mode
        # If relax_ngt_post=True (default), post_process_like_evaluate_model already corrected
        # all nodes (including ZIB) in full space, and applying Kron reconstruction here would
        # UNDO those corrections and snap back to the (potentially infeasible) manifold.
        # Strategy: Only apply Kron reconstruction when using strict subspace mode.
        use_strict_subspace = dbg_info.get('mode') == 'strict_subspace'
        if use_strict_subspace and ngt_data.get('param_ZIMV') is not None and ngt_data.get('bus_ZIB_all') is not None:
            # Extract corrected non-ZIB voltages (these are the ones we want to keep)
            bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
            bus_slack = int(sys_data.bus_slack)
            bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
            
            # Get corrected non-ZIB voltages
            Vm_nonZIB_corrected = Vm_corrected[:, bus_Pnet_all].copy()
            
            # Reconstruct Va_nonZIB with slack insertion (need full Va for all non-ZIB buses)
            # bus_Pnet_all includes slack, so we need to extract Va for all non-ZIB buses
            # Va_corrected shape: [batch, Nbus] - contains all buses including slack and ZIB
            # Simply extract Va for bus_Pnet_all (slack angle is already 0 in Va_corrected)
            Va_nonZIB_with_slack = Va_corrected[:, bus_Pnet_all].copy()
            # Ensure slack bus angle is 0 (should already be, but enforce it)
            idx_slack_in_Pnet = np.where(bus_Pnet_all == bus_slack)[0]
            if len(idx_slack_in_Pnet) > 0:
                Va_nonZIB_with_slack[:, idx_slack_in_Pnet[0]] = 0.0
            
            # Convert to complex and reconstruct ZIB using Kron
            Vx_corrected = Vm_nonZIB_corrected * np.exp(1j * Va_nonZIB_with_slack)
            Vy_reconstructed = np.dot(ngt_data['param_ZIMV'], Vx_corrected.T).T
            
            # Update corrected voltages: keep non-ZIB corrections, use Kron-reconstructed ZIB
            Vm_corrected_final = Vm_corrected.copy()
            Va_corrected_final = Va_corrected.copy()
            Vm_corrected_final[:, ngt_data['bus_ZIB_all']] = np.abs(Vy_reconstructed)
            Va_corrected_final[:, ngt_data['bus_ZIB_all']] = np.angle(Vy_reconstructed)
            
            Vm_corrected = Vm_corrected_final
            Va_corrected = Va_corrected_final
        
        # Convert back to NGT format
        V_ngt_corrected = _full_to_ngt_voltage(Vm_corrected, Va_corrected, ngt_data, sys_data, config)
        
        return torch.tensor(V_ngt_corrected, dtype=torch.float32, device=device)
        
    except Exception as e:
        # If post-processing fails, return original voltage
        if verbose:
            print(f"[Warning] Post-processing failed: {e}, returning original voltage")
        return V_ngt_batch if torch.is_tensor(V_ngt_batch) else torch.tensor(V_ngt_batch, dtype=torch.float32, device=device)

 
# ===================== [MO-PREF] Loss / Eval helpers ===================== 

def main():
    """
    Main function with support for training
    """
    # Load configuration
    config = get_config()
    config.print_config()
    
    # Create output directories if they don't exist
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True) 
    # Load data
    sys_data, _, _ = load_all_data(config) 
    results = train_unsupervised_ngt_gradient_descent(
                config=config, sys_data=sys_data, device=config.device,  
                num_iterations=500,   # 迭代次数
                learning_rate=1e-5,   # 更小的学习率 (1e-5 to 1e-6 recommended)
                tb_logger=None,       # 可选：TensorBoard logger 
                grad_clip_norm=1.0,  # 梯度裁剪
                use_post_processing=False,  # 启用后处理
                )    

if __name__ == "__main__":
    main()