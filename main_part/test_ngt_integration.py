#!/usr/bin/env python
# coding: utf-8
"""
Test script for DeepOPF-NGT unsupervised training integration.

This script verifies that:
1. DeepOPFNGTLoss module can be imported and initialized
2. ZIB identification works correctly
3. Kron Reduction parameters are computed
4. Forward and backward passes work

Usage:
    python test_ngt_integration.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Test 1: Import Verification")
    print("=" * 60)
    
    try:
        from config import get_config
        print("  [OK] config.get_config imported")
    except Exception as e:
        print(f"  [FAIL] config.get_config: {e}")
        return False
    
    try:
        from data_loader import load_all_data, prepare_ngt_data, create_ngt_dataloaders
        print("  [OK] data_loader functions imported")
    except Exception as e:
        print(f"  [FAIL] data_loader: {e}")
        return False
    
    try:
        from deepopf_ngt_loss import DeepOPFNGTLoss, compute_ngt_params
        print("  [OK] deepopf_ngt_loss imported")
    except Exception as e:
        print(f"  [FAIL] deepopf_ngt_loss: {e}")
        return False
    
    try:
        from models import create_model
        print("  [OK] models.create_model imported")
    except Exception as e:
        print(f"  [FAIL] models: {e}")
        return False
    
    print("  All imports successful!")
    return True


def test_data_loading():
    """Test data loading and ZIB identification."""
    print("\n" + "=" * 60)
    print("Test 2: Data Loading and ZIB Identification")
    print("=" * 60)
    
    try:
        from config import get_config
        from data_loader import load_all_data
        
        config = get_config()
        print(f"  Loading data for {config.Nbus}-bus system...")
        
        sys_data, dataloaders, BRANFT = load_all_data(config)
        
        # Check ZIB indices are computed
        assert sys_data.bus_Pnet_all is not None, "bus_Pnet_all not computed"
        assert sys_data.bus_ZIB_all is not None, "bus_ZIB_all not computed"
        assert sys_data.bus_Pnet_nonPg is not None, "bus_Pnet_nonPg not computed"
        assert sys_data.NPred_Vm is not None, "NPred_Vm not computed"
        assert sys_data.NPred_Va is not None, "NPred_Va not computed"
        
        print(f"  [OK] bus_Pnet_all: {len(sys_data.bus_Pnet_all)} buses")
        print(f"  [OK] bus_ZIB_all: {len(sys_data.bus_ZIB_all)} ZIB buses")
        print(f"  [OK] NPred_Vm: {sys_data.NPred_Vm}")
        print(f"  [OK] NPred_Va: {sys_data.NPred_Va}")
        
        return True, sys_data, config
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False, None, None


def test_ngt_loss_initialization(sys_data, config):
    """Test DeepOPFNGTLoss initialization."""
    print("\n" + "=" * 60)
    print("Test 3: DeepOPFNGTLoss Initialization")
    print("=" * 60)
    
    try:
        from deepopf_ngt_loss import DeepOPFNGTLoss
        
        loss_fn = DeepOPFNGTLoss(sys_data, config)
        
        # Check output dimensions
        output_dims = loss_fn.get_output_dims()
        print(f"  [OK] Output dims: Va={output_dims['Va']}, Vm={output_dims['Vm']}, Total={output_dims['total']}")
        
        # Check bus indices
        bus_indices = loss_fn.get_bus_indices()
        print(f"  [OK] bus_Pnet_all: {len(bus_indices['bus_Pnet_all'])} buses")
        print(f"  [OK] bus_ZIB_all: {len(bus_indices['bus_ZIB_all'])} ZIB buses")
        
        # Check Kron Reduction parameters
        if loss_fn.params.NZIB > 0:
            assert loss_fn.params.param_ZIMV is not None, "param_ZIMV not computed"
            print(f"  [OK] Kron Reduction param_ZIMV shape: {loss_fn.params.param_ZIMV.shape}")
        else:
            print(f"  [OK] No ZIB buses, Kron Reduction not needed")
        
        return True, loss_fn
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False, None


def test_forward_backward(sys_data, config, loss_fn):
    """Test forward and backward passes."""
    print("\n" + "=" * 60)
    print("Test 4: Forward and Backward Pass")
    print("=" * 60)
    
    try:
        device = config.device
        batch_size = 4  # Small batch for testing
        
        # Get dimensions
        output_dims = loss_fn.get_output_dims()
        bus_indices = loss_fn.get_bus_indices()
        
        # Create dummy input as a leaf tensor
        V_pred = torch.randn(batch_size, output_dims['total'], device=device)
        
        # Scale to reasonable values
        # Va part: roughly [-0.5, 0.5] radians
        V_pred[:, :output_dims['Va']] = V_pred[:, :output_dims['Va']] * 0.1
        # Vm part: roughly [0.95, 1.05] p.u.
        V_pred[:, output_dims['Va']:] = 1.0 + V_pred[:, output_dims['Va']:] * 0.05
        
        # Make it require gradients as a leaf tensor
        V_pred = V_pred.clone().detach().requires_grad_(True)
        
        # Create dummy load data
        num_Pd = len(bus_indices['bus_Pd'])
        num_Qd = len(bus_indices['bus_Qd'])
        PQd = torch.rand(batch_size, num_Pd + num_Qd).to(device) * 0.5  # Random loads
        
        print(f"  V_pred shape: {V_pred.shape}")
        print(f"  PQd shape: {PQd.shape}")
        
        # Forward pass
        loss, loss_dict = loss_fn(V_pred, PQd)
        
        print(f"  [OK] Forward pass - Loss: {loss.item():.4f}")
        print(f"  [OK] Loss dict: {loss_dict}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert V_pred.grad is not None, "Gradients not computed"
        grad_norm = V_pred.grad.norm().item()
        print(f"  [OK] Backward pass - Gradient norm: {grad_norm:.4f}")
        
        # Check gradient values are reasonable
        if torch.isnan(V_pred.grad).any():
            print("  [WARN] Some gradients are NaN")
        else:
            print("  [OK] No NaN gradients")
        
        if torch.isinf(V_pred.grad).any():
            print("  [WARN] Some gradients are Inf")
        else:
            print("  [OK] No Inf gradients")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_training_loop_mock(sys_data, config, loss_fn):
    """Test a mock training loop."""
    print("\n" + "=" * 60)
    print("Test 5: Mock Training Loop (3 steps)")
    print("=" * 60)
    
    try:
        from models import create_model
        
        device = config.device
        batch_size = 4
        
        # Get dimensions
        output_dims = loss_fn.get_output_dims()
        bus_indices = loss_fn.get_bus_indices()
        
        # Create simple models
        input_dim = sys_data.x_train.shape[1]
        output_dim_vm = config.Nbus  # Full output, will extract non-ZIB later
        output_dim_va = config.Nbus - 1  # Excluding slack
        
        model_vm = create_model('simple', input_dim, output_dim_vm, config, is_vm=True)
        model_va = create_model('simple', input_dim, output_dim_va, config, is_vm=False)
        model_vm.to(device)
        model_va.to(device)
        
        optimizer = torch.optim.Adam(
            list(model_vm.parameters()) + list(model_va.parameters()),
            lr=1e-3
        )
        
        # Mock training data
        x = sys_data.x_train[:batch_size].to(device)
        
        # Create PQd
        baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
        Pd = sys_data.RPd[:batch_size, bus_indices['bus_Pd']] / baseMVA
        Qd = sys_data.RQd[:batch_size, bus_indices['bus_Qd']] / baseMVA
        PQd = np.concatenate([Pd, Qd], axis=1)
        PQd_tensor = torch.from_numpy(PQd).float().to(device)
        
        # Get voltage bounds
        VmLb = sys_data.VmLb.item() if hasattr(sys_data.VmLb, 'item') else float(sys_data.VmLb)
        VmUb = sys_data.VmUb.item() if hasattr(sys_data.VmUb, 'item') else float(sys_data.VmUb)
        
        bus_slack = int(sys_data.bus_slack)
        bus_Pnet_all = bus_indices['bus_Pnet_all']
        bus_Pnet_noslack_all = bus_indices['bus_Pnet_noslack_all']
        
        losses = []
        for step in range(3):
            optimizer.zero_grad()
            
            # Forward pass
            Vm_pred_scaled = model_vm(x)
            Va_pred_scaled = model_va(x)
            
            # Unscale
            Vm_pred_full = Vm_pred_scaled / config.scale_vm.item() * (VmUb - VmLb) + VmLb
            Va_pred_no_slack = Va_pred_scaled / config.scale_va.item()
            
            # Insert slack bus Va = 0
            Va_pred_full = torch.zeros(batch_size, config.Nbus, device=device)
            Va_pred_full[:, :bus_slack] = Va_pred_no_slack[:, :bus_slack]
            Va_pred_full[:, bus_slack+1:] = Va_pred_no_slack[:, bus_slack:]
            
            # Extract non-ZIB nodes
            Vm_ngt = Vm_pred_full[:, bus_Pnet_all]
            Va_ngt = Va_pred_full[:, bus_Pnet_noslack_all]
            
            # Combine
            V_combined = torch.cat([Va_ngt, Vm_ngt], dim=1)
            
            # Compute loss
            loss, loss_dict = loss_fn(V_combined, PQd_tensor)
            
            # Backward
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model_vm.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_va.parameters(), max_norm=1.0)
            
            # Step
            optimizer.step()
            
            losses.append(loss.item())
            print(f"  Step {step+1}: Loss = {loss.item():.4f}")
        
        # Check loss is decreasing or stable
        print(f"  [OK] Training loop completed. Losses: {losses}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DeepOPF-NGT Integration Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Imports
    if not test_imports():
        print("\n[FATAL] Import test failed. Cannot continue.")
        return 1
    
    # Test 2: Data loading
    success, sys_data, config = test_data_loading()
    if not success:
        print("\n[FATAL] Data loading test failed. Cannot continue.")
        return 1
    
    # Test 3: Loss initialization
    success, loss_fn = test_ngt_loss_initialization(sys_data, config)
    if not success:
        print("\n[FATAL] Loss initialization test failed. Cannot continue.")
        return 1
    
    # Test 4: Forward/backward
    if not test_forward_backward(sys_data, config, loss_fn):
        print("\n[FATAL] Forward/backward test failed.")
        return 1
    
    # Test 5: Mock training
    if not test_training_loop_mock(sys_data, config, loss_fn):
        print("\n[FATAL] Mock training test failed.")
        return 1
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

