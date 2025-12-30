#!/usr/bin/env python
# coding: utf-8
"""
DeepOPF-V Training Entry Point

Routes to appropriate training module based on configuration:
- Standard mode: train_standard.py (separate Vm/Va models)
- Multi-preference mode: train_multi_preference.py (preference-conditioned model)

Author: Peng Yue
Date: December 2025

Usage:
    # Standard training (single-objective)
    MULTI_PREF_SUPERVISED=False python train_supervised.py
    
    # Multi-preference training (default)
    python train_supervised.py
    MULTI_PREF_SUPERVISED=True MODEL_TYPE=flow python train_supervised.py
    
    # Or run modules directly:
    python train_standard.py
    python train_multi_preference.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(debug=False):
    """
    Main entry point - routes to appropriate training module.
    
    Set MULTI_PREF_SUPERVISED=True (default) for multi-preference mode.
    Set MULTI_PREF_SUPERVISED=False for standard Vm/Va training.
    """
    print("=" * 60)
    print("DeepOPF-V Training")
    print("=" * 60)
    
    # Check environment variable for mode selection
    use_multi_objective = os.environ.get('MULTI_PREF_SUPERVISED', 'True').lower() == 'true'
    
    if use_multi_objective:
        print("\n[Mode] Multi-Preference Training")
    print("=" * 60)
        from train_multi_preference import main as main_multi_pref
        return main_multi_pref(debug)
        else:
        print("\n[Mode] Standard Training (Vm/Va)")
    print("=" * 60)
        from train_standard import main as main_std
        return main_std(debug)


if __name__ == "__main__": 
    debug = bool(int(os.environ.get('DEBUG', '0')))
    results = main(debug=debug) 
