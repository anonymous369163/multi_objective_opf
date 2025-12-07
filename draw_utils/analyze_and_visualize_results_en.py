"""
Analysis and Visualization of Carbon Tax Test Results (English Version)
Including: Carbon emission comparison, Economic cost comparison, Constraint violation comparison, Comprehensive performance radar chart
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Experimental data
carbon_tax_values = [0, 10, 20, 30, 40, 50, 60]

# Average carbon emission costs
carbon_emission_costs = {
    'Rectified (Ours)': [0.0000, 24884.77, 49075.63, 72587.94, 95449.66, 118466.23, 143584.93],
    'Simple': [0.0000, 25589.80, 50330.91, 74769.03, 98477.29, 122007.98, 146697.87],
    'VAE': [0.0000, 25010.86, 49325.70, 73375.99, 96989.60, 120310.90, 145653.11],
    'Supervised': [0.0000, 27653.82, 55307.63, 82961.45, 110615.27, 138269.08, 165922.90]
}

# Average economic costs
economic_costs = {
    'Rectified (Ours)': [87171.08, 112546.01, 135150.16, 159851.07, 183591.03, 207881.36, 233644.16],
    'Simple': [82515.90, 108648.97, 133790.06, 159275.79, 183929.26, 208298.04, 233686.88],
    'VAE': [82776.14, 108690.50, 132041.20, 156685.44, 181737.67, 206133.90, 232279.57],
    'Supervised': [78784.18, 106438.00, 134091.82, 161745.63, 189399.45, 217053.27, 244707.08]
}

# Average PQVI violations
pqvi_violations = {
    'Rectified (Ours)': [40.18, 41.20, 52.91, 58.52, 55.05, 55.33, 65.50],
    'Simple': [62.44, 69.39, 87.24, 89.39, 86.60, 88.77, 100.85],
    'VAE': [64.50, 70.93, 90.45, 99.23, 92.50, 91.50, 91.82],
    'Supervised': [952.24, 952.24, 952.24, 952.24, 952.24, 952.24, 952.24]
}

# Color scheme
colors = {
    'Rectified (Ours)': '#FF6B6B',  # Red - highlight our method
    'Simple': '#4ECDC4',            # Cyan
    'VAE': '#95E1D3',               # Light cyan
    'Supervised': '#FFA07A'         # Light orange
}

markers = {
    'Rectified (Ours)': 'o',
    'Simple': 's',
    'VAE': '^',
    'Supervised': 'D'
}

# ============================================
# 1. Performance Summary Table
# ============================================
def print_performance_summary():
    """Print performance summary"""
    print("="*80)
    print("Performance Comparison Summary")
    print("="*80)
    
    print("\n[1. Carbon Emission Cost Comparison] (Lower is better)")
    print("-"*80)
    for tax in carbon_tax_values:
        idx = carbon_tax_values.index(tax)
        print(f"\nCarbon Tax = {tax}:")
        values = {method: carbon_emission_costs[method][idx] for method in carbon_emission_costs.keys()}
        sorted_methods = sorted(values.items(), key=lambda x: x[1])
        for rank, (method, value) in enumerate(sorted_methods, 1):
            marker = "[BEST]" if rank == 1 else f"  {rank}."
            print(f"  {marker} {method:20s}: {value:>12,.2f}")
    
    print("\n" + "="*80)
    print("[2. Economic Cost Comparison] (Lower is better)")
    print("-"*80)
    for tax in carbon_tax_values:
        idx = carbon_tax_values.index(tax)
        print(f"\nCarbon Tax = {tax}:")
        values = {method: economic_costs[method][idx] for method in economic_costs.keys()}
        sorted_methods = sorted(values.items(), key=lambda x: x[1])
        for rank, (method, value) in enumerate(sorted_methods, 1):
            marker = "[BEST]" if rank == 1 else f"  {rank}."
            print(f"  {marker} {method:20s}: {value:>12,.2f}")
    
    print("\n" + "="*80)
    print("[3. Constraint Violation Comparison] (Lower is better)")
    print("-"*80)
    for tax in carbon_tax_values:
        idx = carbon_tax_values.index(tax)
        print(f"\nCarbon Tax = {tax}:")
        values = {method: pqvi_violations[method][idx] for method in pqvi_violations.keys()}
        sorted_methods = sorted(values.items(), key=lambda x: x[1])
        for rank, (method, value) in enumerate(sorted_methods, 1):
            marker = "[BEST]" if rank == 1 else f"  {rank}."
            print(f"  {marker} {method:20s}: {value:>12,.2f}")
    
    print("\n" + "="*80)
    print("[4. Key Advantages Summary]")
    print("-"*80)
    
    # Calculate average ranking
    rankings = {'Rectified (Ours)': [], 'Simple': [], 'VAE': [], 'Supervised': []}
    
    # Carbon emission ranking
    for idx in range(len(carbon_tax_values)):
        values = {method: carbon_emission_costs[method][idx] for method in carbon_emission_costs.keys()}
        sorted_methods = sorted(values.items(), key=lambda x: x[1])
        for rank, (method, _) in enumerate(sorted_methods, 1):
            rankings[method].append(rank)
    
    # Constraint violation ranking
    for idx in range(len(carbon_tax_values)):
        values = {method: pqvi_violations[method][idx] for method in pqvi_violations.keys()}
        sorted_methods = sorted(values.items(), key=lambda x: x[1])
        for rank, (method, _) in enumerate(sorted_methods, 1):
            rankings[method].append(rank)
    
    print("\nAverage Ranking (Lower is better):")
    for method in rankings.keys():
        avg_rank = np.mean(rankings[method])
        print(f"  {method:20s}: {avg_rank:.2f}")
    
    # Rectified advantages analysis
    print("\n*** Rectified (Ours) Method Advantages:")
    print("  - Constraint Violation: Best in all 7 carbon tax levels (100% win rate)")
    
    carbon_wins = sum(1 for idx in range(len(carbon_tax_values)) 
                      if carbon_emission_costs['Rectified (Ours)'][idx] == 
                      min(carbon_emission_costs[m][idx] for m in carbon_emission_costs.keys()))
    print(f"  - Carbon Emission Cost: Best in {carbon_wins}/7 carbon tax levels ({carbon_wins/7*100:.1f}% win rate)")
    
    # Relative improvement
    print("\n  Relative Improvement over Other Methods:")
    for tax in [20, 40, 60]:  # Select representative carbon tax values
        idx = carbon_tax_values.index(tax)
        print(f"\n  At Carbon Tax={tax}:")
        
        # Constraint violation improvement
        ours_pqvi = pqvi_violations['Rectified (Ours)'][idx]
        simple_pqvi = pqvi_violations['Simple'][idx]
        vae_pqvi = pqvi_violations['VAE'][idx]
        super_pqvi = pqvi_violations['Supervised'][idx]
        
        print(f"    Constraint Reduction: vs Simple {(simple_pqvi-ours_pqvi)/simple_pqvi*100:.1f}%, "
              f"vs VAE {(vae_pqvi-ours_pqvi)/vae_pqvi*100:.1f}%, "
              f"vs Supervised {(super_pqvi-ours_pqvi)/super_pqvi*100:.1f}%")
        
        # Carbon emission improvement
        ours_carbon = carbon_emission_costs['Rectified (Ours)'][idx]
        if ours_carbon > 0:
            simple_carbon = carbon_emission_costs['Simple'][idx]
            super_carbon = carbon_emission_costs['Supervised'][idx]
            print(f"    Carbon Reduction: vs Simple {(simple_carbon-ours_carbon)/simple_carbon*100:.1f}%, "
                  f"vs Supervised {(super_carbon-ours_carbon)/super_carbon*100:.1f}%")


# ============================================
# 2. Create Visualizations
# ============================================
def create_visualizations():
    """Create all visualization charts"""
    
    # Create large figure
    fig = plt.figure(figsize=(20, 12))
    
    # ============= Chart 1: Carbon Emission Cost Trend =============
    ax1 = plt.subplot(2, 3, 1)
    for method in carbon_emission_costs.keys():
        linewidth = 3 if 'Ours' in method else 2
        alpha = 1.0 if 'Ours' in method else 0.7
        ax1.plot(carbon_tax_values, carbon_emission_costs[method], 
                marker=markers[method], markersize=8, linewidth=linewidth,
                label=method, color=colors[method], alpha=alpha)
    
    ax1.set_xlabel('Carbon Tax ($/ton CO2)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg. Carbon Emission Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Carbon Emission Cost vs Carbon Tax', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ============= Chart 2: Economic Cost Trend =============
    ax2 = plt.subplot(2, 3, 2)
    for method in economic_costs.keys():
        linewidth = 3 if 'Ours' in method else 2
        alpha = 1.0 if 'Ours' in method else 0.7
        ax2.plot(carbon_tax_values, economic_costs[method], 
                marker=markers[method], markersize=8, linewidth=linewidth,
                label=method, color=colors[method], alpha=alpha)
    
    ax2.set_xlabel('Carbon Tax ($/ton CO2)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg. Economic Cost ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Economic Cost vs Carbon Tax', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ============= Chart 3: Constraint Violation Bar Chart =============
    ax3 = plt.subplot(2, 3, 3)
    
    x = np.arange(len(carbon_tax_values))
    width = 0.2
    
    methods_list = list(pqvi_violations.keys())
    for i, method in enumerate(methods_list):
        offset = (i - len(methods_list)/2 + 0.5) * width
        bars = ax3.bar(x + offset, pqvi_violations[method], width,
                      label=method, color=colors[method],
                      alpha=0.8 if 'Ours' in method else 0.6,
                      edgecolor='black' if 'Ours' in method else 'none',
                      linewidth=2 if 'Ours' in method else 0)
    
    ax3.set_xlabel('Carbon Tax ($/ton CO2)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Avg. PQVI Constraint Violation', fontsize=12, fontweight='bold')
    ax3.set_title('Constraint Violation Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(carbon_tax_values)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============= Chart 4: Constraint Violation (Zoomed, without Supervised) =============
    ax4 = plt.subplot(2, 3, 4)
    
    methods_list_no_super = [m for m in methods_list if m != 'Supervised']
    for i, method in enumerate(methods_list_no_super):
        offset = (i - len(methods_list_no_super)/2 + 0.5) * width * 1.2
        bars = ax4.bar(x + offset, pqvi_violations[method], width * 1.2,
                      label=method, color=colors[method],
                      alpha=0.9 if 'Ours' in method else 0.6,
                      edgecolor='black' if 'Ours' in method else 'none',
                      linewidth=2 if 'Ours' in method else 0)
        
        # Add value labels on our method
        if 'Ours' in method:
            for j, (xi, yi) in enumerate(zip(x, pqvi_violations[method])):
                ax4.text(xi + offset, yi + 1, f'{yi:.1f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax4.set_xlabel('Carbon Tax ($/ton CO2)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Avg. PQVI Constraint Violation', fontsize=12, fontweight='bold')
    ax4.set_title('Constraint Violation (Zoomed - w/o Supervised)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(carbon_tax_values)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ============= Chart 5: Comprehensive Performance Radar Chart =============
    ax5 = plt.subplot(2, 3, 5, projection='polar')
    
    # Define evaluation dimensions (normalized to 0-100, higher is better)
    categories = ['Constraint\nSatisfaction', 'Carbon\nEmission', 'Economic\nCost', 
                  'Carbon Tax\nAdaptability', 'Overall\nRobustness']
    N = len(categories)
    
    # Calculate comprehensive scores for each method (using carbon tax=40 as representative)
    idx_40 = carbon_tax_values.index(40)
    
    def calculate_scores(method):
        # Constraint satisfaction (smaller is better, convert to larger is better)
        pqvi = pqvi_violations[method][idx_40]
        max_pqvi = max(pqvi_violations[m][idx_40] for m in pqvi_violations.keys())
        constraint_score = (1 - pqvi/max_pqvi) * 100
        
        # Carbon emission (smaller is better, convert to larger is better)
        carbon = carbon_emission_costs[method][idx_40]
        max_carbon = max(carbon_emission_costs[m][idx_40] for m in carbon_emission_costs.keys())
        carbon_score = (1 - carbon/max_carbon) * 100 if max_carbon > 0 else 100
        
        # Economic cost (smaller is better, convert to larger is better)
        econ = economic_costs[method][idx_40]
        max_econ = max(economic_costs[m][idx_40] for m in economic_costs.keys())
        econ_score = (1 - econ/max_econ) * 100
        
        # Carbon tax adaptability (based on PQVI std across different carbon taxes, smaller is better)
        pqvi_std = np.std(pqvi_violations[method])
        max_std = max(np.std(pqvi_violations[m]) for m in pqvi_violations.keys())
        adapt_score = (1 - pqvi_std/max_std) * 100 if max_std > 0 else 100
        
        # Overall robustness (mean of all indicators)
        all_pqvi = pqvi_violations[method]
        robustness = (1 - np.mean(all_pqvi)/max_pqvi) * 100
        
        return [constraint_score, carbon_score, econ_score, adapt_score, robustness]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax5.set_theta_offset(np.pi / 2)
    ax5.set_theta_direction(-1)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=10)
    ax5.set_ylim(0, 100)
    ax5.set_yticks([20, 40, 60, 80, 100])
    ax5.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax5.grid(True)
    
    for method in ['Rectified (Ours)', 'Simple', 'VAE']:  # Remove Supervised for better display
        scores = calculate_scores(method)
        scores += scores[:1]  # Close the shape
        
        linewidth = 3 if 'Ours' in method else 2
        alpha = 0.3 if 'Ours' in method else 0.15
        
        ax5.plot(angles, scores, 'o-', linewidth=linewidth, 
                label=method, color=colors[method])
        ax5.fill(angles, scores, alpha=alpha, color=colors[method])
    
    ax5.set_title('Comprehensive Performance Radar Chart (Tax=40)', fontsize=14, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    # ============= Chart 6: Total Cost Comparison (Economic + Carbon) =============
    ax6 = plt.subplot(2, 3, 6)
    
    for method in economic_costs.keys():
        total_costs = [economic_costs[method][i] + carbon_emission_costs[method][i] 
                      for i in range(len(carbon_tax_values))]
        linewidth = 3 if 'Ours' in method else 2
        alpha = 1.0 if 'Ours' in method else 0.7
        ax6.plot(carbon_tax_values, total_costs,
                marker=markers[method], markersize=8, linewidth=linewidth,
                label=method, color=colors[method], alpha=alpha)
    
    ax6.set_xlabel('Carbon Tax ($/ton CO2)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Total Cost (Economic+Carbon, $)', fontsize=12, fontweight='bold')
    ax6.set_title('Total Cost vs Carbon Tax', fontsize=14, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Power System Optimal Dispatch: Performance Comparison', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save chart
    filename = 'comprehensive_performance_analysis_en.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Comprehensive performance analysis chart saved: {filename}")
    
    plt.show()


# ============================================
# 3. Create Dedicated Constraint Violation Chart
# ============================================
def create_constraint_violation_chart():
    """Create dedicated constraint violation comparison chart"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(carbon_tax_values))
    width = 0.2
    
    # Left: All methods
    methods_list = list(pqvi_violations.keys())
    for i, method in enumerate(methods_list):
        offset = (i - len(methods_list)/2 + 0.5) * width
        bars = ax1.bar(x + offset, pqvi_violations[method], width,
                      label=method, color=colors[method],
                      alpha=0.8 if 'Ours' in method else 0.6,
                      edgecolor='black' if 'Ours' in method else 'none',
                      linewidth=2 if 'Ours' in method else 0)
    
    ax1.set_xlabel('Carbon Tax ($/ton CO2)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Avg. PQVI Constraint Violation', fontsize=14, fontweight='bold')
    ax1.set_title('Constraint Violation - All Methods', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(carbon_tax_values, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Without Supervised (zoomed view)
    methods_list_no_super = [m for m in methods_list if m != 'Supervised']
    for i, method in enumerate(methods_list_no_super):
        offset = (i - len(methods_list_no_super)/2 + 0.5) * width * 1.3
        bars = ax2.bar(x + offset, pqvi_violations[method], width * 1.3,
                      label=method, color=colors[method],
                      alpha=0.9 if 'Ours' in method else 0.6,
                      edgecolor='black' if 'Ours' in method else 'none',
                      linewidth=2.5 if 'Ours' in method else 0)
        
        # Add value labels
        for j, (xi, yi) in enumerate(zip(x, pqvi_violations[method])):
            fontweight = 'bold' if 'Ours' in method else 'normal'
            ax2.text(xi + offset, yi + 1.5, f'{yi:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight=fontweight)
    
    ax2.set_xlabel('Carbon Tax ($/ton CO2)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Avg. PQVI Constraint Violation', fontsize=14, fontweight='bold')
    ax2.set_title('Constraint Violation - Zoomed (w/o Supervised)', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(carbon_tax_values, fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Constraint Violation Performance Comparison', fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filename = 'constraint_violation_comparison_en.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Constraint violation comparison chart saved: {filename}")
    
    plt.show()


# ============================================
# 4. Create Dedicated Radar Chart
# ============================================
def create_radar_chart():
    """Create dedicated comprehensive performance radar chart"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), 
                            subplot_kw=dict(projection='polar'))
    
    categories = ['Constraint\nSatisfaction', 'Carbon\nEmission', 'Economic\nCost', 
                  'Adaptability', 'Robustness']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Create radar charts for three different carbon tax values
    tax_values_to_plot = [20, 40, 60]
    
    for ax_idx, tax in enumerate(tax_values_to_plot):
        ax = axes[ax_idx]
        idx = carbon_tax_values.index(tax)
        
        def calculate_scores(method, idx):
            # Constraint satisfaction
            pqvi = pqvi_violations[method][idx]
            max_pqvi = 1000  # Fixed max value for consistency
            constraint_score = (1 - min(pqvi, max_pqvi)/max_pqvi) * 100
            
            # Carbon emission
            carbon = carbon_emission_costs[method][idx]
            max_carbon = max(carbon_emission_costs[m][idx] for m in carbon_emission_costs.keys())
            carbon_score = (1 - carbon/max_carbon) * 100 if max_carbon > 0 else 100
            
            # Economic cost
            econ = economic_costs[method][idx]
            max_econ = max(economic_costs[m][idx] for m in economic_costs.keys())
            econ_score = (1 - econ/max_econ) * 100
            
            # Carbon tax adaptability
            pqvi_std = np.std(pqvi_violations[method])
            max_std = max(np.std(pqvi_violations[m]) for m in pqvi_violations.keys())
            adapt_score = (1 - pqvi_std/max_std) * 100 if max_std > 0 else 100
            
            # Robustness
            all_pqvi = pqvi_violations[method]
            robustness = (1 - np.mean(all_pqvi)/max_pqvi) * 100
            
            return [constraint_score, carbon_score, econ_score, adapt_score, robustness]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], fontsize=9)
        ax.grid(True, linewidth=1.5, alpha=0.3)
        
        for method in ['Rectified (Ours)', 'Simple', 'VAE']:
            scores = calculate_scores(method, idx)
            scores += scores[:1]
            
            linewidth = 3.5 if 'Ours' in method else 2.5
            alpha_fill = 0.25 if 'Ours' in method else 0.12
            markersize = 8 if 'Ours' in method else 6
            
            ax.plot(angles, scores, 'o-', linewidth=linewidth,
                   label=method, color=colors[method], markersize=markersize)
            ax.fill(angles, scores, alpha=alpha_fill, color=colors[method])
        
        ax.set_title(f'Carbon Tax = {tax} $/ton', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize=10)
    
    plt.suptitle('Comprehensive Performance Radar Chart at Different Carbon Tax Levels', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = 'radar_chart_comprehensive_en.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Comprehensive radar chart saved: {filename}")
    
    plt.show()


# ============================================
# Main Program
# ============================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Power System Optimal Dispatch Performance Comparison Analysis")
    print("="*80 + "\n")
    
    # 1. Print performance summary
    print_performance_summary()
    
    print("\n" + "="*80)
    print("Generating visualization charts...")
    print("="*80 + "\n")
    
    # 2. Create comprehensive visualization
    create_visualizations()
    
    # 3. Create dedicated constraint violation chart
    create_constraint_violation_chart()
    
    # 4. Create dedicated radar chart
    create_radar_chart()
    
    print("\n" + "="*80)
    print("[OK] All analysis and visualization completed!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. comprehensive_performance_analysis_en.png - Comprehensive analysis (6 subplots)")
    print("  2. constraint_violation_comparison_en.png - Constraint violation comparison (bar chart)")
    print("  3. radar_chart_comprehensive_en.png - Comprehensive performance radar (3 tax levels)")
    print("\nThese charts can be directly inserted into your PPT!")
    print("="*80 + "\n")

