#!/usr/bin/env python3
"""
Visualize archetype criteria thresholds vs feature distributions.

Creates a multi-panel figure showing:
1. Feature distributions (histograms) with current thresholds overlaid
2. Archetype criteria satisfaction percentages
3. Comparison of current vs suggested thresholds
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def load_analysis():
    """Load threshold analysis results."""
    analysis_path = ROOT / "patterns" / "archetype_threshold_analysis.json"
    with open(analysis_path, 'r') as f:
        data = json.load(f)
    return data

def create_threshold_visualization(analysis_data, output_path=None):
    """Create visualization of thresholds vs distributions."""
    analysis = analysis_data.get("analysis", {})
    
    if not analysis:
        print("No analysis data found")
        return
    
    # Group by archetype
    archetypes = set()
    for key, data in analysis.items():
        archetypes.add(data["archetype"])
    
    archetypes = sorted(archetypes)
    
    # Create figure
    fig, axes = plt.subplots(len(archetypes), 1, figsize=(12, 4 * len(archetypes)))
    if len(archetypes) == 1:
        axes = [axes]
    
    for idx, arch in enumerate(archetypes):
        ax = axes[idx]
        
        # Get criteria for this archetype
        arch_criteria = [d for d in analysis.values() if d["archetype"] == arch]
        
        # Prepare data for plotting
        features = []
        current_thresholds = []
        suggested_thresholds = []
        meet_percentages = []
        predicates = []
        
        for crit in arch_criteria:
            feature = crit["feature"].split('.')[-1]  # Short name
            features.append(feature)
            current_thresholds.append(crit["current_value"])
            suggested_thresholds.append(crit["suggested_value"])
            meet_percentages.append(crit["meet_percentage"])
            predicates.append(crit["predicate"])
        
        # Create positions
        x = np.arange(len(features))
        width = 0.35
        
        # Plot current vs suggested thresholds
        bars1 = ax.bar(x - width/2, current_thresholds, width, label='Current', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, suggested_thresholds, width, label='Suggested', alpha=0.8, color='lightcoral')
        
        # Add meet percentage as text above bars
        for i, (curr, sugg, meet) in enumerate(zip(current_thresholds, suggested_thresholds, meet_percentages)):
            # Determine color based on meet percentage
            if meet < 25:
                color = 'red'
            elif meet > 75:
                color = 'green'
            else:
                color = 'orange'
            
            ax.text(i - width/2, curr + 0.02 * max(curr, sugg), f'{meet:.1f}%', 
                   ha='center', va='bottom', fontsize=8, color=color)
            ax.text(i + width/2, sugg + 0.02 * max(curr, sugg), f'p{25 if predicates[i] == "gt" else 75}', 
                   ha='center', va='bottom', fontsize=8, color='black')
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Threshold Value')
        ax.set_title(f'{arch.upper()} Archetype: Current vs Suggested Thresholds')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    return fig

def create_satisfaction_heatmap(analysis_data, output_path=None):
    """Create heatmap of criteria satisfaction percentages."""
    analysis = analysis_data.get("analysis", {})
    
    if not analysis:
        return None
    
    # Prepare data for heatmap
    archetypes = sorted(set(d["archetype"] for d in analysis.values()))
    features = sorted(set(d["feature"] for d in analysis.values()))
    
    # Create matrix
    matrix = np.full((len(features), len(archetypes)), np.nan)
    
    for key, data in analysis.items():
        arch_idx = archetypes.index(data["archetype"])
        feat_idx = features.index(data["feature"])
        matrix[feat_idx, arch_idx] = data["meet_percentage"]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(len(features)):
        for j in range(len(archetypes)):
            value = matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                       color='black' if 30 < value < 70 else 'white', fontsize=9)
    
    # Labels
    ax.set_xticks(np.arange(len(archetypes)))
    ax.set_xticklabels([a.upper() for a in archetypes], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([f.split('.')[-1] for f in features])
    
    ax.set_title('Criteria Satisfaction Percentage by Archetype and Feature')
    plt.colorbar(im, ax=ax, label='Percentage of Protocols Meeting Criterion')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    
    return fig

def main():
    print("=" * 70)
    print("Archetype Threshold Visualization")
    print("=" * 70)
    
    # Load analysis data
    analysis_data = load_analysis()
    if not analysis_data:
        print("No analysis data found. Run analyze_archetype_thresholds.py first.")
        return
    
    # Create output directory
    output_dir = ROOT / "output" / "archetype_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("Creating threshold comparison visualization...")
    fig1 = create_threshold_visualization(
        analysis_data,
        output_path=output_dir / "threshold_comparison.png"
    )
    
    print("Creating satisfaction heatmap...")
    fig2 = create_satisfaction_heatmap(
        analysis_data,
        output_path=output_dir / "satisfaction_heatmap.png"
    )
    
    # Create summary report
    summary_path = output_dir / "visualization_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Archetype Threshold Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        
        analysis = analysis_data.get("analysis", {})
        for arch in sorted(set(d["archetype"] for d in analysis.values())):
            f.write(f"{arch.upper()} ARCHETYPE:\n")
            arch_criteria = [d for d in analysis.values() if d["archetype"] == arch]
            for crit in sorted(arch_criteria, key=lambda x: x["feature"]):
                feature = crit["feature"]
                predicate = crit["predicate"]
                current = crit["current_value"]
                suggested = crit["suggested_value"]
                meet = crit["meet_percentage"]
                
                if predicate == "between":
                    tol = crit["current_tolerance"]
                    current_str = f"{current:.3f}+{tol:.3f}"
                else:
                    current_str = f"{current:.3f}"
                
                status = "STRICT" if meet < 25 else "LOOSE" if meet > 75 else "MODERATE"
                f.write(f"  {feature}: {predicate} {current_str} (meets {meet:.1f}%) [{status}]\n")
            f.write("\n")
    
    print(f"\nVisualizations saved to {output_dir}")
    print(f"Summary saved to {summary_path}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()