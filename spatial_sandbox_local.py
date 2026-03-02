"""spatial_sandbox_local.py

Phase 9 Proof-of-Concept: Spatial Mapping of ODE States.
Loads binary STL metadata and maps family metabolic data to 3D volumes.

NOTE: This file is ignored by git.
"""

import numpy as np
import struct
import os

def parse_stl_metadata(file_path):
    """Lightweight binary STL parser to get facet count and bounding box."""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(80)
            count_data = f.read(4)
            if len(count_data) < 4: return None
            num_facets = struct.unpack('<I', count_data)[0]
            
            v_min = np.array([np.inf, np.inf, np.inf])
            v_max = np.array([-np.inf, -np.inf, -np.inf])
            
            # Sample first 1000 facets
            for _ in range(min(num_facets, 1000)):
                f.read(12) # skip normal
                for _ in range(3):
                    v_raw = f.read(12)
                    if len(v_raw) < 12: break
                    v = np.array(struct.unpack('<fff', v_raw))
                    v_min = np.minimum(v_min, v)
                    v_max = np.maximum(v_max, v)
                f.read(2) # skip attribute
                
        return {
            "facets": num_facets,
            "bbox_min": v_min,
            "bbox_max": v_max,
            "volume_approx_cm3": np.prod(v_max - v_min) / 1000.0 if np.all(np.isfinite(v_min)) else 0
        }
    except Exception as e:
        print(f"Error parsing STL: {e}")
        return None

def map_metabolic_state_to_spatial(subject_name, atp_level, mesh_info):
    print(f"Mapping {subject_name}'s metabolic state to 3D mesh...")
    print(f"  Target Mesh Facets: {mesh_info['facets']:,}")
    print(f"  ATP Intensity: {atp_level:.4f} MU/day")
    
    center = (mesh_info['bbox_min'] + mesh_info['bbox_max']) / 2.0
    print(f"  Spatial Anchor (Center): {center}")
    if mesh_info['volume_approx_cm3'] > 0:
        print(f"  Metabolic Density: {atp_level / mesh_info['volume_approx_cm3']:.6f} MU/cm3")

if __name__ == "__main__":
    print("PHASE 9 SPATIAL BRIDGE: ORGAN MAPPING PROTOTYPE")
    
    # Try multiple possible locations for the asset
    possible_paths = ["../heart.stl", "heart.stl", "/Users/kathryncramer/heart.stl"]
    heart_file = None
    for p in possible_paths:
        if os.path.exists(p):
            heart_file = p
            break
    
    if heart_file:
        info = parse_stl_metadata(heart_file)
        if info:
            print(f"Mesh Loaded: {heart_file}")
            print(f"  Facets: {info['facets']:,}")
            print(f"  Bounding Box: {info['bbox_min']} to {info['bbox_max']}")
            
            print("-" * 30)
            map_metabolic_state_to_spatial("Jasper (EDS 1.45x)", 0.7145, info)
            
            print("-" * 30)
            map_metabolic_state_to_spatial("John Jr. (Rescue)", 0.5290, info)
        else:
            print("Failed to parse mesh metadata.")
    else:
        print("Error: Could not locate heart.stl.")
