# Design Doc: Phase 9 - Spatial Digital Twin Bridge

**Status:** Technical Specification  
**Objective:** Transform the 37-state lumped ODE model into a 3D-spatially resolved "Digital Twin" using existing organ meshes.

---

## 1. Concept: Beyond the Average Cell
Currently, the simulator calculates health for a "Generic Brain Cell." Phase 9 introduces **Anatomical Heterogeneity**. By voxelizing STLs of the brain and heart, we can assign different metabolic parameters to different regions (e.g., high metabolic demand in the hippocampus vs. low demand in white matter).

## 2. Architecture

### 2.1 Mesh Ingestion & Voxelization
- **Input:** `brain_fixed.stl`, `heart.stl`, `brainstem.stl`.
- **Process:** Convert surface meshes into 3D voxel grids (Metabolic Tensors) using `trimesh` or `numpy-stl`.
- **Resolution:** 1mm to 5mm voxels (configurable).

### 2.2 Spatial Coupling (The Diffusion Layer)
- **PDE Extension:** Introduce a Laplacian term ($
abla^2$) to the ODEs for ROS and Amyloid.
- **Rationale:** ROS doesn't just stay in one cell; it diffuses to neighbors. A high-damage area (like a surgical margin or amyloid plaque) can "poison" surrounding healthy tissue.
- **NRZ Precision:** The Neural Recovery Zone (State 36) is no longer a number; it is a 3D volume with explicit boundaries.

### 2.3 Structural Feedback (EDS Module)
- **Mechanical Analysis:** Use the `heart.stl` mesh to calculate wall stress based on geometry.
- **Metabolic Cost:** Map mechanical strain to the `structural_drag` parameter. High-strain regions require more ATP for repair, creating a spatial map of EDS-related energy debt.

## 3. Targeted Use Cases

### 3.1 PBM (Red Light) Depth Simulation
- **Photon Mapping:** Model red light penetration through the `brain_fixed.stl` geometry.
- **Dosage Optimization:** Calculate exactly how much light intensity is needed to reach the deep brain mitochondria for subjects like Ratio and Jasper.

### 3.2 Amyloid/Tau "Wildfire" Modeling
- **Seeding:** Map tau seeding onto the 3D mesh.
- **Propagation:** Simulate how tau "jumps" across anatomical connections (Connectome-guided diffusion).

### 3.3 The "Full Farm" Sensory Input
- **Systemic Distribution:** Model the 1.8x BDNF boost as a signal originating from the sensory cortex and diffusing throughout the 3D brain mesh.

## 4. Technical Stack Requirements
- `trimesh`: For STL/3MF processing.
- `scipy.ndimage`: For 3D diffusion convolution.
- `pyvista` or `VTK`: For high-fidelity 3D visualization of the metabolic tensors.

---

## 5. Success Metric: The "Metabolic Heatmap"
Phase 9 is successful when the simulator can produce a 3D color-coded map of a family member's brain (e.g., "Kathryn's Brain 2046") showing exactly where the heteroplasmy cliff is most likely to trigger first.
