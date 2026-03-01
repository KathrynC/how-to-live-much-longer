#!/usr/bin/env python3
"""
Corrected Sobol index estimators based on Saltelli (2002) and Jansen (1999).
"""

import numpy as np

def sobol_indices_corrected(y_A, y_B, y_AB, y_BA):
    """Compute first-order (S1) and total-order (ST) Sobol indices.
    
    Uses Saltelli's estimator for first-order:
        S_i = (1/N) Σ f(A) * (f(A_B^i) - f(B)) / V
    
    Uses Jansen's estimator for total-order:
        ST_i = (1/(2N)) Σ (f(A) - f(B_A^i))^2 / V
    
    where V is total variance estimated from A and B samples.
    
    Args:
        y_A: Model output for matrix A, shape (N,).
        y_B: Model output for matrix B, shape (N,).
        y_AB: Model output for AB cross-matrices, shape (D, N).
        y_BA: Model output for BA cross-matrices, shape (D, N).
    
    Returns:
        (S1, ST) — arrays of shape (D,) clipped to [0, 1] with ST >= S1.
    """
    n = len(y_A)
    d = y_AB.shape[0]
    
    # Total variance estimator using both A and B
    f0 = np.mean(np.concatenate([y_A, y_B]))
    var_total = np.var(np.concatenate([y_A, y_B]))
    
    if var_total < 1e-12:
        return np.zeros(d), np.zeros(d)
    
    S1 = np.zeros(d)
    ST = np.zeros(d)
    
    for i in range(d):
        # First-order: Saltelli (2002) estimator
        V_i = np.mean(y_A * (y_AB[i] - y_B))
        S1[i] = V_i / var_total
        
        # Total-order: Jansen (1999) estimator
        VT_i = 0.5 * np.mean((y_A - y_BA[i]) ** 2)
        ST[i] = VT_i / var_total
    
    # Clip to [0, 1] and enforce ST >= S1
    S1 = np.clip(S1, 0.0, 1.0)
    ST = np.clip(ST, 0.0, 1.0)
    ST = np.maximum(ST, S1)
    
    return S1, ST

def test_corrected():
    """Test the corrected estimator with known functions."""
    from sobol_sensitivity import saltelli_sample
    
    print("=== Testing corrected Sobol estimator ===\n")
    
    # 1. Linear additive
    rng = np.random.default_rng(42)
    n_base = 5000
    d = 5
    samples = saltelli_sample(n_base, d, rng)
    y = np.sum(samples, axis=1)
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    S1, ST = sobol_indices_corrected(y_A, y_B, y_AB, y_BA)
    theoretical = 1.0 / d
    print(f"Linear additive f(x)=sum(x), d={d}")
    print(f"Theoretical S1_i = {theoretical:.4f}")
    print(f"Computed S1: {S1}")
    print(f"Computed ST: {ST}")
    print(f"Mean S1: {np.mean(S1):.4f}")
    print(f"ST - S1: {ST - S1}")
    print()
    
    # 2. Linear weighted
    d = 2
    samples = saltelli_sample(n_base, d, rng)
    y = 2*samples[:,0] + 3*samples[:,1]
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    S1, ST = sobol_indices_corrected(y_A, y_B, y_AB, y_BA)
    theoretical = np.array([4/13, 9/13])  # (2^2)/(2^2+3^2) etc
    print(f"Linear weighted f(x)=2*x1 + 3*x2")
    print(f"Theoretical S1: {theoretical}")
    print(f"Computed S1: {S1}")
    print(f"Computed ST: {ST}")
    print(f"ST - S1: {ST - S1}")
    print()
    
    # 3. Ishigami function
    d = 3
    samples_01 = saltelli_sample(n_base, d, rng)
    samples = -np.pi + 2*np.pi * samples_01
    x1 = samples[:,0]
    x2 = samples[:,1]
    x3 = samples[:,2]
    a = 7.0
    b = 0.1
    y = np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    S1, ST = sobol_indices_corrected(y_A, y_B, y_AB, y_BA)
    print(f"Ishigami function")
    print(f"Literature S1: [0.3139, 0.4424, 0.0]")
    print(f"Literature ST: [0.5576, 0.4424, 0.2437]")
    print(f"Computed S1: {S1}")
    print(f"Computed ST: {ST}")
    print(f"ST - S1: {ST - S1}")
    print()
    
    # 4. Discrete output (floor)
    d = 2
    samples = saltelli_sample(n_base, d, rng)
    y = np.floor(3 * samples[:,0])  # 0,1,2
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    S1, ST = sobol_indices_corrected(y_A, y_B, y_AB, y_BA)
    print(f"Discrete f(x)=floor(3*x1)")
    print(f"Computed S1: {S1}")
    print(f"Computed ST: {ST}")
    print(f"ST - S1: {ST - S1}")
    print("Note: x2 should have zero effect")
    
    return True

if __name__ == "__main__":
    test_corrected()