#!/usr/bin/env python3
"""
Validate the Sobol estimator with known analytic functions.
"""

import numpy as np
from sobol_sensitivity import saltelli_sample, sobol_indices

def test_linear_additive():
    """Test with f(x) = sum_i x_i, where each x_i ~ U(0,1)."""
    print("=== Linear additive f(x) = sum_i x_i ===")
    rng = np.random.default_rng(42)
    n_base = 1000
    d = 5
    samples = saltelli_sample(n_base, d, rng)
    y = np.sum(samples, axis=1)
    
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    S1, ST = sobol_indices(y_A, y_B, y_AB, y_BA)
    
    # Theoretical: Var(Y) = d * Var(U(0,1)) = d * 1/12
    # Var_i = Var(x_i) = 1/12, so S1_i = (1/12) / (d/12) = 1/d
    theoretical = 1.0 / d
    print(f"d = {d}, n_base = {n_base}")
    print(f"Theoretical S1_i = {theoretical:.4f}")
    print("Computed S1:", S1)
    print("Computed ST:", ST)
    print("Mean S1:", np.mean(S1))
    print("Sum S1:", np.sum(S1))
    print("ST - S1:", ST - S1)
    print()

def test_linear_weighted():
    """Test with f(x) = 2*x1 + 3*x2, where x1,x2 ~ U(0,1)."""
    print("=== Linear weighted f(x) = 2*x1 + 3*x2 ===")
    rng = np.random.default_rng(123)
    n_base = 1000
    d = 2
    samples = saltelli_sample(n_base, d, rng)
    y = 2 * samples[:,0] + 3 * samples[:,1]
    
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    S1, ST = sobol_indices(y_A, y_B, y_AB, y_BA)
    
    # Theoretical: Var(Y) = 4*Var(x1) + 9*Var(x2) = 4/12 + 9/12 = 13/12
    # S1_1 = (4/12)/(13/12) = 4/13 ≈ 0.3077
    # S1_2 = (9/13) ≈ 0.6923
    theoretical = np.array([4/13, 9/13])
    print(f"d = {d}, n_base = {n_base}")
    print(f"Theoretical S1: {theoretical}")
    print("Computed S1:", S1)
    print("Computed ST:", ST)
    print("ST - S1:", ST - S1)
    print()

def test_ishigami():
    """Test with Ishigami function, a standard benchmark for Sobol."""
    print("=== Ishigami function (3 parameters) ===")
    # f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1)
    # where x_i ~ U(-pi, pi), a=7, b=0.1
    # Known Sobol indices from literature:
    # S1_1 = 0.3139, S1_2 = 0.4424, S1_3 = 0
    # ST_1 = 0.5576, ST_2 = 0.4424, ST_3 = 0.2437
    rng = np.random.default_rng(456)
    n_base = 5000
    d = 3
    a = 7.0
    b = 0.1
    
    # Generate samples in [0,1] then rescale to [-pi, pi]
    samples_01 = saltelli_sample(n_base, d, rng)
    samples = -np.pi + 2*np.pi * samples_01
    
    x1 = samples[:,0]
    x2 = samples[:,1]
    x3 = samples[:,2]
    y = np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)
    
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    S1, ST = sobol_indices(y_A, y_B, y_AB, y_BA)
    
    print(f"n_base = {n_base}")
    print("Computed S1:", S1)
    print("Computed ST:", ST)
    print("ST - S1:", ST - S1)
    print("Literature S1: [0.3139, 0.4424, 0.0]")
    print("Literature ST: [0.5576, 0.4424, 0.2437]")
    print()

def test_discrete_output():
    """Test with discrete (integer) output to see if estimator breaks."""
    print("=== Discrete output f(x) = floor(3*x1) (0,1,2) ===")
    rng = np.random.default_rng(789)
    n_base = 1000
    d = 2
    samples = saltelli_sample(n_base, d, rng)
    y = np.floor(3 * samples[:,0])  # 0,1,2 with equal probability
    
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    S1, ST = sobol_indices(y_A, y_B, y_AB, y_BA)
    
    print(f"d = {d}, n_base = {n_base}")
    print("Output values:", np.unique(y))
    print("Computed S1:", S1)
    print("Computed ST:", ST)
    print("ST - S1:", ST - S1)
    print("Note: x2 should have zero effect")
    print()

def main():
    test_linear_additive()
    test_linear_weighted()
    test_ishigami()
    test_discrete_output()

if __name__ == "__main__":
    main()