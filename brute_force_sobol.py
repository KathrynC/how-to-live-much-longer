import numpy as np

def brute_force_sobol(func, d, bounds, N=100000):
    """Brute-force Monte Carlo estimation of Sobol indices.
    
    Uses the definition: S_i = Var[E[Y|X_i]] / Var[Y]
    Estimates by sampling X and computing conditional expectations via binning.
    """
    rng = np.random.default_rng(12345)
    # Sample X uniformly in bounds
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    X = lo + (hi - lo) * rng.random((N, d))
    Y = func(X)
    var_total = np.var(Y)
    
    # For each parameter, bin its range and compute conditional expectation
    n_bins = 50
    S1 = np.zeros(d)
    for i in range(d):
        x_i = X[:, i]
        bins = np.linspace(lo[i], hi[i], n_bins+1)
        # digitize
        indices = np.digitize(x_i, bins) - 1
        indices = np.clip(indices, 0, n_bins-1)
        # compute mean per bin
        bin_means = np.zeros(n_bins)
        for b in range(n_bins):
            mask = indices == b
            if np.sum(mask) > 0:
                bin_means[b] = np.mean(Y[mask])
        # variance of conditional expectation
        var_cond = np.var(bin_means[indices])  # approximate
        S1[i] = var_cond / var_total
    
    return S1, var_total

def linear_func(X):
    return 2*X[:,0] + 3*X[:,1]

def main():
    d = 2
    bounds = [(0,1), (0,1)]
    S1, var = brute_force_sobol(linear_func, d, bounds, N=200000)
    print("Brute-force S1:", S1)
    print("Total variance:", var)
    print("Theoretical S1: [0.3077, 0.6923]")
    print("Theoretical var:", 2**2/12 + 3**2/12)
    
    # Also compute using Saltelli samples with corrected estimator
    from sobol_sensitivity import saltelli_sample
    rng = np.random.default_rng(42)
    n_base = 10000
    samples = saltelli_sample(n_base, d, rng)
    y = 2*samples[:,0] + 3*samples[:,1]
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    # Try different estimators
    var_total = np.var(np.concatenate([y_A, y_B]))
    f0 = np.mean(np.concatenate([y_A, y_B]))
    
    # Jansen estimator as implemented
    S1_jansen = np.zeros(d)
    for i in range(d):
        V_i = np.mean(y_B * (y_AB[i] - y_A))
        S1_jansen[i] = V_i / var_total
    print("\nJansen estimator S1:", S1_jansen)
    
    # Alternative estimator: S_i = (1/N) Σ f(A)*(f(A_B^i)-f(B)) / V
    S1_alt = np.zeros(d)
    for i in range(d):
        V_i = np.mean(y_A * (y_AB[i] - y_B))
        S1_alt[i] = V_i / var_total
    print("Alternative estimator S1:", S1_alt)
    
    # Centered version
    S1_cent = np.zeros(d)
    for i in range(d):
        V_i = np.mean((y_B - f0) * (y_AB[i] - y_A))
        S1_cent[i] = V_i / var_total
    print("Centered Jansen S1:", S1_cent)
    
    # Saltelli's original estimator
    # S_i = (1/N) Σ f(A)*(f(A_B^i)-f(A)) / V
    S1_salt = np.zeros(d)
    for i in range(d):
        V_i = np.mean(y_A * (y_AB[i] - y_A))
        S1_salt[i] = V_i / var_total
    print("Saltelli estimator S1:", S1_salt)

if __name__ == "__main__":
    main()