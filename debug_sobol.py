import numpy as np
from sobol_sensitivity import saltelli_sample, sobol_indices

def debug_linear():
    rng = np.random.default_rng(999)
    n_base = 4
    d = 1
    samples = saltelli_sample(n_base, d, rng)
    print("samples shape:", samples.shape)
    print("samples:")
    print(samples)
    # f(x) = x
    y = samples[:,0]
    print("y:", y)
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    print("y_A:", y_A)
    print("y_B:", y_B)
    print("y_AB:", y_AB)
    print("y_BA:", y_BA)
    
    # compute manually
    var_total = np.var(np.concatenate([y_A, y_B]))
    print("var_total:", var_total)
    # V_i
    V_i = np.mean(y_B * (y_AB[0] - y_A))
    print("V_i:", V_i)
    S1 = V_i / var_total
    print("S1:", S1)
    # VT_i
    VT_i = 0.5 * np.mean((y_A - y_BA[0])**2)
    print("VT_i:", VT_i)
    ST = VT_i / var_total
    print("ST:", ST)
    
    # call function
    S1_f, ST_f = sobol_indices(y_A, y_B, y_AB, y_BA)
    print("Function S1:", S1_f)
    print("Function ST:", ST_f)

def debug_two_params():
    rng = np.random.default_rng(888)
    n_base = 100
    d = 2
    samples = saltelli_sample(n_base, d, rng)
    # f(x) = x0 + 2*x1
    y = samples[:,0] + 2*samples[:,1]
    n = n_base
    y_A = y[:n]
    y_B = y[n:2*n]
    y_AB = y[2*n:2*n + d*n].reshape(d, n)
    y_BA = y[2*n + d*n:].reshape(d, n)
    
    # compute theoretical
    # Var(Y) = Var(x0) + 4*Var(x1) = 1/12 + 4/12 = 5/12
    # S1_0 = (1/12)/(5/12) = 0.2
    # S1_1 = (4/12)/(5/12) = 0.8
    print("Theoretical S1: [0.2, 0.8]")
    
    S1, ST = sobol_indices(y_A, y_B, y_AB, y_BA)
    print("Computed S1:", S1)
    print("Computed ST:", ST)
    print("ST - S1:", ST - S1)
    
    # Let's compute variance contributions manually
    var_total = np.var(np.concatenate([y_A, y_B]))
    print("var_total:", var_total)
    print("expected var_total:", 5/12)
    for i in range(d):
        V_i = np.mean(y_B * (y_AB[i] - y_A))
        print(f"V_{i}:", V_i)
        print(f"S1_{i} (V_i/var):", V_i/var_total)
        VT_i = 0.5 * np.mean((y_A - y_BA[i])**2)
        print(f"VT_{i}:", VT_i)
        print(f"ST_{i}:", VT_i/var_total)

if __name__ == "__main__":
    print("=== Debug d=1 ===")
    debug_linear()
    print("\n=== Debug d=2 ===")
    debug_two_params()