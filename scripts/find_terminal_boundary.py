import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('coeffs'))
from design_local_poly import design_local
from design_bf16_poly import cheb_eval_bf16, bf16_round_f32, all_bf16_values_in_interval

def find_terminal_boundary(r=2.0):
    """
    Finds the exact maximum rho_in where the polynomial can STILL hit the absolute bf16 noise floor (0.0078125).
    """
    print("Searching for the absolute terminal boundary (maximum rho_in to hit 0.0078125)...")
    
    # We know from previous runs it's somewhere between 0.05 and 0.0895
    rho_ins = np.linspace(0.05, 0.1, 50)
    
    best_rhos = {2: 0, 3: 0, 4: 0, 5: 0}
    
    for d in [2, 3, 4, 5]:
        max_valid_rho = 0.0
        for rho in rho_ins:
            try:
                out = design_local(
                    rho=rho, deg=d, basis='cheb', r=r, 
                    proxy_log=1000, proxy_lin=2000, coef_bound=1e6, refine_bf16=False
                )
                err = out['bf16_max_cert_err']
                
                # 0.0078125 is exactly 1/128, the bf16 epsilon for numbers near 1
                if err <= 0.0078125 + 1e-6:
                    max_valid_rho = rho
                else:
                    break # since error is monotonically increasing with rho
            except Exception:
                break
                
        best_rhos[d] = max_valid_rho
        print(f"Degree {d}: max terminal rho_in = {max_valid_rho:.4f}")

if __name__ == "__main__":
    find_terminal_boundary()
