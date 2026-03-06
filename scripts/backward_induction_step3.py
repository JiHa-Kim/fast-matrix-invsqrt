import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('coeffs'))
from design_local_poly import design_local

def compute_ladder(target_rho0=0.0816, eps_gemm=0.008):
    print(f"Target Terminal Rho: {target_rho0}")
    print(f"Assumed GEMM Noise Margin: {eps_gemm}")
    print("=" * 60)
    
    current_target_rho_out = target_rho0 - eps_gemm
    ladder = []
    
    rho_ins = np.linspace(0.1, 0.99, 100)
    
    # We want to find the sequence backwards
    # Let's see what a d=2, d=3, d=4 polynomial can achieve
    
    step = 1
    while True:
        best_d = None
        best_rho_in = 0.0
        best_scalar_out = 0.0
        
        for d in [2, 3]:  # Prefer low degree for phase 2
            # Find largest rho_in that satisfies scalar_rho_out <= current_target_rho_out
            valid_rho_in = 0.0
            valid_out = 0.0
            
            for rho in rho_ins:
                try:
                    out = design_local(
                        rho=rho, deg=d, basis='cheb', r=2.0, 
                        proxy_log=1000, proxy_lin=2000, coef_bound=1e6, refine_bf16=False
                    )
                    scalar_rho_out = out['bf16_max_cert_err']
                    
                    if scalar_rho_out <= current_target_rho_out:
                        valid_rho_in = rho
                        valid_out = scalar_rho_out
                    else:
                        break  # since error increases monotonically with rho_in
                except Exception:
                    break
            
            if valid_rho_in > best_rho_in:
                best_rho_in = valid_rho_in
                best_d = d
                best_scalar_out = valid_out
                
        if best_rho_in <= current_target_rho_out + eps_gemm:
            print(f"Cannot find a polynomial that expands the interval safely.")
            break
            
        ladder.append({
            'step': step,
            'degree': best_d,
            'rho_in': best_rho_in,
            'scalar_out': best_scalar_out,
            'expected_hw_out': best_scalar_out + eps_gemm
        })
        
        print(f"Step {step} backwards: Use degree {best_d}")
        print(f"  Input interval:  [1 - {best_rho_in:.4f}, 1 + {best_rho_in:.4f}]")
        print(f"  Guarantees pure scalar output rho: {best_scalar_out:.4f}")
        print(f"  Expected HW output rho (incl. GEMM noise): {best_scalar_out + eps_gemm:.4f} <= {current_target_rho_out + eps_gemm:.4f}")
        print("-" * 60)
        
        current_target_rho_out = best_rho_in - eps_gemm
        step += 1
        
        if best_rho_in > 0.8:
            print(f"Reached extremely large input radius ({best_rho_in:.4f}), which overlaps with Phase 1 global domain.")
            break

if __name__ == "__main__":
    compute_ladder()
