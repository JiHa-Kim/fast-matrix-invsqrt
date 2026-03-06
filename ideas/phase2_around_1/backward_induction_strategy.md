# Reverse-Engineering the Preconditioning Sequence via Backward Induction

## 1. The Terminal State: The bf16 Noise Floor
Based on exact hardware emulation, the absolute minimum error achievable for the whitening certificate $S = Z^T B Z$ in pure `bf16` arithmetic is bounded by the machine epsilon: $u = 2^{-7} \approx 7.8 \times 10^{-3}$. 

We have proven that a single Phase 2 local minimax polynomial of degree $d=3$, designed for the interval $[0.9, 1.1]$ (i.e., $\rho = 0.1$), will successfully compress any matrix spectrum within that interval directly into the absolute hardware noise floor. 

Therefore, our **terminal target** for all preceding iterations is to reach $\rho_0 \le 0.1$. Once $\rho \le 0.1$, the algorithm is effectively finished (one application of the terminal $d=3$ polynomial completes the process).

## 2. The Problem Statement
Our objective is to work *backwards* from this terminal state ($\rho_0 = 0.1$) to define the optimal, minimal sequence of preconditioning steps. 

Let $\rho_k$ be the spectral radius (interval $[1-\rho_k, 1+\rho_k]$) at step $k$. 
Let $P_d$ be a polynomial of degree $d$. 

We want to find a sequence of intervals $\rho_0, \rho_1, \rho_2, \dots$ and degrees $d_1, d_2, \dots$ such that:
$$ \text{Applying } P_{d_k} \text{ over } [1-\rho_k, 1+\rho_k] \implies \text{Resulting spectrum is bounded by } \rho_{k-1} $$

We want to maximize $\rho_k$ for a given $d_k$ to cover as much ground as possible per step, minimizing the total number of GEMMs. 

## 3. The Challenge: bf16 GEMM Quantization Noise
If we were operating in theoretical continuous math, we could just evaluate the minimax approximation error: $\rho_{k-1} = \max_{x \in [1-\rho_k, 1+\rho_k]} |1 - x P_{d_k}(x)^r|$. 

However, in hardware, evaluating $S_{\text{new}} = Z_{\text{new}}^T B Z_{\text{new}}$ is performed via highly quantized `bf16` matrix multiplications. 
The quantization introduces non-deterministic noise (spectral drift) at every iteration. 
If theoretical math predicts an output interval of $[0.91, 1.09]$ ($\rho_{out} = 0.09$), the actual hardware GEMMs might produce eigenvalues like $1.12$, breaking the strict bounds of the next polynomial in the ladder and causing catastrophic divergence.

To counteract this, we must reverse-engineer the minimum safe noise margin $\epsilon_{\text{gemm}}$.
The true mapping is:
$$ \rho_{k-1}^{\text{hardware}} = \rho_{k-1}^{\text{theoretical}} + \epsilon_{\text{gemm}}(\rho_k, d_k) $$

We must make $\epsilon_{\text{gemm}}$ mathematically safe, but as *tight as possible* so we do not over-damp the algorithm and artificially destroy convergence speed.

## 4. Proposed Strategy for Backward Induction

We will execute the following steps to construct the optimal ladder:

**Step 1: Define the Pure Scalar Mapping**
Create an exhaustive lookup table of $\rho_{out}$ as a function of $(\rho_{in}, d)$ using exact `bf16` scalar emulation. For a given target $\rho_{out}$, what is the absolute largest $\rho_{in}$ that maps into it for $d \in \{2, 3, 4, 5\}$?

**Step 2: Model the GEMM Drift ($\epsilon_{\text{gemm}}$)**
Empirically and theoretically bound the worst-case eigenvalue drift introduced by computing $Z^T B Z$ in pure `bf16` relative to the scalar evaluation of the polynomial. This bound will depend on the matrix dimension $N$ and the condition number/norm of the matrices being multiplied.

**Step 3: Compute the Safe Ladder**
Starting at $\rho_0 = 0.1$:
1. Define target $\rho_{target} = 0.1 - \epsilon_{\text{safety}}$. 
2. Find the largest $\rho_1$ such that a $d=2$ or $d=3$ polynomial maps $\rho_1 \to \rho_{target}$. 
3. Repeat the process, setting the new target to $\rho_1 - \epsilon_{\text{safety}}$, to find $\rho_2$, and so on, until we reach the boundary of the Phase 1 output (e.g., $\rho \approx 0.5$ or wherever the global minimax transitions to local).

**Step 4: Dynamic Hardware Verification**
Write a script that physically runs the sequence using `torch` in `bf16` with real matrices of varying dimensions to prove that the theoretically safe ladder holds true under adversarial noise.
