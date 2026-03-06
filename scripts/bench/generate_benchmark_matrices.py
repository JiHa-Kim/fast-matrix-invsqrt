import numpy as np

sizes = [64, 128, 256, 512, 1024, 2048]
for n in sizes:
    Bs = []
    # Generate an SPD matrix close to identity
    for _ in range(5):
        A = np.random.randn(n, n)
        S = (A + A.T) * 0.5
        B = S @ S.T
        Bs.append(B)
    np.savez(f"../../data/benchmark_{n}.npz", Bs=np.array(Bs))
print("Generated .npz files for testing.")
