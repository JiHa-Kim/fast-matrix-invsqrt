import pandas as pd

sizes = [64, 128, 256, 512, 1024, 2048]
degs = [2, 3, 4, 5]

print("## Performance Analysis (GPU execution time)\n")
print("| Matrix Size | Degree | Monomial (ms) | Chebyshev (ms) | Speed Advantage |")
print("| :--- | :--- | :--- | :--- | :--- |")

for size in sizes:
    df = pd.read_csv(f"../../results/bench/run_{size}.csv")
    df = df[
        df["iters"] == 5
    ]  # use iters=5 for execution times (avoids compile overhead)
    df = df.groupby(["basis", "deg"])["total_ms"].mean().reset_index()
    for d in degs:
        mono_ms = df[(df["basis"] == "mono") & (df["deg"] == d)]["total_ms"].values[0]
        cheb_ms = df[(df["basis"] == "cheb") & (df["deg"] == d)]["total_ms"].values[0]
        if mono_ms < cheb_ms:
            adv = f"**Monomial ({cheb_ms / mono_ms:.2f}x)**"
        else:
            adv = f"**Chebyshev ({mono_ms / cheb_ms:.2f}x)**"
        size_str = f"**{size} x {size}**" if d == degs[0] else ""
        print(f"| {size_str} | {d} | {mono_ms:.2f} | {cheb_ms:.2f} | {adv} |")

print("\n## Numerical Stability (Relative Frobenius Error)\n")
print("| Size | Degree | Monomial Final Error | Chebyshev Final Error |")
print("| :--- | :--- | :--- | :--- |")

for size in sizes:
    df = pd.read_csv(f"../../results/bench/run_{size}.csv")
    df = df[df["iters"] == 5]  # Final error after 5 iterations
    df = df.groupby(["basis", "deg"])["deltaF_final"].mean().reset_index()
    for d in degs:
        mono_err = df[(df["basis"] == "mono") & (df["deg"] == d)][
            "deltaF_final"
        ].values[0]
        cheb_err = df[(df["basis"] == "cheb") & (df["deg"] == d)][
            "deltaF_final"
        ].values[0]
        size_str = f"**{size} x {size}**" if d == degs[0] else ""
        print(f"| {size_str} | {d} | {mono_err:.2f} | {cheb_err:.2f} |")
