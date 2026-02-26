#!/usr/bin/env python3
"""
benchmarks/run_benchmarks.py

Orchestrates running the solver performance benchmarks for SPD and non-SPD cases.
"""

import subprocess
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_command(cmd, log_filename):
    log_path = os.path.join(RESULTS_DIR, log_filename)
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_path}")

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=REPO_ROOT,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        process.wait()
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            sys.exit(process.returncode)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. SPD Benchmarks: n=1024, k=1,16,64,1024, p=1,2,4
    for p in [1, 2, 4]:
        cmd = [
            "uv",
            "run",
            "python",
            "benchmarks/solve/matrix_solve.py",
            "--p",
            str(p),
            "--sizes",
            "1024",
            "--k",
            "1,16,64,1024",
            "--trials",
            "5",  # Reduced trials for speed if needed, but keeping it decent
            "--dtype",
            "bf16",
        ]
        log_name = f"spd_p{p}_n1024_k1-1024_{timestamp}.log"
        run_command(cmd, log_name)

    # 2. Non-SPD Benchmarks: p=1, n=1024,2048, k=1,16,64,n
    # Note: matrix_solve_nonspd.py might need separate calls if k isn't easy to specify as 'n'
    # Actually, matrix_solve_nonspd.py takes --k as CSV of integers.

    # n=1024
    cmd_1024 = [
        "uv",
        "run",
        "python",
        "benchmarks/solve/matrix_solve_nonspd.py",
        "--sizes",
        "1024",
        "--k",
        "1,16,64,1024",
        "--trials",
        "5",
        "--dtype",
        "bf16",
    ]
    run_command(cmd_1024, f"nonspd_n1024_k1-1024_{timestamp}.log")

    # n=2048
    cmd_2048 = [
        "uv",
        "run",
        "python",
        "benchmarks/solve/matrix_solve_nonspd.py",
        "--sizes",
        "2048",
        "--k",
        "1,16,64,2048",
        "--trials",
        "5",
        "--dtype",
        "bf16",
    ]
    run_command(cmd_2048, f"nonspd_n2048_k1-2048_{timestamp}.log")


if __name__ == "__main__":
    main()

