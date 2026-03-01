#!/usr/bin/env python3
"""
scripts/compare_production_benchmarks.py

Compares the Markdown tables in docs/benchmarks/benchmark_results_production.md 
between two git references.
"""

import argparse
import re
import subprocess
from typing import Dict

def get_file_at_ref(ref: str, path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path}"], 
            stderr=subprocess.STDOUT,
            encoding="utf-8"
        )
    except subprocess.CalledProcessError:
        print(f"Error: Could not find {path} at {ref}")
        return ""

def parse_markdown_tables(content: str) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Parses the production markdown into a nested dict:
    results[section_key][case_key][method] = {metric: value}
    """
    results = {}
    current_p = "p=unknown"
    current_size = "size=unknown"
    current_case = "case=unknown"
    
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Match p = X
        pm = re.match(r"^#+ p = (\d+)", line)
        if pm:
            current_p = f"p={pm.group(1)}"
        
        # Match Size XxX
        sm = re.match(r"^#+ Size (\d+x\d+)", line)
        if sm:
            current_size = sm.group(1)
            
        # Match Case: `name`
        cm = re.match(r"^#+ Case: `([^`]+)`", line)
        if cm:
            current_case = cm.group(1)
            
        # Match Table Header
        if line.startswith("| method |") and i + 2 < len(lines):
            header = [h.strip() for h in line.split("|") if h.strip()]
            i += 2 # Skip header and divider
            
            section_key = f"{current_p} | {current_size}"
            if section_key not in results:
                results[section_key] = {}
            if current_case not in results[section_key]:
                results[section_key][current_case] = {}
                
            while i < len(lines) and lines[i].strip().startswith("|"):
                row_line = lines[i].strip()
                # Remove bold markers **
                row_line = row_line.replace("**", "")
                cols = [c.strip() for c in row_line.split("|") if c.strip()]
                
                if len(cols) >= 2:
                    method = cols[0]
                    metrics = {}
                    for idx, val_str in enumerate(cols[1:], 1):
                        if idx < len(header):
                            metric_name = header[idx]
                            try:
                                # Handle percentages
                                if val_str.endswith("%"):
                                    val = float(val_str[:-1]) / 100.0
                                else:
                                    val = float(val_str)
                                metrics[metric_name] = val
                            except ValueError:
                                pass
                    results[section_key][current_case][method] = metrics
                i += 1
            continue
        i += 1
    return results

def compare_results(results_a, results_b):
    sections = sorted(set(results_a.keys()) | set(results_b.keys()))
    
    found_any = False
    for section in sections:
        cases_a = results_a.get(section, {})
        cases_b = results_b.get(section, {})
        cases = sorted(set(cases_a.keys()) | set(cases_b.keys()))
        
        section_printed = False
        
        for case in cases:
            methods_a = cases_a.get(case, {})
            methods_b = cases_b.get(case, {})
            methods = sorted(set(methods_a.keys()) | set(methods_b.keys()))
            
            case_printed = False
            
            for method in methods:
                metrics_a = methods_a.get(method, {})
                metrics_b = methods_b.get(method, {})
                
                all_metrics = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
                for metric in all_metrics:
                    val_a = metrics_a.get(metric)
                    val_b = metrics_b.get(metric)
                    
                    if val_a is not None and val_b is not None:
                        if val_a != 0:
                            delta_pct = ((val_b / val_a) - 1.0) * 100.0
                        else:
                            delta_pct = 0.0
                            
                        # Show if change > 0.1% or it's a key metric
                        if abs(delta_pct) > 0.1 or metric == "total_ms":
                            if not section_printed:
                                print(f"\n=== {section} ===")
                                section_printed = True
                            if not case_printed:
                                print(f"\n  Case: {case}")
                                print(f"    {'method':<25} | {'metric':<10} | {'side A':>10} | {'side B':>10} | {'delta %':>8}")
                                print(f"    {'-'*25}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*8}")
                                case_printed = True
                            
                            print(f"    {method:<25} | {metric:<10} | {val_a:10.4g} | {val_b:10.4g} | {delta_pct:+7.1f}%")
                            found_any = True
                    elif val_a is not None:
                        if not section_printed:
                            print(f"\n=== {section} ===")
                            section_printed = True
                        if not case_printed:
                            print(f"\n  Case: {case}")
                            print(f"    {'method':<25} | {'metric':<10} | {'side A':>10} | {'side B':>10} | {'delta %':>8}")
                            print(f"    {'-'*25}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*8}")
                            case_printed = True
                        print(f"    {method:<25} | {metric:<10} | {val_a:10.4g} | {'MISSING':>10} |")
                        found_any = True
                    elif val_b is not None:
                        if not section_printed:
                            print(f"\n=== {section} ===")
                            section_printed = True
                        if not case_printed:
                            print(f"\n  Case: {case}")
                            print(f"    {'method':<25} | {'metric':<10} | {'side A':>10} | {'side B':>10} | {'delta %':>8}")
                            print(f"    {'-'*25}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*8}")
                            case_printed = True
                        print(f"    {method:<25} | {metric:<10} | {'MISSING':>10} | {val_b:10.4g} |")
                        found_any = True
    if not found_any:
        print("No significant differences found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_a", help="First git reference (e.g. HEAD~1)")
    parser.add_argument("ref_b", nargs="?", default="", help="Second git reference (default: current working copy)")
    args = parser.parse_args()
    
    path = "docs/benchmarks/benchmark_results_production.md"
    
    content_a = get_file_at_ref(args.ref_a, path)
    if not content_a:
        return
        
    if args.ref_b:
        content_b = get_file_at_ref(args.ref_b, path)
    else:
        # Use working copy
        try:
            with open(path, "r", encoding="utf-8") as f:
                content_b = f.read()
        except FileNotFoundError:
            print(f"Error: {path} not found in working copy")
            return
            
    if not content_b:
        return
        
    results_a = parse_markdown_tables(content_a)
    results_b = parse_markdown_tables(content_b)
    
    compare_results(results_a, results_b)

if __name__ == "__main__":
    main()
