"""
run_analysis_outputs.py

Runs the current canonical figure/table scripts for the stroke + SLP timing
study after the database tables and primary PSM match have been built.

This is the single entrypoint for manuscript-ready outputs. The scripts it
runs live in analysis/. Older exploratory and diagnostic scripts live in
archive/ and are intentionally not run here.

Usage:
    python run_analysis_outputs.py
    python run_analysis_outputs.py 4   # resume from output step 4
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_DIR / "analysis"

OUTPUT_STEPS = {
    1: ("Figure 1 cohort flowchart", ANALYSIS_DIR / "make_flowchart.py"),
    2: ("Table 1 baseline characteristics", ANALYSIS_DIR / "make_table1.py"),
    3: ("Table 2 comprehensive current-method workbook", ANALYSIS_DIR / "make_table2.py"),
    4: ("Primary TV Cox results", ANALYSIS_DIR / "make_results_outputs.py"),
    5: ("Supplement 1 Elixhauser-stratified results", ANALYSIS_DIR / "make_gradient_elix.py"),
    6: ("Supplement 2 week-by-week dose-response", ANALYSIS_DIR / "comp_b_sensitivity.py"),
    7: ("Supplement 3 subgroup analysis", ANALYSIS_DIR / "comp_b_stratified.py"),
    8: ("Supplement 4 covariate balance love plot", ANALYSIS_DIR / "make_love_plot.py"),
}


def run_step(step_num, label, script_path):
    if not script_path.exists():
        raise FileNotFoundError(f"Missing output script: {script_path}")

    print(f"\n{'=' * 70}")
    print(f"  OUTPUT STEP {step_num}: {label}")
    print(f"  Script: {script_path.relative_to(PROJECT_DIR)}")
    print(f"{'=' * 70}")

    t0 = time.time()
    subprocess.run([sys.executable, str(script_path)], cwd=ANALYSIS_DIR, check=True)
    print(f"  Done in {time.time() - t0:.1f}s")


def main():
    start_step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Generating stroke + SLP manuscript outputs from step {start_step} ...")
    print(f"  Scripts: {ANALYSIS_DIR}")

    for step_num in sorted(OUTPUT_STEPS):
        if start_step <= step_num:
            run_step(step_num, *OUTPUT_STEPS[step_num])

    print(f"\n{'=' * 70}")
    print("  OUTPUT GENERATION COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
