"""
main.py
Run this to execute the full Loan Approval Predictor pipeline.
  1. Generates synthetic loan applicant dataset (1000 records)
  2. Creates EDA visualizations
  3. Trains classifiers and evaluates performance
"""

print("=" * 62)
print("  LOAN APPROVAL PREDICTOR")
print("  CodTech IT Solutions — ML Internship Task 3")
print("=" * 62)

import subprocess, sys

steps = [
    ("Step 1: Generating loan dataset...",        "generate_data.py"),
    ("Step 2: Creating visualizations...",         "visualize.py"),
    ("Step 3: Training models & evaluating...",    "model.py"),
]

for msg, script in steps:
    print(f"\n{'─'*62}")
    print(f"  {msg}")
    print(f"{'─'*62}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Error in {script}. Stopping.")
        sys.exit(1)

print("\n" + "=" * 62)
print("  ✅ ALL DONE! Check the outputs/ folder for images.")
print("=" * 62)
