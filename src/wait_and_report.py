"""
Wait for checkpoints and generate final report.
"""

import time
from pathlib import Path
import subprocess

def count_files(dir_path):
    """Count gsm8k_*.json files."""
    p = Path(dir_path)
    if not p.exists():
        return 0
    return len(list(p.glob("gsm8k_*.json")))

def main():
    print("=" * 80)
    print("WAITING FOR CHECKPOINTS")
    print("=" * 80)
    print("")
    
    checkpoints = [
        ("Grok man_implicits", "results/grok_4_1_fast/man_implicits/gsm8k_raw", 1000),
        ("Grok woman_implicits", "results/grok_4_1_fast/woman_implicits/gsm8k_raw", 1000),
        ("Nebulon man_implicits", "results/bert_nebulon_alpha/man_implicits/gsm8k_raw", 600),
        ("Nebulon woman_implicits", "results/bert_nebulon_alpha/woman_implicits/gsm8k_raw", 600),
    ]
    
    reached = {}
    
    while len(reached) < len(checkpoints):
        for name, path, target in checkpoints:
            if name in reached:
                continue
            
            count = count_files(path)
            status = "✓" if count >= target else "•"
            print(f"\r{status} {name}: {count}/{target}", end="", flush=True)
            
            if count >= target:
                print(f" [REACHED]")
                reached[name] = True
        
        if len(reached) < len(checkpoints):
            time.sleep(10)
    
    print("\n")
    print("=" * 80)
    print("ALL CHECKPOINTS REACHED - Generating reports...")
    print("=" * 80)
    print("")
    
    # Run analysis
    result = subprocess.run(
        ["python3", "src/analyze_results.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("\n" + "=" * 80)
    print("REPORTS GENERATED")
    print("Files saved to: results/reports/")
    print("=" * 80)

if __name__ == "__main__":
    main()
