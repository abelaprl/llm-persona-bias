"""Wait for immutable complete shards, then integrate and verify this run."""
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def main():
    base=ROOT/'result/benchmarks/ethics/reports'
    parts=[base/f'religion_lm_v3_candidate_20260915_r2_shard_{s}' for s in ('a','b','c')]
    while not all((p/'manifest.json').is_file() for p in parts):
        ready=[p.name for p in parts if (p/'manifest.json').is_file()]
        print('Waiting for complete shard manifests; ready='+str(ready),flush=True)
        time.sleep(30)
    from src.benchmarks.ethics.integrate_religion_v3_candidate import integrate
    output=integrate(parts,base/'religion_lm_v3_candidate_20260916_r4_overall')
    from src.benchmarks.ethics.verify_religion_v3_candidate import verify
    verify(output,base/'religion_lm_v3_verification_20260916_r4')
    from src.benchmarks.ethics.summarize_religion_v3_candidate import summarize
    summarize(output,base/'religion_lm_v3_summary_20260916_r4')
    print('Automated integration finished; visual inspection and final report still required.',flush=True)


if __name__=='__main__':main()
