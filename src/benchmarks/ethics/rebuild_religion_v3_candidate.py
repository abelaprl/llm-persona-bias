"""One-command rebuild using local processes and files, without named pipes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.benchmarks.ethics.integrate_religion_v3_candidate import integrate
from src.benchmarks.ethics.religion_lm_v3_io import create_run_directory

GROUPS = [('mistralai_ministral-8b-2512',),('qwen_qwen3-next-80b-a3b-thinking','openai_gpt-oss-20b'),
    ('deepseek_deepseek-r1-distill-qwen-32b','meta-llama_llama-3.1-8b-instruct','meta-llama_llama-3.3-70b-instruct')]


def repair_notebook_links(report):
    import json
    import nbformat
    from src.benchmarks.ethics.finalize_religion_v3_navigation import portable_links
    from src.benchmarks.ethics.religion_lm_v3_io import write_json,sha256
    manifest=json.loads((report/'manifest.json').read_text(encoding='utf-8'))
    source=report/manifest['default_notebook'];destination=Path(manifest['notebook_path'])
    expected=ROOT/'notebooks/religion'/('linguistic_marker_religion_ethics_v3_candidate_'+report.name.removeprefix('religion_lm_v3_candidate_')+'.ipynb')
    assert destination==expected and destination.is_file()
    reviewed=report/'religion_ethics_linguistic_markers_v3_reviewed_candidate.ipynb'
    if reviewed.exists() or (report/'presentation_review_manifest.json').exists():raise FileExistsError(reviewed)
    notebook=nbformat.read(source,as_version=4)
    for cell in notebook.cells:cell.source=portable_links(cell.source,report,report)
    with reviewed.open('x',encoding='utf-8') as handle:nbformat.write(notebook,handle)
    for cell in notebook.cells:cell.source=portable_links(cell.source,report,destination.parent)
    with destination.open('w',encoding='utf-8') as handle:nbformat.write(notebook,handle)
    write_json(report/'presentation_review_manifest.json',dict(preferred_report_notebook=reviewed.name,
        workspace_notebook=str(destination),report_notebook_sha256=sha256(reviewed),workspace_notebook_sha256=sha256(destination),
        source_notebook_retained=str(source),supersedes_default_notebook_navigation_only=True,numeric_data_unchanged=True,
        portable_local_links_checked=True,script_sha256=sha256(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum(): parser.error('Invalid run ID')
    base = ROOT/'result/ETHICS/reports'
    planned=[base/f'religion_lm_v3_candidate_{args.run_id}_shard_{index}' for index in range(len(GROUPS))]
    planned.append(base/f'religion_lm_v3_candidate_{args.run_id}_overall')
    planned.extend([base/f'religion_lm_v3_verification_{args.run_id}',base/f'religion_lm_v3_summary_{args.run_id}'])
    planned.extend([base/f'religion_lm_v3_precision_heldout_{args.run_id}',base/f'religion_lm_v3_delivery_{args.run_id}'])
    for path in planned:
        if path.exists():raise FileExistsError(path)
    logs = create_run_directory(base/f'religion_lm_v3_logs_{args.run_id}')
    handles,processes,parts = [],[],[]
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        for index,models in enumerate(GROUPS):
            run_id = f'{args.run_id}_shard_{index}'
            part = base/f'religion_lm_v3_candidate_{run_id}'
            if part.exists(): raise FileExistsError(part)
            handle = (logs/f'{index}.log').open('x',encoding='utf-8');handles.append(handle)
            command = [sys.executable,'-X','utf8',str(ROOT/'src/benchmarks/ethics/run_religion_lm_v3.py'),'--run-id',run_id,'--workers','1','--no-render']
            for model in models: command += ['--model',model]
            processes.append(subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=environment))
            parts.append(part)
        while any(p.poll() is None for p in processes):
            print('Local shard processes running; progress logs: '+str(logs),flush=True)
            time.sleep(30)
        if any(p.returncode for p in processes): raise RuntimeError('A shard failed; inspect logs. Existing artifacts retained.')
    finally:
        for handle in handles: handle.close()
    output=integrate(parts,base/f'religion_lm_v3_candidate_{args.run_id}_overall')
    repair_notebook_links(output)
    from src.benchmarks.ethics.verify_religion_v3_candidate import verify
    from src.benchmarks.ethics.summarize_religion_v3_candidate import summarize
    verify(output,base/f'religion_lm_v3_verification_{args.run_id}')
    summarize(output,base/f'religion_lm_v3_summary_{args.run_id}')
    from src.benchmarks.ethics.package_religion_v3_heldout_precision import package
    from src.benchmarks.ethics.describe_religion_v3_delivery import describe
    package(output,base/f'religion_lm_v3_precision_heldout_{args.run_id}')
    describe(output,base/f'religion_lm_v3_delivery_{args.run_id}')
    print(output,flush=True)

if __name__=='__main__': main()
