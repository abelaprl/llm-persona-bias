"""Cross-check final summaries and documentation without rewriting result files."""
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.benchmarks.ethics.religion_lm_v3_io import sha256, write_json


def main():
    base = ROOT / 'result/benchmarks/ethics/reports'
    report = base / 'religion_lm_v3_candidate_20260916_r4_overall'
    summary = base / 'religion_lm_v3_summary_20260916_r4'
    verification = base / 'religion_lm_v3_verification_20260916_r5/verification.json'
    output = base / 'religion_lm_v3_delivery_20260916_r4/final_checks.json'
    if output.exists():
        raise FileExistsError(output)
    directions = pd.read_csv(report / 'statistics/sensitivity_directions.csv.gz', dtype=str, keep_default_na=False)
    left = pd.to_numeric(directions.effect, errors='coerce')
    right = pd.to_numeric(directions.effect_primary, errors='coerce')
    evaluable = np.isfinite(left) & np.isfinite(right)
    reversed_effect = (left * right < 0) & evaluable
    assert directions.direction_evaluable.eq('True').equals(evaluable)
    assert directions.direction_reversed.eq('True').equals(reversed_effect)
    assert directions.direction_reversed.eq('').equals(~evaluable)
    directions['evaluated'] = evaluable.astype(int)
    directions['reversed_effect'] = reversed_effect.astype(int)
    actual = directions.groupby(['population', 'metric']).agg(total=('evaluated', 'size'),
        evaluable=('evaluated', 'sum'), direction_reversed=('reversed_effect', 'sum'))
    expected = pd.read_csv(summary / 'sensitivity_direction_summary.csv').set_index(['population', 'metric'])
    pd.testing.assert_frame_equal(actual.sort_index(), expected.sort_index(), check_dtype=False)
    ablation = pd.read_csv(summary / 'factorial_ablation_summary.csv').groupby('metric').value.sum()
    assert ablation['baseline_replay_mismatch_responses'] == 0
    assert ablation['new_surface_new_registry'] == 2865351
    scope = pd.read_csv(summary / 'scope_counts.csv')
    assert scope.occurrences.sum() == 2865351
    assert scope.loc[scope.speaker_role.eq('unresolved') & scope.discourse_scope.eq('unresolved'), 'occurrences'].sum() == 2731632
    contrasts = pd.read_csv(report / 'statistics/paired_contrasts.csv.gz', usecols=['ci_status', 'inference_scope_status'])
    assert len(contrasts) == 165672
    docs = [ROOT / 'docs' / name for name in ['religion_lm_v3_results.md', 'religion_lm_v3_execution_progress.md',
        'religion_lm_v3_completeness.md', 'religion_lm_v3_rule_audit.md', 'religion_lm_v3_codebook.md',
        'superpowers/plans/2026-09-15-religion-ethics-marker-coverage.md']]
    links = []
    for path in docs:
        text = path.read_text(encoding='utf-8')
        assert '\ufffd' not in text
        prose = re.sub(r'```.*?```', '', text, flags=re.S)
        prose = re.sub(r'`[^`]*`', '', prose)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', prose):
            if target.startswith(('https://', 'http://', '#')):
                continue
            target = target.split('#')[0]
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), (path, target)
            links.append(dict(document=str(path.relative_to(ROOT)), target=str(resolved.relative_to(ROOT))))
    verified = json.loads(verification.read_text(encoding='utf-8'))
    assert verified['status'] == 'automated_checks_passed_human_validation_pending'
    visual = json.loads((report / 'visual_review.json').read_text(encoding='utf-8'))
    assert len(visual['reviewed_images']) == 37
    navigation = json.loads((report / 'presentation_review_manifest.json').read_text(encoding='utf-8'))
    notebook = Path(navigation['workspace_notebook'])
    assert sha256(notebook) == navigation['workspace_notebook_sha256']
    preferred = report / navigation['preferred_report_notebook']
    assert sha256(preferred) == navigation['report_notebook_sha256']
    heldout = base / 'religion_lm_v3_precision_heldout_20260916_r1/manifest.json'
    precision = json.loads(heldout.read_text(encoding='utf-8'))
    assert precision['development_and_calibration_excluded'] and precision['human_reviews_completed'] == 0
    result = dict(status='candidate_delivery_checks_passed_human_validation_pending',
        summary_direction_flags_match_numeric_effects=True, sensitivity_summary=actual.reset_index().to_dict('records'),
        ablation_totals=ablation.to_dict(), ci_status_counts=contrasts.ci_status.value_counts().to_dict(),
        inference_scope_counts=contrasts.inference_scope_status.value_counts().to_dict(),
        document_links_checked=len(links), document_links=links, reviewed_figures=37,
        prior_full_suite=dict(tests=49, seconds=49.871, status='passed_recorded_execution'),
        subsequent_navigation_suite=dict(tests=3, new_tests=1, seconds=0.408, status='passed_recorded_execution'),
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in docs + [Path(__file__), verification,
            report/'visual_review.json', report/'presentation_review_manifest.json', heldout, notebook, preferred,
            ROOT/'src/benchmarks/ethics/rebuild_religion_v3_candidate.py']})
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['document_links','source_hashes']}, indent=2))


if __name__ == '__main__':
    main()
