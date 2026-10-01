"""Reuse must reject changed detectors and non-verdict surface changes."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from src.benchmarks.ethics.gender_lm_v2 import revision
from src.benchmarks.ethics.gender_lm_v2.revision import check_reuse, check_transition


class RevisionTests(unittest.TestCase):
    def test_unbound_reviewed_evidence_rejected_before_output_creation(self):
        def read(path, **kwargs):
            if path.name == 'extraction_manifest.json':
                return json.dumps({'registry_sha256': 'same'})
            if path.name == 'analysis_input_provenance.json':
                return json.dumps({'instrument_hashes': {}, 'artifacts': {'original.csv.gz': 'hash'}})
            return '[]'
        with patch.object(Path, 'read_text', read), patch.object(revision.run, 'instrument_hashes', return_value={}), \
                patch.object(revision, 'sha256', return_value='same'), patch.object(revision.run, 'load_registry', return_value=[]), \
                patch.object(revision.run, 'evidence_path', return_value=Path('reviewed.csv.gz')), \
                patch.object(revision, 'reserve_output') as reserve:
            with self.assertRaisesRegex(ValueError, 'Selected parent evidence'):
                revision.revise(Path('parent'), Path('output'))
            reserve.assert_not_called()

    def test_only_surface_hash_may_change(self):
        check_reuse({'surfaces.py': 'old', 'detectors.py': 'same'},
                    {'surfaces.py': 'new', 'detectors.py': 'same'})
        with self.assertRaises(ValueError):
            check_reuse({'surfaces.py': 'old', 'detectors.py': 'old'},
                        {'surfaces.py': 'new', 'detectors.py': 'new'})

    def test_missing_instrument_entry_is_rejected(self):
        with self.assertRaises(ValueError):
            check_reuse({'surfaces.py': 'old'}, {'surfaces.py': 'new', 'detectors.py': 'new'})

    def test_revision_is_limited_to_pure_verdict_reclassification(self):
        check_transition('substantive_explanation', 'verdict_only')
        for before, after in [('missing', 'verdict_only'), ('substantive_explanation', 'substantive_explanation'),
                              ('substantive_explanation', 'technical_only')]:
            with self.assertRaises(ValueError):
                check_transition(before, after)
