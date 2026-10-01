"""Instrument snapshots must bind extraction, not only later report execution."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.benchmarks.ethics.gender_lm_v2 import run


REGISTRY = [{"marker_id": "test_marker", "patterns": [{"rule_id": "test_marker.r1"}]}]


class ExtractionProvenanceTests(unittest.TestCase):
    def extraction(self, hashes, registry_hashes=None):
        source, output = Path("synthetic_source.csv"), Path("synthetic_output")
        config = {"source": str(source), "sha256": "source_hash", "expected_rows": 1}
        record = {"response_id": "r1", "model_id": "model", "persona": "neutral", "subset": "justice",
                  "question_id": "q1", "file": "fixture", "response_content_full": "Full explanation.",
                  "text_status": "substantive_explanation", "raw_text": "Full explanation.",
                  "visible_text": "Full explanation.", "explanation_text": "Full explanation.",
                  "structure_text": "Full explanation."}
        captured = {}
        registry_versions = iter(registry_hashes or ["source_hash"] * 4)
        def artifact_hash(path):
            return next(registry_versions) if path.name == "registry.json" else "source_hash"
        with patch.object(run, "instrument_hashes", side_effect=hashes) as snapshot, \
                patch.object(run, "sha256", side_effect=artifact_hash), \
                patch.object(run, "load_source", return_value=pd.DataFrame([record])), \
                patch.object(run, "load_registry", return_value=REGISTRY), \
                patch.object(Path, "read_text", return_value=json.dumps(config)), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(run.gzip, "open", side_effect=lambda *args, **kwargs: io.StringIO()), \
                patch.object(run, "extract_batch", return_value=[(record, [])]) as worker, \
                patch.object(run, "write_json_new", side_effect=lambda path, data: captured.update({path.name: data})):
            def checked_worker(rows):
                self.assertEqual(snapshot.call_count, 1, "Snapshot must precede response extraction")
                return [(record, [])]
            worker.side_effect = checked_worker
            result = run.extract(source, output, workers=1)
        return result, captured

    def test_snapshot_captured_before_extraction_and_persisted(self):
        expected = {"surfaces.py": "original"}
        result, files = self.extraction([expected, expected])
        self.assertEqual(result["instrument_hashes_at_extraction"], expected)
        self.assertEqual(files["extraction_manifest.json"]["instrument_hashes_at_extraction"], expected)

    def test_code_change_during_extraction_rejects_completion(self):
        with self.assertRaisesRegex(ValueError, "[Ii]nstrument|[Ss]urface|[Dd]etector"):
            self.extraction([{"surfaces.py": "before"}, {"surfaces.py": "after"}])

    def test_registry_change_during_extraction_rejects_completion(self):
        stable = {"surfaces.py": "same"}
        with self.assertRaisesRegex(ValueError, "Registry"):
            self.extraction([stable, stable], ["registry_before", "registry_after"])

    def verification(self, manifest, current):
        def read(path, *args, **kwargs):
            return json.dumps(manifest if path.name == "extraction_manifest.json" else REGISTRY)
        with patch.object(Path, "read_text", read), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(run, "sha256", return_value="registry_hash"), \
                patch.object(run, "load_registry", return_value=REGISTRY), \
                patch.object(run, "instrument_hashes", return_value=current), \
                patch.object(run.importlib.metadata, "version", return_value="fixture-version"):
            return run.verify_extraction(Path("synthetic_output"))

    def test_later_code_change_is_rejected_even_when_registry_matches(self):
        manifest = {"registry_sha256": "registry_hash", "source_sha256": "source_hash",
                    "instrument_hashes_at_extraction": {"surfaces.py": "before"}}
        with self.assertRaisesRegex(ValueError, "extraction"):
            self.verification(manifest, {"surfaces.py": "after"})

    def test_historical_missing_snapshot_is_explicit_not_retrofitted(self):
        manifest = {"registry_sha256": "registry_hash", "source_sha256": "source_hash"}
        current = {"surfaces.py": "current"}
        result = self.verification(manifest, current)
        self.assertEqual(result["extraction_instrument_hash_status"], "historical_hash_unavailable")
        self.assertIsNone(result["instrument_hashes_at_extraction"])
        self.assertEqual(result["instrument_hashes"], current)

    def test_matching_snapshot_remains_bound(self):
        current = {"surfaces.py": "same"}
        manifest = {"registry_sha256": "registry_hash", "source_sha256": "source_hash",
                    "instrument_hashes_at_extraction": current}
        result = self.verification(manifest, current)
        self.assertEqual(result["extraction_instrument_hash_status"], "verified_against_extraction")
        self.assertEqual(result["instrument_hashes_at_extraction"], current)


if __name__ == "__main__":
    unittest.main()
