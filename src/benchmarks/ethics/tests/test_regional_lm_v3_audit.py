import hashlib
import json
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.benchmarks.ethics.regional_lm_v3_audit import (
    inventory,
    import_audit,
    make_audit_packets,
    rule_gate,
    summarize_audits,
)


class RegionalAuditTest(unittest.TestCase):
    @contextmanager
    def _tempdir(self):
        base = (Path.cwd() / "src" / "benchmarks" / "ethics" / "tests" / ".regional_audit_tmp").resolve()
        base.mkdir(mode=0o777, parents=True, exist_ok=True)
        root = (base / uuid4().hex).resolve()
        if base not in root.parents:
            raise RuntimeError("Temporary test path escaped its base")
        root.mkdir(mode=0o777)
        try:
            yield root
        finally:
            if root.exists() and base in root.resolve().parents:
                shutil.rmtree(root)

    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        rows = []
        subsets = ["commonsense"] * 3 + ["deontology"] * 3 + ["justice"] * 2 + ["virtue"] * 2
        for model in ("m1", "m2"):
            for persona in ("neutral", "east_asian_explicit"):
                for i, subset in enumerate(subsets):
                    qid = f"ethics_{subset}_{i:04d}"
                    rows.append(
                        {
                            "model": model,
                            "persona_folder": persona,
                            "question_id": qid,
                            "subset": subset,
                            "response_content_full": (
                                "As a helpful AI assistant, ask me if you will. "
                                "Customs vary.\n- **Reason:** evidence."
                            ),
                            "answer_status": "committed" if i else "abstain",
                            "extraction_confidence": "semantic_unaudited" if i == 1 else "explicit",
                        }
                    )
        source = root / "source.csv"
        pd.DataFrame(rows).to_csv(source, index=False)

        frozen = root / "response_level.csv.gz"
        frozen_rows = []
        for row in rows:
            frozen_rows.append(
                {
                    "model_id": row["model"],
                    "persona": row["persona_folder"],
                    "question_id": row["question_id"],
                    "subset": row["subset"],
                    "answer_status": row["answer_status"],
                    "text_status": "technical_only" if row["question_id"].endswith("0000") else "available",
                    "text_available": not row["question_id"].endswith("0000"),
                    "marker_a__count": 1,
                    "marker_a__density": 1.0,
                    "marker_a__present": 1,
                }
            )
        pd.DataFrame(frozen_rows).to_csv(frozen, index=False, compression="gzip")

        hits = root / "marker_hits.csv.gz"
        hit_rows = []
        for row in rows:
            if not row["question_id"].endswith("0000"):
                key = f'{row["model"]}|{row["persona_folder"]}|{row["question_id"]}'
                hit_rows.append(
                    {
                        "response_key": key,
                        "marker_id": "marker_a",
                        "rule_id": "marker_a:p0",
                        "surface": "final",
                        "start": 0,
                        "end": 2,
                        "matched_text": "As",
                        "decision": "accepted",
                        "reason": "pattern",
                        "rule_version": "v1",
                    }
                )
        first = rows[0]
        hit_rows.append(
            {
                "response_key": f'{first["model"]}|{first["persona_folder"]}|{first["question_id"]}',
                "marker_id": "marker_zero",
                "rule_id": "marker_zero:p0",
                "surface": "final",
                "start": 1,
                "end": 12,
                "matched_text": "if you will",
                "decision": "rejected",
                "reason": "idiomatic_if_you_will",
                "rule_version": "v1",
                "exclusion_rule_id": "connective_conditional:exclude_if_you_will",
            }
        )
        pd.DataFrame(hit_rows).to_csv(hits, index=False, compression="gzip")
        (root / "registry.json").write_text(
            json.dumps(
                [
                    {
                        "marker_id": "marker_a",
                        "patterns": [{"pattern_index": 0, "pattern": "As"}],
                        "audit_class": "explicit",
                    },
                    {
                        "marker_id": "marker_zero",
                        "patterns": [{"pattern_index": 0, "pattern": "never"}],
                        "audit_class": "semantic",
                        "candidate_version": "v3",
                        "candidate_exclusions": [
                            {"rule_id": "connective_conditional:exclude_if_you_will", "pattern": "if you will"}
                        ],
                    },
                    {
                        "marker_id": "marker_never",
                        "patterns": [{"pattern_index": 0, "pattern": "absent"}],
                        "audit_class": "semantic",
                    },
                ]
            ),
            encoding="utf-8",
        )
        exposure = root / "old_audit.csv"
        pd.DataFrame(
            [
                {
                    "model_id": "m1",
                    "persona": "neutral",
                    "question_id": "ethics_deontology_0003",
                    "codex_review_label": "TP",
                },
                {
                    "model_id": "m2",
                    "persona": "east_asian_explicit",
                    "question_id": "ethics_justice_0006",
                    "codex_review_label": "",
                },
            ]
        ).to_csv(exposure, index=False)
        return source, frozen, exposure

    def test_inventory_joins_metadata_and_materializes_zero_rules(self):
        with self._tempdir() as root:
            source, frozen, _ = self._fixture(root)
            out = root / "inventory_out"
            out.mkdir()
            shutil.copy2(frozen, out / "response_level.csv.gz")
            shutil.copy2(root / "marker_hits.csv.gz", out / "marker_hits.csv.gz")
            shutil.copy2(root / "registry.json", out / "registry.json")
            before = hashlib.sha256((out / "response_level.csv.gz").read_bytes()).hexdigest()
            result = inventory(source, frozen, out)
            after = hashlib.sha256((out / "response_level.csv.gz").read_bytes()).hexdigest()
            response = pd.read_csv(result["audit_response_index_path"])
            inv = pd.read_csv(result["inventory_path"])
            self.assertEqual(before, after)
            self.assertEqual(len(response), 40)
            self.assertIn("response_key", response)
            self.assertIn("answer_status_source", response)
            self.assertNotIn("extraction_confidence_x", response)
            self.assertNotIn("extraction_confidence_y", response)
            demoted = response[response["extraction_confidence_source"] == "semantic_unaudited"]
            self.assertTrue((demoted["analysis_answer_status"] == "abstain").all())
            zero = inv[inv["rule_id"] == "marker_never:p0"]
            self.assertEqual(len(zero), 16)
            self.assertEqual(int(zero["population_hit_count"].sum()), 0)
            excluded = inv[inv["rule_id"] == "marker_zero:p0"]
            self.assertEqual(int(excluded["population_hit_count"].sum()), 1)
            self.assertEqual(int(excluded["rejected_hit_count"].sum()), 1)
            exclusion_rule = inv[inv["rule_id"] == "connective_conditional:exclude_if_you_will"]
            self.assertEqual(int(exclusion_rule["population_hit_count"].sum()), 1)
            self.assertEqual(int(exclusion_rule["rejected_hit_count"].sum()), 1)
            self.assertIn("technical_only", inv.columns)

    def test_inventory_reads_frozen_csv_registry_patterns(self):
        with self._tempdir() as root:
            source, frozen, _ = self._fixture(root)
            (root / "registry.json").unlink()
            pd.DataFrame(
                [
                    {"marker_id": "marker_a", "patterns": json.dumps([{"pattern": "As"}]), "version": "v1"},
                    {"marker_id": "marker_zero", "patterns": json.dumps([{"pattern": "never"}]), "version": "v1"},
                    {"marker_id": "marker_never", "patterns": json.dumps([{"pattern": "absent"}]), "version": "v1"},
                ]
            ).to_csv(root / "marker_registry.csv", index=False)
            out = root / "inventory_out"
            out.mkdir()
            shutil.copy2(frozen, out / "response_level.csv.gz")
            shutil.copy2(root / "marker_hits.csv.gz", out / "marker_hits.csv.gz")
            result = inventory(source, frozen, out)
            rules = pd.read_csv(result["rules_path"])
            self.assertEqual(set(rules["rule_id"]), {"marker_a:p0", "marker_zero:p0", "marker_never:p0"})

    def test_packets_are_seeded_group_partitioned_and_blinded(self):
        with self._tempdir() as root:
            source, frozen, exposure = self._fixture(root)
            for name in ("a", "b"):
                run = root / name
                run.mkdir()
                shutil.copy2(frozen, run / "response_level.csv.gz")
                shutil.copy2(root / "marker_hits.csv.gz", run / "marker_hits.csv.gz")
                shutil.copy2(root / "registry.json", run / "registry.json")
                inventory(source, frozen, run)
            (root / "marker_hits.csv.gz").unlink()
            (root / "registry.json").unlink()
            first = make_audit_packets(source, exposure, root / "a", seed=7)
            second = make_audit_packets(source, exposure, root / "b", seed=7)
            a = pd.read_csv(first["recall_path"])
            b = pd.read_csv(second["recall_path"])
            self.assertEqual(a["audit_id"].tolist(), b["audit_id"].tolist())
            self.assertNotIn("model_id", a.columns)
            self.assertNotIn("persona", a.columns)
            self.assertNotIn("answer_status", a.columns)
            self.assertFalse(any(column.endswith("__count") for column in a.columns))
            index = pd.read_csv(first["recall_index_path"])
            partition = pd.read_csv(first["partition_path"])
            self.assertFalse(set(partition.query("audit_partition == 'development'")["question_id"]) & set(partition.query("audit_partition == 'evaluation'")["question_id"]))
            self.assertNotIn("ethics_deontology_0003", set(index["question_id"]))
            reviewed_partition = partition.set_index("question_id")
            self.assertEqual(reviewed_partition.loc["ethics_deontology_0003", "partition_reason"], "recorded_review_or_readback")
            self.assertTrue(reviewed_partition.loc["ethics_justice_0006", "partition_reason"].startswith("no_recorded_review_seeded_"))
            manifest = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
            self.assertIn("sampling", manifest)
            self.assertIn("deficits", manifest)
            self.assertEqual(manifest["exposure_ledger"]["questions_with_recorded_review_or_readback"], 3)
            self.assertEqual(manifest["exposure_ledger"]["questions_selected_only_unfilled"], 1)
            self.assertTrue(Path(first["targeted_path"]).exists())
            self.assertTrue(Path(first["eligibility_path"]).exists())

    def test_import_audit_rejects_invented_ids_and_ground_truth(self):
        with self._tempdir() as root:
            text_hash = hashlib.sha256(b"text").hexdigest()
            pd.DataFrame([{"audit_id": "PRE-known", "response_key": "m|p|q", "rule_id": "r", "rule_version": "v1", "surface": "visible", "start": 0, "end": 1, "audit_partition": "evaluation", "full_response_sha256": text_hash}]).to_csv(
                root / "audit_precision_index.csv", index=False
            )
            pd.DataFrame([{"rule_id": "r", "rule_version": "v1"}]).to_csv(root / "rules.csv", index=False)
            valid = root / "review.csv"
            pd.DataFrame(
                [{"audit_id": "PRE-known", "rule_id": "r", "rule_version": "v1", "full_response": "text", "label": "TP", "reviewer_kind": "human", "reviewer": "h1"}]
            ).to_csv(valid, index=False)
            result = import_audit(valid, root)
            self.assertEqual(result["rows"], 1)
            self.assertTrue(Path(result["audit_path"]).exists())
            imported = pd.read_csv(result["audit_path"])
            self.assertEqual((imported.loc[0, "surface"], imported.loc[0, "start"], imported.loc[0, "end"]), ("visible", 0, 1))

            tampered = root / "tampered.csv"
            pd.DataFrame(
                [{"audit_id": "PRE-known", "rule_id": "r", "rule_version": "v1", "start": 0, "end": 1, "full_response": "changed", "label": "TP", "reviewer_kind": "human", "reviewer": "h2"}]
            ).to_csv(tampered, index=False)
            with self.assertRaises(ValueError):
                import_audit(tampered, root)

            bad = root / "bad.csv"
            pd.DataFrame(
                [{"audit_id": "PRE-invented", "rule_id": "r", "rule_version": "v1", "start": 0, "end": 1, "full_response": "tampered", "label": "TP", "reviewer_kind": "human", "reviewer": "h1", "ground_truth": 1}]
            ).to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                import_audit(bad, root / "bad_import")

    def test_summary_without_import_preserves_every_rule_id(self):
        with self._tempdir() as root:
            pd.DataFrame(
                [
                    {"marker_id": "a", "rule_id": "a:p0", "audit_class": "explicit", "rule_version": "v1"},
                    {"marker_id": "b", "rule_id": "b:p0", "audit_class": "semantic", "rule_version": "v1"},
                    {"marker_id": "b", "rule_id": "connective_conditional:exclude_if_you_will", "audit_class": "semantic", "rule_version": "v3"},
                ]
            ).to_csv(root / "rules.csv", index=False)
            pd.DataFrame(
                [
                    {"rule_id": "a:p0", "population_hit_count": 2, "accepted_hit_count": 2, "rejected_hit_count": 0},
                    {"rule_id": "b:p0", "population_hit_count": 0, "accepted_hit_count": 0, "rejected_hit_count": 0},
                    {"rule_id": "connective_conditional:exclude_if_you_will", "population_hit_count": 2, "accepted_hit_count": 0, "rejected_hit_count": 2},
                ]
            ).to_csv(root / "inventory.csv", index=False)
            (root / "diagnostic_labels.json").write_text(
                json.dumps({"labels": [{"rule_id": "a:p0", "label": "TP", "response_key": "m|p|q"}]}), encoding="utf-8"
            )
            (root / "idiom_census.json").write_text(
                json.dumps({"responses": [{"rule_id": "connective_conditional:exclude_if_you_will", "label": "TP"}] * 2}), encoding="utf-8"
            )
            result = summarize_audits(root)
            summary = pd.read_csv(result["path"])
            self.assertEqual(summary["rule_id"].tolist(), ["a:p0", "b:p0", "connective_conditional:exclude_if_you_will"])
            self.assertTrue(summary["rule_id"].notna().all())
            self.assertTrue(summary["rule_id"].is_unique)
            exclusion = summary[summary["rule_id"] == "connective_conditional:exclude_if_you_will"].iloc[0]
            self.assertEqual(exclusion["development_n_reviewed"], 2)
            self.assertFalse(bool(exclusion["release_eligible"]))

    def test_rule_gate_keeps_ai_and_human_gates_separate(self):
        rows = [
            {
                "rule_id": "r",
                "response_key": str(i),
                "start": i,
                "end": i + 1,
                "label": "TP",
                "reviewer_kind": "ai",
                "reviewer": "codex",
                "audit_partition": "evaluation",
                "rule_version": "v1",
            }
            for i in range(100)
        ]
        result = rule_gate(rows, semantic=True, population_hits=1000)
        self.assertTrue(result["quota_met"])
        self.assertEqual(result["n_hits_population"], 1000)
        self.assertEqual(result["ai_review_status"], "passed_zero_fp")
        self.assertEqual(result["human_review_status"], "not_reviewed")
        self.assertFalse(result["release_eligible"])

    def test_rule_gate_rejects_duplicates_uncertain_and_mixed_versions(self):
        base = {
            "rule_id": "r",
            "response_key": "k",
            "start": 0,
            "end": 1,
            "reviewer_kind": "human",
            "reviewer": "h1",
            "audit_partition": "evaluation",
        }
        rows = [dict(base, label="TP", rule_version="v1"), dict(base, label="uncertain", rule_version="v2")]
        result = rule_gate(rows, semantic=False)
        self.assertFalse(result["release_eligible"])
        self.assertIn("population_hits_unknown", result["blockers"])
        self.assertIn("mixed_rule_versions", result["blockers"])
        self.assertIn("duplicate_labels", result["blockers"])
        self.assertIn("uncertain_labels", result["blockers"])

    def test_rule_gate_counts_unique_hits_and_requires_two_reviews_per_hit(self):
        rows = []
        for i in range(50):
            for reviewer in ("h1", "h2"):
                rows.append(
                    {
                        "rule_id": "r", "response_key": str(i), "start": i, "end": i + 1,
                        "label": "TP", "reviewer_kind": "human", "reviewer": reviewer,
                        "audit_partition": "evaluation", "rule_version": "v1",
                    }
                )
        passed = rule_gate(rows, semantic=False, population_hits=50)
        self.assertEqual(passed["n_reviewed"], 50)
        self.assertTrue(passed["release_eligible"])

        partial = rows[:2] + [
            {
                "rule_id": "r", "response_key": str(i), "start": i, "end": i + 1,
                "label": "TP", "reviewer_kind": "ai", "reviewer": "codex",
                "audit_partition": "evaluation", "rule_version": "v1",
            }
            for i in range(1, 50)
        ]
        failed = rule_gate(partial, semantic=False, population_hits=50)
        self.assertTrue(failed["quota_met"])
        self.assertFalse(failed["release_eligible"])
        self.assertEqual(failed["human_review_status"], "incomplete_two_reviewer_coverage")


if __name__ == "__main__":
    unittest.main()
