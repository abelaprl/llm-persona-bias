"""Read-only report verification, with exclusive creation of a new JSON record."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


class ReportHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.questions = 0
        self.source_responses = 0
        self.placeholder_responses = 0
        self.persona_counts = {}
        self._persona = None
        self._has_source_response = False
        self._summary = None

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        self.links.extend(attributes[name] for name in ("href", "src") if attributes.get(name))
        if tag == "section" and "data-question" in attributes:
            self.questions += 1
        elif tag == "article":
            self._persona = attributes.get("data-persona", "unknown")
            self._has_source_response = False
        elif tag == "summary":
            self._summary = []

    def handle_data(self, text):
        if self._summary is not None:
            self._summary.append(text)

    def handle_endtag(self, tag):
        if tag == "summary" and self._summary is not None:
            if "".join(self._summary) == "Visible penuh":
                self._has_source_response = True
            self._summary = None
        elif tag == "article":
            if self._has_source_response:
                self.source_responses += 1
                self.persona_counts[self._persona] = self.persona_counts.get(self._persona, 0) + 1
            else:
                self.placeholder_responses += 1
            self._persona = None
            self._has_source_response = False


def resolve_local_link(root, page, link):
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    candidate = (page.parent / unquote(parsed.path)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Local link escapes report directory: {link}") from error
    return candidate


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_verification(path, report):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_report(root, registry_path, *, expected_figures=88, expected_markers=41,
                  expected_pages=306, expected_responses=81756, expected_links=967,
                  visual_reviewed=(), visual_note=""):
    root = Path(root).resolve()
    stage_manifest = root / "report_stage_manifest.json"
    if not stage_manifest.is_file():
        raise FileNotFoundError(f"Report is not ready: {stage_manifest}")
    stage = json.loads(stage_manifest.read_text(encoding="utf-8"))
    errors, hashes, checks = [], {}, []

    def check(name, actual, expected):
        passed = actual == expected
        checks.append(dict(check=name, actual=actual, expected=expected, passed=passed))
        if not passed:
            errors.append(f"{name}: expected {expected!r}, found {actual!r}")

    def contained(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Indexed asset escapes report: {relative}")
        return path

    figures = read_csv(root / "figure_index.csv")
    evidence = read_csv(root / "evidence_index.csv")
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if isinstance(registry, dict):
        registry = registry["markers"]
    registered = {row["marker_id"] for row in registry}
    check("stage_manifest_figures", stage.get("figures"), expected_figures)
    check("stage_status", stage.get("status"), "candidate")
    check("figure_index_rows", len(figures), expected_figures)
    check("unique_figure_ids", len({row["figure_id"] for row in figures}), expected_figures)
    check("figure_default_scope", sorted({row["scope"] for row in figures}), ["overall"])
    check("figure_audit_status", sorted({row["audit_status"] for row in figures}), ["candidate"])
    atlas_markers = set()
    for index, row in enumerate(figures, 1):
        for field in ("png_path", "svg_path", "source_csv"):
            relative = row[field]
            path = contained(relative)
            if not path.is_file():
                errors.append(f"Missing indexed asset: {relative}")
                continue
            digest = sha256_file(path)
            hashes[relative] = digest
            expected_hash = row.get("source_sha256" if field == "source_csv" else "png_sha256" if field == "png_path" else "svg_sha256")
            if field in ("png_path", "source_csv") and not expected_hash:
                errors.append(f"Missing expected SHA256 in figure index: {relative}")
            if expected_hash and digest != expected_hash:
                errors.append(f"Hash mismatch: {relative}")
            if field == "png_path":
                with path.open("rb") as stream:
                    if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                        errors.append(f"Invalid PNG signature: {relative}")
            elif field == "svg_path":
                try:
                    if not ET.parse(path).getroot().tag.endswith("svg"):
                        errors.append(f"Invalid SVG root: {relative}")
                except ET.ParseError:
                    errors.append(f"Invalid SVG XML: {relative}")
        if row["figure_kind"] == "marker_atlas" and contained(row["source_csv"]).is_file():
            atlas_markers.update(item["marker_id"] for item in read_csv(contained(row["source_csv"])))
        if index % 25 == 0:
            print(f"Asset verification: {index}/{len(figures)} figures", flush=True)
    for suffix in ("png", "svg", "csv"):
        check(f"figure_{suffix}_files", len(list((root / "figures").glob(f"*.{suffix}"))), expected_figures)
    check("registered_marker_count", len(registered), expected_markers)
    check("atlas_marker_count", len(atlas_markers), expected_markers)
    check("atlas_marker_ids", sorted(atlas_markers), sorted(registered))
    check("evidence_index_pages", len(evidence), expected_pages)
    check("evidence_html_files", len(list((root / "evidence").glob("*.html"))), expected_pages)
    check("evidence_index_source_responses", sum(int(row["n_responses"]) for row in evidence), expected_responses)
    check("evidence_models", len({row["model_id"] for row in evidence}), 6)
    check("evidence_subsets", sorted({row["subset"] for row in evidence}), ["commonsense", "deontology", "justice", "virtue"])
    evidence_by_path = {row["path"]: row for row in evidence}
    check("evidence_unique_paths", len(evidence_by_path), expected_pages)
    pages = ["report.html", "evidence_explorer.html"] + list(evidence_by_path)
    link_count, response_count, placeholder_count, question_count = 0, 0, 0, 0
    persona_counts = {}
    for index, relative in enumerate(pages, 1):
        path = contained(relative)
        if not path.is_file():
            errors.append(f"Missing HTML: {relative}")
            continue
        parser = ReportHTMLParser()
        try:
            with path.open(encoding="utf-8") as stream:
                for block in iter(lambda: stream.read(256 * 1024), ""):
                    parser.feed(block)
            parser.close()
        except UnicodeDecodeError:
            errors.append(f"Invalid UTF-8 HTML: {relative}")
            continue
        for link in parser.links:
            try:
                target = resolve_local_link(root, path, link)
            except ValueError as error:
                errors.append(f"{relative}: {error}")
                continue
            if target is not None:
                link_count += 1
                if not target.is_file():
                    errors.append(f"Broken local link in {relative}: {link}")
        if relative in evidence_by_path:
            record = evidence_by_path[relative]
            if parser.source_responses != int(record["n_responses"]):
                errors.append(f"HTML response count mismatch: {relative}")
            if parser.questions != int(record["n_questions"]):
                errors.append(f"HTML question count mismatch: {relative}")
            response_count += parser.source_responses
            placeholder_count += parser.placeholder_responses
            question_count += parser.questions
            for persona, count in parser.persona_counts.items():
                persona_counts[persona] = persona_counts.get(persona, 0) + count
        if index % 50 == 0:
            print(f"HTML verification: {index}/{len(pages)} pages", flush=True)
    check("local_link_occurrences_href_and_src", link_count, expected_links)
    check("html_source_responses", response_count, expected_responses)
    check("html_placeholder_responses", placeholder_count, 0)
    check("html_personas", len(persona_counts), 9)
    check("html_questions", question_count, sum(int(row["n_questions"]) for row in evidence))
    expected_scopes = ["commonsense", "deontology", "justice", "overall", "virtue"]
    for filename in ("marker_summary.csv", "response_length_summary.csv", "report_status_coverage.csv"):
        rows = read_csv(root / filename)
        check(filename + "_scopes", sorted({row["scope"] for row in rows}), expected_scopes)
        hashes[filename] = sha256_file(root / filename)
    visual_review = []
    png_paths = {row["png_path"] for row in figures}
    for relative in visual_reviewed:
        if relative not in png_paths:
            errors.append(f"Visual review path is not an indexed PNG: {relative}")
        else:
            visual_review.append(dict(path=relative, sha256=hashes.get(relative),
                                      reviewer_type="agent_visual_inspection", note=visual_note))
    for filename in ("figure_index.csv", "evidence_index.csv", "report_stage_manifest.json"):
        hashes[filename] = sha256_file(root / filename)
    return dict(status="pass" if not errors else "fail", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                report_root=str(root), checks=checks, errors=errors, figure_count=len(figures),
                evidence_pages=len(evidence), html_source_responses=response_count,
                html_persona_counts=persona_counts, local_link_occurrences=link_count,
                atlas_marker_ids=sorted(atlas_markers), artifact_sha256=hashes,
                registry_sha256=sha256_file(registry_path), verifier_sha256=sha256_file(__file__),
                svg_hash_note="SVG SHA256 recorded here; figure index does not provide an expected SVG hash unless stated.",
                visual_review=visual_review,
                limitations=["Checks cover artifact consistency, not detector precision/recall or semantic validity.",
                             "Visual inspection is by the agent; it is not human annotation or an audit promotion."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--registry", type=Path, default=Path(__file__).resolve().parent.parent / "registry.json")
    parser.add_argument("--check-only", action="store_true", help="Read-only validation; write no JSON.")
    parser.add_argument("--visual-reviewed", action="append", default=[], help="Indexed PNG already inspected by the agent.")
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()
    target = args.report_dir / "verification_report.json"
    if not args.check_only and target.exists():
        raise FileExistsError(target)
    report = verify_report(args.report_dir, args.registry, visual_reviewed=args.visual_reviewed,
                           visual_note=args.visual_note)
    if not args.check_only:
        write_verification(target, report)
    print(json.dumps({key: report[key] for key in ("status", "figure_count", "evidence_pages", "html_source_responses", "local_link_occurrences", "errors")}, ensure_ascii=True))
    return 0 if report["status"] == "pass" else 1


class VerifierTests(unittest.TestCase):
    def test_html_counts_links_and_source_panels_without_parsing_escaped_text(self):
        parser = ReportHTMLParser()
        parser.feed('<a href="../evidence_explorer.html">Back</a><img src="plot.png">'
                    '<section data-question="q1"><article data-persona="neutral">'
                    '<details><summary>Visible penuh</summary><pre>&lt;a href="fake"&gt;</pre></details>'
                    '</article><article data-persona="man">unavailable</article></section>')
        self.assertEqual(parser.links, ["../evidence_explorer.html", "plot.png"])
        self.assertEqual(parser.source_responses, 1)
        self.assertEqual(parser.placeholder_responses, 1)
        self.assertEqual(parser.questions, 1)

    def test_local_link_resolution_stays_inside_report(self):
        root = Path.cwd() / "sample_report"
        page = root / "evidence" / "page.html"
        self.assertEqual(resolve_local_link(root, page, "../report.html#x"), root / "report.html")
        self.assertIsNone(resolve_local_link(root, page, "https://example.com/"))
        with self.assertRaises(ValueError):
            resolve_local_link(root, page, "../../outside.txt")

    def test_hash_check_detects_tampering_and_never_overwrites_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "source.csv"
            asset.write_bytes(b"value\n1\n")
            expected = sha256_file(asset)
            asset.write_bytes(b"value\n2\n")
            self.assertNotEqual(sha256_file(asset), expected)
            target = root / "verification_report.json"
            write_verification(target, {"status": "pass"})
            with self.assertRaises(FileExistsError):
                write_verification(target, {"status": "changed"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["status"], "pass")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        raise SystemExit(main())
