"""Read-only audit for DeepSeek GSM8K Religion and Gender persona results.

The auditor validates exact 1..1319 filename coverage per persona, JSON/schema
integrity, warmups, HTTP 4XX/5XX errors, unsuccessful responses, and empty final
or reasoning output. It never writes experiment files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import tarfile
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SLUG = "deepseek_deepseek-r1-distill-qwen-32b"
DEFAULT_ROOT = REPO_ROOT / "result" / "GMS8K"
EXPECTED_COUNT = 1319
GROUPS: dict[str, tuple[str, ...]] = {
    "Religion_Persona": (
        "atheist_explicit",
        "atheist_implicit_1",
        "atheist_implicit_2",
        "atheist_implicit_3",
        "christian_explicit",
        "christian_implicit_1",
        "christian_implicit_2",
        "christian_implicit_3",
        "islam_explicit",
        "islam_implicit_1",
        "islam_implicit_2",
        "islam_implicit_3",
        "jewish_explicit",
        "jewish_implicit_1",
        "jewish_implicit_2",
        "jewish_implicit_3",
    ),
    "Gender_Persona": (
        "man_explicit_1",
        "man_implicit_1",
        "man_implicit_2",
        "man_implicit_3",
        "neutral",
        "woman_explicit_1",
        "woman_implicit_1",
        "woman_implicit_2",
        "woman_implicit_3",
    ),
}
HTTP_RE = re.compile(r"\bHTTP\s*([45]\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class HttpIssue:
    code: int
    group: str
    persona: str
    file: str
    message: str
    success: bool | None
    final_empty: bool
    all_output_empty: bool


@dataclass(frozen=True)
class ResponseIssue:
    group: str
    persona: str
    file: str
    error: str
    success: bool | None
    finish_reason: str
    final_empty: bool
    all_output_empty: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--groups",
        default="Religion_Persona,Gender_Persona",
        help="Comma-separated group folder names.",
    )
    parser.add_argument("--detail-limit", type=int, default=1000)
    parser.add_argument(
        "--manifest-dir",
        default="",
        help="Optional directory for derived HTTP/non-HTTP/invalid rerun CSV manifests.",
    )
    parser.add_argument(
        "--source",
        choices=("worktree", "git-head"),
        default="worktree",
        help=(
            "Read individual worktree files, or stream tracked blobs from Git HEAD. "
            "Use git-head only after confirming the selected folders have no local changes."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )
    args = parser.parse_args(argv)
    args.root = Path(args.root).expanduser()
    if not args.root.is_absolute():
        args.root = REPO_ROOT / args.root
    args.groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
    unknown = sorted(set(args.groups) - set(GROUPS))
    if unknown:
        parser.error(f"unknown group(s): {unknown}; available: {sorted(GROUPS)}")
    if args.detail_limit < 0:
        parser.error("--detail-limit must be zero or greater")
    if args.manifest_dir:
        args.manifest_dir = Path(args.manifest_dir).expanduser()
        if not args.manifest_dir.is_absolute():
            args.manifest_dir = REPO_ROOT / args.manifest_dir
    return args


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON is not an object")
    return payload


def text_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def first_choice(response: dict[str, Any]) -> dict[str, Any]:
    raw_api = response.get("raw_api_response")
    choices = raw_api.get("choices") if isinstance(raw_api, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def response_texts(response: dict[str, Any]) -> tuple[str, str, str]:
    choice = first_choice(response)
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    final_text = text_value(response.get("content")) or text_value(message.get("content"))
    reasoning = ""
    for key in ("reasoning", "reasoning_content", "thinking", "analysis"):
        reasoning = text_value(message.get(key))
        if reasoning:
            break
    if not reasoning:
        details = message.get("reasoning_details")
        reasoning = text_value(details)
    finish_reason = str(choice.get("finish_reason") or "")
    return final_text, reasoning, finish_reason


def http_error(response: dict[str, Any]) -> tuple[int | None, str]:
    raw_api = response.get("raw_api_response")
    raw_error = raw_api.get("error") if isinstance(raw_api, dict) else None
    if isinstance(raw_error, dict):
        try:
            code = int(raw_error.get("code"))
        except (TypeError, ValueError):
            code = None
        if code is not None and 400 <= code <= 599:
            return code, str(raw_error.get("message") or "")

    error_text = str(response.get("error") or "")
    match = HTTP_RE.search(error_text)
    if match:
        return int(match.group(1)), error_text
    return None, ""


def finish_result(
    args: argparse.Namespace,
    totals: Counter[str],
    persona_rows: list[dict[str, Any]],
    http_issues: list[HttpIssue],
    non_http_failures: list[ResponseIssue],
    empty_outputs: list[ResponseIssue],
    invalid_files: list[dict[str, Any]],
    id_mismatches: list[dict[str, object]],
    persona_mismatches: list[dict[str, object]],
    warmup_issues: list[dict[str, str]],
) -> dict[str, Any]:
    if args.manifest_dir:
        write_manifests(
            args.manifest_dir,
            http_issues,
            non_http_failures,
            invalid_files,
        )
    http_code_counts = Counter(issue.code for issue in http_issues)
    http_message_counts = Counter(
        (issue.code, issue.message) for issue in http_issues
    )
    group_counts = Counter(issue.group for issue in http_issues)
    limit = args.detail_limit
    return {
        "source": args.source,
        "model_slug": MODEL_SLUG,
        "groups": list(args.groups),
        "expected_per_persona": EXPECTED_COUNT,
        "personas": sum(len(GROUPS[group]) for group in args.groups),
        "totals": dict(totals),
        "persona_rows": persona_rows,
        "http_code_counts": {str(key): value for key, value in sorted(http_code_counts.items())},
        "http_group_counts": dict(sorted(group_counts.items())),
        "http_message_counts": [
            {"code": code, "message": message, "count": count}
            for (code, message), count in sorted(http_message_counts.items())
        ],
        "http_issues_total": len(http_issues),
        "http_issues": [asdict(item) for item in http_issues[:limit]],
        "non_http_failures_total": len(non_http_failures),
        "non_http_failures": [asdict(item) for item in non_http_failures[:limit]],
        "empty_outputs_total": len(empty_outputs),
        "empty_outputs": [asdict(item) for item in empty_outputs[:limit]],
        "invalid_files_total": len(invalid_files),
        "invalid_files": invalid_files[:limit],
        "id_mismatches_total": len(id_mismatches),
        "id_mismatches": id_mismatches[:limit],
        "persona_mismatches_total": len(persona_mismatches),
        "persona_mismatches": persona_mismatches[:limit],
        "warmup_issues_total": len(warmup_issues),
        "warmup_issues": warmup_issues[:limit],
    }


def write_csv_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model_slug",
        "group",
        "persona_folder",
        "question_id",
        "issue_category",
        "http_code",
        "error_message",
        "json_invalid",
        "relative_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def invalid_relative_path(path: str) -> str:
    normalized = Path(path).as_posix()
    prefix = "result/benchmarks/gsm8k/"
    return normalized[len(prefix):] if normalized.startswith(prefix) else normalized


def write_manifests(
    manifest_dir: Path,
    http_issues: list[HttpIssue],
    non_http_failures: list[ResponseIssue],
    invalid_files: list[dict[str, Any]],
) -> None:
    invalid_by_relative = {
        invalid_relative_path(str(item["path"])): item
        for item in invalid_files
    }

    def base_row(
        *,
        group: str,
        persona: str,
        filename: str,
        category: str,
        error_message: str,
        http_code: int | str = "",
    ) -> dict[str, object]:
        relative = f"{group}/{MODEL_SLUG}/{persona}/{filename}"
        return {
            "model_slug": MODEL_SLUG,
            "group": group,
            "persona_folder": persona,
            "question_id": Path(filename).stem,
            "issue_category": category,
            "http_code": http_code,
            "error_message": error_message,
            "json_invalid": str(relative in invalid_by_relative).lower(),
            "relative_path": relative,
        }

    http_rows = [
        base_row(
            group=issue.group,
            persona=issue.persona,
            filename=issue.file,
            category=f"http_{issue.code}",
            error_message=issue.message,
            http_code=issue.code,
        )
        for issue in http_issues
    ]
    non_http_rows = [
        base_row(
            group=issue.group,
            persona=issue.persona,
            filename=issue.file,
            category="non_http_infra",
            error_message=issue.error,
        )
        for issue in non_http_failures
    ]
    failed_paths = {
        str(row["relative_path"])
        for row in (*http_rows, *non_http_rows)
    }

    invalid_unrecoverable_rows: list[dict[str, object]] = []
    invalid_repair_rows: list[dict[str, object]] = []
    for relative, item in invalid_by_relative.items():
        parts = relative.split("/")
        if len(parts) != 4 or parts[1] != MODEL_SLUG:
            raise RuntimeError(f"Unexpected invalid JSON path: {relative}")
        group, _, persona, filename = parts
        if not item.get("prefix_recovered"):
            invalid_unrecoverable_rows.append(
                base_row(
                    group=group,
                    persona=persona,
                    filename=filename,
                    category="invalid_unrecoverable",
                    error_message=str(item.get("error") or ""),
                )
            )
        elif relative not in failed_paths:
            invalid_repair_rows.append(
                base_row(
                    group=group,
                    persona=persona,
                    filename=filename,
                    category="invalid_repair_only",
                    error_message=str(item.get("error") or ""),
                )
            )

    sort_key = lambda row: (
        str(row["group"]),
        str(row["persona_folder"]),
        str(row["question_id"]),
    )
    http_rows.sort(key=sort_key)
    non_http_rows.sort(key=sort_key)
    invalid_unrecoverable_rows.sort(key=sort_key)
    invalid_repair_rows.sort(key=sort_key)
    infra_rows = sorted(
        [*http_rows, *non_http_rows, *invalid_unrecoverable_rows],
        key=sort_key,
    )
    unique_infra_paths = {str(row["relative_path"]) for row in infra_rows}
    if len(unique_infra_paths) != len(infra_rows):
        raise RuntimeError("Generated infrastructure manifest contains duplicate paths")

    outputs = {
        "deepseek_http_4xx_5xx.csv": http_rows,
        "deepseek_non_http_infra.csv": non_http_rows,
        "deepseek_invalid_unrecoverable.csv": invalid_unrecoverable_rows,
        "deepseek_invalid_repair_only.csv": invalid_repair_rows,
        "deepseek_infra_all.csv": infra_rows,
    }
    for filename, rows in outputs.items():
        write_csv_manifest(manifest_dir / filename, rows)
        print(
            f"[MANIFEST] {manifest_dir / filename}: {len(rows)} row(s)",
            file=sys.stderr,
            flush=True,
        )


class GitBatchObjects:
    """Read all selected Git blobs through one ``cat-file --batch`` process."""

    def __init__(self, prefixes: list[str]) -> None:
        self.prefixes = prefixes
        self.process: subprocess.Popen[bytes] | None = None
        self.current_name = ""
        self.current_blob = b""
        self.writer_error: Exception | None = None

    def __enter__(self) -> "GitBatchObjects":
        return self

    def __exit__(self, *_: object) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    @staticmethod
    def _read_exact(stream: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise RuntimeError(
                    f"Unexpected end of git cat-file stream; {remaining} byte(s) missing"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _tracked_paths(self) -> list[str]:
        command = [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            "HEAD",
            "--",
            *self.prefixes,
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git ls-tree failed with code {result.returncode}: {error}")
        return [
            item.decode("utf-8", errors="surrogateescape")
            for item in result.stdout.split(b"\0")
            if item and item.lower().endswith(b".json")
        ]

    def __iter__(self):
        paths = self._tracked_paths()
        print(f"[GIT] cat-file batch: {len(paths)} JSON blobs", file=sys.stderr, flush=True)
        self.process = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if (
            self.process.stdin is None
            or self.process.stdout is None
            or self.process.stderr is None
        ):
            self.process.kill()
            raise RuntimeError("Cannot open git cat-file batch pipes")

        def write_queries() -> None:
            try:
                for path in paths:
                    self.process.stdin.write(f"HEAD:{path}\n".encode("utf-8"))
                self.process.stdin.close()
            except Exception as exc:  # propagated by the reader after the process exits
                self.writer_error = exc

        writer = threading.Thread(target=write_queries, name="git-cat-file-writer", daemon=True)
        writer.start()

        for index, path in enumerate(paths, start=1):
            header = self.process.stdout.readline()
            if not header:
                raise RuntimeError(f"Missing git cat-file header for {path}")
            header_parts = header.rstrip(b"\n").rsplit(b" ", 2)
            if len(header_parts) != 3 or header_parts[1] != b"blob":
                raise RuntimeError(
                    f"Unexpected git cat-file header for {path}: "
                    f"{header.decode('utf-8', errors='replace').strip()}"
                )
            try:
                size = int(header_parts[2])
            except ValueError as exc:
                raise RuntimeError(f"Invalid blob size for {path}: {header!r}") from exc
            self.current_name = path
            self.current_blob = self._read_exact(self.process.stdout, size)
            terminator = self.process.stdout.read(1)
            if terminator != b"\n":
                raise RuntimeError(f"Missing git cat-file terminator for {path}")
            member = tarfile.TarInfo(path)
            member.size = size
            yield member
            if index % 2000 == 0 or index == len(paths):
                print(f"[GIT] decoded {index}/{len(paths)}", file=sys.stderr, flush=True)

        writer.join()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = self.process.wait()
        self.process = None
        if self.writer_error is not None:
            raise RuntimeError(f"git cat-file query writer failed: {self.writer_error}")
        if return_code != 0:
            raise RuntimeError(f"git cat-file failed with code {return_code}: {stderr}")

    def extractfile(self, member: tarfile.TarInfo) -> io.BytesIO | None:
        if member.name != self.current_name or not self.current_blob:
            return None
        return io.BytesIO(self.current_blob)


def audit_git_head(args: argparse.Namespace) -> dict[str, Any]:
    """Audit current tracked results as sequential per-persona Git streams."""
    expected_names = {f"gsm8k_{index:05d}.json" for index in range(1, EXPECTED_COUNT + 1)}
    prefixes = [
        f"result/benchmarks/gsm8k/{group}/{MODEL_SLUG}/{persona}"
        for group in args.groups
        for persona in GROUPS[group]
    ]

    totals: Counter[str] = Counter()
    rows = {
        (group, persona): Counter()
        for group in args.groups
        for persona in GROUPS[group]
    }
    actual_names = {key: set() for key in rows}
    seen_warmups: set[tuple[str, str]] = set()
    http_issues: list[HttpIssue] = []
    non_http_failures: list[ResponseIssue] = []
    empty_outputs: list[ResponseIssue] = []
    invalid_files: list[dict[str, Any]] = []
    id_mismatches: list[dict[str, object]] = []
    persona_mismatches: list[dict[str, object]] = []
    warmup_issues: list[dict[str, str]] = []

    def invalid(
        path: str,
        error: Exception | str,
        *,
        warmup: bool,
    ) -> dict[str, Any] | None:
        if warmup:
            totals["warmup_invalid"] += 1
            warmup_issues.append({"path": path, "issue": f"invalid: {error}"})
            return None
        else:
            totals["invalid"] += 1
            entry: dict[str, Any] = {
                "path": path,
                "error": str(error),
                "prefix_recovered": False,
            }
            invalid_files.append(entry)
            return entry

    with GitBatchObjects(prefixes) as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".json"):
                continue
            parts = Path(member.name).as_posix().split("/")
            # result/benchmarks/gsm8k/<group>/<model>/<persona>/<file>
            if len(parts) < 6 or parts[:2] != ["result", "GMS8K"]:
                continue
            group, model_slug, persona = parts[2], parts[3], parts[4]
            key = (group, persona)
            if group not in args.groups or model_slug != MODEL_SLUG or key not in rows:
                continue
            is_main = len(parts) == 6 and re.fullmatch(r"gsm8k_\d{5}\.json", parts[5])
            is_warmup = (
                len(parts) == 7
                and parts[5] == "warmup"
                and parts[6] == "persona_warmup.json"
            )
            if not is_main and not is_warmup:
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                invalid(member.name, "cannot extract blob", warmup=is_warmup)
                continue
            blob = extracted.read()
            try:
                payload = json.loads(blob)
                if not isinstance(payload, dict):
                    raise ValueError("top-level JSON is not an object")
            except (OSError, ValueError, TypeError) as exc:
                invalid_entry = invalid(member.name, exc, warmup=is_warmup)
                if is_main:
                    rows[key]["invalid"] += 1
                try:
                    text = blob.decode("utf-8")
                    payload, _ = json.JSONDecoder().raw_decode(text)
                    if not isinstance(payload, dict):
                        raise ValueError("recovered prefix is not a JSON object")
                except (UnicodeError, ValueError, TypeError):
                    if is_main:
                        actual_names[key].add(parts[5])
                        rows[key]["files"] += 1
                        totals["files"] += 1
                    continue
                if invalid_entry is not None:
                    invalid_entry["prefix_recovered"] = True

            if is_warmup:
                seen_warmups.add(key)
                response = payload.get("response")
                response = response if isinstance(response, dict) else {}
                content, reasoning, _ = response_texts(response)
                if response.get("success") is not True or not (content or reasoning):
                    totals["warmup_failed"] += 1
                    warmup_issues.append(
                        {"path": member.name, "issue": "unsuccessful or empty"}
                    )
                continue

            filename = parts[5]
            relative = f"{group}/{persona}/{filename}"
            row = rows[key]
            row["files"] += 1
            totals["files"] += 1
            actual_names[key].add(filename)

            expected_question_id = int(filename[6:11])
            input_data = payload.get("input")
            input_data = input_data if isinstance(input_data, dict) else {}
            actual_question_id = input_data.get("question_id")
            try:
                actual_question_id = int(actual_question_id)
            except (TypeError, ValueError):
                pass
            if expected_question_id != actual_question_id:
                row["id_mismatch"] += 1
                totals["id_mismatch"] += 1
                id_mismatches.append(
                    {
                        "path": relative,
                        "filename_id": expected_question_id,
                        "json_id": actual_question_id,
                    }
                )

            persona_data = input_data.get("persona")
            persona_data = persona_data if isinstance(persona_data, dict) else {}
            actual_persona = str(persona_data.get("type") or "")
            if actual_persona != persona:
                row["persona_mismatch"] += 1
                totals["persona_mismatch"] += 1
                persona_mismatches.append(
                    {"path": relative, "json_persona": actual_persona}
                )

            response = payload.get("response")
            response = response if isinstance(response, dict) else {}
            final_text, reasoning, finish_reason = response_texts(response)
            final_empty = not bool(final_text)
            all_empty = final_empty and not bool(reasoning)
            if final_empty:
                row["final_empty"] += 1
                totals["final_empty"] += 1
            if all_empty:
                row["all_output_empty"] += 1
                totals["all_output_empty"] += 1
                empty_outputs.append(
                    ResponseIssue(
                        group=group,
                        persona=persona,
                        file=filename,
                        error=str(response.get("error") or ""),
                        success=response.get("success"),
                        finish_reason=finish_reason,
                        final_empty=True,
                        all_output_empty=True,
                    )
                )

            code, message = http_error(response)
            if code is not None:
                row["http"] += 1
                totals["http"] += 1
                http_issues.append(
                    HttpIssue(
                        code=code,
                        group=group,
                        persona=persona,
                        file=filename,
                        message=message,
                        success=response.get("success"),
                        final_empty=final_empty,
                        all_output_empty=all_empty,
                    )
                )
            elif response.get("success") is not True:
                row["success_false_no_http"] += 1
                totals["success_false_no_http"] += 1
                non_http_failures.append(
                    ResponseIssue(
                        group=group,
                        persona=persona,
                        file=filename,
                        error=str(response.get("error") or ""),
                        success=response.get("success"),
                        finish_reason=finish_reason,
                        final_empty=final_empty,
                        all_output_empty=all_empty,
                    )
                )

            analysis = payload.get("analysis")
            analysis = analysis if isinstance(analysis, dict) else {}
            if analysis.get("extracted_answer") in (None, ""):
                row["extracted_null"] += 1
                totals["extracted_null"] += 1
            if analysis.get("is_correct") is None:
                row["is_correct_null"] += 1
                totals["is_correct_null"] += 1

    persona_rows: list[dict[str, Any]] = []
    for group in args.groups:
        for persona in GROUPS[group]:
            key = (group, persona)
            row = rows[key]
            missing = sorted(expected_names - actual_names[key])
            extra = sorted(actual_names[key] - expected_names)
            row["missing"] = len(missing)
            row["extra"] = len(extra)
            totals["missing"] += len(missing)
            totals["extra"] += len(extra)
            if "_implicit_" in persona:
                totals["warmup_expected"] += 1
                if key not in seen_warmups:
                    totals["warmup_missing"] += 1
                    warmup_issues.append(
                        {
                            "path": f"{group}/{persona}/warmup/persona_warmup.json",
                            "issue": "missing",
                        }
                    )
            persona_rows.append(
                {
                    "group": group,
                    "persona": persona,
                    **dict(row),
                    "missing_preview": missing[:10],
                    "extra_preview": extra[:10],
                }
            )

    return finish_result(
        args,
        totals,
        persona_rows,
        http_issues,
        non_http_failures,
        empty_outputs,
        invalid_files,
        id_mismatches,
        persona_mismatches,
        warmup_issues,
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.source == "git-head":
        return audit_git_head(args)

    expected_names = {f"gsm8k_{index:05d}.json" for index in range(1, EXPECTED_COUNT + 1)}
    totals: Counter[str] = Counter()
    persona_rows: list[dict[str, Any]] = []
    http_issues: list[HttpIssue] = []
    non_http_failures: list[ResponseIssue] = []
    empty_outputs: list[ResponseIssue] = []
    invalid_files: list[dict[str, Any]] = []
    id_mismatches: list[dict[str, object]] = []
    persona_mismatches: list[dict[str, object]] = []
    warmup_issues: list[dict[str, str]] = []

    for group in args.groups:
        model_root = args.root / group / MODEL_SLUG
        if not model_root.is_dir():
            raise RuntimeError(f"Model directory is missing: {model_root}")

        for persona in GROUPS[group]:
            persona_root = model_root / persona
            files = sorted(persona_root.glob("gsm8k_*.json"))
            actual_names = {path.name for path in files}
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            row: Counter[str] = Counter(
                files=len(files),
                missing=len(missing),
                extra=len(extra),
            )
            totals.update(row)

            for path in files:
                relative = f"{group}/{persona}/{path.name}"
                manifest_relative = (
                    f"{group}/{MODEL_SLUG}/{persona}/{path.name}"
                )
                try:
                    text = path.read_text(encoding="utf-8")
                    payload = json.loads(text)
                    if not isinstance(payload, dict):
                        raise ValueError("top-level JSON is not an object")
                except OSError as exc:
                    row["invalid"] += 1
                    totals["invalid"] += 1
                    invalid_files.append(
                        {
                            "path": manifest_relative,
                            "error": str(exc),
                            "prefix_recovered": False,
                        }
                    )
                    continue
                except (TypeError, ValueError) as exc:
                    row["invalid"] += 1
                    totals["invalid"] += 1
                    invalid_entry: dict[str, Any] = {
                        "path": manifest_relative,
                        "error": str(exc),
                        "prefix_recovered": False,
                    }
                    invalid_files.append(invalid_entry)
                    try:
                        start = len(text) - len(text.lstrip())
                        payload, _ = json.JSONDecoder().raw_decode(text, start)
                        if not isinstance(payload, dict):
                            raise ValueError("recovered prefix is not a JSON object")
                    except (TypeError, ValueError):
                        continue
                    invalid_entry["prefix_recovered"] = True

                match = re.fullmatch(r"gsm8k_(\d{5})", path.stem)
                expected_question_id = int(match.group(1)) if match else None
                input_data = payload.get("input")
                input_data = input_data if isinstance(input_data, dict) else {}
                actual_question_id = input_data.get("question_id")
                try:
                    actual_question_id = int(actual_question_id)
                except (TypeError, ValueError):
                    pass
                if expected_question_id != actual_question_id:
                    row["id_mismatch"] += 1
                    totals["id_mismatch"] += 1
                    id_mismatches.append(
                        {
                            "path": relative,
                            "filename_id": expected_question_id,
                            "json_id": actual_question_id,
                        }
                    )

                persona_data = input_data.get("persona")
                persona_data = persona_data if isinstance(persona_data, dict) else {}
                actual_persona = str(persona_data.get("type") or "")
                if actual_persona != persona:
                    row["persona_mismatch"] += 1
                    totals["persona_mismatch"] += 1
                    persona_mismatches.append(
                        {"path": relative, "json_persona": actual_persona}
                    )

                response = payload.get("response")
                response = response if isinstance(response, dict) else {}
                final_text, reasoning, finish_reason = response_texts(response)
                final_empty = not bool(final_text)
                all_empty = final_empty and not bool(reasoning)
                if final_empty:
                    row["final_empty"] += 1
                    totals["final_empty"] += 1
                if all_empty:
                    row["all_output_empty"] += 1
                    totals["all_output_empty"] += 1
                    empty_outputs.append(
                        ResponseIssue(
                            group=group,
                            persona=persona,
                            file=path.name,
                            error=str(response.get("error") or ""),
                            success=response.get("success"),
                            finish_reason=finish_reason,
                            final_empty=True,
                            all_output_empty=True,
                        )
                    )

                code, message = http_error(response)
                if code is not None:
                    row["http"] += 1
                    totals["http"] += 1
                    http_issues.append(
                        HttpIssue(
                            code=code,
                            group=group,
                            persona=persona,
                            file=path.name,
                            message=message,
                            success=response.get("success"),
                            final_empty=final_empty,
                            all_output_empty=all_empty,
                        )
                    )
                elif response.get("success") is not True:
                    row["success_false_no_http"] += 1
                    totals["success_false_no_http"] += 1
                    non_http_failures.append(
                        ResponseIssue(
                            group=group,
                            persona=persona,
                            file=path.name,
                            error=str(response.get("error") or ""),
                            success=response.get("success"),
                            finish_reason=finish_reason,
                            final_empty=final_empty,
                            all_output_empty=all_empty,
                        )
                    )

                analysis = payload.get("analysis")
                analysis = analysis if isinstance(analysis, dict) else {}
                if analysis.get("extracted_answer") in (None, ""):
                    row["extracted_null"] += 1
                    totals["extracted_null"] += 1
                if analysis.get("is_correct") is None:
                    row["is_correct_null"] += 1
                    totals["is_correct_null"] += 1

            if "_implicit_" in persona:
                totals["warmup_expected"] += 1
                warmup_path = persona_root / "warmup" / "persona_warmup.json"
                if not warmup_path.is_file():
                    totals["warmup_missing"] += 1
                    warmup_issues.append(
                        {"path": f"{group}/{persona}/warmup/persona_warmup.json", "issue": "missing"}
                    )
                else:
                    try:
                        warmup = read_json(warmup_path)
                        response = warmup.get("response")
                        response = response if isinstance(response, dict) else {}
                        content, reasoning, _ = response_texts(response)
                        if response.get("success") is not True or not (content or reasoning):
                            totals["warmup_failed"] += 1
                            warmup_issues.append(
                                {
                                    "path": f"{group}/{persona}/warmup/persona_warmup.json",
                                    "issue": "unsuccessful or empty",
                                }
                            )
                    except (OSError, ValueError) as exc:
                        totals["warmup_invalid"] += 1
                        warmup_issues.append(
                            {
                                "path": f"{group}/{persona}/warmup/persona_warmup.json",
                                "issue": f"invalid: {exc}",
                            }
                        )

            persona_rows.append(
                {
                    "group": group,
                    "persona": persona,
                    **dict(row),
                    "missing_preview": missing[:10],
                    "extra_preview": extra[:10],
                }
            )

    return finish_result(
        args,
        totals,
        persona_rows,
        http_issues,
        non_http_failures,
        empty_outputs,
        invalid_files,
        id_mismatches,
        persona_mismatches,
        warmup_issues,
    )


def print_text(result: dict[str, Any]) -> None:
    print("=" * 80)
    print("GSM8K DeepSeek Religion/Gender audit (read-only)")
    print("=" * 80)
    print(f"Model: {result['model_slug']}")
    print(f"Source: {result['source']}")
    print(f"Groups: {result['groups']}")
    print(f"Personas: {result['personas']}")
    print(f"Expected per persona: {result['expected_per_persona']}")
    print("Totals:")
    for key, value in sorted(result["totals"].items()):
        print(f"  {key}={value}")
    print(f"HTTP codes: {result['http_code_counts']}")
    print(f"HTTP by group: {result['http_group_counts']}")
    print("Persona rows:")
    for row in result["persona_rows"]:
        values = "|".join(
            f"{key}={row.get(key, 0)}"
            for key in (
                "files",
                "missing",
                "extra",
                "invalid",
                "http",
                "success_false_no_http",
                "final_empty",
                "all_output_empty",
                "extracted_null",
                "is_correct_null",
            )
        )
        print(f"  {row['group']}/{row['persona']}|{values}")
    for section in (
        "http_issues",
        "non_http_failures",
        "empty_outputs",
        "invalid_files",
        "id_mismatches",
        "persona_mismatches",
        "warmup_issues",
    ):
        total_key = f"{section}_total"
        print(f"{section} ({result.get(total_key, len(result[section]))}):")
        for item in result[section]:
            print("  " + json.dumps(item, ensure_ascii=False, sort_keys=True))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    try:
        result = audit(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
