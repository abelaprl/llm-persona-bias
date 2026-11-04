"""Rebuild the active thesis without changing its sources or figure assets."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess

ROOT = Path(r"D:\ProposalTA-18222008")
AUDIT = Path(r"D:\TugasAkhir\.codex_tmp\verify_ta_lm_20261011")
BUILD = ROOT / ".latex-build/verify-lm-no-ratios-20261011"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    AUDIT.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "thesis_compile", ROOT / "analisis/penyelarasan_lm_regional_20261011/compile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = AUDIT
    module.BUILD = BUILD
    sources = module.active()
    assets = []
    for source in sources.values():
        assets.extend(ROOT / p for p in re.findall(
            r"\\includegraphics\s*(?:\[[^]]*\])?\s*\{([^}]+)\}", source
        ))
    protected = {p: digest(p) for p in set(sources) | set(assets) if p.is_file()}
    target_assets = [p for p in set(assets) if "gsm8k" in p.parts and
                     ("agama" in p.parts or "regional_kultural" in p.parts)]
    BUILD.mkdir(exist_ok=False)
    for path in (ROOT / ".latex-build/lm-tanpa-rasio-20261011").glob("TA.*"):
        if path.suffix in {".aux", ".toc", ".lof", ".lot", ".bbl", ".out", ".lol", ".loa", ".equ"}:
            shutil.copy2(path, BUILD / path.name)
    module.main()
    pdf = BUILD / "TA.pdf"
    assert pdf.is_file()
    log = (BUILD / "TA.log").read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"There were undefined references|undefined citations|Rerun to get cross-references right|Label\(s\) may have changed|Please rerun LaTeX|Table widths have changed", log)
    subprocess.run([str(module.BIN / "miktex-pdftotext.exe"), "-layout", str(pdf),
                    str(AUDIT / "TA.text.txt")], env=module.environment(), check=True)
    text = (AUDIT / "TA.text.txt").read_text(encoding="utf-8")
    assert not re.search(r"\bk\s*/\s*n\b", text)
    assert all(digest(p) == old for p, old in protected.items())
    pages = int(re.search(r"\((\d+) pages, \d+ bytes\)", log)[1])
    result = {"pages": pages, "pdf_sha256": digest(pdf),
              "sources_and_assets_unchanged": True,
              "protected_files": len(protected), "k_over_n_absent_in_pdf_text": True,
              "note": "Raster assets separately checked against their plotting source; PDF text extraction does not inspect raster text.",
              "domain_assets": [str(p) for p in sorted(target_assets)],
              "backup": str(AUDIT / "TA.before.pdf")}
    assert (AUDIT / "TA.before.pdf").is_file()
    shutil.copy2(pdf, ROOT / "TA.pdf")
    assert digest(ROOT / "TA.pdf") == result["pdf_sha256"]
    result["published"] = True
    (AUDIT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
