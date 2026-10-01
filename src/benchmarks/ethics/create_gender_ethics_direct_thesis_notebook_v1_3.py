"""Create the readable ETHICS Gender direct-comparison notebook v1.3.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_NOTEBOOK = (
    REPO_ROOT
    / "notebooks"
    / "gender"
    / "linguistic marker"
    / "template"
    / "linguistic_marker_gender_ethics_direct_thesis_v1_2.ipynb"
)
OLD_BUNDLE = "gender_ethics_direct_thesis_visuals_v1_2_0_r4_20260825"
NEW_BUNDLE = "gender_ethics_direct_thesis_visuals_v1_3_1_r4_20260825"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_notebook(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite notebook: {output}")
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(SOURCE_NOTEBOOK)

    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    replacements = (
        (OLD_BUNDLE, NEW_BUNDLE),
        ("gender-ethics-direct-thesis-visuals-v1.2.0", "gender-ethics-direct-thesis-visuals-v1.3.1"),
        ("Perbandingan Langsung Man dan Woman v1.2", "Perbandingan Langsung Man dan Woman v1.3.1"),
    )
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        for old, new in replacements:
            source = source.replace(old, new)
        source = source.replace("kondisi", "skenario").replace("Kondisi", "Skenario")
        if cell.get("cell_type") == "code" and "from IPython.display import Image" in source:
            source = source.replace(
                "from IPython.display import Image, Markdown, display",
                "from IPython.display import HTML, Image, Markdown, display",
            )
            css = '''

display(HTML("""
<style>
.jp-RenderedHTMLCommon, .rendered_html {
    font-size: 17px !important;
    line-height: 1.5 !important;
}
.jp-RenderedHTMLCommon table, .rendered_html table,
.jp-OutputArea-output table, .output_subarea table {
    font-size: 15px !important;
}
.jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td,
.rendered_html th, .rendered_html td {
    padding: 7px 10px !important;
}
.jp-OutputArea pre, .output_area pre {
    font-size: 15px !important;
    line-height: 1.45 !important;
}
</style>
"""))
'''
            source = source.replace(
                "pd.set_option(\"display.max_columns\", 40)",
                css + "\npd.set_option(\"display.max_columns\", 40)",
            )
        # Use essentially the full notebook width, as in the GSM8K v3 notebook.
        source = source.replace("width=1300", "width=1700")
        source = source.replace("width=1700", "width=1900")
        source = source.replace("width=1750", "width=1900")
        source = source.replace("width=1800", "width=1950")
        source = source.replace("width=1850", "width=2000")
        source = source.replace("width=1900", "width=2000")
        cell["source"] = source.splitlines(keepends=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[DONE] Notebook created: {output}")
    return output


def main() -> int:
    args = _parser().parse_args()
    build_notebook(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
