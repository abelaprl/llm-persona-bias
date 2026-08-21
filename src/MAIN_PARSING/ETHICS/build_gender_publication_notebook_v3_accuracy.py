"""Create the executable ETHICS Gender Persona v3 publication notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "ETHICS_Gender_Persona_Publication_Analysis_v3_Accuracy.ipynb"
)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    markdown(
        """
        # ETHICS Gender Persona — Analisis Akurasi v3

        Notebook final ini mengikuti metode Bab VI: Commonsense, Deontology,
        Justice, dan Virtue dinilai sebagai akurasi terhadap ground truth.
        Seluruh respons masuk denominator; abstain dan missing bernilai salah.
        Semua uji inferensial dipasangkan berdasarkan ID soal yang sama.
        """
    ),
    markdown(
        """
        ## 1. Konfigurasi dan keluaran

        Analisis menggunakan seluruh 81.756 baris, enam model, sembilan kondisi,
        empat subset, dan 1.514 ID soal unik. Output baru disimpan pada direktori
        v3 sehingga artefak v2 tetap utuh.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import sys
        import pandas as pd
        from IPython.display import Image, display

        start = Path.cwd().resolve()
        REPO_ROOT = next(
            path for path in [start, *start.parents]
            if (path / "AGENT.md").exists() and (path / "src").exists()
        )
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from src.MAIN_PARSING.ETHICS.gender_publication_analysis_v3_accuracy import (
            run_analysis,
        )

        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_rows", 200)
        """
    ),
    code(
        """
        result = run_analysis()
        OUTPUT_DIR = Path(result["output_dir"])
        TABLE_DIR = OUTPUT_DIR / "tables"
        FIGURE_DIR = OUTPUT_DIR / "figures"
        print(f"Output: {OUTPUT_DIR}")
        """
    ),
    markdown(
        """
        ## 2. Validasi data dan denominator

        Pemeriksaan berikut memastikan semua baris digunakan, sumber tidak
        berubah, respons tanpa jawaban biner bernilai salah, dan pairing lengkap
        tanpa duplikasi.
        """
    ),
    code(
        """
        validation = result["validation"]
        display(pd.DataFrame({"check": validation.keys(), "value": validation.values()}))
        assert validation["valid"]
        """
    ),
    markdown("## 3. Akurasi deskriptif per subset, model, persona, dan skenario"),
    code(
        """
        accuracy = result["condition_accuracy"]
        display(
            accuracy[
                ["subset_label", "model_label", "persona_label", "scenario",
                 "total_responses", "n_correct", "n_incorrect", "accuracy_pct"]
            ].head(36).style.format({"accuracy_pct": "{:.2f}"})
        )
        """
    ),
    code(
        """
        display(
            result["tables"]["accuracy_combined_four_subsets_by_model_condition.csv"]
            [["model_label", "persona_label", "scenario", "total_responses", "n_correct", "accuracy_pct"]]
            .style.format({"accuracy_pct": "{:.2f}"})
        )
        """
    ),
    markdown("## 4. Ringkasan laki-laki dan perempuan"),
    code(
        """
        for table_name in [
            "accuracy_gender_overall.csv",
            "accuracy_gender_comparison_by_subset.csv",
            "accuracy_gender_comparison_by_model.csv",
            "accuracy_gender_comparison_by_scenario.csv",
        ]:
            print(table_name)
            display(result["tables"][table_name].style.format(precision=2))
        """
    ),
    markdown(
        """
        ## 5. Cochran’s Q

        Untuk setiap subset–model–gender, kondisi N, E, I1, I2, dan I3 diuji
        pada ID soal yang sama.
        """
    ),
    code(
        """
        display(
            result["cochran_q"]
            [["subset_label", "model_label", "identity_label", "n_pairs",
              "cochran_q_statistic", "degrees_of_freedom", "p_value", "significant_0_05"]]
            .style.format({"cochran_q_statistic": "{:.3f}", "p_value": "{:.4g}"})
        )
        """
    ),
    markdown(
        """
        ## 6. Exact McNemar dan koreksi Holm

        Tujuh perbandingan skenario dikoreksi dalam setiap kelompok
        subset–model–gender. Empat perbandingan laki-laki–perempuan dikoreksi
        dalam setiap kelompok subset–model. Cohen’s h dilaporkan sebagai effect size.
        """
    ),
    code(
        """
        comparison_columns = [
            "comparison_category", "subset_label", "model_label", "comparison_label",
            "n_pairs", "accuracy_a_pct", "accuracy_b_pct", "delta_accuracy_pp",
            "discordant_a_correct_b_wrong", "discordant_a_wrong_b_correct",
            "discordant_total", "exact_mcnemar_p_value", "holm_adjusted_p_value",
            "significant_holm_0_05", "cohens_h",
        ]
        display(result["all_comparisons"][comparison_columns].head(30).style.format(precision=4))
        """
    ),
    markdown("## 7. Ringkasan hasil signifikan"),
    code(
        """
        print("Keseluruhan")
        display(result["significant_overall"])
        print("Per subset")
        display(result["significant_by_subset"])
        print("Per model")
        display(result["significant_by_model"])
        """
    ),
    markdown(
        """
        ## 8. Diagnosis deskriptif respons

        Status respons dan alasan abstain digunakan untuk menjelaskan mengapa
        sebagian respons tidak memperoleh nilai benar, bukan sebagai ukuran hasil utama.
        """
    ),
    code(
        """
        display(result["tables"]["response_status_by_model.csv"].style.format({"percentage": "{:.2f}"}))
        display(result["tables"]["abstain_reason_by_persona.csv"].head(40).style.format({"percentage_within_persona_abstain": "{:.2f}"}))
        """
    ),
    markdown("## 9. Gambar publikasi"),
    code(
        """
        png_files = sorted(FIGURE_DIR.glob("*.png"))
        print(f"{len(png_files)} gambar PNG")
        for path in png_files:
            print(path.name)
            display(Image(filename=str(path), width=1500))
        """
    ),
    markdown(
        """
        ## 10. Indeks artefak

        Setiap tabel, gambar, dan metadata memiliki ukuran serta SHA-256 pada
        `output_index.csv`. File `metadata/validation.json` menyimpan seluruh
        pemeriksaan yang harus lulus.
        """
    ),
    code(
        """
        display(result["output_index"])
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
