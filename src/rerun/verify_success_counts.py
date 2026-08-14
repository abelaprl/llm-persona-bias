"""Audit kelengkapan hasil berdasarkan ``response.success``, bukan jumlah file.

Kenapa ini ada
--------------
Jumlah file menyesatkan: runner tetap menulis JSON ketika panggilan API gagal.
Pada sesi 10-11 Agustus 2026, ETHICS "lengkap" 24.224 file padahal 3.660 di
antaranya berisi error koneksi. Verifikasi yang sah harus membaca
``response.success`` setiap file.

Keluaran
--------
Per benchmark: jumlah file, jumlah sukses, gagal, rusak, dan daftar persona yang
belum penuh beserta kekurangannya -- angka yang langsung bisa dipakai untuk
memperkirakan sisa waktu dan biaya GPU.

Contoh
------
    python src/rerun/verify_success_counts.py
    python src/rerun/verify_success_counts.py --model deepseek_deepseek-r1-distill-qwen-32b
"""

from __future__ import annotations

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (label, path relatif, prefix, jumlah soal per persona)
BENCHMARKS = [
    ("GSM8K", os.path.join("result", "GMS8K", "regional_cultural_persona"), "gsm8k_", 1319),
    ("ETHICS", os.path.join("result", "ETHICS", "v9 [THIS IS THE MAIN]", "Regional Cultural Persona"), "ethics_", 1514),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="deepseek_deepseek-r1-distill-qwen-32b")
    parser.add_argument("--personas", type=int, default=16,
                        help="Jumlah persona yang diharapkan (default 16 regional).")
    args = parser.parse_args()

    exit_code = 0
    for label, rel_root, prefix, per_persona in BENCHMARKS:
        root = os.path.join(REPO_ROOT, rel_root, args.model)
        if not os.path.isdir(root):
            print(f"===== {label}: folder tidak ada =====\n   {root}\n")
            exit_code = 1
            continue

        total = ok = fail = bad = 0
        kurang: dict[str, int] = {}
        for persona in sorted(os.listdir(root)):
            pdir = os.path.join(root, persona)
            if not os.path.isdir(pdir):
                continue
            persona_ok = 0
            for name in os.listdir(pdir):
                if not (name.startswith(prefix) and name.endswith(".json")):
                    continue
                total += 1
                try:
                    with open(os.path.join(pdir, name), encoding="utf-8") as handle:
                        payload = json.load(handle)
                except Exception:
                    bad += 1
                    continue
                if (payload.get("response", {}) or {}).get("success") is True:
                    ok += 1
                    persona_ok += 1
                else:
                    fail += 1
            if persona_ok < per_persona:
                kurang[persona] = per_persona - persona_ok

        target = per_persona * args.personas
        print(f"===== {label} =====")
        print(f"  file di disk        : {total}")
        print(f"  SUKSES              : {ok} / {target}")
        print(f"  gagal (success=false): {fail}")
        print(f"  JSON rusak          : {bad}")
        print(f"  persona belum penuh : {len(kurang)}")
        for persona, sisa in sorted(kurang.items(), key=lambda item: -item[1]):
            print(f"     {persona:34s} kurang {sisa}")
        print()
        if ok < target:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
