"""Hapus hasil per-soal yang GAGAL agar bisa dijalankan ulang oleh runner.

Latar belakang
--------------
Runner menulis file JSON untuk setiap soal, termasuk ketika panggilan API gagal
(misalnya tunnel SSH ke vLLM putus di tengah run). File gagal itu tetap ada di
disk, sehingga **menghitung jumlah file bukan cara yang sah untuk menilai
kelengkapan** -- pelajaran mahal dari sesi 10-11 Agustus 2026, di mana 3.674
file "ada" tetapi isinya

    {"success": false,
     "error": "HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded",
     "raw_api_response": {"choices": []}}

Skrip ini membuang file semacam itu. Karena runner dipanggil dengan
``--skip_existing``, soal yang filenya hilang otomatis dikerjakan ulang dan
soal yang sudah benar dilewati. Jadi: prune -> jalankan runner = hanya yang
gagal yang diulang.

Yang TIDAK dihapus
------------------
File dengan ``finish_reason == "length"`` dan ``content`` kosong **dibiarkan**.
Itu bukan kegagalan infrastruktur melainkan perilaku model: DeepSeek R1-distill
menghabiskan seluruh 4096 token untuk reasoning tanpa sempat menulis jawaban.
Menjalankannya ulang pada ``temperature=0`` menghasilkan keluaran identik, jadi
tidak ada gunanya -- angkanya sebaiknya dilaporkan apa adanya sebagai temuan.

Contoh
------
    python src/rerun/prune_failed_results.py --model deepseek_deepseek-r1-distill-qwen-32b
    python src/rerun/prune_failed_results.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (label, path relatif ke root repo, prefix nama file)
SCOPES = {
    "regional": [
        ("GSM8K", os.path.join("result", "GMS8K", "regional_cultural_persona"), "gsm8k_"),
        ("ETHICS", os.path.join("result", "ETHICS", "v9 [THIS IS THE MAIN]", "Regional Cultural Persona"), "ethics_"),
    ],
}


def is_failed(path: str) -> bool:
    """True bila file tidak terbaca atau response.success bukan True."""
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return True
    return (payload.get("response", {}) or {}).get("success") is not True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="deepseek_deepseek-r1-distill-qwen-32b",
                        help="Slug folder model di bawah tiap direktori scope.")
    parser.add_argument("--scope", default="regional", choices=sorted(SCOPES),
                        help="Kelompok persona yang diperiksa.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Hanya melaporkan, tidak menghapus apa pun.")
    args = parser.parse_args()

    grand = 0
    for label, rel_root, prefix in SCOPES[args.scope]:
        root = os.path.join(REPO_ROOT, rel_root, args.model)
        if not os.path.isdir(root):
            print(f"===== {label}: folder tidak ada, dilewati =====")
            print(f"   {root}\n")
            continue

        per_persona: Counter[str] = Counter()
        for persona in sorted(os.listdir(root)):
            pdir = os.path.join(root, persona)
            if not os.path.isdir(pdir):
                continue
            for name in os.listdir(pdir):
                if not (name.startswith(prefix) and name.endswith(".json")):
                    continue
                path = os.path.join(pdir, name)
                if not is_failed(path):
                    continue
                if not args.dry_run:
                    os.remove(path)
                per_persona[persona] += 1

        total = sum(per_persona.values())
        grand += total
        verb = "akan dihapus" if args.dry_run else "dihapus"
        print(f"===== {label}: {verb} {total} file =====")
        for persona, count in per_persona.most_common():
            print(f"   {persona:34s} {count}")
        print()

    print(f"TOTAL: {grand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
