"""
Hapus semua hasil GSM8K woman_implicit_2 untuk GPT-OSS-20B
(Warmup TIDAK dihapus)
"""
from pathlib import Path

print("="*80)
print("  MENGHAPUS HASIL GSM8K: openai_gpt-oss-20b woman_implicit_2")
print("="*80)
print()

# Find all gsm8k result files (NOT warmup)
folder = Path("result/openai_gpt-oss-20b/woman_implicit_2")
files_to_delete = list(folder.glob("gsm8k_*.json"))

print(f"Found {len(files_to_delete)} files to delete")
print()

deleted_count = 0
for file in files_to_delete:
    try:
        file.unlink()
        deleted_count += 1
        if deleted_count % 100 == 0:
            print(f"  Deleted {deleted_count}/{len(files_to_delete)} files...")
    except Exception as e:
        print(f"[ERROR] {file.name}: {e}")

print()
print("="*80)
print(f"SELESAI: {deleted_count}/{len(files_to_delete)} files dihapus")
print("="*80)
print()
print("Warmup file TIDAK dihapus - masih ada di:")
print("result\\openai_gpt-oss-20b\\woman_implicit_2\\warmup\\persona_warmup.json")
