"""
Hapus semua warmup man_implicit_3 supaya bisa diulang dengan prompt baru
"""
import os
from pathlib import Path

# Find all man_implicit_3 warmup files
pattern = "result/*/man_implicit_3/warmup/persona_warmup.json"
files_to_delete = list(Path(".").glob(pattern))

print("="*80)
print(f"MENGHAPUS {len(files_to_delete)} FILE WARMUP MAN_IMPLICIT_3")
print("="*80)

deleted = 0
for file_path in files_to_delete:
    try:
        # Get model name from path
        model_name = file_path.parts[1]
        print(f"\n[DELETE] {model_name}")
        print(f"         {file_path}")
        
        # Delete file
        file_path.unlink()
        deleted += 1
        print(f"         [OK] Deleted")
        
    except Exception as e:
        print(f"         [ERROR] {str(e)}")

print(f"\n{'='*80}")
print(f"SELESAI: {deleted}/{len(files_to_delete)} files dihapus")
print(f"{'='*80}")
print(f"\nSekarang jalankan warmup lagi untuk prompt 3:")
print(f"python src\\run_warmup_man_implicit.py")
