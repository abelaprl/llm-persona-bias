"""
Hapus semua warmup woman_implicit_3 untuk re-run dengan prompt baru
"""
from pathlib import Path

print("="*80)
print("          MENGHAPUS 12 FILE WARMUP WOMAN_IMPLICIT_3")
print("="*80)
print()

# Find all woman_implicit_3 warmup files
files_to_delete = list(Path(".").glob("result/*/woman_implicit_3/warmup/persona_warmup.json"))

for file in files_to_delete:
    # Extract model name from path
    model_name = file.parts[1]
    
    print(f"[DELETE] {model_name}")
    print(f"         {file}")
    
    try:
        file.unlink()
        print("         [OK] Deleted")
    except Exception as e:
        print(f"         [ERROR] {e}")
    
    print()

print("="*80)
print(f"SELESAI: {len(files_to_delete)}/12 files dihapus")
print("="*80)
print()
print("Sekarang jalankan warmup lagi (prompt 1 & 2 akan di-skip):")
print("python src\\run_warmup_woman_implicit.py")
