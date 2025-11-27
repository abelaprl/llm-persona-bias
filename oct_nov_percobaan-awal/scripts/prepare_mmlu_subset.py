import os
import json
import random
from collections import defaultdict, Counter

# ------------------------------
# Konfigurasi
# ------------------------------

# Berapa soal per subject yang mau diambil
QUESTIONS_PER_SUBJECT = 10  # bisa kamu ubah ke 20/30 nanti

# Subject yang kita anggap high-quality + diverse
# Pastikan nama-namanya persis seperti yang ada di cais/mmlu
SELECTED_SUBJECTS = {
    "business_ethics",
    "management",
    "marketing",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "sociology",
    "us_foreign_policy",
    "world_religions",
    "philosophy",
    "formal_logic",
    "high_school_mathematics",
    "high_school_physics",
    "abstract_algebra",
}

# Seed biar sampling konsisten / reproducible
RANDOM_SEED = 42

# ------------------------------
# Path setup (sesuai struktur repo kamu)
# ------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")

MMLU_RAW = os.path.join(DATA_DIR, "mmlu_raw.json")
MMLU_OUT = os.path.join(DATA_DIR, f"mmlu_subset_{QUESTIONS_PER_SUBJECT}q_per_subject.jsonl")


def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(MMLU_RAW):
        print(f"⚠️ MMLU raw file not found at: {MMLU_RAW}")
        print("   Pastikan kamu sudah menjalankan scripts/download_mmlu_all.py")
        return

    # 1. Load raw MMLU
    print(f"📂 Loading raw MMLU from: {MMLU_RAW}")
    with open(MMLU_RAW, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("rows", [])
    print(f"📦 Loaded {len(rows)} total rows")

    # 2. Kumpulkan soal per subject
    by_subject = defaultdict(list)
    all_subjects = set()

    for row in rows:
        d = row.get("row", {})
        subj = d.get("subject") or d.get("category")
        if subj is None:
            continue
        all_subjects.add(subj)
        by_subject[subj].append(d)

    print("\n📚 Unique subjects found in your mmlu_raw.json:")
    for s in sorted(all_subjects):
        print("  -", s)

    # 3. Cek apakah semua SELECTED_SUBJECTS ada di data
    missing = SELECTED_SUBJECTS - all_subjects
    if missing:
        print("\n⚠️ WARNING: These selected subjects are NOT present in your mmlu_raw.json:")
        for s in sorted(missing):
            print("  -", s)
        print("   ➜ Cek lagi ejaan subject, atau lihat daftar subject di atas.")
        # tetap lanjut untuk subject yang ada, tapi kamu mungkin mau revisi set-nya
    else:
        print("\n✅ All selected subjects are present in your mmlu_raw.json")

    # 4. Sampling per subject
    print(f"\n🎯 Sampling up to {QUESTIONS_PER_SUBJECT} questions per selected subject...")
    total_kept = 0
    subj_counter = Counter()

    with open(MMLU_OUT, "w", encoding="utf-8") as out:
        for subj in sorted(SELECTED_SUBJECTS):
            questions = by_subject.get(subj, [])
            if not questions:
                print(f"   ⚠️ No questions found for subject: {subj}")
                continue

            # Kalau jumlah soal kurang dari target, ambil semua
            if len(questions) <= QUESTIONS_PER_SUBJECT:
                sampled = questions
            else:
                sampled = random.sample(questions, QUESTIONS_PER_SUBJECT)

            for i, ex in enumerate(sampled):
                qid = ex.get("id", f"{subj}_{i}")
                question = ex.get("question", "")
                choices = ex.get("choices", [])
                answer = ex.get("answer", None)

                record = {
                    "id": qid,
                    "subject": subj,
                    "question": question,
                    "A": choices[0] if len(choices) > 0 else "",
                    "B": choices[1] if len(choices) > 1 else "",
                    "C": choices[2] if len(choices) > 2 else "",
                    "D": choices[3] if len(choices) > 3 else "",
                    "answer": answer,
                }

                out.write(json.dumps(record) + "\n")
                total_kept += 1
                subj_counter[subj] += 1

    # 5. Ringkasan
    print(f"\n✅ Saved sampled subset to: {MMLU_OUT}")
    print(f"   ➜ Total kept questions: {total_kept}")
    print("   ➜ Breakdown per subject:")
    for subj, cnt in subj_counter.most_common():
        print(f"      - {subj}: {cnt} questions")


if __name__ == "__main__":
    main()
