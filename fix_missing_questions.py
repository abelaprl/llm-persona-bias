import os
import json
from tqdm import tqdm

def check_and_fix_questions(base_dir="result/allenai_olmo-3-32b-think"):
    fixed = 0
    missing = 0
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                    # Cek dan perbaiki jika field question kosong/null
                    if "input" in data and ("question" not in data["input"] or not data["input"]["question"]):
                        # Coba ambil dari request.messages
                        question = None
                        if "request" in data and "messages" in data["request"]:
                            for m in data["request"]["messages"]:
                                if m["role"] == "user" and "Solve the following math problem" in m["content"]:
                                    # Ambil baris setelah 'Solve the following math problem:'
                                    parts = m["content"].split("Solve the following math problem:")
                                    if len(parts) > 1:
                                        question = parts[1].strip().split("\nAnswer:")[0].strip()
                                    else:
                                        question = m["content"].strip()
                                    break
                                elif m["role"] == "user" and "Solve the following question:" in m["content"]:
                                    parts = m["content"].split("Solve the following question:")
                                    if len(parts) > 1:
                                        question = parts[1].strip().split("\nAnswer:")[0].strip()
                                    else:
                                        question = m["content"].strip()
                                    break
                        if question:
                            data["input"]["question"] = question
                            with open(fpath, "w", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)
                            fixed += 1
                        else:
                            missing += 1
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
    print(f"Selesai. {fixed} file berhasil diperbaiki. {missing} file masih tidak ada question.")

if __name__ == "__main__":
    check_and_fix_questions()
