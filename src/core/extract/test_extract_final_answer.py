import sys
sys.path.insert(0, '.')
from src.core.extract.rebuild_results_csv_root import extract_final_answer, normalize_number

def test_extract():
    tests = [
        ("Total time = (40 + 20 + 100 = 160) minutes, i.e. 2 hours 40 minutes", 160),
        ("n >= 90/7.50 = 12 ... (13th year ...)", 12),
        ("John spends **35 hours a week** taking care of the 10 dogs.", 35),
        ("**Answer:** $25,000 per year ... full pension $50,000 per year", 25000),
        ("**Answer:** The hospital earns a profit of **$10,000 per day** ... (annual $3.65 million)", 10000),
        ("**Answer: 95 minutes (1 hour 35 minutes).**", 95),
        ("Total weight = **2â€¯600 grams**", 2600),
        ("**Answer: 160 red roses must be delivered by 5 pm.**", 160),
        ("So Johnny picks up **2,125 blocks**.", 2125),
        ("Adding them together: (35 + 70 = 105) questions ...", 105),
    ]
    for i, (text, expected) in enumerate(tests):
        got, reason = extract_final_answer(text)
        if got != expected:
            print(f"FAILED: {i+1}. Text: {text}\n  Got: {got} ({reason}) | Expected: {expected}")
        else:
            print(f"PASS: {i+1} | {got} ({reason})")

def test_normalize():
    cases = [
        ("2,600", 2600),
        ("2â€¯600", 2600),
        ("$25,000", 25000),
        ("$3.65", 3.65),
    ]
    for raw, expected in cases:
        got = normalize_number(raw)
        if got != expected:
            print(f"FAILED normalize: {raw} -> {got} (expected {expected})")
        else:
            print(f"PASS normalize: {raw} -> {got}")

if __name__ == '__main__':
    test_extract()
    test_normalize()
