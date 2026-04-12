import re

# Import actual functions from rebuild_results_csv
import sys
sys.path.insert(0, r'E:\Tugas Akhir\TugasAkhir\src')
from rebuild_results_csv import extract_boxed, normalize_number, NUMBER_RE

# Test text from test case
text = "### ✅ Final Answer:\n\n$$\n\\boxed{224,\\!000} \\text{ dollars per month}\n$$"

print("Text:", repr(text))
print()
print("NUMBER_RE:", repr(NUMBER_RE))
print()

# Test with actual extract_boxed function
num, found = extract_boxed(text)
print("extract_boxed result:", num, found)
print()

# Manual test - extract boxed content
boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
print("Manual boxed extract:", boxed)

if boxed:
    val = boxed[0]
    print("First val:", repr(val))
    
    # Test NUMBER_RE match
    matches = re.findall(NUMBER_RE, val)
    print("NUMBER_RE matches in val:", matches)
    print()
    
    # Test normalization
    if matches:
        for m in matches:
            norm = normalize_number(m)
            print(f"normalize_number({repr(m)}) = {norm}")
    print()
    
    # Manual normalization
    s = str(val)
    print("1. Input:", repr(s))
    
    s = s.replace('\\!', '')
    print("2. After remove \\!:", repr(s))
    
    s = s.replace(',', '')
    print("3. After remove comma:", repr(s))
    
    try:
        n = int(s)
        print("4. Result:", n)
    except Exception as e:
        print("4. Error:", e)
