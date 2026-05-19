import sys
sys.path.insert(0, r'E:\Tugas Akhir\TugasAkhir\src')
from rebuild_results_csv import extract_final_answer

text = """**Blue shoe boxes left:** 4
**Red shoe boxes left:** 6

✅ **Answer: 4 blue and 6 red shoe boxes are left.**"""

result, reason = extract_final_answer(text)
print(f"Result: {result}, Reason: {reason}")
print(f"Expected: 10")
print()

# Debug: show what bolds are found
import re
last_lines = text.splitlines()[-20:]
print("Lines:")
for i, line in enumerate(last_lines):
    print(f"  {i}: {repr(line)}")
print()

marker_re = re.compile(r'(Final Answer|Answer|So the answer is|Therefore|Thus|Hence)', re.I)
for i, line in enumerate(reversed(last_lines)):
    if marker_re.search(line):
        print(f"Marker line (index {len(last_lines)-1-i}): {repr(line)}")
        parts = marker_re.split(line, maxsplit=1)
        search_text = ''.join(parts[2:]) if len(parts) >= 3 else line
        search_text = re.sub(r'^[\*:\s]+', '', search_text)
        print(f"Search text: {repr(search_text)}")
        
        bolds = re.findall(r'\*\*([^*]+)\*\*', search_text)
        print(f"Bolds in search_text: {bolds}")
        print(f"Has 'and' in search_text: {'and' in search_text}")
        break
