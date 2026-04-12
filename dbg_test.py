import re, sys
sys.path.insert(0, 'src')
from rebuild_results_csv import NUMBER_RE, normalize_number, extract_final_answer

# Test 1: 3 loaves
text1 = '**Answer:** 3 loaves of bread cost **$4 more** than 2 bagels.'
val, label = extract_final_answer(text1)
print(f'loaves: got={val}({label}) expected=4')

# Debug the marker split
marker_re = re.compile(r'(Final Answer|(?<![a-zA-Z-])Answer|So the answer is|Therefore|Thus|Hence)', re.I)
parts = marker_re.split(text1, maxsplit=1)
print('parts:', parts)
st = ''.join(parts[2:]) if len(parts) >= 3 else text1
print('before new strip:', repr(st))
st2 = re.sub(r'^[:\s]+', '', st)
print('after new strip:', repr(st2))
bgs = re.findall(r'\*\*([^*]+)\*\*', st2)
print('bold groups:', bgs)

# Test 2: three bolds
text2 = '**Answer:** After ten years, Rong has **2400** coins, Neil has **3360** coins, and together they have **5760** coins.'
val2, label2 = extract_final_answer(text2)
print(f'\nthree_bolds: got={val2}({label2}) expected=5760')
parts2 = marker_re.split(text2, maxsplit=1)
print('parts2[-1]:', repr(parts2[-1][:80]))
st2b = re.sub(r'^[:\s]+', '', ''.join(parts2[2:]))
print('search_text:', repr(st2b[:80]))
print('bolds:', re.findall(r'\*\*([^*]+)\*\*', st2b))
