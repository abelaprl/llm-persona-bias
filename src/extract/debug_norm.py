from scripts.rebuild_results_csv import normalize_number, extract_final_answer
cases = [r'\\boxed{$140{,}000}', r'\\boxed{12%}', r'\\boxed{$140{,}000}$', 'Some work... \\boxed{$140{,}000}$', "95 minutes — that's 1 hour 35 minutes", 'Total time = (40 + 20 + 100 = 160) minutes, i.e. 2 hours 40 minutes']
for c in cases:
    print('INPUT:', repr(c))
    print('normalize:', normalize_number(c))
    print('extract:', extract_final_answer(c))
    print('---')
from scripts.rebuild_results_csv import extract_boxed
print('\nBoxed details:')
for c in cases[:4]:
    print('INPUT:', repr(c))
    print('boxed_matches:', __import__('re').findall(r'\\\\boxed\{([^}]*)\}', c) or __import__('re').findall(r'\\boxed\{([^}]*)\}', c))
    print('extract_boxed:', extract_boxed(c))
    print('---')
