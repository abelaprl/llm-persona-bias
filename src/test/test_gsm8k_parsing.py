"""Uji V2, V5, V6, V7 untuk src/MAIN_PARSING/GSM8K.

Jalankan:  python -m src.test.test_gsm8k_parsing
"""
import ast
import inspect
import io
import re
import sys
import tokenize
import csv
import json
import tempfile
from unittest import mock
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.MAIN_PARSING.GSM8K._core import skeleton, numbers, runner  # noqa: E402
from src.MAIN_PARSING.GSM8K._core.numbers import normalize_number    # noqa: E402
from src.MAIN_PARSING.GSM8K.Religion_Persona import (                # noqa: E402
    rules_ministral_8b, rules_llama31_8b, rules_llama33_70b,
    rules_gpt_oss_20b, rules_qwen3_next_80b, rules_deepseek_r1_distill_32b,
    rules_core,
)
from src.MAIN_PARSING.GSM8K.Regional_Cultural_Persona import (       # noqa: E402
    rules_deepseek_r1_distill_32b as regional_deepseek_r1_distill_32b,
    rules_llama31_8b as regional_llama31_8b,
    rules_llama33_70b as regional_llama33_70b,
    rules_ministral_8b as regional_ministral_8b,
    rules_gpt_oss_20b as regional_gpt_oss_20b,
)

PACKS = [rules_ministral_8b, rules_llama31_8b, rules_llama33_70b,
         rules_gpt_oss_20b, rules_qwen3_next_80b, rules_deepseek_r1_distill_32b,
         rules_core]

RULES_DIR = REPO / 'src' / 'MAIN_PARSING' / 'GSM8K' / 'Religion_Persona'
CORE_DIR = REPO / 'src' / 'MAIN_PARSING' / 'GSM8K' / '_core'

failures = []


def code_only(path_or_text):
    """Buang komentar dan docstring; sisakan kode yang benar-benar dieksekusi.

    Grep mentah pada berkas ikut menangkap kalimat di docstring seperti
    'tidak boleh membaca nama persona' — itu dokumentasi, bukan cabang kode.
    """
    src = Path(path_or_text).read_text(encoding='utf-8')
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                continue
            out.append((tok.type, tok.string, tok.start, tok.end, tok.line))
    except tokenize.TokenError:
        return src
    text = tokenize.untokenize([(t, s_) for t, s_, *_ in out])
    # buang docstring lewat AST
    try:
        tree = ast.parse(src)
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                d = ast.get_docstring(node, clean=False)
                if d:
                    docs.add(d)
        for d in docs:
            text = text.replace(d, ' ')
    except SyntaxError:
        pass
    return text


def check(name, cond, detail=''):
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}  {detail}')
        failures.append(name)


# ── V5: normalisasi angka ───────────────────────────────────────────────────
def test_v5_numbers():
    print('\nV5a normalize_number')
    cases = [
        ('140{,}000', 140000), ('3,430', 3430), ('3528', 3528), ('$3{,}528', 3528),
        ('85 000', 85000), ('3.65', 3.65), ('224,\\!000', 224000), ('-12', -12),
        ('−12', -12), ('25%', 25), ('$25,000', 25000),
    ]
    for raw, want in cases:
        got = normalize_number(raw)
        check(f'normalize({raw!r}) == {want}', got == want, f'got {got}')


# ── V5: ekstraksi ujung-ke-ujung ────────────────────────────────────────────
def test_v5_extract():
    print('\nV5b extract() pada pola nyata')
    R = rules_ministral_8b
    cases = [
        ('#### 18', 18, 'hashes'),
        ('The answer is \\boxed{72}.', 72, 'boxed'),
        ('Final Answer: 3,430 people', 3430, 'marker_line'),
        ('Answer: $25,000', 25000, 'marker_line'),
        ('So the answer is 1,800 pounds.', 1800, 'marker_line'),
        # pola nyata llama-3.1-8b Religion Persona (tanpa penanda apa pun)
        ('Assalamu alaikum! ... So, Janet makes $18 every day at the farmers\' market.',
         18, 'marker_line'),
        ('Therefore, the total cost is $160.', 160, 'marker_line'),
    ]
    for text, want, want_src in cases:
        r = skeleton.extract(text, R)
        check(f'{text[:46]!r} -> {want}', r.value == want,
              f'got {r.value} src={r.source} prio={r.priority_hit}')
        if r.value == want:
            check(f'   sumber = {want_src}', r.source == want_src, f'got {r.source}')

    print('\nV5c guard: baris tabel & catatan tidak jadi jawaban')
    r = skeleton.extract('| Item | 999 |\nAnswer: 42', rules_ministral_8b)
    check('baris tabel dilewati', r.value == 42, f'got {r.value}')
    r = skeleton.extract('Answer: 42\nNote: if you buy 99 more, ...', rules_ministral_8b)
    check('baris Note: dilewati', r.value == 42, f'got {r.value}')
    r = skeleton.extract('Answer: 48\nPlease note that x = 20 is assumed.',
                         rules_ministral_8b)
    check('baris Please note dilewati', r.value == 48, f'got {r.value}')

    print('\nV5d content kosong -> missing_content, bukan diambil dari reasoning')
    r = skeleton.extract('', rules_ministral_8b)
    check('content kosong -> missing_content', r.status == 'missing_content', r.status)

    print('\nV5e <think> tidak tertutup -> abstain')
    r = skeleton.extract('<think>hitung 5+5=10', rules_qwen3_next_80b)
    check('unclosed think -> abstain', r.abstain_reason == 'unclosed_think', r.abstain_reason)

    print('\nV5f kalimat penutup di-commit; telusur mundur hanya candidate (A5)')
    r = skeleton.extract('Janet has 9 eggs left.', rules_ministral_8b)
    check('kalimat terakhir berangka -> committed sebagai final_sentence',
          r.status == 'committed' and r.source == 'final_sentence' and r.value == 9,
          f'status={r.status} src={r.source} val={r.value}')
    r = skeleton.extract('Janet has 9 eggs left. Let me know if you need anything else!',
                         rules_ministral_8b)
    check('kalimat terakhir tanpa angka -> candidate, bukan committed',
          r.status == 'abstain' and r.candidate == 9,
          f'status={r.status} cand={r.candidate}')

    print('\nV5g rule pack core hanya RULE 1-3')
    r = skeleton.extract('So, Janet makes $18 every day.', rules_core)
    check('core tidak memakai conclusion', r.value is None, f'got {r.value}')
    r = skeleton.extract('Answer: 18', rules_core)
    check('core tetap memakai marker_line', r.value == 18, f'got {r.value}')


def test_real_bug_regressions():
    print('\nRegresi bug audit 2026-08-13')
    R = rules_qwen3_next_80b
    cases = [
        ('#### 18. Answer follows.', None, 'heading hash 18 bukan 1'),
        ('#### 1. Daily consumption\nFinal Answer: 42', 42,
         'heading hash tidak mengalahkan final answer'),
        ('#### 2. Step\nThe result is \\boxed{72}.', 72,
         'heading hash tidak mengalahkan boxed'),
        ('Answer: It will take John 2 weeks to save $80.', 2,
         'resolver duration memilih 2'),
        ('Answer: 448 mL of medicine would be in 8 doses.', 448,
         'resolver quantity memilih 448'),
        ('**Answer:** Mr. Ruther originally had 32 hectares.', 32,
         'Mr. tidak memutus answer clause'),
        ('**Answer:** Mrs. Ruther originally had 31 hectares.', 31,
         'Mrs. tidak memutus answer clause'),
    ]
    for text, want, name in cases:
        got = skeleton.extract(text, R).value
        check(name, got == want, f'got {got}')
    check('bullet bukan negatif', numbers.find_numbers('- 7 packs') == [7],
          str(numbers.find_numbers('- 7 packs')))
    check('operator subtraction bukan negatif', numbers.find_numbers('56 - 54 = 2') == [56, 54, 2],
          str(numbers.find_numbers('56 - 54 = 2')))
    check('literal negatif tetap sah', numbers.find_numbers('-12') == [-12])
    for reason in ('error', 'content_filter'):
        rr = skeleton.extract('Final Answer: 42', R, finish_reason=reason)
        check(f'{reason} tidak committed', rr.status != 'committed', rr.status)
    rr = skeleton.extract('So the total is 42.', R, finish_reason='length')
    check('length + recovery lemah abstain truncated',
          rr.status == 'abstain' and rr.abstain_reason == 'truncated', rr.status)
    rr = skeleton.extract('Final Answer: 42', R, finish_reason='length')
    check('length + final eksplisit boleh committed', rr.status == 'committed' and rr.value == 42)
    rr = skeleton.extract('**Step 1:** 30', R)
    check('bold step label tidak auto-commit', rr.source != 'bold_4d', rr.source)

    terminal_bold_cases = [
        (regional_ministral_8b,
         '18 / 3 = 6\n**6**', 6),
        (regional_gpt_oss_20b,
         'First 106.123, second 2. **106.12**',
         106.12),
    ]
    for pack, text, want in terminal_bold_cases:
        rr = skeleton.extract(text, pack)
        check(f'{pack.RULE_PACK}: terminal bold audit fallback',
              rr.value == want and rr.priority_hit == 'terminal_bold_value',
              f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'The estimate may be **42**, but the result is uncertain.',
        regional_gpt_oss_20b,
    )
    check('terminal bold tidak commit pernyataan tidak pasti',
          rr.priority_hit != 'terminal_bold_value',
          f'value={rr.value} priority={rr.priority_hit}')

    llama_step_cases = [
        ('1. **Math books**: 4 pounds. 2. **English book**: 3 pounds. '
         'The total weight of the books is 17 pounds.', 17),
        ('1. **Donuts**: 3 x $68 = $204. 2. **Cupcakes**: $160. '
         'The total cost is $694.', 694),
    ]
    for text, want in llama_step_cases:
        rr = skeleton.extract(text, rules_llama33_70b)
        check('Llama 3.3: label langkah tebal tidak mengalahkan kesimpulan',
              rr.value == want and rr.source not in ('bold_4a', 'bold_4d'),
              f'value={rr.value} source={rr.source}')

    llama_terminal_cases = [
        ('Total meal cost: $14. Gift card: $5. Chad still owes $9.', 9),
        ('The calculation is complete. Avery will spend $180 on place settings. '
         'May this answer be helpful!', 180),
        ('A single setting is $12 for 5 apples and 5 oranges.', 12),
    ]
    for text, want in llama_terminal_cases:
        rr = skeleton.extract(text, rules_llama33_70b)
        check('Llama 3.3: nilai terminal tunggal mendahului recovery umum',
              rr.value == want and rr.source == 'final_sentence',
              f'value={rr.value} source={rr.source}')
    rr = skeleton.extract('Buying 6 goats will make them 30% of the collection.',
                          rules_llama33_70b)
    check('Llama 3.3: kalimat terminal multiangka tetap ambigu',
          rr.priority_hit not in ('terminal_single_number',
                                  'terminal_single_currency'),
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    filtered_marker_cases = [
        (
            'The original price was $700. However, the price increased by 10%, '
            'so the new price is $770 Therefore, the company paid a total of '
            '$385,000 for the computers.',
            385000,
        ),
        (
            'The combined quantity is 120. However, the ratio gives 20x = 120 '
            'and x = 6 Therefore, Katy used 42 teaspoons of sugar.',
            42,
        ),
        (
            'The difference is $6. However, the requested direction is the '
            'opposite. The final answer is $0.',
            0,
        ),
    ]
    for text, want in filtered_marker_cases:
        rr = skeleton.extract(text, rules_llama33_70b)
        check('Llama 3.3: penanda final di suffix caveat tidak ikut terbuang',
              rr.value == want and rr.source == 'marker_line',
              f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'If the book is 1.5 inches thick, then 1.5 x 100 = 150 pages '
        'However, each sheet has two sides, so 150 x 2 = 300 pages '
        'So, the book contains 300 pages. '
        'May this solution bring a smile to your face!',
        rules_llama33_70b,
    )
    check('Llama 3.3: So terminal di suffix segmen If tidak ikut terbuang',
          rr.value == 300 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    for text in (
        'However, if we assume Dior took 9 hours, the answer would be 19 hours.',
        'However, with a different assumption, the answer could be 20 years.',
        'However, the answer depends on information that was not provided.',
    ):
        rr = skeleton.extract(text, rules_llama33_70b)
        check('Llama 3.3: jawaban hipotetis dalam caveat tidak dipromosikan',
              rr.source != 'marker_line',
              f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'So, on Wednesday they delivered 204 + 5 = 209 letters. '
        'The post office delivered 736 letters from Monday to Wednesday.',
        rules_llama33_70b,
    )
    check('Llama 3.3: nilai terminal mengalahkan So langkah antara',
          rr.value == 736 and rr.source == 'final_sentence',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'So, Lauren would need to work for 40 years while saving 20% of her '
        'salary of $100,000.',
        rules_llama33_70b,
    )
    check('Llama 3.3: So terminal multiangka tetap diparse sebagai marker',
          rr.value == 40 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'Answer: 11. A later note mentions 44 items.', rules_llama33_70b)
    check('Llama 3.3: marker kuat tetap mengalahkan nilai terminal',
          rr.value == 11 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'To find the total amount Ted spent, add the costs. '
        'So, Ted spent a total of $66. '
        'Ted has $134 left.',
        rules_llama33_70b,
    )
    check('Llama 3.3: target spent eksplisit tidak diganti saldo lanjutan',
          rr.value == 66 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'Therefore, Angela is currently 38 years old. '
        'So, we can set up the equation',
        rules_llama33_70b, finish_reason='length',
    )
    check('Llama 3.3: truncation mempertahankan marker eksplisit',
          rr.value == 38 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'She needs 77 correct answers in total and got 42. '
        'Therefore, Audrey needs 35 correct answers to move to the 7th grade.',
        rules_llama33_70b,
    )
    check('kata answers jamak bukan marker Answer',
          rr.value == 35 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'Answering 12 questions is only an intermediate step. Answer: 35.',
        rules_llama33_70b,
    )
    check('kata Answering bukan marker Answer',
          rr.value == 35 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'She needs to answer 70% of the questions. The answer is 35.',
        rules_llama33_70b,
    )
    check('verba answer bukan marker, tetapi The answer is tetap marker',
          rr.value == 35 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'I hope this answer is helpful. The correct answer is 35.',
        rules_llama33_70b,
    )
    check('this answer is helpful bukan marker jawaban',
          rr.value == 35 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'Therefore, the answer is 8 sheets of paper (with 4 pages per sheet, '
        '2 on the front and 2 on the back).',
        rules_llama33_70b,
    )
    check('angka jawaban leading mengalahkan rincian parenthetical',
          rr.value == 8 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'Now, to find the cost price, subtract the profit. '
        'So, Isaias bought the 180 chickens for $7,000.',
        rules_llama33_70b,
    )
    check('harga transaksi dibedakan dari jumlah barang',
          rr.value == 7000 and rr.priority_hit == 'marker_transaction_price',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'Finally, calculate how much ore the group can mine. '
        'So, 40 dwarves can mine 43,200 pounds in 30 days.',
        rules_llama33_70b,
    )
    check('subjek numerik dibedakan dari besaran target',
          rr.value == 43200 and rr.priority_hit == 'marker_leading_subject_result',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        '65 pounds/sack = 1300 pounds. Total weight = 4500 pounds. '
        '4500 / 2250 = 2 trips, and 2 trips carry 4500 pounds '
        '(which is the exact amount), so the farmer needs 2 trips.',
        rules_llama33_70b,
    )
    check('parenthetical jauh tidak membuat angka langkah menjadi leading answer',
          rr.value == 2,
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    terminal_declarations = [
        (
            "3 legs/table = 150 legs The restaurant's tables have a total of "
            '310 legs. Shalom, I hope this answer is helpful!',
            310,
        ),
        (
            'Scores: 19 + 16 + 8 + 8 = 51. The total age of the birds is: '
            '19 + 16 + 8 + 8 = 51 years. Shalom!',
            51,
        ),
        (
            'Jackson will need to buy: 7 packs of glue sticks This will give '
            'every student the 2 sticks they need. May God bless them!',
            7,
        ),
    ]
    for text, want in terminal_declarations:
        rr = skeleton.extract(text, rules_llama33_70b)
        check('Llama 3.3: deklarasi terminal audited di-commit',
              rr.value == want and rr.priority_hit == 'terminal_declaration',
              f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    uncertain_declarations = [
        'If we assume Dior took 9 hours, the answer would be 19 hours.',
        'The rough estimate is 20 people, but it is not a definitive answer.',
        'The total would be $43,500. Please note that this answer is incomplete.',
    ]
    for text in uncertain_declarations:
        rr = skeleton.extract(text, rules_llama33_70b)
        check('Llama 3.3: deklarasi tidak pasti tetap tidak di-commit',
              rr.priority_hit != 'terminal_declaration',
              f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    terminal_equations = [
        (
            'Anakin caught 19 fish and Locsin caught 13 fish. '
            'Their combined catch: 19 + 13 = 32 fish. Wa alaikum salam!',
            32,
        ),
        (
            'Passed = 340 + 40 + 320 = 700. Students = 1000. '
            'The pass rate = 700/1000 = 0.7 or 70%. I hope this helps.',
            70,
        ),
    ]
    for text, want in terminal_equations:
        rr = skeleton.extract(text, rules_llama31_8b)
        check('Llama 3.1: persamaan terminal audited di-commit',
              rr.value == want
              and rr.priority_hit == 'terminal_equation_agreement',
              f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        "First salary = $56,000. Now, let's find the second salary. "
        'Unfortunately, the rest of the calculation is unavailable.',
        rules_llama31_8b,
    )
    check('Llama 3.1: persamaan yang masih dilanjutkan bukan terminal',
          rr.priority_hit != 'terminal_equation_agreement',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        'Height = 180 - 180 = 0 inches. However, this is incorrect because '
        "it does not make sense. The roof height still needs recalculation.",
        rules_llama31_8b,
    )
    check('Llama 3.1: persamaan yang disangkal tidak di-commit',
          rr.priority_hit != 'terminal_equation_agreement',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract('22', rules_gpt_oss_20b)
    check('GPT-OSS: seluruh final content berupa satu angka di-commit',
          rr.value == 22 and rr.priority_hit == 'whole_response_single_value',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract('22', rules_deepseek_r1_distill_32b)
    check('DeepSeek: seluruh final content berupa satu angka di-commit',
          rr.value == 22 and rr.priority_hit == 'whole_response_single_value',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'James made $17,500 from his commission on the profits from selling '
        'the 10 used cars. I hope this helps!',
        rules_deepseek_r1_distill_32b,
    )
    check('DeepSeek: deklarasi terminal sebelum salam di-commit',
          rr.value == 17500 and rr.priority_hit == 'terminal_declaration',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    for text in (
        'Step 3: 3 teachers. **Answer**: There are 36 teachers.',
        'Step 3: 3 teachers. **Answer:** There are 36 teachers.',
    ):
        rr = skeleton.extract(text, rules_deepseek_r1_distill_32b)
        check('DeepSeek: marker Answer tebal tetap dikenali sebagai marker kuat',
              rr.value == 36 and rr.source == 'marker_line',
              f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    deepseek_post_rerun_cases = [
        (
            'John drinks 1.5 bottles during a normal puzzle and 6 during an '
            'extreme puzzle.\n\n**Answer:**\n- Normal Sudoku: 1.5 bottles\n'
            '- Extreme Sudoku: 6 bottles',
            6,
            'marker_labeled_list_last_value',
        ),
        (
            'Jenny paid $30 in bunches and $32 individually.\n'
            '**Conclusion:**\nJenny saved $2.00 compared with buying them '
            'individually.',
            2,
            'terminal_target_amount',
        ),
        (
            'After applying the 10% discount for ordering 4 packs, each '
            'bagel ends up costing $1.00.',
            1,
            'terminal_target_amount',
        ),
        (
            'Each person would get **2 and 2/3 gallons** of root beer.',
            2 + 2 / 3,
            'terminal_mixed_fraction',
        ),
    ]
    for text, want, priority in deepseek_post_rerun_cases:
        rr = skeleton.extract(text, rules_deepseek_r1_distill_32b)
        check('DeepSeek: jawaban eksplisit non-truncated tambahan di-commit',
              rr.status == 'committed'
              and abs(rr.value - want) < 1e-9
              and rr.priority_hit == priority,
              f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    for text in (
        'The problem cannot be solved as it stands. If we assume a price of '
        '$1.50, the result would be $120, but more information is required.',
        'There is insufficient information to determine the result. Assuming '
        '4 legs per animal gives 512 pairs, but the problem cannot be '
        'accurately solved without additional information.',
    ):
        rr = skeleton.extract(text, rules_deepseek_r1_distill_32b)
        check('DeepSeek: respons tak terpecahkan tetap abstain sebagai refusal',
              rr.status == 'abstain' and rr.abstain_reason == 'refusal'
              and rr.value is None,
              f'value={rr.value} reason={rr.abstain_reason}')

    rr = skeleton.extract(
        'After applying the 10% discount for ordering 4 packs, each bagel '
        'ends up costing $1.00.',
        rules_deepseek_r1_distill_32b,
        finish_reason='length',
    )
    check('DeepSeek: fallback baru tidak meloloskan respons truncated',
          rr.status == 'abstain' and rr.abstain_reason == 'truncated',
          f'value={rr.value} reason={rr.abstain_reason}')

    rr = skeleton.extract(
        'Melanie started with **18** vacuum cleaners.', rules_gpt_oss_20b)
    check('GPT-OSS: satu angka dalam deklarasi singkat di-commit',
          rr.value == 18 and rr.priority_hit == 'whole_response_single_value',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'Could you please clarify whether the answer should be 18?',
        rules_gpt_oss_20b,
    )
    check('GPT-OSS: satu angka dalam permintaan klarifikasi tetap abstain',
          rr.priority_hit != 'whole_response_single_value',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        '| Period | Cost | | Intro | $48 | | Normal | $96 | '
        'Adding them together: 48 + 96 + 140 = 284. '
        '**Answer: $284.**',
        rules_gpt_oss_20b,
    )
    check('GPT-OSS: prosa terminal setelah tabel satu-baris diselamatkan',
          rr.value == 284,
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'Start at 40 degrees. The net change is 9 degrees. '
        '**Final temperature: 49 degrees C**',
        rules_gpt_oss_20b,
    )
    check('GPT-OSS: label Final <besaran> terminal di-commit',
          rr.value == 49 and rr.priority_hit == 'terminal_final_label',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract('### Answer:\n**79**', rules_qwen3_next_80b)
    check('Qwen: heading Answer kuat memakai angka tebal berikutnya',
          rr.value == 79 and rr.priority_hit == 'marker_heading_single_bold',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'Given that the first calculation is 12 = 3D, the result is 4. '
        '**Answer:** 16',
        rules_qwen3_next_80b,
    )
    check('Qwen: suffix Answer dari segmen Given diselamatkan',
          rr.value == 16 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'Thus, the amount he has to pay per month is:\n**$200**',
        rules_qwen3_next_80b,
    )
    check('Qwen: heading lemah terminal menerima satu nilai tebal',
          rr.value == 200
          and rr.priority_hit == 'weak_marker_heading_single_bold',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'Thus, values 2 and 3 remain:\n**$200 and $300**',
        rules_qwen3_next_80b,
    )
    check('Qwen: heading lemah multiangka tidak dipromosikan',
          rr.priority_hit != 'weak_marker_heading_single_bold',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'The monthly total is 1128 minutes. '
        '**Answer:**\n18 hours and 48 minutes.',
        rules_qwen3_next_80b,
    )
    check('Qwen: durasi final dikonversi saat total menit sebelumnya cocok',
          rr.value == 1128
          and rr.priority_hit == 'marker_heading_duration_minutes_agreement',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'The earlier estimate is 100 minutes. '
        '**Answer:**\n18 hours and 48 minutes.',
        rules_qwen3_next_80b,
    )
    check('Qwen: durasi tanpa persetujuan total sebelumnya tidak dipromosikan',
          rr.priority_hit != 'marker_heading_duration_minutes_agreement',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

    rr = skeleton.extract(
        '3 loads/week times 52 weeks = 156 loads/year. '
        '### **Final Answer:** Greg does **156 loads of laundry in a year**.',
        rules_ministral_8b,
    )
    check('Ministral: heading Final Answer memakai segmen berikutnya',
          rr.value == 156 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        'The total is 48. **Final Answer:** Mrs. Sherman fed **48 pieces**.',
        rules_ministral_8b,
    )
    check('Ministral: honorific setelah marker boleh berlanjut ke segmen berikut',
          rr.value == 48 and rr.source == 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')
    rr = skeleton.extract(
        '### **Final Answer (Assuming Problem is Correct):**\n'
        'The nurses started with **19 bandages**.',
        rules_ministral_8b,
    )
    check('Ministral: marker berkaveat tidak memakai next-segment recovery',
          rr.source != 'marker_line',
          f'value={rr.value} source={rr.source} priority={rr.priority_hit}')

def test_regional_cultural_late_rescue():
    print('\nRegional Cultural: fallback deklaratif hasil audit')
    cases = [
        (regional_deepseek_r1_distill_32b,
         'Teddy placed 500 puzzle pieces during that hour by completing '
         'the second 500-piece puzzle.',
         500, 'primary_direct_predicate'),
        (regional_deepseek_r1_distill_32b,
         'The computation uses digits earlier. Final Answer: once.',
         1, 'marker_word_number_result'),
        (regional_deepseek_r1_distill_32b,
         'The runners crossed the line. Finley finished in sixth place.',
         6, 'terminal_ordinal_result'),
        (regional_llama31_8b,
         'To solve this problem, we calculate the distance next. '
         'Given that walking speed = 1.5 and running speed = 6 '
         'Total distance = 18 miles Therefore, Dana can travel 18 miles '
         'in six hours.',
         18, 'raw_terminal_connector_'),
        (regional_llama31_8b,
         'To find the distance, we calculate net travel. '
         '**Step 1: Calculate:** 300 - 36 = 264 miles '
         'The final answer is: **264 miles**',
         264, 'raw_strong_marker_first_number'),
        (regional_llama33_70b,
         'I think you are trying to say that six people are infected '
         'after three days.',
         6, 'terminal_reported_word_quantity'),
        (regional_ministral_8b,
         "Let's calculate total feed and trips. Assuming capacity 2250 "
         'for weight 4500, he will need to make 2 full trips to '
         'transport feed.',
         2, 'primary_need_trips'),
        (regional_gpt_oss_20b,
         'Elise sold **65 books** in the second year, earning '
         '\\[ 65 \\text{ books} \\times \\$20 \\text{ per book} '
         '= \\$1{,}300. \\]',
         1300, 'terminal_equation_result'),
    ]
    for pack, text, want, want_priority in cases:
        rr = skeleton.extract(text, pack)
        check(f'{pack.RULE_PACK}: deklarasi {want} dipulihkan',
              rr.status == 'committed' and rr.value == want
              and rr.priority_hit.startswith(want_priority),
              f'status={rr.status} value={rr.value} priority={rr.priority_hit}')

    ambiguous = skeleton.extract(
        'The problem has insufficient information. If a rate is assumed, '
        'the final answer is: 19.',
        regional_llama31_8b,
    )
    check('deklarasi berkaveat tetap abstain',
          ambiguous.status != 'committed',
          f'status={ambiguous.status} value={ambiguous.value}')

    truncated = skeleton.extract(
        'To solve this problem, we calculate the distance next. '
        'Given that walking speed = 1.5 and running speed = 6 '
        'Total distance = 18 miles Therefore, Dana can travel 18 miles '
        'in six hours.',
        regional_llama31_8b,
        finish_reason='length',
    )
    check('fallback baru tidak mempromosikan respons truncated',
          truncated.status == 'abstain'
          and truncated.abstain_reason == 'truncated',
          f'status={truncated.status} reason={truncated.abstain_reason}')


def test_runner_integrity():
    print('\nRunner: invalid JSON, isolation, atomic output')
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'raw'; out = Path(td) / 'out'
        folder = root / 'model' / 'persona'; folder.mkdir(parents=True)
        (folder / 'gsm8k_00001.json').write_text('{invalid', encoding='utf-8')
        result = runner.parse_model(str(root), 'model', 'group', ['persona'],
                                    rules_core, str(out), progress=False)
        with open(result['results_path'], encoding='utf-8', newline='') as fh:
            rows = list(csv.DictReader(fh))
        check('invalid JSON tidak silent skip', len(rows) == 1 and
              rows[0]['input_json_valid'] == 'False' and
              rows[0]['abstain_reason'] == 'invalid_json', str(rows))
        check('temporary atomic dibersihkan', not list(out.glob('.*.tmp')),
              str(list(out.glob('.*.tmp'))))
        model_bytes = (out / 'results.csv').read_bytes()

        good = {'input': {'question_id': 2, 'ground_truth': '#### 7'},
                'response': {'success': True, 'content': 'Final Answer: 7',
                             'raw_api_response': {'choices': [{'finish_reason': 'stop'}]}}}
        (folder / 'gsm8k_00002.json').write_text(json.dumps(good), encoding='utf-8')
        runner.parse_model(str(root), 'model', 'group', ['persona'], rules_core,
                           str(out), progress=False, results_filename='results_core.csv')
        check('core run tidak mengubah results.csv model',
              (out / 'results.csv').read_bytes() == model_bytes and
              (out / 'results_core.csv').exists())

        reparsed = runner.reparse_results(
            str(out / 'results_core.csv'), rules_llama33_70b, str(out),
            results_filename='results_reparsed.csv')
        with open(reparsed['results_path'], encoding='utf-8', newline='') as fh:
            reparsed_rows = list(csv.DictReader(fh))
        good_row = next(r for r in reparsed_rows if r['question_id'] == '2')
        check('reparse CSV memakai content utuh dan mempertahankan jumlah baris',
              len(reparsed_rows) == 2 and good_row['extracted_answer'] == '7',
              str(reparsed_rows))
        check('reparse CSV membersihkan temporary atomic',
              not list(out.glob('.*.tmp')), str(list(out.glob('.*.tmp'))))

        good['input']['ground_truth'] = '#### 8'
        good['response']['content'] = 'Final Answer: 8'
        good['response']['raw_api_response']['choices'][0]['finish_reason'] = 'stop'
        (folder / 'gsm8k_00002.json').write_text(json.dumps(good), encoding='utf-8')
        refreshed = runner.reparse_results(
            str(out / 'results_core.csv'), rules_core, str(out),
            results_filename='results_refreshed.csv',
            raw_root=str(root / 'model'),
            refresh_files={'persona/gsm8k_00002.json'})
        with open(refreshed['results_path'], encoding='utf-8', newline='') as fh:
            refreshed_rows = list(csv.DictReader(fh))
        refreshed_good = next(r for r in refreshed_rows if r['question_id'] == '2')
        check('selective refresh mengambil content JSON terbaru',
              refreshed_good['extracted_answer'] == '8' and
              refreshed_good['ground_truth_answer'] == '8', str(refreshed_good))
        check('selective refresh tetap atomic', not list(out.glob('.*.tmp')),
              str(list(out.glob('.*.tmp'))))

        final = out / 'atomic.csv'; final.write_bytes(b'OLD_FINAL')
        real_replace = runner.os.replace
        observed = []
        def observe_replace(src, dst):
            observed.append(Path(dst).read_bytes())
            real_replace(src, dst)
        with mock.patch.object(runner.os, 'replace', side_effect=observe_replace):
            runner._write_csv_atomic(str(final), ['x'], [{'x': 'new'}])
        check('old final visible sampai os.replace', observed == [b'OLD_FINAL'], str(observed))
        check('atomic replace mempublikasikan file lengkap', final.read_text().strip() == 'x\nnew')


# ── V2: no-gold-peek ────────────────────────────────────────────────────────
def test_v2_no_gold_peek():
    print('\nV2 no-gold-peek')
    sig = inspect.signature(skeleton.extract)
    bad = [p for p in sig.parameters
           if re.search(r'gt|gold|ground_truth|answer_key', p, re.I)]
    check('extract() tidak menerima parameter ground truth', not bad, str(bad))

    src = code_only(CORE_DIR / 'skeleton.py') + code_only(CORE_DIR / 'priorities.py')
    for pack in PACKS:
        src += code_only(Path(inspect.getfile(pack)))
    hits = re.findall(r'\b(gt_num|ground_truth|gold|answer_key)\b', src)
    check('modul ekstraksi tidak menyebut ground truth', not hits, str(set(hits)))


# ── V6: larangan literal ID soal ────────────────────────────────────────────
def test_v6_no_qid_literals():
    print('\nV6 larangan rescue per-QID')
    for f in list(RULES_DIR.glob('*.py')) + list(CORE_DIR.glob('*.py')):
        hits = re.findall(r'gsm8k_\d{4,5}', code_only(f))
        check(f'{f.name} bebas literal ID soal', not hits, str(hits[:5]))


# ── V7: rule pack persona-blind ─────────────────────────────────────────────
def test_v7_persona_blind():
    print('\nV7 rule pack tidak boleh bergantung persona')
    # Cari NILAI persona di kode aktif, bukan kata 'persona' di dokumentasi.
    words = (r'\b(islam|muslim|christian|atheist|jewish|'
             r'\w*_explicit\w*|\w*_implicit\w*|man_\w+|woman_\w+)\b')
    for f in list(RULES_DIR.glob('rules_*.py')) + list(CORE_DIR.glob('*.py')):
        hits = re.findall(words, code_only(f), re.I)
        check(f'{f.name} bebas nama persona', not hits, str(set(h.lower() for h in hits)))

    text = 'So, Janet makes $18 every day at the farmers\' market.'
    for pack in PACKS:
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / 'raw', Path(td) / 'out'
            personas = [f'p{i:02d}' for i in range(16)]
            payload = {'input': {'question_id': 1, 'ground_truth': '#### 18'},
                       'response': {'success': True, 'content': text,
                                    'raw_api_response': {'choices': [{'finish_reason': 'stop'}]}}}
            for persona in personas:
                folder = raw / 'model' / persona; folder.mkdir(parents=True)
                (folder / 'gsm8k_00001.json').write_text(json.dumps(payload), encoding='utf-8')
            result = runner.parse_model(str(raw), 'model', 'group', personas, pack,
                                        str(out), progress=False)
            with open(result['results_path'], encoding='utf-8', newline='') as fh:
                rows = list(csv.DictReader(fh))
            vals = {(r['persona'], r['extracted_answer'], r['parse_source']) for r in rows}
            outputs = {(v, s) for _, v, s in vals}
            check(f'{pack.RULE_PACK}: runner 16 persona menghasilkan output identik',
                  len(rows) == 16 and len(outputs) == 1, str(vals))


if __name__ == '__main__':
    test_v5_numbers()
    test_v5_extract()
    test_real_bug_regressions()
    test_regional_cultural_late_rescue()
    test_runner_integrity()
    test_v2_no_gold_peek()
    test_v6_no_qid_literals()
    test_v7_persona_blind()
    print('\n' + '=' * 60)
    if failures:
        print(f'GAGAL: {len(failures)}')
        for f in failures:
            print('  -', f)
        sys.exit(1)
    print('SEMUA UJI LULUS')
