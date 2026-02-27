from scripts.rebuild_results_csv import normalize_number, extract_final_answer


def test_answer_marker_comma():
    text = "Answer: 3,000 quarts ... 48 hours"
    val, reason = extract_final_answer(text)
    assert val == 3000


def test_boxed_latex_thousands():
    text = "Some work... \\\boxed{$140{,}000}$"
    val, reason = extract_final_answer(text)
    assert val == 140000


def test_boxed_percent():
    text = "Result: \\\boxed{12%}"
    val, reason = extract_final_answer(text)
    assert val == 12


def test_minutes_preference():
    text = "95 minutes — that's 1 hour 35 minutes"
    val, reason = extract_final_answer(text)
    assert val == 95


def test_total_parenthetical_ie():
    text = "Total time = (40 + 20 + 100 = 160) minutes, i.e. 2 hours 40 minutes"
    val, reason = extract_final_answer(text)
    assert val == 160
from scripts.rebuild_results_csv import normalize_number, extract_final_answer


def test_normalize_percent_and_latex_commas():
    assert normalize_number(r'\boxed{$140{,}000}') == 140000
    assert normalize_number(r'\boxed{12%}') == 12


def test_extract_cases():
    t1 = 'Answer: 3,000 quarts ... 48 hours'
    assert extract_final_answer(t1)[0] == 3000

    t2 = r'\boxed{$140{,}000}'
    assert extract_final_answer(t2)[0] == 140000

    t3 = r'\boxed{12%}'
    assert extract_final_answer(t3)[0] == 12

    t4 = "95 minutes — that's 1 hour 35 minutes"
    assert extract_final_answer(t4)[0] == 95

    t5 = 'Total time = (40 + 20 + 100 = 160) minutes, i.e. 2 hours 40 minutes'
    assert extract_final_answer(t5)[0] == 160
