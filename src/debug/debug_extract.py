import re, sys
sys.stdout.reconfigure(encoding='utf-8')

NUMBER_RE = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'

test_cases = [
    "Therefore, Allen's age 10 years from now is 109.",
    "Therefore, 160 flowers in Arianna's garden have bloomed.",
    "Therefore, it will take Lauren 22 days to create all of the picture scenes.",
    "Therefore, Vanessa took 64 minutes to complete the test.",
    "Therefore, Frankie watched 3 episodes of the 30-minute show on Wednesday.",
    "So, James responds to 320 emails in a 5-day work week.",
    "Rounded to the nearest integer, the percentage chance is 33%.",
    "So, Tony, Bobby, Ricky, and Kathy can pick 29 quarts of strawberries per hour on Grandma Concetta's farm.",
    "Therefore, 9 of the cards had the letter D on them.",
    "approximately 25 candles will both smell like wet dog and explode",
    "Therefore, 5 loaves of bread are sold in the afternoon.",
    "So, 2 Lego sets still remain to be built.",
]

for line_s in test_cases:
    print(f"\nLine: {repr(line_s)}")
    all_nums = re.findall(NUMBER_RE, line_s)
    print(f"  all_nums: {all_nums} (count={len(all_nums)})")
    
    m_there = re.search(
        r'(?:There\s+(?:are|is)|\w+\s+(?:has|have|can|will|got|gets?|contains?|holds?|needs?|takes?|makes?|earns?|costs?|weighs?|measures?|equals?|totals?))\s+(' + NUMBER_RE + r')\b',
        line_s, re.I
    )
    print(f"  m_there: {m_there.group(1) if m_there else None}")
    
    m_start = re.match(r'^(' + NUMBER_RE + r')\s+\w', line_s)
    print(f"  m_start: {m_start.group(1) if m_start else None}")
    
    m_sent = re.search(r'(' + NUMBER_RE + r')\s+(?:\w+\s+){0,6}(?:\w+)\s*[.!]?\s*$', line_s)
    print(f"  m_sent: {m_sent.group(1) if m_sent else None}")
    
    m_plain = re.match(r'^((?:[-\u2212\u2013]\s*)?(?:[\$])?(?:\d{1,3}(?:(?:,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?)\s*(?:[°\u00b0a-zA-Z\.%]*)?\s*$', line_s)
    print(f"  m_plain: {m_plain.group(1) if m_plain else None}")
