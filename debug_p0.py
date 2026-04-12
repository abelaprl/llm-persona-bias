import sys, re
sys.path.insert(0, 'src')
from rebuild_results_csv_llama import normalize_number

NUMBER_RE = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'

_ANSWER_VERB_RE = re.compile(
    r'\b(?:'
    r'will\s+(?:take|cost|last|be|earn|save|spend|make|have|need|weigh|give|receive|pay|create|score)|'
    r'would\s+(?:take|cost|last|be|earn|save|spend|make|have|need|weigh|pay|create|score)|'
    r'is|are|was|were|costs?|earns?|saves?|takes?|makes?|needs?|weighs?|spends?|'
    r'pays?|paid|creates?|created|scores?|scored|spent|brought|can\s+\w+|'
    r'remains?|add\s+up\s+to|'
    r'had|got|gotten|collected?|picked?|won|saved?|bought|ran|walked?|watched?|'
    r'responds?\s+to|responded?\s+to|finished?|completed?|solved?|reached?|'
    r'arrived?|found|learns?|learned?|grew|planted?|painted?|wrote|written|drew|'
    r'sang|cooked?|baked?|cleaned?|fixed?|built|sold|hired?|owned?|holds?|held|'
    r'planted?|visited?|drove|driven|flew|flown|swam|swum|fell|risen?|rose'
    r')\b', re.I)

sentence = "3 trucks will be needed to transport 80 flagstones in one trip."
print('Sentence:', repr(sentence))

_first_sent = sentence.split('.', 1)[0].strip()
print('_first_sent:', repr(_first_sent))

_first_lead_m = re.match(r'^(?:\*\*)?\s*(' + NUMBER_RE + r')\s+\w+', _first_sent)
print('_first_lead_m:', _first_lead_m)
if _first_lead_m:
    print('  group(1):', _first_lead_m.group(1))
    print('  start(1):', _first_lead_m.start(1))

_verb_m_p0 = _ANSWER_VERB_RE.search(_first_sent)
print('_verb_m_p0:', _verb_m_p0)
if _verb_m_p0:
    print('  match:', _verb_m_p0.group(), 'at', _verb_m_p0.start())

if _first_lead_m:
    _verb_before_num = _verb_m_p0 and _verb_m_p0.start() < _first_lead_m.start(1)
    print('_verb_before_num:', _verb_before_num)
    if not _verb_before_num:
        n0 = normalize_number(_first_lead_m.group(1))
        print('PRIORITY 0 RETURN:', n0)
