import os, json, re
from collections import defaultdict

MODEL_DIR = 'result/openai_gpt-oss-120b'
OUT_LEN = 'lenlim_files_model34.txt'
OUT_TRUNC_WRONG = 'truncated_wrong_model34.txt'

# reuse extractor and truncation heuristics

def extract_final_answer(response_content):
    if not response_content:
        return None
    text = response_content.strip()
    boxed_matches = list(re.finditer(r'\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)[^}]*\}', text))
    if boxed_matches:
        ans = boxed_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    hash_matches = list(re.finditer(r'####\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text))
    if hash_matches:
        ans = hash_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    lines = text.strip().split('\n')
    last_lines = lines[-15:] if len(lines)>=15 else lines
    for line in reversed(last_lines):
        ls = line.strip()
        if re.search(r'^\*?\*?[Aa]nswer', ls) or re.search(r'\*\*[Aa]nswer', ls):
            m = re.search(r'[Aa]nswer[:\s\*]*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', ls)
            if m:
                return float(m.group(1).replace(',', ''))
            m = re.search(r'[Aa]nswer[:\s\*]+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', ls)
            if m:
                return float(m.group(1).replace(',', ''))
            bold = re.findall(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', ls)
            if bold:
                return float(bold[-1].replace(',', ''))
    for line in reversed(last_lines):
        m = re.search(r'[Tt]he\s+answer\s+is[:\s]*\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if m:
            return float(m.group(1).replace(',', ''))
    for line in reversed(last_lines[-5:]):
        m = re.search(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|dollars?|minutes?|hours?|pages?|items?|pieces?|days?|meters?|%|percent)?\*\*', line, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(',', ''))
    for line in reversed(last_lines[-5:]):
        m = re.search(r'\b(?:is|are)\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:dollars?|minutes?|hours?|items?|people|meters|yards|pounds|lb)\b', line, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(',', ''))
    for line in reversed(last_lines[-5:]):
        m = re.search(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$', line, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(',', ''))
    bd = re.findall(r'\*\*\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\*\*', text)
    if bd:
        try:
            return float(bd[-1].replace(',', ''))
        except:
            pass
    last_num = None
    for m in re.finditer(r'\$?([+-]?\d{1,3}(?:[\d,]*\d)?(?:\.\d+)?)', text):
        last_num = m.group(1)
    if last_num:
        try:
            return float(last_num.replace(',', ''))
        except:
            pass
    return None


def is_truncated(text):
    if not text:
        return True, 'EMPTY'
    text = text.strip()
    if not text:
        return True, 'EMPTY'
    stripped = re.sub(r'[\*`\$\s]+$', '', text)
    stripped = re.sub(r'\\\]|\\\)$', '', stripped)
    if not stripped:
        return True, 'EMPTY_AFTER_STRIP'
    last_char = stripped[-1]
    ok_endings = set('.!?)\]"\'0123456789%')
    if last_char in ok_endings:
        return False, 'OK'
    if re.search(r'\$?\d+(?:,\d{3})*(?:\.\d+)?\s*\*\*?\s*$', text):
        return False, 'OK_NUM_WITH_MARKUP'
    return True, f"ENDS_WITH: '{text[-30:]}'"


def extract_ground_truth(gt_text):
    if not gt_text:
        return None
    m = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', gt_text)
    if m:
        return float(m.group(1).replace(',', ''))
    return None


lenlim = []
trunc_wrong = []

for persona in sorted(os.listdir(MODEL_DIR)):
    pdir = os.path.join(MODEL_DIR, persona)
    if not os.path.isdir(pdir):
        continue
    for fname in sorted([f for f in os.listdir(pdir) if f.endswith('.json')]):
        fpath = os.path.join(pdir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            continue
        raw = data.get('response', {}).get('raw_api_response', {})
        choices = raw.get('choices', [])
        finish_reason = ''
        if choices:
            finish_reason = choices[0].get('finish_reason', '')
        if finish_reason == 'length':
            lenlim.append((persona, fname, finish_reason))
        content = data.get('response', {}).get('content', '')
        gt_text = data.get('input', {}).get('ground_truth', '')
        gt = extract_ground_truth(gt_text)
        extracted = extract_final_answer(content)
        is_tr, info = is_truncated(content)
        # consider truncated-wrong only when heuristic says truncated and extracted exists and doesn't equal GT
        if is_tr and extracted is not None and gt is not None and float(extracted) != float(gt):
            trunc_wrong.append((persona, fname, info, gt, extracted, content[:300].replace('\n',' ')))

# Write outputs
with open(OUT_LEN, 'w', encoding='utf-8') as out:
    for p,f,fr in lenlim:
        out.write(f"{p}/{f} | finish_reason={fr}\n")

with open(OUT_TRUNC_WRONG, 'w', encoding='utf-8') as out:
    for p,f,info,gt,ex,sn in trunc_wrong:
        out.write(f"{p}/{f} | {info} | GT:{gt} | EX:{ex} | Snippet: {sn}\n")

print(f'LenLim count: {len(lenlim)} -> wrote {OUT_LEN}')
print(f'Truncated+Wrong count: {len(trunc_wrong)} -> wrote {OUT_TRUNC_WRONG}')
