import json, os, re, random

MODEL_DIR = 'result/openai_gpt-oss-120b'
PERSONAS = sorted([p for p in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR,p))])

# copy of improved extractor from analyze_model34.py

def extract_final_answer(response_content):
    if not response_content:
        return None
    text = response_content.strip()
    # boxed
    boxed_matches = list(re.finditer(r'\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)[^}]*\}', text))
    if boxed_matches:
        ans = boxed_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    # ####
    hash_matches = list(re.finditer(r'####\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text))
    if hash_matches:
        ans = hash_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    # Answer lines
    lines = text.strip().split('\n')
    last_lines = lines[-15:] if len(lines) >= 15 else lines
    for line in reversed(last_lines):
        line_stripped = line.strip()
        if re.search(r'^\*?\*?[Aa]nswer', line_stripped) or re.search(r'\*\*[Aa]nswer', line_stripped):
            match = re.search(r'[Aa]nswer[:\s\*]*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped)
            if match:
                return float(match.group(1).replace(',', ''))
            match = re.search(r'[Aa]nswer[:\s\*]+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped)
            if match:
                return float(match.group(1).replace(',', ''))
            bold_matches = re.findall(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', line_stripped)
            if bold_matches:
                return float(bold_matches[-1].replace(',', ''))
    # The answer is
    for line in reversed(last_lines):
        match = re.search(r'[Tt]he\s+answer\s+is[:\s]*\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if match:
            return float(match.group(1).replace(',', ''))
    # Bold number
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|sheep|miles?|dollars?|minutes?|hours?|downloads?|eggs?|pages?|books?|items?|pieces?|days?|meters?|dozens?|years?|bolts?|feet|foot|centimeters?|cm|inches?|%|percent)?\*\*', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    # is/are NUMBER unit
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b(?:is|are)\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:cups?|sheep|miles?|dollars?|minutes?|hours?|downloads?|eggs?|pages?|books?|items?|pieces?|glasses?|points?|people|students?|meters?|dozens?|yards?|burritos?|pounds?|lb|leftover)\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    # = NUMBER at end
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    # bold/dollar anywhere
    bold_dollar = re.findall(r'\*\*\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\*\*', text)
    if bold_dollar:
        try:
            return float(bold_dollar[-1].replace(',', ''))
        except:
            pass
    # last standalone number
    last_num = None
    for m in re.finditer(r'\$?([+-]?\d{1,3}(?:[\d,]*\d)?(?:\.\d+)?)', text):
        last_num = m.group(1)
    if last_num:
        try:
            return float(last_num.replace(',', ''))
        except:
            pass
    return None


def extract_ground_truth(ground_truth_text):
    if not ground_truth_text:
        return None
    m = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', ground_truth_text)
    if m:
        return float(m.group(1).replace(',', ''))
    return None


def main():
    wrongs = []
    for persona in PERSONAS:
        persona_path = os.path.join(MODEL_DIR, persona)
        if not os.path.isdir(persona_path):
            continue
        for fname in sorted([f for f in os.listdir(persona_path) if f.endswith('.json')]):
            fpath = os.path.join(persona_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                continue
            content = data.get('response', {}).get('content', '')
            gt_text = data.get('input', {}).get('ground_truth', '')
            gt = extract_ground_truth(gt_text)
            extracted = extract_final_answer(content)
            if gt is None:
                continue
            if extracted is None:
                continue
            # compare numerically allowing small float differences
            try:
                if float(extracted) != float(gt):
                    snippet = (content or '')[:300].replace('\n',' ')[:300]
                    wrongs.append((persona, fname, gt, extracted, snippet))
            except Exception:
                continue
    if not wrongs:
        print('No wrong examples found')
        return
    sample = wrongs
    if len(wrongs) > 100:
        sample = random.sample(wrongs, 100)
    out_path = 'wrong_examples_model34.txt'
    with open(out_path, 'w', encoding='utf-8') as out:
        for p,f,gt,ex,sn in sample:
            line = f"{p}/{f} | GT: {gt} | Extracted: {ex} | Snippet: {sn}\n"
            out.write(line)
    print(f'Wrote {len(sample)} examples to {out_path}')

if __name__ == '__main__':
    main()
