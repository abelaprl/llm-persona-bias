"""Test extraction accuracy across many files"""
import json
import re
from pathlib import Path
from collections import defaultdict

def extract_final_answer(response_content):
    if not response_content:
        return None
    
    text = response_content.strip()
    
    # Priority 1: boxed pattern - handle with/without %, \text{}, etc., take LAST match
    # Handle \boxed{60\%}, \boxed{160\ \text{minutes}}, \boxed{45\text{ miles}}
    boxed_matches = list(re.finditer(r'\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)[^}]*\}', text))
    if boxed_matches:
        ans = boxed_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    
    # Priority 2: #### pattern - take LAST match
    hash_matches = list(re.finditer(r'####\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text))
    if hash_matches:
        ans = hash_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    
    # Priority 3: Look for "Answer" line and extract carefully
    lines = text.strip().split('\n')
    last_lines = lines[-15:] if len(lines) >= 15 else lines
    
    for line in reversed(last_lines):
        line_stripped = line.strip()
        
        # Check if this is an Answer line
        if re.search(r'^\*?\*?[Aa]nswer', line_stripped) or re.search(r'\*\*[Aa]nswer', line_stripped):
            # Pattern 1: "Answer: $NUMBER" or "**Answer: $NUMBER**"
            match = re.search(r'[Aa]nswer[:\s\*]*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 2: "Answer: NUMBER" directly after colon  
            match = re.search(r'[Aa]nswer[:\s\*]+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 3: "have NUMBER unit" in answer line
            match = re.search(r'\bhave\s+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sheep|miles?|dollars?|eggs?|items?|points?|downloads?|pages?|books?|apples?)\b', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 4: "is NUMBER unit" - get NUMBER before unit  
            match = re.search(r'\bis\s+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:miles?|dollars?|eggs?|vacuum\s*cleaners?|glasses?|items?|people|students?|apples?|books?|pages?|points?|pieces?|downloads?|minutes?|sheep)\b', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 5: "pay $NUMBER"
            match = re.search(r'\bpay\s+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 6: Bold number **NUMBER** in answer line (any position)
            bold_matches = re.findall(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', line_stripped)
            if bold_matches:
                return float(bold_matches[-1].replace(',', ''))
            
            # Pattern 7: "give (her chickens)? NUMBER cups/unit" - for chicken feed problem
            match = re.search(r'\bgive\s+(?:her\s+)?(?:chickens?\s+)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|bolts?|items?|pieces?|dollars?)\b', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 8: "needs to give NUMBER" 
            match = re.search(r'\bneeds?\s+to\s+give\s+(?:her\s+)?(?:chickens?\s+)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|bolts?|items?|pieces?)\b', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 9: Generic - find LAST "NUMBER unit" in Answer line (fallback)
            all_unit_matches = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|items?|dollars?|miles?|eggs?|pieces?|minutes?|hours?|days?|years?|meters?|bolts?|sheep|glasses?|vacuum\s*cleaners?|downloads?|pages?|books?|people|students?|feet|foot|centimeters?|cm|inches?|inch|shorter|longer|times?|laps?|boxes?|bags?|cans?|bottles?|pairs?|sets?|rows?|groups?|packs?|slices?|pounds?|kilograms?|kg|grams?|g|calories?|liters?|gallons?|ounces?|oz|seconds?|weeks?|months?|cars?|trucks?|bikes?|trees?|flowers?|cookies?|candies?|toys?|balls?|marbles?|stickers?|pencils?|pens?|crayons?|tickets?|coins?|bills?|points?|goals?|runs?|hits?|wins?|losses?|games?|matches?|rounds?|levels?|floors?|rooms?|houses?|apartments?|buildings?|blocks?|streets?|roads?|ways?|paths?|steps?|stairs?|doors?|windows?|walls?|tables?|chairs?|desks?|beds?|sofas?|shelves?|drawers?|boxes?|baskets?|buckets?|jars?|plates?|bowls?|cups?|mugs?|forks?|spoons?|knives?)\b', line_stripped, re.IGNORECASE)
            if all_unit_matches:
                return float(all_unit_matches[-1].replace(',', ''))
    
    # Priority 4: Look for "The answer is NUMBER"
    for line in reversed(last_lines):
        match = re.search(r'[Tt]he\s+answer\s+is[:\s]*\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 5: Bold number **NUMBER unit** anywhere in last 5 lines
    for line in reversed(last_lines[-5:]):
        # Bold number followed by common units (extended list)
        match = re.search(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|sheep|miles?(?:\s+per\s+hour)?|dollars?|minutes?|downloads?|eggs?|pages?|books?|items?|pieces?|hours?|days?|meters?|dozens?|years?|bolts?|feet|foot|centimeters?|cm|inches?|shorter|longer|times?|laps?|boxes?|bags?|pounds?|kg|grams?|g|liters?|gallons?|seconds?|weeks?|months?|cookies?|candies?|marbles?|stickers?|pencils?|tickets?|coins?|points?|floors?|steps?|calories?|%|percent)?\*\*', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 6: "makes/made/profit of $NUMBER" / "speed of NUMBER" patterns
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b(?:makes?|made|profit\s+of|speed\s+of)\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 7: "needs NUMBER" / "need NUMBER" / "have NUMBER" / "takes NUMBER" patterns 
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b(?:needs?|have|has|gets?|total\s+of|runs?\s+a\s+total\s+of|run|eats?|will\s+eat|takes?|pay\s+a\s+total\s+of|it\s+takes|used\s+a\s+total\s+of|used|are|there\s+are)\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:cups?|sheep|miles?|dollars?|minutes?|hours?|downloads?|eggs?|pages?|books?|items?|pieces?|meters?|dozens?|bolts?|years?|yards?|burritos?|pounds?|lb|students?|people|leftover)?\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 8: "is/are NUMBER unit" with extended units
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b(?:is|are)\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:cups?|sheep|miles?(?:\s+per\s+hour)?|dollars?|minutes?|hours?|downloads?|eggs?|pages?|books?|items?|pieces?|glasses?|points?|people|students?|meters?|dozens?|yards?|burritos?|pounds?|lb|leftover)\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 9: = NUMBER unit (not just at end)
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|downloads?|sheep|meters?|cups?|bolts?|minutes?|years?)', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 10: = NUMBER at end of line  
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 11: "give NUMBER" pattern
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\bgive\s+(?:her\s+)?(?:chickens?\s+)?\*?\*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:cups?|bolts?|items?|pieces?)\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 12: "distance/salary/profit/earnings is/are $NUMBER" patterns
    for line in reversed(last_lines[-5:]):
        match = re.search(r'(?:distance|salary|profit|earnings?|would\s+be)\s+(?:is|are|of)?\s*\*?\*?\\?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 13: "covers NUMBER miles/distance" pattern
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\bcovers?\s+(?:a\s+)?(?:total\s+)?(?:distance\s+of\s+)?\*?\*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:miles?|meters?|km)\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 14: "NUMBER%" pattern for percentages 
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 15: "Answer: \(NUMBER\)" LaTeX inline math
    for line in reversed(last_lines[-5:]):
        match = re.search(r'[Aa]nswer[:\s]*\\\(([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\\\)', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 16: "= NUMBER m." or "= NUMBER m" at end of line
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:m|meters?)\.?\s*\*?\*?\s*$', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 17: "NUMBER\ \text{min}" LaTeX format
    for line in reversed(last_lines[-5:]):
        match = re.search(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\\?\s*\\text\{', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 18: escaped dollar \$NUMBER pattern
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\\?\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)[.\s]*(?:$|[^0-9])', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 19: "began with **NUMBER" pattern
    for line in reversed(last_lines[-5:]):
        match = re.search(r'began\s+with\s+\*?\*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    return None
    
    return None

def extract_ground_truth(ground_truth_text):
    if not ground_truth_text:
        return None
    match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', ground_truth_text)
    if match:
        return float(match.group(1).replace(',', ''))
    return None

def main():
    import random
    random.seed(42)  # Reproducible random selection
    
    result_dir = Path('result')
    models = sorted([d for d in result_dir.iterdir() if d.is_dir() and d.name not in ['testing', 'reports']])
    
    # 8 personas to test
    PERSONAS = [
        'man_explicit_1',
        'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
        'woman_explicit_1',
        'woman_implicit_1', 'woman_implicit_2', 'woman_implicit_3'
    ]
    
    FILES_PER_PERSONA = 100  # 100 random files per persona
    
    all_results = []
    
    for model in models:
        for persona_name in PERSONAS:
            persona_dir = model / persona_name
            if not persona_dir.exists():
                continue
            
            # Get all JSON files and randomly select 100
            all_json_files = list(persona_dir.glob('gsm8k_*.json'))
            if len(all_json_files) == 0:
                continue
            
            # Random sample (or all if less than 100)
            sample_size = min(FILES_PER_PERSONA, len(all_json_files))
            json_files = random.sample(all_json_files, sample_size)
            
            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    gt_text = data.get('input', {}).get('ground_truth', '')
                    gt = extract_ground_truth(gt_text)
                    
                    response = data.get('response', {})
                    content = response.get('content', '')
                    
                    # Fallback to reasoning if content empty
                    if not content:
                        raw = response.get('raw_api_response', {})
                        choices = raw.get('choices', [])
                        if choices:
                            msg = choices[0].get('message', {})
                            content = msg.get('content', '') or msg.get('reasoning', '')
                    
                    extracted = extract_final_answer(content)
                    
                    is_correct = False
                    if extracted is not None and gt is not None:
                        is_correct = abs(extracted - gt) < 0.01
                    
                    all_results.append({
                        'model': model.name,
                        'persona': persona_name,
                        'file': jf.name,
                        'gt': gt,
                        'extracted': extracted,
                        'correct': is_correct,
                        'content_end': content[-800:] if content else ''
                    })
                except Exception as e:
                    pass
    
    # Summary per model
    print("="*80)
    print("EXTRACTION TEST RESULTS")
    print("="*80)
    
    model_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'no_extract': 0})
    
    for r in all_results:
        model_stats[r['model']]['total'] += 1
        if r['correct']:
            model_stats[r['model']]['correct'] += 1
        if r['extracted'] is None:
            model_stats[r['model']]['no_extract'] += 1
    
    print(f"\n{'Model':<45} {'Correct':<8} {'NoExtract':<10} {'Total':<8} {'Accuracy':<10}")
    print("-"*85)
    
    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        acc = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{model:<45} {stats['correct']:<8} {stats['no_extract']:<10} {stats['total']:<8} {acc:.1f}%")
    
    total_correct = sum(s['correct'] for s in model_stats.values())
    total_no_extract = sum(s['no_extract'] for s in model_stats.values())
    total_all = sum(s['total'] for s in model_stats.values())
    print("-"*85)
    print(f"{'TOTAL':<45} {total_correct:<8} {total_no_extract:<10} {total_all:<8} {100*total_correct/total_all:.1f}%")
    
    # Show cases where no answer was extracted
    no_extract = [r for r in all_results if r['extracted'] is None]
    print(f"\n\nNO ANSWER EXTRACTED: {len(no_extract)}")
    print("-"*75)
    for r in no_extract[:8]:
        print(f"\n{r['model'][:30]} / {r['file']}")
        print(f"  GT: {r['gt']}")
        print(f"  End: ...{r['content_end'][-100:]}")
    
    # Show some mismatches
    mismatches = [r for r in all_results if not r['correct'] and r['extracted'] is not None]
    print(f"\n\nMISMATCHES (extracted != ground_truth): {len(mismatches)}")
    print("-"*75)
    
    # Group by type of mismatch
    extraction_issues = []  # Cases where extraction might be wrong
    model_errors = []       # Cases where model gave wrong answer
    
    for r in mismatches:
        content = r['content_end']
        extracted = r['extracted']
        gt = r['gt']
        
        # Check if extracted number appears near end (likely correct extraction, model error)
        if str(int(extracted)) in content[-200:]:
            model_errors.append(r)
        else:
            extraction_issues.append(r)
    
    print(f"\n\n=== LIKELY MODEL ERRORS (extracted number in response): {len(model_errors)} ===")
    for r in model_errors[:10]:
        print(f"\n{'='*80}")
        print(f"FILE: {r['model']} / {r['persona']} / {r['file']}")
        print(f"GROUND TRUTH: {r['gt']}")
        print(f"EXTRACTED: {r['extracted']}")
        print(f"\nRESPONSE ENDING (last 500 chars):")
        content = r['content_end'][-500:] if len(r.get('content_end','')) > 0 else 'EMPTY'
        # Remove problematic unicode chars
        content = content.encode('ascii', 'replace').decode('ascii')
        print(content)
    
    print(f"\n\n=== POSSIBLE EXTRACTION ISSUES: {len(extraction_issues)} ===")
    for r in extraction_issues[:10]:
        print(f"\n{'='*80}")
        print(f"FILE: {r['model']} / {r['persona']} / {r['file']}")
        print(f"GROUND TRUTH: {r['gt']}")
        print(f"EXTRACTED: {r['extracted']}")
        print(f"\nRESPONSE ENDING (last 500 chars):")
        content = r['content_end'][-500:] if len(r.get('content_end','')) > 0 else 'EMPTY'
        content = content.encode('ascii', 'replace').decode('ascii')
        print(content)

if __name__ == "__main__":
    main()
