# GSM8K Extraction Error Analysis - Quick Reference

## Problem Breakdown

```
20 Total Error Cases Analyzed
│
├─ 14 cases (70%): EXPECTED_NOT_IN_CANDIDATES
│  └─ Expected number missing from regex extraction
│     ├─ Root cause: Split calculations (1 * 4 * 1 * 2 → captures [1,4,1,2] not 1412)
│     ├─ Root cause: LaTeX/formatting issues (1{,}412 not parsed)
│     └─ Fix: Enhanced regex + answer-sentence parsing
│
├─ 2 cases (10%): WRONG_INDEX
│  └─ Expected in candidates but wrong position selected
│     ├─ Issue: Extracting early intermediate instead of final
│     └─ Fix: Positional weighting + keyword detection
│
├─ 2 cases (10%): EXTRACTED_LAST
│  └─ Last-number heuristic fired but answer was first/middle
│     ├─ Issue: "total cost is $21... plus $10" → picked $10
│     └─ Fix: Semantic keyword priority over position
│
├─ 1 case (5%): EXTRACTED_FIRST
│  └─ First-number heuristic fired but answer was later
│     └─ Fix: Same as EXTRACTED_LAST
│
└─ 1 case (5%): NO_EXTRACTION
   └─ Extraction returned None for expected=3
      └─ Fix: Fallback word-to-digit + full-text scan
```

---

## Top 3 Failure Patterns

### Pattern 1: EXPECTED_NOT_IN_CANDIDATES (14 cases, 70%)

**What's happening:**
- Model output: "95 × 10 + 3 = 953"
- Regex finds: [95, 10, 3]
- Expected: 953
- Extracted: First/last/random from [95, 10, 3] ❌

**Why it matters:** Dominates all errors

**Quick fix:**
```
1. Check for patterns like "X * Y + Z" 
2. Try to evaluate: 95 * 10 + 3 = 953
3. If result makes sense, use it
```

**Real examples:**
- `gsm8k_00175`: Expected 953, model shows "95 × 10 + 3"
- `gsm8k_00325`: Expected 1412, model shows "14 × 100 + 12"
- `gsm8k_00006`: Expected 646, model shows "64 × 10 + 6"

---

### Pattern 2: WRONG_INDEX (2 cases, 10%)

**What's happening:**
- Candidates: [5 (index 0), 2000 (index 1), ..., 114200 (index 18)]
- Extracted: 5 ❌
- Expected: 114200 ✓

**Why it matters:** Early numbers look plausible; extraction stops there

**Quick fix:**
```
1. Prefer numbers near end of text (last 30%)
2. Multiply end-position scores by 2-3x
3. Use keyword "total" or "final" for tie-breaking
```

**Real examples:**
- `gsm8k_00202`: $5 early vs $114,200 final
- `gsm8k_00018`: $35,000 at index 7 vs $57,500 at index 21

---

### Pattern 3: EXTRACTED_LAST (2 cases, 10%)

**What's happening:**
- Text: "total cost is $29.00... delivery fee $5.00"
- Candidates: [29, 5] (in order)
- Extracted: 5 (last) ❌
- Expected: 29 (marked as "total") ✓

**Why it matters:** Position heuristic ignores semantic meaning

**Quick fix:**
```
1. Before picking "last number"
2. Check for keywords: "total", "final", "answer", "result"
3. If found, use that number instead
```

**Real examples:**
- `gsm8k_00357`: "Dean has $21 left" but picked later "$10"
- `gsm8k_00138`: "total cost is $29" but picked later "$5"

---

## Extraction Rule Improvements (Priority Order)

### 🔴 IMMEDIATE (Fixes 70%)

#### 1. Enhanced NUMBER_RE
**Current problem:** Misses numbers split by operators
```regex
# Current (too restrictive)
\d+(?:\.\d+)?

# Needed (capture more)
# Include: compound patterns, scientific notation, fractions
```

**Impact:** ~50% of EXPECTED_NOT_IN_CANDIDATES cases

#### 2. Answer-Sentence Parsing
**Current problem:** Doesn't look for "The answer is X" format
```python
# Add after standard rules fail:
patterns = [
    r'answer\s+(?:is|=):\s*(\d+)',
    r'total\s+(?:is|=):\s*(\d+)',
    r'final\s+(?:is|=):\s*(\d+)',
]
```

**Impact:** ~30% of EXPECTED_NOT_IN_CANDIDATES cases

#### 3. Arithmetic Expression Evaluation
**Current problem:** Captures "95", "10", "3" but not "95*10+3=950+3=953"
```python
# Detect: X OP Y pattern and evaluate
if re.match(r'\d+\s*[*+-/]\s*\d+'):
    result = eval(...)
    add_to_candidates(result)
```

**Impact:** ~20% of EXPECTED_NOT_IN_CANDIDATES cases

---

### 🟡 SHORT-TERM (Fixes 20%)

#### 1. Semantic Keyword Scoring
```python
score = 1.0
if 'total' nearby: score *= 5
if 'final' nearby: score *= 4
if 'answer is' nearby: score *= 3
return max_score_candidate
```
**Impact:** All EXTRACTED_LAST cases

#### 2. Positional Weighting
```python
# Score by distance from end
recency_score = 1.0 + (total_len - pos) / total_len * 2.0
final_score = base_score * recency_score
```
**Impact:** Most WRONG_INDEX cases

#### 3. Reduce Last-Number Fallback
```python
# Before using last-number heuristic:
if 'total' found or 'final' found:
    use_keyword_candidate
elif has_marker:
    use_marker_candidate
else:
    use_last_number  # Only as true fallback
```
**Impact:** All EXTRACTED_LAST cases

---

### 🟢 MEDIUM-TERM (Fixes 10%)

#### 1. Multi-Pass Extraction
```python
# Pass 1: Look for explicit markers (####, \boxed)
result = extract_with_markers(text)
if result: return result, confidence=0.95

# Pass 2: Look for keywords (total, final, answer)
result = extract_with_keywords(text)
if result: return result, confidence=0.85

# Pass 3: Use positional heuristics
result = extract_with_position(text)
return result, confidence=0.60
```

#### 2. Word-to-Digit Conversion
```python
# Handle written numbers
word_to_num = {
    'zero': 0, 'one': 1, ..., 'twenty': 20, ...,
    'hundred': 100, 'thousand': 1000, ...
}
```

---

## Testing Strategy

### Phase 1: Regression Testing
```
Input: 20 test cases
Expected: Fix all 14 EXPECTED_NOT_IN_CANDIDATES cases
Measure: Success rate before/after for these cases
```

### Phase 2: Broader Testing
```
Input: Full GSM8K dataset
Expected: Improvement across all persona types
Measure: Overall extraction accuracy
```

### Phase 3: Edge Case Validation
```
Input: Manually curated edge cases
Expected: No regression in currently-passing cases
Measure: False positive detection
```

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Total cases analyzed | 20 |
| Cases with missing expected number | 14 (70%) |
| Cases with wrong index | 2 (10%) |
| Cases with aggressive fallback | 2 (10%) |
| Cases with first-number error | 1 (5%) |
| Cases with no extraction | 1 (5%) |
| Implicit persona failures | 19 (95%) |
| Explicit persona failures | 1 (5%) |

---

## Implementation Checklist

### Phase 1 (Urgent - Week 1)
- [ ] Add regex patterns for compound numbers
- [ ] Implement answer-sentence parsing
- [ ] Add arithmetic expression evaluation
- [ ] Test on 20 error cases

### Phase 2 (Important - Week 2)
- [ ] Add semantic keyword scoring
- [ ] Implement positional weighting
- [ ] Reduce last-number heuristic weight
- [ ] Test on full GSM8K dataset

### Phase 3 (Nice-to-have - Week 3)
- [ ] Multi-pass extraction framework
- [ ] Confidence scoring system
- [ ] Word-to-digit fallback
- [ ] Final validation and documentation

---

## Expected Outcomes

### After Phase 1
- Expected improvement: 70% → 85% success rate on these 20 cases
- Focus: Fix the dominant pattern

### After Phase 2
- Expected improvement: 85% → 95% success rate on these 20 cases
- Focus: Fix positional/semantic issues

### After Phase 3
- Expected improvement: 95% → 98%+ success rate on these 20 cases
- Focus: Edge cases and edge case handling

---

## References

- Main analysis: `GSM8K_EXTRACTION_ERROR_ANALYSIS.md`
- Detailed data: `gsm8k_extraction_analysis_detailed.json`
- Summary: `EXTRACTION_ERROR_ANALYSIS_SUMMARY.md`
- Ground truth format: `data/gsm8k_test.json` (uses `#### <number>` markers)
- Current extraction code: `src/rebuild_results_csv_ministrail copy.py`
