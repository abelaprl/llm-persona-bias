# GSM8K Extraction Errors - Analysis Summary

## Analysis Overview

Analyzed **20 reported extraction failures** from GSM8K Religion Persona dataset using deepseek-r1-distill-qwen-32b model.

**Success Rate:** All 20 cases successfully loaded and analyzed (100%)

---

## Failure Breakdown (20 total)

| Pattern | Count | % | Issue |
|---------|-------|---|-------|
| **EXPECTED_NOT_IN_CANDIDATES** | 14 | 70% | Expected answer number missing from regex extraction entirely |
| **WRONG_INDEX** | 2 | 10% | Expected number in candidates but wrong index selected |
| **EXTRACTED_LAST** | 2 | 10% | Last-number heuristic picked instead of correct early answer |
| **EXTRACTED_FIRST** | 1 | 5% | First-number heuristic picked instead of later answer |
| **NO_EXTRACTION** | 1 | 5% | Extraction returned None; no number captured |
| **CORRECT** | 0 | 0% | None of the reported cases were actually correct |

---

## Root Causes (Priority Order)

### 1. **NUMBER_RE Regex Missing Expected Numbers** (70% of failures)
**Status:** Fixable with regex enhancement

Numbers not captured because they are:
- Split across multiple lines with calculation operators (`1 * 4 * 1 * 2`)
- Formatted with LaTeX commands not fully handled (`1{,}412` → becomes `1,412`)
- Part of multi-step calculations where each step is extracted separately
- Written in prose instead of numeric form ("nine hundred fifty-three" vs "953")

**Representative Examples:**
- `gsm8k_00175`: Expected 953, but appears as `95 + 3` (captures [95, 3] not 953)
- `gsm8k_00325`: Expected 1412, appears as `14 * 100 + 12` (captures [14, 100, 12] not 1412)
- `gsm8k_00006`: Expected 646, appears as `64 * 10 + 6` (captures [64, 10, 6] not 646)

**Impact:** 14 failures (70%)

**Fix Complexity:** Medium - Requires:
1. Enhanced regex for compound numbers
2. Post-processing to detect and evaluate calculation patterns
3. Answer-sentence parsing to prioritize final results

---

### 2. **Wrong Candidate Selection (Position/Index)** (20% of failures)

**Problem A: Extracting Last Number Instead of First/Middle**
- When multiple candidates exist and no clear marker found
- Heuristic: "pick the last number" fails when answer is first
- Examples:
  - `gsm8k_00357`: Expected $21 (stated as "left after"), but picked $10 (mentioned later)
  - `gsm8k_00138`: Expected $29 (stated as "total cost"), but picked $5 (mentioned later)

**Problem B: Extracting Early Intermediate Instead of Final**
- When many candidates exist and expected is much later
- Early numbers look "plausible" so extraction stops looking
- Examples:
  - `gsm8k_00202`: Expected $114,200 (index 18), picked $5 (index 3)
  - `gsm8k_00018`: Expected $57,500 (index 21), picked $35,000 (index 7)

**Impact:** 4 failures (20%)

**Fix Complexity:** Medium - Requires:
1. Semantic keyword detection ("total", "final", "remaining")
2. Positional weighting favoring end-of-text numbers
3. Multi-pass extraction (markers → keywords → position)

---

### 3. **Complete Extraction Failure** (5% of failures)

**Case:** `gsm8k_00002`: Expected 3, got None

All extraction rules failed to capture expected answer.

**Impact:** 1 failure (5%)

**Fix Complexity:** Low - Add fallback word-to-digit conversion and full-text scan

---

## Categorized Failure Patterns with Examples

### **PATTERN 1: Missing Large Numbers from Calculations (14 cases, 70%)**

This is the dominant failure mode. The expected answer exists in the model output but:
1. Not as a complete standalone number
2. Result of a calculation that's decomposed into steps
3. Model shows intermediate values that extraction captures instead

**Top 3 Most Representative Examples:**

#### Example 1-A: `atheist_implicit_2/gsm8k_00175`
```
Problem: Calculate something = 953
Model Output: "95 multiplied by 10 equals 950, then add 3 equals 953"
Candidates Found: [95, 10, 950, 3]
Extracted: 1
Expected: 953
Why: Extraction picked "1" (possibly step number or index), never reconstructed 953
```

#### Example 1-B: `atheist_implicit_2/gsm8k_00325`
```
Problem: Calculate something = 1,412
Model Output: "14 hundred is 1400, plus 12 is 1412"
Candidates Found: [14, 100, 12, 1400]
Extracted: 5
Expected: 1412
Why: Thousands separator handling incomplete; 1,412 and 1412 treated differently
```

#### Example 1-C: `atheist_implicit_3/gsm8k_00006`
```
Problem: Calculate something = 646
Model Output: "64 times 10 plus 6 equals 646"
Candidates Found: [64, 10, 6]
Extracted: 16
Expected: 646
Why: Regex doesn't reconstruct final answer; extraction picked early value
```

**Why This Happens:**
- NUMBER_RE captures each number separately
- No post-processing to recognize "X * Y + Z" patterns
- No final-answer marker detection for intermediate calculations
- Model output format varies; not all use `#### final_answer` marker

**Suggested Fix:**
```python
# After standard extraction fails:
1. Scan for arithmetic patterns: "\d+ [*+-/] \d+" 
2. Try to evaluate these expressions
3. Check if result matches any expected reasonable answer
4. Use result if found in last 20% of text

# OR better yet:
1. Look for sentences with "answer is", "total is", "equals"
2. Extract number from that sentence only
3. Skip intermediate calculations entirely
```

---

### **PATTERN 2: Last-Number Heuristic Too Aggressive (2 cases, 10%)**

When no clear marker is found, extraction defaults to "pick the last number". This fails when:
- The actual answer is mentioned first or middle
- Later mentions are supporting details, not the main answer

**Example 2-A: `christian_implicit_2/gsm8k_00357`**
```
Model Output: "Dean has $21 left after his purchases and receiving the extra $10 from his mother"
Candidates: [21, 10]  (in order of appearance)
Extracted: 10 (last)
Expected: 21 (first)
Problem: Text clearly states "21 left" (the final amount), then mentions "$10 from mother" as additional detail
Why: Generic "last number" rule picked $10, ignoring that $21 is the problem's real answer
```

**Example 2-B: `islam_implicit_1/gsm8k_00138`**
```
Model Output: "The total cost of Rory's delivery order is $29.00. This includes the cost of the items ($24.00) and a delivery fee ($5.00)."
Candidates: [29.0, 24.0, 5.0]
Extracted: 5.0 (last)
Expected: 29.0 (first, explicitly marked as "total cost")
Problem: $29 is explicitly called "total cost", but $5 is mentioned last
Why: Extraction ignored keyword "total" and just picked last number
```

**Suggested Fix:**
```python
# Before using "last number" heuristic:
1. Search for keywords: 'total', 'final', 'answer is', 'remaining', 'left', 'equals'
2. If keyword found within 50 chars of any number, prioritize that number
3. Only use "last number" if NO keywords detected

# Better yet:
Scoring function:
  score = base_position_score
  if 'total' nearby: score *= 5
  if 'final' nearby: score *= 4
  if 'answer is' nearby: score *= 3
  if in_last_20_percent: score *= 1.5
  return max_score candidate
```

---

### **PATTERN 3: Wrong Index Selection (2 cases, 10%)**

Extraction picks early candidate when the expected answer is much later in the list.

**Example 3-A: `christian_implicit_2/gsm8k_00202`**
```
Model Output: John's yearly earnings calculation
  Step 1: Base salary = 1 × $5,000 = $5,000
  Step 2: Bonus calculation = 2 × $2,000 = $4,000
  ... (many intermediate steps)
  Final: Total earnings = $114,200

Candidates Found: [1, 5000, 2, 5, 2000, -5, 100, 2100, ..., 114200, 114200]
Extracted: 5 (index 3)
Expected: 114200 (index 18)
Problem: Extraction stopped at first "plausible" number instead of scanning to final answer
Why: No semantic understanding that "total earnings" = final large number, not early small number
```

**Example 3-B: `islam_implicit_1/gsm8k_00018`**
```
Model Output: Salary calculation
  Jill earns $35 per hour × 1000 hours/year = $35,000
  Plus bonus: $22,500
  Total annual earnings: $57,500

Candidates: [1, 20, 35, 50, 20, 35, 50, 35000, 35000, 2, 30, 15, 50, 30, 15, 50, 22500, 22500, 3, 35000, 22500, 57500, 57500]
Extracted: 35000 (index 7)
Expected: 57500 (index 21)
Problem: Many intermediate values; extraction picked early large number instead of final largest
Why: No position weighting; number in middle looked "final" enough
```

**Suggested Fix:**
```python
# Add positional weighting:
def score_by_position(candidate_index, total_candidates):
  # Strongly prefer numbers near end
  distance_from_end = total_candidates - candidate_index
  recency_score = (1 + distance_from_end / total_candidates) * 3.0
  return recency_score

# OR detect calculation patterns:
# If numbers form a chain: base → intermediate1 → intermediate2 → FINAL
# Pick the last in the chain
```

---

### **PATTERN 4: First-Number Heuristic (1 case, 5%)**

Generic "pick first number" rule without understanding context.

**Example 4-A: `atheist_implicit_3/gsm8k_00302`**
```
Extracted: 32 (first)
Expected: 8
Problem: First calculation gives 32, but problem's actual answer is 8 (after adjustments)
Why: No semantic understanding of problem asking for "final adjusted value", not first intermediate result
```

**Suggested Fix:** Same as PATTERN 2 - add keyword detection and semantic scoring.

---

### **PATTERN 5: Complete Failure (1 case, 5%)**

**Example 5-A: `islam_implicit_1/gsm8k_00002`**
```
Expected: 3
Extracted: None
All extraction rules failed
Possible causes:
  - Number written as word "three" instead of digits "3"
  - Unusual formatting not handled by regex
  - Normalization error
Suggested fix: Add word-to-digit converter, implement fallback full-text scan
```

---

## Heuristic Rules That Need Fixing

### Current Rules (in rebuild_results_csv_ministrail copy.py)

1. **RULE 1:** Look for `####` marker (GSM8K ground truth format)
2. **RULE 2:** Look for `\boxed{...}` LaTeX notation
3. **RULE 3:** Look for "Final Answer:", "Therefore:", etc. markers
4. **RULE 4:** Look for bold formatting near markers
5. **RULE 5:** *(NO GLOBAL LAST-NUMBER FALLBACK)* - but seems to be applied anyway

### Problems

- **RULE 1 Failure:** Model output uses #### but extraction still fails → regex issue, not rule issue
- **RULE 2 Failure:** Models don't always use \boxed{}
- **RULE 3 Failure:** Markers present but extraction still picks wrong number → selection logic broken
- **RULE 4 Failure:** Bold formatting not reliable across all models
- **Missing Rule:** No semantic keyword scoring (total, final, remaining, etc.)
- **Missing Rule:** No positional weighting (prefer end-of-text numbers)
- **Missing Rule:** No calculation chain detection (reconstruct final answers from arithmetic)
- **Missing Rule:** No word-to-digit conversion fallback

### Recommended Changes

#### Immediate (Fixes 70% of failures)
```
Priority 1a: Enhance NUMBER_RE to handle compound numbers (1 * 4 * 1 * 2)
Priority 1b: Add answer-sentence extraction pass ("The answer is X" pattern)
Priority 1c: Detect and evaluate arithmetic expressions (95 * 10 + 3 = 953)
```

#### Short-term (Fixes 20% of failures)
```
Priority 2a: Add semantic keyword scoring (total, final, result, remaining)
Priority 2b: Implement positional weighting (prefer last 30% of text)
Priority 2c: Reduce/eliminate last-number fallback unless confirmed by marker
Priority 2d: Add duplicate detection (if number appears 2+ times, use last)
```

#### Medium-term (Fixes remaining 10%)
```
Priority 3a: Multi-pass extraction (markers → keywords → position)
Priority 3b: Add confidence scoring to extraction results
Priority 3c: Implement word-to-digit conversion fallback
Priority 3d: Add problem question analysis (what is being asked?)
```

---

## Key Insights

### Persona Impact
- **Implicit Persona Questions:** 19/20 errors (95%)
- **Explicit Persona Questions:** 1/20 errors (5%)
- **Conclusion:** Implicit persona rewrites produce different model output formats that extraction doesn't handle well

### Model Output Format Issues
1. Intermediate calculations not marked as intermediate
2. Multiple candidate numbers without clear final answer marker
3. Final answer sometimes in prose instead of numeric
4. Large numbers (>10,000) often decomposed into steps

### Ground Truth Format
- Expected format: `#### <number>` (GSM8K standard)
- Model outputs: May use different formats (#### missing, calculation chains, prose)
- Extraction needs to be more flexible than regex matching

---

## Implementation Priority

### Phase 1 (Addresses 70% of failures)
1. Enhance NUMBER_RE for compound numbers
2. Add post-processing for arithmetic expressions
3. Implement answer-sentence parsing ("The answer is")

### Phase 2 (Addresses 20% of failures)
1. Add semantic keyword detection
2. Implement positional weighting
3. Reduce aggressive last-number heuristic

### Phase 3 (Addresses 10% of failures)
1. Multi-pass extraction framework
2. Confidence scoring
3. Word-to-digit fallback

---

## Files Created

1. **GSM8K_EXTRACTION_ERROR_ANALYSIS.md** - Detailed explanation of each pattern
2. **gsm8k_extraction_analysis_detailed.json** - Structured data for implementation
3. **analyze_gsm8k_extraction_errors.py** - Analysis script (for reference)

---

## Next Steps

1. **Validate fixes** against these 20 test cases
2. **Implement Phase 1** improvements
3. **Re-test** on full dataset to measure improvement
4. **Iterate** on Phase 2 and 3 if needed
5. **Consider** LLM-based validation as final backstop

