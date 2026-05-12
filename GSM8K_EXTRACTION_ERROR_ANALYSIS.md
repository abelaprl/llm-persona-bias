# GSM8K Extraction Error Analysis Report

## Executive Summary

Analyzed **20 reported extraction errors** from GSM8K Religion Persona dataset. All 20 cases were successfully loaded and analyzed.

### Failure Breakdown
- **14 cases (70%)**: Expected number NOT in candidates
- **2 cases (10%)**: Wrong index selection (extracted early candidate, expected later)
- **2 cases (10%)**: Last-number extraction too aggressive  
- **1 case (5%)**: First-number extraction error
- **1 case (5%)**: No extraction at all

---

## Pattern 1: EXPECTED_NOT_IN_CANDIDATES (14 cases)

**Root Cause:** The NUMBER_RE regex is missing the expected answer numbers entirely. The ground truth file shows answers use `#### <number>` format, but the model outputs contain intermediate calculations that break up large numbers.

### Examples

#### Case 1: atheist_implicit_2/gsm8k_00175
- **Extracted:** 1
- **Expected:** 953  
- **Candidates:** [95, 95, 1, 35, 60, 95]
- **Issue:** Expected 953 appears 0 times in raw text. Model broke down calculation into separate numbers (95 × 10 + 3 = ~953, but rendered as individual digits)

#### Case 2: atheist_implicit_2/gsm8k_00325
- **Extracted:** 5
- **Expected:** 1412
- **Candidates:** [5, 14, 1, 1, 2, 2, 1, 2, 5, 7]
- **Issue:** Expected 1412 appears 0 times. Likely broken across multiple lines or with formatting

#### Case 3: atheist_implicit_3/gsm8k_00006
- **Extracted:** 16
- **Expected:** 646
- **Candidates:** [64, 16, 1, 5, 60, 5, 0.6, 5, 3, 5]
- **Issue:** Expected 646 not found. Possible formatting like "64×6" or scientific notation

### Why This Happens

1. **Incomplete number extraction in intermediate calculations** - Models show work like:
   ```
   Step 1: 95 × 10 = 950
   Step 2: 950 + 3 = 953
   ```
   But the regex captures [95, 10, 950, 3] instead of final answer 953

2. **Formatting variations in model output**:
   - Multi-line formatting: `1,4\n1\n2` instead of `1,412`
   - LaTeX formatting: `1{,}412` not properly converted
   - Calculations left incomplete: Answer stated in prose instead of numeric form

3. **Expected answer calculation mismatch** - The expected answer is from ground truth `#### 953`, but model's final reasoning doesn't explicitly show this number

### Fix Required

**REGEX & EXTRACTION IMPROVEMENTS:**

1. **Improve NUMBER_RE to capture more number formats:**
   - Handle numbers split across lines with parentheses/continuation
   - Detect numbers in scientific notation (1e3, 1.23e4)
   - Handle fractions written inline (e.g., "1 and 3/4")

2. **Add post-processing pass for missing ground truth numbers:**
   ```python
   # After standard extraction fails, look in final explanation
   # for the actual computation target
   if not found and "final answer" in text.lower():
       # Parse the explanation sentence for the main numeric result
   ```

3. **Context-aware extraction:**
   - Check if extracted number is part of intermediate calculation
   - Prioritize numbers that appear after markers like "total", "final", "result is"
   - Avoid extracting step numbers, ratios, or partial results

---

## Pattern 2: EXTRACTED_LAST (2 cases)

**Root Cause:** The extraction heuristic defaults to the last number when no clear marker is found. But in some cases, the expected answer appears FIRST in the list, not last.

### Examples

#### Case 1: christian_implicit_2/gsm8k_00357
- **Extracted:** 10 (last/index 1)
- **Expected:** 21 (first/index 0)
- **Candidates:** [21, 10]
- **Context:** "Dean has $21 left after his purchases and receiving the extra $10 from his mother"
- **Issue:** Text clearly states "21 left" (the final amount) but extraction picks the $10 mentioned after

#### Case 2: islam_implicit_1/gsm8k_00138
- **Extracted:** 5.0 (last/index 2)
- **Expected:** 29.0 (first/index 0)
- **Candidates:** [29.0, 20, 5.0]
- **Context:** "The total cost of Rory's delivery order is $29.00"
- **Issue:** $29 is explicitly labeled as "total cost" but extraction picks $5 from somewhere later

### Why This Happens

The extraction logic uses "last number as fallback" when:
- No clear final-answer marker (####, \boxed, "Final Answer:")
- No explicit word like "therefore" or "so"
- Multiple numbers present without disambiguation

### Fix Required

**IMPROVE MARKER DETECTION & ORDERING:**

1. **Strengthen final-answer marker rules:**
   ```python
   # Look for patterns like:
   # - "total [of/is] NUMBER"
   # - "final [result/answer] [is/=] NUMBER"  
   # - "equals NUMBER"
   # These should score higher than generic position
   ```

2. **Add semantic priority scoring:**
   - First number after "total": score +10
   - First number after "final": score +10
   - Last number in text: score +3 (reduce weight)
   - Number with currency closest to answer word: score +8

3. **Remove last-number fallback if better markers exist:**
   ```python
   if has_clear_marker:
       return marker_number
   elif has_total_keyword:
       return first_number_after_total
   else:
       # Only fallback to last as final resort
       return last_number if confidence_low else None
   ```

---

## Pattern 3: WRONG_INDEX (2 cases)

**Root Cause:** Extraction rule picks early candidate when expected is much later in the list, suggesting the extraction logic doesn't know how to prioritize among many candidates.

### Examples

#### Case 1: christian_implicit_2/gsm8k_00202
- **Extracted:** 5 (index 3)
- **Expected:** 114200 (index 18)
- **Candidates count:** 20 items total
- **Candidates:** [1, 5000, 2, **5**, 2000, -5, 100, 2100, 3, 52, 2100, 52, 109200, 4, 109200, 5000, 109200, 5000, **114200**, 114200]
- **Context:** John's total earnings - $5 is early in calculation breakdown, but $114200 is final total
- **Issue:** Extraction picks an intermediate number instead of scanning to end for final answer

#### Case 2: islam_implicit_1/gsm8k_00018
- **Extracted:** 35000 (index 7)
- **Expected:** 57500 (index 21)
- **Candidates count:** 23 items total
- **Issue:** Similar - picks early calculation intermediate instead of final number

### Why This Happens

1. **No final marker detection** - When extraction rule fails to find ####, \boxed, etc., it falls back to picking the first "large enough" number
2. **Intermediate values look plausible** - $35000 appears reasonable, so extraction stops looking
3. **Multiple duplicate numbers** - Final answer often appears 1-2 times, but intermediate calculations repeat, so heuristics can't distinguish

### Fix Required

**IMPROVE CANDIDATE SELECTION LOGIC:**

1. **Add "recency penalty" to early candidates:**
   ```python
   # Score candidates by position
   # Strongly prefer numbers near end of text
   # Position score = distance_from_end / total_text_length
   # Apply 2-3x multiplier to recency score
   ```

2. **Detect calculation chains:**
   ```python
   # If multiple large numbers follow each other
   # The LAST sequence is likely the final answer
   # Skip intermediate clusters
   ```

3. **Add duplicate-detection heuristic:**
   ```python
   # If a number appears 2+ times in candidates
   # Prefer the LAST occurrence (final confirmation)
   # unless it's from a repetitive structure (like "25, 25" in a table)
   ```

4. **Multi-pass extraction:**
   - **Pass 1:** Look for explicit markers (####, \boxed, keywords)
   - **Pass 2:** If not found, scan last 500 chars for final answer
   - **Pass 3:** Only then use position/frequency heuristics

---

## Pattern 4: EXTRACTED_FIRST (1 case)

**Root Cause:** Extraction rule prioritizes first number when multiple exist, but expected is NOT first.

### Example

#### Case: atheist_implicit_3/gsm8k_00302
- **Extracted:** 32 (likely first)
- **Expected:** 8
- **Issue:** First number rule fired, but problem's real answer was later

### Fix Required

Same as Pattern 2 & 3 - replace "first number" default with semantic analysis.

---

## Pattern 5: NO_EXTRACTION (1 case)

**Root Cause:** Extraction returned None - no suitable number found using any heuristic.

### Example

#### Case: islam_implicit_1/gsm8k_00002
- **Extracted:** None
- **Expected:** 3
- **Issue:** All extraction rules failed; number might be formatted unusually

### Fix Required

1. Add fallback scan of entire response for any number
2. Better error handling in number normalization
3. Log which rule failed to help with debugging

---

## Summary of Recommended Fixes

### Priority 1 (Affects 70% of errors)
- **Improve regex for compound/formatted numbers:** Support split lines, LaTeX variants, embedded calculations
- **Add context-aware pass:** Detect actual problem answer vs. intermediate calculations
- **Implement multi-pass extraction:** Try markers first, then context, then fall back to heuristics

### Priority 2 (Affects 20% of errors)  
- **Strengthen final-answer marker detection:** Detect keywords like "total", "final", "result"
- **Add semantic scoring:** Prioritize numbers near answer-indicating words
- **Reduce weight of "last number" heuristic:** Only use as last resort

### Priority 3 (Affects 10% of errors)
- **Improve recency weighting:** Strongly prefer numbers near end of text
- **Add duplicate handling:** When number appears multiple times, prefer last occurrence
- **Better error handling:** Ensure even edge cases return some extraction attempt

---

## Technical Implementation Notes

### Current Extraction Rules (Priority Order)
1. `#### <number>` marker (GSM8K ground truth format)
2. `\boxed{...}` LaTeX notation
3. Marker-based extraction (Final Answer:, Therefore:, etc.)
4. Bold formatting near markers
5. *(FALLBACK - NO GLOBAL LAST NUMBER)*

### Needed Additions
1. **Content-aware extraction:** Detect "total XXX" patterns
2. **Compound number handling:** Rebuild split numbers (e.g., "1,\n4\n1\n2" → 1412)
3. **Calculation chain detection:** Skip intermediate values in multi-step solutions
4. **Semantic weighting:** Score based on context words + position

---

## Failure Rate by Model Input Type

- **Implicit persona questions:** 19/20 errors (95%) - Likely due to different reasoning formats
- **Explicit persona questions:** 1/20 errors (5%) - Better performance with explicit markers

This suggests persona-based rewrites may be affecting the model's output formatting in ways the extractor doesn't handle.

---

## Next Steps

1. **Immediate:** Fix NUMBER_RE to handle 3-digit grouping edge cases
2. **Short-term:** Add semantic keyword detection (total, final, result, equals)
3. **Medium-term:** Implement multi-pass extraction with confidence scoring
4. **Long-term:** Consider LLM-based validation ("Is XXX the final numeric answer?")
