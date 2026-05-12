"""
GSM8K Extraction Error Fixes - Code Examples
Suggested implementations for each failure pattern
"""

import re
from typing import Tuple, List, Optional

# Current NUMBER_RE (from rebuild_results_csv_ministrail copy.py)
NUMBER_RE_CURRENT = r'(?:[-\u2212\u2013]\s*)?(?:\$)?(?:\d{1,3}(?:(?:,\\!|,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?'

# ============================================================================
# FIX 1: ENHANCE NUMBER_RE FOR COMPOUND NUMBERS
# ============================================================================
# Problem: Misses large numbers split by calculation operators
# Current: Captures [95, 10, 3] but not 953
# Fix: Add patterns to capture more number formats

def create_enhanced_number_re():
    """Create enhanced regex that captures more number formats."""
    return (
        r'(?:'
        # Original pattern (thousands with separators)
        r'(?:[-\u2212\u2013]\s*)?(?:\$)?(?:\d{1,3}(?:(?:,\\!|,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?'
        r'|'
        # Scientific notation: 1e3, 1.2E4, 1.5e-2
        r'(?:[-\u2212\u2013]\s*)?(?:\$)?\d+(?:\.\d+)?[eE][-+]?\d+'
        r'|'
        # Fractions written numerically: 1/2, 3/4
        r'(?:[-\u2212\u2013]\s*)?(?:\$)?\d+(?:\.\d+)?/\d+(?:\.\d+)?'
        r'|'
        # Currency with no digits before: $10, €50
        r'(?:\$|€|£)\d+(?:\.\d+)?'
        r')'
    )

NUMBER_RE_ENHANCED = create_enhanced_number_re()


# ============================================================================
# FIX 2: DETECT AND EVALUATE ARITHMETIC EXPRESSIONS
# ============================================================================
# Problem: Model shows "95 * 10 + 3" but extraction only gets [95, 10, 3]
# Fix: Detect arithmetic patterns and evaluate them

def find_arithmetic_expressions(text: str) -> List[Tuple[float, str]]:
    """
    Find arithmetic expressions like "95 * 10 + 3" and evaluate them.
    Returns: [(result, expression_string), ...]
    """
    results = []
    
    # Pattern: NUMBER OP NUMBER [OP NUMBER]...
    # Where OP is +, -, *, /, mod
    pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\d+(?:\.\d+)?)*)'
    
    for match in re.finditer(pattern, text):
        expr = match.group(0)
        try:
            # Safely evaluate (would be better with ast.literal_eval for security)
            result = eval(expr)
            results.append((result, expr))
        except Exception:
            pass
    
    return results


# ============================================================================
# FIX 3: ANSWER-SENTENCE EXTRACTION
# ============================================================================
# Problem: Model ends with "The answer is 953" but extraction doesn't find it
# Fix: Specifically look for answer-indicator sentences

def extract_from_answer_sentence(text: str) -> Optional[float]:
    """
    Extract number from explicit answer sentences.
    Patterns: "The answer is 953", "Total is 100", "Result equals 42"
    """
    # Patterns with capture group for number
    patterns = [
        r'(?:the\s+)?answer\s+(?:is|=|:)\s*([^\s,\.]+)',  # "answer is 953"
        r'(?:the\s+)?(?:final\s+)?(?:total|result|sum)\s+(?:is|=|:)\s*([^\s,\.]+)',  # "total is 100"
        r'(?:the\s+)?(?:final\s+)?(?:amount|value|number)\s+(?:is|=|:)\s*([^\s,\.]+)',  # "amount is 50"
        r'(?:he|she|they|it)\s+(?:has?|have|takes?|spends?|earns?|makes?)\s+(?:a\s+)?(?:total\s+)?(?:of\s+)?([^\s,\.]+)',
        r'(?:equals?|=)\s*([^\s,\.]+)',  # "= 100" (risky, might match "= N" in formulas)
    ]
    
    # Search in last 1000 chars (where answers usually are)
    search_text = text[-1000:] if len(text) > 1000 else text
    
    for pattern in patterns:
        for match in re.finditer(pattern, search_text, re.IGNORECASE):
            raw_num = match.group(1).strip()
            # Clean up: remove trailing punctuation, brackets
            raw_num = re.sub(r'[\.)\]],\s*$', '', raw_num)
            try:
                num = float(raw_num) if '.' in raw_num else int(raw_num)
                return num
            except ValueError:
                continue
    
    return None


# ============================================================================
# FIX 4: SEMANTIC KEYWORD SCORING
# ============================================================================
# Problem: "total is 21, received bonus 10" → picks 10 instead of 21
# Fix: Score candidates based on nearby keywords

def score_candidates_by_keywords(
    candidates: List[Tuple[float, int, str]],  # (number, position, line_context)
    text: str
) -> List[Tuple[float, float]]:  # (number, score)
    """
    Score candidates based on semantic keywords.
    Returns: [(number, confidence_score), ...]
    """
    
    # Keywords that indicate final answer (with weights)
    answer_keywords = {
        'answer': 3.0,
        'total': 3.0,
        'final': 3.0,
        'result': 2.5,
        'sum': 2.5,
        'final answer': 4.0,
        'total amount': 3.5,
        'total cost': 3.5,
        'total earnings': 3.5,
        'remaining': 2.0,
        'left': 2.0,
        'spent': 2.0,
        'earned': 2.0,
    }
    
    scored = []
    
    for number, pos, line in candidates:
        score = 1.0
        
        # Check for keywords near this candidate
        context_start = max(0, pos - 100)
        context_end = min(len(text), pos + 100)
        context = text[context_start:context_end].lower()
        
        for keyword, weight in answer_keywords.items():
            if keyword in context:
                score *= weight
        
        # Position bonus: prefer end of text
        position_ratio = (len(text) - pos) / len(text)
        position_bonus = 1.0 + position_ratio * 2.0  # 3x at end, 1x at start
        score *= position_bonus
        
        scored.append((number, score))
    
    return scored


# ============================================================================
# FIX 5: POSITIONAL WEIGHTING
# ============================================================================
# Problem: Early candidate picked instead of final answer
# Fix: Strongly prefer numbers near end of text

def add_position_weights(
    candidates: List[Tuple[float, int]],  # (number, position_in_text)
    text: str
) -> List[Tuple[float, float]]:  # (number, weighted_score)
    """
    Add position-based weighting.
    Numbers near end of text get higher score.
    """
    
    text_len = len(text)
    weighted = []
    
    for number, pos in candidates:
        # Distance from end (0 at start, 1 at end)
        distance_from_end = (text_len - pos) / text_len
        
        # Create sigmoid-like weighting that heavily prefers end
        # Formula: 1 + 3 * (distance_from_end)
        # At start (pos=0): 1 + 3 * 1.0 = 4.0
        # At end (pos=text_len): 1 + 3 * 0.0 = 1.0
        # This actually reverses - we want: high at END, low at START
        
        # Corrected formula:
        position_weight = 1.0 + (distance_from_end ** 2) * 3.0
        
        weighted.append((number, position_weight))
    
    return weighted


# ============================================================================
# FIX 6: MULTI-PASS EXTRACTION WITH CONFIDENCE
# ============================================================================
# Problem: No way to know if extraction is confident or guessing
# Fix: Try multiple strategies and return confidence score

def extract_with_confidence(text: str) -> Tuple[Optional[float], str, float]:
    """
    Multi-pass extraction with confidence scoring.
    
    Returns:
        (extracted_number, extraction_method, confidence_score)
        
    confidence_score:
        0.95+: Very confident (explicit marker found)
        0.80-0.94: Confident (keyword + position match)
        0.60-0.79: Uncertain (position heuristic only)
        <0.60: Fallback/guess
    """
    
    # PASS 1: Explicit markers (####, \boxed)
    # (Use existing extract_hashes(), extract_boxed() from original code)
    markers = []  # Would extract using existing code
    if markers:
        return markers[0], 'hashes_or_boxed', 0.95
    
    # PASS 2: Answer sentences ("The answer is", "Total is", etc.)
    answer_num = extract_from_answer_sentence(text)
    if answer_num is not None:
        return answer_num, 'answer_sentence', 0.90
    
    # PASS 3: Arithmetic expression evaluation
    expressions = find_arithmetic_expressions(text)
    if expressions:
        # Take the last (most likely final) expression
        result, expr = expressions[-1]
        return result, 'arithmetic_expr', 0.85
    
    # PASS 4: Keyword-based selection
    # (Would need to adapt extract_candidates from original code)
    candidates = []  # extract_candidates(text)
    if candidates:
        scored = score_candidates_by_keywords(candidates, text)
        if scored:
            best_num, score = max(scored, key=lambda x: x[1])
            confidence = 0.70 + (score / 10.0) * 0.20  # 0.70 to 0.90 range
            return best_num, 'keyword_scoring', min(confidence, 0.90)
    
    # PASS 5: Position-based fallback
    if candidates:
        weighted = add_position_weights(candidates, text)
        if weighted:
            best_num, weight = max(weighted, key=lambda x: x[1])
            confidence = 0.50 + (weight / 5.0) * 0.15
            return best_num, 'position_heuristic', min(confidence, 0.70)
    
    # PASS 6: Word-to-digit conversion
    # (Not implemented here, but would try "three" → 3, etc.)
    
    # No extraction possible
    return None, 'no_extraction', 0.0


# ============================================================================
# FIX 7: WORD-TO-DIGIT CONVERSION (Fallback)
# ============================================================================
# Problem: "three" instead of "3" → extraction returns None
# Fix: Convert written numbers to digits

WORD_TO_NUM = {
    # Singles
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    # Teens
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19,
    # Tens
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    # Scales
    'hundred': 100, 'thousand': 1000, 'million': 1000000,
}

def text_to_number(text: str) -> Optional[float]:
    """
    Convert written number text to numeric value.
    Examples: "twenty-three" → 23, "one thousand five hundred" → 1500
    """
    text = text.lower().strip()
    
    # Handle simple cases first
    if text in WORD_TO_NUM:
        return float(WORD_TO_NUM[text])
    
    # Handle hyphenated numbers: "twenty-three" → 23
    parts = text.split('-')
    if len(parts) == 2 and all(p in WORD_TO_NUM for p in parts):
        return float(sum(WORD_TO_NUM[p] for p in parts))
    
    # Handle compound like "one thousand five hundred"
    words = text.split()
    total = 0
    current = 0
    
    for word in words:
        if word in WORD_TO_NUM:
            val = WORD_TO_NUM[word]
            if val >= 1000:
                current *= val
                total += current
                current = 0
            else:
                current += val
        else:
            # Unknown word
            return None
    
    total += current
    return float(total) if total > 0 else None


# ============================================================================
# INTEGRATION EXAMPLE: Improved Extract Final Answer
# ============================================================================

def extract_final_answer_improved(text: str) -> Tuple[Optional[float], str, float]:
    """
    Improved extraction using all fixes above.
    
    Usage:
        number, method, confidence = extract_final_answer_improved(model_output)
        if confidence > 0.75:
            use_extraction(number)
        elif confidence > 0.60:
            flag_for_manual_review(number)
        else:
            mark_as_failed()
    """
    
    return extract_with_confidence(text)


# ============================================================================
# TESTING / VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Test cases from analysis
    test_cases = [
        # (model_output, expected_answer, test_name)
        (
            "95 multiplied by 10 equals 950, then add 3 equals 953",
            953,
            "gsm8k_00175: Arithmetic expression"
        ),
        (
            "The total cost of Rory's delivery order is $29.00. The delivery fee is $5.00.",
            29.0,
            "gsm8k_00138: Keyword 'total cost'"
        ),
        (
            "Dean has $21 left after his purchases and receiving the extra $10 from his mother",
            21,
            "gsm8k_00357: Keyword 'left' should beat 'last'"
        ),
    ]
    
    print("Testing improved extraction functions:\n")
    for text, expected, name in test_cases:
        extracted, method, confidence = extract_final_answer_improved(text)
        match = "✓" if extracted == expected else "✗"
        print(f"{match} {name}")
        print(f"  Expected: {expected}, Extracted: {extracted}")
        print(f"  Method: {method}, Confidence: {confidence:.2f}\n")


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================
"""
Integration points in rebuild_results_csv_ministrail copy.py:

1. Replace NUMBER_RE_CURRENT with NUMBER_RE_ENHANCED (line 217)
2. Add arithmetic expression detection before marker extraction (line 492)
3. Add answer sentence extraction as Pass 3 rule (after Rule 2: boxed)
4. Add semantic keyword scoring (modify extract_marker function)
5. Add positional weighting when multiple candidates exist
6. Implement multi-pass extraction framework with confidence scoring

Priority order:
  1. Fix RULE 1 (#### marker) with enhanced regex
  2. Add arithmetic expression evaluation (FIX 2)
  3. Add answer sentence extraction (FIX 3)
  4. Add semantic scoring (FIX 4)
  5. Add position weighting (FIX 5)
  6. Implement full multi-pass (FIX 6)
  7. Add word-to-digit fallback (FIX 7)

Testing strategy:
  - Regression test on 20 error cases
  - Measure improvement per fix
  - Validate no regression on passing cases
  - Broad test on full GSM8K dataset
"""
