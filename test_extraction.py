import re

# Test different extraction methods
test_outputs = [
    "Sunday + Monday combined = 23 minutes + 16 minutes = 39 minutes.  \nTwice the combined time = 2 × 39 minutes = 78 minutes.  \nPeter must exercise for 78 minutes on Tuesday.\n\n**Final answer: 78**",
    "John arm wrestles 20 people and beats 80% of them.\n80% = 0.8\n0.8 × 20 = 16\nSo, John beats 16 people.\n\n**Final answer: 16**",
]

def extract_final_answer_v2(text: str) -> str:
    """Extract answer after 'Final answer:' more carefully."""
    if not text:
        return ""
    
    # Look for Final answer pattern
    match = re.search(r'[Ff]inal\s+answer\s*:\s*\*?\*?(\d+(?:\.\d+)?)\*?\*?', text)
    if match:
        return match.group(1)
    
    # If no match, return nothing
    return ""

for output in test_outputs:
    answer = extract_final_answer_v2(output)
    print(f"Output: {output[:80]}...")
    print(f"Extracted: {answer}\n")
