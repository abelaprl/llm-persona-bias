"""
Fix OLMo 3 32B Think warmup files - extract content from reasoning field
"""
import json
import os
from pathlib import Path

def fix_olmo_warmup_file(file_path):
    """Fix single warmup file - extract content from reasoning field"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if content is empty
    if data['response']['content'] == "":
        # Extract from raw_api_response
        raw_response = data['response']['raw_api_response']
        message = raw_response['choices'][0]['message']
        
        # Get reasoning content
        if 'reasoning' in message and message['reasoning']:
            reasoning_text = message['reasoning']
            
            # Remove <think> tags if present
            if '<think>' in reasoning_text and '</think>' in reasoning_text:
                # Extract everything after </think>
                final_content = reasoning_text.split('</think>\n\n')[-1]
            else:
                # Use the whole reasoning as content
                final_content = reasoning_text
            
            # Update the content field
            data['response']['content'] = final_content.strip()
            
            # Save back
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"[FIXED] {file_path}")
            print(f"  Content length: {len(final_content)} chars")
            return True
        else:
            print(f"[SKIP] {file_path} - No reasoning field found")
            return False
    else:
        print(f"[SKIP] {file_path} - Content already exists")
        return False

def main():
    """Fix all OLMo warmup files"""
    result_dir = Path("result/allenai_olmo-3-32b-think")
    
    if not result_dir.exists():
        print(f"Directory not found: {result_dir}")
        return
    
    # Find all warmup files
    warmup_files = list(result_dir.glob("*/warmup/persona_warmup.json"))
    
    print(f"Found {len(warmup_files)} OLMo warmup files\n")
    
    fixed_count = 0
    for file_path in sorted(warmup_files):
        if fix_olmo_warmup_file(file_path):
            fixed_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Total files: {len(warmup_files)}")
    print(f"Fixed: {fixed_count}")
    print(f"Skipped: {len(warmup_files) - fixed_count}")

if __name__ == "__main__":
    main()
