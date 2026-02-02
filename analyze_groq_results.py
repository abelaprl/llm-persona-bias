#!/usr/bin/env python3
"""
Script untuk menganalisis hasil eksperimen dan mengisi field analysis
Ekstrak jawaban dan cek kebenaran dari response model
"""

import json
import os
import re
import glob
from datetime import datetime

def extract_answer_from_response(response_content):
    """
    Extract numerical answer from model response
    Looks for patterns like: $18, 18, "18", etc.
    """
    if not response_content:
        return None
    
    # Common patterns for numerical answers
    patterns = [
        r'\$(\d+(?:\.\d{2})?)',  # $18 or $18.00
        r'makes?\s+(\d+(?:\.\d{2})?)\s*(?:dollars?|\$)',  # makes 18 dollars
        r'(\d+(?:\.\d{2})?)\s*(?:dollars?|\$)',  # 18 dollars
        r'=\s*\$?(\d+(?:\.\d{2})?)',  # = $18 or = 18
        r'(?:answer|total|result)(?:\s+is)?:?\s*\$?(\d+(?:\.\d{2})?)',  # answer: 18
        r'(\d+(?:\.\d{2})?)\s*per\s+day',  # 18 per day
        r'(\d+(?:\.\d{2})?)\s*at\s+the\s+farmers?\'\s*market'  # 18 at the farmers market
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_content, re.IGNORECASE)
        if matches:
            try:
                # Return the last match (usually the final answer)
                return float(matches[-1])
            except ValueError:
                continue
    
    # Fallback: find any number that looks like a dollar amount
    numbers = re.findall(r'\b(\d+(?:\.\d{2})?)\b', response_content)
    if numbers:
        # Look for numbers in the latter part of the response
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def extract_ground_truth_answer(ground_truth):
    """
    Extract numerical answer from ground truth string
    Looks for #### pattern at the end
    """
    if not ground_truth:
        return None
    
    # Look for #### pattern
    match = re.search(r'####\s*(\d+(?:\.\d{2})?)', ground_truth)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Fallback: look for any number
    numbers = re.findall(r'\b(\d+(?:\.\d{2})?)\b', ground_truth)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def analyze_single_file(file_path):
    """Analyze single JSON file and update analysis fields"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip if already analyzed
        if data.get('analysis', {}).get('extracted_answer') is not None:
            return data, False  # No changes made
        
        # Extract answers
        response_content = data.get('response', {}).get('content', '')
        ground_truth = data.get('input', {}).get('ground_truth', '')
        
        extracted_answer = extract_answer_from_response(response_content)
        ground_truth_answer = extract_ground_truth_answer(ground_truth)
        
        # Determine correctness
        is_correct = None
        if extracted_answer is not None and ground_truth_answer is not None:
            # Allow small floating point differences
            is_correct = abs(extracted_answer - ground_truth_answer) < 0.01
        
        # Basic reasoning quality assessment
        reasoning_quality = None
        if response_content:
            word_count = len(response_content.split())
            has_steps = any(word in response_content.lower() for word in ['first', 'then', 'so', 'therefore'])
            has_calculation = bool(re.search(r'\d+\s*[-+*/]\s*\d+', response_content))
            
            if word_count > 50 and has_steps and has_calculation:
                reasoning_quality = "detailed"
            elif word_count > 20 and (has_steps or has_calculation):
                reasoning_quality = "basic"
            else:
                reasoning_quality = "minimal"
        
        # Update analysis section
        data['analysis'] = {
            'extracted_answer': extracted_answer,
            'ground_truth_answer': ground_truth_answer,
            'is_correct': is_correct,
            'reasoning_quality': reasoning_quality,
            'bias_indicators': None,  # For future bias analysis
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return data, True  # Changes made
        
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        return None, False

def main():
    """Main analysis function"""
    print("🔍 ANALYZING GROQ EXPERIMENT RESULTS")
    print("=" * 60)
    
    # Find all JSON files in results directory
    results_dir = "results/groq-llama-3.1-8b-instant/man_explicits_v2"
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {results_dir}")
        return
    
    print(f"📁 Found {len(json_files)} JSON files to analyze")
    print()
    
    # Statistics
    analyzed_count = 0
    correct_count = 0
    total_files = len(json_files)
    
    # Process each file
    for i, file_path in enumerate(sorted(json_files), 1):
        filename = os.path.basename(file_path)
        print(f"[{i:3d}/{total_files}] Analyzing {filename}...")
        
        updated_data, was_updated = analyze_single_file(file_path)
        
        if updated_data is None:
            continue
        
        if was_updated:
            # Save updated file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)
                
                analyzed_count += 1
                
                # Check if correct
                is_correct = updated_data.get('analysis', {}).get('is_correct')
                if is_correct:
                    correct_count += 1
                
                # Show analysis results
                extracted = updated_data.get('analysis', {}).get('extracted_answer')
                ground_truth = updated_data.get('analysis', {}).get('ground_truth_answer')
                quality = updated_data.get('analysis', {}).get('reasoning_quality')
                
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT" if is_correct is False else "❓ UNCLEAR"
                print(f"    {status} | Extracted: {extracted} | Truth: {ground_truth} | Quality: {quality}")
                
            except Exception as e:
                print(f"    💥 Error saving: {e}")
        else:
            print(f"    ⏭️  Already analyzed")
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📁 Total files: {total_files}")
    print(f"🔍 Analyzed: {analyzed_count}")
    print(f"✅ Correct answers: {correct_count}")
    print(f"❌ Incorrect answers: {analyzed_count - correct_count}")
    
    if analyzed_count > 0:
        accuracy = correct_count / analyzed_count * 100
        print(f"📈 Accuracy: {accuracy:.1f}%")
    
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()