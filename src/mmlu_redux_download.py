import json
from pathlib import Path
from datasets import load_dataset
from typing import List
import numpy as np

def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def get_mmlu_subjects() -> List[str]:
    """
    Get list of all available subjects in MMLU Redux
    """
    return [
        'anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_medicine',
        'college_physics', 'econometrics', 'electrical_engineering', 'formal_logic',
        'global_facts', 'high_school_chemistry', 'high_school_mathematics',
        'high_school_physics', 'high_school_statistics', 'human_aging',
        'logical_fallacies', 'machine_learning', 'miscellaneous', 'philosophy',
        'professional_accounting', 'public_relations', 'virology', 'conceptual_physics',
        'high_school_us_history', 'astronomy', 'high_school_geography',
        'high_school_macroeconomics', 'professional_law'
    ]

def download_mmlu_redux():
    """
    Download all MMLU Redux test sets using Hugging Face datasets library
    """
    print("Starting MMLU Redux download using Hugging Face datasets...")
    print("=" * 60)
    
    subjects = get_mmlu_subjects()
    all_data = []
    successful_subjects = []
    failed_subjects = []
    
    for i, subject in enumerate(subjects, 1):
        try:
            print(f"[{i}/{len(subjects)}] Loading {subject}...")
            dataset = load_dataset("edinburgh-dawg/mmlu-redux", subject, split="test")
            
            # Convert to list of dicts
            subject_data = dataset.to_pandas().to_dict(orient='records')
            
            # Add subject field to each record
            for record in subject_data:
                record['subject'] = subject
            
            all_data.extend(subject_data)
            successful_subjects.append(subject)
            print(f"  ✓ {subject}: {len(subject_data)} questions")
            
        except Exception as e:
            print(f"  ✗ Error loading {subject}: {e}")
            failed_subjects.append(subject)
    
    if not all_data:
        print("\n✗ No data downloaded. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("Saving combined data...")
    
    # Convert numpy types to native Python types
    all_data = convert_numpy_types(all_data)
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    output_path = data_dir / "mmlu_redux.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully saved to {output_path}")
    print(f"Total records: {len(all_data)}")
    print(f"Successful subjects: {len(successful_subjects)}")
    
    if failed_subjects:
        print(f"\n⚠ Failed subjects ({len(failed_subjects)}):")
        for subject in failed_subjects:
            print(f"  - {subject}")
    
    # Show column names and sample
    if all_data:
        print("\nColumn names:", list(all_data[0].keys()))
        print("\nSample record:")
        print(json.dumps(all_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    download_mmlu_redux()
