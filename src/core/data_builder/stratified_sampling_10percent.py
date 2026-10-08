import json
import math
import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Dict

def detect_question_type(question: str) -> str:
    """
    Detect question type using rule-based prefix matching.
    
    Args:
        question: The question text
        
    Returns:
        str: One of {who, what, when, where, why, how, other}
    """
    if not isinstance(question, str):
        return 'other'
    
    q_lower = question.strip().lower()
    prefixes = ['who', 'what', 'when', 'where', 'why', 'how']
    
    for prefix in prefixes:
        if q_lower.startswith(prefix):
            return prefix
    
    return 'other'

def safely_parse_answers(answers) -> List[Dict]:
    """
    Safely parse answers field which may be stringified JSON or list.
    
    Args:
        answers: The answers field (can be string, list, or other)
        
    Returns:
        List[Dict]: Parsed answers or empty list
    """
    try:
        if isinstance(answers, str):
            return json.loads(answers)
        elif isinstance(answers, list):
            return answers
        else:
            return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []

def get_answer_length(answers, is_impossible: bool) -> int:
    """
    Extract answer length from answers field.
    
    Args:
        answers: The answers field
        is_impossible: Whether question is unanswerable
        
    Returns:
        int: Number of words in first answer, or 0 if unanswerable
    """
    if is_impossible:
        return 0
    
    parsed_answers = safely_parse_answers(answers)
    
    if not parsed_answers or len(parsed_answers) == 0:
        return 0
    
    try:
        first_answer = parsed_answers[0]
        if isinstance(first_answer, dict):
            answer_text = first_answer.get('text', '')
        else:
            answer_text = str(first_answer)
        
        if not answer_text:
            return 0
        
        return len(answer_text.split())
    except (IndexError, AttributeError, KeyError):
        return 0

def get_answer_length_category(length: int) -> str:
    """
    Categorize answer length.
    
    Args:
        length: Number of words in answer
        
    Returns:
        str: One of {none, short, medium, long}
    """
    if length == 0:
        return 'none'
    elif 1 <= length <= 3:
        return 'short'
    elif 4 <= length <= 10:
        return 'medium'
    else:
        return 'long'

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features for stratification.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new columns (type, answer_len, ans_len_cat)
    """
    df['type'] = df['question'].apply(detect_question_type)
    df['answer_len'] = df.apply(
        lambda row: get_answer_length(row['answers'], row['is_impossible']),
        axis=1
    )
    df['ans_len_cat'] = df['answer_len'].apply(get_answer_length_category)
    
    return df

def proportional_stratified_sample(
    df: pd.DataFrame,
    strata_cols: List[str],
    sample_fraction: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform proportional stratified sampling.
    
    Args:
        df: Input dataframe
        strata_cols: Columns to stratify by
        sample_fraction: Fraction to sample (0.1 = 10%)
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Sampled dataframe
    """
    sampled_dfs = []
    
    for stratum, group in df.groupby(strata_cols):
        n_total = len(group)
        # Use ceiling to ensure at least 1 sample per stratum
        n_sample = max(1, math.ceil(sample_fraction * n_total))
        
        # If requested sample exceeds available, take all
        n_sample = min(n_sample, n_total)
        
        sampled = group.sample(n=n_sample, random_state=random_state)
        sampled_dfs.append(sampled)
    
    return pd.concat(sampled_dfs, ignore_index=True)

def print_distribution_comparison(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    columns: List[str]
) -> Dict[str, float]:
    """
    Print and compute distribution comparison.
    
    Args:
        original_df: Original dataframe
        sampled_df: Sampled dataframe
        columns: Columns to compare
        
    Returns:
        Dict[str, float]: Mean absolute percentage difference
    """
    results = {}
    
    for col in columns:
        print(f'\n{col}:')
        print('-' * 60)
        
        orig_dist = (original_df[col].value_counts(normalize=True) * 100).round(2)
        samp_dist = (sampled_df[col].value_counts(normalize=True) * 100).round(2)
        
        print('Original:')
        print(orig_dist.to_string())
        print('\nSampled:')
        print(samp_dist.to_string())
        
        # Compute percentage difference
        all_categories = set(orig_dist.index).union(samp_dist.index)
        diffs = []
        
        for cat in all_categories:
            orig_pct = orig_dist.get(cat, 0)
            samp_pct = samp_dist.get(cat, 0)
            diff = abs(samp_pct - orig_pct)
            diffs.append(diff)
        
        mean_diff = np.mean(diffs) if diffs else 0
        results[col] = mean_diff
        print(f'\nMean absolute % difference: {mean_diff:.3f}%')
    
    return results

def main():
    """Main execution function."""
    
    # Define paths
    input_path = r'D:\Abel\TugasAkhir\data\squad\squad_v2_validation.csv'
    output_path = r'D:\Abel\TugasAkhir\data\squad\squad_v2_sampled_10percent.csv'
    
    # Validate input file exists
    if not os.path.exists(input_path):
        print(f'Error: Input file {input_path} not found.')
        return
    
    print('=' * 60)
    print('SQuAD v2 Stratified Sampling Pipeline')
    print('=' * 60)
    
    # Load dataset
    print('\n1. Loading dataset...')
    try:
        df = pd.read_csv(input_path)
        print(f'   Loaded {len(df)} rows')
    except Exception as e:
        print(f'Error loading CSV: {e}')
        return
    
    # Engineer features
    print('\n2. Engineering features...')
    df = engineer_features(df)
    print('   Created columns: type, answer_len, ans_len_cat')
    
    # Create stratum identifier
    df['_stratum'] = (
        df['is_impossible'].astype(str) + '_' +
        df['type'] + '_' +
        df['ans_len_cat']
    )
    
    n_strata = df['_stratum'].nunique()
    print(f'   Created {n_strata} strata')
    
    # Perform stratified sampling
    print('\n3. Performing 10% stratified sampling...')
    sampled_df = proportional_stratified_sample(
        df,
        strata_cols=['is_impossible', 'type', 'ans_len_cat'],
        sample_fraction=0.1,
        random_state=42
    )
    
    # Shuffle final dataset
    sampled_df = sampled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f'   Sampled {len(sampled_df)} rows ({len(sampled_df)/len(df)*100:.1f}%)')
    
    # Drop temporary stratum column
    sampled_df = sampled_df.drop(columns=['_stratum'])
    
    # Validation & Distribution Comparison
    print('\n4. Validation & Distribution Comparison')
    print('=' * 60)
    
    print(f'\nTotal rows: {len(sampled_df)} (Target: ~{int(len(df)*0.1)})')
    
    mean_diffs = print_distribution_comparison(
        df,
        sampled_df,
        ['is_impossible', 'type', 'ans_len_cat']
    )
    
    overall_mean_diff = np.mean(list(mean_diffs.values()))
    print(f'\n{"="*60}')
    print(f'Overall mean absolute % difference: {overall_mean_diff:.3f}%')
    
    # Save output
    print(f'\n5. Saving output...')
    try:
        sampled_df.to_csv(output_path, index=False)
        print(f'   Saved to {output_path}')
    except Exception as e:
        print(f'Error saving CSV: {e}')
        return
    
    print('\n' + '=' * 60)
    print('Pipeline completed successfully!')
    print('=' * 60)

if __name__ == '__main__':
    main()
