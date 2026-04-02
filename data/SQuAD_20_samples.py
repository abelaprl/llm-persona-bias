"""Quick script to generate and save 20-sample JSON file"""
import json
import random
from collections import defaultdict

def count_words(text):
    return len(text.split())

def is_yes_no_question(question):
    prefix = question.strip().lower().split()[0].rstrip('?')
    return prefix in ['is', 'are', 'do', 'does', 'did', 'can', 'will', 'would', 'could', 'have', 'has']

def is_list_type(answer):
    has_comma = ',' in answer
    words = count_words(answer)
    has_and = 'and' in answer.lower() and words >= 6
    return has_comma or has_and

def get_question_type(question):
    q_lower = question.lower().split()[0].rstrip('?')
    if q_lower in ['what', 'who', 'when', 'where', 'why', 'how']:
        return q_lower
    return 'other'

# Load data
with open('data/SQuAD_validation.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

data = []
for item in dataset:
    data.append({
        'id': item['id'],
        'question': item['question'],
        'context': item['context'],
        'answer': item['answers']['text'][0] if item.get('answers', {}).get('text') else ''
    })

# Pre-filter
filtered = []
for item in data:
    if not item['answer'].strip():
        continue
    if count_words(item['answer']) < 1:
        continue
    if is_yes_no_question(item['question']):
        continue
    if count_words(item['context']) > 400:
        continue
    filtered.append(item)

# Compute features
for item in filtered:
    item['answer_length'] = count_words(item['answer'])
    item['is_list_type'] = is_list_type(item['answer'])
    item['question_type'] = get_question_type(item['question'])

# Categorize
pools = defaultdict(list)
for item in filtered:
    if item['is_list_type']:
        category = 'list'
    elif 1 <= item['answer_length'] <= 3:
        category = 'short'
    elif 4 <= item['answer_length'] <= 10:
        category = 'medium'
    elif item['answer_length'] > 10:
        category = 'long'
    else:
        continue
    
    item['category'] = category
    pools[category].append(item)

# Sample
samples = []
for category in ['short', 'medium', 'long', 'list']:
    pool = pools[category]
    random.shuffle(pool)
    samples.extend(pool[:5])

random.shuffle(samples)
samples = samples[:20]

# Save
with open('data/SQuAD_20_samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"✓ Saved 20 samples to data/SQuAD_20_samples.json")
