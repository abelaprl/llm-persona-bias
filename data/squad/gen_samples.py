"""Generate 20 balanced SQuAD samples"""
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
    return q_lower if q_lower in ['what', 'who', 'when', 'where', 'why', 'how'] else 'other'

with open('data/squad/SQuAD_validation.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

data = []
for item in dataset:
    data.append({
        'id': item['id'],
        'question': item['question'],
        'context': item['context'],
        'answer': item['answers']['text'][0] if item.get('answers', {}).get('text') else ''
    })

filtered = [item for item in data if item['answer'].strip() and count_words(item['answer']) >= 1 
            and not is_yes_no_question(item['question']) and count_words(item['context']) <= 400]

for item in filtered:
    item['answer_length'] = count_words(item['answer'])
    item['is_list_type'] = is_list_type(item['answer'])
    item['question_type'] = get_question_type(item['question'])

pools = defaultdict(list)
for item in filtered:
    if item['is_list_type']:
        cat = 'list'
    elif 1 <= item['answer_length'] <= 3:
        cat = 'short'
    elif 4 <= item['answer_length'] <= 10:
        cat = 'medium'
    elif item['answer_length'] > 10:
        cat = 'long'
    else:
        continue
    item['category'] = cat
    pools[cat].append(item)

samples = []
for cat in ['short', 'medium', 'long', 'list']:
    pool = pools[cat]
    random.shuffle(pool)
    samples.extend(pool[:5])

random.shuffle(samples)
samples = samples[:20]

with open('data/squad/SQuAD_20_samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

cat_counts = {cat: sum(1 for s in samples if s['category'] == cat) for cat in ['short', 'medium', 'long', 'list']}
print(f'✓ Saved 20 samples to data/squad/SQuAD_20_samples.json')
print(f'Categories: {cat_counts}')
