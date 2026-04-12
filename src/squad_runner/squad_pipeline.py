"""
SQuAD 20-Sample Evaluation Pipeline with 9 Persona Integration
Pipeline untuk menjalankan 20 sampel SQuAD melalui 9 persona (implicit, explicit, neutral)
Menggunakan: Allen AI: OLMo 3 32B Think
Total runs: 180 (20 samples x 9 personas)
"""

import json
import random
import requests
import time
import re
import string
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse


def sanitize_user_content(text):
    """Remove disallowed phrase from any user message content in SQuAD runs."""
    if not text:
        return text
    cleaned = text.replace("Solve the following question:", "")
    cleaned = cleaned.replace("solve the following question:", "")
    return cleaned.strip()


def load_samples_file(filepath):
    """Load pre-selected 20 samples from JSON file (no resampling)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError(f"Input sample file must be a list: {filepath}")

    return samples

# ============================================================================
# HELPER FUNCTIONS - SQuAD Metrics
# ============================================================================

def normalize_answer(s):
    """Normalize answer untuk SQuAD scoring (lowercase, remove punct, remove articles)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    return white_space_fix(remove_punc(remove_articles(s.lower())))

def f1_score(prediction, ground_truth):
    """Compute F1 score antara prediction dan ground truth"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = set(prediction_tokens) & set(ground_truth_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

def exact_match_score(prediction, ground_truth):
    """Compute exact match"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def extract_text_from_message(message):
    """Prefer assistant content; fallback to reasoning text when content is empty."""
    content = (message.get('content') or '').strip()
    if content:
        return content

    reasoning = (message.get('reasoning') or '').strip()
    if reasoning:
        # Some providers append final answer after </think>
        if '</think>' in reasoning:
            tail = reasoning.split('</think>')[-1].strip()
            if tail:
                return tail
        return reasoning

    details = message.get('reasoning_details') or []
    if isinstance(details, list):
        texts = []
        for item in details:
            if isinstance(item, dict):
                txt = (item.get('text') or '').strip()
                if txt:
                    texts.append(txt)
        if texts:
            merged = "\n".join(texts)
            if '</think>' in merged:
                tail = merged.split('</think>')[-1].strip()
                if tail:
                    return tail
            return merged

    return ''

# ============================================================================
# STEP 1: LOAD & FLATTEN SQuAD
# ============================================================================

def load_and_flatten_squad(filepath):
    """Load SQuAD validation dan flatten ke format [id, question, context, answer]"""
    print("[STEP 1] Loading SQuAD validation...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flattened = []
    for article in data.get('data', []):
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', '')
            for qa in paragraph.get('qas', []):
                qa_id = qa.get('id', '')
                question = qa.get('question', '')
                answers = qa.get('answers', [])
                
                if answers:
                    answer = answers[0]['text']
                else:
                    answer = ''
                
                flattened.append({
                    'id': qa_id,
                    'question': question,
                    'context': context,
                    'answer': answer
                })
    
    print(f"  Total samples: {len(flattened)}")
    return flattened

# ============================================================================
# STEP 2: SMART SAMPLING (20 samples)
# ============================================================================

def prefilter_samples(samples):
    """Pre-filter sampel yang tidak memenuhi syarat"""
    print("\n[STEP 2] Pre-filtering samples...")
    
    filtered = []
    removed_reasons = defaultdict(int)
    
    yes_no_starts = ('is ', 'are ', 'do ', 'does ', 'did ', 'can ', 'could ', 'would ', 'should ', 'will ', 'have ', 'has ')
    
    seen_questions = set()
    
    for sample in samples:
        answer = sample['answer'].strip()
        question = sample['question'].strip()
        context = sample['context'].strip()
        
        # Rule 1: Empty answer
        if not answer:
            removed_reasons['empty_answer'] += 1
            continue
        
        # Rule 2: Answer < 1 word
        if len(answer.split()) < 1:
            removed_reasons['too_short_answer'] += 1
            continue
        
        # Rule 3: Yes/No question
        if question.lower().startswith(yes_no_starts):
            removed_reasons['yes_no_question'] += 1
            continue
        
        # Rule 4: Context > 400 words
        if len(context.split()) > 400:
            removed_reasons['context_too_long'] += 1
            continue
        
        # Rule 5: Duplicate question
        if question in seen_questions:
            removed_reasons['duplicate'] += 1
            continue
        
        seen_questions.add(question)
        filtered.append(sample)
    
    print(f"  Removed: {sum(removed_reasons.values())} samples")
    for reason, count in removed_reasons.items():
        print(f"    - {reason}: {count}")
    print(f"  Remaining: {len(filtered)}")
    
    return filtered

def compute_features(samples):
    """Compute answer_length, is_list_type, question_type untuk setiap sample"""
    assert_threshold = 0.001
    
    featured = []
    
    for sample in samples:
        answer = sample['answer'].strip()
        question = sample['question'].lower().strip()
        
        # Feature 1: answer_length
        answer_length = len(answer.split())
        
        # Feature 2: is_list_type
        is_list = (',' in answer or (
            'and' in answer and len(answer.split()) >= 6
        ))
        
        # Feature 3: question_type
        qtype = 'other'
        for qt in ['what', 'who', 'when', 'where', 'why', 'how']:
            if question.startswith(qt):
                qtype = qt
                break
        
        sample_with_features = sample.copy()
        sample_with_features['answer_length'] = answer_length
        sample_with_features['is_list_type'] = is_list
        sample_with_features['question_type'] = qtype
        
        featured.append(sample_with_features)
    
    return featured

def smart_sample_20(samples):
    """Kategorisasi dan sampling 20 sampel dengan kategori balanced"""
    print("\n[STEP 2.2] Smart sampling for 20 balanced samples...")
    
    # Compute features
    samples = compute_features(samples)
    
    # Kategorisasi
    pools = {
        'short': [],
        'medium': [],
        'long': [],
        'list': []
    }
    
    for sample in samples:
        if sample['is_list_type']:
            pools['list'].append(sample)
        elif sample['answer_length'] <= 3:
            pools['short'].append(sample)
        elif sample['answer_length'] <= 10:
            pools['medium'].append(sample)
        else:
            pools['long'].append(sample)
    
    print("  Pool sizes:")
    for cat, items in pools.items():
        print(f"    - {cat}: {len(items)}")
    
    # Shuffle & sample
    samples_20 = []
    for cat, count in [('short', 5), ('medium', 5), ('long', 5), ('list', 5)]:
        random.shuffle(pools[cat])
        selected = pools[cat][:count]
        
        # Diversity check: hindari tipe pertanyaan yang terlalu sama
        # Simple strategy: prefer diverse question types
        selected = ensure_question_diversity(selected)[:count]
        samples_20.extend(selected)
    
    # Post-check
    print("\n  Post-check validation:")
    
    cat_counts = defaultdict(int)
    qtype_counts = defaultdict(int)
    
    for s in samples_20:
        # Determine category
        if s['is_list_type']:
            cat = 'list'
        elif s['answer_length'] <= 3:
            cat = 'short'
        elif s['answer_length'] <= 10:
            cat = 'medium'
        else:
            cat = 'long'
        
        cat_counts[cat] += 1
        qtype_counts[s['question_type']] += 1
    
    print(f"    - Total: {len(samples_20)}")
    for cat in ['short', 'medium', 'long', 'list']:
        print(f"    - {cat}: {cat_counts[cat]}")
    
    why_how = qtype_counts.get('why', 0) + qtype_counts.get('how', 0)
    print(f"    - why/how questions: {why_how}")
    print(f"    - Unique questions: {len(set(s['id'] for s in samples_20))}")
    
    return samples_20

def ensure_question_diversity(samples):
    """Hindari terlalu banyak tipe pertanyaan yang sama"""
    # Sort by question_type to spread them
    return sorted(samples, key=lambda x: x['question_type'])

# ============================================================================
# STEP 3: LOAD PERSONAS
# ============================================================================

def load_implicit_warmup(persona_name):
    """Load implicit persona warmup"""
    result_path = Path(f"result/allenai_olmo-3-32b-think/{persona_name}/warmup/persona_warmup.json")
    
    if not result_path.exists():
        print(f"    [WARN] Warmup not found for {persona_name}: {result_path}")
        return None
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"    [ERR] Failed to load warmup for {persona_name}: {e}")
        return None

def load_explicit_persona_prompt(persona_name):
    """Load explicit persona prompt"""
    input_types = {
        'man_explicit_1': 'explicits_man_prompts.json',
        'woman_explicit_1': 'explicits_woman_prompts.json'
    }
    
    if persona_name not in input_types:
        return None
    
    input_file = input_types[persona_name]
    filepath = Path(f"inputs/{input_file}")
    
    if not filepath.exists():
        print(f"    [WARN] Explicit prompt not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Assume first entry is persona system prompt
            if isinstance(data, list) and len(data) > 0:
                return data[0].get('prompt', '')
            elif isinstance(data, dict):
                return data.get('prompt', '')
    except Exception as e:
        print(f"    [ERR] Failed to load explicit prompt for {persona_name}: {e}")
        return None

def load_persona_configs():
    """Load semua 9 persona configurations"""
    print("\n[STEP 3] Loading persona configurations...")
    
    personas = {
        # Implicit: 6
        'man_implicit_1': {
            'type': 'implicit',
            'warmup': load_implicit_warmup('man_implicit_1'),
            'explicit_prompt': None
        },
        'man_implicit_2': {
            'type': 'implicit',
            'warmup': load_implicit_warmup('man_implicit_2'),
            'explicit_prompt': None
        },
        'man_implicit_3': {
            'type': 'implicit',
            'warmup': load_implicit_warmup('man_implicit_3'),
            'explicit_prompt': None
        },
        'woman_implicit_1': {
            'type': 'implicit',
            'warmup': load_implicit_warmup('woman_implicit_1'),
            'explicit_prompt': None
        },
        'woman_implicit_2': {
            'type': 'implicit',
            'warmup': load_implicit_warmup('woman_implicit_2'),
            'explicit_prompt': None
        },
        'woman_implicit_3': {
            'type': 'implicit',
            'warmup': load_implicit_warmup('woman_implicit_3'),
            'explicit_prompt': None
        },
        # Explicit: 2
        'man_explicit_1': {
            'type': 'explicit',
            'warmup': None,
            'explicit_prompt': load_explicit_persona_prompt('man_explicit_1')
        },
        'woman_explicit_1': {
            'type': 'explicit',
            'warmup': None,
            'explicit_prompt': load_explicit_persona_prompt('woman_explicit_1')
        },
        # Neutral: 1
        'neutral': {
            'type': 'neutral',
            'warmup': None,
            'explicit_prompt': None
        }
    }
    
    # Status report
    for pname, pconfig in personas.items():
        status = "OK"
        if pconfig['type'] == 'implicit' and pconfig['warmup'] is None:
            status = "MISSING WARMUP"
        elif pconfig['type'] == 'explicit' and pconfig['explicit_prompt'] is None:
            status = "MISSING PROMPT"
        
        print(f"  [{pname:20s}] type={pconfig['type']:10s} status={status}")
    
    return personas

# ============================================================================
# STEP 4: BUILD MESSAGES FOR EACH PERSONA
# ============================================================================

def build_messages_implicit(persona_config, question, context):
    """Build messages untuk implicit persona (dengan warmup history)"""
    warmup = persona_config['warmup']
    
    if not warmup:
        return None
    
    warmup_q = sanitize_user_content(warmup.get('input', {}).get('prompt', ''))
    warmup_a = warmup.get('response', {}).get('content', '')
    
    system_msg = "You are a helpful AI assistant."
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": warmup_q},
        {"role": "assistant", "content": warmup_a},
        {
            "role": "user",
            "content": sanitize_user_content(
                f"Answer questions using ONLY the given context. Keep the answer concise.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            )
        }
    ]
    
    return messages

def build_messages_explicit(persona_config, question, context):
    """Build messages untuk explicit persona (dengan persona prompt)"""
    persona_prompt = sanitize_user_content(persona_config['explicit_prompt'])
    
    if not persona_prompt:
        return None
    
    system_msg = "You are a helpful AI assistant."

    combined_user = sanitize_user_content(
        f"{persona_prompt}\n\n"
        f"Answer questions using ONLY the given context. Keep the answer concise.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": combined_user}
    ]
    
    return messages

def build_messages_neutral(question, context):
    """Build messages untuk neutral persona (direct QA)"""
    system_msg = "You are a helpful AI assistant."
    
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": sanitize_user_content(
                f"Answer questions using ONLY the given context. Keep the answer concise.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            )
        }
    ]
    
    return messages

# ============================================================================
# STEP 5: API CALL TO OPENROUTER
# ============================================================================

def call_openrouter_api(model_config, messages, retry_count=0, max_retries=3):
    """Call OpenRouter API dengan retry mechanism"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_config['api_key_alias']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_config['model_id'],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            api_response = response.json()
            return {
                "success": True,
                "api_response": api_response,
                "latency": latency,
                "error": None,
                "retry_count": retry_count
            }
        elif response.status_code in [429, 500, 502, 503, 504] and retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            print(f"      [RETRY] Status {response.status_code} - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_config, messages, retry_count + 1, max_retries)
        else:
            return {
                "success": False,
                "api_response": None,
                "latency": latency,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "retry_count": retry_count
            }
    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            wait_time = 5 * (2 ** retry_count)
            print(f"      [RETRY] Timeout - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_config, messages, retry_count + 1, max_retries)
        else:
            return {
                "success": False,
                "api_response": None,
                "latency": time.time() - start_time,
                "error": "Timeout after retries",
                "retry_count": retry_count
            }
    except Exception as e:
        if retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            print(f"      [RETRY] Exception - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_config, messages, retry_count + 1, max_retries)
        else:
            return {
                "success": False,
                "api_response": None,
                "latency": time.time() - start_time,
                "error": str(e),
                "retry_count": retry_count
            }

# ============================================================================
# STEP 6: SAVE JSON & AGGREGATE
# ============================================================================

def get_answer_category(sample):
    """Determine category dari sample"""
    if sample.get('is_list_type'):
        return 'list'
    elif sample.get('answer_length', 0) <= 3:
        return 'short'
    elif sample.get('answer_length', 0) <= 10:
        return 'medium'
    else:
        return 'long'

def create_result_json(model_config, persona_name, persona_type, sample, messages, api_result):
    """Create result JSON"""
    timestamp = datetime.now().isoformat()
    category = get_answer_category(sample)
    
    result = {
        "metadata": {
            "run_id": f"squad_{sample['id']}_{persona_name}",
            "experiment_type": "squad_bias_study",
            "dataset": "SQuAD_validation",
            "timestamp": timestamp,
            "model_info": {
                "provider": "openrouter",
                "model_id": model_config['model_id'],
                "model_name": model_config['name']
            }
        },
        "input": {
            "sample_id": sample['id'],
            "question": sample['question'],
            "context": sample['context'],
            "ground_truth": sample['answer'],
            "persona": {
                "name": persona_name,
                "type": persona_type
            },
            "category": category
        },
        "request": {
            "messages": messages,
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048,
                "model": model_config['model_id']
            }
        }
    }
    
    if api_result['success']:
        api_resp = api_result['api_response']
        choice = api_resp.get('choices', [{}])[0]
        message = choice.get('message', {})
        finish_reason = (choice.get('finish_reason') or '').lower()
        usage = api_resp.get('usage', {}) or {}
        completion_tokens = usage.get('completion_tokens', 0) or 0
        max_tokens = result['request']['parameters'].get('max_tokens', 0) or 0

        # Avoid promoting incomplete reasoning text when generation hit token limit.
        if finish_reason == 'length' or (max_tokens and completion_tokens >= max_tokens):
            prediction = ''
        else:
            prediction = extract_text_from_message(message)
        em = exact_match_score(prediction, sample['answer'])
        f1 = f1_score(prediction, sample['answer'])
        
        result["response"] = {
            "success": True,
            "prediction": prediction,
            "raw_api_response": api_resp,
            "error": None,
            "performance": {
                "latency_sec": api_result['latency'],
                "tokens": api_resp.get('usage', {}),
                "retry_count": api_result['retry_count']
            }
        }
        
        result["analysis"] = {
            "exact_match": em,
            "f1_score": f1
        }
    else:
        result["response"] = {
            "success": False,
            "prediction": None,
            "raw_api_response": None,
            "error": api_result['error'],
            "performance": {
                "latency_sec": api_result['latency'],
                "tokens": {},
                "retry_count": api_result['retry_count']
            }
        }
        
        result["analysis"] = {
            "exact_match": 0.0,
            "f1_score": 0.0
        }
    
    return result

def output_base_folder(model_config):
    """Output base folder based on model_id."""
    model_slug = model_config['model_id'].replace('/', '_')
    return Path("result") / "SQuAD" / model_slug


def save_result_json(result_json, model_config, persona_name, sample_id):
    """Save result ke JSON file"""
    folder = output_base_folder(model_config) / persona_name
    folder.mkdir(parents=True, exist_ok=True)
    
    filename = f"squad_{sample_id}.json"
    filepath = folder / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        return str(filepath)
    except Exception as e:
        print(f"    [ERR] Failed to save JSON: {e}")
        return None


def result_exists(model_config, persona_name, sample_id):
    """Check if result JSON already exists (resume support)."""
    filepath = output_base_folder(model_config) / persona_name / f"squad_{sample_id}.json"
    return filepath.exists()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_model_config(order):
    """Load model configuration dari config/model_keys.json"""
    with open('config/model_keys.json', 'r', encoding='utf-8') as f:
        models = json.load(f)
    for m in models:
        if m.get('order') == order:
            return m
    return None

def main():
    parser = argparse.ArgumentParser(description='SQuAD 20-Sample Pipeline dengan 9 Persona')
    parser.add_argument('--dry-run', action='store_true', help='Hanya buat sampling tanpa API call')
    parser.add_argument('--samples_file', type=str, default='data/squad_20_cleaned.json', help='Path ke file 20 sampel siap pakai')
    parser.add_argument('--limit_samples', type=int, default=0, help='Limit jumlah sample dari awal file (0 = semua)')
    args = parser.parse_args()
    
    print("="*80)
    print("SQuAD 20-SAMPLE EVALUATION PIPELINE")
    print("="*80)
    
    # STEP 1: Load fixed 20 samples (no resampling)
    print("[STEP 1] Loading fixed sample file...")
    samples_20 = load_samples_file(args.samples_file)
    samples_20 = sorted(samples_20, key=lambda x: x['id'])  # Sort by ID
    if args.limit_samples and args.limit_samples > 0:
        samples_20 = samples_20[:args.limit_samples]
    print(f"  Loaded samples: {len(samples_20)}")
    
    print("\n[STEP 2] Using provided samples:")
    for i, s in enumerate(samples_20, 1):
        cat = get_answer_category(s)
        print(f"  {i:2d}. [{cat:6s}] {s['id']:20s} Q: {s['question'][:40]}")
    
    # STEP 3: Load personas
    personas = load_persona_configs()
    
    if args.dry_run:
        print("\n[DRY RUN] Skipping API calls.")
        print(f"\nWould run: {len(samples_20)} samples x {len(personas)} personas = {len(samples_20) * len(personas)} total tasks")
        return
    
    # STEP 4: Load model
    model_config = load_model_config(42)  # Order 42 = Allen AI: OLMo 3 32B Think
    if not model_config:
        print("[ERR] Model not found!")
        return
    
    print(f"\n[STEP 4] Using model: {model_config['name']}")
    print(f"         Model ID: {model_config['model_id']}")
    
    # STEP 5: Main loop
    print(f"\n{'='*80}")
    print(f"EXECUTION LOOP: {len(samples_20)} samples x {len(personas)} personas = {len(samples_20) * len(personas)} tasks")
    print(f"{'='*80}")
    
    csv_rows = []
    overall_start = time.time()
    task_count = 0
    success_count = 0
    failed_count = 0
    
    for persona_name in sorted(personas.keys()):
        persona_config = personas[persona_name]
        persona_type = persona_config['type']
        
        print(f"\n[PERSONA: {persona_name:20s} ({persona_type:10s})]")
        
        for sample in samples_20:
            task_count += 1
            progress = f"{task_count}/{len(samples_20) * len(personas)}"
            
            print(f"  [{progress}] {sample['id']:20s} ...", end=' ', flush=True)

            if result_exists(model_config, persona_name, sample['id']):
                print("[SKIP] exists")
                continue
            
            # Build messages
            if persona_type == 'implicit':
                messages = build_messages_implicit(persona_config, sample['question'], sample['context'])
            elif persona_type == 'explicit':
                messages = build_messages_explicit(persona_config, sample['question'], sample['context'])
            else:  # neutral
                messages = build_messages_neutral(sample['question'], sample['context'])
            
            if not messages:
                print("[SKIP] Failed to build messages")
                failed_count += 1
                continue
            
            # Call API
            api_result = call_openrouter_api(model_config, messages)
            
            # Create & save JSON
            result_json = create_result_json(model_config, persona_name, persona_type, sample, messages, api_result)
            filepath = save_result_json(result_json, model_config, persona_name, sample['id'])
            
            # Add to CSV aggregation
            if api_result['success']:
                prediction = result_json['response']['prediction']
                em = result_json['analysis']['exact_match']
                f1 = result_json['analysis']['f1_score']
                status = "SUCCESS"
                print(f"[OK] EM={em:.1f} F1={f1:.3f}")
                success_count += 1
            else:
                prediction = None
                em = 0.0
                f1 = 0.0
                status = "FAILED"
                print(f"[FAIL]")
                failed_count += 1
            
            csv_rows.append({
                'id': sample['id'],
                'persona_name': persona_name,
                'persona_type': persona_type,
                'question': sample['question'],
                'prediction': prediction if prediction else '',
                'ground_truth': sample['answer'],
                'category': get_answer_category(sample),
                'exact_match': em,
                'f1_score': f1,
                'status': status
            })
            
            # Rate limiting
            time.sleep(0.5)
    
    # STEP 6: Save CSV
    overall_elapsed = time.time() - overall_start
    
    model_slug = model_config['model_id'].replace('/', '_')
    csv_path = output_base_folder(model_config) / f"result_squad_{model_slug}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'persona_name', 'persona_type', 'question', 'prediction', 'ground_truth', 'category', 'exact_match', 'f1_score', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\n[CSV] Saved: {csv_path}")
    except Exception as e:
        print(f"\n[ERR] Failed to save CSV: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {task_count}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    
    # Compute aggregate metrics
    total_em = sum(r['exact_match'] for r in csv_rows)
    total_f1 = sum(r['f1_score'] for r in csv_rows)
    
    print(f"\nAggregate Metrics:")
    print(f"  Avg EM: {total_em / len(csv_rows):.4f}")
    print(f"  Avg F1: {total_f1 / len(csv_rows):.4f}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
