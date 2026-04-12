"""
SQuAD 20-Sample Evaluation Pipeline - Test Mode
Generates test JSONs for 3 persona types (implicit, explicit, neutral)
"""
import requests
import json
import argparse
import time
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import random

class SQuADSampler:
    def __init__(self):
        self.data = []
        self.load_squad()
    
    def load_squad(self):
        """Load and flatten SQuAD validation data"""
        with open('data/SQuAD_validation.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Data is already flat (format: list of QA items)
        for item in dataset:
            self.data.append({
                'id': item['id'],
                'question': item['question'],
                'context': item['context'],
                'answer': item['answers']['text'][0] if item.get('answers', {}).get('text') else ''
            })
        print(f"[OK] Loaded {len(self.data)} SQuAD samples")
    
    def _is_yes_no_question(self, question):
        """Check if question is yes/no type"""
        prefix = question.strip().lower().split()[0].rstrip('?')
        return prefix in ['is', 'are', 'do', 'does', 'did', 'can', 'will', 'would', 'could', 'have', 'has']
    
    def _count_words(self, text):
        """Count words in text"""
        return len(text.split())
    
    def _is_list_type(self, answer):
        """Detect if answer is list-type"""
        has_comma = ',' in answer
        words = self._count_words(answer)
        has_and = 'and' in answer.lower() and words >= 6
        return has_comma or has_and
    
    def _get_question_type(self, question):
        """Classify question type"""
        q_lower = question.lower().split()[0].rstrip('?')
        if q_lower in ['what', 'who', 'when', 'where', 'why', 'how']:
            return q_lower
        return 'other'
    
    def _prefilter(self):
        """Remove invalid samples"""
        original_count = len(self.data)
        filtered = []
        
        for item in self.data:
            # Check: empty answer
            if not item['answer'].strip():
                continue
            
            # Check: answer < 1 word
            if self._count_words(item['answer']) < 1:
                continue
            
            # Check: yes/no question
            if self._is_yes_no_question(item['question']):
                continue
            
            # Check: context > 400 words
            if self._count_words(item['context']) > 400:
                continue
            
            filtered.append(item)
        
        self.data = filtered
        print(f"[FILTER] {original_count} -> {len(self.data)} (removed {original_count - len(self.data)})")
    
    def _compute_features(self):
        """Add features to each sample"""
        for item in self.data:
            item['answer_length'] = self._count_words(item['answer'])
            item['is_list_type'] = self._is_list_type(item['answer'])
            item['question_type'] = self._get_question_type(item['question'])
    
    def _categorize(self):
        """Categorize samples (short/medium/long/list)"""
        pools = defaultdict(list)
        
        for item in self.data:
            if item['is_list_type']:
                category = 'list'
            elif 1 <= item['answer_length'] <= 3:
                category = 'short'
            elif 4 <= item['answer_length'] <= 10:
                category = 'medium'
            elif item['answer_length'] > 10:
                category = 'long'
            else:
                continue  # Skip if doesn't fit
            
            item['category'] = category
            pools[category].append(item)
        
        return pools
    
    def _diversity_filter(self, samples):
        """Ensure diversity: avoid similar questions, same entities"""
        if len(samples) <= 5:
            return samples
        
        selected = []
        seen_patterns = set()
        
        # Shuffle before selecting
        random.shuffle(samples)
        
        for item in samples:
            # Simple pattern: first 3 words of question
            pattern = tuple(item['question'].split()[:3])
            if pattern not in seen_patterns:
                selected.append(item)
                seen_patterns.add(pattern)
                if len(selected) >= 5:
                    break
        
        # If not enough viable samples, just take first 5
        if len(selected) < 5:
            selected = samples[:5]
        
        return selected
    
    def sample_20(self):
        """Select EXACTLY 20 balanced samples"""
        print("\n[SAMPLING] Starting smart sampling...")
        
        # Step 1: Pre-filter
        self._prefilter()
        
        # Step 2: Compute features
        self._compute_features()
        
        # Step 3: Categorize
        pools = self._categorize()
        
        print(f"[POOLS] short={len(pools['short'])}, medium={len(pools['medium'])}, long={len(pools['long'])}, list={len(pools['list'])}")
        
        # Step 4: Ensure each pool has enough samples
        min_required = 5
        for cat in ['short', 'medium', 'long', 'list']:
            if len(pools[cat]) < min_required:
                print(f"[WARNING] {cat} has only {len(pools[cat])} samples (need {min_required})")
        
        # Step 5: Sample 5 from each pool
        samples = []
        for category in ['short', 'medium', 'long', 'list']:
            pool = pools[category]
            if len(pool) < 5:
                selected = pool  # Take all available
            else:
                # Apply diversity filter then sample
                selected = self._diversity_filter(pool)
            
            samples.extend(selected)
            print(f"[SAMPLE] {category}: {len(selected)}/5")
        
        # Shuffle final 20
        random.shuffle(samples)
        
        # Step 6: Validation
        validate_msg = self._validate_samples(samples)
        print(f"[VALIDATE] {validate_msg}")
        
        return samples[:20]
    
    def _validate_samples(self, samples):
        """Validate sampling results"""
        by_category = defaultdict(int)
        by_qtype = defaultdict(int)
        
        for s in samples:
            by_category[s['category']] += 1
            by_qtype[s['question_type']] += 1
        
        why_how = by_qtype.get('why', 0) + by_qtype.get('how', 0)
        
        return f"total={len(samples)}, categories={dict(by_category)}, why/how={why_how}"

def load_model_config():
    """Load OLMo 3 32B Think model config"""
    with open('config/model_keys.json', 'r', encoding='utf-8') as f:
        models = json.load(f)
    for m in models:
        if m['order'] == 42:  # OLMo 3 32B Think
            return m
    raise ValueError("Model order 42 (OLMo 3 32B Think) not found")

def load_warmup(persona_name):
    """Load warmup history for implicit persona"""
    warmup_path = Path(f"result/allenai_olmo-3-32b-think/{persona_name}/warmup/persona_warmup.json")
    if not warmup_path.exists():
        return None
    
    with open(warmup_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_explicit_prompt(persona_name):
    """Load explicit persona prompt"""
    gender = 'man' if 'man' in persona_name else 'woman'
    prompts_path = Path(f"inputs/explicits_{gender}_prompts.json")
    
    if not prompts_path.exists():
        return None
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    # Return first prompt as default
    return prompts[0]['prompt'] if prompts else None

def build_messages(squad_item, persona_name, persona_type, model_config):
    """Build messages array based on persona type"""
    system_msg = "You are a helpful AI assistant."
    context_msg = f"Solve the following question:\n\nAnswer the question using ONLY the given context. Keep the answer concise.\n\nContext: {squad_item['context']}\n\nQuestion: {squad_item['question']}\n\nAnswer:"
    
    if persona_type == 'implicit':
        # Load warmup and prepend
        warmup = load_warmup(persona_name)
        if not warmup:
            raise ValueError(f"Warmup not found for {persona_name}")
        
        warmup_q = warmup['input']['prompt']
        warmup_a = warmup['response']['content']
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": warmup_q},
            {"role": "assistant", "content": warmup_a},
            {"role": "user", "content": context_msg}
        ]
    
    elif persona_type == 'explicit':
        # Load explicit prompt
        explicit_prompt = load_explicit_prompt(persona_name)
        if not explicit_prompt:
            raise ValueError(f"Explicit prompt not found for {persona_name}")
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": explicit_prompt},
            {"role": "user", "content": context_msg}
        ]
    
    else:  # neutral
        # Direct QA without persona
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": context_msg}
        ]
    
    return messages

def call_openrouter_api(model_config, messages, retry_count=0, max_retries=3):
    """Call OpenRouter API with retry"""
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
            return {
                "success": True,
                "api_response": response.json(),
                "error": None,
                "latency": latency,
                "retry_count": retry_count
            }
        elif response.status_code in [429, 500, 502, 503, 504] and retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            print(f"        [RETRY] Status {response.status_code}, waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_config, messages, retry_count + 1, max_retries)
        else:
            return {
                "success": False,
                "api_response": None,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "latency": latency,
                "retry_count": retry_count
            }
    except Exception as e:
        if retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            print(f"        [RETRY] Exception, waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_config, messages, retry_count + 1, max_retries)
        return {
            "success": False,
            "api_response": None,
            "error": str(e),
            "latency": time.time() - start_time,
            "retry_count": retry_count
        }

def create_result_json(model_config, squad_item, persona_name, persona_type, messages, api_result):
    """Create result JSON"""
    timestamp = datetime.now().isoformat()
    
    result = {
        "metadata": {
            "run_id": f"squad_{persona_type}_{persona_name}_{squad_item['id']}",
            "experiment_type": "squad_persona_evaluation",
            "timestamp": timestamp,
            "model_info": {
                "provider": "openrouter",
                "model_id": model_config['model_id'],
                "model_name": model_config['name']
            }
        },
        "input": {
            "sample_id": squad_item['id'],
            "question": squad_item['question'],
            "context": squad_item['context'],
            "ground_truth": squad_item['answer'],
            "category": squad_item['category'],
            "persona": {
                "name": persona_name,
                "type": persona_type
            }
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
        choice = api_resp['choices'][0]
        message = choice['message']
        
        result["response"] = {
            "success": True,
            "prediction": message.get('content', ''),
            "raw_api_response": api_resp,
            "error": None,
            "performance": {
                "latency_sec": api_result.get('latency', 0),
                "tokens": api_resp.get('usage', {}),
                "retry_count": api_result.get('retry_count', 0)
            }
        }
    else:
        result["response"] = {
            "success": False,
            "prediction": None,
            "raw_api_response": None,
            "error": api_result['error'],
            "performance": {
                "latency_sec": api_result.get('latency', 0),
                "tokens": {},
                "retry_count": api_result.get('retry_count', 0)
            }
        }
    
    return result

def save_result(persona_name, persona_type, squad_item, result_json):
    """Save result JSON"""
    folder_path = Path(f"result/SQuAD/allenai_olmo-3-32b-think/{persona_type}_{persona_name}")
    folder_path.mkdir(parents=True, exist_ok=True)
    
    file_path = folder_path / f"squad_{squad_item['id']}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    return str(file_path)

def main():
    parser = argparse.ArgumentParser(description='SQuAD Test Generation - 3 Personas')
    parser.add_argument('--skip_api', action='store_true', help='Skip API call (mock data)')
    args = parser.parse_args()
    
    print("="*80)
    print("SQUAD TEST GENERATION - 3 PERSONAS")
    print("="*80)
    
    # Step 1: Load model
    model_config = load_model_config()
    print(f"[OK] Model: {model_config['name']}")
    
    # Step 2: Sample SQuAD
    sampler = SQuADSampler()
    samples = sampler.sample_20()
    print(f"[OK] Selected {len(samples)} samples")
    
    # Step 3: Test 3 personas with first 3 samples
    test_personas = [
        ('man_explicit_1', 'explicit'),
        ('man_implicit_1', 'implicit'),
        ('neutral', 'neutral')
    ]
    test_samples = samples[:3]
    
    print(f"\n{'='*80}")
    print("RUNNING TEST GENERATION: 3 personas × 1 sample each")
    print(f"{'='*80}")
    
    for persona_name, persona_type in test_personas:
        print(f"\n[PERSONA] {persona_type}/{persona_name}")
        
        sample = test_samples[0]  # Use first sample for all
        print(f"  Sample: {sample['id']}")
        print(f"  Category: {sample['category']}")
        print(f"  Question: {sample['question'][:60]}...")
        
        try:
            # Build messages
            messages = build_messages(sample, persona_name, persona_type, model_config)
            print(f"  [MSG] Built {len(messages)} messages")
            
            # Call API (or skip if --skip_api)
            if args.skip_api:
                print(f"  [SKIP] Skipping API call")
                api_result = {
                    "success": True,
                    "api_response": {
                        "choices": [{
                            "message": {
                                "content": "Mock prediction for testing"
                            }
                        }],
                        "usage": {
                            "prompt_tokens": 178,
                            "completion_tokens": 96,
                            "total_tokens": 274,
                            "cost": 1.398e-05,
                            "is_byok": False,
                            "prompt_tokens_details": {
                                "cached_tokens": 0,
                                "audio_tokens": 0
                            },
                            "cost_details": {
                                "upstream_inference_cost": 1.398e-05,
                                "upstream_inference_prompt_cost": 5.34e-06,
                                "upstream_inference_completions_cost": 8.64e-06
                            },
                            "completion_tokens_details": {
                                "reasoning_tokens": 0,
                                "audio_tokens": 0
                            }
                        }
                    },
                    "error": None,
                    "latency": 0.0,
                    "retry_count": 0
                }
            else:
                print(f"  [API] Calling OpenRouter...")
                api_result = call_openrouter_api(model_config, messages)
            
            if api_result['success']:
                print(f"  [OK] Got response ({len(api_result['api_response']['choices'][0]['message']['content'])} chars)")
            else:
                print(f"  [FAIL] {api_result['error']}")
            
            # Create and save JSON
            result_json = create_result_json(model_config, sample, persona_name, persona_type, messages, api_result)
            file_path = save_result(persona_name, persona_type, sample, result_json)
            print(f"  [SAVE] {file_path}")
            
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
    
    print(f"\n{'='*80}")
    print("TEST GENERATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
