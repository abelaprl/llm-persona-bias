"""Kolom results.csv — sama untuk semua model."""

COLUMNS = [
    'model', 'persona_group', 'persona', 'question_id', 'file',
    'ground_truth_raw', 'ground_truth_answer',
    'response_content_full',
    'response_success', 'response_error_present', 'error_class', 'error_message',
    'input_json_valid', 'technical_failure', 'content_empty',
    'reasoning_present', 'think_block_stripped',
    'extracted_answer', 'candidate_answer', 'answer_status',
    'rule_pack', 'parse_source', 'priority_hit', 'extraction_confidence',
    'abstain_reason', 'is_correct',
    'finish_reason', 'truncated_by_metadata', 'truncated_by_content',
    'prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens',
    'latency_sec', 'retry_count',
    'parser_version', 'text_tail',
]
