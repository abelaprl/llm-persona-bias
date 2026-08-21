"""16 persona religion. Urutan tetap agar CSV antar model sebanding."""

PERSONA_GROUP = 'Religion_Persona'

PERSONAS = [
    'atheist_explicit', 'atheist_implicit_1', 'atheist_implicit_2', 'atheist_implicit_3',
    'christian_explicit', 'christian_implicit_1', 'christian_implicit_2', 'christian_implicit_3',
    'islam_explicit', 'islam_implicit_1', 'islam_implicit_2', 'islam_implicit_3',
    'jewish_explicit', 'jewish_implicit_1', 'jewish_implicit_2', 'jewish_implicit_3',
]

MODELS = [
    'deepseek_deepseek-r1-distill-qwen-32b',
    'meta-llama_llama-3.1-8b-instruct',
    'meta-llama_llama-3.3-70b-instruct',
    'mistralai_ministral-8b-2512',
    'openai_gpt-oss-20b',
    'qwen_qwen3-next-80b-a3b-thinking',
]
