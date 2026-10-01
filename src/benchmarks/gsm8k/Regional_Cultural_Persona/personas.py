"""Taksonomi Regional Cultural utama: 4 region × 4 varian persona."""

PERSONA_GROUP = 'Regional_Cultural_Persona'

PERSONAS = [
    'north_american_explicit',
    'north_american_implicit_1',
    'north_american_implicit_2',
    'north_american_implicit_3',
    'east_asian_explicit',
    'east_asian_implicit_1',
    'east_asian_implicit_2',
    'east_asian_implicit_3',
    'south_asian_explicit',
    'south_asian_implicit_1',
    'south_asian_implicit_2',
    'south_asian_implicit_3',
    'sub_saharan_african_explicit',
    'sub_saharan_african_implicit_1',
    'sub_saharan_african_implicit_2',
    'sub_saharan_african_implicit_3',
]

# OLMo sengaja tidak masuk: data parsialnya adalah eksperimen lama dan bukan
# bagian dari enam model utama penelitian.
MODELS = [
    'deepseek_deepseek-r1-distill-qwen-32b',
    'meta-llama_llama-3.1-8b-instruct',
    'meta-llama_llama-3.3-70b-instruct',
    'mistralai_ministral-8b-2512',
    'openai_gpt-oss-20b',
    'qwen_qwen3-next-80b-a3b-thinking',
]

