from transformers import pipeline

# Use the high-level pipeline for text generation
pipe = pipeline("text-generation", model="google/gemma-3-1b-it")
messages = [
    {"role": "user", "content": "Who are you?"},
]

# Run inference using the pipeline
result = pipe(messages)
print(result)
