from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model and tokenizer
model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Example messages (can be replaced with your own prompt)
messages = [
    {"role": "user", "content": "Who are you?"},
]

# Prepare input for chat template
tokens = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
tokens = tokens.to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**tokens, max_new_tokens=40)

# Decode and print result
result = tokenizer.decode(outputs[0][tokens["input_ids"].shape[-1]:], skip_special_tokens=True)
print(result)
