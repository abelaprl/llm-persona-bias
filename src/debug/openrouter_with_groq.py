#!/usr/bin/env python3
"""
Updated OpenRouter CLI dengan dukungan Groq API
Usage:
    python openrouter_with_groq.py "What is the meaning of life?"
    python openrouter_with_groq.py --interactive
    python openrouter_with_groq.py --groq "Hello from Groq"
"""

import os
import sys
import json
import time
import requests

# API configurations
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# API Keys
OPENROUTER_API_KEY = "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"
GROQ_API_KEY = "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"

# Available models
RECOMMENDED_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "gemma_2b": "google/gemma-3n-e2b-it:free", 
    "gemma_4b": "google/gemma-3-4b-it:free",
    "nvidia": "nvidia/nemotron-nano-12b-v2-vl:free",
    "glm": "z-ai/glm-4.5-air:free"
}

class ChatHistory:
    """Manages conversation history"""
    def __init__(self, persona=None):
        self.messages = []
        if persona:
            self.messages.append({"role": "system", "content": persona})
    
    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self):
        return self.messages.copy()
    
    def clear_history(self):
        # Keep system message if it exists
        system_msgs = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_msgs
    
    def save_to_file(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                self.messages = json.load(f)

def call_groq(model, messages, temperature=0.2, max_tokens=300):
    """Call the Groq API directly"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Groq API error: {e}")
        return None

def call_openrouter(model, messages, temperature=0.2, max_tokens=300):
    """Call the OpenRouter API and return response JSON"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "include_reasoning": False
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ OpenRouter API error: {e}")
        return None

def call_api(model, messages, temperature=0.2, max_tokens=300):
    """Smart API caller - automatically choose Groq or OpenRouter"""
    if model in ["llama-3.1-8b-instant", "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]:
        print(f"🚀 Using Groq API for {model}")
        return call_groq(model, messages, temperature, max_tokens)
    else:
        print(f"🌐 Using OpenRouter API for {model}")
        return call_openrouter(model, messages, temperature, max_tokens)

def run_chat_with_history(chat_history, prompt, model="llama-3.1-8b-instant"):
    """Run prompt with conversation history"""
    chat_history.add_user_message(prompt)
    
    start = time.time()
    result = call_api(model, chat_history.get_messages())
    latency = time.time() - start

    if not result:
        return None

    content = result["choices"][0]["message"]["content"]
    chat_history.add_assistant_message(content)
    
    print("\n🧠 Response:")
    print(content)
    print(f"\n⏱️ Latency: {latency:.2f}s | Tokens: {result.get('usage', {})}")
    
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "prompt": prompt,
        "response": content,
        "latency": latency,
        "raw": result,
    }

def interactive_mode(persona=None, model="llama-3.1-8b-instant"):
    """Interactive chat mode with history"""
    chat_history = ChatHistory(persona)
    
    print("🤖 Interactive Chat Mode dengan Groq & OpenRouter")
    print("Commands: 'quit' = exit, 'clear' = reset, 'save' = save chat")
    print("          'models' = show available models, 'switch <model>' = change model")
    if persona:
        print(f"📝 Persona: {persona}")
    print(f"🔧 Current model: {model}")
    print()
    
    out_file = "chat_results.jsonl"
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'clear':
                chat_history.clear_history()
                print("🧹 Chat history cleared!")
                continue
            elif user_input.lower() == 'save':
                filename = f"chat_history_{int(time.time())}.json"
                chat_history.save_to_file(filename)
                print(f"💾 Chat saved to {filename}")
                continue
            elif user_input.lower() == 'models':
                print("\n📋 Available models:")
                for key, model_id in RECOMMENDED_MODELS.items():
                    api_type = "Groq" if key == "groq" else "OpenRouter"
                    print(f"  {key}: {model_id} ({api_type})")
                continue
            elif user_input.lower().startswith('switch '):
                new_model_key = user_input[7:].strip()
                if new_model_key in RECOMMENDED_MODELS:
                    model = RECOMMENDED_MODELS[new_model_key]
                    print(f"🔄 Switched to model: {model}")
                else:
                    # Try as direct model ID
                    model = new_model_key
                    print(f"🔄 Switched to model: {model}")
                continue
            elif not user_input:
                continue
            
            data = run_chat_with_history(chat_history, user_input, model)
            
            if data:
                with open(out_file, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
    
    print(f"\n✅ Chat history saved to {out_file}")

def run_single_prompt(prompt, persona=None, model="llama-3.1-8b-instant"):
    """Run one prompt (with optional persona/system role)"""
    chat_history = ChatHistory(persona)
    return run_chat_with_history(chat_history, prompt, model)

def show_available_models():
    """Show available models"""
    print("🤖 AVAILABLE MODELS:")
    print("=" * 50)
    print("🚀 GROQ MODELS (Super Fast):")
    print("  • llama-3.1-8b-instant (Recommended)")
    print("  • llama3-8b-8192")
    print("  • llama3-70b-8192")
    print("  • mixtral-8x7b-32768")
    print()
    print("🌐 OPENROUTER MODELS (Working):")
    for key, model_id in RECOMMENDED_MODELS.items():
        if key != "groq":
            print(f"  • {model_id}")
    print()
    print("💡 Usage:")
    print("  --groq : Use Groq Llama (fastest)")
    print("  --gemma : Use Google Gemma")
    print("  --nvidia : Use NVIDIA Nemotron")

def main():
    persona = "You are a helpful AI assistant. Respond clearly and concisely."
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python openrouter_with_groq.py '<prompt>'")
        print("  python openrouter_with_groq.py --interactive")
        print("  python openrouter_with_groq.py --groq '<prompt>'")
        print("  python openrouter_with_groq.py --models")
        sys.exit(0)

    arg = sys.argv[1]
    
    if arg == "--interactive" or arg == "-i":
        interactive_mode(persona)
    elif arg == "--groq":
        if len(sys.argv) > 2:
            prompt = " ".join(sys.argv[2:])
            data = run_single_prompt(prompt, persona, "llama-3.1-8b-instant")
            if data:
                with open("results.jsonl", "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                print(f"\n✅ Result saved to results.jsonl")
        else:
            interactive_mode(persona, "llama-3.1-8b-instant")
    elif arg == "--gemma":
        if len(sys.argv) > 2:
            prompt = " ".join(sys.argv[2:])
            data = run_single_prompt(prompt, persona, RECOMMENDED_MODELS["gemma_4b"])
        else:
            interactive_mode(persona, RECOMMENDED_MODELS["gemma_4b"])
    elif arg == "--models":
        show_available_models()
    elif os.path.exists(arg):
        # treat as file - run with shared history
        with open(arg, "r", encoding="utf-8") as f:
            tasks = [line.strip() for line in f if line.strip()]
        
        chat_history = ChatHistory(persona)
        out_file = "batch_results.jsonl"
        
        with open(out_file, "a", encoding="utf-8") as fout:
            for i, task in enumerate(tasks, start=1):
                print(f"\n=== [{i}/{len(tasks)}] {task} ===")
                data = run_chat_with_history(chat_history, task, "llama-3.1-8b-instant")
                if data:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        print(f"\n✅ Done! Results saved to {out_file}")
    else:
        # treat as single prompt
        data = run_single_prompt(arg, persona, "llama-3.1-8b-instant")
        if data:
            with open("results.jsonl", "a", encoding="utf-8") as fout:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            print(f"\n✅ Done! Result saved to results.jsonl")

if __name__ == "__main__":
    main()