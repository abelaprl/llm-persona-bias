#!/usr/bin/env python3
"""
Simple CLI client for OpenRouter with chat history support
Usage:
    python openrouter_cli.py "What is the meaning of life?"
or:
    python openrouter_cli.py tasks.txt
or:
    python openrouter_cli.py --interactive
"""

import os
import sys
import json
import time
import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"

if not API_KEY:
    print("❌ Error: Please set OPENROUTER_API_KEY environment variable.")
    sys.exit(1)


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


def call_openrouter(model, messages, temperature=0.2, max_tokens=300):
    """Call the OpenRouter API and return response JSON"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API error: {e}")
        return None


def run_chat_with_history(chat_history, prompt, model="mistralai/mistral-7b-instruct:free"):
    """Run prompt with conversation history"""
    chat_history.add_user_message(prompt)
    
    start = time.time()
    result = call_openrouter(model, chat_history.get_messages())
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


def interactive_mode(persona=None, model="mistralai/mistral-7b-instruct:free"):
    """Interactive chat mode with history"""
    chat_history = ChatHistory(persona)
    
    print("🤖 Interactive Chat Mode (type 'quit' to exit, 'clear' to reset history, 'save' to save chat)")
    if persona:
        print(f"📝 Persona: {persona}")
    
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


def run_single_prompt(prompt, persona=None, model="mistralai/mistral-7b-instruct:free"):
    """Run one prompt (with optional persona/system role) - legacy function"""
    chat_history = ChatHistory(persona)
    return run_chat_with_history(chat_history, prompt, model)


def run_batch_with_history(tasks, persona=None, model="mistralai/mistral-7b-instruct:free"):
    """Run multiple prompts with shared history"""
    chat_history = ChatHistory(persona)
    out_file = "batch_results.jsonl"
    
    with open(out_file, "a", encoding="utf-8") as fout:
        for i, task in enumerate(tasks, start=1):
            print(f"\n=== [{i}/{len(tasks)}] {task} ===")
            data = run_chat_with_history(chat_history, task, model)
            if data:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # Save final chat history
    chat_history.save_to_file(f"batch_chat_history_{int(time.time())}.json")
    print(f"\n✅ Done! Results saved to {out_file}")


def main():
    persona = "You are a polite assistant who loves discussing daily routines, skincare, and self-care. You ask clarifying questions first before giving concise tips."
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python openrouter_cli.py '<prompt>'")
        print("  python openrouter_cli.py tasks.txt")
        print("  python openrouter_cli.py --interactive")
        sys.exit(0)

    arg = sys.argv[1]
    
    if arg == "--interactive" or arg == "-i":
        interactive_mode(persona)
    elif os.path.exists(arg):
        # treat as file - run with shared history
        with open(arg, "r", encoding="utf-8") as f:
            tasks = [line.strip() for line in f if line.strip()]
        run_batch_with_history(tasks, persona)
    else:
        # treat as single prompt
        data = run_single_prompt(arg, persona)
        if data:
            with open("results.jsonl", "a", encoding="utf-8") as fout:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            print(f"\n✅ Done! Result saved to results.jsonl")


if __name__ == "__main__":
    main()
