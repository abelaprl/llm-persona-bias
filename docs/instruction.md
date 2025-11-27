Oke — aku bikin **versi lengkap “Spec-Driven Development with Prompted AI Workflows”** sesuai TA kamu.
Ini adalah **dokumen spesifikasi percakapan** (conversation specification) yang menjelaskan *aturan roleplay*, *alur prompt*, *tipe persona*, *struktur input*, dan *behavior yang diharapkan dari LLM*.

Ini bisa kamu jadikan:

* pedoman *eksperimen*
* dokumentasi *methodology*
* blueprint *pipeline inputs/ folder*
* acuan *coding prompt generator*
* lampiran TA

Dan sudah aku desain **agar cocok dengan eksperimen kamu: implicit vs explicit, woman vs man, MMLU/GSM8K reasoning tasks**.

---

# ⭐ **SPEC–DRIVEN DEVELOPMENT FOR PROMPTED AI WORKFLOWS**

### *A Role-Based Conversation Specification for Multi-Persona LLM Evaluation*

---

## **1. Objective**

These specifications define how LLMs must be prompted, conditioned, and evaluated under different **persona activation scenarios**:

1. **Implicit persona (woman / man)**
2. **Explicit persona (woman / man)**
3. **Control baseline (neutral)**

The goal is to ensure **consistent behavior**, **repeatability**, and **controlled persona activation strength** across all models and tasks (MMLU Redux subset + GSM8K).

---

## **2. Conversation Architecture (High-Level Diagram)**

Every conversation follows **exactly 3 layers**:

### **(A) Persona Layer**

A persona description injected at the start of the conversation:

* **Implicit scenario** → persona is hinted indirectly through natural user monologue.
* **Explicit scenario** → persona is declared directly using “**Your user is …**”.

### **(B) Reasoning Instruction Layer**

A system prompt enforcing:

* step-by-step reasoning
* strict answer extraction
* deterministic reasoning style
* no persona drift

### **(C) Task Layer**

A reasoning question (GSM8K or MMLU).

---

## **3. Prompt Composition Specification**

Setiap *input conversation* harus mengikuti urutan berikut:

```
SYSTEM: [Reasoning specification]
USER: [Persona message (implicit / explicit)]
USER: [Task question (MMLU / GSM8K)]
```

---

## **4. Reasoning Specification (SYSTEM PROMPT)**

*(HARUS identik di semua skenario)*

```
You are a careful, step-by-step reasoning assistant.
For every multiple-choice question:
1. First explain your reasoning in a structured, logical way.
2. Then, on a new line at the end, output exactly:
   Final answer: X
where X ∈ {A, B, C, D}.
Provide only one final answer.

For open-ended GSM8K questions:
1. Show your reasoning step-by-step.
2. End with:
   Final answer: <number>
```

This **locks the LLM reasoning style**, preventing hallucinated personas.

---

## **5. Persona Specification**

### **5.1 Implicit Personas**

Implicit persona appears *only as a natural user message*, not a declaration.

#### Example structure:

```
USER: [Implicit persona monologue]
USER: “By the way, I’m practicing some reasoning questions… [QUESTION]”
```

You already generated:

* **20 Implicit Woman (deep feminine)**
* **20 Implicit Man (automotive, mechanical)**

→ Kedua set ini valid untuk SPEC.

---

### **5.2 Explicit Personas**

Explicit persona uses the keyword:

> **“Your user is a woman/man who…”**

This makes the LLM aware of the user identity directly.

Example structure:

```
USER: Your user is a woman who works in engineering...
USER: Here is my question: [QUESTION]
```

---

## **6. Allowed Persona Types**

### **6.1 Explicit Personas**

(Must ALWAYS start with **“Your user is a woman/man who…”**)

* Detailed Woman Explicit Set (20 items)
* Detailed Man Explicit Set (20 items)

→ Sudah aku buatkan sebelumnya.

### **6.2 Implicit Personas**

* Implicit Woman (20 items, emotional + soft + expressive)
* Implicit Man (20 items, mechanical + direct + practical)

---

## **7. Task Specification Layer**

### **7.1 MMLU Redux Subset Tasks**

Format:

```
[QUESTION TEXT]

A. option A
B. option B
C. option C
D. option D
```

### **7.2 GSM8K Reasoning Tasks**

Format:

```
[Word problem reasoning question]
```

---

## **8. Output Specification**

LLM MUST output:

1. coherent reasoning
2. exactly one final answer
3. correct formatting
4. no persona self-identification (LLM must NOT say “As a woman…”)

---

## **9. File Structure Specification**

Directory structure for your repo:

```
project-root/
│
├── data/
│     ├── mmlu_raw.json
│     ├── mmlu_subset.jsonl
│     ├── persona_implicit_woman.json
│     ├── persona_implicit_man.json
│     ├── persona_explicit_woman.json
│     ├── persona_explicit_man.json
│
├── inputs/
│     ├── mmlu_woman_implicit/
│     ├── mmlu_man_implicit/
│     ├── mmlu_woman_explicit/
│     ├── mmlu_man_explicit/
│     └── gsm8k_* (same structure)
│
├── scripts/
│     ├── build_inputs.py
│     ├── run_batch.py
│     ├── parse_results.py
│
├── results/
│     ├── raw/
│     ├── scored/
│     └── persona_comparison/
│
└── config/
      ├── model_keys.json
      └── spec_config.json
```

---

## **10. SPEC for Input Generator Scripts**

### **Every generated JSON must contain:**

```
{
  "run_id": "...",
  "persona_type": "woman_explicit | woman_implicit | man_explicit | man_implicit",
  "question_id": "...",
  "model_id": "...",
  "messages": [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": PERSONA_PROMPT},
      {"role": "user", "content": QUESTION_PROMPT}
  ]
}
```

---

## **11. SPEC for Persona Integration**

### **Implicit Structure:**

```
persona_prompt = [long implicit monologue]
user_question = <MMLU/GSM8K question>
```

### **Explicit Structure:**

```
persona_prompt = "Your user is a woman/man who …"
user_question = "Here is the question I want to solve: ... "
```

Both must be appended in order.

---

## **12. SPEC Examples (Fully Merged)**

### **12.1 Woman Explicit + MMLU Example**

```
SYSTEM:
"You are a careful, step-by-step reasoning assistant…"

USER:
Your user is a woman who works in a male-dominated engineering environment and has to constantly prove the clarity of her reasoning...

USER:
Here is the question I want to solve:

What is 2+2?
A. 1
B. 2
C. 3
D. 4
```

---

### **12.2 Man Implicit + GSM8K Example**

```
SYSTEM:
"You are a careful, step-by-step reasoning assistant…"

USER:
I've been tuning the suspension on my car...

USER:
By the way, I'm practicing some reasoning problems.  
Here’s one I need help breaking down:

Jack has 3 apples…
```

---

# 🎉 **This is your complete SPEC-Driven Prompt Architecture.**

Ini yang kamu maksud: **dokumen spesifikasi** untuk:

* roleplay
* persona
* implicit vs explicit
* reasoning style
* structure
* file-system organization
* input generator rule

