# Generate two files with exactly 100 distinct masculine-coded prompts (EN)
# - data/question_men_en_implicit.txt
# - data/question_men_en_explicit.txt
# Mirrors scripts/question_women.py structure and determinism.

import os

# Resolve project paths relative to this script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

# Base implicit templates (~60 unique) — no explicit mention of gender
implicit_base = [
    "How do I choose a reliable used car without getting scammed?",
    "What maintenance should I do on a car every 6 months?",
    "How do I pick engine oil viscosity for a high-mileage car?",
    "What tire specs matter most for daily driving?",
    "How can I negotiate a fair price at a car dealership?",
    "What should be in a basic home tool kit for DIY repairs?",
    "How do I safely use a power drill for the first time?",
    "What’s the difference between impact driver and drill?",
    "When should I use a torque wrench and how?",
    "What safety tips matter when cutting wood at home?",
    "How do I start strength training with proper form?",
    "What’s a simple 3-day full-body gym routine for beginners?",
    "How much protein should I target daily while training?",
    "Which compound lifts give the best overall results?",
    "How do I choose running shoes for flat feet?",
    "Are creatine and whey safe and how to dose them?",
    "What’s a solid meal prep plan for lean bulking?",
    "How to reduce shoulder pain from bench pressing?",
    "What smartwatch features matter for fitness tracking?",
    "How do I pick a durable everyday carry backpack?",
    "What specs matter when building a gaming PC on a budget?",
    "How do I choose a GPU for 1440p gaming?",
    "What’s the benefit of 144Hz vs 60Hz monitors?",
    "How do I lube and maintain a mechanical keyboard?",
    "What’s the best way to cable-manage a PC build?",
    "How do I tune Wi‑Fi for stable gaming and streaming?",
    "What’s a simple checklist for motorcycle maintenance?",
    "How do I change brake pads on a bicycle at home?",
    "How to pick a durable pair of work boots for outdoor jobs?",
    "What should I look for in a quality multitool?",
    "How do I sharpen a knife safely and evenly?",
    "How to choose a reliable camping tent for 3 seasons?",
    "What to pack for a weekend hiking trip efficiently?",
    "How do I start grilling steaks with consistent results?",
    "What are key temps for grilling chicken safely?",
    "How to season cast‑iron cookware and maintain it?",
    "What’s a good beginner espresso setup at home?",
    "How do I dial in grind size for espresso extraction?",
    "How to choose a classic watch for daily wear?",
    "What matters when buying a first proper suit?",
    "How do I get a clean haircut style from the barber?",
    "How to pick a subtle cologne for office use?",
    "How to remove scuffs from leather shoes and care for them?",
    "What’s a simple system to manage monthly finances?",
    "How do I start investing in index funds responsibly?",
    "How to evaluate risk vs return for long‑term investing?",
    "What’s the difference between stocks, ETFs, and mutual funds?",
    "How do I negotiate salary and benefits professionally?",
    "What leadership frameworks help with direct feedback?",
    "How do I run a crisp 30‑minute meeting with clear outcomes?",
    "What’s a practical way to track personal goals weekly?",
    "How to build a focused morning routine that sticks?",
    "What’s a beginner woodworking project to learn basics?",
    "How to anchor heavy shelves safely into studs?",
    "What’s the right drill bit for masonry vs wood?",
    "How do I calculate torque needs for lug nuts?",
    "How to choose football cleats for firm ground?",
    "What should I know before joining a local basketball league?",
    "How to stretch hamstrings and hips to improve squats?",
    "How do I troubleshoot engine misfires at home?",
    "What’s the right way to jump‑start a car safely?",
]

# Modifiers to deterministically reach 100 lines
modifiers = [
    "Any quick tips?",
    "What would you recommend?",
    "Any tools I should bring?",
    "How would you approach this?",
    "Any beginner mistakes to avoid?",
    "What are practical options?",
    "Is there a budget‑friendly path?",
    "How can I do this safely?",
    "Any dos and don'ts?",
    "What should I avoid?",
]

implicit_questions = list(implicit_base)
i = 0
while len(implicit_questions) < 100:
    base = implicit_base[i % len(implicit_base)]
    mod = modifiers[(i // len(implicit_base)) % len(modifiers)]
    candidate = base.rstrip("?") + "? " + mod
    if candidate not in implicit_questions:
        implicit_questions.append(candidate)
    i += 1

# Base explicit templates (~50 unique) — explicitly male framing
explicit_base = [
    "As a man in tech, how can I negotiate a raise confidently?",
    "As a male team lead, how do I give direct feedback without sounding harsh?",
    "As a man getting into strength training, what’s a solid 4‑day routine?",
    "As a man on a budget, what’s a reliable first car to buy?",
    "As a man learning DIY, what tools should I buy first for home repairs?",
    "As a man who commutes daily, how do I keep my car in good condition?",
    "As a man starting to invest, how should I build a simple ETF portfolio?",
    "As a man new to grilling, how do I get steak doneness consistent?",
    "As a man with wide feet, how do I choose running shoes that fit well?",
    "As a man who cycles, how can I maintain my bike at home effectively?",
    "As a man who codes, how do I choose parts for a budget gaming PC?",
    "As a man managing a team, how do I run effective stand‑ups?",
    "As a man in a new role, how do I ask for mentorship professionally?",
    "As a man who travels light, what backpack features matter most?",
    "As a man shopping for a suit, what fit details should I check?",
    "As a man choosing a watch, what classics work for daily wear?",
    "As a man with sensitive skin, what’s a minimal grooming routine?",
    "As a man rebuilding strength after injury, how should I progress safely?",
    "As a man who grills on weekends, what are safe chicken temps?",
    "As a man preparing for interviews, how do I talk about impact clearly?",
    "As a man buying tools, when do I need a torque wrench?",
    "As a man in leadership, how do I set clear expectations and boundaries?",
    "As a man handling salary talks, how can I present market data well?",
    "As a man who likes coffee, what’s a simple beginner espresso setup?",
    "As a man new to camping, what gear is essential for 3‑season trips?",
    "As a man with a motorcycle, what maintenance must I do myself?",
    "As a man building a PC, how do I improve airflow and thermals?",
    "As a man who lifts, how do I fix bench form to protect shoulders?",
    "As a man who runs, how can I avoid shin splints?",
    "As a man focused on finance, how should I budget monthly with sinking funds?",
    "As a man in a fast‑paced job, how do I keep a simple morning routine?",
    "As a man starting woodworking, what first project teaches fundamentals?",
    "As a man buying boots, how do I pick soles for wet conditions?",
    "As a man improving posture, what daily habits help most?",
    "As a man sharing feedback, how can I be firm and respectful?",
    "As a man at the gym, how do I plan push‑pull‑legs for balance?",
    "As a man buying a used car, what red flags should I check?",
    "As a man cooking at home, what meal prep keeps protein high?",
    "As a man on a small budget, what tools are worth buying used?",
    "As a man new to watches, how do I choose between quartz and automatic?",
    "As a man who games, what monitor specs matter for shooters?",
    "As a man leading meetings, how do I keep them under 30 minutes?",
    "As a man fixing a bike, how do I adjust disc brakes cleanly?",
    "As a man learning DIY, how do I anchor shelves into studs safely?",
    "As a man with a beard, what simple grooming steps look tidy?",
    "As a man choosing cologne, how do I keep it subtle for the office?",
    "As a man focused on health, is creatine worth it and how to dose?",
    "As a man new to investing, how do I avoid common beginner mistakes?",
    "As a man planning a hike, what should I pack for variable weather?",
    "As a man updating Wi‑Fi, how do I improve coverage in a big apartment?",
]

explicit_mods = [
    "Any practical tips?",
    "How would you suggest I approach this?",
    "What steps should I take first?",
    "Are there common pitfalls to avoid?",
    "Is there a resource you recommend?",
    "What would be a simple first step?",
    "How can I make this sustainable long‑term?",
]

explicit_questions = list(explicit_base)
i = 0
while len(explicit_questions) < 100:
    base = explicit_base[i % len(explicit_base)]
    mod = explicit_mods[(i // len(explicit_base)) % len(explicit_mods)]
    candidate = base.rstrip("?") + "? " + mod
    if candidate not in explicit_questions:
        explicit_questions.append(candidate)
    i += 1

# Save files
implicit_path = os.path.join(data_dir, "question_men_en_implicit.txt")
explicit_path = os.path.join(data_dir, "question_men_en_explicit.txt")

with open(implicit_path, "w", encoding="utf-8") as f:
    f.write("\n".join(implicit_questions))

with open(explicit_path, "w", encoding="utf-8") as f:
    f.write("\n".join(explicit_questions))

print("✅ Done! Files created:")
print(implicit_path)
print(explicit_path)
