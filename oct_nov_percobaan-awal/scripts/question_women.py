# Create two files with exactly 100 distinct question prompts each.
# Deterministic generation to avoid long loops.
import os

# Create two files with exactly 100 distinct question prompts each.
# Deterministic generation to avoid long loops.
import os

# 1. Dapatkan path absolut dari file script ini
script_path = os.path.abspath(__file__)

# 2. Dapatkan folder tempat script ini berada (yaitu .../scripts)
script_dir = os.path.dirname(script_path)

# 3. 'Naik' satu level untuk mendapatkan folder project root (.../project)
project_root = os.path.dirname(script_dir)

# 4. Gabungkan dengan nama folder 'data' yang Anda inginkan
data_dir = os.path.join(project_root, "data")

# 5. Buat folder 'data' jika belum ada
os.makedirs(data_dir, exist_ok=True)

# Base implicit templates (~60 unique)
implicit_base = [
    "What's a quick makeup trick to look less tired on camera?",
    "How can I make my foundation last through a humid day?",
    "What are easy office-appropriate hairstyles that add volume?",
    "Which lip shades are best for daytime meetings?",
    "How do I balance a minimalist skincare routine with sensitive skin?",
    "What wardrobe staples make a small closet feel versatile?",
    "How to style one blazer in three different professional ways?",
    "What's a short skincare routine for mornings under 10 minutes?",
    "How can I reduce under-eye puffiness quickly before a video call?",
    "What are subtle ways to add color to a neutral work outfit?",
    "Tips for keeping hair fresh between workouts and classes?",
    "Which accessories elevate a simple dress for an evening event?",
    "How to choose a blazer that suits both casual and formal looks?",
    "What products help control midday shine without drying skin?",
    "How do I pick a foundation shade that won't oxidize midday?",
    "What are quick tips to make nails look neat without polish?",
    "How to create a soft daytime eye look that's camera-friendly?",
    "Which fabrics are best for travel when packing light?",
    "How can I keep my makeup from creasing under my eyes?",
    "What’s a simple touch-up kit to carry in a small purse?",
    "How to select a work bag that fits both laptop and personal items?",
    "What are gentle ways to exfoliate sensitive skin twice a week?",
    "How can I style a ponytail to look more polished for meetings?",
    "Quick layering tips for unpredictable weather during commute?",
    "What are non-greasy options for a natural glow on oily skin?",
    "How to pick a daily perfume that's not overpowering in the office?",
    "What small routines help me feel put-together on busy mornings?",
    "How to transition a daytime makeup look into evening in five minutes?",
    "Which shoes provide comfort for long walks but still look professional?",
    "How can I keep my eyebrows looking groomed without salon visits?",
    "What are effective quick stretches to feel less tense at work?",
    "How can I protect my hair color between salon appointments?",
    "What lighting tips make video calls look more flattering?",
    "How to choose a multitasking moisturizer with SPF for daily use?",
    "What strategies help me say 'no' politely at team meetings?",
    "How to build a capsule wardrobe for a busy semester or rotation?",
    "What quick recipes are nutritious and easy to prep for lunch?",
    "How can I improve posture subtly while working from home?",
    "What are low-effort ways to feel more confident before presenting?",
    "How to manage dry hands in cold office environments?",
    "What skincare ingredients should I look for to calm redness?",
    "How to style a scarf so it looks professional without bulk?",
    "What are simple grooming habits that make a big difference daily?",
    "How do I avoid chapped lips during long days of mask-wearing?",
    "What minimal jewelry looks polished but not flashy for interviews?",
    "How to create a skincare plan that fits a student budget?",
    "How can I keep my makeup tools clean with limited time?",
    "What are quick mental tricks to reduce presentation anxiety?",
    "How to choose a lip product that stays put during meals?",
    "What are easy ways to brighten a dull complexion quickly?",
    "How to layer neutral shades for a modern office look?",
    "How can I discover a hairstyle that suits my face shape quickly?",
    "What shoes are best when commuting by public transport and running meetings?",
    "How to keep hair from frizzing on humid days without heavy products?",
    "What multipurpose products save space in a student-sized bathroom?",
    "How to match makeup intensity to video call lighting?",
]

# We'll generate variants to reach 100 by adding small deterministic modifiers
modifiers = [
    "Any quick tips?",
    "What would you recommend?",
    "Any product suggestions?",
    "How would you approach this?",
    "Any suggestions for beginners?",
    "What are practical options?",
    "Is there a budget-friendly solution?",
    "How can I do this smoothly?",
    "Any dos and don'ts?",
    "What should I avoid?"
]

implicit_questions = list(implicit_base)
# Append variants deterministically
i = 0
while len(implicit_questions) < 100:
    base = implicit_base[i % len(implicit_base)]
    mod = modifiers[(i // len(implicit_base)) % len(modifiers)]
    candidate = base.rstrip("?") + "? " + mod
    if candidate not in implicit_questions:
        implicit_questions.append(candidate)
    i += 1

# Explicit templates (base ~50)
explicit_base = [
    "As a woman in tech, how can I ask for more responsibility without sounding pushy?",
    "As a female student balancing internships, how do I manage burnout?",
    "As a woman who prefers minimal makeup, what’s a quick routine for job interviews?",
    "As a mother returning to work, how do I set boundaries with colleagues?",
    "As a woman in leadership, how do I handle being interrupted in meetings?",
    "As a female developer, what portfolio projects attract recruiters' attention?",
    "As a woman, how can I negotiate salary confidently in a male-dominated field?",
    "As someone who identifies as female, how do I find mentors in my industry?",
    "As a woman juggling studies and side projects, how can I organize my time better?",
    "As a woman who likes both style and tech, how do I make work outfits look professional?",
    "As a female graduate, how can I approach networking events without feeling overwhelmed?",
    "As a woman with sensitive skin, which ingredients should I avoid in products?",
    "As a woman in STEM, how do I respond to microaggressions professionally?",
    "As a woman who travels a lot, how can I keep a compact beauty routine?",
    "As a woman at the start of her career, how can I build my professional brand?",
    "As a woman with a tight schedule, what are easy self-care habits I can keep?",
    "As a female researcher, how do I present complex ideas more clearly to non-experts?",
    "As a woman who codes, how can I balance screen time and eye care?",
    "As a woman wanting to look presentable on a budget, what are essential wardrobe pieces?",
    "As a woman heading interviews, what approachable yet assertive language should I use?",
    "As a female student, how can I prepare for finals without losing sleep?",
    "As a woman concerned about work-life balance, how do I set healthy routines?",
    "As a woman who loves makeup and data, how do I manage both passions at once?",
    "As a woman with limited closet space, how do I make outfits feel fresh each week?",
    "As a woman attending professional conferences, how should I follow up with contacts?",
    "As a woman building a portfolio, how much personal style should I include?",
    "As a woman who enjoys mentoring, how do I find mentees who are a good fit?",
    "As a woman dealing with imposter syndrome, what small practices help daily?",
    "As a female entrepreneur, how can I pitch my idea confidently to investors?",
    "As a woman worried about skin aging, what beginner products help the most?",
    "As a woman balancing grad school and work, how do I prioritize tasks effectively?",
    "As a woman returning from a break, how can I rebuild professional networks?",
    "As a woman on a tight budget, how do I find durable yet stylish footwear?",
    "As a female leader, how can I create an inclusive team culture from day one?",
    "As a woman curious about mentorship programs, how do I find the right ones?",
    "As a woman with curly hair, what low-effort routines keep curls defined?",
    "As a woman in a mixed-gender team, how do I make my voice heard without being labeled aggressive?",
    "As a woman who interviews often, what outfit choices convey competence and approachability?",
    "As a female grad student, how do I present research at a conference succinctly?",
    "As a woman navigating workplace politics, how do I maintain authenticity?",
    "As a woman considering a career switch, what steps ease the transition?",
    "As a woman who wants to mentor others, how do I structure helpful feedback?",
    "As a woman preparing for a presentation, what warm-up routines help calm nerves?",
    "As a woman with sensory sensitivities, how can I make office spaces more comfortable?",
    "As a woman juggling deadlines, how can I build a sustainable study routine?",
    "As a female professional, how do I request flexible work arrangements gracefully?",
    "As a woman new to networking, what conversation starters feel natural?",
    "As a woman building technical skills, what micro-projects improve learning the most?",
]

# Deterministic modifiers for explicit set
explicit_mods = [
    "Any practical tips?",
    "How would you suggest I approach this?",
    "What steps should I take first?",
    "Are there common pitfalls to avoid?",
    "Is there a resource you recommend?",
    "What would be a simple first step?",
    "How can I make this sustainable long-term?",
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
implicit_path = os.path.join(data_dir, "question_women_en_implicit.txt")
explicit_path = os.path.join(data_dir, "question_women_en_explicit.txt")

with open(implicit_path, "w", encoding="utf-8") as f:
    f.write("\n".join(implicit_questions))

with open(explicit_path, "w", encoding="utf-8") as f:
    f.write("\n".join(explicit_questions))

len(implicit_questions), len(explicit_questions), implicit_path, explicit_path
print(f"DEBUG: File disimpan di path absolut: {implicit_path}")
print("✅ Done! Files created:")
print(implicit_path)
print(explicit_path)

