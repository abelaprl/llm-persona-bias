# Create two files with exactly 100 distinct question prompts each.
# Deterministic generation to avoid long loops.
import os

data_dir = "/mnt/data/project-root/data"
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