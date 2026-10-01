"""Shared display terminology; never transform data keys or quoted responses."""
import re

MODEL_NAMES = {
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill 32B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B-A3B",
}
REPLACEMENTS = {
    "DeepSeek R1 32B": "DeepSeek R1-Distill 32B",
    "Qwen3 Next 80B": "Qwen3 Next 80B-A3B",
    "penanda bahasa umum": "penanda linguistik umum",
    "penanda bahasa etika": "penanda linguistik etika",
    "penanda wacana gender": "penanda linguistik wacana gender",
    "Median 6 model": "Median enam model",
    "Man": "Laki-laki", "Woman": "Perempuan",
    "Neutral": "Netral", "neutral": "netral",
    "Explicit": "Eksplisit", "explicit": "eksplisit",
    "Implicit": "Implisit", "implicit": "implisit",
    "Kondisi": "Skenario", "kondisi": "skenario",
    "Technical-only": "Hanya konten teknis",
    "Masculine pronouns": "Pronomina maskulin",
    "Feminine pronouns": "Pronomina feminin",
    "pronoun": "pronomina", "opening": "pembuka",
    "Engagement": "Sapaan kepada pengguna",
    "Konektif logis": "Kata penghubung",
    "Organisasi": "Cara menyusun jawaban",
    "Keragaman leksikal": "Variasi kata",
    "Sikap epistemik": "Kepastian dan keraguan",
    "Rujukan diri": "Kata ganti diri",
    "Elaborasi": "Penjelasan tambahan",
    "Presentasi matematis": "Bahasa matematis",
}


def display_text(value):
    text = str(value)
    for before, after in REPLACEMENTS.items():
        # Hyphens protect already expanded model names and compound keys.
        text = re.sub(r"(?<![\w-])" + re.escape(before) + r"(?![\w-])", lambda _: after, text)
    return text


def model_display(value):
    """Two-line model labels fit categorical axes without smaller type."""
    return display_text(value).replace("DeepSeek R1-Distill 32B", "DeepSeek R1-Distill\n32B").replace(
        "Qwen3 Next 80B-A3B", "Qwen3 Next\n80B-A3B")
