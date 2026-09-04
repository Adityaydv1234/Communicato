import re

FILLER_PATTERNS = [
    r"\bum\b",
    r"\buh\b",
    r"\blike\b",
    r"\byou know\b",
    r"\bactually\b",
    r"\bbasically\b",
    r"\bliterally\b",
    r"\bsort of\b",
    r"\bkind of\b",
    r"\bi mean\b",
]
FILLER_REGEX = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)


def count_fillers(text: str) -> int:
    return len(FILLER_REGEX.findall(text))


def count_words(text: str) -> int:
    return len(text.split())
